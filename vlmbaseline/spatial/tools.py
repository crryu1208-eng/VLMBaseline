"""Spatial queries over a scene graph, computed in code (the VLM only picks the query and phrases the answer).

    idx = SpatialIndex(load_export("data/library_merged"))
    idx.count("chair"); idx.nearest("O817", "potted plant"); idx.route("O817", "O834")

Distances are horizontal (x, y). Walking distances follow the 2D floor places of the graph (Dijkstra over
place-place edges); an object is reached through the place it is attached to.
Every tool returns a JSON-serialisable dict; TOOLS holds their function-calling schemas.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np

from vlmbaseline.scene import Scene
from vlmbaseline.spatial.search import rank

# visible states that call for an action (augment/describe_objects.py STATES); other states are normal
ACTIONS = {
    "잎이 시들거나 누렇게 변함": "물 주기 또는 식물 교체",
    "거의 가득 참": "쓰레기통 비우기",
    "넘침": "쓰레기통 바로 비우기",
    "물건이 어지럽게 놓여 있음": "책상 위 정리",
    "어지러움": "선반 정리",
    "위에 물건이 쌓여 있음": "위에 쌓인 물건 정리",
    "이상이 보임": "상태 점검",
    "열림": "닫혀 있어야 하면 닫기",
}

SKIP_LANDMARK = {"book", "paper", "sticky note", "document", "folder", "ceiling light"}


class SpatialIndex:
    def __init__(self, scene: Scene, encoder=None):
        self.s = scene
        self.encoder = encoder  # search.TextEncoder for free-text search (optional)
        self.obj = {o.symbol: o for o in scene.objects}
        self.place = {p.id: p for p in scene.places}
        self.adj = defaultdict(list)
        for a, b in scene.edges:
            if a in self.place and b in self.place:
                d = float(np.linalg.norm(self.place[a].center[:2] - self.place[b].center[:2]))
                self.adj[a].append((b, d))
                self.adj[b].append((a, d))
        self.on_place = defaultdict(list)
        for o in scene.objects:
            self.on_place[o.place].append(o)
        self.labels = Counter(o.label for o in scene.objects)

    # ---------------------------------------------------------------- helpers
    def _o(self, sym):
        o = self.obj.get(str(sym).strip().upper())
        if o is None:
            raise KeyError(f"no object {sym}")
        return o

    def _match(self, label):
        label = label.strip().lower()
        hit = [o for o in self.s.objects if o.label == label]
        if not hit:  # 'plant' also finds 'potted plant', 'cabinet' finds 'filing cabinet'
            hit = [o for o in self.s.objects if label in o.label or o.label in label]
        return hit

    @staticmethod
    def _xy(o):
        return [round(float(o.center[0]), 2), round(float(o.center[1]), 2)]

    def _brief(self, o, ref=None):
        r = {"symbol": o.symbol, "label": o.label, "xy": self._xy(o), "top_z": round(float(o.center[2] + o.size[2] / 2), 2)}
        if o.vlm:
            r["name"] = o.vlm.get("이름")
        if ref is not None:
            r["dist_m"] = round(float(np.linalg.norm(o.center[:2] - ref.center[:2])), 2)
        return r

    def _dijkstra(self, src):
        dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                if d + w < dist.get(v, 1e18):
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(pq, (d + w, v))
        return dist, prev

    # ---------------------------------------------------------------- tools
    def list_labels(self):
        return {"labels": dict(self.labels.most_common())}

    def count(self, label):
        hit = self._match(label)
        return {"label": label, "count": len(hit), "matched_labels": sorted({o.label for o in hit}),
                "objects": [o.symbol for o in hit]}

    def find(self, label):
        hit = self._match(label)
        return {"label": label, "objects": [self._brief(o) for o in hit]}

    def object_info(self, symbol):
        o = self._o(symbol)
        near = sorted((x for x in self.s.objects if x is not o and x.label not in SKIP_LANDMARK),
                      key=lambda x: np.linalg.norm(x.center[:2] - o.center[:2]))[:4]
        r = self._brief(o)
        r.update(size_m=[round(float(v), 2) for v in o.size], room=o.room,
                 nearby=[self._brief(x, o) for x in near])
        if o.vlm:
            r["description"] = o.vlm
        return r

    def search(self, query, k=5):
        """Objects matching a free-text description (Korean or English)."""
        out = []
        for score, o, parts in rank(self.s.objects, query, self.encoder, int(k)):
            b = self._brief(o)
            if o.vlm:
                b.update(state=o.vlm.get("상태"), uses=o.vlm.get("용도"), summary=o.vlm.get("설명"))
            b["match"] = parts
            out.append(b)
        return {"query": query, "objects": out}

    def inventory(self, label=None):
        """Objects whose visible state calls for an action (cleaning, tidying, repair), with the action, and the
        number of objects per state; states are as seen when the scene was recorded."""
        objs = [o for o in self.s.objects if label is None or o.label == label]
        todo, states, unknown = [], Counter(), 0
        for o in objs:
            v = o.vlm or {}
            st = v.get("상태")
            if not st:
                unknown += 1
                continue
            states[f"{o.label}: {st}"] += 1
            act = ACTIONS.get(st) or ("상태 점검" if "이상" in st else None)
            if o.label in ("door", "glass door") and st == "열림":
                act = None  # open doors are normal
            if v.get("점검_필요"):
                act = (act + "; " if act else "") + (v.get("점검_이유") or "점검")
            if act:
                todo.append({"symbol": o.symbol, "label": o.label, "name": v.get("이름"), "state": st, "action": act})
        return {"actionable": todo, "state_counts": dict(states.most_common()), "not_described": unknown,
                "note": "only 'actionable' objects need work; states are as seen when the scene was recorded"}

    def nearest(self, symbol, label, k=1):
        o = self._o(symbol)
        hit = [x for x in self._match(label) if x is not o]
        hit.sort(key=lambda x: np.linalg.norm(x.center[:2] - o.center[:2]))
        return {"from": symbol, "label": label, "nearest": [self._brief(x, o) for x in hit[:int(k)]]}

    def distance(self, a, b):
        oa, ob = self._o(a), self._o(b)
        straight = float(np.linalg.norm(oa.center[:2] - ob.center[:2]))
        dist, _ = self._dijkstra(oa.place)
        walk = dist.get(ob.place)
        return {"a": a, "b": b, "straight_m": round(straight, 2),
                "walking_m": round(walk, 2) if walk is not None else None}

    def extreme(self, kind):
        key = {"largest": lambda o: float(np.prod(o.size)), "highest": lambda o: float(o.center[2] + o.size[2] / 2),
               "lowest": lambda o: -float(o.center[2] - o.size[2] / 2), "smallest": lambda o: -float(np.prod(o.size))}
        if kind not in key:
            raise KeyError(f"kind must be one of {sorted(key)}")
        o = max(self.s.objects, key=key[kind])
        return {"kind": kind, "object": self._brief(o)}

    def route(self, start, goal):
        """Walking route between two objects: length, and the landmarks passed in order."""
        a, b = self._o(start), self._o(goal)
        dist, prev = self._dijkstra(a.place)
        if b.place not in dist:
            return {"start": start, "goal": goal, "reachable": False}
        path, u = [b.place], b.place
        while u != a.place:
            u = prev[u]
            path.append(u)
        path.reverse()
        pts = np.array([self.place[p].center[:2] for p in path])
        along = np.array([dist[p] for p in path])
        near = []  # objects within 1.5 m of the path, at the distance along the path where they are passed
        for o in self.s.objects:
            if o.symbol in (start, goal) or o.label in SKIP_LANDMARK or o.label == b.label:
                continue
            d = np.linalg.norm(pts - o.center[:2], axis=1)
            if d.min() <= 1.5:
                near.append((float(along[d.argmin()]), -float(np.prod(o.size)), o))
        near.sort(key=lambda t: (t[0], t[1]))
        marks, last = [], -1e9  # at most one landmark per 2 m, the biggest object there
        for k, (a_m, _, o) in enumerate(near):
            if a_m - last < 2.0:
                continue
            group = [t for t in near[k:] if t[0] - a_m < 2.0]
            o = min(group, key=lambda t: t[1])[2]
            marks.append({"symbol": o.symbol, "label": o.label, "after_m": round(a_m, 1)})
            last = a_m
        return {"start": self._brief(a), "goal": self._brief(b), "reachable": True,
                "walking_m": round(dist[b.place], 2),
                "straight_m": round(float(np.linalg.norm(a.center[:2] - b.center[:2])), 2),
                "landmarks": marks[:8]}

    def guide_to(self, start, label):
        """Nearest object of a label from the start object, and the walking route to it."""
        n = self.nearest(start, label)["nearest"]
        if not n:
            return {"start": start, "label": label, "error": "no such object"}
        return self.route(start, n[0]["symbol"])

    def call(self, name, args):
        fn = {t["function"]["name"]: getattr(self, t["function"]["name"]) for t in TOOLS}
        if name not in fn:
            return {"error": f"unknown tool {name}"}
        try:
            return fn[name](**args)
        except (KeyError, TypeError, ValueError) as e:
            return {"error": str(e)}


def _t(name, desc, **props):
    req = [k for k, v in props.items() if not v.pop("optional", False)]
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": req}}}


S = {"type": "string"}
TOOLS = [
    _t("list_labels", "All object labels in the scene with their counts."),
    _t("count", "Number of objects with a label (English label from the scene's label list).",
       label=dict(S, description="English object label, e.g. 'chair'")),
    _t("find", "All objects with a label and their positions.", label=dict(S, description="English object label")),
    _t("search", "Find objects from a free-text description in Korean or English (kind, use, appearance), "
       "e.g. '앉을 수 있는 곳', '음식을 데우는 기계', '주황색 안내판'. Use this for any '... 어디 있어?' question "
       "that has no object symbol. Returns the best matches with symbols and descriptions.",
       query=dict(S, description="description of the object"),
       k={"type": "integer", "description": "how many (default 5)", "optional": True}),
    _t("inventory", "ONLY for task suggestions (cleaning, tidying, repair): objects whose visible state needs work, "
       "with the action. Do not use it to locate objects.",
       label={"type": "string", "description": "only this English label (optional)", "optional": True}),
    _t("object_info", "Label, position, size, room and nearby objects of one object.",
       symbol=dict(S, description="object symbol, e.g. 'O817'")),
    _t("nearest", "Objects of a label closest (horizontal straight-line distance) to a given object.",
       symbol=dict(S, description="reference object symbol"), label=dict(S, description="English label to look for"),
       k={"type": "integer", "description": "how many (default 1)", "optional": True}),
    _t("distance", "Horizontal straight-line and walking distance between two objects.",
       a=dict(S, description="object symbol"), b=dict(S, description="object symbol")),
    _t("extreme", "The largest, smallest, highest (top) or lowest (bottom) object.",
       kind={"type": "string", "enum": ["largest", "smallest", "highest", "lowest"]}),
    _t("guide_to", "Guidance: the nearest object of a label from a start object and the walking route to it "
       "(length, landmarks passed in order). Use for '가장 가까운 X까지 가는 길'.",
       start=dict(S, description="start object symbol"), label=dict(S, description="English label of the goal")),
    _t("route", "Walking route from one object to another: length and landmarks passed on the way.",
       start=dict(S, description="start object symbol"), goal=dict(S, description="goal object symbol")),
]
