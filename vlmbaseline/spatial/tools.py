"""Spatial queries over a scene graph, computed in code (the VLM only picks the query and phrases the answer).

    idx = SpatialIndex(load_export("data/library_merged"))
    idx.count("chair"); idx.nearest("O817", "potted plant"); idx.route("O817", "O834")

Distances are horizontal (x, y). Walking distances follow the 2D floor places of the graph (Dijkstra over
place-place edges); an object is reached through the place it is attached to.
Every tool returns a JSON-serialisable dict; TOOLS holds their function-calling schemas.
"""
from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict

import numpy as np

from vlmbaseline.scene import Scene
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.search import rank


def _bi(s):
    """Two-character pieces of a text without spaces (single characters match too easily)."""
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)} - _COMMON


_COMMON = {"공간", "구역", "있는", "는곳", "있어", "어디", "가는", "으로", "까지"}  # too common to tell areas apart

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
        # areas (scene.areas, from attach_areas before building the index): place -> area id
        self.area_of = {p: a for a, r in scene.areas.items() for p in r.get("places", [])}
        self.pose = None  # (x, y, yaw) of the user, set by set_pose(); enables around_me / guide_me

    def set_pose(self, x, y, yaw):
        """User position (m, world) and heading (radians from the world x axis)."""
        self.pose = (float(x), float(y), float(yaw))

    def set_pose_from_keyframe(self, kf):
        self.pose = nav.pose_from_T(kf.world_T_sensor)

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

    def _path(self, prev, a, b):
        path, u = [b], b
        while u != a:
            u = prev[u]
            path.append(u)
        return path[::-1]

    def _pose_place(self):
        """Graph place closest to the user (only places with edges, so routes exist)."""
        x, y, _ = self.pose
        ids = [p for p in self.adj]
        c = np.array([self.place[p].center[:2] for p in ids])
        return ids[int(np.linalg.norm(c - [x, y], axis=1).argmin())]

    def _rel(self, o):
        x, y, yaw = self.pose
        v = o.center[:2] - [x, y]
        a = nav.rel_angle(yaw, v)
        return {"direction": nav.direction_word(a), "clock": nav.clock(a),
                "dist_m": round(float(np.linalg.norm(v)), 1)}

    def _area_brief(self, aid):
        r = self.s.areas.get(aid)
        if r is None:
            return None
        return {"area": aid, "name": r.get("이름"), "kind": r.get("종류")}

    def _area_arg(self, area):
        """Area by id or by name / description text."""
        if not self.s.areas:
            raise ValueError("구역 정보가 없습니다")
        a = str(area).strip()
        if re.fullmatch(r"\d+", a) and int(a) in self.s.areas:
            return int(a)
        exact = [k for k, r in self.s.areas.items() if r.get("이름") == a]
        if len(exact) == 1:
            return exact[0]
        best = self.areas(query=a, k=5)["areas"]
        if not best or best[0]["score"] < 1:
            raise KeyError(f"no area matching {area}")
        tie = [b["name"] for b in best if b["score"] == best[0]["score"]]
        if len(tie) > 1:  # e.g. '열람 공간' fits several areas: do not pick one at random
            raise ValueError(f"'{area}'에 맞는 구역이 여러 개입니다: {', '.join(tie)}. 사용자가 말한 구역 이름 전체로 다시 호출하세요.")
        return best[0]["area"]

    def _need_pose(self):
        if self.pose is None:
            raise ValueError("현재 위치를 모릅니다. 출발 물체의 기호가 필요합니다.")

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
        r.update(size_m=[round(float(v), 2) for v in o.size], area=self._area_brief(self.area_of.get(o.place)),
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

    def guide_to(self, start, label=None, area=None):
        """Nearest object of a label from the start object and the walking route to it; or the route to an area."""
        if area is not None and not label:
            a = self._o(start)
            dist, prev = self._dijkstra(a.place)
            aid = self._area_arg(area)
            saved = self.pose
            try:  # same steps as from a pose, with an unknown heading
                self.pose = (float(a.center[0]), float(a.center[1]), None)
                r = self._guide_area(a.place, dist, prev, aid)
            finally:
                self.pose = saved
            return dict(r, start=self._brief(a))
        if not label:
            raise ValueError("label or area is needed")
        n = self.nearest(start, label)["nearest"]
        if not n:
            return {"start": start, "label": label, "error": "no such object"}
        return self.route(start, n[0]["symbol"])

    SIDES = {"앞": {"앞", "왼쪽 앞", "오른쪽 앞"}, "왼쪽": {"왼쪽", "왼쪽 앞", "왼쪽 뒤"},
             "오른쪽": {"오른쪽", "오른쪽 앞", "오른쪽 뒤"}, "뒤": {"뒤", "왼쪽 뒤", "오른쪽 뒤"}}

    def _area_name_at(self, xy):
        if not self.s.areas:
            return None
        ids = list(self.adj)
        c = np.array([self.place[p].center[:2] for p in ids])
        aid = self.area_of.get(ids[int(np.linalg.norm(c - xy, axis=1).argmin())])
        return (self.s.areas.get(aid) or {}).get("이름")

    def _guide_area(self, src, dist, prev, aid):
        x, y, yaw = self.pose
        if self.area_of.get(src) == aid:
            return {"goal_area": self._area_brief(aid), "already_there": True}
        tgt = self._area_target(dist, aid)
        path = self._path(prev, src, tgt)
        pts = np.vstack([[x, y]] + [self.place[p].center[:2] for p in path])
        lm = [(c.symbol, c.label, (c.vlm or {}).get("이름"), c.center[:2], float(np.prod(c.size)))
              for c in self.s.objects if c.label not in SKIP_LANDMARK]
        st, _ = nav.steps(pts, yaw, lm, area_at=self._area_name_at)
        return {"goal_area": self._area_brief(aid), "walking_m": round(float(np.linalg.norm(pts[1] - pts[0])) + dist[tgt], 1),
                "steps": st[:8]}

    def around_me(self, label=None, side=None, k=8, radius=4.0):
        """Objects around the user with direction relative to the heading (앞, 왼쪽, 오른쪽 뒤 ...);
        side (앞 / 왼쪽 / 오른쪽 / 뒤) keeps only the objects on that side."""
        self._need_pose()
        if side and side not in self.SIDES:
            raise ValueError(f"side must be one of {list(self.SIDES)}")
        x, y, _ = self.pose
        objs = self._match(label) if label else [o for o in self.s.objects if o.label not in SKIP_LANDMARK]
        objs = sorted(objs, key=lambda o: np.linalg.norm(o.center[:2] - [x, y]))
        out = []
        for o in objs:
            r = self._rel(o)
            if r["dist_m"] > float(radius) and out:
                break
            if side and r["direction"] not in self.SIDES[side]:
                continue
            b = {"symbol": o.symbol, "label": o.label}
            if o.vlm and o.vlm.get("이름"):
                b["name"] = o.vlm["이름"]
            b.update(r)
            out.append(b)
            if len(out) >= int(k):
                break
        if side and not out:
            return {"side": side, "objects": [], "note": f"{radius} m 안에서 {side}쪽에 있는 물체가 없습니다"}
        return {"objects": out}

    def guide_me(self, label=None, goal=None, area=None):
        """Turn-by-turn directions from the user's pose to an object (goal symbol), the nearest object of a
        label by walking distance, or an area (id or name)."""
        self._need_pose()
        src = self._pose_place()
        dist, prev = self._dijkstra(src)
        if area is not None and not goal and not label:
            return self._guide_area(src, dist, prev, self._area_arg(area))
        if goal:
            o = self._o(goal)
        elif label:
            cand = [c for c in self._match(label) if c.place in dist]
            if not cand:
                return {"label": label, "error": "no reachable object with this label"}
            o = min(cand, key=lambda c: dist[c.place])
        else:
            raise ValueError("label or goal is needed")
        if o.place not in dist:
            return {"goal": o.symbol, "reachable": False}
        x, y, yaw = self.pose
        path = self._path(prev, src, o.place)
        pts = np.vstack([[x, y]] + [self.place[p].center[:2] for p in path] + [o.center[:2]])
        lm = [(c.symbol, c.label, (c.vlm or {}).get("이름"), c.center[:2], float(np.prod(c.size)))
              for c in self.s.objects if c is not o and c.label not in SKIP_LANDMARK and c.label != o.label]
        st, h = nav.steps(pts[:-1], yaw, lm, area_at=self._area_name_at)
        walk = float(np.linalg.norm(pts[1] - pts[0])) + dist[o.place]
        g = self._brief(o)
        g.pop("xy", None)
        g.pop("top_z", None)
        g["now"] = self._rel(o)  # where the goal is from the user now
        last = o.center[:2] - pts[-2]
        if h is not None and np.linalg.norm(last) > 0.7:
            g["at_arrival"] = nav.direction_word(nav.rel_angle(h, last))  # side of the goal at the end
        off = float(np.linalg.norm(pts[1] - pts[0]))
        extra = {"off_floor_m": round(off, 1), "warning": "현재 위치가 지도에 있는 바닥에서 멀리 떨어져 있습니다"} \
            if off > 1.5 else {}
        return {"goal": g, "walking_m": round(walk, 1), "steps": st[:8], **extra,
                "note": "steps are turns relative to the user's heading; 'passing' objects are on that side"}

    def areas(self, query=None, k=20):
        """Areas of the floor (name, kind, one sentence); with a query, the best matching areas first."""
        out = []
        q = _bi(query) if query else None
        for aid, r in self.s.areas.items():
            b = {"area": aid, "name": r.get("이름"), "kind": r.get("종류"), "summary": r.get("설명"),
                 "objects": sum(1 for o in self.s.objects if self.area_of.get(o.place) == aid)}
            if q:
                txt = " ".join([r.get("이름") or "", r.get("종류") or "", r.get("설명") or ""]
                               + list(r.get("특징") or []) + list(r.get("용도") or []))
                name_hit = len(q & _bi(r.get("이름"))) + len(q & _bi(r.get("종류")))
                b["score"] = round(2 * name_hit + len(q & _bi(txt)) / 2, 1)
            out.append(b)
        if q:
            out.sort(key=lambda b: -b["score"])
        return {"areas": out[:int(k)]}

    def area_info(self, area):
        """One area: name, kind, description, uses, notable things, neighbouring areas, objects in it."""
        aid = self._area_arg(area)
        r = self.s.areas[aid]
        objs = [o for o in self.s.objects if self.area_of.get(o.place) == aid and o.label not in SKIP_LANDMARK]
        return {"area": aid, "name": r.get("이름"), "kind": r.get("종류"), "description": r.get("설명"),
                "uses": r.get("용도"), "notable": r.get("특징"),
                "neighbours": [self._area_brief(n) for n in r.get("neighbours", [])],
                "objects": [{"symbol": o.symbol, "label": o.label, "name": (o.vlm or {}).get("이름")} for o in objs[:20]]}

    def where_am_i(self):
        """The area the user is in now, and the closest objects with their direction."""
        self._need_pose()
        aid = self.area_of.get(self._pose_place())
        r = {"area": self.area_info(aid) if aid is not None else None}
        if r["area"]:
            r["area"].pop("objects", None)
        r["around"] = self.around_me(k=4)["objects"]
        return r

    def _area_target(self, dist, aid):
        """The place of an area closest by walking (from a Dijkstra result)."""
        ps = [p for p in self.s.areas[aid].get("places", []) if p in dist]
        if not ps:
            raise ValueError("구역에 갈 수 없습니다")
        return min(ps, key=dist.get)

    def call(self, name, args):
        fn = {t["function"]["name"]: getattr(self, t["function"]["name"]) for t in TOOLS + POSE_TOOLS}
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
    _t("guide_to", "Guidance from a start object: to the nearest object of a label (walking route, landmarks passed) "
       "or to an area (turn-by-turn steps). Use for '가장 가까운 X까지 가는 길', 'O12에서 휴게 공간까지'.",
       start=dict(S, description="start object symbol"),
       label={"type": "string", "description": "English label of the goal", "optional": True},
       area={"type": "string", "description": "goal area: id or name / description in Korean", "optional": True}),
    _t("areas", "Areas (zones) of the floor with name, kind and summary, e.g. 복도, 열람 공간, 엘리베이터 앞. With a query "
       "(Korean words of the area such as '휴게', '엘리베이터'), the best matching areas first (score 0 = no word "
       "match: then judge from the names and summaries).",
       query={"type": "string", "description": "description of the area (optional)", "optional": True}),
    _t("area_info", "One area: name, kind, description, uses, notable things, neighbouring areas and objects in it.",
       area=dict(S, description="area id or name")),
    _t("route", "Walking route from one object to another: length and landmarks passed on the way.",
       start=dict(S, description="start object symbol"), goal=dict(S, description="goal object symbol")),
]

POSE_TOOLS = [
    _t("guide_me", "Guidance from where the user is now (their position and facing direction are known): "
       "turn-by-turn steps (turn, walk m, objects passed on the left / right, area entered). Use for "
       "'여기서 X까지 어떻게 가?', '가장 가까운 X로 안내해줘', '휴게 공간으로 데려다줘' when no start object is given.",
       label={"type": "string", "description": "English label of the goal (nearest by walking)", "optional": True},
       goal={"type": "string", "description": "goal object symbol", "optional": True},
       area={"type": "string", "description": "goal area: id or name / description in Korean", "optional": True}),
    _t("where_am_i", "The area the user is in now (name, kind, description, neighbouring areas) and the closest "
       "objects with their direction. Use for '여기가 어디야?', '지금 어디쯤이야?'."),
    _t("around_me", "Objects around the user now, with direction relative to where they face (앞, 왼쪽, 오른쪽 뒤, "
       "clock 12 = ahead) and distance. Use for '내 앞에/왼쪽에 뭐 있어?', '근처에 X 있어?'.",
       label={"type": "string", "description": "only this English label (optional)", "optional": True},
       side={"type": "string", "enum": ["앞", "왼쪽", "오른쪽", "뒤"],
             "description": "only objects on this side of the user (optional)", "optional": True}),
]
