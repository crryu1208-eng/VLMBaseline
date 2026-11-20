"""Local web page to try the agent: top-down map, set your position / heading, ask in Korean, see the route.

    python -m vlmbaseline.demo.server [--model Qwen/Qwen3.5-9B] [--port 8765] [--scenes library_merged lab_03]
    open http://127.0.0.1:8765

--no-model serves the map and runs only the tools (for checking the page without a GPU).
Listens on 127.0.0.1 only: the page shows recorded images of the space.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from vlmbaseline.scene import attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.tools import SpatialIndex

HERE = Path(__file__).parent


class Demo:
    def __init__(self, scenes, model_id, aug, use_model=True):
        self.aug = Path(aug)
        self.idx, self.agent = {}, {}
        enc = None
        try:
            from vlmbaseline.spatial.search import TextEncoder
            enc = TextEncoder()
        except Exception as e:  # search still works on descriptions only
            print("text encoder not loaded:", e)
        proc = model = None
        if use_model:
            from vlmbaseline.bench.run import load
            proc, model = load(model_id, "4bit")
        for name in scenes:
            s = load_export(Path("data") / name)
            attach_vlm(s, self.aug / name / "objects_vlm.json")
            self.idx[name] = SpatialIndex(s, enc)
            if model is not None:
                from vlmbaseline.spatial.agent import Agent
                self.agent[name] = Agent(proc, model, self.idx[name])
        self.lock = threading.Lock()  # one GPU, one question at a time

    # ------------------------------------------------------------ map data
    def scene_json(self, name):
        idx = self.idx[name]
        s = idx.s
        r2 = lambda v: [round(float(x), 2) for x in v]
        places = [{"id": p.id, "c": r2(p.center[:2]), "b": [r2(q[:2]) for q in p.boundary]} for p in s.places]
        edges = [[r2(idx.place[a].center[:2]), r2(idx.place[b].center[:2])] for a, b in s.edges
                 if a in idx.place and b in idx.place]
        objs = []
        for o in s.objects:
            v = o.vlm or {}
            objs.append({"sym": o.symbol, "label": o.label, "c": r2(o.center[:2]), "s": r2(o.size[:2]),
                         "name": v.get("이름"), "state": v.get("상태"), "desc": v.get("설명"),
                         "check": bool(v.get("점검_필요")),
                         "view": (self.aug / name / "views" / f"{o.symbol}.jpg").exists()})
        return {"name": name, "places": places, "edges": edges, "objects": objs,
                "keyframes": [list(nav.pose_from_T(k.world_T_sensor)) for k in s.keyframes[::10]]}

    # ------------------------------------------------------------ route polyline for the map
    def route_points(self, idx, tool, args, result):
        if not isinstance(result, dict) or "error" in result or not isinstance(result.get("goal"), dict):
            return None
        goal = idx.obj.get(result["goal"]["symbol"])
        if goal is None:
            return None
        if tool == "guide_me":
            if idx.pose is None:
                return None
            src, head = idx._pose_place(), [list(idx.pose[:2])]
        else:
            st = result.get("start", {})
            a = idx.obj.get(st.get("symbol") if isinstance(st, dict) else None)
            if a is None:
                return None
            src, head = a.place, [a.center[:2].tolist()]
        dist, prev = idx._dijkstra(src)
        if goal.place not in dist:
            return None
        path = idx._path(prev, src, goal.place)
        pts = head + [idx.place[p].center[:2].tolist() for p in path] + [goal.center[:2].tolist()]
        return [[round(x, 2), round(y, 2)] for x, y in pts]

    # ------------------------------------------------------------ ask
    def ask(self, name, q, pose):
        idx = self.idx[name]
        t = time.perf_counter()
        with self.lock:
            if name in self.agent:
                ans, trace = self.agent[name].ask(q, pose=pose)
            else:
                idx.pose = pose
                ans, trace = "(모델 없이 실행 중: 도구 결과만 보여줍니다)", self._tools_only(idx, q)
            route = None
            for tr in reversed(trace):
                if tr["tool"] in ("guide_me", "route", "guide_to"):
                    route = self.route_points(idx, tr["tool"], tr["args"], tr["result"])
                    if route:
                        break
        warn = next((tr["result"]["warning"] for tr in trace
                     if isinstance(tr.get("result"), dict) and tr["result"].get("warning")), None)
        cited = [x for x in dict.fromkeys(re.findall(r"O\d+", ans)) if x in idx.obj]
        return {"answer": ans, "trace": trace, "route": route, "cited": cited, "warning": warn,
                "sec": round(time.perf_counter() - t, 1)}

    @staticmethod
    def _tools_only(idx, q):
        res = idx.around_me() if idx.pose is not None else idx.search(q)
        return [{"tool": "around_me" if idx.pose is not None else "search", "args": {}, "result": res}]


def make_handler(demo: Demo):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False, default=_np).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/scenes":
                return self._send(200, {"scenes": list(demo.idx), "model": bool(demo.agent)})
            if u.path == "/api/scene" and qs.get("name") in demo.idx:
                return self._send(200, demo.scene_json(qs["name"]))
            if u.path == "/api/view" and qs.get("name") in demo.idx and re.fullmatch(r"O\d+", qs.get("sym", "")):
                f = demo.aug / qs["name"] / "views" / f"{qs['sym']}.jpg"
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/ask":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                name, q = req["scene"], str(req["q"]).strip()[:300]
                if name not in demo.idx or not q:
                    return self._send(400, {"error": "scene / question"})
                pose = req.get("pose")
                pose = tuple(float(v) for v in pose) if pose else None
                self._send(200, demo.ask(name, q, pose))
            except Exception as e:  # report to the page instead of dropping the connection
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    demo = Demo(a.scenes, a.model, a.aug, use_model=not a.no_model)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(demo))
    print(f"http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
