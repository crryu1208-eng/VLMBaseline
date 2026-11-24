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

from vlmbaseline.demo.eventlog import EventLog, compact_trace, wav_seconds
from vlmbaseline.scene import attach_areas, attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.follow import Follower
from vlmbaseline.spatial.tools import SpatialIndex

HERE = Path(__file__).parent
STT_HINT = False  # docs/bench_2026-11-19.md, voice input


# said once when the guidance holds because the pose may be wrong (live/posecore.py, by the reason)
HOLD_SAY = {"jump": "위치를 다시 확인하고 있습니다. 천천히 걸어 주세요.",
            "lost": "위치가 정확하지 않을 수 있습니다. 지도와 다시 맞출 때까지 안내를 멈춥니다. 주변을 천천히 둘러봐 주세요.",
            "far": "위치가 정확하지 않을 수 있습니다. 지도와 다시 맞출 때까지 안내를 멈춥니다."}

class Demo:
    def __init__(self, scenes, model_id, aug, use_model=True, use_tts=True, use_stt=True, log=None):
        self.aug = Path(aug)
        self.log = log  # eventlog.EventLog or None
        self.speaker = self.listener = None
        if use_tts:
            from vlmbaseline.demo.tts import Speaker
            self.speaker = Speaker()  # loads in the background
        if use_stt:
            from vlmbaseline.demo.stt import Listener
            self.listener = Listener()  # loads in the background
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
            attach_areas(s, self.aug / name / "areas_vlm.json")
            self.idx[name] = SpatialIndex(s, enc)
            if model is not None:
                from vlmbaseline.spatial.agent import Agent
                self.agent[name] = Agent(proc, model, self.idx[name])
        self.stt_hint = {}  # scene -> area and object names given to Whisper (stt.scene_hint), when STT_HINT
        if self.listener is not None and STT_HINT:
            from vlmbaseline.demo.stt import scene_hint
            self.stt_hint = {n: scene_hint(i) for n, i in self.idx.items()}
        self.follow = {}  # scene -> Follower of the last guidance from the user's pose
        self.lock = threading.Lock()  # one GPU, one question at a time
        self.flock = threading.Lock()  # followers (pose updates must not wait for a question)
        self.live = {}  # scene -> latest pose from the mapping pipeline (live/bridge.py) and announcements

    # ------------------------------------------------------------ map data
    def scene_json(self, name):
        idx = self.idx[name]
        s = idx.s
        r2 = lambda v: [round(float(x), 2) for x in v]
        places = [{"id": str(p.id), "c": r2(p.center[:2]), "b": [r2(q[:2]) for q in p.boundary]} for p in s.places]
        edges = [[r2(idx.place[a].center[:2]), r2(idx.place[b].center[:2])] for a, b in s.edges
                 if a in idx.place and b in idx.place]
        objs = []
        for o in s.objects:
            v = o.vlm or {}
            objs.append({"sym": o.symbol, "label": o.label, "c": r2(o.center[:2]), "s": r2(o.size[:2]),
                         "name": v.get("이름"), "state": v.get("상태"), "desc": v.get("설명"),
                         "check": bool(v.get("점검_필요")),
                         "view": (self.aug / name / "views" / f"{o.symbol}.jpg").exists()})
        areas = [{"id": a, "name": r.get("이름"), "kind": r.get("종류"), "desc": r.get("설명"), "c": r.get("center"),
                  "places": [str(p) for p in r.get("places", [])]} for a, r in s.areas.items()]
        return {"name": name, "places": places, "areas": areas, "edges": edges, "objects": objs,
                "keyframes": [list(nav.pose_from_T(k.world_T_sensor)) for k in s.keyframes[::10]]}

    # ------------------------------------------------------------ route polyline for the map
    def route_points(self, idx, tool, args, result):
        if not isinstance(result, dict) or "error" in result:
            return None
        if isinstance(result.get("goal_area"), dict) and "steps" in result:
            return self._area_route(idx, tool, result)
        if not isinstance(result.get("goal"), dict):
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

    def _area_route(self, idx, tool, result):
        if tool == "guide_me":
            if idx.pose is None:
                return None
            src, head = idx._pose_place(), [list(idx.pose[:2])]
        else:
            a = idx.obj.get((result.get("start") or {}).get("symbol"))
            if a is None:
                return None
            src, head = a.place, [a.center[:2].tolist()]
        dist, prev = idx._dijkstra(src)
        tgt = idx._area_target(dist, result["goal_area"]["area"])
        pts = head + [idx.place[p].center[:2].tolist() for p in idx._path(prev, src, tgt)]
        return [[round(x, 2), round(y, 2)] for x, y in pts]

    # ------------------------------------------------------------ ask
    def ask(self, name, q, pose, history=None):
        idx = self.idx[name]
        t = time.perf_counter()
        with self.lock:
            if name in self.agent:
                ans, trace = self.agent[name].ask(q, pose=pose, history=history)
            else:
                idx.pose = pose
                ans, trace = "(모델 없이 실행 중: 도구 결과만 보여줍니다)", self._tools_only(idx, q)
            route = None
            for tr in reversed(trace):
                if tr["tool"] in ("guide_me", "route", "guide_to"):
                    route = self.route_points(idx, tr["tool"], tr["args"], tr["result"])
                    if route:
                        break
        goal_area = next((tr["result"]["goal_area"]["area"] for tr in reversed(trace)
                          if isinstance(tr.get("result"), dict) and isinstance(tr["result"].get("goal_area"), dict)), None)
        warn = next((tr["result"]["warning"] for tr in trace
                     if isinstance(tr.get("result"), dict) and tr["result"].get("warning")), None)
        cited = [x for x in dict.fromkeys(re.findall(r"O\d+", ans)) if x in idx.obj]
        follow = self._start_follow(name, trace, pose)
        self.ev("ask", scene=name, q=q, pose=pose, history=len(history or []), answer=ans, tools=compact_trace(trace),
                follow=follow and follow["goal"], sec=round(time.perf_counter() - t, 1))
        return {"answer": ans, "trace": trace, "route": route, "cited": cited, "warning": warn, "goal_area": goal_area,
                "follow": follow, "sec": round(time.perf_counter() - t, 1)}

    # ------------------------------------------------------------ guidance while walking
    def _start_follow(self, name, trace, pose):
        """After guidance from the user's pose, keep a Follower so pose updates get announcements. Any other answer
        (a question asked while walking, '그 옆에 뭐가 있어?') leaves the current guidance running."""
        if pose is None:
            return None
        for tr in reversed(trace):
            r = tr.get("result")
            if tr["tool"] != "guide_me" or not isinstance(r, dict) or "error" in r:
                continue
            if isinstance(r.get("goal"), dict) and r["goal"].get("symbol"):
                f = Follower(self.idx[name], goal=r["goal"]["symbol"])
            elif isinstance(r.get("goal_area"), dict):
                f = Follower(self.idx[name], area=r["goal_area"]["area"])
            else:
                continue
            with self.flock:
                st = f.start(pose)
                self.follow[name] = f
            lv = self.live.get(name)
            if lv is not None:
                lv["state"] = None  # the page shows the new guidance from the answer, not the last one's state
            self.ev("follow_start", scene=name, goal=f.name, remaining_m=st["remaining_m"], pose=pose)
            return {"goal": f.name, "route": st["route"], "remaining_m": st["remaining_m"]}
        return None

    def stop_follow(self, name):
        if self.follow.pop(name, None) is not None:
            self.ev("follow_stop", scene=name)
        if name in self.live:
            self.live[name]["state"] = None

    def follow_update(self, name, pose):
        f = self.follow.get(name)
        if f is None:
            return {"error": "길 안내를 먼저 받아야 합니다"}
        with self.flock:
            return f.update(pose)

    # ------------------------------------------------------------ live pose (mapping pipeline)
    def ev(self, kind, **kw):
        if self.log is not None:
            try:
                self.log.add(kind, **kw)
            except OSError as e:  # a full disk must not stop the guidance
                print("log:", e, flush=True)

    def _say(self, lv, texts, why=None, scene=None):
        for t in texts:
            self.ev("say", scene=scene, text=t, why=why)
            lv["seq"] += 1
            lv["say"].append([lv["seq"], t])
        del lv["say"][:-50]

    def live_post(self, rec):
        """A pose from live/bridge.py: keep it; with an active guidance, update it and queue the announcements."""
        name = rec["scene"]
        lv = self.live.setdefault(name, {"seq": 0, "say": []})
        unc = bool(rec.get("uncertain"))
        now = time.time()
        state = "ok" if rec.get("localized") and not unc else f"uncertain:{rec.get('why')}" if unc else "searching"
        gap = now - lv["t"] if "t" in lv else 0
        if state != lv.get("logstate") or gap > 3:
            lv["logstate"] = state
            self.ev("live", scene=name, state=state, pose=rec.get("pose"), since_loc_s=rec.get("since_loc_s"),
                    gap_s=round(gap, 1) if gap > 3 else None)
        lv.update(t=now, localized=bool(rec.get("localized")), since_loc_s=rec.get("since_loc_s"), uncertain=unc,
                  why=rec.get("why"))
        if not rec.get("localized") or not rec.get("pose"):
            return {"ok": True, "follow": None}
        lv["pose"] = [float(v) for v in rec["pose"]]
        st = None
        f = self.follow.get(name)
        if f is not None and not f.arrived and unc:
            # the pose may be wrong (live/posecore.py): no turn, reroute or arrival from it; say so once
            if not lv.get("held"):
                lv["held"] = True
                self._say(lv, [HOLD_SAY.get(rec.get("why"), HOLD_SAY["jump"])], "hold", name)
            return {"ok": True, "follow": None, "held": True}
        if lv.get("held") and f is not None and not f.arrived:
            self._say(lv, ["위치를 다시 맞췄습니다. 안내를 이어 갑니다."], "resume", name)
        lv["held"] = False
        if f is not None and not f.arrived:
            st = self.follow_update(name, tuple(lv["pose"]))
            self._say(lv, st.get("say", []), "follow", name)
            if st.get("arrived"):
                self.ev("arrived", scene=name, goal=f.name, pose=lv["pose"])
            lv["state"] = {k: st.get(k) for k in ("next", "remaining_m", "arrived", "goal", "route")}
        return {"ok": True, "follow": st}

    def live_get(self, name, after=0):
        lv = self.live.get(name)
        if lv is None:
            return {"live": False}
        return {"live": True, "age_s": round(time.time() - lv["t"], 1), "localized": lv.get("localized"),
                "uncertain": lv.get("uncertain", False), "why": lv.get("why"),
                "since_loc_s": lv.get("since_loc_s"), "pose": lv.get("pose"), "state": lv.get("state"),
                "say": [x for x in lv["say"] if x[0] > after], "seq": lv["seq"]}

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
                return self._send(200, {"scenes": list(demo.idx), "model": bool(demo.agent),
                                        "tts": bool(demo.speaker and demo.speaker.ready),
                                        "stt": bool(demo.listener and demo.listener.ready)})
            if u.path == "/api/tts":  # the guidance sentence as speech (wav), made on this computer
                w = demo.speaker.wav(qs.get("text", "")) if demo.speaker else None
                if w is None:
                    return self._send(503, {"error": "speech not loaded"})
                return self._send(200, w, "audio/wav")
            if u.path == "/api/scene" and qs.get("name") in demo.idx:
                return self._send(200, demo.scene_json(qs["name"]))
            if u.path == "/api/live" and qs.get("scene") in demo.idx:
                return self._send(200, demo.live_get(qs["scene"], int(qs.get("after", 0) or 0)))
            if u.path == "/api/view" and qs.get("name") in demo.idx and re.fullmatch(r"O\d+", qs.get("sym", "")):
                f = demo.aug / qs["name"] / "views" / f"{qs['sym']}.jpg"
                if f.exists():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/live":
                try:
                    rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    if rec.get("scene") not in demo.idx:
                        return self._send(400, {"error": "scene"})
                    return self._send(200, demo.live_post(rec))
                except Exception as e:
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            if path == "/api/stt":  # a spoken question (wav) -> text, recognized on this computer
                n = int(self.headers.get("Content-Length", 0))
                if not demo.listener or n > 5_000_000:
                    return self._send(503 if not demo.listener else 413, {"error": "voice input"})
                sc = parse_qs(urlparse(self.path).query).get("scene", [None])[0]
                t0 = time.perf_counter()
                wav = self.rfile.read(n)
                try:
                    t = demo.listener.text(wav, demo.stt_hint.get(sc))
                except Exception as e:
                    demo.ev("stt_error", scene=sc, error=f"{type(e).__name__}: {e}")
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
                demo.ev("stt", scene=sc, text=t, audio_s=wav_seconds(wav),
                        sec=round(time.perf_counter() - t0, 2), audio=demo.log.audio(wav) if demo.log else None)
                if t is None:
                    return self._send(503, {"error": "voice input not loaded"})
                return self._send(200, {"text": t})
            if path == "/api/event":  # said by the page itself (searching, signal lost, back)
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(min(n, 4000)) or b"{}")
                if req.get("kind") == "page_say" and req.get("scene") in demo.idx:
                    demo.ev("page_say", scene=req["scene"], text=str(req.get("text", ""))[:300])
                return self._send(200, {"ok": True})
            if path == "/api/follow_stop":
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                demo.stop_follow(req.get("scene"))
                return self._send(200, {"ok": True})
            if path == "/api/follow":
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    if req.get("scene") not in demo.idx:
                        return self._send(400, {"error": "scene"})
                    pose = tuple(float(v) for v in req["pose"])
                    st = demo.follow_update(req["scene"], pose)
                    for t in st.get("say", []):
                        demo.ev("say", scene=req["scene"], text=t, why="follow_map", pose=pose)
                    return self._send(200, st)
                except Exception as e:
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            if path != "/api/ask":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                name, q = req["scene"], str(req["q"]).strip()[:300]
                if name not in demo.idx or not q:
                    return self._send(400, {"error": "scene / question"})
                pose = req.get("pose")
                pose = tuple(float(v) for v in pose) if pose else None
                hist = [{"q": str(h.get("q", ""))[:300], "a": str(h.get("a", ""))[:1500]}
                        for h in (req.get("history") or [])[-3:] if isinstance(h, dict)]
                self._send(200, demo.ask(name, q, pose, hist))
            except Exception as e:  # report to the page instead of dropping the connection
                demo.ev("ask_error", q=locals().get("q"), error=f"{type(e).__name__}: {e}")
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
    ap.add_argument("--no-tts", action="store_true", help="no speech on the server (the page uses the browser's voice)")
    ap.add_argument("--no-stt", action="store_true", help="no voice input (Whisper on the GPU, ~1.8 GB)")
    ap.add_argument("--no-log", action="store_true", help="no session log (outputs/logs/, demo/eventlog.py)")
    ap.add_argument("--keep-audio", action="store_true", help="keep the spoken questions next to the log")
    a = ap.parse_args()
    log = None if a.no_log else EventLog(keep_audio=a.keep_audio)
    demo = Demo(a.scenes, a.model, a.aug, use_model=not a.no_model, use_tts=not a.no_tts, use_stt=not a.no_stt,
                log=log)
    demo.ev("start", scenes=a.scenes, model=None if a.no_model else a.model, tts=not a.no_tts, stt=not a.no_stt,
            keep_audio=a.keep_audio)
    if log:
        print("log:", log.path, flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(demo))
    print(f"http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
