"""Benchmark follow-up questions: two-turn conversations where the second question points back ('거기', '그 옆').

    python -m vlmbaseline.spatial.bench_dialog Qwen/Qwen3.5-9B [--aug outputs/augmented] [--no-history]

go_there    (pose) '여기서 가장 가까운 X 어디 있어?' -> '거기까지 어떻게 가?'
            right when guidance went to an object of label X named in the first answer.
beside      '<X> 어디 있어?' -> '그 옆에 뭐가 있어?'  right when object_info / nearest ran on an object of the first answer.
from_there  '<X> 어디 있어?' -> '거기서 가장 가까운 Y까지 가는 길 알려줘.'  right when route / guide_to ran from an
            object of the first answer.
area_go     (pose) '<area>은 어떤 곳이야?' -> '거기로 가는 길 알려줘.'  right when guidance went to that area.
other       (pose) '여기서 가장 가까운 X 어디 있어?' -> '다른 거는 없어?'  right when the answer names another X.
switch      any first turn -> '<X>는 몇 개 있어?'  right when the count is in the answer (history must not confuse it).
no_start    '<X> 어디 있어?' -> 'Y까지 가는 길 알려줘.' (no pose, no '거기')  right when no route was computed from an
            object the user did not give as start.
--no-history asks the second question alone (shows what the conversation adds).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from vlmbaseline.bench.run import load
from vlmbaseline.scene import attach_areas, attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.agent import Agent
from vlmbaseline.spatial.bench_areas import went_to
from vlmbaseline.spatial.bench_qa import KO
from vlmbaseline.spatial.search import TextEncoder
from vlmbaseline.spatial.tools import SpatialIndex

GUIDE = ("guide_me", "guide_to", "route")


def syms(idx, text, label=None):
    out = [x for x in dict.fromkeys(re.findall(r"O\d+", text)) if x in idx.obj]
    return [x for x in out if label is None or idx.obj[x].label == label]


def ran(trace, tools):
    return [t for t in trace if t["tool"] in tools and isinstance(t["result"], dict) and "error" not in t["result"]]


def goal_of(t):
    g = t["result"].get("goal")
    return g.get("symbol") if isinstance(g, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--n", type=int, default=3, help="dialogues per kind and scene")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-history", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    proc, model = load(a.model, "4bit")
    enc = TextEncoder()
    rows = []

    for name in a.scenes:
        s = load_export(Path("data") / name)
        attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        attach_areas(s, Path(a.aug) / name / "areas_vlm.json")
        idx = SpatialIndex(s, enc)
        agent = Agent(proc, model, idx)
        labels = [l for l in KO if idx.labels.get(l)]
        many = [l for l in labels if idx.labels[l] >= 2]
        poses = [nav.pose_from_T(k.world_T_sensor) for k in s.keyframes]
        uniq = {}
        for aid, r in s.areas.items():
            uniq.setdefault(r.get("이름"), []).append(aid)
        area_names = [n for n, ids in uniq.items() if n and len(ids) == 1]

        def dialog(kind, q1, q2, score, pose=None):
            a1, t1 = agent.ask(q1, pose=pose)
            hist = None if a.no_history else [{"q": q1, "a": a1}]
            a2, t2 = agent.ask(q2, pose=pose, history=hist)
            ok = bool(score(a1, t1, a2, t2))
            rows.append(dict(scene=name, kind=kind, q1=q1, a1=a1, q2=q2, a2=a2, ok=ok,
                             tools1=[t["tool"] for t in t1], tools2=[t["tool"] for t in t2],
                             args2=[t["args"] for t in t2]))
            print(f"[{'OK' if ok else '--'}] {kind:10s} {name} | {q1} -> {a1[:70]!r} | {q2} -> {a2[:90]!r} "
                  f"{[t['tool'] for t in t2]}", flush=True)

        for _ in range(a.n):
            lab = rng.choice(labels)
            dialog("go_there", f"여기서 가장 가까운 {KO[lab]} 어디 있어?", "거기까지 어떻게 가?",
                   lambda a1, t1, a2, t2: any(goal_of(t) in syms(idx, a1) for t in ran(t2, GUIDE)),
                   pose=rng.choice(poses))
        for _ in range(a.n):
            lab = rng.choice(labels)
            dialog("beside", f"{KO[lab]} 어디 있어?", "그 옆에 뭐가 있어?",
                   lambda a1, t1, a2, t2: any(str(t["args"].get("symbol", "")).upper() in syms(idx, a1)
                                              for t in ran(t2, ("object_info", "nearest"))))
        for _ in range(a.n):
            lab, lab2 = rng.sample(labels, 2)
            dialog("from_there", f"{KO[lab]} 어디 있어?", f"거기서 가장 가까운 {KO[lab2]}까지 가는 길 알려줘.",
                   lambda a1, t1, a2, t2: any(str(t["args"].get("start", "")).upper() in syms(idx, a1)
                                              for t in ran(t2, ("route", "guide_to"))))
        for tgt in rng.sample(area_names, min(a.n, len(area_names))):
            aid = uniq[tgt][0]
            for _ in range(10):  # start outside the area
                pose = rng.choice(poses)
                idx.pose = pose
                if idx.area_of.get(idx._pose_place()) != aid:
                    break
            dialog("area_go", f"{tgt}은 어떤 곳이야?", "거기로 가는 길 알려줘.",
                   lambda a1, t1, a2, t2, aid=aid: went_to(t2, aid), pose=pose)
        for _ in range(a.n):
            lab = rng.choice(many)
            dialog("other", f"여기서 가장 가까운 {KO[lab]} 어디 있어?", "다른 거는 없어?",
                   lambda a1, t1, a2, t2, lab=lab: bool(set(syms(idx, a2, lab)) - set(syms(idx, a1, lab))),
                   pose=rng.choice(poses))
        for _ in range(a.n):
            lab, lab2 = rng.sample(labels, 2)
            n2 = idx.count(lab2)["count"]
            dialog("switch", f"{KO[lab]} 어디 있어?", f"{KO[lab2]}는 몇 개 있어?",
                   lambda a1, t1, a2, t2, n2=n2: re.search(rf"(?<!\d){n2}\s*(개|대|곳)", a2))
        for _ in range(a.n):
            lab, lab2 = rng.sample(labels, 2)
            dialog("no_start", f"{KO[lab]} 어디 있어?", f"{KO[lab2]}까지 가는 길 알려줘.",
                   lambda a1, t1, a2, t2: not ran(t2, ("route", "guide_to")))

    tag = "_nohist" if a.no_history else ""
    Path(f"outputs/bench_dialog{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    out = {}
    for k in dict.fromkeys(r["kind"] for r in rows):
        v = [r["ok"] for r in rows if r["kind"] == k]
        out[k] = f"{sum(v)}/{len(v)}"
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
