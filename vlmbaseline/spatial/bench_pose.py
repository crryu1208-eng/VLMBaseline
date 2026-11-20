"""Benchmark guidance and surroundings from the user's pose (keyframe camera poses stand in for the user).

    python -m vlmbaseline.spatial.bench_pose Qwen/Qwen3.5-9B [--aug outputs/augmented] [--n 8]

guide   '여기서 가장 가까운 X까지 어떻게 가?'  right when the answer names the goal of guide_me, starts with the
        right first turn (직진 / 왼쪽 / 오른쪽 / 뒤) and guide_me was called.
around  '내 왼쪽에 뭐가 있어?' (앞 / 왼쪽 / 오른쪽)  right when the first object symbol cited is on that side.
nopose  guidance asked without a known pose: right when the answer asks for the start / says it cannot tell and
        no route was computed from a start the user did not give.
Ground truth comes from SpatialIndex (code), which the camera projection check in tests/test_nav.py validates.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from vlmbaseline.bench.run import load
from vlmbaseline.scene import attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.agent import Agent
from vlmbaseline.spatial.bench_qa import KO
from vlmbaseline.spatial.tools import SpatialIndex

TURN = {"직진": r"직진|앞으로|똑바로|쭉", "왼쪽": r"왼쪽", "오른쪽": r"오른쪽", "뒤로": r"뒤로|돌아서|반대"}
SIDE = {"앞": {"앞", "왼쪽 앞", "오른쪽 앞"}, "왼쪽": {"왼쪽", "왼쪽 앞", "왼쪽 뒤"}, "오른쪽": {"오른쪽", "오른쪽 앞", "오른쪽 뒤"}}
NOPOSE = re.compile(r"현재 위치|어디에 (계|있)|출발|기준이 되는|확인할 수 없|알 수 없|알려주")


def first_turn_key(turn):
    for k in ("직진", "왼쪽", "오른쪽", "뒤로"):
        if turn.startswith(k):
            return k
    return "직진"


def answer_first_turn(ans):
    """The turn word that comes first in the answer."""
    pos = {k: m.start() for k, r in TURN.items() if (m := re.search(r, ans))}
    return min(pos, key=pos.get) if pos else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    proc, model = load(a.model, "4bit")
    from vlmbaseline.spatial.search import TextEncoder
    enc = TextEncoder()
    rows = []
    for name in a.scenes:
        s = load_export(Path("data") / name)
        attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        idx = SpatialIndex(s, enc)
        agent = Agent(proc, model, idx)
        labels = [l for l in KO if idx.labels.get(l)]
        kfs = s.keyframes[len(s.keyframes) // (2 * a.n)::max(1, len(s.keyframes) // a.n)][:a.n]
        for j, kf in enumerate(kfs):
            pose = nav.pose_from_T(kf.world_T_sensor)
            idx.pose = pose
            lab = rng.choice(labels)
            gt = idx.guide_me(label=lab)
            if "steps" in gt and gt["steps"]:
                q = f"여기서 가장 가까운 {KO[lab]}까지 어떻게 가?"
                ans, tr = agent.ask(q, pose=pose)
                key = first_turn_key(gt["steps"][0]["turn"])
                ok = (gt["goal"]["symbol"] in ans and answer_first_turn(ans) == key
                      and any(t["tool"] == "guide_me" for t in tr))
                rows.append(dict(scene=name, kf=kf.node, kind="guide", q=q, gt=gt["goal"]["symbol"], gt_turn=key,
                                 answer=ans, ok=ok, tools=[t["tool"] for t in tr]))
                print(f"[{'OK' if ok else '--'}] guide  {name} {kf.node} {q} gt={gt['goal']['symbol']}/{key} -> {ans[:120]!r}",
                      flush=True)
            idx.pose = pose
            around = idx.around_me(k=8)["objects"]
            for side in rng.sample(list(SIDE), 1):
                good = {o["symbol"] for o in around if o["direction"] in SIDE[side]}
                if not good:
                    continue
                q = f"내 {side}에 뭐가 있어?" if side != "앞" else "내 앞에 뭐가 있어?"
                ans, tr = agent.ask(q, pose=pose)
                syms = [x for x in re.findall(r"O\d+", ans) if x in idx.obj]
                ok = bool(syms) and syms[0] in good
                rows.append(dict(scene=name, kf=kf.node, kind="around", q=q, gt=sorted(good), answer=ans, ok=ok,
                                 tools=[t["tool"] for t in tr]))
                print(f"[{'OK' if ok else '--'}] around {name} {kf.node} {q} -> {ans[:120]!r}", flush=True)
        for lab in rng.sample(labels, 2):
            q = f"여기서 가장 가까운 {KO[lab]}까지 어떻게 가?"
            ans, tr = agent.ask(q, pose=None)
            # a route the guard blocked (start not given by the user) is fine; a route that ran is invented
            ran = [t for t in tr if t["tool"] in ("route", "guide_to") and "error" not in t["result"]]
            ok = bool(NOPOSE.search(ans)) and not ran
            rows.append(dict(scene=name, kind="nopose", q=q, answer=ans, ok=ok, tools=[t["tool"] for t in tr]))
            print(f"[{'OK' if ok else '--'}] nopose {name} {q} -> {ans[:120]!r}", flush=True)
    Path("outputs/bench_pose.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    out = {}
    for k in ("guide", "around", "nopose"):
        v = [r["ok"] for r in rows if r["kind"] == k]
        out[k] = f"{sum(v)}/{len(v)}"
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
