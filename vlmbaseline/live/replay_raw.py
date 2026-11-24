"""Recompute the live pose offline from the bridge's raw log (bridge.py --raw), with other jump-check settings.

    python -m vlmbaseline.live.replay_raw <raw.jsonl> --align <align.json> -o poses.jsonl [--no-gate] [--jump-m 1.5] ...

The inputs are fed in the order the bridge received them, one pose every 0.5 s of odometry time (the bridge's
2 Hz timer), so the output is the bridge's poses.jsonl with these settings (score it with live/bench_live.py).
"""
from __future__ import annotations

import argparse
import json

from vlmbaseline.live.posecore import PoseCore, unflat


def replay(raw_path, align, scene=None, rate=2.0, **kw):
    core = PoseCore(align, **kw)
    scene = scene or align["scene"]
    out, next_t = [], None
    for line in open(raw_path):
        e = json.loads(line)
        if e["k"] == "o":
            core.on_odom(e["t"], unflat(e["T"]))
            if next_t is None:
                next_t = e["t"]
            while e["t"] >= next_t:
                rec = core.pose()
                if rec is not None:
                    out.append({"scene": scene, **rec})
                next_t += int(1e9 / rate)
        elif e["k"] == "lost":
            core.on_odom(e["t"], None, lost=True)
        elif e["k"] == "l":
            core.on_loc(e["t"], unflat(e["T"]))
        elif e["k"] == "i":
            core.on_info(e["t"], e["n"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--align", required=True)
    ap.add_argument("--scene")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--jump-m", type=float, default=2.0)
    ap.add_argument("--confirm", type=int, default=3)
    ap.add_argument("--trust-m", type=float, default=25.0)
    ap.add_argument("--stale-m", type=float, default=0.0, help="odometry alone this far: uncertain (0: never)")
    a = ap.parse_args()
    recs = replay(a.raw, json.load(open(a.align)), a.scene, gate=not a.no_gate, jump_m=a.jump_m,
                  confirm=a.confirm, trust_m=a.trust_m,
                  stale_m=a.stale_m or None)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} poses -> {a.out}")


if __name__ == "__main__":
    main()
