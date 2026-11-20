"""Benchmark guidance while walking, replaying the recorded camera trajectories as a user walking (no model).

    python -m vlmbaseline.spatial.bench_follow [--n 40]

A trial: a stretch of the recording (8-35 m of walking); the goal is an object within 1.5 m of where the stretch
ends; the keyframe poses of the stretch are fed in order (about every 0.25 m, as a 2 Hz stream would), then a few
steps straight to the goal. Only stretches where the walk was close to the planned route are used
(walked <= 1.3 x planned + 2 m), so the walker 'follows' the guidance.

arrived     arrival announced, and not while still more than 2.5 m from the goal
reroutes    'left the route' announcements in these trials (each is a false alarm: the walker followed)
turns       a left / right turn said at a corner, and the walker's direction over the next 3 m turned that way
wrong_way   the same start, walking the recording backwards (away from the goal): '반대 방향' or a new route
            within 6 m of walking
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from vlmbaseline.scene import attach_areas, attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.follow import Follower
from vlmbaseline.spatial.tools import SKIP_LANDMARK, SpatialIndex


def walked(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])


def at_walk(P, cum, k, m):
    """Index of the pose m metres of walking after pose k (clamped)."""
    j = int(np.searchsorted(cum, cum[k] + m))
    return min(max(j, 0), len(P) - 1)


def side_of(turn):
    return 1 if turn.startswith("왼쪽") else -1 if turn.startswith("오른쪽") else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--scenes", nargs="+", default=["library_merged", "lab_03"])
    ap.add_argument("--n", type=int, default=40, help="trials per scene")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows, times = [], []
    for name in a.scenes:
        s = load_export(Path("data") / name)
        attach_vlm(s, Path(a.aug) / name / "objects_vlm.json")
        attach_areas(s, Path(a.aug) / name / "areas_vlm.json")
        idx = SpatialIndex(s)
        P = np.array([nav.pose_from_T(k.world_T_sensor) for k in s.keyframes])
        cum = walked(P)
        objs = [o for o in s.objects if o.label not in SKIP_LANDMARK]
        oxy = np.array([o.center[:2] for o in objs])
        got, tries = 0, 0
        while got < a.n and tries < 4000:
            tries += 1
            i = rng.randrange(len(P) - 10)
            j = at_walk(P, cum, i, rng.uniform(8, 35))
            if cum[j] - cum[i] < 8:
                continue
            dd = np.linalg.norm(oxy - P[j, :2], axis=1)
            near = [k for k in np.argsort(dd)[:5] if dd[k] <= 1.5]
            if not near:
                continue
            goal = objs[near[0]]
            f = Follower(idx, goal=goal.symbol)
            try:
                first = f.start(P[i])
            except (ValueError, KeyError):
                continue
            plan = f.total
            if not (cum[j] - cum[i] <= 1.3 * plan + 2.0 and plan > 4.0):
                continue
            got += 1
            said = [(0, first["say"])]
            stream = [tuple(P[k]) for k in range(i + 1, j + 1)]
            last = np.asarray(P[j, :2])
            v = goal.center[:2] - last
            n = int(np.linalg.norm(v) / 0.25)
            yaw = math.atan2(v[1], v[0])
            stream += [(*(last + v * (t + 1) / max(n, 1)), yaw) for t in range(n)]
            arrived_at, reroutes, turn_ok, turn_n = None, 0, 0, 0
            for k, pose in enumerate(stream):
                t0 = time.perf_counter()
                r = f.update(pose)
                times.append(time.perf_counter() - t0)
                if r.get("rerouted"):
                    reroutes += 1
                if r["say"]:
                    said.append((k + 1, r["say"]))
                for msg in r["say"]:
                    trn = next((t for t in ("왼쪽으로 도세요", "오른쪽으로 도세요", "왼쪽으로 비스듬히", "오른쪽으로 비스듬히")
                                if msg.startswith(t)), None)
                    if trn and i + 1 + k < j:  # judge on the recorded part only
                        kk = i + 1 + k
                        a0 = at_walk(P, cum, kk, 1.5)  # the corner is ~2 m ahead when this is said
                        a1 = at_walk(P, cum, kk, 5.0)
                        b0 = max(0, int(np.searchsorted(cum, cum[kk] - 2.0)))
                        before, after = P[a0, :2] - P[b0, :2], P[a1, :2] - P[a0, :2]
                        if np.linalg.norm(before) > 0.5 and np.linalg.norm(after) > 0.5 and a1 < j:
                            ang = nav.rel_angle(math.atan2(before[1], before[0]), after)
                            turn_n += 1
                            turn_ok += int(abs(ang) > 15 and np.sign(ang) == side_of(trn))
                if r["arrived"] and arrived_at is None:
                    arrived_at = float(np.linalg.norm(goal.center[:2] - np.asarray(pose[:2])))
            ok_arrive = arrived_at is not None and arrived_at <= 2.5
            # wrong way: walk the recording backwards from the same start
            g = Follower(idx, goal=goal.symbol)
            g.start(P[i])
            back = at_walk(P, cum, i, 0)
            k0 = int(np.searchsorted(cum, cum[i] - 8.0))
            warned = None
            for k in range(i - 1, max(k0, 0) - 1, -1):
                mv = P[k, :2] - P[k + 1, :2]
                pose = (P[k, 0], P[k, 1], math.atan2(mv[1], mv[0]) if np.linalg.norm(mv) > 1e-3 else P[k, 2] + math.pi)
                r = g.update(pose)
                if any("반대 방향" in m or "새 길" in m for m in r["say"]):
                    warned = cum[i] - cum[k]
                    break
            # the backward walk only tests something when it moves away from the goal along the route
            away = k0 >= 0 and i > 0 and np.linalg.norm(goal.center[:2] - P[max(k0, 0), :2]) > \
                np.linalg.norm(goal.center[:2] - P[i, :2]) + 2.0
            rows.append(dict(scene=name, start=s.keyframes[i].node, end=s.keyframes[j].node, goal=goal.symbol,
                             planned_m=round(plan, 1), walked_m=round(float(cum[j] - cum[i]), 1),
                             arrived_at_m=None if arrived_at is None else round(arrived_at, 1), ok_arrive=ok_arrive,
                             reroutes=reroutes, turns=[turn_ok, turn_n], wrong_way_test=bool(away),
                             wrong_way_warned_m=None if warned is None else round(float(warned), 1),
                             said=[[k, m] for k, ms in said for m in ms]))
            print(f"{name} {s.keyframes[i].node}->{s.keyframes[j].node} goal {goal.symbol} plan {plan:.1f} walk "
                  f"{cum[j]-cum[i]:.1f}: arrive {'OK' if ok_arrive else '--'} ({arrived_at}), reroutes {reroutes}, "
                  f"turns {turn_ok}/{turn_n}, wrong way {'-' if not away else warned}", flush=True)
    Path("outputs/bench_follow.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    ww = [r for r in rows if r["wrong_way_test"]]
    out = {"trials": len(rows),
           "arrived": f"{sum(r['ok_arrive'] for r in rows)}/{len(rows)}",
           "false_reroutes": sum(r["reroutes"] for r in rows),
           "trials_with_reroute": sum(r["reroutes"] > 0 for r in rows),
           "turns": f"{sum(r['turns'][0] for r in rows)}/{sum(r['turns'][1] for r in rows)}",
           "wrong_way_warned": f"{sum(r['wrong_way_warned_m'] is not None and r['wrong_way_warned_m'] <= 6 for r in ww)}/{len(ww)}",
           "update_ms_median": round(1000 * float(np.median(times)), 2),
           "update_ms_max": round(1000 * float(np.max(times)), 1)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
