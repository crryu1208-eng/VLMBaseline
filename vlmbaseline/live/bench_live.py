"""Benchmark the live pose (scripts/localize_bag.sh -> poses.jsonl) and guidance while walking on it (no model).

    python -m vlmbaseline.live.bench_live <poses.jsonl> --scene lab_03 [--gt] [--n 40]

localized    share of the 2 Hz updates with a pose in the saved map; seconds until the first one
gap          seconds since the last match with the saved map (between matches the pose follows odometry)
error        --gt (the replayed session is the one the export was built from): distance and heading difference to the
             export's keyframe pose at the same stamp
on_map       distance from the pose to the nearest walkable place (a pose off the floor plan is wrong)
jumps        consecutive updates more than 1 m apart (a person walks ~0.7 m in 0.5 s)
uncertain    share of the updates marked uncertain (live/posecore.py: a localization far from odometry not taken yet);
             with --gt, the error when marked and when not
follow       guidance on the live stream as bench_follow does on keyframes: a stretch of 8-35 m, the goal an object
             within 1.5 m of its end; arrived / false reroutes; as the demo server, no update from an uncertain pose
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from vlmbaseline.scene import attach_areas, attach_vlm, load_export
from vlmbaseline.spatial import nav
from vlmbaseline.spatial.follow import Follower
from vlmbaseline.spatial.tools import SKIP_LANDMARK, SpatialIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("--scene", default="lab_03")
    ap.add_argument("--aug", default="outputs/augmented")
    ap.add_argument("--gt", action="store_true", help="the replayed session built the export: compare to its keyframes")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    s = load_export(Path("data") / a.scene)
    attach_vlm(s, Path(a.aug) / a.scene / "objects_vlm.json")
    attach_areas(s, Path(a.aug) / a.scene / "areas_vlm.json")
    idx = SpatialIndex(s)
    R = [json.loads(l) for l in open(a.poses)]
    t0 = R[0]["stamp_ns"]
    loc = [r for r in R if r.get("pose")]
    out = {"updates": len(R), "localized": f"{len(loc)}/{len(R)}",
           "first_s": round((loc[0]["stamp_ns"] - t0) / 1e9, 1) if loc else None}
    if not loc:
        print(json.dumps(out, ensure_ascii=False))
        return
    gap = np.array([r["since_loc_s"] for r in loc])
    out["gap_s_median_90_max"] = [round(float(np.median(gap)), 1), round(float(np.percentile(gap, 90)), 1),
                                  round(float(gap.max()), 1)]
    P = np.array([r["pose"] for r in loc])
    pc = np.array([idx.place[p].center[:2] for p in idx.adj])
    on = np.array([np.linalg.norm(pc - p[:2], axis=1).min() for p in P])
    out["on_map_m_median_90_max"] = [round(float(np.median(on)), 2), round(float(np.percentile(on, 90)), 2),
                                     round(float(on.max()), 2)]
    step = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    out["jumps_over_1m"] = int((step > 1.0).sum())
    U = np.array([bool(r.get("uncertain")) for r in loc])
    out["uncertain"] = f"{int(U.sum())}/{len(U)}"
    if a.gt:
        kt = np.array([k.timestamp_ns for k in s.keyframes]) if hasattr(s.keyframes[0], "timestamp_ns") else None
        if kt is None:
            K = [json.loads(l) for l in open(Path("data") / a.scene / "keyframes" / "keyframes.jsonl")]
            kt = np.array([k["timestamp_ns"] for k in K])
        G = [nav.pose_from_T(k.world_T_sensor) for k in s.keyframes]
        e, h, eu = [], [], []
        for r, p, u in zip(loc, P, U):
            i = int(np.argmin(abs(kt - r["stamp_ns"])))
            if abs(kt[i] - r["stamp_ns"]) > 0.1e9:
                continue
            e.append(float(np.hypot(p[0] - G[i][0], p[1] - G[i][1])))
            h.append(abs((math.degrees(p[2] - G[i][2]) + 180) % 360 - 180))
            eu.append(u)
        e, h, eu = np.array(e), np.array(h), np.array(eu, bool)
        out["gt_pairs"] = len(e)
        out["error_m_median_90_max"] = [round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2),
                                        round(float(e.max()), 2)]
        out["heading_deg_median_90"] = [round(float(np.median(h)), 1), round(float(np.percentile(h, 90)), 1)]
        out["error_under_1m"] = f"{int((e < 1).sum())}/{len(e)}"
        out["error_over_2m_certain_uncertain"] = [int((e[~eu] > 2).sum()), int((e[eu] > 2).sum())]
        out["error_m_90_certain"] = round(float(np.percentile(e[~eu], 90)), 2) if (~eu).any() else None
    # guidance on the live stream
    rng = random.Random(a.seed)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    objs = [o for o in s.objects if o.label not in SKIP_LANDMARK]
    oxy = np.array([o.center[:2] for o in objs])
    trials, tries = [], 0
    while len(trials) < a.n and tries < 4000 and len(P) > 20:
        tries += 1
        i = rng.randrange(len(P) - 10)
        j = min(int(np.searchsorted(cum, cum[i] + rng.uniform(8, 35))), len(P) - 1)
        if cum[j] - cum[i] < 8:
            continue
        dd = np.linalg.norm(oxy - P[j, :2], axis=1)
        if dd.min() > 1.5:
            continue
        goal = objs[int(dd.argmin())]
        f = Follower(idx, goal=goal.symbol)
        try:
            f.start(tuple(P[i]))
        except (ValueError, KeyError):
            continue
        if not (cum[j] - cum[i] <= 1.3 * f.total + 2.0 and f.total > 4.0):
            continue
        stream = [tuple(P[k]) for k in range(i + 1, j + 1) if not U[k]]
        last, v = P[j, :2], goal.center[:2] - P[j, :2]
        n = int(np.linalg.norm(v) / 0.25)
        stream += [(*(last + v * (t + 1) / max(n, 1)), math.atan2(v[1], v[0])) for t in range(n)]
        arrived_at, rer = None, 0
        for pose in stream:
            r = f.update(pose)
            rer += int(bool(r.get("rerouted")))
            if r["arrived"] and arrived_at is None:
                arrived_at = float(np.linalg.norm(goal.center[:2] - np.asarray(pose[:2])))
        trials.append({"goal": goal.symbol, "planned_m": round(f.total, 1), "arrived_at_m": arrived_at, "reroutes": rer})
    out["follow_trials"] = len(trials)
    out["follow_arrived"] = f"{sum(t['arrived_at_m'] is not None and t['arrived_at_m'] <= 2.5 for t in trials)}/{len(trials)}"
    out["follow_false_reroutes"] = sum(t["reroutes"] for t in trials)
    print(json.dumps(out, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": out, "trials": trials}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
