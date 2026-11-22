"""Corrections from the RTAB-Map map frame to the export's world frame, per map node.

    python -m vlmbaseline.live.align <poses.txt> data/lab_03 [--level level_transform.json] -o outputs/live/lab_03_align.json

poses.txt: the optimized node poses of the saved map, `rtabmap-export --poses --poses_format 11 <db>` (stamp x y z qx qy qz
qw id). The export's poses are not the same trajectory up to one rigid transform (they come from a different
correction, up to ~2 m apart where a place was passed twice), so one correction per map node is kept,
C_k = world_T_body(k) * map_T_body(k)^-1, with the node matched to the export keyframe by stamp. A localized pose is then
brought into the world frame with the correction of a node of the same pass (live/bridge.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mat(t, q_wxyz):
    w, x, y, z = q_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("export")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tol", type=float, default=0.02, help="stamp match tolerance (s)")
    a = ap.parse_args()
    K = [json.loads(l) for l in open(Path(a.export) / "keyframes" / "keyframes.jsonl")]
    kt = np.array([k["timestamp_ns"] / 1e9 for k in K])
    P = np.loadtxt(a.poses, comments="#")
    nodes, used = [], 0
    for p in P:
        i = int(np.argmin(abs(kt - p[0])))
        if abs(kt[i] - p[0]) > a.tol:
            continue
        D = mat(p[1:4], [p[7], p[4], p[5], p[6]])
        W = mat(K[i]["world_T_body"]["t"], K[i]["world_T_body"]["q_wxyz"])
        nodes.append({"id": int(p[8]), "pos": [round(float(v), 4) for v in D[:3, 3]],
                      "corr": np.round(W @ np.linalg.inv(D), 6).tolist()})
        used += 1
    k0 = K[0]
    B = mat(k0["world_T_body"]["t"], k0["world_T_body"]["q_wxyz"])
    S = mat(k0["world_T_sensor"]["t"], k0["world_T_sensor"]["q_wxyz"])
    out = {"scene": Path(a.export).name, "body_T_sensor": np.round(np.linalg.inv(B) @ S, 6).tolist(), "nodes": nodes}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"{used} of {len(P)} map nodes matched to keyframes -> {a.out}")


if __name__ == "__main__":
    main()
