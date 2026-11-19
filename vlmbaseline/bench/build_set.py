"""Build the VLM benchmark set from scene graph exports.

    python -m vlmbaseline.bench.build_set data/library_merged data/lab_03 --out outputs/bench_set

For each object (labels capped at --per-label): the keyframe that sees its box best (box inside the image,
not occluded by the depth image, largest on screen), saved with the box drawn in red (`<sym>_full.jpg`)
and as a crop (`<sym>_crop.jpg`). Ground truth = the human-reviewed label of the export.
Scene questions (Korean) with answers computed from the graph: counts, nearest object, distance, largest.
Writes objects.jsonl and scene_qa.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vlmbaseline.scene import load_export


def corners(o):
    h = o.size / 2
    return np.array([o.center + h * (2 * np.array(s) - 1) for s in np.ndindex(2, 2, 2)])


def project(scene, kf, pts):
    c = scene.camera
    T = np.linalg.inv(kf.world_T_sensor)
    p = pts @ T[:3, :3].T + T[:3, 3]
    z = p[:, 2]
    uv = np.stack([c["fx"] * p[:, 0] / z + c["cx"], c["fy"] * p[:, 1] / z + c["cy"]], 1)
    return uv, z


def best_view(scene, o, depth_cache):
    c = scene.camera
    W, H = c["width"], c["height"]
    pts = corners(o)
    best = None
    for kf in scene.keyframes:
        uv, z = project(scene, kf, np.vstack([pts, o.center]))
        if (z[:8] < 0.3).any() or not (0.5 < z[8] < 6.0):
            continue
        x0, y0 = uv[:8].min(0)
        x1, y1 = uv[:8].max(0)
        area = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        inside = (cx1 - cx0) * (cy1 - cy0) / area
        frac = area / (W * H)
        if inside < 0.98 or min(x0, y0) < 4 or x1 > W - 4 or y1 > H - 4 or frac < 0.004 or frac > 0.6:
            continue
        if kf.node not in depth_cache:
            depth_cache[kf.node] = np.asarray(Image.open(kf.depth), np.float32) / 1000.0
        d = depth_cache[kf.node]
        u, v = int(np.clip(uv[8, 0], 0, W - 1)), int(np.clip(uv[8, 1], 0, H - 1))
        patch = d[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
        patch = patch[patch > 0]
        if patch.size == 0 or np.median(patch) < z[8] - (0.5 * o.size.max() + 0.3):
            continue  # something in front of the object
        box = d[int(cy0):int(cy1), int(cx0):int(cx1)]
        box = box[box > 0]
        on = ((box > z[:8].min() - 0.15) & (box < z[:8].max() + 0.15)).mean() if box.size else 0
        if on < 0.35:
            continue  # the box shows mostly background: wrong box or occluded
        score = frac * inside * on
        if best is None or score > best[0]:
            best = (score, kf, (cx0, cy0, cx1, cy1))
    return best


def object_set(scene, name, out, per_label, rng):
    by_label = defaultdict(list)
    for o in scene.objects:
        by_label[o.label].append(o)
    rows, cache = [], {}
    for label, objs in sorted(by_label.items()):
        rng.shuffle(objs)
        n = 0
        for o in objs:
            if n >= per_label:
                break
            b = best_view(scene, o, cache)
            if b is None:
                continue
            _, kf, (x0, y0, x1, y1) = b
            im = Image.open(kf.rgb).convert("RGB")
            full = im.copy()
            ImageDraw.Draw(full).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            m = 0.25 * max(x1 - x0, y1 - y0) + 10
            crop = im.crop((max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, im.width), min(y1 + m, im.height)))
            key = f"{name}_{o.symbol}"
            full.save(out / f"{key}_full.jpg", quality=92)
            crop.save(out / f"{key}_crop.jpg", quality=92)
            rows.append(dict(key=key, scene=name, symbol=o.symbol, label=o.label, source=o.source,
                             keyframe=kf.node, box_px=[round(v, 1) for v in (x0, y0, x1, y1)],
                             full=f"{key}_full.jpg", crop=f"{key}_crop.jpg"))
            n += 1
    return rows


def graph_text(scene):
    lines = ["기호 | 라벨 | 중심 x,y,z (m) | 크기 x,y,z (m)"]
    for o in scene.objects:
        lines.append(f"{o.symbol} | {o.label} | {o.center[0]:.2f},{o.center[1]:.2f},{o.center[2]:.2f} | "
                     f"{o.size[0]:.2f},{o.size[1]:.2f},{o.size[2]:.2f}")
    return "\n".join(lines)


def scene_qa(scene, name, rng, n_each=6):
    objs = scene.objects
    by_label = defaultdict(list)
    for o in objs:
        by_label[o.label].append(o)
    qs = []
    labels = [l for l, v in by_label.items() if len(v) >= 2] + [l for l, v in by_label.items() if len(v) == 1]
    for l in labels[:n_each]:
        qs.append(dict(kind="count", q=f"이 공간에 라벨이 '{l}'인 물체는 몇 개입니까? 숫자만 답하세요.",
                       gt=len(by_label[l])))
    multi = [l for l, v in by_label.items() if len(v) >= 2]
    for _ in range(n_each):
        a = rng.choice(objs)
        l = rng.choice([m for m in multi if m != a.label] or multi)
        cand = by_label[l]
        near = min(cand, key=lambda o: np.linalg.norm(o.center[:2] - a.center[:2]))
        qs.append(dict(kind="nearest", q=f"{a.symbol}({a.label})에서 수평 거리로 가장 가까운 '{l}'은 무엇입니까? "
                                          f"기호(예: O12)만 답하세요.", gt=near.symbol))
    for _ in range(n_each):
        a, b = rng.sample(objs, 2)
        d = float(np.linalg.norm(a.center[:2] - b.center[:2]))
        qs.append(dict(kind="distance", q=f"{a.symbol}({a.label})과 {b.symbol}({b.label}) 중심 사이의 수평 거리는 "
                                           f"몇 m입니까? 소수 첫째 자리까지 숫자만 답하세요.", gt=round(d, 2)))
    big = max(objs, key=lambda o: float(np.prod(o.size)))
    qs.append(dict(kind="largest", q="상자 부피가 가장 큰 물체는 무엇입니까? 기호만 답하세요.", gt=big.symbol))
    hi = max(objs, key=lambda o: o.center[2] + o.size[2] / 2)
    qs.append(dict(kind="highest", q="윗면이 가장 높은 물체는 무엇입니까? 기호만 답하세요.", gt=hi.symbol))
    for i, q in enumerate(qs):
        q.update(id=f"{name}_{i:02d}", scene=name)
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--out", default="outputs/bench_set")
    ap.add_argument("--per-label", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    objs, qa = [], []
    for e in a.exports:
        s = load_export(e)
        name = Path(e).name
        rows = object_set(s, name, out / "img", a.per_label, rng)
        objs += rows
        (out / f"graph_{name}.txt").write_text(graph_text(s))
        qa += scene_qa(s, name, rng)
        print(f"{name}: {len(rows)} objects with a view (of {len(s.objects)}), {len(qa)} questions so far")
    with open(out / "objects.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in objs)
    with open(out / "scene_qa.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in qa)


if __name__ == "__main__":
    main()
