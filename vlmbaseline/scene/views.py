"""Pick the keyframes that show an object best (projected box, depth check)."""
from __future__ import annotations

import numpy as np
from PIL import Image


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


def best_views(scene, o, depth_cache, k=1, strict=True):
    """The k keyframes that show the object best: [(score, keyframe, clipped 2D box)], best first.
    strict: whole box inside the image and depth agreeing (benchmark); else partly cut boxes are allowed."""
    min_inside, edge, min_on = (0.98, 4, 0.35) if strict else (0.7, -1e9, 0.2)
    c = scene.camera
    W, H = c["width"], c["height"]
    pts = corners(o)
    found = []
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
        if inside < min_inside or min(x0, y0) < edge or x1 > W - edge or y1 > H - edge or frac < 0.004 or frac > 0.6:
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
        if on < min_on:
            continue  # the box shows mostly background: wrong box or occluded
        found.append((frac * inside * on, kf, (cx0, cy0, cx1, cy1)))
    found.sort(key=lambda t: -t[0])
    out, used = [], []  # spread the k views over time (keyframes at least 3 s apart)
    for f in found:
        if all(abs(f[1].timestamp_ns - u) > 3e9 for u in used):
            out.append(f)
            used.append(f[1].timestamp_ns)
        if len(out) == k:
            break
    return out
