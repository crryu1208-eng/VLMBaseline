"""Turn-by-turn directions and directions relative to the user's pose, computed in code.

World frame: z up, so a positive (counter-clockwise) angle seen from above is a turn to the left.
A pose is (x, y, yaw); yaw is the walking / looking direction in radians from the world x axis.
A camera keyframe gives a pose through its forward axis (camera z) projected on the floor.
"""
from __future__ import annotations

import math

import numpy as np


def pose_from_T(world_T_sensor):
    """(x, y, yaw) of a camera pose; camera frame x right, y down, z forward."""
    T = np.asarray(world_T_sensor)
    f = T[:3, 2]
    return float(T[0, 3]), float(T[1, 3]), math.atan2(f[1], f[0])


def rel_angle(yaw, vec):
    """Angle of a floor vector relative to the heading, in degrees: + left, - right, in (-180, 180]."""
    a = math.degrees(math.atan2(vec[1], vec[0]) - yaw)
    return (a + 180.0) % 360.0 - 180.0


def direction_word(a):
    """Korean direction for a relative angle (degrees, + left)."""
    s = "왼쪽" if a > 0 else "오른쪽"
    b = abs(a)
    if b <= 30:
        return "앞"
    if b <= 60:
        return f"{s} 앞"
    if b <= 120:
        return s
    if b <= 150:
        return f"{s} 뒤"
    return "뒤"


def clock(a):
    """Clock direction (12 = straight ahead, 3 = right)."""
    h = round(-a / 30.0) % 12
    return 12 if h == 0 else h


def turn_word(a):
    """Korean turn instruction for a change of heading (degrees, + left)."""
    s = "왼쪽" if a > 0 else "오른쪽"
    b = abs(a)
    if b <= 25:
        return "직진"
    if b <= 60:
        return f"{s}으로 비스듬히"
    if b <= 135:
        return f"{s}으로 돌기"
    return "뒤로 돌기"


def simplify(pts, eps=0.5):
    """Douglas-Peucker on a floor polyline (N, 2); keeps the end points."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(d.argmax())
    if d[i] <= eps:
        return np.array([a, b])
    return np.vstack([simplify(pts[:i + 1], eps)[:-1], simplify(pts[i:], eps)])


def _turn_at(p, i):
    u, v = p[i] - p[i - 1], p[i + 1] - p[i]
    return abs(rel_angle(math.atan2(u[1], u[0]), v))


def legs(pts, min_leg=0.8, eps=0.6, min_turn=40.0):
    """Polyline as legs [(start, end)]: simplified, short legs merged, and bends under min_turn degrees
    straightened (removed one at a time, the gentlest first), so a corridor reads as one straight leg."""
    p = [q for q in simplify(pts, eps)]
    changed = True
    while changed and len(p) > 2:
        changed = False
        inner = range(1, len(p) - 1)
        short = [i for i in inner if min(np.linalg.norm(p[i] - p[i - 1]), np.linalg.norm(p[i + 1] - p[i])) < min_leg]
        cand = short or [i for i in inner if _turn_at(p, i) < min_turn]
        if cand:
            i = min(cand, key=lambda j: _turn_at(p, j))
            del p[i]
            changed = True
    return [(p[i], p[i + 1]) for i in range(len(p) - 1)]


def half_m(x):
    return max(0.5, round(x * 2) / 2)


def steps(pts, yaw=None, landmarks=(), goal_xy=None, near=1.5):
    """Turn-by-turn steps along a floor polyline.

    yaw        current heading; None means unknown (the first step then has no turn).
    landmarks  [(symbol, label, name, xy, size)] objects that may be passed.
    Returns (steps, final_yaw); each step: turn, walk_m, passing [{symbol, label, name, side}].
    """
    out, used = [], set()
    head = yaw
    for a, b in legs(pts):
        v = b - a
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            continue
        h = math.atan2(v[1], v[0])
        st = {"turn": "출발" if head is None else turn_word(rel_angle(head, v)), "walk_m": half_m(L)}
        passing = []
        for sym, label, name, xy, size in landmarks:
            if sym in used:
                continue
            w = np.asarray(xy) - a
            t = float(np.dot(w, v) / L)
            if not 0.0 <= t <= L:
                continue
            side = float(v[0] * w[1] - v[1] * w[0]) / L  # + left
            if abs(side) <= near:
                passing.append((-size, t, sym, label, name, "왼쪽" if side > 0 else "오른쪽"))
        passing.sort()
        sel = sorted(passing[:2], key=lambda x: x[1])
        for _, _, sym, label, name, side in sel:
            used.add(sym)
        st["passing"] = [{"symbol": s, "label": lb, **({"name": n} if n else {}), "side": sd}
                         for _, _, s, lb, n, sd in sel]
        out.append(st)
        head = h
    return out, head
