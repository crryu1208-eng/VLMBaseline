"""The jump check of the live pose (live/posecore.py) on a made-up walk: odometry straight along x at 1 m/s."""
import numpy as np

from vlmbaseline.live.posecore import PoseCore

ALIGN = {"body_T_sensor": np.eye(4).tolist(),
         "nodes": [{"id": i, "pos": [float(i), 0.0, 0.0], "corr": np.eye(4).tolist()} for i in range(1, 80)]}


def T(x, y=0.0):
    M = np.eye(4)
    M[:3, 3] = [x, y, 0.0]
    return M


def walk(core, t0, t1, loc=None):
    """Odometry every 0.1 s from t0 to t1; loc(t) -> map pose or None at every whole second."""
    for k in range(int(t0 * 10), int(t1 * 10)):
        t = k / 10
        ns = int(t * 1e9)
        core.on_odom(ns, T(t))
        if loc and k % 10 == 0:
            L = loc(t)
            if L is not None:
                core.on_loc(ns, L)
                core.on_info(ns, int(t) + 1)


def test_wrong_aisle_is_not_taken():
    c = PoseCore(ALIGN, trust_m=25.0)
    walk(c, 0, 5, lambda t: T(t))  # good matches
    walk(c, 5, 15, lambda t: T(t, 4.0))  # look-alike aisle 4 m to the side, ten times
    r = c.pose()
    assert abs(r["pose"][1]) < 0.01 and r["uncertain"] and r["conflict"] == 10
    walk(c, 15, 16, lambda t: T(t))  # a match agreeing with odometry again
    r = c.pose()
    assert not r["uncertain"] and r["conflict"] == 0 and r["n_loc"] == 6


def test_real_correction_is_taken_after_walking():
    c = PoseCore(ALIGN, trust_m=25.0)
    walk(c, 0, 2, lambda t: T(t))
    walk(c, 2, 30)  # odometry alone for 28 m: may have drifted
    assert c.pose()["since_loc_m"] > 27
    walk(c, 30, 33, lambda t: T(t, 2.0))  # the map says 2 m off, three times in a row: taken
    r = c.pose()
    assert abs(r["pose"][1] - 2.0) < 0.01 and not r["uncertain"]


def test_no_gate_takes_every_match():
    c = PoseCore(ALIGN, gate=False)
    walk(c, 0, 5, lambda t: T(t))
    walk(c, 5, 7, lambda t: T(t, 4.0))
    assert abs(c.pose()["pose"][1] - 4.0) < 0.2


def test_lost_odometry_marks_until_the_map_matches():
    c = PoseCore(ALIGN)
    walk(c, 0, 3, lambda t: T(t))
    c.on_odom(int(3e9), None, lost=True)  # odometry restarts from a guess
    walk(c, 3, 10)
    r = c.pose()
    assert r["uncertain"] and r["why"] == "lost"
    walk(c, 10, 11, lambda t: T(t, 3.0))  # after a loss a match far from odometry is taken at once
    r = c.pose()
    assert not r["uncertain"] and r["why"] is None and abs(r["pose"][1] - 3.0) < 0.01


def test_odometry_alone_is_marked_only_with_stale_m():
    c, d = PoseCore(ALIGN), PoseCore(ALIGN, stale_m=10.0)
    for x in (c, d):
        walk(x, 0, 2, lambda t: T(t))
        walk(x, 2, 20)
    assert not c.pose()["uncertain"]
    assert d.pose()["uncertain"] and d.pose()["why"] == "far"
