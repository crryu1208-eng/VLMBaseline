import math
from pathlib import Path

import numpy as np
import pytest

from vlmbaseline.spatial.follow import Follower, josa


def test_josa():
    assert josa("벤치(O223)") == "벤치(O223)는"
    assert josa("화분(O12)", ("이", "가")) == "화분(O12)이"
    assert josa("ETRI 열람 공간", ("으로", "로")) == "ETRI 열람 공간으로"
    assert josa("원형 복도", ("으로", "로")) == "원형 복도로"
    assert josa("엘리베이터 홀", ("으로", "로")) == "엘리베이터 홀로"  # final ㄹ takes 로


@pytest.fixture(scope="module")
def idx():
    root = Path("data") / "lab_03"
    if not root.exists():
        pytest.skip("no export data")
    from vlmbaseline.scene import load_export
    from vlmbaseline.spatial.tools import SpatialIndex
    return SpatialIndex(load_export(root))


def test_walk_the_planned_route_arrives_without_reroute(idx):
    from vlmbaseline.spatial import nav
    kf = idx.s.keyframes[100]
    pose = nav.pose_from_T(kf.world_T_sensor)
    goal = idx.s.objects[-1].symbol
    f = Follower(idx, goal=goal)
    f.start(pose)
    line, n = f.line, 0
    for a, b in zip(line[:-1], line[1:]):  # walk the route itself, 0.25 m a step, facing forward
        v = b - a
        L = float(np.linalg.norm(v))
        yaw = math.atan2(v[1], v[0])
        for t in np.arange(0.25, L + 1e-9, 0.25):
            r = f.update((*(a + v * t / L), yaw))
            n += 1
            assert not r.get("rerouted")
            if r["arrived"]:
                break
        if f.arrived:
            break
    assert f.arrived and f.reroutes == 0
