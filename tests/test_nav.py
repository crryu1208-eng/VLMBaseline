from pathlib import Path

import numpy as np
import pytest

from vlmbaseline.spatial import nav


def test_words():
    assert nav.direction_word(0) == "앞" and nav.direction_word(90) == "왼쪽" and nav.direction_word(-90) == "오른쪽"
    assert nav.direction_word(180) == "뒤" and nav.direction_word(-45) == "오른쪽 앞"
    assert nav.clock(0) == 12 and nav.clock(-90) == 3 and nav.clock(90) == 9 and nav.clock(180) == 6
    assert nav.turn_word(10) == "직진" and nav.turn_word(90) == "왼쪽으로 돌기" and nav.turn_word(-170) == "뒤로 돌기"


def test_steps_corner():
    # heading +x, walk 4 m along x, then 3 m along +y: straight, then a left turn; a box on the right of leg 1
    pts = np.array([[0, 0], [1, 0.05], [2, -0.05], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]], float)
    st, h = nav.steps(pts, 0.0, [("O1", "box", None, np.array([2.0, -1.0]), 1.0)])
    assert [s["turn"] for s in st] == ["직진", "왼쪽으로 돌기"]
    assert st[0]["passing"][0]["side"] == "오른쪽" and st[1]["walk_m"] == 3.0


def test_left_right_matches_camera():
    """Objects left of the heading project left of the image centre (camera x < 0) in real keyframes."""
    root = Path("data/lab_03")
    if not root.exists():
        pytest.skip("no export data")
    from vlmbaseline.scene import load_export
    s = load_export(root)
    agree = n = 0
    for kf in s.keyframes[::10]:
        x, y, yaw = nav.pose_from_T(kf.world_T_sensor)
        Ti = np.linalg.inv(kf.world_T_sensor)
        for o in s.objects:
            pc = Ti[:3, :3] @ o.center + Ti[:3, 3]
            a = nav.rel_angle(yaw, o.center[:2] - [x, y])
            if pc[2] < 0.5 or abs(pc[0] / pc[2]) < 0.15 or abs(a) < 10:
                continue
            n += 1
            agree += (pc[0] > 0) == (a < 0)
    assert n > 100 and agree == n
