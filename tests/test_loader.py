from pathlib import Path

import pytest

from vlmbaseline.scene import load_export

DATA = Path(__file__).resolve().parents[1] / "data" / "library_merged"


@pytest.mark.skipif(not DATA.exists(), reason="export not linked in data/")
def test_library_merged():
    s = load_export(DATA)
    assert len(s.objects) == 112
    assert s.object("O817").label == "desk"
    assert len(s.rooms) == 1 and len(s.places) > 0
    assert sum(o.source == "ov" for o in s.objects) == 100
    assert all(o.place is not None for o in s.objects)
    assert s.keyframes and s.keyframes[0].rgb.exists()
