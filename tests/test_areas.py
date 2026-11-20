from pathlib import Path

import pytest

from vlmbaseline.spatial.areas import split_areas


@pytest.mark.parametrize("scene", ["lab_03", "library_merged"])
def test_split_areas(scene):
    root = Path("data") / scene
    if not root.exists():
        pytest.skip("no export data")
    from vlmbaseline.scene import load_export
    from vlmbaseline.spatial.tools import SpatialIndex
    idx = SpatialIndex(load_export(root))
    areas = split_areas(idx, radius=7.0)
    every = [p for ps in areas.values() for p in ps]
    assert sorted(every) == sorted(idx.adj) and len(every) == len(set(every))  # each place in exactly one area
    for ps in areas.values():
        assert len(ps) >= 12
        inside, seen, todo = set(ps), set(), [ps[0]]
        while todo:  # connected inside the area
            u = todo.pop()
            if u not in seen:
                seen.add(u)
                todo += [v for v, _ in idx.adj[u] if v in inside]
        assert seen == inside
