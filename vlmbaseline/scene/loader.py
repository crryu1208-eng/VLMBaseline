"""Read a scene graph export (format v1, see docs/export_format_v1.md) without spark_dsg.

    from vlmbaseline.scene import load_export
    scene = load_export("data/library_merged")
    scene.objects[0].label, scene.objects[0].center, scene.rooms, scene.keyframes[0].rgb
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# spark_dsg (layer, partition) of each node kind in the export
OBJECTS, AGENTS, PLACES, ROOMS, BUILDINGS = (2, 0), (2, 97), (3, 1), (4, 0), (5, 0)


@dataclass
class Object:
    id: int
    symbol: str          # e.g. "O817": the name used in the export README and review decisions
    label: str
    center: np.ndarray   # (3,) world, m, floor at z = 0
    size: np.ndarray     # (3,) axis-aligned box dimensions, m
    feature: np.ndarray | None  # SigLIP2 image embedding (768,), or None
    source: str          # "ov" (open-vocabulary fusion) or "hydra" (closed-set)
    top3: list[str]
    n_obs: int
    place: int | None = None
    room: int | None = None


@dataclass
class Place:
    id: int
    center: np.ndarray
    boundary: np.ndarray  # (N, 3) floor polygon
    room: int | None = None


@dataclass
class Room:
    id: int
    center: np.ndarray
    size: np.ndarray
    name: str


@dataclass
class Keyframe:
    node: str            # agent node name, e.g. "a1"
    session: str
    timestamp_ns: int
    rgb: Path
    depth: Path          # 16-bit PNG, mm
    world_T_sensor: np.ndarray  # (4, 4), camera frame x right, y down, z forward


@dataclass
class Scene:
    root: Path
    objects: list[Object]
    places: list[Place]
    rooms: list[Room]
    keyframes: list[Keyframe]
    camera: dict         # pinhole intrinsics: width, height, fx, fy, cx, cy
    edges: list[tuple[int, int]] = field(repr=False)

    def object(self, symbol: str) -> Object:
        return next(o for o in self.objects if o.symbol == symbol)


def _symbol(node_id: int) -> str:
    """spark_dsg NodeSymbol: high byte = category letter, low 56 bits = index."""
    return f"{chr(node_id >> 56)}{node_id & ((1 << 56) - 1)}"


def _pose(p: dict) -> np.ndarray:
    w, x, y, z = p["q_wxyz"]
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = p["t"]
    return T


def load_export(root: str | Path) -> Scene:
    root = Path(root)
    g = json.loads((root / "hydra/backend/dsg.json").read_text())
    kind = {n["id"]: (n["layer"], n.get("partition", 0)) for n in g["nodes"]}
    edges = [(e["source"], e["target"]) for e in g["edges"]]
    parent = {}  # child -> parent across layers (object -> place, place -> room)
    for s, t in edges:
        if kind.get(s) and kind.get(t) and kind[s][0] != kind[t][0]:
            lo, hi = (s, t) if kind[s][0] < kind[t][0] else (t, s)
            parent[lo] = hi

    objects, places, rooms = [], [], []
    for n in g["nodes"]:
        a, k = n["attributes"], kind[n["id"]]
        if k == OBJECTS:
            ov = (a.get("metadata") or {}).get("ov", {})
            f = a.get("semantic_feature") or {}
            feat = np.asarray(f["data"], np.float32) if f.get("data") else None
            bb = a["bounding_box"]
            objects.append(Object(
                id=n["id"], symbol=_symbol(n["id"]), label=a["name"],
                center=np.asarray(bb["world_P_center"]), size=np.asarray(bb["dimensions"]),
                feature=feat, source="hydra" if ov.get("source") == "hydra" else "ov", top3=ov.get("top3", []),
                n_obs=ov.get("n_obs", 0), place=parent.get(n["id"])))
        elif k == PLACES:
            places.append(Place(id=n["id"], center=np.asarray(a["position"]),
                                boundary=np.asarray(a.get("boundary") or np.zeros((0, 3))),
                                room=parent.get(n["id"])))
        elif k == ROOMS:
            bb = a["bounding_box"]
            rooms.append(Room(id=n["id"], center=np.asarray(a["position"]),
                              size=np.asarray(bb["dimensions"]), name=a.get("name") or _symbol(n["id"])))
    room_of_place = {p.id: p.room for p in places}
    for o in objects:
        o.room = room_of_place.get(o.place)

    keyframes = []
    kf_file = root / "keyframes/keyframes.jsonl"
    if kf_file.exists():
        for line in kf_file.read_text().splitlines():
            r = json.loads(line)
            keyframes.append(Keyframe(node=r["node"], session=r.get("session", ""),
                                      timestamp_ns=r["timestamp_ns"], rgb=root / r["rgb"],
                                      depth=root / r["depth"], world_T_sensor=_pose(r["world_T_sensor"])))
    calib = json.loads((root / "camera_calib.json").read_text())
    camera = calib["sensors"]["camera"]
    return Scene(root, objects, places, rooms, keyframes, camera, edges)
