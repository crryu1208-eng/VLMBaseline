# Scene graph export, format v1

Written by `realtime/export_keyframes.py` and the OV post-processing in hydra-observation-ws
(frozen at tag `freeze-2026-11-19`). One folder per scene:

| Path | Content |
|---|---|
| `hydra/backend/dsg.json` | spark_dsg graph (JSON), no mesh |
| `hydra/backend/dsg_with_mesh.json`, `mesh.ply` | same graph with the mesh; mesh alone |
| `hydra/backend/ov_objects.json`, `closed_*.json` | human review decisions (labels, boxes, removals) |
| `keyframes/keyframes.jsonl` | one line per agent node: `node`, `session`, `timestamp_ns`, `rgb`, `depth`, `world_T_body`, `world_T_sensor` (`t`, `q_wxyz`) |
| `keyframes/rgb/<node>.png`, `keyframes/depth/<node>.png` | colour; depth 16-bit, mm |
| `camera_calib.json` | pinhole intrinsics (`sensors.camera`: width, height, fx, fy, cx, cy), no distortion |
| `observations/` | all recorded frames (`frames.jsonl`) |
| `README.md` | how the scene was made, review history |

World frame: floor at z = 0, z up, metres. Camera frame: x right, y down, z forward.

## Graph layers (layer, partition)

| Nodes | Key | Attributes used |
|---|---|---|
| Objects | (2, 0), symbol `O` | `name` (label), `bounding_box` (AABB center / dimensions), `semantic_feature` (SigLIP2 image embedding, 768), `metadata.ov` (`top3`, `n_obs`; `source: "hydra"` for closed-set objects, else open-vocabulary) |
| Agent nodes | (2, 97) | keyframe poses (same as `keyframes.jsonl`) |
| Places | (3, 1) | 2D floor places: `position`, `boundary` polygon |
| Rooms | (4, 0) | `position`, `bounding_box` |
| Building | (5, 0) | one node |

Edges between layers: object -> nearest place, place -> room, room -> building.
Current exports have one room; multi-room scenes come with the multi-session merge (hydra-observation-ws,
branch `multi-session-merge`).
