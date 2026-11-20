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

## VLM augmentation (VLMBaseline output)

`python -m vlmbaseline.augment.describe_objects <export>`, `describe_areas <export>`, then `write_dsg <export>` write
`outputs/augmented/<scene>/`:

| File | Content |
|---|---|
| `objects_vlm.json` | per object symbol: the VLM's answer, the keyframes shown (`views`), raw text |
| `dsg_vlm.json`, `dsg_with_mesh_vlm.json` | the export graph with `attributes.metadata.vlm` on each described object (loads in spark_dsg as before) |
| `areas_vlm.json` | per area id: places, centre, neighbours, shape (area m², longest walk, width), objects, the VLM's answer |
| `views/<symbol>.jpg`, `area_views/<id>.jpg` | the images given to the VLM, for checking |

`metadata.vlm` keys (values in Korean): `이름` name, `라벨_맞음` label fits (bool), `색` colour, `재질` material,
`상태` visible state (from a per-label list, e.g. 화분: 건강함 / 잎이 시들거나 누렇게 변함; 쓰레기통: 비어 있음 … 넘침),
`점검_필요` needs cleaning / repair (bool) and `점검_이유`, `용도` uses (list), `설명` one sentence, `views`, `model`.
States are as seen when the scene was recorded.

Areas: 2D place nodes get `attributes.metadata.vlm_area = {area, 이름}`; the graph metadata gets `vlm.areas` =
{id: {`이름` name, `종류` kind (복도, 출입구·로비, 엘리베이터·계단 앞, 열람·학습 공간, 서가, 사무 공간, 회의 공간, 휴게 공간,
실험·작업 공간, 창고·보관 공간, 기타), `설명`, `용도`, `특징` notable things, `center`, `neighbours`, `n_places`}}.
Areas are split in code (walkable regions of about 7 m), not Hydra rooms.

