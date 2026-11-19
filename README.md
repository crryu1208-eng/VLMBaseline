# VLMBaseline

Spatial intelligence on a 3D scene graph built by the Hydra realtime pipeline
([hydra-observation-ws](https://github.com/crryu1208-eng/hydra-observation-ws), `realtime` branch):

1. **Augment** the merged scene graph with a VLM: object attributes, states, affordances, room / area
   descriptions, from the keyframe images that saw each node.
2. **Act** on it: answer questions about the space, guide a person to a place or an object, suggest tasks.

The only input is a scene graph export (format v1, [docs/export_format_v1.md](docs/export_format_v1.md)):
spark_dsg graph JSON, mesh, keyframe RGB / depth with poses, camera intrinsics.
Producing and merging scene graphs stays in hydra-observation-ws.

## Layout

    vlmbaseline/scene/    export loader (pure Python + numpy, no spark_dsg)
    docs/                 export format, design notes
    tests/
    data/                 exports, not tracked (symlinks; see data/README.md)

## Setup

    uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
    ln -s ~/hydra_data/realtime/export/library_merged_01_03_04_yaw015_ov_data data/library_merged
    .venv/bin/python -m pytest -q

## Licenses

Deployment constraint: no AGPL code or weights (no Ultralytics / YOLO). Check code, weights and training-data
licenses of every model before adding it. This repository is public: keep data, images and weights out of git.
