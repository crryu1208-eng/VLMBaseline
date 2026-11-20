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

    vlmbaseline/scene/    export loader (pure Python + numpy, no spark_dsg), best keyframe views of an object
    vlmbaseline/augment/  VLM descriptions of every object and of floor areas, written into the graph (metadata.vlm)
    vlmbaseline/spatial/  spatial tools (count, nearest, distances, routes, search, turn-by-turn from the user's pose),
                      tool-using QA / guidance agent
    vlmbaseline/demo/     local web page to try the agent (stdlib http.server, one HTML file)
    vlmbaseline/bench/    VLM benchmark (object naming, descriptions, scene QA)
    docs/                 export format, design notes
    tests/
    data/                 exports, not tracked (symlinks; see data/README.md)

## Setup

    uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
    ln -s ~/hydra_data/realtime/export/library_merged_01_03_04_yaw015_ov_data data/library_merged
    .venv/bin/python -m pytest -q

## Use

    python -m vlmbaseline.augment.describe_objects data/library_merged      # ~4 s per object (Qwen3.5-9B, 4-bit)
    python -m vlmbaseline.augment.describe_areas data/library_merged       # areas (zones) named by the VLM
    python -m vlmbaseline.augment.write_dsg data/library_merged
    python -m vlmbaseline.spatial.bench_qa Qwen/Qwen3.5-9B                   # QA / guidance benchmark
    python -m vlmbaseline.spatial.bench_pose Qwen/Qwen3.5-9B                 # guidance from the user's pose
    python -m vlmbaseline.spatial.bench_areas Qwen/Qwen3.5-9B                # questions about areas
    python -m vlmbaseline.spatial.bench_dialog Qwen/Qwen3.5-9B               # follow-up questions ('거기까지 어떻게 가?')
    python -m vlmbaseline.spatial.bench_follow                               # guidance while walking (no model, replays the recording)

Try it in a browser (map, set your position / heading by dragging, ask in Korean, see the route; the page keeps the
conversation, so follow-ups like '거기까지 어떻게 가?' work, '새 대화' starts over; after a guidance answer, moving
your position or '걸어 보기' gives turn-by-turn announcements while walking, '음성' speaks them):

    python -m vlmbaseline.demo.server            # http://127.0.0.1:8765 (local only); --no-model for the map only
    # http://127.0.0.1:8765/?scene=lab_03&pose=29.8,-23.0,131&q=여기서 가장 가까운 화분까지 어떻게 가?

In code: `Agent(proc, model, SpatialIndex(scene)).ask("여기서 가장 가까운 화분까지 어떻게 가?", pose=(x, y, yaw))`;
without `pose`, guidance needs a start object named in the question.

Results: [docs/bench_2026-11-19.md](docs/bench_2026-11-19.md).

## Licenses

Deployment constraint: no AGPL code or weights (no Ultralytics / YOLO). Check code, weights and training-data
licenses of every model before adding it. This repository is public: keep data, images and weights out of git.
