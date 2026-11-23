#!/usr/bin/env bash
# Walking test with the camera, on the host, in a terminal of its own (works without network: everything is local).
#   bash scripts/walk_test.sh [scene]          # default lab_03
# 1. starts the demo server if it is not running (model on the GPU, ~1 min)
# 2. copies the live files into the hydra-dev container and runs live_localize.sh with BAG=1:
#    the walk is kept in ~/hydra_data/vlm_walks/<time>/ (poses.jsonl, logs, bag)
# 3. open http://127.0.0.1:8765/?scene=<scene>&live=1 , ask, walk. Enter in this terminal stops (or after MAX_MIN).
# Needs: outputs/live/maps/<scene>.db (scripts/build_map.sh) and outputs/live/<scene>_align.json (live/align.py).
set -eo pipefail
SCENE="${1:-lab_03}"
cd "$(dirname "$0")/.."
DB=outputs/live/maps/$SCENE.db; ALIGN=outputs/live/${SCENE}_align.json
[ -f "$DB" ] && [ -f "$ALIGN" ] || { echo "no saved map for $SCENE ($DB, $ALIGN)"; exit 1; }
docker ps --format '{{.Names}}' | grep -qx hydra-dev || { echo "the hydra-dev container is not running"; exit 1; }
if docker exec hydra-dev pgrep -f realsense2_camera_node > /dev/null; then
  echo "the camera is in use (a recording session?)"; exit 1
fi
if ! curl -s -m 2 http://127.0.0.1:8765/api/scenes > /dev/null; then
  echo "starting the demo server (log: outputs/demo.log) ..."
  setsid nohup .venv/bin/python -m vlmbaseline.demo.server --port 8765 > outputs/demo.log 2>&1 < /dev/null &
  for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:8765/api/scenes > /dev/null && break; sleep 1; done
  curl -s -m 2 http://127.0.0.1:8765/api/scenes > /dev/null || { echo "demo server did not start, see outputs/demo.log"; exit 1; }
fi
echo "demo server: http://127.0.0.1:8765/?scene=$SCENE&live=1"
docker exec hydra-dev mkdir -p /tmp/vlmlive
docker cp vlmbaseline/live/bridge.py hydra-dev:/tmp/vlmlive/
docker cp vlmbaseline/live/posecore.py hydra-dev:/tmp/vlmlive/
docker cp scripts/live_localize.sh hydra-dev:/tmp/vlmlive/
docker cp "$DB" hydra-dev:/tmp/vlmlive/$SCENE.db
docker cp "$ALIGN" hydra-dev:/tmp/vlmlive/${SCENE}_align.json
OUT=/root/data/vlm_walks/$(date +%Y%m%d_%H%M%S)_$SCENE
docker exec -it -e BAG=1 -e SCENE="$SCENE" -e MAX_MIN="${MAX_MIN:-15}" hydra-dev \
  bash /tmp/vlmlive/live_localize.sh /tmp/vlmlive/$SCENE.db /tmp/vlmlive/${SCENE}_align.json "$OUT"
echo "kept in ~/hydra_data/vlm_walks/$(basename "$OUT")"
