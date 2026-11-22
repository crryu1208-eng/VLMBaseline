#!/usr/bin/env bash
# Replay a recorded D435i session as if it were live and localize it in a saved RTAB-Map map; the pose bridge writes the
# user's pose in the export's world frame (and POSTs it to the demo server when SERVER is set).
# Runs inside the hydra-dev container (ROS 2 Jazzy, rtabmap), from a copy of this repository's files:
#   docker cp vlmbaseline/live hydra-dev:/tmp/vlmlive && docker cp scripts/localize_bag.sh hydra-dev:/tmp/vlmlive/
#   docker exec hydra-dev bash /tmp/vlmlive/localize_bag.sh <saved map rtabmap.db> <session bag dir> <align.json> <out dir> [rate]
# The map database is copied first (localization mode does not add nodes, but rtabmap still writes to it).
# Env: SERVER=http://127.0.0.1:8765 posts to the demo; SCENE overrides the scene name in align.json.
# A live camera runs the same nodes with realsense + rtabmap_odom instead of the bag (scripts/live_localize.sh).
set -eo pipefail
DB="${1:?usage: localize_bag.sh <rtabmap.db> <bag dir> <align.json> <out dir> [rate]}"
BAG="${2:?}"; ALIGN="${3:?}"; OUT="${4:?}"; RATE="${5:-1.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CAM=/camera/camera
source /opt/ros/jazzy/setup.bash
[ -f /root/hydra_ws/install/setup.bash ] && source /root/hydra_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# own DDS domain: never mix with a running session
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
mkdir -p "$OUT"
cp "$DB" "$OUT/map.db"

# the session's camera topics and odometry, without null odometry (apply_loop_closures.py of the Hydra workspace);
# /tf_static gives camera_link -> optical frames, /tf is dropped (its odom frame would clash with nothing here, but
# rtabmap only needs odom from the topic)
python3 /root/hydra_ws/scripts/apply_loop_closures.py "$BAG" "$OUT/input" --filter-slam-input \
  $CAM/color/image_raw $CAM/color/camera_info $CAM/aligned_depth_to_color/image_raw /rtabmap/odom /tf /tf_static \
  > "$OUT/filter.json" 2> "$OUT/filter.log"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# localization only: the saved map is loaded whole into working memory, no new nodes; publish_tf off (the bridge pairs
# localization_pose with odometry itself)
/opt/ros/jazzy/lib/rtabmap_slam/rtabmap --ros-args -r __ns:=/rtabmap -p use_sim_time:=true \
  -p frame_id:=camera_link -p map_frame_id:=map -p odom_frame_id_init:=odom -p publish_tf:=false -p pub_loc_pose_only_when_localizing:=true \
  -p subscribe_depth:=true -p subscribe_rgb:=true -p approx_sync:=false -p odom_sensor_sync:=false \
  -p qos:=1 -p qos_image:=1 -p qos_camera_info:=1 -p qos_odom:=1 -p topic_queue_size:=30 -p sync_queue_size:=30 \
  -p database_path:=$OUT/map.db \
  -p "Mem/IncrementalMemory:='false'" -p "Mem/InitWMWithAllNodes:='true'" -p "Rtabmap/DetectionRate:='2'" \
  -r rgb/image:=$CAM/color/image_raw -r depth/image:=$CAM/aligned_depth_to_color/image_raw \
  -r rgb/camera_info:=$CAM/color/camera_info -r odom:=/rtabmap/odom \
  < /dev/null > "$OUT/rtabmap.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 120); do
  grep -qiE "rtabmap.*(subscribed|Sync)|Subscribed to" "$OUT/rtabmap.log" 2>/dev/null && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { echo "rtabmap exited early, see $OUT/rtabmap.log"; exit 1; }
  sleep 1
done
python3 "$HERE/bridge.py" --align "$ALIGN" ${SCENE:+--scene $SCENE} ${SERVER:+--server $SERVER} \
  --log "$OUT/poses.jsonl" --ros-args -p use_sim_time:=true < /dev/null > "$OUT/bridge.log" 2>&1 &
PIDS+=($!)
sleep 3
echo "localizing $BAG in $DB at rate $RATE"
QOS=$OUT/qos.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw $CAM/color/camera_info /rtabmap/odom; do
    echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
  done
} > "$QOS"
ros2 bag play "$OUT/input" --clock -r "$RATE" --disable-keyboard-controls --qos-profile-overrides-path "$QOS" \
  < /dev/null > "$OUT/bag_play.log" 2>&1
sleep 3
rm -rf "$OUT/input" "$OUT/map.db"
echo "done: $(wc -l < "$OUT/poses.jsonl") poses, $(grep -c '"localized": true' "$OUT/poses.jsonl") localized -> $OUT/poses.jsonl"
