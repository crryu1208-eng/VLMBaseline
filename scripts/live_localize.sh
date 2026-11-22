#!/usr/bin/env bash
# Live guidance from the D435i: camera -> RTAB-Map odometry -> localization in the saved map (scripts/build_map.sh)
# -> pose bridge -> demo server (/api/live). No Hydra, no segmentation: only the pose is needed here.
# Runs inside the hydra-dev container (files copied as in localize_bag.sh); the demo server runs on the host
# (the container uses host networking):
#   docker exec -it hydra-dev bash /tmp/vlmlive/live_localize.sh <map.db> <align.json> [out dir]
# Then open http://127.0.0.1:8765/?scene=<scene>&live=1, ask '여기서 ... 어떻게 가?' and walk. Enter (or Ctrl-C) stops.
# Check first that no recording session is running (one camera, one user).
# Env: FPS (default 15), RES (default 848x480: odometry and localization do not need 720p), SERVER
# (default http://127.0.0.1:8765), SCENE, BAG=1 also records the camera, odometry and localization topics to
# <out dir>/bag (~40 MB/s) so the walk can be replayed later with localize_bag.sh, MAX_MIN (default 15) stops by itself.
set -eo pipefail
set -m
DB="${1:?usage: live_localize.sh <map.db> <align.json> [out dir]}"; ALIGN="${2:?}"
OUT="${3:-/tmp/vlmlive/live_$(date +%Y%m%d_%H%M%S)}"
FPS="${FPS:-15}"; RES="${RES:-848x480}"; W="${RES%x*}"; H="${RES#*x}"
SERVER="${SERVER:-http://127.0.0.1:8765}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CAM=/camera/camera
source /opt/ros/jazzy/setup.bash
[ -f /root/hydra_ws/install/setup.bash ] && source /root/hydra_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
if pgrep -f "realsense2_camera_node" > /dev/null; then echo "the camera is in use (a recording session?)"; exit 1; fi
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
mkdir -p "$OUT"
cp "$DB" "$OUT/map.db"
PIDS=()
# each job is its own process group (set -m): signal the group, so `ros2 run` children stop too; SIGTERM after 10 s
cleanup() {
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  for i in $(seq 1 10); do
    alive=0; for p in "${PIDS[@]}"; do kill -0 -- "-$p" 2>/dev/null && alive=1; done
    [ $alive = 0 ] && break; sleep 1
  done
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  rm -f "$OUT/map.db"
}
trap cleanup EXIT

ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=$W,$H,$FPS depth_module.depth_profile:=$W,$H,$FPS \
  align_depth.enable:=true enable_sync:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  < /dev/null > "$OUT/camera.log" 2>&1 &
PIDS+=($!)
# odometry as in the Hydra workspace's d435i_obs.launch.yaml (IMU orientation, reset at the last pose after a loss)
ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -r __ns:=/rtabmap -r __node:=imu_filter \
  -p use_mag:=false -p publish_tf:=false -p world_frame:=enu -r imu/data_raw:=$CAM/imu -r imu/data:=/rtabmap/imu \
  < /dev/null > "$OUT/imu.log" 2>&1 &
PIDS+=($!)
ros2 run rtabmap_odom rgbd_odometry --ros-args -r __ns:=/rtabmap \
  -p frame_id:=camera_link -p odom_frame_id:=odom -p publish_tf:=true -p approx_sync:=false \
  -p qos:=1 -p qos_camera_info:=1 -p topic_queue_size:=30 -p sync_queue_size:=30 -p wait_for_transform:=0.2 \
  -p "Odom/ResetCountdown:='1'" -p wait_imu_to_init:=true -p always_check_imu_tf:=true \
  -p "Vis/MaxFeatures:='600'" -p "OdomF2M/MaxSize:='1200'" \
  -r rgb/image:=$CAM/color/image_raw -r depth/image:=$CAM/aligned_depth_to_color/image_raw \
  -r rgb/camera_info:=$CAM/color/camera_info -r imu:=/rtabmap/imu \
  < /dev/null > "$OUT/odom.log" 2>&1 &
PIDS+=($!)
# localization only in the saved map (as localize_bag.sh)
/opt/ros/jazzy/lib/rtabmap_slam/rtabmap --ros-args -r __ns:=/rtabmap \
  -p frame_id:=camera_link -p map_frame_id:=map -p publish_tf:=false -p pub_loc_pose_only_when_localizing:=true \
  -p subscribe_depth:=true -p subscribe_rgb:=true -p approx_sync:=false \
  -p qos:=1 -p qos_image:=1 -p qos_camera_info:=1 -p qos_odom:=1 -p topic_queue_size:=30 -p sync_queue_size:=30 \
  -p database_path:=$OUT/map.db \
  -p "Mem/IncrementalMemory:='false'" -p "Mem/InitWMWithAllNodes:='true'" -p "Rtabmap/DetectionRate:='2'" \
  -r rgb/image:=$CAM/color/image_raw -r depth/image:=$CAM/aligned_depth_to_color/image_raw \
  -r rgb/camera_info:=$CAM/color/camera_info -r odom:=/rtabmap/odom \
  < /dev/null > "$OUT/rtabmap.log" 2>&1 &
PIDS+=($!)
python3 "$HERE/bridge.py" --align "$ALIGN" ${SCENE:+--scene $SCENE} --server "$SERVER" --log "$OUT/poses.jsonl" \
  < /dev/null > "$OUT/bridge.log" 2>&1 &
PIDS+=($!)
if [ "${BAG:-0}" = 1 ]; then
  ros2 bag record -s sqlite3 -o "$OUT/bag" $CAM/color/image_raw $CAM/color/camera_info \
    $CAM/aligned_depth_to_color/image_raw $CAM/aligned_depth_to_color/camera_info $CAM/imu \
    /rtabmap/odom /rtabmap/info /rtabmap/localization_pose /tf /tf_static \
    < /dev/null > "$OUT/bag_record.log" 2>&1 &
  PIDS+=($!)
fi
MAX_MIN="${MAX_MIN:-15}"
echo "live localization running ($OUT); open $SERVER/?scene=${SCENE:-<scene>}&live=1"
echo "Enter stops (or after $MAX_MIN min)."
T0=$(date +%s)
status() {  # one line every 5 s: how many updates had a pose in the saved map
  local n=0 k=0
  [ -f "$OUT/poses.jsonl" ] && n=$(wc -l < "$OUT/poses.jsonl") && k=$(grep -c '"localized": true' "$OUT/poses.jsonl" || true)
  local last=$(tail -1 "$OUT/poses.jsonl" 2>/dev/null | grep -o '"localized": [a-z]*' | cut -d' ' -f2)
  printf '\r%4ds  updates %d, in the map %d, now %s   ' $(( $(date +%s) - T0 )) "$n" "$k" "${last:-waiting}"
}
while ! read -r -t 5 _; do
  status
  [ $(( $(date +%s) - T0 )) -ge $(( MAX_MIN * 60 )) ] && { echo; echo "time limit"; break; }
  # stdin closed (non-interactive run): keep going until the time limit
  [ -t 0 ] || [ -p /dev/stdin ] || sleep 5
done
echo
status; echo
