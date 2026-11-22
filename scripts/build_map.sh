#!/usr/bin/env bash
# Build the saved map for live localization from a session bag: one connected RTAB-Map graph.
# The live database of a session can be in many pieces (a new map after every odometry loss; lab_03: 25 maps, the
# middle of the walk not connected to the rest), and a pose localized in a piece that is not connected drifts off the
# export. Here the recorded odometry is replayed with null poses dropped and every reset turned into an uncertain link
# (apply_loop_closures.py --filter-slam-input of the Hydra workspace), so rtabmap keeps one graph and loop-closes it.
# Runs inside the hydra-dev container (see localize_bag.sh for copying the files in):
#   docker exec hydra-dev bash /tmp/vlmlive/build_map.sh <session bag dir> <out dir> [rate]
# -> <out dir>/map.db and <out dir>/map_poses.txt (optimized node poses, input of vlmbaseline.live.align)
set -eo pipefail
BAG="${1:?usage: build_map.sh <bag dir> <out dir> [rate]}"; OUT="${2:?}"; RATE="${3:-1.0}"
CAM=/camera/camera
source /opt/ros/jazzy/setup.bash
[ -f /root/hydra_ws/install/setup.bash ] && source /root/hydra_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-57}"
if [ -e "$OUT" ]; then echo "$OUT already exists"; exit 1; fi
mkdir -p "$OUT"
python3 /root/hydra_ws/scripts/apply_loop_closures.py "$BAG" "$OUT/input" --filter-slam-input \
  $CAM/color/image_raw $CAM/color/camera_info $CAM/aligned_depth_to_color/image_raw /rtabmap/odom /tf /tf_static \
  > "$OUT/filter.json" 2> "$OUT/filter.log"
/opt/ros/jazzy/lib/rtabmap_slam/rtabmap -d --ros-args -r __ns:=/rtabmap -p use_sim_time:=true \
  -p frame_id:=camera_link -p map_frame_id:=map -p publish_tf:=false \
  -p subscribe_depth:=true -p subscribe_rgb:=true -p approx_sync:=false -p odom_sensor_sync:=false \
  -p qos:=1 -p qos_image:=1 -p qos_camera_info:=1 -p qos_odom:=1 -p topic_queue_size:=30 -p sync_queue_size:=30 \
  -p database_path:=$OUT/map.db -p "Rtabmap/DetectionRate:='2'" -p "Mem/IncrementalMemory:='true'" \
  -r rgb/image:=$CAM/color/image_raw -r depth/image:=$CAM/aligned_depth_to_color/image_raw \
  -r rgb/camera_info:=$CAM/color/camera_info -r odom:=/rtabmap/odom \
  < /dev/null > "$OUT/rtabmap.log" 2>&1 &
PID=$!
trap 'kill -INT $PID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
  grep -qiE "subscribed to|Sync" "$OUT/rtabmap.log" 2>/dev/null && break
  kill -0 $PID 2>/dev/null || { echo "rtabmap exited early, see $OUT/rtabmap.log"; exit 1; }
  sleep 1
done
sleep 2
QOS=$OUT/qos.yaml
{
  echo "/tf_static: {depth: 1, durability: transient_local}"
  for t in $CAM/color/image_raw $CAM/aligned_depth_to_color/image_raw $CAM/color/camera_info /rtabmap/odom; do
    echo "$t: {history: keep_last, depth: 500, reliability: reliable, durability: volatile}"
  done
} > "$QOS"
ros2 bag play "$OUT/input" --clock -r "$RATE" --disable-keyboard-controls --qos-profile-overrides-path "$QOS" \
  --wait-for-all-acked 30000 < /dev/null > "$OUT/bag_play.log" 2>&1
rm -rf "$OUT/input"
sleep 3
kill -INT $PID
for i in $(seq 1 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -0 $PID 2>/dev/null && kill -TERM $PID
wait $PID 2>/dev/null || true
trap - EXIT
rtabmap-export --poses --poses_format 11 --output_dir "$OUT" --output map "$OUT/map.db" > "$OUT/export.log" 2>&1
[ -f "$OUT/map_poses.txt" ] || mv "$(ls "$OUT"/*poses*.txt | head -1)" "$OUT/map_poses.txt"
python3 - "$OUT/map.db" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
print("nodes", c.execute("select count(*) from Node").fetchone()[0],
      "maps", c.execute("select count(distinct map_id) from Node").fetchone()[0],
      "loop closures", c.execute("select count(*) from Link where type in (1, 2)").fetchone()[0] // 2)
EOF
echo "map: $OUT/map.db, poses: $OUT/map_poses.txt ($(grep -vc '^#' "$OUT/map_poses.txt") nodes)"
