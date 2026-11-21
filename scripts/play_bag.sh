#!/usr/bin/env bash
# Play a ROS2 bag directory. Usage: play_bag.sh <bag_dir> [rate=1.0] [extra ros2 bag play args]
# Start OpenVINS and RViz2 first so no early IMU/camera data is lost.
source "$(dirname "$0")/common.sh"
BAG=${1:?usage: play_bag.sh <ros2_bag_dir> [rate]}
RATE=${2:-1.0}
shift $(( $# > 2 ? 2 : $# ))
ros2 bag info "$BAG" | tee "$LOG_DIR/bag_info_$STAMP.log"
# --clock is intentionally NOT used: OpenVINS here runs on header stamps, not sim time.
ros2 bag play "$BAG" -r "$RATE" "$@" 2>&1 | tee "$LOG_DIR/bagplay_$STAMP.log"
