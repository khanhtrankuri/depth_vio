#!/usr/bin/env bash
# Convert a ROS1 .bag to a ROS2 (sqlite3) bag directory using the `rosbags` library (no ROS1 install needed).
# Usage: convert_bag.sh <in.bag> [out_dir]
source "$(dirname "$0")/common.sh"
set -e
IN=${1:?usage: convert_bag.sh <in.bag> [out_dir]}
OUT=${2:-$VIO_HOME/bags/$(basename "${IN%.bag}")_ros2}
[ -e "$OUT" ] && { echo "Refusing to overwrite existing $OUT"; exit 1; }
# Humble's rosbag2 needs the older metadata format (v6, no type_description_hash) and Humble message types.
~/venv_bag/bin/rosbags-convert --src "$IN" --dst "$OUT" --dst-version 8 --dst-typestore ros2_humble
echo "Converted -> $OUT"
ros2 bag info "$OUT"
