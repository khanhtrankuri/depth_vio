#!/usr/bin/env bash
# One-shot demo: RViz2 + OpenVINS + bag playback, in the background, then output checks.
# Usage: run_demo.sh <ros2_bag_dir> <config_name|estimator_config.yaml> [rate=1.0] [max_cameras=2] [use_stereo=true]
D=$(cd "$(dirname "$0")" && pwd)
BAG=${1:?usage: run_demo.sh <bag_dir> <config> [rate]}
CFG=${2:?config}
RATE=${3:-1.0}
MAXCAM=${4:-2}
STEREO=${5:-true}
source "$D/common.sh"

bash "$D/stop_all.sh" >/dev/null 2>&1
nohup bash "$D/run_rviz.sh" >/dev/null 2>&1 &
sleep 6
nohup bash "$D/run_openvins.sh" "$CFG" "$MAXCAM" "$STEREO" >/dev/null 2>&1 &
sleep 8
nohup bash "$D/play_bag.sh" "$BAG" "$RATE" >/dev/null 2>&1 &
echo "Started. Logs in $LOG_DIR"
sleep 30
bash "$D/check_output.sh"
