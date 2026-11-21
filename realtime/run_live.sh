#!/usr/bin/env bash
# Live stereo-inertial odometry with learned depth (inside WSL / Ubuntu 22.04 + ROS2 Humble).
# Starts the GPU depth node and the patched VINS-Fusion; feed it a camera driver or `ros2 bag play`.
# Usage: run_live.sh <cam0 sensor.yaml> <cam1 sensor.yaml> [vins_config.yaml] [model=rt:sceneflow:6]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CAM0=${1:?cam0 sensor.yaml}; CAM1=${2:?cam1 sensor.yaml}
CFG=${3:-$HERE/config/euroc_stereo_imu_depth.yaml}; MODEL=${4:-rt:sceneflow:6}
source /opt/ros/humble/setup.bash
source "$HOME/ws_vins_depth/install/setup.bash"
mkdir -p /tmp/vins_depth
(cd "$HERE/depth_node" && exec "$HOME/venv_depth/bin/python" depth_node.py --ros-args \
   -p model:=$MODEL -p cam0_yaml:=$CAM0 -p cam1_yaml:=$CAM1) &
DN=$!
trap 'kill -INT $DN $VINS 2>/dev/null; wait' EXIT INT TERM
sleep 15   # model load + CUDA-graph capture
ros2 run vins vins_node "$CFG" &
VINS=$!
wait
