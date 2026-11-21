#!/usr/bin/env bash
# Build a separate VINS-Fusion ROS2 workspace (~/ws_vins_depth) with the live-depth patch (run inside WSL).
# The source is re-synced from the pristine package and re-patched on every run; ~/ws_vins is untouched.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WS=${WS:-$HOME/ws_vins_depth}
SRC_PKG=${SRC_PKG:-$HOME/ws_vins/src/VINS-Fusion-ROS2-Humble}
CERES21=$HOME/opt/ceres-2.1.0/lib/cmake/Ceres
source /opt/ros/humble/setup.bash
mkdir -p "$WS/src"
rsync -a --checksum --exclude .git "$SRC_PKG/" "$WS/src/VINS-Fusion-ROS2-Humble/"
python3 "$HERE/apply_ros2_depth_patch.py" "$WS/src/VINS-Fusion-ROS2-Humble"
cd "$WS"
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DCeres_DIR="$CERES21" --parallel-workers 4 \
  --event-handlers console_cohesion+ --packages-select camera_models vins 2>&1 | grep -E "error|Finished|Summary|failed" || true
test -x "$WS/install/vins/lib/vins/vins_node" && echo BUILD_OK
