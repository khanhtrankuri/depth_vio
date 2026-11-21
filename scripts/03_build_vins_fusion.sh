#!/usr/bin/env bash
# Build the ROS2 Humble port of VINS-Fusion (fanhong-li) in its own workspace ~/ws_vins (CPU mode). Idempotent.
set -eo pipefail
WS=${WS:-$HOME/ws_vins}
source /opt/ros/humble/setup.bash
mkdir -p "$WS/src"
if [ ! -d "$WS/src/VINS-Fusion-ROS2-Humble" ]; then
  if [ -d "$HOME/vins_candidates/VINS-Fusion-ROS2-Humble" ]; then
    cp -r "$HOME/vins_candidates/VINS-Fusion-ROS2-Humble" "$WS/src/"
  else
    git clone https://github.com/fanhong-li/VINS-Fusion-ROS2-Humble.git "$WS/src/VINS-Fusion-ROS2-Humble"
  fi
fi
sudo apt-get install -y libgeographic-dev ros-humble-rcpputils >/dev/null
cd "$WS"
CERES21=$HOME/opt/ceres-2.1.0/lib/cmake/Ceres   # run 03a_build_ceres_2_1.sh first (repo needs ceres::CUDA enum => Ceres >= 2.1)
[ -f "$CERES21/CeresConfig.cmake" ] || { echo "Missing Ceres 2.1.0; run 03a_build_ceres_2_1.sh"; exit 1; }
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DCeres_DIR="$CERES21" --parallel-workers 4 \
  --event-handlers console_cohesion+ --packages-select camera_models vins loop_fusion
echo "VINS-Fusion build finished: exit $?"
