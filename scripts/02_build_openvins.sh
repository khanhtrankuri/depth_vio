#!/usr/bin/env bash
# Clone (if needed) and build OpenVINS (ROS2 Humble) in ~/ws_ov. Idempotent.
set -eo pipefail
WS=${WS:-$HOME/ws_ov}
source /opt/ros/humble/setup.bash   # not compatible with `set -u`

sudo apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
  python3-dev python3-matplotlib python3-numpy python3-psutil python3-tk

mkdir -p "$WS/src"
[ -d "$WS/src/open_vins" ] || git clone https://github.com/rpng/open_vins.git "$WS/src/open_vins"

cd "$WS"
# Parallelism kept modest: Eigen-heavy C++ can use >2GB RAM per job.
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  --parallel-workers 4 --event-handlers console_cohesion+ \
  --packages-select ov_core ov_init ov_msckf ov_eval
echo "OpenVINS build finished: exit $?"
