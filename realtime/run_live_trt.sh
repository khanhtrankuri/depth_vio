#!/usr/bin/env bash
# Live stereo-inertial odometry with learned depth on a Jetson, without PyTorch:
# TensorRT depth node (depth_node_trt.py) + patched VINS-Fusion (~/ws_vins_depth). Feed it a camera driver or a bag.
#
# Usage: run_live_trt.sh <engine> <cam0 sensor.yaml> <cam1 sensor.yaml> <vins_config.yaml> [image0_topic] [image1_topic]
#   RUN_DIR=<dir>   where logs and the trajectory go (default /tmp/vio_run_<time>):
#                   depth.log, vins.log, est.txt (+ est.txt.recv), depth_stats.csv
#   NO_DEPTH=1      baseline: VINS only, no depth node (use a config without depth_topic)
#   DEPTH_ARGS="-p zmax:=8.0 -p zmin:=0.4"   extra depth-node parameters
# Stop with Ctrl-C after the bag has finished; then run the summary printed at the end.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
abs() { readlink -f "$1"; }
ENGINE=$(abs "${1:?engine}"); CAM0=$(abs "${2:?cam0 sensor.yaml}"); CAM1=$(abs "${3:?cam1 sensor.yaml}")
CFG=$(abs "${4:?vins config yaml}")
IMG0=${5:-/cam0/image_thermal}; IMG1=${6:-/cam1/image_thermal}
[ -f "$CFG" ] || { echo "missing file: $CFG"; exit 1; }
if [ -z "$NO_DEPTH" ]; then
  for f in "$ENGINE" "$CAM0" "$CAM1"; do [ -f "$f" ] || { echo "missing file: $f"; exit 1; }; done
  grep -q '^depth_topic' "$CFG" || echo "WARNING: $CFG has no depth_topic -> VINS will ignore the depth"
fi
RUN_DIR=${RUN_DIR:-/tmp/vio_run_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN_DIR" || { echo "cannot create RUN_DIR $RUN_DIR"; exit 1; }
RUN_DIR=$(abs "$RUN_DIR")
echo "run directory: $RUN_DIR"
cp "$CFG" "$RUN_DIR/config.yaml"
source /opt/ros/humble/setup.bash
source "$HOME/ws_vins_depth/install/setup.bash"
PY=${PY:-/usr/bin/python3}
PIDS=()
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; pgrep -f depth_node_trt.py >/dev/null; }
stop_all() {  # $1 = signal
  for p in "${PIDS[@]}"; do kill "-$1" "$p" 2>/dev/null || true; done
  pkill "-$1" -f depth_node_trt.py 2>/dev/null || true
  pkill "-$1" -f "lib/vins/vins_node" 2>/dev/null || true
  for i in $(seq 10); do alive || return 0; sleep 0.5; done
}
cleanup() {
  # INT first (graceful, writes depth_stats.csv); TERM covers background launches, where SIGINT is ignored
  stop_all INT; alive && stop_all TERM; alive && stop_all KILL
  wait 2>/dev/null || true
  echo
  echo "run directory: $RUN_DIR"
  echo "summary: python3 $HERE/tools/run_summary.py $RUN_DIR"
}
trap cleanup EXIT
trap 'exit 0' INT TERM

if [ -z "$NO_DEPTH" ]; then
  (cd "$HERE/depth_node" && exec "$PY" depth_node_trt.py --ros-args -p engine:="$ENGINE" \
     -p cam0_yaml:="$CAM0" -p cam1_yaml:="$CAM1" -p image0_topic:="$IMG0" -p image1_topic:="$IMG1" \
     -p stats_csv:="$RUN_DIR/depth_stats.csv" $DEPTH_ARGS) > "$RUN_DIR/depth.log" 2>&1 &
  PIDS+=($!)
  for i in $(seq 60); do grep -q "learned depth ready" "$RUN_DIR/depth.log" && break; sleep 1; done
  grep -q "learned depth ready" "$RUN_DIR/depth.log" || { cat "$RUN_DIR/depth.log"; echo "depth node did not start"; exit 1; }
  grep "learned_depth_trt" "$RUN_DIR/depth.log"
fi
"$PY" "$HERE/../scripts/record_traj.py" "$RUN_DIR/est.txt" /odometry odom > "$RUN_DIR/recorder.log" 2>&1 &
PIDS+=($!)
ros2 run vins vins_node "$CFG" > "$RUN_DIR/vins.log" 2>&1 &
PIDS+=($!)
echo "running (logs in $RUN_DIR). Play the bag now; Ctrl-C here when it has finished."
# live one-line status every 5 s
while sleep 5; do
  d=$(grep -o 'depth frames=.*' "$RUN_DIR/depth.log" 2>/dev/null | tail -1)
  n=$(grep -c LIVE_DEPTH_OBS "$RUN_DIR/vins.log" 2>/dev/null || true)
  s=$(grep -o 'solver costs: [0-9.]*' "$RUN_DIR/vins.log" 2>/dev/null | tail -1)
  p=$(grep -o 'time: [0-9.]*' "$RUN_DIR/vins.log" 2>/dev/null | wc -l)
  echo "[status] vins_poses=$p depth_used_frames=$n ${s} | ${d:-no depth node}"
  grep -qE "terminate called|Aborted|what\(\)" "$RUN_DIR/vins.log" && { tail -5 "$RUN_DIR/vins.log"; echo "VINS died"; exit 1; }
done
