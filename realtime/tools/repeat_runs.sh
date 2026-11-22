#!/usr/bin/env bash
# Unattended repeated runs on the Jetson: for each repetition, one baseline run (VINS only) and one depth run
# (TensorRT depth node + patched VINS), alternating, each with a full real-time bag playback. Then compares them.
#
# Usage: repeat_runs.sh <bag_dir> <out_dir> <reps> [rate]
#   Env (defaults match the thermal setup on the Orin):
#     CFG_DIR    dir with cam0/cam1_sensor.yaml, thermal_nodepth_td0_f10.yaml, thermal_depth_f10.yaml
#     ENGINE     TensorRT engine
#     BASE_CFG / DEPTH_CFG   override the two VINS configs
#     ONLY=base|depth         run only one of the two
# Each run goes to <out_dir>/<base|depth>_rN (logs, est.txt, depth_stats.csv).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); RT=$(cd "$HERE/.." && pwd)
BAG=$(readlink -f "${1:?bag dir}"); OUT=${2:?out dir}; REPS=${3:?reps}; RATE=${4:-1.0}
CFG_DIR=${CFG_DIR:-$HOME/Desktop/lantl/VINS-Fusion/config/thermal/CalibIMU_CAM_11_11_Chuan}
ENGINE=${ENGINE:-$RT/rtigev_i6_512x640_fp16.engine}
BASE_CFG=${BASE_CFG:-$CFG_DIR/thermal_nodepth_td0_f10.yaml}
DEPTH_CFG=${DEPTH_CFG:-$CFG_DIR/thermal_depth_f10.yaml}
[ -d "$BAG" ] || { echo "bag not found: $BAG"; exit 1; }
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
source /opt/ros/humble/setup.bash

one_run() {  # $1 = base|depth  $2 = run dir
  local kind=$1 dir=$2 log=$2.launcher.log pid n prev
  rm -rf "$dir"
  if [ "$kind" = base ]; then
    RUN_DIR="$dir" NO_DEPTH=1 bash "$RT/run_live_trt.sh" x x x "$BASE_CFG" > "$log" 2>&1 &
  else
    RUN_DIR="$dir" DEPTH_ARGS="-p zmin:=0.4 -p zmax:=8.0" bash "$RT/run_live_trt.sh" "$ENGINE" \
      "$CFG_DIR/cam0_sensor.yaml" "$CFG_DIR/cam1_sensor.yaml" "$DEPTH_CFG" > "$log" 2>&1 &
  fi
  pid=$!
  for i in $(seq 90); do grep -q "Play the bag now" "$log" && break
    kill -0 $pid 2>/dev/null || { cat "$log"; echo "run failed to start: $dir"; return 1; }; sleep 1; done
  sleep 3   # VINS subscriptions up
  echo "  playing bag (rate $RATE) ..."
  ros2 bag play "$BAG" -r "$RATE" < /dev/null > "$dir/bagplay.log" 2>&1 || true
  # wait until VINS has worked off its queue (pose count unchanged for 6 s)
  prev=-1
  for i in $(seq 60); do
    n=$(grep -c "time: " "$dir/vins.log" 2>/dev/null || echo 0)
    [ "$n" = "$prev" ] && break
    prev=$n; sleep 3
  done
  kill -TERM $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  sleep 3
  echo "  done: $(grep -c 'time: ' "$dir/vins.log" 2>/dev/null || echo 0) VINS poses"
}

for r in $(seq "$REPS"); do
  # alternate the order so slow drifts (temperature, background load) do not favour one variant
  if [ $((r % 2)) = 1 ]; then order="base depth"; else order="depth base"; fi
  for k in $order; do
    [ -n "$ONLY" ] && [ "$ONLY" != "$k" ] && continue
    echo "=== repetition $r/$REPS: $k"
    one_run "$k" "$OUT/${k}_r$r"
  done
done
python3 "$HERE/compare_runs.py" "$OUT"
