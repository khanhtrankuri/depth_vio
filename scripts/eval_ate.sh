#!/usr/bin/env bash
# Headless run (no RViz) of OpenVINS on a bag, record trajectory, then compute ATE/RPE vs ground truth with ov_eval.
# Usage: eval_ate.sh <ros2_bag_dir> <config> <gt_euroc_data.csv> [name] [rate=1.0] [max_cameras=2] [use_stereo=true]
D=$(cd "$(dirname "$0")" && pwd)
BAG=${1:?bag dir}; CFG=${2:?config}; GTCSV=${3:?euroc data.csv}
NAME=${4:-run_$(date +%H%M%S)}; RATE=${5:-1.0}; MAXCAM=${6:-2}; STEREO=${7:-true}
source "$D/common.sh"
OUT=$VIO_HOME/results/$NAME; mkdir -p "$OUT"

bash "$D/stop_all.sh" >/dev/null 2>&1
python3 "$D/convert_gt.py" "$GTCSV" "$OUT/gt.txt"

nohup bash "$D/run_openvins.sh" "$CFG" "$MAXCAM" "$STEREO" >/dev/null 2>&1 &
sleep 8
python3 "$D/record_traj.py" "$OUT/est.txt" > "$OUT/recorder.log" 2>&1 &
REC=$!
sleep 2
ros2 bag play "$BAG" -r "$RATE" > "$OUT/bagplay.log" 2>&1     # blocks until the bag ends
sleep 3
kill -INT $REC 2>/dev/null; wait $REC 2>/dev/null
cat "$OUT/recorder.log"
cp "$(ls -t "$LOG_DIR"/openvins_*.log | head -1)" "$OUT/openvins.log"
bash "$D/stop_all.sh" >/dev/null 2>&1

for MODE in posyaw se3; do
  echo; echo "######## align mode: $MODE ########"
  ros2 run ov_eval error_singlerun $MODE "$OUT/gt.txt" "$OUT/est.txt" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$OUT/ate_$MODE.txt"
done
echo "Results in $OUT"
