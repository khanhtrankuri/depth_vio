#!/usr/bin/env bash
# Headless VINS-Fusion (ROS2) run on a bag, record /odometry, compute ATE/RPE with ov_eval.
# Usage: eval_vins.sh <ros2_bag_dir> <vins_config_dir_or_yaml> <gt_euroc_data.csv> [name] [rate=1.0]
# <config> may be a yaml file, or a config dir copied under ~/vio_data/configs (the yaml name is then required as <dir>/<file>).
D=$(cd "$(dirname "$0")" && pwd)
BAG=${1:?bag dir}; CFG=${2:?vins yaml}; GTCSV=${3:?euroc data.csv}
NAME=${4:-vins_$(date +%H%M%S)}; RATE=${5:-1.0}
source "$D/common.sh"
source "$HOME/ws_vins/install/setup.bash"
OUT=$VIO_HOME/results/$NAME; mkdir -p "$OUT"

pkill -INT -f "vins_node" 2>/dev/null; sleep 1
python3 "$D/convert_gt.py" "$GTCSV" "$OUT/gt.txt"

ros2 run vins vins_node "$CFG" > "$OUT/vins.log" 2>&1 &
VINS=$!
sleep 8
python3 "$D/record_traj.py" "$OUT/est.txt" /odometry odom > "$OUT/recorder.log" 2>&1 &
REC=$!
sleep 2
ros2 bag play "$BAG" -r "$RATE" > "$OUT/bagplay.log" 2>&1
sleep 4
kill -INT $REC 2>/dev/null; wait $REC 2>/dev/null
kill -INT $VINS 2>/dev/null; sleep 2; pkill -KILL -f vins_node 2>/dev/null
cat "$OUT/recorder.log"
bash "$D/ate_only.sh" "$OUT"
echo "Results in $OUT"
