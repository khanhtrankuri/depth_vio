#!/usr/bin/env bash
# Compute ATE/RPE (ov_eval) for an existing results dir containing gt.txt and est.txt. No GUI windows.
# Usage: ate_only.sh <results_dir>
source "$(dirname "$0")/common.sh"
OUT=${1:?results dir}
export MPLBACKEND=Agg
for MODE in posyaw se3; do
  echo; echo "######## align mode: $MODE ########"
  stdbuf -oL ros2 run ov_eval error_singlerun $MODE "$OUT/gt.txt" "$OUT/est.txt" 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | tee "$OUT/ate_$MODE.txt"
done
