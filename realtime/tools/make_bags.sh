#!/usr/bin/env bash
# Convert the EuRoC Vicon Room 1 sequences into ROS2 bags and copy GT (run inside WSL).
set -eo pipefail
R=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/venv_bag/bin/python}
for s in V1_01_easy V1_02_medium V1_03_difficult; do
  [ -d "$HOME/vio_data/bags/${s}_r2" ] || "$PY" "$R/realtime/tools/euroc_to_ros2bag.py" "$R/data/$s/mav0" "$HOME/vio_data/bags/${s}_r2" | tail -1
  mkdir -p "$HOME/vio_data/gt/$s"
  cp "$R/data/$s/mav0/state_groundtruth_estimate0/data.csv" "$HOME/vio_data/gt/$s/"
done
du -sh "$HOME"/vio_data/bags/*
