#!/usr/bin/env bash
# Shallow-clone candidate ROS2 VINS-Fusion ports into ~/vins_candidates and summarise them (read-only probe).
mkdir -p ~/vins_candidates && cd ~/vins_candidates
for r in JanekDev/VINS-Fusion-ROS2-humble-arm fanhong-li/VINS-Fusion-ROS2-Humble zinuok/VINS-Fusion-ROS2; do
  d=$(basename "$r")
  [ -d "$d" ] || git clone --depth 1 "https://github.com/$r.git" "$d" >/dev/null 2>&1 || { echo "== $r: CLONE FAILED"; continue; }
  echo "=================== $r"
  git -C "$d" log -1 --format='last commit: %ad | %s' --date=short
  ls "$d"
  grep -n -i -E 'humble|foxy|galactic|ceres|opencv|eigen|colcon|cuda|gpu' "$d"/README.md | head -14
  find "$d" -name package.xml | head -8
done
