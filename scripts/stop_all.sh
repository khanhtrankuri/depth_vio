#!/usr/bin/env bash
# Gracefully stop OpenVINS, bag playback and RViz2 (SIGINT first, SIGKILL only for stragglers).
for pat in "ros2 bag play" "run_subscribe_msckf" "ros2 launch ov_msckf" "rviz2"; do
  pkill -INT -f "$pat" 2>/dev/null && echo "SIGINT -> $pat"
done
sleep 3
for pat in "ros2 bag play" "run_subscribe_msckf" "ros2 launch ov_msckf" "rviz2"; do
  pgrep -f "$pat" >/dev/null && { pkill -KILL -f "$pat"; echo "SIGKILL -> $pat"; }
done
echo "Stopped."
