#!/usr/bin/env bash
# Open RViz2 on the Windows desktop (WSLg) with OpenVINS' ROS2 display config.
source "$(dirname "$0")/common.sh"
CFG=$(ros2 pkg prefix ov_msckf)/share/ov_msckf/launch/display_ros2.rviz
[ -f "$CFG" ] || CFG=$HOME/ws_ov/src/open_vins/ov_msckf/launch/display_ros2.rviz
rviz2 -d "$CFG" --ros-args --log-level warn 2>&1 | tee "$LOG_DIR/rviz_$STAMP.log"
