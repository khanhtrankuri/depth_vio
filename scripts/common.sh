#!/usr/bin/env bash
# Shared environment for all run_*.sh scripts. Source, don't execute.
# ROS setup scripts are incompatible with `set -u`, so it is not enabled here.
source /opt/ros/humble/setup.bash
[ -f "$HOME/ws_ov/install/setup.bash" ] && source "$HOME/ws_ov/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=1
export VIO_HOME=${VIO_HOME:-$HOME/vio_data}
export LOG_DIR=${LOG_DIR:-$VIO_HOME/logs}
export PID_DIR=${PID_DIR:-$VIO_HOME/pids}
mkdir -p "$LOG_DIR" "$PID_DIR" "$VIO_HOME/bags" "$VIO_HOME/configs" "$VIO_HOME/results"
STAMP=$(date +%Y%m%d_%H%M%S)
# Set VIO_SOFTWARE_GL=1 to force software rendering if the D3D12 GPU path misbehaves.
[ "${VIO_SOFTWARE_GL:-0}" = "1" ] && export LIBGL_ALWAYS_SOFTWARE=1
