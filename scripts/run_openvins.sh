#!/usr/bin/env bash
# Run the OpenVINS subscriber node. Usage:
#   run_openvins.sh <config_name | /path/to/estimator_config.yaml> [max_cameras=1|2] [use_stereo=true|false] [extra ros args]
# Config names are the folders under open_vins/config (euroc_mav, tum_vi, ...) or ~/vio_data/configs.
source "$(dirname "$0")/common.sh"
CFG=${1:?usage: run_openvins.sh <config_name|estimator_config.yaml> [max_cameras] [use_stereo]}
MAXCAM=${2:-2}
STEREO=${3:-true}
shift $(( $# > 3 ? 3 : $# ))

if [ -f "$CFG" ]; then ARGS=(config_path:="$CFG"); else ARGS=(config:="$CFG"); fi
LOG="$LOG_DIR/openvins_$STAMP.log"
echo "Logging to $LOG"
ros2 launch ov_msckf subscribe.launch.py "${ARGS[@]}" max_cameras:="$MAXCAM" use_stereo:="$STEREO" \
  verbosity:=INFO save_total_state:=false "$@" 2>&1 | tee "$LOG"
