#!/usr/bin/env bash
# Copy the EuRoC VINS-Fusion configs to ~/vio_data/configs/vins_euroc and point output paths at a results dir.
# Usage: prep_vins_euroc_cfg.sh <results_name>
NAME=${1:?results name}
SRC=$HOME/ws_vins/src/VINS-Fusion-ROS2-Humble/config/euroc
DST=$HOME/vio_data/configs/vins_euroc
RES=$HOME/vio_data/results/$NAME
mkdir -p "$DST" "$RES/pose_graph"
cp "$SRC"/*.yaml "$DST"/
sed -i "s#^output_path:.*#output_path: \"$RES\"#; s#^pose_graph_save_path:.*#pose_graph_save_path: \"$RES/pose_graph/\"#" "$DST/euroc_stereo_imu_config.yaml"
grep -n -E 'output_path|pose_graph_save|cam._calib|use_gpu|multiple_thread|^imu:|num_of_cam' "$DST/euroc_stereo_imu_config.yaml"
