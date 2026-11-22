#!/usr/bin/env bash
# One live run: [learned depth node] + patched VINS-Fusion + bag playback at real-time rate, then ATE/RPE.
# Usage (inside WSL): run_vio.sh <SEQ> <run_name> <depth:0|1> [key=value ...]
#   keys: model=rt:sceneflow:6 weight=4 period=1.0 wait_ms=80 rel_range=0.08 init=1 rate=1.0
#         sigma_px=1.0 relative_sigma=0.025 zmax=4.0 max_rate_hz=0 photo=0 lr=0 delay=0 gate=10
#         dist=0 keep=0 maxcnt=150 mindist=30 freq=10 solver_time=0.04 iters=8 loop=0
#   final config: weight=16 gate=0.1 period=0.2 maxcnt=200 mindist=25 delay=1 dist=0.5
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
SEQ=${1:?seq}; NAME=${2:?name}; DEPTH=${3:?0|1}; shift 3
model=rt:sceneflow:6; weight=4; period=1.0; wait_ms=80; rel_range=0.08; init=1; rate=1.0
sigma_px=1.0; relative_sigma=0.025; zmax=4.0; max_rate_hz=0; maxcnt=150; photo=0; lr=0; delay=0; gate=10
loop=0; solver_time=0.04; iters=8; freq=10; mindist=30; keep=0; dist=0
for kv in "$@"; do eval "$kv"; done

source /opt/ros/humble/setup.bash
source "$HOME/ws_vins_depth/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-43} ROS_LOCALHOST_ONLY=1
OUT=$HOME/vio_data/rt/runs/$SEQ/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"
BAG=$HOME/vio_data/bags/${SEQ}_r2
CFGSRC=$HOME/vio_data/configs/vins_euroc
cp "$CFGSRC"/cam0_mei.yaml "$CFGSRC"/cam1_mei.yaml "$OUT/"
sed -e "s#^output_path:.*#output_path: \"$OUT/\"#" -e "s#^show_track:.*#show_track: 0#" \
    -e "s#^max_cnt:.*#max_cnt: $maxcnt#" -e "s#^min_dist:.*#min_dist: $mindist#" -e "s#^freq:.*#freq: $freq#" \
    -e "s#^max_solver_time:.*#max_solver_time: $solver_time#" -e "s#^max_num_iterations:.*#max_num_iterations: $iters#" \
    -e "s#^pose_graph_save_path:.*#pose_graph_save_path: \"$OUT/pg/\"#" -e "s#^save_image:.*#save_image: 0#" \
    "$CFGSRC/euroc_stereo_imu_config.yaml" > "$OUT/config.yaml"
if [ "$DEPTH" = 1 ]; then
  cat >> "$OUT/config.yaml" <<EOF

# live learned depth
depth_topic: "/depth/inverse_depth"
depth_weight: $weight
depth_period: $period
depth_wait_ms: $wait_ms
depth_max_rel_range: $rel_range
depth_init_landmarks: $init
depth_start_delay: $delay
depth_gate: $gate
depth_keep_stereo: $keep
depth_start_dist: $dist
EOF
fi
printf '%s\n' "seq=$SEQ depth=$DEPTH $*" > "$OUT/params.txt"

PAT="vins_node|loop_fusion_node|depth_node.py|record_traj.py|ros2 bag play"
cleanup() { pkill -INT -f "$PAT" 2>/dev/null; sleep 3; pkill -KILL -f "$PAT" 2>/dev/null; sleep 1; }
cleanup  # never share the ROS graph with leftovers of a previous run
trap cleanup EXIT
DN=
if [ "$DEPTH" = 1 ]; then
  MAV=$REPO/data/$SEQ/mav0
  (cd "$REPO/realtime/depth_node" && exec "$HOME/venv_depth/bin/python" depth_node.py --ros-args \
     -p model:=$model -p cam0_yaml:=$MAV/cam0/sensor.yaml -p cam1_yaml:=$MAV/cam1/sensor.yaml \
     -p sigma_px:=$sigma_px -p relative_sigma:=$relative_sigma -p zmax:=$zmax -p max_rate_hz:=$max_rate_hz -p photo_thresh:=$photo -p lr_thresh:=$lr \
     -p stats_csv:=$OUT/depth_stats.csv) > "$OUT/depth_node.log" 2>&1 &
  DN=$!
  for i in $(seq 120); do grep -q "learned depth ready" "$OUT/depth_node.log" && break; sleep 1; done
  grep -q "learned depth ready" "$OUT/depth_node.log" || { echo "depth node failed"; tail "$OUT/depth_node.log"; exit 1; }
fi
ros2 run vins vins_node "$OUT/config.yaml" > "$OUT/vins.log" 2>&1 &
VINS=$!
"$HOME/venv_depth/bin/python" "$REPO/scripts/record_traj.py" "$OUT/est.txt" /odometry odom > "$OUT/recorder.log" 2>&1 &
REC=$!
if [ "$loop" = 1 ]; then
  # 4-DoF pose-graph loop closure; /odometry_rect is the causal (real-time) loop-corrected pose
  ros2 run loop_fusion loop_fusion_node "$OUT/config.yaml" --ros-args \
     -r /vins_estimator/odometry:=/odometry -r /vins_estimator/keyframe_pose:=/keyframe_pose \
     -r /vins_estimator/extrinsic:=/extrinsic -r /vins_estimator/keyframe_point:=/keyframe_point \
     -r /vins_estimator/margin_cloud:=/margin_cloud -r camera_pose_visual:=loop_camera_pose_visual \
     < /dev/null > "$OUT/loop.log" 2>&1 &
  "$HOME/venv_depth/bin/python" "$REPO/scripts/record_traj.py" "$OUT/est_loop.txt" /odometry_rect odom > "$OUT/recorder_loop.log" 2>&1 &
fi
sleep 4
ros2 bag play "$BAG" -r "$rate" > "$OUT/bagplay.log" 2>&1
sleep 3
cleanup; trap - EXIT
wait 2>/dev/null
"$HOME/venv_depth/bin/python" "$HERE/eval_traj.py" "$OUT/est.txt" "$HOME/vio_data/gt/$SEQ/data.csv" --out "$OUT/metrics.json" \
   --vins-log "$OUT/vins.log" --depth-stats "$OUT/depth_stats.csv"
if [ "$loop" = 1 ]; then
  mv "$OUT/metrics.json" "$OUT/metrics_vio.json"   # summaries use metrics.json = loop-corrected real-time pose
  "$HOME/venv_depth/bin/python" "$HERE/eval_traj.py" "$OUT/est_loop.txt" "$HOME/vio_data/gt/$SEQ/data.csv" --out "$OUT/metrics.json" \
     --vins-log "$OUT/vins.log" --depth-stats "$OUT/depth_stats.csv"
fi
