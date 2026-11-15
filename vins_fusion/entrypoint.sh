#!/bin/bash
set -eo pipefail
source /opt/ros/melodic/setup.bash
source /catkin_ws/devel/setup.bash
set -u
export ROS_MASTER_URI=http://localhost:11311 ROS_HOSTNAME=localhost ROS_HOME=/tmp/ros
if [ "$1" = "bag" ]; then
    exec python /integration/make_bag.py /data /out/euroc.bag
fi
cp -r /catkin_ws/src/VINS-Fusion/config/euroc /tmp/euroc
python /integration/configure_euroc.py
sed -i 's|output_path:.*|output_path: "/out/"|' /tmp/euroc/euroc_stereo_imu_config.yaml
# Fixed official EuRoC extrinsics and td=0 match the mapper's T_I_C contract.
cp /tmp/euroc/euroc_stereo_imu_config.yaml /out/estimator_config.yaml
cp /tmp/euroc/cam?_mei.yaml /out/
roscore > /out/roscore.log 2>&1 &
master=$!
estimator=''
cleanup() {
    if [ -n "$estimator" ]; then kill -INT "$estimator" 2>/dev/null || true; wait "$estimator" 2>/dev/null || true; fi
    kill "$master" 2>/dev/null || true
}
trap cleanup EXIT
python /integration/wait_ros.py master
stdbuf -oL -eL rosrun vins vins_node /tmp/euroc/euroc_stereo_imu_config.yaml > /out/estimator.log 2>&1 &
estimator=$!
python /integration/wait_ros.py sensors
rosbag play --clock --delay=2 --rate="${PLAYBACK_RATE:-0.5}" "${BAG_PATH:-/out/euroc.bag}" > /out/playback.log 2>&1
# Allow asynchronous optimisation to drain after the last input.
sleep "${DRAIN_SECONDS:-15}"
kill -0 "$estimator"
python3 /integration/convert_trajectory.py /out/vio.csv /out/vins_fusion_traj_tum.txt
