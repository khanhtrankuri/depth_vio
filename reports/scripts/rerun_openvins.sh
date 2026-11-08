#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
export ROS_HOME=/tmp/ros
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
exec roslaunch /cfg/run_euroc.launch bag:="/bags/${EUROC_SEQ}.bag"
