#!/usr/bin/env bash
# Environment health check (run inside WSL Ubuntu-22.04). Prints PASS/FAIL per component.
source /opt/ros/humble/setup.bash 2>/dev/null
[ -f "$HOME/ws_ov/install/setup.bash" ] && source "$HOME/ws_ov/install/setup.bash"

row() { printf '| %-22s | %-5s | %-28s | %s\n' "$1" "$2" "$3" "$4"; }
echo "| Component              | State | Version                      | Notes"
echo "|------------------------|-------|------------------------------|------"

. /etc/os-release
[[ "$VERSION_ID" == "22.04" ]] && row Ubuntu PASS "$PRETTY_NAME" "" || row Ubuntu FAIL "$PRETTY_NAME" "need 22.04"
row "WSL kernel" PASS "$(uname -r)" ""

[[ -n "${DISPLAY:-}" && -d /mnt/wslg ]] && row WSLg PASS "DISPLAY=$DISPLAY" "" || row WSLg FAIL "" "no DISPLAY or /mnt/wslg"
[[ -e /dev/dxg ]] && row "GPU (/dev/dxg)" PASS "" "$(glxinfo -B 2>/dev/null | grep 'OpenGL renderer' | cut -d: -f2)" || row "GPU (/dev/dxg)" FAIL "" "software rendering only"

[[ "${ROS_DISTRO:-}" == "humble" ]] && row ROS2 PASS "humble" "$(ros2 --version 2>&1 | head -1)" || row ROS2 FAIL "${ROS_DISTRO:-none}" ""
command -v rviz2 >/dev/null && row RViz2 PASS "$(dpkg-query -W -f='${Version}' ros-humble-rviz2 2>/dev/null)" "" || row RViz2 FAIL "" "not installed"
ros2 bag --help >/dev/null 2>&1 && row rosbag2 PASS "" "" || row rosbag2 FAIL "" ""
command -v colcon >/dev/null && row colcon PASS "" "" || row colcon FAIL "" ""
command -v gdb >/dev/null && row gdb PASS "$(gdb --version | head -1 | awk '{print $NF}')" "" || row gdb FAIL "" ""
pkg-config --exists opencv4 && row OpenCV PASS "$(pkg-config --modversion opencv4)" "" || row OpenCV FAIL "" ""
pkg-config --exists eigen3 && row Eigen PASS "$(pkg-config --modversion eigen3)" "" || row Eigen FAIL "" ""
[[ -f /usr/include/ceres/ceres.h ]] && row Ceres PASS "$(dpkg-query -W -f='${Version}' libceres-dev 2>/dev/null)" "" || row Ceres FAIL "" ""

if ros2 pkg prefix ov_msckf >/dev/null 2>&1; then
  row OpenVINS PASS "ov_msckf built" "$(ros2 pkg prefix ov_msckf)"
else
  row OpenVINS FAIL "" "ov_msckf not found in sourced workspace"
fi
df -h / | tail -1 | awk '{print "disk /: used " $3 " free " $4}'
free -g | awk '/Mem:/{print "RAM total " $2 "G free " $7 "G"}'
