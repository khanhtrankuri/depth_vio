#!/usr/bin/env bash
# Install ROS2 Humble (desktop) + build tooling on Ubuntu 22.04. Idempotent.
set -euo pipefail

. /etc/os-release
[[ "$VERSION_ID" == "22.04" ]] || { echo "Requires Ubuntu 22.04, found $VERSION_ID"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y software-properties-common curl gnupg lsb-release locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  ros-humble-desktop ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins \
  ros-humble-cv-bridge ros-humble-image-transport ros-humble-tf2-geometry-msgs \
  ros-dev-tools python3-colcon-common-extensions python3-rosdep python3-vcstool \
  build-essential cmake git gdb libeigen3-dev libopencv-dev libboost-all-dev \
  libceres-dev mesa-utils x11-apps

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

grep -q "opt/ros/humble/setup.bash" ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# --- ROS2 Humble ---
source /opt/ros/humble/setup.bash
[ -f "$HOME/ws_ov/install/setup.bash" ] && source "$HOME/ws_ov/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=1
EOF
echo "ROS2 Humble install finished."
