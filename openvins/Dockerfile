FROM ros:noetic-ros-base-focal
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3-catkin-tools libeigen3-dev libboost-all-dev libceres-dev \
      ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-tf ros-noetic-rosbag \
      ros-noetic-message-filters ros-noetic-nav-msgs ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone --depth 1 --branch v2.7 https://github.com/rpng/open_vins.git
WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin config --extend /opt/ros/noetic -DCMAKE_BUILD_TYPE=Release && catkin build -j8 ov_msckf"
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin build -j8 ov_eval"
