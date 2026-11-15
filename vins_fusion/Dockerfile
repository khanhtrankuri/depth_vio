FROM ros:melodic-perception-bionic
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python-catkin-tools python3 libceres-dev libeigen3-dev libboost-all-dev \
    ros-melodic-tf ros-melodic-image-transport ros-melodic-rosbag \
    && rm -rf /var/lib/apt/lists/*
ARG VINS_REF=be55a937a57436548ddfb1bd324bc1e9a9e828e0
RUN mkdir -p /catkin_ws/src && git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git /catkin_ws/src/VINS-Fusion \
    && cd /catkin_ws/src/VINS-Fusion && git checkout ${VINS_REF}
WORKDIR /catkin_ws
RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash && catkin config --extend /opt/ros/melodic -DCMAKE_BUILD_TYPE=Release && catkin build -j2 vins'
COPY vins_fusion /integration
ENTRYPOINT ["/bin/bash", "/integration/entrypoint.sh"]
