FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV RL_BASE_WS=/workspace
ENV RL_BASE_MODEL_DIR=/workspace/navigation_model
ENV GAZEBO_MODEL_DATABASE_URI=
ENV TURTLEBOT3_MODEL=waffle

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-numpy \
    python3-pip \
    python3-torch \
    python3-venv \
    ros-humble-ament-cmake \
    ros-humble-cv-bridge \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-image-geometry \
    ros-humble-image-transport \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-slam-toolbox \
    ros-humble-tf2-geometry-msgs \
    ros-humble-turtlebot3-gazebo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN source /opt/ros/humble/setup.bash \
    && colcon build \
      --symlink-install \
      --packages-select depthimage_to_laserscan goal_seeker_rl

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
