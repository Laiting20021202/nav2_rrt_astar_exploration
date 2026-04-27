#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${RL_BASE_DOCKER_IMAGE:-rl-base-navigation:stretch3}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION_VALUE="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

exec docker run --rm -it \
  --name rl-base-navigation-stretch3 \
  --network host \
  --ipc host \
  --privileged \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID_VALUE}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION_VALUE}" \
  -e RL_BASE_WS=/workspace \
  -e RL_BASE_MODEL_DIR=/workspace/navigation_model \
  "${IMAGE_NAME}" \
  ros2 launch goal_seeker_rl start_stretch3_far_planner.launch.py "$@"
