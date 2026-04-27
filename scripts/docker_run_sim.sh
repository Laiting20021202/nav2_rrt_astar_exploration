#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${RL_BASE_DOCKER_IMAGE:-rl-base-navigation:stretch3}"
ROS_DOMAIN_ID_VALUE="${ROS_DOMAIN_ID:-0}"

X11_ARGS=()
if [ -n "${DISPLAY:-}" ]; then
  X11_ARGS+=(-e DISPLAY="${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ]; then
    X11_ARGS+=(-e XAUTHORITY=/tmp/.docker.xauth -v "${XAUTHORITY}:/tmp/.docker.xauth:ro")
  fi
fi

exec docker run --rm -it \
  --name rl-base-navigation-sim \
  --network host \
  --ipc host \
  --privileged \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID_VALUE}" \
  -e RL_BASE_WS=/workspace \
  -e RL_BASE_MODEL_DIR=/workspace/navigation_model \
  "${X11_ARGS[@]}" \
  "${IMAGE_NAME}" \
  ros2 launch goal_seeker_rl start_hrl_navigation.launch.py headless:=true use_rviz:=false "$@"
