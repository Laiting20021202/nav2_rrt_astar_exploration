#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f "${RL_BASE_WS:-/workspace}/install/local_setup.bash" ]; then
  source "${RL_BASE_WS:-/workspace}/install/local_setup.bash"
fi

export GAZEBO_MODEL_DATABASE_URI="${GAZEBO_MODEL_DATABASE_URI:-}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"

exec "$@"
