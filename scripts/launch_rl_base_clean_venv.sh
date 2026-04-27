#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_BASE_WS:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${RL_BASE_VENV:-${WORKSPACE}/.venv}"
LAUNCH_FILE="${RL_BASE_LAUNCH_FILE:-start_rl_base_navigation.launch.py}"

if [ "${RL_BASE_CLEAN_ENV:-}" != "1" ]; then
  EXEC_USER="${USER:-$(id -un)}"
  EXEC_LOGNAME="${LOGNAME:-${EXEC_USER}}"

  CLEAN_ENV=(
    env -i
    HOME="${HOME}"
    USER="${EXEC_USER}"
    LOGNAME="${EXEC_LOGNAME}"
    SHELL="/bin/bash"
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    TERM="${TERM:-xterm-256color}"
    LANG="${LANG:-C.UTF-8}"
    RL_BASE_CLEAN_ENV="1"
    RL_BASE_WS="${WORKSPACE}"
    RL_BASE_VENV="${VENV_DIR}"
    RL_BASE_LAUNCH_FILE="${LAUNCH_FILE}"
  )

  [ -n "${DISPLAY+x}" ] && CLEAN_ENV+=(DISPLAY="${DISPLAY}")
  [ -n "${XAUTHORITY+x}" ] && CLEAN_ENV+=(XAUTHORITY="${XAUTHORITY}")
  [ -z "${XAUTHORITY+x}" ] && [ -n "${DISPLAY+x}" ] && CLEAN_ENV+=(XAUTHORITY="${HOME}/.Xauthority")
  [ -n "${WAYLAND_DISPLAY+x}" ] && CLEAN_ENV+=(WAYLAND_DISPLAY="${WAYLAND_DISPLAY}")
  [ -n "${XDG_RUNTIME_DIR+x}" ] && CLEAN_ENV+=(XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}")
  [ -n "${DBUS_SESSION_BUS_ADDRESS+x}" ] && CLEAN_ENV+=(DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS}")
  [ -n "${ROS_DOMAIN_ID+x}" ] && CLEAN_ENV+=(ROS_DOMAIN_ID="${ROS_DOMAIN_ID}")
  [ -n "${RMW_IMPLEMENTATION+x}" ] && CLEAN_ENV+=(RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}")
  [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && CLEAN_ENV+=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  [ -n "${NVIDIA_VISIBLE_DEVICES+x}" ] && CLEAN_ENV+=(NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}")
  [ -n "${NVIDIA_DRIVER_CAPABILITIES+x}" ] && CLEAN_ENV+=(NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}")

  exec "${CLEAN_ENV[@]}" bash "${SCRIPT_DIR}/launch_rl_base_clean_venv.sh" "$@"
fi

cd "${WORKSPACE}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
unset PYTHONNOUSERSITE
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_IP="${GAZEBO_IP:-127.0.0.1}"
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"

if [ -d /usr/local/cuda/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

set +u
source /opt/ros/humble/setup.bash
set -u

if [ ! -f "${WORKSPACE}/install/local_setup.bash" ]; then
  echo "Missing install/local_setup.bash. Build first:"
  echo "  source /opt/ros/humble/setup.bash"
  echo "  colcon build --symlink-install --packages-select depthimage_to_laserscan goal_seeker_rl --allow-overriding depthimage_to_laserscan"
  exit 2
fi

set +u
source "${WORKSPACE}/install/local_setup.bash"
set -u

if [ "${RL_BASE_KILL_OLD:-1}" = "1" ]; then
  pkill -9 -f "ros2 launch goal_seeker_rl start_rl_base_navigation[.]launch[.]py" 2>/dev/null || true
  pkill -9 -f "ros2 launch goal_seeker_rl start_goal_seeker[.]launch[.]py" 2>/dev/null || true
  pkill -9 -f "ros2 launch goal_seeker_rl start_hrl_navigation[.]launch[.]py" 2>/dev/null || true
  killall -9 -q gzserver gzclient gzmaster rviz2 2>/dev/null || true
  pkill -9 -f "goal_seeker_main|hrl_global_planner|rl_local_driver|depthimage_to_laserscan_node" 2>/dev/null || true
  sleep 1
fi

LAUNCH_ARGS=()
ROS_LAUNCH_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --*) ROS_LAUNCH_ARGS+=("${arg}") ;;
    *) LAUNCH_ARGS+=("${arg}") ;;
  esac
done
HAS_HEADLESS=0
HAS_RVIZ=0

for arg in "${LAUNCH_ARGS[@]}"; do
  case "${arg}" in
    headless:=*) HAS_HEADLESS=1 ;;
    use_rviz:=*) HAS_RVIZ=1 ;;
  esac
done

if [ "${HAS_HEADLESS}" = "0" ]; then
  LAUNCH_ARGS+=(headless:=true)
fi

if [ "${HAS_RVIZ}" = "0" ]; then
  LAUNCH_ARGS+=(use_rviz:=true)
fi

set -- "${LAUNCH_ARGS[@]}" "${ROS_LAUNCH_ARGS[@]}"

echo "Workspace: ${WORKSPACE}"
echo "Virtualenv: ${VIRTUAL_ENV}"
echo "Launch file: ${LAUNCH_FILE}"
echo "Launching with args: $*"

exec ros2 launch goal_seeker_rl "${LAUNCH_FILE}" "$@"
