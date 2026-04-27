#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_BASE_WS:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${RL_BASE_VENV:-${WORKSPACE}/.venv}"

if [ "${RL_BASE_CLEAN_ENV:-}" != "1" ]; then
  exec env -i \
    HOME="${HOME}" \
    USER="${USER:-$(id -un)}" \
    LOGNAME="${LOGNAME:-${USER:-$(id -un)}}" \
    SHELL="/bin/bash" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    TERM="${TERM:-xterm-256color}" \
    LANG="${LANG:-C.UTF-8}" \
    RL_BASE_CLEAN_ENV="1" \
    RL_BASE_WS="${WORKSPACE}" \
    RL_BASE_VENV="${VENV_DIR}" \
    bash "${SCRIPT_DIR}/build_rl_base_clean_venv.sh" "$@"
fi

cd "${WORKSPACE}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
unset PYTHONNOUSERSITE

set +u
source /opt/ros/humble/setup.bash
set -u

colcon build \
  --symlink-install \
  --packages-select depthimage_to_laserscan goal_seeker_rl \
  --allow-overriding depthimage_to_laserscan "$@"
