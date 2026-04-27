#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_BASE_WS:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MAP_DIR="${1:-${WORKSPACE}/maps}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${MAP_DIR}/rl_map_${STAMP}"

mkdir -p "${MAP_DIR}"
cd "${WORKSPACE}"

if [ -d "${WORKSPACE}/.venv" ]; then
  source "${WORKSPACE}/.venv/bin/activate"
fi

set +u
source /opt/ros/humble/setup.bash
[ -f "${WORKSPACE}/install/local_setup.bash" ] && source "${WORKSPACE}/install/local_setup.bash"
set -u

ros2 run nav2_map_server map_saver_cli -f "${OUTPUT}"
echo "Saved map: ${OUTPUT}.yaml"
