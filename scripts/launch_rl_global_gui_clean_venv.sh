#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RL_BASE_LAUNCH_FILE="start_hrl_navigation.launch.py"
exec "${SCRIPT_DIR}/launch_rl_base_clean_venv.sh" headless:=false use_rviz:=true "$@"
