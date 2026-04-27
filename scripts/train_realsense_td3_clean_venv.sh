#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_BASE_WS:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_DIR="${WORKSPACE}/navigation_model"
TRAIN_DIR="${RL_BASE_TRAIN_DIR:-${MODEL_DIR}/realsense_td3_current}"

mkdir -p "${TRAIN_DIR}"

export RL_BASE_LAUNCH_FILE="start_goal_seeker.launch.py"
exec "${SCRIPT_DIR}/launch_rl_base_clean_venv.sh" \
  headless:=true \
  use_rviz:=false \
  inference_mode:=false \
  auto_goal_training:=true \
  resume_model_path:="${MODEL_DIR}/td3_latest.pth" \
  checkpoint_dir:="${TRAIN_DIR}" \
  checkpoint_interval_steps:=1000 \
  warmup_steps:=500 \
  exploration_std:=0.10 \
  goal_tolerance:=0.38 \
  collision_distance:=0.34 \
  linear_speed_max:=0.20 \
  angular_speed_max:=1.0 \
  "$@"
