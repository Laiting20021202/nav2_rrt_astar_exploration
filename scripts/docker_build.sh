#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${RL_BASE_DOCKER_IMAGE:-rl-base-navigation:stretch3}"

cd "${WORKSPACE}"
docker build -t "${IMAGE_NAME}" .
