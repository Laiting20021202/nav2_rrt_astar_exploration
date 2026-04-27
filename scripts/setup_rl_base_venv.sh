#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${RL_BASE_WS:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${RL_BASE_VENV:-${WORKSPACE}/.venv}"

cd "${WORKSPACE}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
unset PYTHONNOUSERSITE
set +u
source /opt/ros/humble/setup.bash
set -u

python - <<'PY'
import importlib

for module_name in ("numpy", "torch", "rclpy"):
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "ok")
    print(f"{module_name}: {version}")
PY

echo "Virtual environment ready: ${VENV_DIR}"
