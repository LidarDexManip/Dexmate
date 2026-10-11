#!/usr/bin/env bash
# Jog each simulated left-arm joint inside its limits. No glove or headset needed.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0"
  echo "Opens a jog window (slider and -/+ per joint) beside the MuJoCo viewer."
  echo "Limits live in ARM_JOINT_LIMITS in wuji_teleop/vega_arm_sim.py."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
log_run "$@"

python_cmd=("$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_range)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
