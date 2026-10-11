#!/usr/bin/env bash
# One Wuji glove drives simulated Hand 2 fingers and Vega's three wrist joints.
# Vega's shoulder and elbow are physically fixed in this MuJoCo scene.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [LEFT_GLOVE_SERIAL]"
  echo "Default: the only glove found on the network (or \$GLOVE_SN)."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
set_glove_args "${1:-}"
log_run "$@"

echo "Starting glove fingers + palm IMU wrist rotation in MuJoCo."
echo "Hold the glove still for one second to calibrate. Press R to recenter."
python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input glove-wrist --finger-input glove "${glove_args[@]}"
  --report /tmp/vega-glove-wrist-live.json
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
