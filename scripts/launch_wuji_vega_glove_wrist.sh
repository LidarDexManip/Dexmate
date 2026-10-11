#!/usr/bin/env bash
# One Wuji glove drives simulated Hand 2 fingers and Vega's three wrist joints.
# Vega's shoulder and elbow are physically fixed in this MuJoCo scene.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
glove_serial="${1:-WG1JA06260812032}"

if [[ "$glove_serial" == "--help" || "$glove_serial" == "-h" ]]; then
  echo "Usage: $0 [LEFT_GLOVE_SERIAL]"
  echo "Default glove: WG1JA06260812032"
  exit 0
fi
if [[ ! -x "$workspace/.venv/bin/python" ]]; then
  echo "Python environment missing: ${workspace}/.venv/bin/python" >&2
  exit 1
fi

cd "$workspace"
# shellcheck source=scripts/_log_run.sh
source "$(dirname "${BASH_SOURCE[0]}")/_log_run.sh"
log_run "$@"

echo "Starting glove fingers + palm IMU wrist rotation in MuJoCo."
echo "Hold the glove still for one second to calibrate. Press R to recenter."
python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input glove-wrist --finger-input glove
  --glove-sn "$glove_serial"
  --report /tmp/vega-glove-wrist-live.json
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
