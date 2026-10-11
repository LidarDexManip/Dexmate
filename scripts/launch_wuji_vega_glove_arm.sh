#!/usr/bin/env bash
# Wuji glove fingers plus whole-arm IK.
# Palm orientation and directional strokes set a 6D wrist target.
# Damped least squares solves Vega's seven left-arm joints.
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

echo "Starting glove fingers + whole-arm IK in MuJoCo."
echo "Hold the glove still for one second to calibrate."
echo "Point the fingers up to raise the whole arm from the shoulder (straight up raises it overhead); point down to lower it."
echo "Tip side to side to sweep. The hand stays at full reach, so the elbow holds its bend."
echo "Lower the hand to tuck the elbow first; the shoulder follows after the elbow starts bending."
echo "Twist the forearm to roll the robot forearm. The arm follows slowly and returns with the calibrated pose."
echo "Press R while still to recenter."
python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input glove-arm --finger-input glove "${glove_args[@]}"
  --report /tmp/vega-glove-arm-live.json
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
