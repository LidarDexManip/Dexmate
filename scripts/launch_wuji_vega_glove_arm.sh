#!/usr/bin/env bash
# Wuji glove fingers plus whole-arm IK.
# Palm orientation and directional strokes set a 6D wrist target.
# Damped least squares solves Vega's seven left-arm joints.
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

echo "Starting glove fingers + whole-arm IK in MuJoCo."
echo "Hold the glove still for one second to calibrate."
echo "Point the fingers up to raise the whole arm from the shoulder (straight up raises it overhead); point down to lower it."
echo "Tip side to side to sweep. The hand stays at full reach, so the elbow holds its bend."
echo "Lower the hand to tuck the elbow first; the shoulder follows after the elbow starts bending."
echo "Twist the forearm to roll the robot forearm. The arm follows slowly and returns with the calibrated pose."
echo "Press R while still to recenter."
python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input glove-arm --finger-input glove
  --glove-sn "$glove_serial"
  --report /tmp/vega-glove-arm-live.json
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
