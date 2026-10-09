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
echo "Starting glove fingers + whole-arm IK in MuJoCo."
echo "Hold the glove still for one second to calibrate."
echo "Tip the fingers up and down to swing the shoulder forward and back. Tip side to side to sweep."
echo "The arm follows slowly, and returns when you return to the calibrated pose."
echo "Twist to aim the wrist once the hand is close. A firm push nudges the hand without retipping."
echo "Press R while still to recenter."
exec "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim \
  --arm-input glove-arm --finger-input glove \
  --glove-sn "$glove_serial" \
  --report /tmp/vega-glove-arm-live.json
