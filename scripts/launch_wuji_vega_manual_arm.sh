#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [J1 J2 J3 J4 J5 J6 J7]   (radians; opens a 60 s viewer with sliders)"
  exit 0
fi
cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
log_run "$@"

# Values are radians. Optional arguments override these defaults. A slider
# panel opens with the MuJoCo viewer for live editing.
J1="${1:-1.39}"
J2="${2:-0.06}"
J3="${3:-0.14}"
J4="${4:--1.38}"
J5="${5:-0.0}"
J6="${6:-0.0}"
J7="${7:-0.0}"

python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input manual --finger-input synthetic --duration 60
  --joints "$J1" "$J2" "$J3" "$J4" "$J5" "$J6" "$J7"
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"