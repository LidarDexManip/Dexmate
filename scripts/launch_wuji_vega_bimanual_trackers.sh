#!/usr/bin/env bash
# PICO Motion Trackers on both wrists (XRoboToolkit object tracking) -> both
# simulated Vega arms through GMR-style IK; matching Wuji gloves -> fingers.
# Never starts robot control.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [LEFT_GLOVE_SERIAL] [RIGHT_GLOVE_SERIAL]"
  echo "Defaults: left glove .100 and right glove .101; trackers assigned by position."
  echo "Env overrides: LEFT_TRACKER_SN, RIGHT_TRACKER_SN, HUMAN_HEIGHT (m), IK_CONFIG,"
  echo "  FINGER_INPUT=synthetic (no gloves) or left-glove/right-glove (one glove)."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
if ! "$workspace/.venv/bin/python" -c 'import mink, xrobotoolkit_sdk' 2>/dev/null; then
  echo "Tracker input needs mink and xrobotoolkit_sdk in .venv." >&2
  echo "Run scripts/setup_sim.sh, then scripts/install_xrobotoolkit_sdk.sh." >&2
  exit 1
fi
service_dir=/opt/apps/roboticsservice
if ! pgrep -f RoboticsServiceProcess >/dev/null; then
  if [[ ! -f "$service_dir/runService.sh" ]]; then
    echo "XRoboToolkit PC Service is not installed (${service_dir})." >&2
    exit 1
  fi
  echo "Starting the XRoboToolkit PC Service."
  (cd "$service_dir" && bash runService.sh >/dev/null 2>&1)
  sleep 2
fi
log_run "$@"

echo "On the Pico: open XRoboToolkit, connect to this laptop's Wi-Fi address"
echo "  ($(find_wifi_ip 2>/dev/null || echo 'see ip addr')), then in the Pico Motion Tracker tab"
echo "  choose object tracking and press Send. Strap one tracker on each wrist."
echo "The first tracker frame calibrates. To recalibrate, hold both forearms forward,"
echo "  palms down, look straight ahead, and press R; both arms return to their home pose."

python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_bimanual_sim
  --arm-input tracker --report /tmp/vega-bimanual-tracker-live.json
)
[[ -n "${1:-}" ]] && python_cmd+=(--left-glove-sn "$1")
[[ -n "${2:-}" ]] && python_cmd+=(--right-glove-sn "$2")
[[ -n "${LEFT_TRACKER_SN:-}" ]] && python_cmd+=(--left-tracker-sn "$LEFT_TRACKER_SN")
[[ -n "${RIGHT_TRACKER_SN:-}" ]] && python_cmd+=(--right-tracker-sn "$RIGHT_TRACKER_SN")
[[ -n "${HUMAN_HEIGHT:-}" ]] && python_cmd+=(--human-height "$HUMAN_HEIGHT")
[[ -n "${IK_CONFIG:-}" ]] && python_cmd+=(--ik-config "$IK_CONFIG")
[[ -n "${FINGER_INPUT:-}" ]] && python_cmd+=(--finger-input "$FINGER_INPUT")
log_cmd python "${python_cmd[@]}"
"${python_cmd[@]}"
