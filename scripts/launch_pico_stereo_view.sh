#!/usr/bin/env bash
# Live stereo passthrough from the PICO headset (stock XRoboToolkit app), shown
# left | right on this laptop. Touches neither the robot nor the gloves.
#   scripts/launch_pico_stereo_view.sh [--headset IP] [options]
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--headset IP] [--width 2160 --height 810 --fps 30] [--record PREFIX] [--no-show] [--duration S]"
  echo "Asks the XRoboToolkit app on the headset (TCP 13579) to stream its passthrough"
  echo "cameras to this laptop (TCP 12345) and shows them side by side, always on top."
  echo "The headset needs PICO's enterprise camera access, switched on in the app, and"
  echo "must not be recording (Data Collection) at the same time. Without --headset the"
  echo "IP is read over adb. Keys: s save PNG (~/Pictures/pico-stereo), t on top, q quit."
  echo "--record PREFIX saves the raw H.264 and each packet's arrival time."
  exit 0
fi

retargeting="${RETARGETING_DIR:-$HOME/Desktop/retargeting}"
python="$retargeting/.venv-vision/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing $python (it provides OpenCV for the window). In $retargeting run:" >&2
  echo "  uv venv --python /usr/bin/python3.12 .venv-vision && ./setup_retarget.sh" >&2
  exit 1
fi
cd "$workspace"
# OpenCV's Qt looks for fonts inside the wheel, which ships none.
if [[ -z "${QT_QPA_FONTDIR:-}" && -d /usr/share/fonts/truetype/dejavu ]]; then
  export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu
fi
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
log_run "$@"
python_cmd=("$python" -m wuji_teleop.demo.headset_stream "$@")
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
