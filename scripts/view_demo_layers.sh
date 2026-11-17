#!/usr/bin/env bash
# Camera-view inspector of a synced live-stereo session: left image and depth side
# by side, with trackers, wrist, hand orientation, finger skeleton and the fitted
# hand model switched on and off (wuji_teleop/demo/layer_view.py).
#   scripts/view_demo_layers.sh SESSION_DIR [--model mano] [--start FRAME] [--near 0.2 --far 1.0]
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
retargeting="${RETARGETING_DIR:-$HOME/Desktop/retargeting}"
python="$retargeting/.venv-vision/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing $python (see HUMAN_DEMO_CAPTURE.md)." >&2
  exit 1
fi
cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
log_run "$@"
exec "$python" -m wuji_teleop.demo.layer_view "$@"
