#!/usr/bin/env bash
# Record a human demonstration: both Wuji gloves (skeleton, fingertip sensors,
# palm IMU) and the PICO wrist trackers/headset from the XRoboToolkit PC Service.
# The headset itself records the stereo passthrough video (XRoboToolkit Data
# Collection: Tracking + Vision), or, with --stereo HEADSET_IP, the laptop
# records the live stereo stream with Fast-FoundationStereo disparity of every
# frame. Never connects to the robot or the robot hands.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--out SESSION_DIR] [--hands both|left|right|none] [--user-id NAME] [--skip-calibration]"
  echo "          [--stereo HEADSET_IP [--stereo-checkpoint 23-36-37] [--no-stereo-window]]"
  echo "Default session directory: ~/vega-demos/<time>. Ctrl-C stops (stop the headset Record first)."
  echo "Then: scripts/process_human_demo.sh pull SESSION_DIR, and the stages in HUMAN_DEMO_CAPTURE.md;"
  echo "with --stereo: scripts/process_human_demo.sh sync SESSION_DIR."
  exec "$workspace/.venv/bin/python" -m wuji_teleop.demo.record --help
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
if ! "$workspace/.venv/bin/python" -c 'import xrobotoolkit_sdk, wuji_sdk' 2>/dev/null; then
  echo "Recording needs xrobotoolkit_sdk and wuji_sdk in .venv (scripts/install_xrobotoolkit_sdk.sh)." >&2
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
echo "Headset: open XRoboToolkit, connect to $(find_wifi_ip 2>/dev/null || echo 'this laptop'),"
echo "  turn on Head tracking, Pico Motion Tracker tab -> object tracking -> Send,"
if [[ " $* " == *" --stereo "* ]]; then
  echo "  and keep the app in front with the headset awake; do not press Data Collection Record."
  echo "  Close any other stereo viewer first: the app serves one camera client at a time."
else
  echo "  then Data Collection: Tracking + Vision -> Record (needs the enterprise camera permission)."
fi
python_cmd=("$workspace/.venv/bin/python" -m wuji_teleop.demo.record "$@")
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
