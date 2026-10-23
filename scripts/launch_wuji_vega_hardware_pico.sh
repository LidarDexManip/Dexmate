#!/usr/bin/env bash
# PICO wrist trackers (default) or Pico hand tracking + both Wuji gloves -> the
# PHYSICAL Vega-1U arms and both Wuji hands, with a Reset / Start / Pause /
# E-stop dashboard. MOVES HARDWARE once Reset or Start is pressed; it holds the
# current pose until then.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [WIFI_IP] [hardware_teleop options, e.g. --fake-robot]"
  echo "Defaults: ARM_INPUT=tracker; ZENOH_CONFIG=${workspace}/dm_vgce08fed0a4-1u.dzcfg."
  echo "Env overrides: ARM_INPUT (tracker | pico-hand), ROBOT_NAME, ZENOH_CONFIG,"
  echo "  LEFT_GLOVE_SN, RIGHT_GLOVE_SN, LEFT_HAND_SN, RIGHT_HAND_SN,"
  echo "  LEFT_TRACKER_SN, RIGHT_TRACKER_SN, IK_CONFIG, HUMAN_HEIGHT,"
  echo "  PICO_SCALE, RESET_SHOULDER_BACK, WIFI_INTERFACE, DEXSTREAM_BIN."
  echo "Rehearse with the real inputs but no robot: $0 --fake-robot"
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
if ! "$workspace/.venv/bin/python" -c "import dexcontrol" 2>/dev/null; then
  echo "dexcontrol missing. Run: .venv/bin/python -m pip install -r requirements-wuji-hardware.txt" >&2
  exit 1
fi
wifi_ip=""
if [[ -n "${1:-}" && "${1:-}" != -* ]]; then
  wifi_ip="$1"
  shift
fi
: "${ARM_INPUT:=tracker}"
: "${ROBOT_NAME:=dm/vgce08fed0a4-1u}"
: "${ZENOH_CONFIG:=$workspace/dm_vgce08fed0a4-1u.dzcfg}"
if [[ ! -f "$ZENOH_CONFIG" ]]; then
  echo "Robot communication config not found: ${ZENOH_CONFIG}" >&2
  exit 1
fi
export ROBOT_NAME ZENOH_CONFIG
log_run "$@"
echo "Dashboard: R reset (collision-checked), S start, P/Space pause, Esc E-stop, U release."

if [[ "$ARM_INPUT" == "tracker" ]]; then
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
  echo "On the headset: XRoboToolkit, Motion Tracker tab, Send. Wear one tracker per wrist."
  echo "Start (S) calibrates: hold your forearms like the robot's (after Reset: forward, palms down)."
elif [[ "$ARM_INPUT" == "pico-hand" ]]; then
  wifi_ip="${wifi_ip:-$(find_wifi_ip)}"
  dexstream_bin="$(find_dexstream)"
  echo "Pico page https://${wifi_ip}:8091/: accept the certificate, controllers down, Enter AR."
  if ss -ltn 2>/dev/null | grep -qE ':8091([[:space:]]|$)'; then
    echo "DexStream already running on port 8091; starting only the teleop."
  else
    dexstream_cmd=(env -u ZENOH_CONFIG -u ROBOT_NAME "$dexstream_bin"
      --host "$wifi_ip" --port 8090 --mode wireless --input hand
      --disable-world-frame-snap --no-shm)
    log_cmd dexstream "${dexstream_cmd[@]}"
    "${dexstream_cmd[@]}" &
    dexstream_pid=$!

    cleanup() {
      kill -TERM "$dexstream_pid" 2>/dev/null || true
      wait "$dexstream_pid" 2>/dev/null || true
    }
    trap cleanup EXIT

    sleep 2
    if ! kill -0 "$dexstream_pid" 2>/dev/null; then
      echo "DexStream stopped during startup; inspect its output above." >&2
      wait "$dexstream_pid"
      exit 1
    fi
  fi
else
  echo "ARM_INPUT must be tracker or pico-hand, not ${ARM_INPUT}" >&2
  exit 2
fi

python_cmd=("$workspace/.venv/bin/python" -m wuji_teleop.hardware_teleop --arm-input "$ARM_INPUT")
[[ -n "${LEFT_GLOVE_SN:-}" ]] && python_cmd+=(--left-glove-sn "$LEFT_GLOVE_SN")
[[ -n "${RIGHT_GLOVE_SN:-}" ]] && python_cmd+=(--right-glove-sn "$RIGHT_GLOVE_SN")
[[ -n "${LEFT_HAND_SN:-}" ]] && python_cmd+=(--left-hand-sn "$LEFT_HAND_SN")
[[ -n "${RIGHT_HAND_SN:-}" ]] && python_cmd+=(--right-hand-sn "$RIGHT_HAND_SN")
[[ -n "${LEFT_TRACKER_SN:-}" ]] && python_cmd+=(--left-tracker-sn "$LEFT_TRACKER_SN")
[[ -n "${RIGHT_TRACKER_SN:-}" ]] && python_cmd+=(--right-tracker-sn "$RIGHT_TRACKER_SN")
[[ -n "${IK_CONFIG:-}" ]] && python_cmd+=(--ik-config "$IK_CONFIG")
[[ -n "${HUMAN_HEIGHT:-}" ]] && python_cmd+=(--human-height "$HUMAN_HEIGHT")
[[ -n "${PICO_SCALE:-}" ]] && python_cmd+=(--pico-scale "$PICO_SCALE")
[[ -n "${RESET_SHOULDER_BACK:-}" ]] && python_cmd+=(--reset-shoulder-back "$RESET_SHOULDER_BACK")
python_cmd+=("$@")
log_cmd python "${python_cmd[@]}"
"${python_cmd[@]}"
