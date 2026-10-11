#!/usr/bin/env bash
# Pico hand tracking -> simulated Vega arm (j1-j4); Wuji glove -> wrist (j5-j7) + fingers.
# No physical robot command process is started.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [WIFI_IP] [LEFT_GLOVE_SERIAL]"
  echo "Defaults: detected Wi-Fi address; the only glove found on the network."
  echo "Env overrides: WIFI_INTERFACE, DEXSTREAM_BIN, GLOVE_SN, PICO_SCALE (sim --pico-scale)."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
wifi_ip="${1:-$(find_wifi_ip)}"
dexstream_bin="$(find_dexstream)"
set_glove_args "${2:-}"
log_run "$@"

echo "Starting DexStream on ${wifi_ip}:8090 and Pico page https://${wifi_ip}:8091/"
echo "On the Pico, open the page, accept its certificate, and enter the VR session with NO controllers (put them down; hand tracking on)."
echo "Hold the gloved left hand in view, keep still, then press R in the sim window to recenter (maps current hand position to the robot hand)."
echo "Pico hand position moves the arm (shoulder/elbow); glove orientation drives the wrist; glove fingers drive the hand. Arm holds if tracking is lost."
echo "XYZ log CSV path is printed by the sim (default /tmp/vega-pico-hand-<time>.csv)."
if ss -ltn 2>/dev/null | grep -qE ':8091([[:space:]]|$)'; then
  # scripts/launch_dexstream.sh is already serving the Pico page; leave it running.
  echo "DexStream already running on port 8091; starting only the sim."
else
  dexstream_cmd=(env -u ZENOH_CONFIG -u ROBOT_NAME "$dexstream_bin"
    --host "$wifi_ip" --port 8090 --mode wireless --input hand --disable-world-frame-snap --no-shm)
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

python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim
  --arm-input pico-hand --finger-input glove "${glove_args[@]}"
  --report /tmp/vega-pico-hand-live.json
)
if [[ -n "${PICO_SCALE:-}" ]]; then
  python_cmd+=(--pico-scale "$PICO_SCALE")
fi
log_cmd python "${python_cmd[@]}"
"${python_cmd[@]}"
