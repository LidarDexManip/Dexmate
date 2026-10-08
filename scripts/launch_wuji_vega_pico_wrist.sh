#!/usr/bin/env bash
# Pico left controller orientation -> simulated Vega wrist; Wuji glove -> fingers.
# No physical robot command process is started.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wifi_interface="${WIFI_INTERFACE:-wlp4s0}"
dexstream_bin="${DEXSTREAM_BIN:-/home/isaac/miniconda3/envs/code_test/bin/dexstream}"
glove_serial="${2:-WG1JA06260812032}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [WIFI_IP] [LEFT_GLOVE_SERIAL]"
  echo "Defaults: current ${wifi_interface} IPv4 address; glove ${glove_serial}"
  echo "Override executable with DEXSTREAM_BIN and Wi-Fi interface with WIFI_INTERFACE."
  exit 0
fi

wifi_ip="${1:-}"
if [[ -z "$wifi_ip" ]]; then
  wifi_ip="$(ip -4 -o address show dev "$wifi_interface" | awk 'NR == 1 { split($4, address, "/"); print address[1] }')"
fi
if [[ -z "$wifi_ip" ]]; then
  echo "No IPv4 address on ${wifi_interface}; pass the laptop Wi-Fi IP as the first argument." >&2
  exit 1
fi
if [[ ! -x "$dexstream_bin" ]]; then
  echo "DexStream executable not found: ${dexstream_bin}" >&2
  exit 1
fi
if [[ ! -x "$workspace/.venv/bin/python" ]]; then
  echo "Python environment missing: ${workspace}/.venv/bin/python" >&2
  exit 1
fi

cd "$workspace"
echo "Starting DexStream on ${wifi_ip}:8090 and Pico page https://${wifi_ip}:8091/"
echo "On the Pico, open the page, accept its certificate, and enter the VR session."
echo "Hold the LEFT index trigger and rotate the controller to move only the simulated wrist."
env -u ZENOH_CONFIG -u ROBOT_NAME "$dexstream_bin" \
  --host "$wifi_ip" --port 8090 --mode wireless --input paddle --no-shm &
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

"$workspace/.venv/bin/python" -m wuji_teleop.vega_arm_sim \
  --arm-input pico-wrist --finger-input glove \
  --glove-sn "$glove_serial" \
  --report /tmp/vega-pico-wrist-live.json
