#!/usr/bin/env bash
# Pico left/right hand tracking -> both simulated Vega arms; matching Wuji
# gloves -> both wrists and both sets of fingers. Never starts robot control.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [WIFI_IP] [LEFT_GLOVE_SERIAL] [RIGHT_GLOVE_SERIAL]"
  echo "Defaults: detected Wi-Fi address; left glove .100 and right glove .101."
  echo "Env overrides: WIFI_INTERFACE, DEXSTREAM_BIN, PICO_SCALE."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
wifi_ip="${1:-$(find_wifi_ip)}"
dexstream_bin="$(find_dexstream)"
log_run "$@"

echo "Starting bilateral simulation and Pico page https://${wifi_ip}:8091/"
echo "On the Pico: accept the certificate, put both controllers down, enter AR, and keep both hands visible."
echo "Hold both gloves still for one second, then press R in MuJoCo to anchor both hands."
echo "Pico positions drive both shoulders/elbows; glove IMUs drive wrists; glove skeletons drive fingers."

if ss -ltn 2>/dev/null | grep -qE ':8091([[:space:]]|$)'; then
  echo "DexStream already running on port 8091; starting only the bilateral sim."
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

python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_bimanual_sim
  --report /tmp/vega-bimanual-pico-live.json
)
[[ -n "${2:-}" ]] && python_cmd+=(--left-glove-sn "$2")
[[ -n "${3:-}" ]] && python_cmd+=(--right-glove-sn "$3")
[[ -n "${PICO_SCALE:-}" ]] && python_cmd+=(--pico-scale "$PICO_SCALE")
log_cmd python "${python_cmd[@]}"
"${python_cmd[@]}"
