#!/usr/bin/env bash
# DexStream only: serves the Pico page on https://<wifi-ip>:8091/ until Ctrl-C.
# Leave it running in its own terminal so the sim can be restarted freely.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [hand|paddle] [WIFI_IP]"
  echo "hand (default): Pico hand tracking, for launch_wuji_vega_pico_hand.sh."
  echo "paddle: Pico controllers, for launch_wuji_vega_pico_wrist.sh."
  exit 0
fi

input_mode="${1:-hand}"
if [[ "$input_mode" != hand && "$input_mode" != paddle ]]; then
  echo "Input mode must be 'hand' or 'paddle', got: ${input_mode}" >&2
  exit 1
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
wifi_ip="${2:-$(find_wifi_ip)}"
dexstream_bin="$(find_dexstream)"
log_run "$@"

extra=()
[[ "$input_mode" == hand ]] && extra=(--disable-world-frame-snap)
echo "Pico page: https://${wifi_ip}:8091/  (Ctrl-C to stop)"
dexstream_cmd=(env -u ZENOH_CONFIG -u ROBOT_NAME "$dexstream_bin"
  --host "$wifi_ip" --port 8090 --mode wireless --input "$input_mode" "${extra[@]}" --no-shm)
log_cmd dexstream "${dexstream_cmd[@]}"
exec "${dexstream_cmd[@]}"
