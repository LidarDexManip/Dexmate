# Sourced by scripts/launch_*.sh. Requires $workspace to be set.
# Auto-detects DexStream, the laptop Wi-Fi address and the glove serial.

# shellcheck source=scripts/_log_run.sh
source "$workspace/scripts/_log_run.sh"

require_venv() {
  if [[ ! -x "$workspace/.venv/bin/python" ]]; then
    echo "Python environment missing: ${workspace}/.venv. Run scripts/setup_sim.sh first." >&2
    exit 1
  fi
}

# Prints the DexStream executable: $DEXSTREAM_BIN, the project .venv, PATH, then conda envs.
find_dexstream() {
  local candidate
  for candidate in "${DEXSTREAM_BIN:-}" "$workspace/.venv/bin/dexstream" \
      "$(command -v dexstream 2>/dev/null || true)" \
      "$HOME"/miniconda3/envs/*/bin/dexstream "$HOME"/anaconda3/envs/*/bin/dexstream; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "DexStream not found. Run scripts/setup_sim.sh or set DEXSTREAM_BIN." >&2
  return 1
}

# Prints the laptop Wi-Fi IPv4 address: $WIFI_INTERFACE if set, else the first
# wireless interface with an address, else the interface holding the default route.
find_wifi_ip() {
  local iface ip
  local candidates=()
  if [[ -n "${WIFI_INTERFACE:-}" ]]; then
    candidates=("$WIFI_INTERFACE")
  else
    for iface in /sys/class/net/*; do
      [[ -d "$iface/wireless" ]] && candidates+=("$(basename "$iface")")
    done
    candidates+=("$(ip -4 route show default | awk 'NR == 1 { print $5 }')")
  fi
  for iface in "${candidates[@]}"; do
    [[ -n "$iface" ]] || continue
    ip="$(ip -4 -o address show dev "$iface" 2>/dev/null | awk 'NR == 1 { split($4, a, "/"); print a[1] }')"
    if [[ -n "$ip" ]]; then
      printf '%s\n' "$ip"
      return 0
    fi
  done
  echo "No Wi-Fi IPv4 address found. Join the Pico's Wi-Fi, or pass the IP / set WIFI_INTERFACE." >&2
  return 1
}

# Sets the glove_args array: --glove-sn when a serial is given (argument or
# $GLOVE_SN), otherwise empty so the sim connects to the only glove it finds.
set_glove_args() {
  local serial="${1:-${GLOVE_SN:-}}"
  glove_args=()
  if [[ -n "$serial" ]]; then
    glove_args=(--glove-sn "$serial")
  fi
}
