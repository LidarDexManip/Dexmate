# Sourced by scripts/launch_wuji_vega_*.sh after cd "$workspace".
# Safe under set -euo pipefail: logging never aborts the launcher.

log_run() {
  local line
  line="$(date -Iseconds) $(basename "$0")"
  if (("$#" > 0)); then
    line+=" $*"
  fi
  printf '%s\n' "$line" >>"$workspace/SIM_RUN_LOG.txt" 2>/dev/null || true
}

# Usage: log_cmd "<label>" <command words...>
log_cmd() {
  local label="$1"
  shift
  printf '  %s %s\n' "$label" "$*" >>"$workspace/SIM_RUN_LOG.txt" 2>/dev/null || true
}
