#!/usr/bin/env bash
# Wuji glove -> simulated Hand 2 fingers on Vega. The arm stays fixed.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [LEFT_GLOVE_SERIAL]"
  echo "Default: the only glove found on the network (or \$GLOVE_SN)."
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
set_glove_args "${1:-}"
log_run "$@"

echo "Starting glove fingers on the simulated Wuji Hand 2. Space pauses finger tracking."
python_cmd=(
  "$workspace/.venv/bin/python" -m wuji_teleop.vega_hand_sim "${glove_args[@]}"
  --report /tmp/vega-hand-live.json
)
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
