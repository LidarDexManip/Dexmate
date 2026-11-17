#!/usr/bin/env bash
# Offline processing of a recorded human demonstration, in the retargeting
# checkout's environment (viser, libigl, its held-segment refinement).
#   scripts/process_human_demo.sh STAGE SESSION_DIR [options]
# Stages: pull inspect hands depth track ground export view all
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
retargeting="${RETARGETING_DIR:-$HOME/Desktop/retargeting}"
python="$retargeting/.venv-vision/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Missing $python. In $retargeting run: uv venv --python /usr/bin/python3.12 .venv-vision && ./setup_retarget.sh" >&2
  exit 1
fi
cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
log_run "$@"
export RETARGETING_DIR="$retargeting"
python_cmd=("$python" -m wuji_teleop.demo.process "$@")
log_cmd python "${python_cmd[@]}"
exec "${python_cmd[@]}"
