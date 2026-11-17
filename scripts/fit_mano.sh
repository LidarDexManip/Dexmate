#!/usr/bin/env bash
# Fit MANO hands to synced live-stereo sessions (process_human_demo.sh sync first).
#   scripts/fit_mano.sh SESSION [SESSION ...] [--mano ~/Downloads/mano_v1_2.zip]
# Runs in Fast-FoundationStereo's environment (torch, GPU). MANO's model files are
# licensed: download mano_v1_2.zip ("Models & Code") from https://mano.is.tue.mpg.de.
# Then view: scripts/process_human_demo.sh view SESSION
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${FFS_PYTHON:-$HOME/Desktop/Fast-FoundationStereo/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
  echo "Missing $python (Fast-FoundationStereo environment; see HUMAN_DEMO_CAPTURE.md)." >&2
  exit 1
fi
cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
log_run "$@"
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m wuji_teleop.demo.mano_fit "$@"
