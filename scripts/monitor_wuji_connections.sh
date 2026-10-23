#!/usr/bin/env bash
# Live check of every teleop link (network, robot, gloves, hands, Pico) with a
# fix for each missing one. Read-only; --connect also opens real connections
# (run that before the teleop, not alongside it).
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--once] [--connect] [--no-pico] [--interval SECONDS]"
  exit 0
fi

cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
exec "$workspace/.venv/bin/python" -m wuji_teleop.connection_monitor "$@"
