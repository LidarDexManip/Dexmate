#!/usr/bin/env bash
# Simulation default. --hardware explicitly connects the physical arms and hands.
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
source scripts/_common.sh
require_venv
export ROBOT_NAME="${ROBOT_NAME:-dm/vgce08fed0a4-1u}"
export ZENOH_CONFIG="${ZENOH_CONFIG:-$workspace/dm_vgce08fed0a4-1u.dzcfg}"
exec "$workspace/.venv/bin/python" -m wuji_teleop.ee_web "$@"
