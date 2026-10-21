#!/usr/bin/env bash
# One-shot setup for the simulation (glove, Pico headset, MuJoCo) on a new machine.
# Installs everything into the project's .venv, including DexStream and its Pico page,
# then verifies the install. Safe to rerun. Never contacts a robot.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace"
run_tests=1
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0 [--skip-tests]"
  echo "Needs: git, python3 (3.11-3.12) with venv, curl. Uses PYTHON to pick the interpreter."
  exit 0
fi
[[ "${1:-}" == "--skip-tests" ]] && run_tests=0

step() { printf '\n== %s\n' "$*"; }

step "Checking system tools"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" && -x .venv/bin/python ]] &&
    .venv/bin/python -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12)' 2>/dev/null; then
  python_bin="$workspace/.venv/bin/python"
fi
if [[ -z "$python_bin" ]]; then
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$python_bin" ]] || ! "$python_bin" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12)' 2>/dev/null; then
  echo "Need Python 3.11-3.12 (NumPy 2.4 requires >=3.11). Install it, e.g.: sudo apt install python3.11 python3.11-venv" >&2
  exit 1
fi
for tool in git curl; do
  command -v "$tool" >/dev/null || { echo "Missing $tool: sudo apt install $tool" >&2; exit 1; }
done
echo "Python: $("$python_bin" --version) at $python_bin"

step "Fetching submodules (DexStream, Vega URDF, Wuji hand model)"
git submodule update --init --recursive dexstream dexmate-urdf third_party/wuji-description

step "Creating .venv and installing Python packages"
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv || {
    echo "venv creation failed. Install the venv package, e.g.: sudo apt install python3-venv" >&2
    exit 1
  }
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-wuji-vr-sim.txt
.venv/bin/python -m pip install -e ./dexstream

step "Installing the Pico web page dependency (three.js)"
three_version="$(sed -n 's/.*"three": *"^\{0,1\}\([0-9.]*\)".*/\1/p' dexstream/frontend/package.json | head -n 1)"
if [[ -f dexstream/frontend/node_modules/three/package.json ]]; then
  echo "three.js already present."
elif command -v npm >/dev/null 2>&1; then
  (cd dexstream/frontend && npm install --no-audit --no-fund)
else
  # No Node.js needed: unpack the same npm tarball the lockfile pins.
  mkdir -p dexstream/frontend/node_modules/three
  curl -fsSL "https://registry.npmjs.org/three/-/three-${three_version}.tgz" |
    tar -xz -C dexstream/frontend/node_modules/three --strip-components=1
  echo "Installed three.js ${three_version} without npm."
fi

step "Building PICO Motion Tracker bindings (XRoboToolkit, optional)"
if [[ -f /opt/apps/roboticsservice/SDK/include/PXREARobotSDK.h ]]; then
  scripts/install_xrobotoolkit_sdk.sh
else
  echo "XRoboToolkit PC Service not installed; skipping. Only the tracker launcher needs it."
fi

step "Verifying"
.venv/bin/python - <<'EOF'
import mink, mujoco, numpy, wuji_sdk, zenoh, dexstream  # noqa: F401
print("imports ok: mujoco", mujoco.__version__)
EOF
test -x .venv/bin/dexstream
test -f dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf
test -f third_party/wuji-description/hand2/hand2_beta2/body/mjcf/left.xml
.venv/bin/python scripts/launch_vega_1u_mujoco.py --convert-only >/dev/null
echo "scene conversion ok"
if (( run_tests )); then
  .venv/bin/python -m unittest tests.test_vega_arm_sim tests.test_vega_pico_hand \
    tests.test_vega_pico_wrist tests.test_vega_bimanual_sim \
    tests.test_imu_arm_source tests.test_vr_pose_source \
    tests.test_gmr_arm_retarget tests.test_xr_tracker_source
fi

cat <<EOF

Setup complete. Next:
  1. Glove: plug in, then  bash scripts/configure_wuji_network.sh
     and check  .venv/bin/python -m wuji_teleop discover --seconds 3
  2. No hardware:          scripts/launch_vega_joint_limits.sh
  3. Glove only:           scripts/launch_wuji_vega_hand_only.sh
  4. Glove + Pico:         scripts/launch_wuji_vega_pico_hand.sh
  5. Both gloves + Pico:   scripts/launch_wuji_vega_bimanual_pico.sh
  6. Wrist trackers:       scripts/launch_wuji_vega_bimanual_trackers.sh
See NEW_MACHINE.md for the full walkthrough.
EOF
