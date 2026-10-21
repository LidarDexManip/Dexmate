#!/usr/bin/env bash
# Build the XRoboToolkit Python bindings (xrobotoolkit_sdk) into the project .venv
# for PICO Motion Tracker input. Needs the XRoboToolkit PC Service .deb installed
# (it provides /opt/apps/roboticsservice/SDK). Safe to rerun. Never contacts a robot.
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: $0"
  echo "Env overrides: XROBOTOOLKIT_SDK_DIR (default /opt/apps/roboticsservice/SDK)."
  exit 0
fi
cd "$workspace"
# shellcheck source=scripts/_common.sh
source "$workspace/scripts/_common.sh"
require_venv
python="$workspace/.venv/bin/python"

sdk_dir="${XROBOTOOLKIT_SDK_DIR:-/opt/apps/roboticsservice/SDK}"
if [[ ! -f "$sdk_dir/include/PXREARobotSDK.h" || ! -f "$sdk_dir/x64/libPXREARobotSDK.so" ]]; then
  echo "XRoboToolkit PC Service SDK not found in ${sdk_dir}." >&2
  echo "Install the PC Service .deb (XRoboToolkit_PC_Service_*_amd64.deb) first." >&2
  exit 1
fi
if "$python" -c 'import xrobotoolkit_sdk' 2>/dev/null; then
  echo "xrobotoolkit_sdk already importable in .venv."
  exit 0
fi
command -v g++ >/dev/null || { echo "Missing g++: sudo apt install g++" >&2; exit 1; }

# The vendored bindings read up to four trackers (upstream stops at three).
bindings_dir="wuji-hand-teleop/src/input_devices/pico_input/vendor/XRoboToolkit-PC-Service-Pybind"
if [[ ! -f "$bindings_dir/bindings/py_bindings.cpp" ]]; then
  git submodule update --init wuji-hand-teleop
fi

build_dir="$workspace/.venv/build/xrobotoolkit"
json_header="$build_dir/include/nlohmann/json.hpp"
json_sha256="9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6"
mkdir -p "$(dirname "$json_header")"
if ! echo "$json_sha256  $json_header" | sha256sum --check --status 2>/dev/null; then
  curl -fsSL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o "$json_header"
  echo "$json_sha256  $json_header" | sha256sum --check --quiet
fi

"$python" -m pip install pybind11==2.13.6
site_packages="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
suffix="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
# shellcheck disable=SC2046  # pybind11 --includes prints several -I flags
g++ -O2 -std=c++17 -shared -fPIC $("$python" -m pybind11 --includes) \
  -I"$build_dir/include" -I"$sdk_dir/include" \
  "$bindings_dir/bindings/py_bindings.cpp" \
  -L"$sdk_dir/x64" -lPXREARobotSDK -Wl,-rpath,"$sdk_dir/x64" \
  -o "$site_packages/xrobotoolkit_sdk$suffix"
"$python" -c 'import xrobotoolkit_sdk; print("xrobotoolkit_sdk ok")'
