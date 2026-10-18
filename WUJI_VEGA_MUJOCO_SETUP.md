# Wuji glove + Wuji Hand 2 + Dexmate Vega 1U in MuJoCo

This guide sets up the **left Wuji glove** to drive a simulated Wuji Hand 2 on
Dexmate Vega 1U. It covers a fixed-arm hand check, a wrist-only scene with the
shoulder and elbow fixed, and an optional articulated seven-joint left arm.
All commands run on the laptop; these launchers do not send commands to a
physical Vega or Wuji Hand.

Run commands from the workspace root:

```bash
cd /home/isaac/Documents/Dexmate
```

The working integration lives in `wuji_teleop/`. A fresh clone includes it
when the repository is cloned with submodules.

## What you need

- Linux desktop with a MuJoCo-capable graphics display for the viewer, Python
  3.11 (the tested version), `git`, `python3-venv`, NetworkManager (`nmcli`), and `rg`
  (ripgrep). A headless check does not need an open viewer.
- The `dexmate-urdf/` submodule containing Vega 1U's URDF. The conversion
  script lives in this repository at `scripts/launch_vega_1u_mujoco.py`.
- The pinned `third_party/wuji-description` submodule for Hand 2's MJCF and
  meshes. Keep the XML and mesh directories together.
- For live tracking: the left Wuji glove, its acquisition board, supplied
  USB-C cable, powered USB-C/Ethernet adapter, and an Ethernet connection to
  the laptop. Synthetic tests need none of this hardware.

The important workspace paths are:

```text
wuji_teleop/vega_hand_sim.py                         fixed Vega arm + moving hand
wuji_teleop/vega_arm_sim.py                          moving left arm + hand
scripts/launch_vega_1u_mujoco.py                    Vega URDF converter
dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf   Vega model
third_party/wuji-description/hand2/hand2_beta2/body/mjcf/left.xml
```

## 1. Install Python and model dependencies

On this Ubuntu laptop, install missing system tools if needed:

```bash
sudo apt install git python3-venv network-manager ripgrep
```

From the workspace root, create the virtual environment if `.venv` does not
already exist, then install the pinned glove/simulation packages:

`dexmate-urdf/` and `third_party/wuji-description/` are submodules.
`dexmate-urdf` is pinned to commit `d8fc9c4`.

```bash
test -d .venv || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
git submodule update --init --recursive
.venv/bin/python -m pip install -r requirements-wuji-sim.txt
.venv/bin/python -m pip check
test -f wuji_teleop/vega_arm_sim.py
test -f scripts/launch_vega_1u_mujoco.py
test -f third_party/wuji-description/hand2/hand2_beta2/body/mjcf/left.xml
test -f dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf
```

`requirements-wuji-sim.txt` pins `wuji-sdk==2026.9.22`, `mujoco==3.15.0`,
`numpy==2.4.6`, `trimesh==5.1.1`, and `urdf2mjcf==0.2.39`. The current
integration uses the SDK's `RetargetSession`; do not install the older
`wuji-retargeting` project for these launchers.

A newer `dexmate-urdf` checkout may need revalidation if its URDF changes.

## 2. Validate the models without the glove

Convert Vega and the assembled fixed-arm hand scene:

```bash
.venv/bin/python scripts/launch_vega_1u_mujoco.py --convert-only
.venv/bin/python -m wuji_teleop.vega_hand_sim --convert-only
```

Open the fixed-arm scene with synthetic finger curls:

```bash
.venv/bin/python -m wuji_teleop.vega_hand_sim --input synthetic --duration 10
```

Open the articulated scene with synthetic wrist and finger motion:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input synthetic --finger-input synthetic --duration 10
```

The fixed-arm scene welds Vega in place and moves only the 20 left hand joints.
The articulated scene fixes Vega's base, head, lift, torso, and right arm,
while seven left arm joints and 20 left hand joints move. Its neutral left arm
hangs with the elbow down beside the body and the hand out. Converted URDF metadata/inertia warnings
may print during startup; the run should report **zero MuJoCo numerical
warnings**.

## 3. Connect and discover the left glove

Connect and power the glove's Ethernet path. This laptop uses `enp3s0`; use the
actual wired interface name on another computer. Check that the interface has
carrier before configuring an IP address:

```bash
ip -brief address
cat /sys/class/net/enp3s0/carrier
```

`1` means link is present. If the interface has no IPv4 address and there is
no existing route to `192.168.1.0/24`, use the guarded temporary NetworkManager
setup. It adds `192.168.1.50/24` without changing the default internet route:

```bash
bash scripts/configure_wuji_network.sh enp3s0
```

If the interface already has an address, inspect it and its route instead of
running the setup script again:

```bash
ip -4 address show dev enp3s0
ip route get 192.168.1.100
ping -c 2 -W 1 -I enp3s0 192.168.1.100
.venv/bin/python -m wuji_teleop discover --seconds 3
```

The current left glove is `WG1JA06260812032` at `192.168.1.100:50001`. Copy
the serial returned by discovery on a different rig. Discovery should show
fresh skeleton frames and exit successfully. The right glove is not required
for this left-arm setup.

When finished, remove the temporary network profile if the setup script
created it:

```bash
nmcli connection delete wuji-glove-sim-enp3s0
```

## 4. Run the live glove in MuJoCo

First check the hand with Vega held still. Move one finger at a time and check
that the matching simulated finger moves smoothly:

```bash
.venv/bin/python -m wuji_teleop.vega_hand_sim \
  --glove-sn WG1JA06260812032 \
  --report /tmp/vega-hand-live.json
```

Then choose one arm input mode. Each command opens one MuJoCo viewer and uses
the same glove connection for finger tracking. Stop the current viewer before
launching another mode.

### Glove IMU for wrist rotation only

This is the current **hand and wrist** launcher. The glove's 20 finger targets
move the simulated hand; its calibrated palm IMU rotates Vega's three wrist
joints. The shoulder and elbow joints are removed from this scene, so glove
acceleration cannot move the whole arm. Hold the glove still for one second
during orientation calibration:

```bash
./scripts/launch_wuji_vega_glove_wrist.sh
```

You can pass a different discovered left glove serial as the first argument.
Press **R** while still to recalibrate and anchor the current wrist angle.
Pico is not needed for this glove-only wrist mode.

### Whole-arm IK from the glove

This launcher keeps the glove on the fingers and solves Vega's shoulder,
elbow, and wrist together. The glove IMU does not measure position, so arm
translation comes from how far the fingers tip away from the pose held during
calibration. That swing is scaled into a wrist-position target, up to 40 cm,
and damped least squares moves all seven joints to reach it. Twisting about
the fingers aims the hand and does not translate the arm. Hold still for one
second at startup. Press **R** while still to anchor again.

```bash
./scripts/launch_wuji_vega_glove_arm.sh
```

Tip the hand to move the arm, or draw it inward while keeping the fingers
aimed the same way. Closer wrist targets fold the elbow; a straight elbow
cannot reach those positions. Small tilts inside the deadband hold the arm
still.

### Larger glove-only arm movement: directional strokes

This is the current glove-only **control** mode. Hold the glove still for its
one-second IMU calibration. Keep the palm roughly level; make a short, quick
hand movement up, down, left, right, forward, or back, then pause. Each detected
stroke commands a smooth 6 cm simulated wrist step. Repeated strokes can move
the robot arm up to 30 cm from the current anchor. The viewer shows stroke
count, commanded distance, and how many of the seven arm joints have moved by
more than 3° since launch:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input glove-gesture --finger-input glove \
  --glove-sn WG1JA06260812032 \
  --report /tmp/vega-arm-gesture-live.json
```

Press **R** while holding still to anchor another range at the arm's current
simulated pose. These are directional commands, not measured hand positions;
the glove does not sense your shoulder or elbow. The arm solver chooses a
bent-elbow posture while moving its seven joints to follow the wrist command.

### Short relative IMU estimate

This experimental mode integrates acceleration and limits the estimated
displacement to 10 cm. It can drift and will not accurately follow sustained
hand translation:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input glove-imu --finger-input glove \
  --glove-sn WG1JA06260812032
```

The glove's fused IMU supplies wrist orientation. The program reads the SDK's
fixed IMU-to-wrist transform so the sensor axes map to the simulated hand.

### Pico hand tracking for the arm

Pico hand position drives shoulder and elbow (j1–j4); the Wuji glove drives wrist
(j5–j7) and fingers. DexStream must run with `--input hand --disable-world-frame-snap`.
Start everything with `./scripts/launch_wuji_vega_pico_hand.sh` (translation
scale defaults to `0.6`; optional `PICO_SCALE` env for `--pico-scale`). On the
Pico, open `https://<WIFI_IP>:8091/`,
accept the certificate, enter VR **without controllers** (hand tracking on). Hold
the gloved left hand in view, stay still, then press **R** in the sim to recenter.
If hand tracking is lost, the arm holds its last pose. The sim prints an XYZ log CSV
(default `/tmp/vega-pico-hand-<time>.csv`) with columns for target, achieved EE,
raw and filtered Pico position, workspace clipping, and `j1`…`j7`. Wearing the
glove can reduce Pico hand-tracking quality; keep the hand visible to the headset.

### Bilateral Pico hand tracking

For two gloves and both simulated Vega arms in one MuJoCo world, use:

```bash
./scripts/launch_wuji_vega_bimanual_pico.sh
```

Pico left/right wrist positions independently drive joints 1-4 on the matching
arms. The matching glove IMUs drive joints 5-7 and their skeletons drive the 20
finger joints per side. Put both Pico controllers down, enter AR with both hands
visible, hold both gloves still for one second, then press **R** to anchor both
sides. The launcher requires exactly one discovered left glove and one right
glove unless their serials are supplied explicitly.

### Pico controller for wrist only

The Pico wrist-only scene welds Vega's shoulder and elbow at their zero pose. Only
the three distal wrist joints and 20 hand joints move. Pico controller
**rotation** drives the wrist; Pico translation is ignored. The glove still
drives the fingers. Install the optional Zenoh subscriber:

```bash
.venv/bin/python -m pip install -r requirements-wuji-vr-sim.txt
./scripts/launch_wuji_vega_pico_wrist.sh
```

The launcher detects the laptop's current `wlp4s0` Wi-Fi address, starts the
installed DexStream service and wrist-only MuJoCo viewer, and prints the Pico
URL (`https://<WIFI_IP>:8091/`). Open that URL in the Pico browser, accept its
local certificate, enter the VR session, then hold the **left index trigger**
to clutch. Rotate the controller to rotate the simulated wrist. Releasing the
trigger holds the wrist. Use the main page, not `/preview`.

If auto-detection chooses the wrong address, pass the Wi-Fi IP and glove serial:

```bash
./scripts/launch_wuji_vega_pico_wrist.sh 10.90.198.129 WG1JA06260812032
```

The default DexStream executable is
`/home/isaac/miniconda3/envs/code_test/bin/dexstream`; set `DEXSTREAM_BIN` to
another installed executable if needed. This starts no real-robot follower or
leader process. The Pico path still needs a wearer check with a live headset.
The older `--arm-input pico-controller` mode remains available for a separate
seven-joint whole-arm experiment; use `pico-wrist` for wrist-only control.

## Controls, reports, and checks

- **Space:** pause/resume finger tracking while holding the current finger
  targets. In the moving-arm viewer it does not pause the arm input.
- **R:** recenter/recalibrate either glove IMU arm mode while holding still.
- **Left Pico index trigger:** hold to clutch the controller to the simulated
  wrist; release to hold the current wrist angle. After a stale controller
  stream, release and press again.
- **Close the viewer or Ctrl-C:** stop; the glove connection is closed.
- **`--duration SECONDS`:** stop after a finite run. **`--headless`** requires
  `--duration`. **`--report PATH`** writes final JSON metrics.

For a finite live check without a window:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input glove-gesture --finger-input glove \
  --glove-sn WG1JA06260812032 \
  --headless --duration 10 --report /tmp/vega-arm-check.json
```

Useful report fields are `finger_status`, `arm_status`, `finger_hz`, `arm_hz`,
`arm_gesture_steps`, `arm_joint_peak_delta_rad`, and `mujoco_warnings`. The
current setup has passed a 44-test suite and a live five-second gesture-mode
headless check with zero MuJoCo numerical warnings. A live viewer has detected
strokes and moved all seven arm joints; how natural the larger commands feel
still requires wearer feedback.

```bash
WUJI_GUI_TEST=1 .venv/bin/python -m unittest discover -s tests -v
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No Ethernet carrier` | Check power, adapter link lights, and the labeled USB-C cable orientation. |
| `Destination Host Unreachable` | Confirm `enp3s0` has a link, the `192.168.1.50/24` address, and a route to `192.168.1.100`. |
| No glove found or stale frames | Run `discover`; verify the glove serial and wired route, then restart after reconnecting the glove. |
| `urdf2mjcf is not installed` | Install `.venv/bin/python -m pip install -r requirements-wuji-sim.txt`. |
| Hand or Vega mesh missing | Initialize `third_party/wuji-description` and keep its mesh folders with the XML; check `dexmate-urdf/` too. |
| IMU arm stays `calibrating` | Hold the glove still for a full second; press **R** and retry. |
| Gesture does not move the arm | Use a short, straight movement with a pause; check that `strokes` increases in the viewer. |
| Arm reaches a limit | Move back in the opposite direction or press **R** while still to recenter. The IK respects Vega joint limits. |
| No viewer window | Use a desktop graphics session or a finite `--headless --duration` check. |

For detailed glove calibration and standalone Hand 2 use, see
[WUJI_TELEOP_RUNBOOK.md](WUJI_TELEOP_RUNBOOK.md). The older sections below the
current-path notes in [WUJI_HAND_MUJOCO_RUNBOOK.md](WUJI_HAND_MUJOCO_RUNBOOK.md)
are historical and use a separate retargeting project.
