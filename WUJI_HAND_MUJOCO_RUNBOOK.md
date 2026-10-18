# Wuji Hand to MuJoCo Runbook

For the current installation and launch steps, use
[WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md). The later sections of
this runbook retain historical experiments and older launch commands.

This runbook covers the current Wuji Glove connection, simulated Wuji Hand launch, bringing up both hands, and the next Vega 1U integration steps.

The current one-window bilateral Vega simulation is now
`scripts/launch_wuji_vega_bimanual_pico.sh`; see
[WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md#bilateral-pico-hand-tracking).
The older two-window instructions under “Deferred: Bring Up Both Hands” remain
historical notes for standalone hand models.

## Current single-arm simulation path — October 8, 2026

The newer `wuji_teleop` pipeline uses the Wuji Hand 2 model and the current
SDK retargeter. It now mounts the simulated **left** hand on Vega 1U's `L_ee`
frame. Vega joints and base are fixed; only the 20 simulated hand joints move.
The physical Wuji glove supplies finger tracking. No robot command transport
is opened. The older `wuji-retargeting/example/teleop_sim.py` instructions
below are retained as historical setup notes and are not the current launcher.

From the workspace root, validate the assembled model without a glove:

```bash
.venv/bin/python -m wuji_teleop.vega_hand_sim --convert-only
.venv/bin/python -m wuji_teleop.vega_hand_sim --input synthetic --duration 10
```

With the left glove on the existing Ethernet path:

```bash
ping -c 2 -I enp3s0 192.168.1.100
.venv/bin/python -m wuji_teleop discover --seconds 2
.venv/bin/python -m wuji_teleop.vega_hand_sim \
  --glove-sn WG1JA06260812032 \
  --report /tmp/vega-left-wuji-live.json
```

Move one finger at a time and confirm the matching simulated finger moves.
Space pauses/resumes tracking; close the viewer or press Ctrl-C to stop. A
finite headless run uses `--headless --duration 30`. The mount turns the
Hand 2 wrist 180 degrees around its local Y axis so fingers point outward
from Vega's `L_ee` frame. This placement was checked in an offscreen render,
but physical mounting dimensions and full live gesture correspondence still
need review in the viewer. The simulated arm does not move from the glove's
21 hand landmarks; arm teleoperation needs a separate arm pose input.

Verification on October 8: left glove `WG1JA06260812032` was discovered at
`192.168.1.100:50001` with fresh tracking near 120 Hz. A five-second live
mounted-hand headless run completed 600 control updates, ended in `tracking`,
and reported zero MuJoCo warnings. A four-second live viewer run also tracked
and exited cleanly. The Vega URDF converter still emits its known inertia and
metadata warnings during model generation.

During the first interactive check, the mounted hand visibly jittered. The
assembled scene had 23 unintended contacts at rest and had lost the Hand 2
default joint armature. The builder now carries over the hand's contact
exclusions and default joint dynamics and disables collision on the fixed Vega
display geometry. The corrected model starts with zero contacts and settles to
zero joint velocity at rest. All 20 tests pass, including that regression check;
a five-second live headless rerun ended in `tracking` with zero MuJoCo warnings.
The wearer confirmed the corrected live viewer looked smooth.

## Optional moving left arm in MuJoCo

For the **Pico wrist-only** setup that fixes the shoulder and elbow, use
[`scripts/launch_wuji_vega_pico_wrist.sh`](scripts/launch_wuji_vega_pico_wrist.sh)
and follow [WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md#pico-controller-for-wrist-only).
For wrist rotation from the glove IMU instead, run
[`scripts/launch_wuji_vega_glove_wrist.sh`](scripts/launch_wuji_vega_glove_wrist.sh).
The `pico-controller` instructions below are the older seven-joint arm mode.

`wuji_teleop.vega_arm_sim` keeps Vega's base, head, lift, torso, and right arm
fixed, while its seven left arm joints and the left Hand 2 are movable. The
Wuji glove supplies the 20 finger targets. Arm input can come from a Pico
controller or the glove's fused palm IMU. The IMU mode uses the same glove
connection as finger tracking, calibrates for one second while held still, and
integrates gravity-compensated acceleration for bounded short relative motion.
The seven-joint IK now starts from a bent-elbow neutral pose and favors that
posture while following wrist targets. The viewer status counts arm joints
that have moved more than 3° since launch. Wrist motion can involve the
shoulder, elbow, and wrist, but the glove does not measure the wearer's
shoulder or elbow pose.
It reads the SDK's fixed wrist-to-palm-IMU transform before applying rotation;
the IMU axes are not the glove wrist axes.
The moving-arm scene mounts the left Hand 2 palm down at its starting pose,
matching the wearer's palm-down calibration pose.
The glove has no absolute wrist-position measurement; this experimental mode
will drift over time and holds the arm target when samples are stale or invalid.
Neither input mode publishes robot commands.

First, check the combined arm IK and finger retargeting without a headset:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input synthetic --finger-input synthetic --duration 10
```

For a worn left glove and a Pico controller, install the optional subscriber
dependency and find the laptop's current Wi-Fi address:

```bash
.venv/bin/python -m pip install -r requirements-wuji-vr-sim.txt
ip -4 -brief address show wlp4s0
```

Start DexStream in one terminal, replacing `<WIFI_IP>` with that address. The
`env -u` flags prevent reuse of a real-robot Zenoh configuration:

```bash
env -u ZENOH_CONFIG -u ROBOT_NAME \
  /home/isaac/miniconda3/envs/code_test/bin/dexstream \
  --host <WIFI_IP> --port 8090 --mode wireless --input paddle --no-shm
```

On a Pico connected to the same Wi-Fi, open `https://<WIFI_IP>:8091/`, accept
the local certificate, and enter the VR session. Use the main page, not
`/preview`. In a second laptop terminal:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input pico-controller --finger-input glove \
  --glove-sn WG1JA06260812032 \
  --report /tmp/vega-left-arm-hand-live.json
```

Hold the **left index trigger** to clutch the tracked controller to the
simulated `L_ee` pose. Small controller position/orientation movements then
move the simulated left arm through limited IK. Releasing the trigger holds
the last arm target. If the controller stream is invalid or older than 250 ms,
the arm holds; release and press the trigger again after tracking returns.
Space pauses/resumes **finger** tracking only. Close the viewer or press Ctrl-C
to stop. Keep arm movements small until wrist axis directions and reach limits
have been visually checked.

Verification on October 8: the combined synthetic arm/finger run completed at
40 arm IK and 120 finger updates per second, with 500 Hz physics and no MuJoCo
warnings. A five-second run with the live left glove and synthetic arm targets
also completed with no MuJoCo warnings. A local Zenoh publisher successfully
sent test Pico-format wrist poses through the controller adapter and arm IK.
Live Pico headset motion still needs an operator test.

To try the experimental glove IMU input, hold the left glove still during its
one-second startup calibration. Its translation is drift-limited and capped at
10 cm from the startup arm pose; it is not a substitute for tracked wrist
position. The status line reports the estimated displacement and the largest
estimate since startup; these are estimator values, not measured wrist
position. Press **R** to recenter at the current simulated wrist pose and
restart the one-second still calibration. Keep movements small and use a short
run first. Translation from steady or slow hand movement will not track reliably
because the IMU does not observe wrist position:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input glove-imu --finger-input glove \
  --glove-sn WG1JA06260812032 --duration 20
```

For larger glove-only arm movements, use directional stroke control. After
the one-second still calibration, keep the palm roughly level and make a
short, quick hand movement in the desired direction, then pause. Each detected
stroke commands a 6 cm wrist step; repeated strokes can accumulate up to
30 cm from the current anchor, so the shoulder and elbow can move much farther.
The arm target moves smoothly toward each step. The viewer shows the detected
stroke count and commanded distance. Press **R** while still to anchor a new
30 cm range at the current simulated wrist position. Strokes are commands,
not measured 3D hand positions:

```bash
.venv/bin/python -m wuji_teleop.vega_arm_sim \
  --arm-input glove-gesture --finger-input glove \
  --glove-sn WG1JA06260812032
```

An October 8 live five-second headless run confirmed the glove fingers and IMU
stream work together, the arm IK updates after calibration, the resting IMU
estimate stays near zero, and MuJoCo reports no warnings. Translation during
actual wrist movement needs a wearer check in the viewer. The wearer confirmed
the corrected palm-down start, matching movement directions up/down,
left/right, and forward/back, and that the arm settles when the glove stops.
Translation accuracy remains limited by the IMU-only position estimate.

## Current Setup

Workspace:

```text
/home/isaac/Documents/Dexmate
```

Python environment:

```text
/home/isaac/Documents/Dexmate/.venv/bin/python
```

Wuji retargeting project:

```text
/home/isaac/Documents/Dexmate/wuji-retargeting
```

Wuji hand description assets:

```text
/home/isaac/Documents/Dexmate/third_party/wuji-description/hand/body/mjcf/
```

The MuJoCo hand models are:

```text
left.xml
right.xml
```

## Network Setup

The wired interface is currently `enp3s0` and the active NetworkManager connection is `Unitree G1`.

The computer needs a temporary address on the glove subnet:

```bash
sudo ip addr add 192.168.1.50/24 dev enp3s0
ip addr show dev enp3s0
```

Do not flush the existing address unless this Ethernet connection is dedicated to the gloves. Flushing can disconnect the Unitree G1 network.

Verify that the route uses the wired interface:

```bash
ip route get 192.168.1.100
ip route get 192.168.1.101
```

Expected route details include:

```text
dev enp3s0 src 192.168.1.50
```

Default glove addresses:

```text
Left glove:  192.168.1.100
Right glove: 192.168.1.101
UDP:         50001
```

Test both devices:

```bash
ping -I 192.168.1.50 192.168.1.100
ping -I 192.168.1.50 192.168.1.101
```

The left glove currently responds reliably through the TP-Link LAN path. The right glove has not been consistently reachable; keep dual-hand setup deferred until its adapter and data path are repaired.

If a ping returns `Destination Host Unreachable`, check the USB-C cable orientation. The white labels on both cable ends must face the same direction. A reversed cable can provide power without data.

## Discover Glove Serial Numbers

Use the active environment:

```bash
/home/isaac/Documents/Dexmate/.venv/bin/python - <<'PY'
from wuji_sdk import SdkManager

for device in SdkManager.instance().scan():
    print(
        f"sn={device.sn!r} "
        f"address={device.address!r} "
        f"transport={device.transport_type}"
    )
PY
```

Current discovered left glove:

```text
Serial:  WG1JA06260812032
Address: 192.168.1.100:50001
Transport: UDP
```

The right glove serial should be recorded after it is powered and reachable.

## Launch One Simulated Hand

The existing live pipeline is in:

```text
/home/isaac/Documents/Dexmate/wuji-retargeting/example/teleop_sim.py
```

Launch the current left glove into the simulated left Wuji Hand:

```bash
cd /home/isaac/Documents/Dexmate/wuji-retargeting/example

/home/isaac/Documents/Dexmate/.venv/bin/python \
  teleop_sim.py \
  --input wuji_glove \
  --hand left \
  --glove-sn WG1JA06260812032
```

This uses the Wuji Hand MuJoCo model, not the Vega robot model.

## Next: Single-Hand Validation and Robot Test

Use the working left glove and simulated left Wuji Hand as the current development path. Do not block robot integration on the second glove.

### 1. Confirm the working network path

The current working arrangement is:

```text
left glove (192.168.1.100) -> TP-Link LAN -> laptop enp3s0
laptop address: 192.168.1.50/24
```

The NetworkManager profile created for this path is `Wuji Gloves Router`. Verify the left glove before each session:

```bash
ping -c 3 -I enp3s0 192.168.1.100
```

Then discover its serial number with the SDK scan command above.

### 2. Start the single-hand simulation

```bash
cd /home/isaac/Documents/Dexmate/wuji-retargeting/example

/home/isaac/Documents/Dexmate/.venv/bin/python \
  teleop_sim.py \
  --input wuji_glove \
  --hand left \
  --glove-sn WG1JA06260812032
```

Confirm that relaxed glove movements track the simulated left Wuji Hand before proceeding.

### 3. Load the Vega 1U simulation separately

Use the existing launcher and asset path:

```text
/home/isaac/Documents/Dexmate/scripts/launch_vega_1u_mujoco.py
/home/isaac/Documents/Dexmate/dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf
```

First verify the Vega model loads and the arm remains stationary. Then attach one simulated Wuji Hand at the target wrist frame and verify the hand joint mapping with the arm and base held still.

### 4. Robot test gate

Only after the single-hand mapping is stable in MuJoCo:

1. Keep the robot stopped with emergency stops available.
2. Confirm the correct wrist/end-effector and joint limits.
3. Send low-rate, low-amplitude hand commands first.
4. Keep arm and base control disabled while testing hand motion.
5. Stop immediately if the mapping, sign convention, or range is wrong.

Record the tested glove serial, model configuration, joint mapping, and observed result before increasing the command rate or adding arm motion.

## Historical: standalone two-window hand setup

Return to the two-hand setup after the right glove's Ethernet/data adapter path is repaired and both fixed addresses respond:

```bash
ping -c 3 -I enp3s0 192.168.1.100
ping -c 3 -I enp3s0 192.168.1.101
```

The right glove serial previously discovered was `WG1KA06260815543`. Do not assume it is online; confirm it with the SDK scan before launching a right-hand process.

The current project runs one simulated hand per process. Two terminals therefore create two MuJoCo windows and two independent SDK connections.

### 1. Power and connect the right glove

Connect the right glove with the same cable orientation and verify:

```bash
ping -I 192.168.1.50 192.168.1.101
```

### 2. Discover both serial numbers

Run the discovery command again and record both devices. Do not guess the right serial number.

### 3. Start the left simulated hand

Terminal A:

```bash
cd /home/isaac/Documents/Dexmate/wuji-retargeting/example

/home/isaac/Documents/Dexmate/.venv/bin/python \
  teleop_sim.py \
  --input wuji_glove \
  --hand left \
  --glove-sn WG1JA06260812032
```

### 4. Start the right simulated hand

Terminal B, replacing `<RIGHT_GLOVE_SERIAL>` with the serial returned by the SDK scan:

```bash
cd /home/isaac/Documents/Dexmate/wuji-retargeting/example

/home/isaac/Documents/Dexmate/.venv/bin/python \
  teleop_sim.py \
  --input wuji_glove \
  --hand right \
  --glove-sn <RIGHT_GLOVE_SERIAL>
```

The current project runs one simulated hand per process. Two terminals therefore create two MuJoCo windows and two independent SDK connections.

### 5. Verify both streams

Move one hand at a time and confirm that only the matching simulated hand moves. Keep fingers relaxed and slightly bent during startup.

If the SDK reports a timeout:

```bash
ip neigh show dev enp3s0
ip route get 192.168.1.100
ip route get 192.168.1.101
```

Both glove addresses should route through `enp3s0`. If multiple interfaces are on `192.168.1.0/24`, use the routing helper in:

```text
/home/isaac/Documents/Dexmate/wuji-hand-teleop/docker/scripts/setup_glove_routes.sh
```

That helper requires the container's `NET_ADMIN` capability and may not work from the restricted VS Code terminal.

## Vega 1U Integration Plan

Do not connect glove joint commands directly to the Vega arm joints. The glove provides a 21-joint hand skeleton. The integration should proceed in stages:

1. **Input stage:** connect both Wuji Glove devices and receive stable 21-joint skeleton frames.
2. **Retargeting stage:** run the existing Wuji retargeter and produce left/right Wuji Hand joint positions.
3. **Hand simulation stage:** verify both simulated Wuji Hands move correctly.
4. **Vega asset stage:** load the Dexmate Vega 1U model separately in MuJoCo using:

   ```text
   /home/isaac/Documents/Dexmate/dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf
   ```

5. **Attachment stage:** select the correct Vega 1U wrist/end-effector asset and attach the simulated Wuji Hand at the left and right wrist frames.
6. **Control stage:** map the retargeted hand joint positions to the attached simulated hand joints while keeping arm and base control separate.
7. **Hardware stage:** only after simulation matches the real glove behavior, connect the same command path to real hardware with the robot stopped and emergency stops available.

The existing Dexstream hand retargeting code is here:

```text
/home/isaac/Documents/Dexmate/dexstream/src/dexstream/retargeting/
```

The existing Vega 1U MuJoCo launcher is here:

```text
/home/isaac/Documents/Dexmate/scripts/launch_vega_1u_mujoco.py
```

Generated Vega 1U MuJoCo files are cached here:

```text
/home/isaac/.cache/dexmate-urdf/vega_1u/
```

## Safety Checklist

Before wearing or streaming:

- Keep fingers slightly bent and relaxed.
- Do not make a fist during startup.
- Keep fingertip EMF modules from crossing.
- Keep the robot and hand workspace clear.
- Keep physical and software emergency stops available.
- Stop the simulation before changing cable or glove connections.
- Do not send commands to real robot hardware until the simulated mapping is verified.
