# Dexmate Vega-1U Teleoperation

Two teleoperation setups are documented here. Install and run the one you are using.

| Setup | What moves | Where it runs |
| --- | --- | --- |
| [Pico on the physical Vega-1U](#1-pico-teleop-on-the-physical-vega-1u) | Both arms, from the Pico controllers | Laptop WebXR page, laptop leader, Jetson follower |
| [Wuji glove in MuJoCo](#2-wuji-glove-teleop-in-mujoco) | Simulated Wuji Hand 2 fingers, plus a chosen wrist or arm mode | Laptop MuJoCo viewer |
| [Gloves + Pico headset quick start](#3-quick-start-glove-and-pico-headset-in-mujoco) | Pico hand tracking moves one or both simulated arms; matching gloves drive wrists and fingers | Laptop MuJoCo viewer + Pico browser |

The Pico section drives the physical robot. The Wuji glove section drives a MuJoCo simulation on the laptop. An optional Pico wrist mode at the end of the glove section uses the headset only to rotate the simulated wrist while the glove still drives the fingers.

Longer network, calibration, and troubleshooting notes are in [TELEOP_VR_RUNBOOK.md](TELEOP_VR_RUNBOOK.md) and [WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md).

## Get the code

```bash
git clone --recurse-submodules https://github.com/LidarDexManip/Dexmate.git
cd Dexmate
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

Commands below assume the repository is at `~/Documents/Dexmate`. Run them from that directory unless a block says otherwise.

The checkout includes `wuji_teleop/`, the launch scripts, and these submodules:

- `dexstream` and `omniteleop` for Pico teleoperation
- `dexmate-urdf` for the Vega 1U model
- `third_party/wuji-description` for the Wuji Hand model
- `mujoco-sim`, `wuji-hand-teleop`, and `wuji-retargeting` for the upstream Wuji projects

Firmware, Debian packages, archives, and robot `.dzcfg` files are not in git. Obtain hardware packages through the approved release or device-provisioning process, and keep passwords out of the repository.

## 1. Pico teleop on the physical Vega-1U

Full notes: [TELEOP_VR_RUNBOOK.md](TELEOP_VR_RUNBOOK.md).

The laptop needs Ethernet to the robot and Wi-Fi for the Pico. The Pico joins the same Wi-Fi network as the laptop.

| | |
| --- | --- |
| Laptop Ethernet to the robot | `192.168.50.10` |
| Jetson | `192.168.50.20` |
| Laptop Wi-Fi for the Pico | `192.168.1.65` |
| Pico page | `https://192.168.1.65:8091/` |
| Robot name | `dm/vgce08fed0a4-1u` |
| Robot profile | `vega_1u` |

Confirm the physical emergency stop and keep the robot workspace clear before the follower starts. The follower can move the arms to the home pose while it initializes.

### Install on the laptop

The laptop uses the `code_test` Conda environment. The robot Zenoh file must be at `/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg`.

```bash
cd ~/Documents/Dexmate
conda activate code_test
conda install -n code_test -c conda-forge nodejs -y

cd ~/Documents/Dexmate/dexstream/frontend
npm install

cd ~/Documents/Dexmate
python -m pip install -e ./dexstream
python -m pip install -e './omniteleop[vr]'
python -m pip install pyserial
```

### Install on the Jetson

The Jetson uses `/home/dexmate/miniconda3` and Python 3.13 ARM64. This install uses OMPL 2.0.1 and dexmotion 0.4.3.

```bash
/home/dexmate/miniconda3/bin/python -m pip install \
  ompl==2.0.1 fastapi hatch joycon-lib

/home/dexmate/miniconda3/bin/python -m pip install \
  --upgrade dexmotion==0.4.3

/home/dexmate/miniconda3/bin/python -m pip install \
  pynput python-socketio pytransform3d rerun-sdk 'uvicorn[standard]'

/home/dexmate/miniconda3/bin/python -m pip install \
  --no-deps -e /home/dexmate/omniteleop

/home/dexmate/miniconda3/bin/python -X faulthandler -c \
'import ompl, dexmotion; print("native imports ok")'
```

### Run

Keep three terminals running. Start dexstream, then the Jetson follower, then the laptop leader.

**Terminal 1, on the laptop:**

```bash
cd ~/Documents/Dexmate
conda activate code_test
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg

dexstream --host 192.168.1.65 --port 8090 --mode wireless --input paddle
```

**On the Pico:** open `https://192.168.1.65:8091/`, accept the certificate warning, and press **Enter AR**. Use the main page. `/preview` only watches the stream.

**Terminal 2, on the Jetson:**

```bash
export ROBOT_CONFIG=vega_1u
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/dexmate/.dexmate/comm/zenoh/dm_vgce08fed0a4-1u.dzcfg

/home/dexmate/miniconda3/bin/python \
  /home/dexmate/omniteleop/src/omniteleop/follower/robot_controller.py
```

Wait for `Robot controller initialized` and `Starting robot control at 100Hz`.

**Terminal 3, on the laptop:**

```bash
cd ~/Documents/Dexmate
conda activate code_test
export ROBOT_CONFIG=vega_1u
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg

omni-paddle --debug
```

Wait until the leader reports that its hot loop is running, then hold an index trigger. The leader should log `Left controller tracking started` or `Right controller tracking started`.

### Controls

- Hold the left or right index trigger to move that arm. Release it to hold the arm.
- Hold both thumbstick clicks for about one second to toggle the software E-stop. Press each stick straight down.
- Vega-1U has no torso, so leave `--unlock-torso` off.
- For the first motion, release the physical E-stop, enter AR, and make a small movement with one trigger held.

## 2. Wuji glove teleop in MuJoCo

Full notes: [WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md). Standalone Hand 2, without Vega, is in [WUJI_TELEOP_RUNBOOK.md](WUJI_TELEOP_RUNBOOK.md).

This path uses a laptop virtualenv, `.venv`, separate from the Pico `code_test` environment. The tested Python is 3.11. A viewer needs a desktop graphics session.

### Install

```bash
sudo apt install git python3-venv network-manager ripgrep

cd ~/Documents/Dexmate
git submodule update --init --recursive
test -d .venv || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-wuji-sim.txt
.venv/bin/python -m pip check
```

`requirements-wuji-sim.txt` pins `wuji-sdk==2026.9.22`, `mujoco==3.15.0`, `numpy==2.4.6`, `trimesh==5.1.1`, and `urdf2mjcf==0.2.39`.

Confirm the models are present:

```bash
test -f scripts/launch_vega_1u_mujoco.py
test -f dexmate-urdf/robots/humanoid/vega_1u/vega_1u.urdf
test -f third_party/wuji-description/hand2/hand2_beta2/body/mjcf/left.xml
```

### Check the scene without a glove

```bash
cd ~/Documents/Dexmate
.venv/bin/python scripts/launch_vega_1u_mujoco.py --convert-only
.venv/bin/python -m wuji_teleop.vega_hand_sim --input synthetic --duration 10
```

### Connect the left glove

Use the glove acquisition board, the supplied USB-C cable, a powered USB-C/Ethernet adapter, and Ethernet to the laptop. This machine's wired interface is `enp3s0`; substitute the interface that shows a link.

```bash
ip -brief address
cat /sys/class/net/enp3s0/carrier
bash scripts/configure_wuji_network.sh enp3s0
.venv/bin/python -m wuji_teleop discover --seconds 3
```

`carrier` prints `1` when the link is up. The setup script gives that interface `192.168.1.50/24` and leaves the normal internet route alone. The left glove used here is `WG1JA06260812032`. Use the serial printed by `discover` if it differs, and pass that serial to the launch commands below.

### Run the glove

Stop one viewer before starting another. Close the window or press Ctrl-C to exit. **Space** pauses finger tracking. **R**, held still, recenters a glove IMU mode.

**Fingers only, Vega arm fixed:**

```bash
cd ~/Documents/Dexmate
.venv/bin/python -m wuji_teleop.vega_hand_sim \
  --glove-sn WG1JA06260812032 \
  --report /tmp/vega-hand-live.json
```

**Fingers plus wrist rotation.** The glove IMU turns the three wrist joints. The shoulder and elbow stay fixed. Hold the glove still for one second at startup.

```bash
./scripts/launch_wuji_vega_glove_wrist.sh
```

Pass another serial as the first argument when discovery reports one: `./scripts/launch_wuji_vega_glove_wrist.sh YOUR_SERIAL`.

**Whole arm from the glove.** Tip the hand, or draw it inward, and the arm follows. The solver bends the elbow when the wrist target moves closer to the shoulder, which is what lets that closer position be reached. Twisting the glove aims the hand. Hold still for one second at startup. Press **R** while still to recenter.

```bash
./scripts/launch_wuji_vega_glove_arm.sh
```

### Optional: Pico rotates the simulated wrist

The glove still drives the fingers. Pico controller rotation drives the three wrist joints while the shoulder and elbow stay fixed. Install the Pico laptop packages in [section 1](#install-on-the-laptop) first, then add the simulation Zenoh package and start the combined launcher:

```bash
cd ~/Documents/Dexmate
.venv/bin/python -m pip install -r requirements-wuji-vr-sim.txt
./scripts/launch_wuji_vega_pico_wrist.sh
```

The script prints `https://<wifi-ip>:8091/`. Open that page on the Pico, accept the certificate, enter VR, and hold the **left index trigger** while rotating the controller. Release the trigger to hold the wrist.

If the script picks the wrong address, pass the laptop Wi-Fi IP and the glove serial:

```bash
./scripts/launch_wuji_vega_pico_wrist.sh 192.168.1.65 WG1JA06260812032
```

The launcher uses `.venv/bin/dexstream` (installed by `scripts/setup_sim.sh`), then `dexstream` on `PATH` or in a conda env; set `DEXSTREAM_BIN` to force one. This launcher starts the MuJoCo viewer only. The Jetson follower and `omni-paddle` leader from section 1 stay stopped.

## 3. Quick start: glove and Pico headset in MuJoCo

Simulation only; nothing here commands the physical robot. The full new-machine walkthrough is [NEW_MACHINE.md](NEW_MACHINE.md), every command is listed in [SIM_LAUNCH_COMMANDS.txt](SIM_LAUNCH_COMMANDS.txt), and every launcher appends what it ran to `SIM_RUN_LOG.txt`. AI assistants: see [AGENTS.md](AGENTS.md).

### Install everything (once)

```bash
cd ~/Documents/Dexmate
scripts/setup_sim.sh
```

This creates `.venv` and installs the pinned packages, DexStream and the Pico page's three.js into it. It then runs the tests and ends with `Setup complete`. It needs `git`, `curl`, and Python 3.11-3.12 with venv.

### Glove setup

1. Connect the left glove: acquisition board -> supplied USB-C cable -> **powered** USB-C/Ethernet adapter -> laptop Ethernet.
2. Configure the port and check the glove:

   ```bash
   bash scripts/configure_wuji_network.sh
   .venv/bin/python -m wuji_teleop discover --seconds 3
   ```

   The left/right gloves normally use `192.168.1.100` and `.101`; the laptop gets `192.168.1.50`. The script uses a temporary profile and leaves the internet route alone. If Wi-Fi is also on `192.168.1.x`, it routes only the two glove addresses over Ethernet so the Pico stays on Wi-Fi.
3. The launchers connect to the only glove found. With several gloves, pass the serial as the first argument or set `GLOVE_SN`.

### Pico headset setup

1. Put the Pico and the laptop on the same Wi-Fi.
2. Start a Pico launcher (below). It finds the Wi-Fi address and DexStream itself and prints `https://<wifi-ip>:8091/`.
3. Open that URL in the Pico browser (the main page, not `/preview`) and accept the certificate.
4. Put both controllers down so the headset switches to hand tracking, then press **Enter AR**.

### Which launcher to run

| Launcher | What it does | Needs |
| --- | --- | --- |
| `scripts/launch_vega_joint_limits.sh` | Joint limit tool: slider and −/+ per left-arm joint inside `ARM_JOINT_LIMITS` (`wuji_teleop/vega_arm_sim.py`) | nothing |
| `scripts/launch_wuji_vega_hand_only.sh` | Hand only: glove drives the fingers, arm fixed | glove |
| `scripts/launch_wuji_vega_pico_hand.sh` | Hand + Pico: headset hand tracking moves the arm (joints 1-4), the glove turns the wrist (joints 5-7) and drives the fingers | glove + Pico |
| `scripts/launch_wuji_vega_bimanual_pico.sh` | Bilateral: Pico tracks both hands; two gloves drive matching wrists and fingers | two gloves + Pico |
| `scripts/launch_wuji_vega_bimanual_trackers.sh` | Wrist trackers: a PICO Motion Tracker on each wrist gives the 6D wrist pose; GMR-style IK moves all 7 joints of each arm; two gloves drive the fingers | two trackers + Pico + XRoboToolkit PC Service (+ two gloves) |

For bilateral use, the gloves normally answer at `192.168.1.100` and `.101`.
The network helper installs host routes for both when Pico Wi-Fi is on the same
subnet. The bilateral launcher auto-selects one discovered glove of each side,
or accepts Wi-Fi IP, left serial, and right serial as positional arguments.

Close the MuJoCo window or press Ctrl-C to stop. Glove-only wrist and whole-arm modes are in [section 2](#run-the-glove).

**Hand + Pico steps:**

1. Run `scripts/launch_wuji_vega_pico_hand.sh`.
2. Open the Pico page and enter AR with the controllers down.
3. Hold the glove still for one second so the wrist calibrates.
4. Wait for the status line to show `hand tracking`.
5. Hold your hand in a comfortable working position and press **R** in the MuJoCo window.

**Wrist tracker steps** (XRoboToolkit instead of DexStream):

1. Once: install the XRoboToolkit PC Service `.deb`, then run `scripts/install_xrobotoolkit_sdk.sh` (`setup_sim.sh` runs it when the service is installed).
2. Strap a PICO Motion Tracker on the back of each wrist.
3. Run `scripts/launch_wuji_vega_bimanual_trackers.sh`. It starts the PC Service if needed.
4. On the Pico, open XRoboToolkit and connect to the laptop. In the **Pico Motion Tracker** tab, choose object tracking and press **Send**. Without Send, no tracker data arrives.
5. The first tracker frame calibrates. To recalibrate, hold both forearms forward, palms down, look straight ahead, and press **R**. Both arms return to their mirrored home pose.

How it retargets (GMR, Ze et al. 2025, Section IV), configured in `wuji_teleop/ik_configs/xr_trackers_to_vega_1u.json` (GMR format):

- Key bodies: left/right wrist tracker -> `L_ee`/`R_ee`, with position and orientation costs per IK stage.
- Rest-pose offsets: **R** sets each wrist's rotation offset and position offset so the current pose maps onto the robot hands at their mirrored home pose. Tracking is relative to that moment. The operator heading comes from the headset.
- Scaling: wrist motion times `human_scale_table` times `HUMAN_HEIGHT / human_height_assumption`.
- IK: two mink stages (`ik_match_table1`, then `ik_match_table2`), each repeated while the error falls by more than 1e-3 (at most 10 extra solves), warm-started from the previous frame. Only the 14 arm joints move.
- Joint limits: every IK step is a QP with hard joint-range bounds. The sim leaves the elbow (`j4`) unlimited, so `joint_limits` in the config adds the URDF elbow range, `[-3.071, 0.244]` rad, for the IK only.
- Trackers are told apart by position at calibration; set `LEFT_TRACKER_SN`/`RIGHT_TRACKER_SN` to pin them. `FINGER_INPUT=synthetic` runs without gloves.
- Each run logs targets, IK error and joints to `/tmp/vega-bimanual-tracker-<time>.csv`.

For the bilateral pre-hardware acceptance sequence and report thresholds, see
[RETARGETING_ACCEPTANCE.md](RETARGETING_ACCEPTANCE.md).

**Details:**

- The arm holds when the headset loses your hand.
- Pico motion defaults to 0.6 robot metres per tracked metre; use `PICO_SCALE`
  to tune it. Tracking is filtered, jump-limited, and projected into the safe
  arm workspace before IK.
- Each run logs raw/filtered Pico XYZ, target, achieved pose, workspace clipping,
  and all seven joints to `/tmp/vega-pico-hand-<time>.csv`.

To restart the sim without dropping the Pico page:

1. Run `scripts/launch_dexstream.sh` in one terminal and leave it running.
2. Run `scripts/launch_wuji_vega_pico_hand.sh` in another. It detects DexStream and starts only the sim.

| Symptom | Fix |
| --- | --- |
| `No wired interface with a link` | Glove adapter unpowered or cable out. |
| Glove not found / `ping 192.168.1.100` fails | Rerun `bash scripts/configure_wuji_network.sh`. If the port has another profile active, `nmcli device disconnect <port>` first. |
| Pico page does not load | DexStream is not running or the IP is wrong; run `scripts/launch_dexstream.sh` and use the URL it prints. |
| Status stays `hand waiting` | The headset is not tracking your hand: enter AR, controllers down, hand in view. |
| Arm is offset or rises too high | Hold your hand in your working position and press **R**. |
