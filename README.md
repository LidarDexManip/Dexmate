# Dexmate Vega-1U Teleoperation

Two teleoperation setups are documented here. Install and run the one you are using.

| Setup | What moves | Where it runs |
| --- | --- | --- |
| [Pico on the physical Vega-1U](#1-pico-teleop-on-the-physical-vega-1u) | Both arms, from the Pico controllers | Laptop WebXR page, laptop leader, Jetson follower |
| [Wuji glove in MuJoCo](#2-wuji-glove-teleop-in-mujoco) | Simulated Wuji Hand 2 fingers, plus a chosen wrist or arm mode | Laptop MuJoCo viewer |

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

The default DexStream executable is `/home/isaac/miniconda3/envs/code_test/bin/dexstream`. Point `DEXSTREAM_BIN` at another executable when that path is different. This launcher starts the MuJoCo viewer only. The Jetson follower and `omni-paddle` leader from section 1 stay stopped.
