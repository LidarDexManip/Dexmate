# Vega-1U VR Teleoperation Runbook

This runbook starts VR teleoperation for the top-half-only Vega-1U using a Pico headset.

## Network

- Laptop Ethernet to robot: `192.168.50.10`
- Jetson: `192.168.50.20`
- Laptop Wi-Fi for the Pico: `192.168.1.65`
- Pico WebXR URL: `https://192.168.1.65:8091/`
- Robot name: `dm/vgce08fed0a4-1u`
- Robot profile: `vega_1u`

The laptop must have both Ethernet and Wi-Fi connected. The Pico must join the same Wi-Fi network as the laptop.

## One-Time Installation

### Laptop

The laptop uses the `code_test` Conda environment.

```bash
cd ~/Documents/Dexmate
conda activate code_test

# Install Node.js/npm without sudo.
conda install -n code_test -c conda-forge nodejs -y

# Install the WebXR frontend dependencies.
cd ~/Documents/Dexmate/dexstream/frontend
npm install

# Install/update Python packages if needed.
cd ~/Documents/Dexmate
python -m pip install -e ./dexstream
python -m pip install -e './omniteleop[vr]'
python -m pip install pyserial
```

The robot Zenoh configuration must be present on the laptop at:

```text
/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg
```

### Jetson

The Jetson uses `/home/dexmate/miniconda3` and Python 3.13 ARM64. The standard OmniTeleop dependency pin for `ompl==1.7.0` is not available for this platform. The working installation uses OMPL 2.0.1 and dexmotion 0.4.3.

```bash
/home/dexmate/miniconda3/bin/python -m pip install \
  ompl==2.0.1 fastapi hatch joycon-lib

/home/dexmate/miniconda3/bin/python -m pip install \
  --upgrade dexmotion==0.4.3

/home/dexmate/miniconda3/bin/python -m pip install \
  pynput python-socketio pytransform3d rerun-sdk 'uvicorn[standard]'

/home/dexmate/miniconda3/bin/python -m pip install \
  --no-deps -e /home/dexmate/omniteleop
```

Verify the native imports before using the robot:

```bash
/home/dexmate/miniconda3/bin/python -X faulthandler -c \
'import ompl, dexmotion; print("native imports ok")'
```

## Startup Order

Start dexstream first, then the Jetson follower, then the laptop leader.
Keep all three terminals running.

### Terminal 1: Laptop dexstream

Run this on the laptop, not inside the Jetson SSH session:

```bash
cd ~/Documents/Dexmate
conda activate code_test
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg

dexstream --host 192.168.1.65 --port 8090 --mode wireless --input paddle
```

The Pico uses HTTPS port `8091` because dexstream serves HTTPS on `port + 1`.

### Pico headset

Connect the Pico to the same Wi-Fi network as the laptop and open this in a WebXR-capable browser:

```text
https://192.168.1.65:8091/
```

Accept the self-signed certificate warning and press **Enter AR**. Use the main page, not `/preview`; `/preview` only observes data and does not publish controller input.

### Terminal 2: Jetson follower

SSH to the Jetson and run:

```bash
export ROBOT_CONFIG=vega_1u
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/dexmate/.dexmate/comm/zenoh/dm_vgce08fed0a4-1u.dzcfg

/home/dexmate/miniconda3/bin/python \
  /home/dexmate/omniteleop/src/omniteleop/follower/robot_controller.py
```

Wait for:

```text
Robot controller initialized
Starting robot control at 100Hz
```

The follower may move the arms to the configured home pose during initialization. Keep the robot area clear.

### Terminal 3: Laptop leader

Run this on the laptop:

```bash
cd ~/Documents/Dexmate
conda activate code_test
export ROBOT_CONFIG=vega_1u
export ROBOT_NAME=dm/vgce08fed0a4-1u
export ZENOH_CONFIG=/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg

omni-paddle --debug
```

Wait for the leader to report that its hot loop is running. Then hold an index trigger. The leader should log:

```text
Left controller tracking started
```

or:

```text
Right controller tracking started
```

## Controls

- Hold left or right index trigger: clutch in and move that arm.
- Release the index trigger: clutch out and hold the arm.
- Hold both thumbstick clicks for about one second: toggle software E-stop.
- Press the thumbstick straight down until it clicks; do not move it sideways.
- Vega-1U has no torso, so do not use `--unlock-torso`.

For the first test, keep the robot area clear, release the physical E-stop, press **Enter AR**, toggle the software E-stop if needed, and make only a small controller movement while holding one index trigger.

## Troubleshooting

### Port 8080 is already in use

An older dexstream process is still running. Either stop it with `Ctrl+C`, or use the working alternate port:

```bash
dexstream --host 192.168.1.65 --port 8090 --mode wireless --input paddle
```

Then use `https://192.168.1.65:8091/` on the Pico.

### Pico page is blank

Ensure the laptop frontend dependencies are installed:

```bash
cd ~/Documents/Dexmate/dexstream/frontend
conda run -n code_test npm install
```

Restart dexstream after installing them. The Pico browser must support modern JavaScript modules and WebXR.

### Preview shows controller positions but the robot does not move

`/preview` only displays the stream. Confirm the Pico is using the main `/` page and has an active **Enter AR** session. Hold an index trigger and check the leader terminal for `controller tracking started`.

### Leader cannot see the controller topic

Both dexstream and omni-paddle must use the same robot Zenoh configuration:

```bash
export ZENOH_CONFIG=/home/isaac/Downloads/dm_vgce08fed0a4-1u.dzcfg
```

Restart both laptop processes after setting it.

### Torso warnings

Warnings such as `Torso joints are not active` are expected for `vega_1u`; this robot has no torso.

### E-Stop

If the Jetson reports `E-STOP BUTTON PRESSED`, release the physical E-stop first. The robot follower intentionally ends home initialization with the software E-stop active. Use both thumbstick clicks to toggle the software E-stop, then test with an index trigger.
