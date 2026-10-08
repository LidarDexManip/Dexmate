# Dexmate Vega-1U Teleoperation

For the current **Wuji glove + Wuji Hand 2 mounted on Vega 1U in MuJoCo**,
including installation and the glove-only moving-arm controls, start with
[WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md).
For glove IMU wrist rotation with a fixed shoulder and elbow, run
[`scripts/launch_wuji_vega_glove_wrist.sh`](scripts/launch_wuji_vega_glove_wrist.sh).
The [Pico wrist-only launcher](scripts/launch_wuji_vega_pico_wrist.sh) is a
separate controller option.

For wearable **Wuji glove → standalone Wuji Hand 2 in MuJoCo**, start with
[WUJI_TELEOP_RUNBOOK.md](WUJI_TELEOP_RUNBOOK.md). This includes a hardware-free
demo, Ethernet setup, live glove input, and verification commands. The left
Hand 2 can also be mounted on a stationary Vega1U in MuJoCo; see
[WUJI_HAND_MUJOCO_RUNBOOK.md](WUJI_HAND_MUJOCO_RUNBOOK.md#current-single-arm-simulation-path--october-8-2026).
The same runbook also covers an optional moving left arm driven by a Pico
controller pose or short relative motion estimated from the glove's palm IMU,
or by glove movement gestures that command larger arm steps, with Wuji glove
fingers, entirely in MuJoCo.

This repository collects the robot-specific setup and the two software
components needed for VR teleoperation:

- [`dexstream`](https://github.com/dexmate-ai/dexstream) serves the WebXR
  controller page.
- [`omniteleop`](https://github.com/dexmate-ai/omniteleop) runs the laptop
  leader and Jetson follower.

The supported setup documented here is the top-half-only Vega-1U with a Pico
headset. Start with [TELEOP_VR_RUNBOOK.md](TELEOP_VR_RUNBOOK.md). It contains
the network details, one-time installation, startup order, controls, and
troubleshooting steps.

## Get the code

```bash
git clone --recurse-submodules https://github.com/LidarDexManip/Dexmate.git
cd Dexmate
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

That checkout includes the local teleop package (`wuji_teleop/`), launch
scripts, tests, and these dependency folders as submodules:

- `dexstream` and `omniteleop` for Pico VR teleoperation
- `dexmate-urdf` for the Vega 1U model
- `third_party/wuji-description` for the Wuji Hand MJCF
- `mujoco-sim`, `wuji-hand-teleop`, and `wuji-retargeting` for the upstream
  Wuji simulation and glove projects

Install and run the glove-to-Vega MuJoCo path from
[WUJI_VEGA_MUJOCO_SETUP.md](WUJI_VEGA_MUJOCO_SETUP.md). Install and run Pico VR
teleoperation from [TELEOP_VR_RUNBOOK.md](TELEOP_VR_RUNBOOK.md).

## Before running teleoperation

1. Confirm the robot-specific Zenoh configuration is available locally. Do
   not commit `.dzcfg` files or passwords.
2. Complete the laptop and Jetson installation steps in the runbook.
3. Verify the physical emergency stop and keep the robot workspace clear.
4. Follow the runbook's startup order: dexstream, Jetson follower, then the
   laptop leader.

The firmware, Debian packages, archives, and robot configuration files in the
working folder are intentionally not part of the Git repository. Obtain
hardware packages through the approved release or device-provisioning process.
