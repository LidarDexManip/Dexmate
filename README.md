# Dexmate Vega-1U Teleoperation

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
