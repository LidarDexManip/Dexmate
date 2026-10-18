# Agent guide

Instructions for AI coding assistants working in this repository.

## Safety

- The `wuji_teleop` simulation (MuJoCo) is the default target. Do not start the Jetson follower, `omni-paddle` leader, or anything that commands the physical Vega-1U unless the user explicitly asks; see `TELEOP_VR_RUNBOOK.md` for that path.
- Do not commit or push unless asked. Never commit `.venv/`, `SIM_RUN_LOG.txt`, firmware, `.deb`, `.zip`, or `.dzcfg` files.
- Do not edit submodules (`dexstream`, `dexmate-urdf`, `omniteleop`, `third_party/*`, `mujoco-sim`, `wuji-*`).
- Network changes: use `scripts/configure_wuji_network.sh` (temporary, in-memory profile). Do not change the default route or saved profiles.

## Set up a machine

```bash
scripts/setup_sim.sh          # idempotent; ends with "Setup complete" after the tests pass
```

It needs `git`, `curl`, and Python 3.11-3.12 with venv. If it fails on a missing system package, tell the user the `sudo apt install ...` line it printed; do not run sudo yourself.

## Verify

```bash
.venv/bin/python -m unittest tests.test_vega_arm_sim tests.test_vega_pico_hand \
  tests.test_vega_pico_wrist tests.test_vega_bimanual_sim \
  tests.test_imu_arm_source tests.test_vr_pose_source
.venv/bin/python -m unittest discover -s tests      # full suite; one GUI test is skipped
```

Without hardware:

- Joint limit tool: `scripts/launch_vega_joint_limits.sh`, which opens windows and needs `DISPLAY`.
- Headless sim check: `.venv/bin/python -m wuji_teleop.vega_arm_sim --arm-input synthetic --finger-input synthetic --headless --duration 5`.

Hardware checks:

- Glove: `bash scripts/configure_wuji_network.sh`, then `ping -c 2 192.168.1.100` and `.venv/bin/python -m wuji_teleop discover --seconds 3`.
- Pico: `scripts/launch_dexstream.sh`, then `curl -sk -o /dev/null -w '%{http_code}' https://<wifi-ip>:8091/` should print `200`.

## Run

Launchers are in `scripts/` and listed in `SIM_LAUNCH_COMMANDS.txt` and `NEW_MACHINE.md`. They auto-detect DexStream (`.venv/bin/dexstream` first), the Wi-Fi IP and the glove, and append each run to `SIM_RUN_LOG.txt`. When starting a viewer from a tool shell, set `DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority` and run it in the background. To stop only the sim, kill the `python -m wuji_teleop.vega_arm_sim` process.

The bilateral mode is `scripts/launch_wuji_vega_bimanual_pico.sh`:

- Pico left/right hand tracking drives matching simulated arms, joints 1-4.
- The two glove palm IMUs drive matching wrists, joints 5-7.
- The two glove skeletons drive matching fingers.
- The user presses **R** to calibrate both sides.

The single-left mode remains `scripts/launch_wuji_vega_pico_hand.sh`:

- Pico hand tracking (DexStream `--input hand`, topic `vr/hands`) drives arm joints 1-4.
- The glove palm IMU drives the wrist, joints 5-7.
- The glove skeleton drives the fingers.
- The user presses **R** to calibrate.
- Each run writes a CSV log to `/tmp/vega-pico-hand-<time>.csv`.

## Code map

- `wuji_teleop/vega_arm_sim.py`:
  - scene build and joint limits (`ARM_JOINT_LIMITS`, `ARM_HOME_QPOS`);
  - IK (`DampedLeastSquaresIK`, `glove_arm_ik`, `pico_upper_ik`, `WristOrientationIK`);
  - `PicoHandArmMapper`;
  - the `run()` loop and the CLI.
- `wuji_teleop/vega_bimanual_sim.py`: both articulated arms/hands and bilateral run loop.
- `wuji_teleop/vr_pose_source.py`: Zenoh readers for Pico controllers and hands.
- `wuji_teleop/imu_arm_source.py`: glove palm IMU to arm or wrist targets.
- `wuji_teleop/vega_arm_range.py`: Tk joint jog tool.
- `wuji_teleop/sources.py`: glove connection.

All scene coordinates are robot frame: x forward, y left, z up, metres.
