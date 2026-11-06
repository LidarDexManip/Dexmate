# Agent guide

Instructions for AI coding assistants working in this repository.

## Safety

- The `wuji_teleop` simulation (MuJoCo) is the default target. Do not start the Jetson follower, `omni-paddle` leader, `scripts/launch_wuji_vega_hardware_pico.sh`, or anything that commands the physical Vega-1U unless the user explicitly asks; see `TELEOP_VR_RUNBOOK.md` for the follower path.
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
  tests.test_imu_arm_source tests.test_vr_pose_source \
  tests.test_gmr_arm_retarget tests.test_xr_tracker_source \
  tests.test_collision_check tests.test_hardware_teleop tests.test_connection_monitor \
  tests.test_hand_log tests.test_hand_compliance tests.test_hand_load
.venv/bin/python -m unittest discover -s tests      # full suite; one GUI test is skipped
```

Without hardware:

- Joint limit tool: `scripts/launch_vega_joint_limits.sh`, which opens windows and needs `DISPLAY`.
- Headless sim check: `.venv/bin/python -m wuji_teleop.vega_arm_sim --arm-input synthetic --finger-input synthetic --headless --duration 5`.

Hardware checks:

- Every link at once: `scripts/monitor_wuji_connections.sh` (live, read-only, with a fix per missing link); `--connect` also opens real connections and catches a device that answers discovery but refuses connections. Do not use `--connect` while the teleop runs.
- Glove: `bash scripts/configure_wuji_network.sh`, then `ping -c 2 192.168.1.100` and `.venv/bin/python -m wuji_teleop discover --seconds 3`.
- Pico: `scripts/launch_dexstream.sh`, then `curl -sk -o /dev/null -w '%{http_code}' https://<wifi-ip>:8091/` should print `200`.

## Run

Launchers are in `scripts/` and listed in `SIM_LAUNCH_COMMANDS.txt` and `NEW_MACHINE.md`. They auto-detect DexStream (`.venv/bin/dexstream` first), the Wi-Fi IP and the glove, and append each run to `SIM_RUN_LOG.txt`. When starting a viewer from a tool shell, set `DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority` and run it in the background. To stop only the sim, kill the `python -m wuji_teleop.vega_arm_sim` process.

The bilateral mode is `scripts/launch_wuji_vega_bimanual_pico.sh`:

- Pico left/right hand tracking drives matching simulated arms, joints 1-4.
- The two glove palm IMUs drive matching wrists, joints 5-7.
- The two glove skeletons drive matching fingers.
- The user presses **R** to calibrate both sides.

The wrist tracker mode is `scripts/launch_wuji_vega_bimanual_trackers.sh` (no DexStream):

- XRoboToolkit object tracking streams one PICO Motion Tracker per wrist through the PC Service (`/opt/apps/roboticsservice`) to `xrobotoolkit_sdk`, built into `.venv` by `scripts/install_xrobotoolkit_sdk.sh`.
- GMR-style two-stage mink IK maps each 6D wrist pose to all seven joints of the matching arm. The config is `wuji_teleop/ik_configs/xr_trackers_to_vega_1u.json`.
- The glove skeletons drive the fingers; `FINGER_INPUT=synthetic` runs without gloves, `FINGER_INPUT=left-glove` (or `right-glove`) with one.
- The first tracker frame calibrates; **R** recalibrates.

The hardware mode is `scripts/launch_wuji_vega_hardware_pico.sh` (moves the physical robot; only when asked):

- Arm input defaults to the wrist-tracker mapping above (`ARM_INPUT=tracker`; the launcher starts the XRoboToolkit PC Service; Start calibrates at the robot's current pose). `ARM_INPUT=pico-hand` uses the bilateral Pico hand mapping and starts DexStream instead.
- `ARM_INPUT=none` (`--arm-input none`) never connects the arms: only the gloves drive the fingers, and a still stand-in at the reset pose serves the finger clearance checks. `HANDS=left` or `right` (`--hands`) connects one hand and its glove only. `HAND_CURRENT_LIMIT` (`--hand-current-limit`, at most 1.5 A) lowers each hand's per-joint current limit while connected and restores it on a normal exit; a killed run leaves it lowered.
- `HAND_COMPLIANCE=1` (`--hand-compliance`) adds HACo-style fingertip force regulation (`wuji_teleop/hand_compliance.py`; Ye et al. 2026, Fig. 3, hand branch). Each fingertip force is estimated from motor current through the Jacobian of the finger's `*_tip` site (the `*_tip_sensor_frame` bodies sit on the DIP axis), since the hands have no tactile sensors; units are uncalibrated N at the SDK's 1 N·m/A. Force along a nearly straight finger barely loads its joints, so it is damped in the estimate and not regulated. In contact it is clipped to `FINGERTIP_FORCE_LIMIT` (`--fingertip-force-limit`, default 4 N), and the glove's finger reference is bent back by a regularized least-squares joint correction. Only fingertip forces are regulated, so keep `HAND_CURRENT_LIMIT` as the backstop. The defaults assume the written kp 1.5 (about 2.85 A/rad in effect); a higher kp needs a larger stiffness or `lam_s`. With it on, the finger log adds `.compliance.bin` (nominal, compliant, observed, force, activation). It runs whenever a hand is connected, also before Start and while paused; E-stop (**Esc**) stops it and holds each hand's last compliant reference. A joint's own load (cover, tendons) and friction would read as force and hold back a thumb touching nothing, so each hand's table (`wuji_teleop/hand_load/<serial>.json`, from `python -m wuji_teleop.hand_load left|right`) is subtracted first; a hand without a table says so on its dashboard dot. That sweep moves the hand (only when asked; nothing may touch it; about 3 minutes; `--plan` does not connect).
- Commands stream at 100 Hz to the real arms through dexcontrol on the laptop (`requirements-wuji-hardware.txt`, robot `.dzcfg` in the repo root) and to both Wuji hands through `wuji_sdk`.
- `ARM_FEEDFORWARD` (`--arm-velocity-feedforward`, 0 to 2, default off) also sends each arm command's rate over the last 100 ms as velocity, times this scale, while following only; hold, reset, pause and E-stop send position alone. With position alone the arm's servo lags by about 0.41 s and shrinks fast motion (test of 2026-11-04, right joints 1 and 5: 36% of the swing and 231-240 ms late at 1 Hz). At 1.5 it made 99-104% of the swing, 35-44 ms late, on small smooth sinusoids. In tracker teleop on the real arms, command to measured fell from 335-435 ms (position alone, 2026-11-01) to 85-140 ms at 1.0 (2026-11-04); 1.5 is still to be tried there. It is the bulk of the arm delay: the IK adds none and the tracker filter about 29 ms.
- Finger commands are position-only (velocity and effort 0). Each hand's MIT gains are written on connect: kp 1.5 / kd 0.1 by default (`--hand-kp`/`--hand-kd`, launcher `HAND_KP`/`HAND_KD`), not saved to the hand. A left hand left at kp 12 / kd 0.3 buzzed and ran hot. Firmware 2.13.1 applies about 1.9 times the written kp and kd (free-motion sweep of both hands, 2026-11-02), and its damping uses the hand's own velocity estimate, which spikes to about 1 rad/s while a joint sticks.
- A finger joint that reports a latched fault when its hand is enabled is left out of the enable mask and sent its own measured angle; the hand runs on its other joints and its dashboard dot names the disabled joint. The right hand's middle.S2 (0x2102 phase overcurrent, 2026-11-02) is always left out (`DISABLED_JOINTS` in `hardware_teleop.py`), even when it reads clear after a restart, so its middle finger's spread is passive.
- A glove finger whose fingertip sensor reads low confidence copies a neighbour (`FINGER_STAND_INS` in `retarget.py`, now ring ← pinky): before retargeting, its three bones are rebuilt along the neighbour's bone directions, keeping its own base and lengths, since the SDK retargeter couples fingers and a bad finger also bends the others. The signal is the glove's EMF fingertip confidence (`emf_poses`, smoothed over 0.3 s), not the skeleton's per-joint confidence, in which healthy fingers dip to 0.4-0.5. It switches on below 0.6, off after 2 s at 0.8 or above, never copies a low neighbour, and is announced in the dashboard log and the hand's status line. The right glove's ring sensor failed this way on 2026-11-06 (confidence 0, no reading in half its frames, the rest near the palm), which curled the robot ring finger into the palm. To read a glove's sensors with the teleop closed: `wuji sub emf_poses --address 192.168.1.101:50001 --count 600 --jsonl` (left: .100); `wuji devices` and `wuji viz` do not find the gloves on this laptop.
- Each run logs finger telemetry to `/tmp/vega-hand-log-<time>.states.bin` (every 1 kHz joint state per hand: position, the hand's velocity, effort = measured motor current in A) and `.commands.bin` (every finger command with its pre-slew retargeted goal), plus `.glove.bin` (every glove skeleton frame the retargeter consumed while running: 21 keypoints in the glove's wrist frame), about 30 MB/min; `--hand-log PREFIX` moves it, `--hand-log ''` turns it off, and `wuji_teleop.hand_log.load(prefix)` reads it back.
- When the PC Service lists a tracker serial more than once (frozen leftovers of a stopped tracker), only the entry that is still updating is used.
- A dashboard drives it: **R** reset (hand-aware collision-checked path to upper arms 0.25 rad behind vertical, forearms level), **S** start (re-anchors relative arm tracking), **P**/**Space** pause (freezes arms and hands), **Esc** software E-stop, **U** release.
- Every teleop target is clearance-checked; a target closer than 1.5 cm is held, with both arms and both hands frozen until the operator's target moves back out. `TELEOP_GUARD` (`--teleop-guard`) changes this for teleop only: `all` (default), `allow-hands` (all but left hand against right hand) or `off` (nothing is held, nothing stops a collision; the dashboard says so). In a two-handed run on 2026-11-04 the guard froze the robot 22 times (25 s of 345 s, up to 3.1 s each); at 21 of them the nearest pair was hand against hand. The reset path always keeps 3 cm over all pairs, and refuses to start while parts overlap in the model. Only the robot is modelled: neither the reset path nor the guard knows about a table or anything else around it.
- The dashboard opens at once and shows a dot per link (robot, each hand, each glove, Pico server, each tracked hand): green working, yellow waiting, red failed with the fix. Failed links are retried; nothing exits. Controls work once the robot arms connect.
- Links must not stall each other. `wuji_sdk` holds the GIL during `scan()` (~3 s) and `connect(sn=...)` without a prior scan (~18 s), freezing the arm loop and starving the glove streams until the SDK drops the gloves. So hand discovery runs in a child process (`wuji_scan_isolated`), hands connect by address with a 3 s timeout, and a glove is pinged before its SDK connect. A glove silent for 3 s (e.g. dropped by the SDK after missed heartbeats) is closed and reconnected.
- A hand is never connected while its laptop source address sits on two interfaces (e.g. 192.168.1.50 on the hands port and the glove adapter): the SDK then connects from the wrong address and the failed attempt locks the hand until a 12 V power cycle.
- `--fake-robot` rehearses with the real Pico and gloves without connecting to the robot; add `--synthetic-input` for no inputs either.
- Below the link dots the dashboard shows the robot's ego (ZED X head, left image) and right wrist (ZED X One S) camera views (`wuji_teleop/robot_cameras.py`; `--no-cameras` hides them). A child process subscribes to `sensors/head_camera/left_rgb` and `sensors/right_wrist_camera/rgb`, decodes about 10 frames/s, and pipes 480-px frames to the dashboard, so decoding never competes with the arm loop. A camera without fresh frames shows "not detected": dexsensor on the Jetson publishes only enabled sensors (`dexsensor enable head_camera`, `dexsensor status`; run `dexsensor` with `/etc/environment.d/99-dexmate-robot.conf` loaded, or it cannot find the service).
- On this robot the right wrist camera is ZED camera ID 1 (a ZED X One S, S/N 309373651, GMSL port 2; the driver and ZED SDK report it as a ZED X One GS, which has the same sensor). dexsensor's default config looks for `camera_id = 2`, which assumes a left wrist camera on ID 1, so plain `dexsensor enable right_wrist_camera` fails with "Init failed after 3 attempts". Enable it with `dexsensor enable -f ~/dexsensor-right-wrist-id1.toml right_wrist_camera` (a single-sensor copy with ID 1 on the Jetson). Sensors start disabled after every dexsensor restart. Before a Jetson reboot (2026-11-02) opening the wrist camera hung Argus capture and stopped the head camera too, even after `systemctl restart dexsensor`; after the reboot both stream together. If it hangs again, reboot the Jetson.

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
- `wuji_teleop/hardware_teleop.py`: hardware reset/start/pause state machine, dexcontrol arms, Wuji hands, Tk dashboard.
- `wuji_teleop/hand_log.py`: background binary writer and reader for the 1 kHz finger telemetry.
- `wuji_teleop/hand_compliance.py`: fingertip force estimate from current and the HACo-style hand force regulator.
- `wuji_teleop/hand_load.py`: per-hand joint load tables, and the sweep that measures them.
- `wuji_teleop/collision_check.py`: mesh clearance (arms, both hands, torso/head) and the reset planner.
- `wuji_teleop/vr_pose_source.py`: Zenoh readers for Pico controllers and hands.
- `wuji_teleop/xr_tracker_source.py`: XRoboToolkit wrist tracker reader, OpenXR to robot frame, left/right assignment.
- `wuji_teleop/gmr_arm_retarget.py`: GMR-style key-body config, rest-pose offsets, scaling, two-stage mink IK, tracker to arm mapping.
- `wuji_teleop/imu_arm_source.py`: glove palm IMU to arm or wrist targets.
- `wuji_teleop/vega_arm_range.py`: Tk joint jog tool.
- `wuji_teleop/sources.py`: glove connection.

All scene coordinates are robot frame: x forward, y left, z up, metres.
