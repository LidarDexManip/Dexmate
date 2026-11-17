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
- Glove: `nmcli device disconnect <glove-port>`, `bash scripts/configure_wuji_network.sh <glove-port>`, then `.venv/bin/python -m wuji_teleop discover --seconds 3`; again after every replug or reboot (`HUMAN_DEMO_CAPTURE.md`, "Connect the gloves"). Always name the port: the gloves sit on the USB Ethernet `enx8cae4cebd67f` (lab profile `Default`, 10.10.x, retakes it on replug), and without a name the script picks the robot port `enp128s31f6` (`robot-lan`), which already holds 192.168.1.50; never disconnect that one. The Wi-Fi `LIDAR_WiFi_5G` is also 192.168.1.0/24, so glove traffic goes to Wi-Fi until the script routes .100/.101 over the glove port. The gloves ignore ARP from senders outside 192.168.1.x: probe them with `arping -s 192.168.1.50 -I <port>`, or a plain probe reads them as absent.
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

Human demonstration capture (no robot) is `scripts/record_human_demo.sh` then `scripts/process_human_demo.sh STAGE SESSION`. `HUMAN_DEMO_CAPTURE.md` is the runbook.

- The laptop logs both gloves (skeleton, fingertip EMF, palm IMU) and every XRoboToolkit tracker/headset frame to `SESSION/pc/`.
- The headset's XRoboToolkit Data Collection (Tracking + Vision; needs PICO enterprise VST camera permission) records the stereo passthrough and the same trackers. `pull` fetches them with adb.
- The trackers stay on the forearms. A recorder-guided calibration (still, then wrists straight while turning the forearms) solves the tracker mounting and the palm-IMU world (AX = YB). After that, the hand's orientation comes from the palm IMU.
- Processing runs in `~/Desktop/retargeting/.venv-vision`, created with `uv venv` because the system Python lacks `python3.12-venv`. Stages: `hands`, `depth` (SGBM+WLS), `view` (viser, also places the object's first pose), `track` (depth ICP, carried by the hand while hidden), `ground` (retargeting's `held_refinement.Problem` with a capsule hand and a moving camera), and `export` (a `retarget.sh` bundle).
- Live view of the headset's stereo passthrough: `scripts/launch_pico_stereo_view.sh [--headset IP]` (`wuji_teleop/demo/headset_stream.py`, `.venv-vision` for the OpenCV window). The stock XRoboToolkit app (v1.1.0+) streams its cameras on request: an `OPEN_CAMERA` message to the app's TCP 13579 makes it connect back to the laptop (TCP 12345) and send side-by-side H.264 (bitrate fixed at about 5 Mbit/s by the app; no capture time, so frames carry their arrival time). It needs the same camera permission and must not run while the headset records. Tested only against a fake headset; the offline pipeline still uses the headset's own recording.
- `--stereo HEADSET_IP` records the live stereo stream on the laptop instead (`stereo_record`, run in `~/Desktop/Fast-FoundationStereo/.venv`: raw H.264 with packet arrival times, left JPEGs, Fast-FoundationStereo disparity PNGs for every frame; default checkpoint 20-30-48 keeps up with 30 fps) after a `headshake` step. Its always-on-top live view shows the ego camera beside depth (metres with `--stereo-baseline-m`, else disparity), the current step and the link status. `scripts/process_human_demo.sh sync SESSION` (`stream_sync`) finds the stream latency from head motion (image vs head-pose rotation) and samples head, trackers, gloves and palm IMUs at each frame's capture time into `processed/synced.npz`. The headset must be awake with the app in front, and no other stereo viewer open (the app serves one camera client at a time). Sync the stand-mounted take `sy_test` (2026-11-16) and later takes with `--camera-offset -0.023 -0.011 -0.074 --baseline-m 0.064`. The 64.0 mm baseline is the headset's factory calibration (`cameraExtrinsics` in line 1 of any `trackingData_*.txt` in the headset's /sdcard/Download, read over adb; 63.97 mm between the two colour cameras; the live stream carries no camera data). The offset (left camera 2.3 cm left, 1.1 cm down, 7.4 cm forward of the head origin, OpenXR head frame) is fitted from the right tracker's puck in the image and depth (3D error 5.9 -> 0.9 cm); the header's camera positions use a device-frame origin about 5 cm off the cameras' midpoint, so they don't give it directly. Without the offset the trackers project about 5 cm beside their pucks. The older 0.083 m estimate assumed the camera at the head origin and is wrong. Depth alignment in `fit_mano` is on hold (`--depth-align off`) until camera, baseline and wrist offset are settled. `sync` pairs trackers to gloves by rotation-speed correlation with the palm IMUs; a headset on a stand needs `--latency 0.062` (measured worn). `scripts/fit_mano.sh SESSION...` (`mano_fit`, FFS venv, needs the user's licensed `~/Downloads/mano_v1_2.zip`) fits MANO per frame (a glove finger at confidence < 0.6 all take, like the right glove's ring, bends with its neighbours instead), then moves each hand onto the stereo depth (`depth_align`; `--depth-align track` by default, since tracker/calibration placement is off by 5-15 cm and 35 degrees and changes through a take; refused for a hand out of view); `process_human_demo.sh view SESSION` opens `hands_view` (MANO meshes or glove keypoints, trackers, camera frustum, depth cloud; hands never in view hidden); `--video` renders it with a headless Chrome client to `processed/hands_view.mp4` beside the camera image. `scripts/view_demo_layers.sh SESSION` (`layer_view`, Tk, `.venv-vision`) is the 2D check: left image and depth side by side, with trackers, wrist, hand orientation, depth, finger skeleton and hand model switched on and off (keys 1-6), any `processed/<name>_<side>.npz` surface as the model. `fit_mano` tracks the depth correction in 0.5 s windows every 0.25 s (1 s was 2% worse on held-out frames, take2).
- Tests: `tests/test_demo.py` runs in either venv; `tests/test_demo_pipeline.py` (synthetic end-to-end) needs `.venv-vision`. No real recording has been processed yet. The headset camera conventions (`--extrinsic`, `--camera-axes`, `--intrinsics`, `--frame-time`) still need checking on the first one.

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
- `wuji_teleop/demo/`: human demonstration capture (`record`, `log`), the live stereo view (`headset_stream`) and offline processing (`headset`, `align`, `hands`, `stereo`, `object_track`, `ground`, `export`, `viewer`, `process`).

All scene coordinates are robot frame: x forward, y left, z up, metres.
