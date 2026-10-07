# Dexmate Vega-1U Setup Handoff

Date: 2026-10-01

## Hardware and scope

- Robot identified by hostname as `vega-1u`.
- This is the top-half-only Vega-1U configuration; there is no base hardware.
- The Vega-1 Pro Jetson Nano instructions using the `dexmate-nano` account do not apply.
- Relevant sensor: `head_camera`.
- Do not enable `base_*` cameras or lidar sensors unless the hardware documentation confirms they are physically present.

## Laptop-to-robot Ethernet connection

The laptop Ethernet interface is `enp3s0`.

Laptop static IPv4 configuration:

```text
Address: 192.168.50.10
Netmask: 255.255.255.0
Gateway: blank
DNS: blank
```

Robot SSH endpoint:

```text
Host: 192.168.50.20
Username: dexmate
```

The robot was confirmed reachable:

```bash
ping -c 5 -W 2 192.168.50.20
```

Result: 5 packets received, 0% packet loss.

SSH port 22 was open, and this connection succeeded:

```bash
ssh dexmate@192.168.50.20
```

Do not store the Jetson password in this file. Enter it directly into the terminal when prompted.

## Display diagnosis

The Jetson initially showed a black screen, but SSH and networking worked.

The following showed that the graphical login manager was not running:

```bash
systemctl status display-manager --no-pager
systemctl get-default
systemctl is-enabled gdm
systemctl status graphical.target --no-pager
```

Important results:

- `gdm.service`: inactive (dead)
- Default target: `multi-user.target`
- `graphical.target`: inactive

The monitor came up after starting GDM manually:

```bash
sudo systemctl start gdm
```

This established that the display hardware was working. Do not enable GDM permanently unless the robot documentation requires a graphical desktop; the robot image may intentionally be headless.

## Dexsensor packages

Files downloaded into `/home/isaac/Documents/Dexmate`:

```text
firmware_mcu_v0.5.2.dpfw
socbridgepkg_v0.18.8.dpsw
dexsensor_0.7.6_amd64.deb
dexsensor_0.7.6_arm64.deb
dexsensor-thor_0.7.6_arm64.deb
```

The laptop is `amd64`. The Jetson reported `aarch64`, so the correct package for the Jetson is:

```text
dexsensor_0.7.6_arm64.deb
```

The `dexsensor-thor` package should not be installed unless the documentation identifies the target computer as Thor.

The host laptop package was installed and verified:

```text
Package: dexsensor
Version: 0.7.6
Architecture: amd64
Status: install ok installed
```

## Installing Dexsensor on the Jetson

The first SCP attempt was made from inside the Jetson SSH session and failed because the source file was not on the Jetson. The corrected transfer must be run from the laptop prompt:

```bash
exit
scp ~/Documents/Dexmate/dexsensor_0.7.6_arm64.deb \
  dexmate@192.168.50.20:/home/dexmate/
```

Then connect to the Jetson and install:

```bash
ssh dexmate@192.168.50.20
sudo apt install -y ./dexsensor_0.7.6_arm64.deb
```

Installation succeeded on the Jetson:

```text
Dexsensor 0.7.6 installed successfully
Architecture: arm64
```

Install and start the service:

```bash
sudo dexsensor install
```

Verify it:

```bash
systemctl status dexsensor.service --no-pager
dexsensor status
```

Successful result:

- `dexsensor.service`: active (running)
- Service enabled at boot
- Zenoh service communication active
- All configured sensors detected
- No sensors enabled by default

## Sensor test

The head camera was enabled with:

```bash
dexsensor enable head_camera
dexsensor status
```

Successful result:

```text
head_camera  zed_x_camera  Running
```

This confirms the head camera and Dexsensor communication are working. Do not enable every sensor at once.

## Firmware and certificate status

Firmware has not been flashed yet.

Staged files:

```text
firmware_mcu_v0.5.2.dpfw
socbridgepkg_v0.18.8.dpsw
```

The manual says to use Method 1, the SocBridge web portal, for certificate setup. It also says to update the dashboard first. For firmware version 0.5.0 or newer, the documented order is:

1. Upload and install the robot-server (`firmware_soc`) package first. This is separate from the SocBridge dashboard package.
2. Restart the robot.
3. Reopen the portal.
4. Upload and install the firmware package.

Do not assume file roles in the portal. Confirm which upload field accepts `.dpsw` and `.dpfw` before installing.

The portal is expected at:

```text
https://192.168.50.21:57832
```

The manual requires an SSH tunnel from the laptop:

```bash
sudo apt install -y sshuttle
sshuttle -r dexmate@192.168.50.20 192.168.50.0/24
```

At the initial handoff, `sshuttle` was not installed. The continuation check on 2026-10-01 confirmed that version `1.0.5-1ubuntu4` is now installed on the laptop. Enter passwords directly in the terminal or portal; do not add them to this file or send them in chat.

## Continuation checks — 2026-10-01

- Laptop Ethernet remains `192.168.50.10/24` on `enp3s0`.
- The SoC portal is reachable from the laptop without starting another tunnel. An HTTPS GET redirects to `/setup`, which returns HTTP 200 and the initial password creation page.
- This HTTP check used `curl -k` for reachability only; it did not validate the portal TLS certificate. The portal TLS certificate is separate from the Dexmate communication certificate.
- The portal asks for a password of at least eight characters and presents recovery codes that must be saved privately. No password was created during these checks.
- Laptop packages verified: `dexsensor 0.7.6 amd64` and `sshuttle 1.0.5-1ubuntu4`.
- All five staged packages listed above are present. Their compatibility and latest release status have not yet been verified through the authenticated portal.
- SSH port responds, but a noninteractive login fails with `Permission denied (publickey,password)`. Jetson package and service results above remain the prior session's results.
- Laptop `sudo` requires interactive authentication.
- `dextop`, `dexcontrol`, and `dexcomm` are absent from the laptop's system Python environment; other Python environments and the Jetson have not been checked.
- No firmware, dashboard, certificate, or robot motion changes were made during these checks.
- The user is unsure whether the communication certificate has been supplied. A filename-only search of `~/Downloads` and this workspace found no obvious communication certificate bundle; `~/.dexmate` is absent on the laptop. This does not establish whether a certificate is already installed on the robot. A portal recovery-code download was found; its contents were not read or copied. Recovery codes are separate from the communication certificate.

## Pause point for next session

- The SocBridge portal is the correct page for continuing setup: `https://192.168.50.21:57832/`.
- The portal is reachable through the working SSH tunnel. Start it from the laptop if needed:
  ```bash
  export ROBOT_IP=192.168.50.20
  sshuttle -r dexmate@$ROBOT_IP 192.168.50.0/24
  ```
- The portal now accepts the communication configuration; the prior `Zenoh configuration missing` block is cleared.
- The matching `.dzcfg` file is `~/Downloads/dm_vgce08fed0a4-1u.dzcfg`.
- The Jetson's control API is installed in `/home/dexmate/miniconda3` (`dexcontrol 0.5.0`). The matching file `dm_vgce08fed0a4-1u.dzcfg` was found in `~/Downloads` and copied to `/home/dexmate/.dexmate/comm/zenoh/` with owner-only permissions. With `ROBOT_NAME=dm/vgce08fed0a4-1u` and `ZENOH_CONFIG` set to that file, `Robot()` connected successfully, initialized head/arms/estop/heartbeat, reported all motors and sensors active, read head joint positions, and shut down cleanly without motion.
- The robot-specific `.dzcfg` has been uploaded through the portal and copied to the Jetson at `/home/dexmate/.dexmate/comm/zenoh/dm_vgce08fed0a4-1u.dzcfg`.
- Do not substitute `socbridgepkg_v0.18.8.dpsw` or `firmware_mcu_v0.5.2.dpfw` for the `.dzcfg` file.
- Do not flash firmware until the portal's current firmware version and update order have been confirmed.
- The portal dashboard reports version `0.18.8` as up to date; the staged `firmware_mcu_v0.5.2.dpfw` has not been installed yet.
- A firmware `v0.5.2` install was attempted on 2026-10-02 and failed at 33% while updating `ARM`. Post-failure verification succeeded: all components remained connected, head positions were readable, and the robot shut down cleanly. Reported versions remain robot-server/firmware `0.5.1`, ARM `1342`, SoC `1320`, and torso `206`; do not retry the firmware blindly.

## Updated todo list

- [x] Establish laptop Ethernet and robot SSH connectivity (prior session).
- [x] Install Dexsensor 0.7.6 on laptop and Jetson (prior session; laptop rechecked).
- [x] Verify `head_camera` running (prior session).
- [x] Install `sshuttle` on the laptop.
- [x] Confirm the SoC portal is reachable from the laptop.
- [ ] Open `https://192.168.50.21:57832/setup`, create a private portal password, save the recovery codes, and log in.
- [ ] Record current dashboard, robot-server, and firmware versions; confirm Vega-1U package compatibility and the roles of the `.dpsw` and `.dpfw` files.
- [ ] Update the dashboard first using its Check for Updates workflow, or the documented upload fallback.
- [ ] Confirm availability of the robot-specific Dexmate communication certificate and install it through the communication window if needed.
- [ ] For firmware >=0.5.0, install the required robot-server package first, restart the robot as directed, reopen the portal, then install the firmware. Do not assume the dashboard package alone completes the robot-server prerequisite.
- [ ] Verify firmware information, communication topics, and robot health using `dextop firmware info`, `dextop topic list --timeout 10`, and `dextop doctor check` in a configured environment.
- [ ] Check installed Python environments and compatible `dextop`, `dexcomm`, and `dexcontrol` versions on the intended development machine; complete the SDK setup.
- [ ] Recheck the Dexsensor service and head camera after updates.

If portal connectivity is lost, the documented laptop tunnel command is:

```bash
sshuttle -r dexmate@192.168.50.20 192.168.50.0/24
```

Leave that terminal open while using the tunnel. It requires interactive authentication.

Do not run a general `apt upgrade` on the Jetson as part of this setup. The Jetson reported many available updates, but broad OS updates should be coordinated with Dexmate documentation.

## Dance diagnosis — 2026-10-02

- User reports both 2.0x and 0.5x dance attempts stopped during pre-positioning with `TimeoutError: Multi-component motion timed out waiting for 'right_arm'`. A prior right-arm diagnostic reportedly passed; its code and log were not available in this session.
- Confirmed repository: `/home/dexmate/dexcontrol`, commit `31f3362`, initially clean. Dance wrapper: `examples/advanced_examples/vega_dance.sh`; recording: `data/vega-1_dance.npz`; playback script: `replay_trajectory.py` in that directory.
- Installed Python is `/home/dexmate/miniconda3/bin/python`. Installed packages: dexcontrol 0.5.0, dexcomm 0.6.0, dexmotion 0.4.1, dexbot-utils 0.5.2. The executing SDK comes from site-packages, not directly from the example repository.
- Live version query still reports server/firmware 0.5.1, ARM 1342, SoC 1320, torso 206. The manufacturer lists dexcontrol 0.5.x as compatible with firmware 0.5.x: https://github.com/dexmate-ai/dexcontrol . This alone does not establish compatibility of every motion feature.
- **The dance speed flag does not slow initial pre-positioning.** It resamples playback. The folded-arms/head move calls `move_to_joint_pos` without a velocity scale and waits five seconds total. The subsequent move to the recording's first frame waits three seconds total. Neither deadline changes with `--speed`.
- `MultiMotionHandle.wait` shares one deadline among all component handles. A timeout naming right_arm identifies the handle being awaited, not a diagnosed hardware fault. A wait timeout does not itself cancel motion. The example lacks a `try/finally` around execution and does not check for a returned `cancelled` state.
- Read-only checks found both arms connected/enabled, fresh position samples, and zero reported velocity. Joint error codes are **unavailable** in the received state; do not describe the robot as fault-free. Brake-release queries returned `enabled: false`, `joints: []`; this is the special over-limit drag setting, not a general diagnosis of physical brake engagement.
- Right-arm folded target is `[-1.57079, 0, 0, -3.1, 0, 0, 0.69813]`. SDK clips joint 4 to its configured lower limit, -3.071 rad. At the 14:39 America/New_York read-only check, joint 6 was approximately 0.27830 rad (target 0), joint 7 approximately 0.39723 rad (target 0.69813). Those residuals are about -15.95 and +17.24 degrees. They do not reveal why the earlier motion failed.
- At 0.5x with the wrapper's smoothing, playback lasts 98.476 seconds. The recording includes torso commands, which are skipped on this Vega-1U. Configured limits are exceeded by L_arm_j4 (up to 0.02909 rad), L_arm_j7 (0.11881), R_arm_j4 (0.03444), and R_arm_j6 (0.10421). Clipping changes the intended trajectory; joint-limit compliance alone does not establish collision clearance. Review the recording before retrying the full dance.
- User confirms no hands or grippers are attached and is unsure how the right arm moved during the failed attempts.
- Prepared `diagnostics/right_arm_check.py`, also copied to `/home/dexmate/dexmate_diagnostics/right_arm_check.py`. Default execution reads state and shuts down. `--test-folded` offers an operator-confirmed right-arm-only move with velocity_scale 0.1 and a 20-second deadline. It logs position residuals and motion-handle status, requests cancellation on timeout/interruption, and always calls SDK shutdown after successful initialization. It preserves SDK limit clipping and does not implement collision checking.
- Verified monitor behavior without hardware for finished/cancelled/error states, timeout cancellation, and interruption cancellation. Ran default read-only mode successfully on the robot. Evidence: `diagnostics/right-arm-readonly-20261002.jsonl` (also on the robot under `dexmate_diagnostics`). No movement command was issued during these checks.
- Operator subsequently confirmed readiness and authorized one right-arm test. At 14:46:57 America/New_York, sent the folded target at velocity_scale 0.1. All 81 samples over 20 seconds remained in the local default `accepted` state; each joint's peak-to-peak position variation was below 0.007 degrees. Cancellation was requested at the deadline and SDK shutdown completed. Evidence: `diagnostics/right-arm-folded-20261002.jsonl`.
- **Immediate blocker identified:** a command-free publisher matching check found one subscriber session each on `dm/vgce08fed0a4-1u/control/arm/left`, `control/arm/right`, and `control/head`, but zero subscribers on the corresponding `motion/target/arm/left`, `motion/target/arm/right`, and `motion/target/head` topics. The pre-position API uses those motion-target topics. A missing, stopped, misconfigured, or unreachable server-side motion plugin is the leading explanation; this check does not establish which of those applies.
- **Misleading right-arm error reproduced without robot motion:** a standalone synthetic `MotionHandle.wait(0.05)` with no status raises `PluginNotAvailableError`. Combining two such handles in left/right order yields `TimeoutError: Multi-component motion timed out waiting for 'right_arm'`. The combined wait defers the missing-plugin exception from the left arm, exhausts its shared deadline, then replaces that diagnosis with the right-arm timeout. The original message is not evidence of a right-arm-specific fault.
- Post-test component query still showed both arms and head connected/enabled and e-stop connected. No further movement was commanded. `dextop topic list --timeout 10` found no discovery keys even though direct publisher matching and state queries work; do not use that discovery result as proof of total communication failure.
- Updated the helper to refuse a motion test when no target subscriber is found and to preserve the SDK's missing-plugin error at a deadline. The original test log predates these improvements.
- Next required step: inspect the robot-server motion-plugin installation, running state, namespace, and routing. The SoC portal at `https://192.168.50.21:57832/` redirects an unauthenticated request to its login page; this session has no authenticated portal access. No SDK, server, or firmware upgrade was performed, and no error was cleared or controller bypassed. Do not retry the dance merely by increasing its timeout.

## Post-restart J3 retry check — 2026-10-02

- User restarted the robot and laptop and requested another J3 move attempt. SSH access was restored through the user's password-authenticated multiplexed connection at `/home/isaac/.ssh/dexmate-control`.
- Ran `right_arm_check.py` in its default state-reading mode. Robot uptime was approximately two minutes when reconnecting. Right arm reported connected/enabled with fresh state samples and zero reported velocity. Joint error codes remain unavailable.
- The right-arm motion-target publisher still reports `has_subscribers: false`, `subscriber_sessions: 0` for `dm/vgce08fed0a4-1u/motion/target/arm/right`. No move was published. Versions remain server/firmware 0.5.1, ARM 1342, SoC 1320, torso 206.
- Evidence saved locally and remotely as `diagnostics/right-arm-restart-check-20261002.jsonl` (remote directory `/home/dexmate/dexmate_diagnostics`).
- Jetson system/user service listings exposed no motion/robot-server service; dexsensor is running. Docker inspection requires sudo authentication, and `sudo -n` reports that a password is required. This does not establish whether containers contain a motion service. SoC portal still requires login.
- Manufacturer changelog confirms these move APIs require the motion plugin on robot-server: https://raw.githubusercontent.com/dexmate-ai/dexcontrol/main/CHANGELOG.md . Need authenticated server/plugin status or logs to distinguish missing, stopped, misconfigured, or unreachable plugin.
- Existing `right_arm_joint_loop.py` commands J3, J5, and J7 together; it is not a J3-only test. Do not run that loop to satisfy the current single-joint request.

## Robot-server download located — 2026-10-03

- Current software page: https://software.dexmate.ai/packages/vega-firmware . Its public release API at https://software.dexmate.ai/api/releases/vega-firmware lists only `firmware_mcu_v0.5.2.dpfw` under 0.5.2; the separate server package is under 0.5.1.
- Downloaded `firmware_soc_v0.5.1_x86_64.dpfw` into this workspace from https://download.dexmate.ai/vega-firmware/v0.5.1/firmware_soc_v0.5.1_x86_64.dpfw . File size is 83,285,975 bytes, matching the official listing. Local SHA-256: `cef20c0e0aa86e91f0fea2c08b188adec6964e3a81a21f36cc8374d6e7c3ec70`. No publisher checksum was supplied by that listing; the hash is a local reference only.
- Official 0.5.0 release notes identify the split into `firmware_soc` and `firmware_mcu` and require SOC first. Official 0.5.1 notes describe a SOC-only update. Sources: https://software.dexmate.ai/api/releases/vega-firmware/0.5.0/notes and https://software.dexmate.ai/api/releases/vega-firmware/0.5.1/notes . No 0.5.2 notes were available at the corresponding endpoint (404).
- `socbridgepkg_v0.18.8.dpsw` is the dashboard update, not the robot-server package. The server package is a portal upload, not an ARM64 Debian package to install on the Jetson. Target compatibility has not been checked in the authenticated robot portal.
- User provided installation instructions: install robot-server first, restart the robot, then reopen the portal and install firmware. Reference: https://platform.dexmate.ai/hub/wiki/software-setup-and-updates?product=vega-1 . The wiki content requires client rendering/authenticated access and was not independently read this session.
- Download only: no package was uploaded or installed, and the robot was not restarted. The prior failed MCU update and missing motion-plugin subscriber remain unresolved.
