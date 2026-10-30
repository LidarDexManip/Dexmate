# New machine: glove + Pico headset simulation

From a fresh Ubuntu laptop to both Vega-1U arms moving in MuJoCo under two Wuji gloves and a Pico headset. Simulation only; nothing here commands the physical robot.

## 1. Install (once)

```bash
sudo apt install git curl python3.11 python3.11-venv network-manager
git clone --recurse-submodules https://github.com/LidarDexManip/Dexmate.git ~/Documents/Dexmate
cd ~/Documents/Dexmate
scripts/setup_sim.sh
```

`setup_sim.sh` fetches the needed submodules and creates `.venv`. It installs the pinned Python packages and DexStream (the Pico web server), plus the three.js file the Pico page needs (Node.js isn't required). It then runs the tests. It ends with `Setup complete`. Rerun it any time; it skips finished steps.

Check it without hardware:

```bash
scripts/launch_vega_joint_limits.sh
```

A jog window and the MuJoCo viewer open. Every launcher appends what it ran to `SIM_RUN_LOG.txt`.

## 2. Glove networking

The gloves normally use `192.168.1.100` (left) and `192.168.1.101` (right), and need the laptop at `192.168.1.50` on the Ethernet port.

1. Glove acquisition board -> supplied USB-C cable -> **powered** USB-C/Ethernet adapter -> laptop Ethernet.
2. Configure and check:

   ```bash
   bash scripts/configure_wuji_network.sh            # picks the wired port with a link
   .venv/bin/python -m wuji_teleop discover --seconds 3
   ```

   The script makes a temporary profile, removed on reboot, and never changes the internet route.
   If Wi-Fi is also on `192.168.1.x` (usual with the Pico router), it routes only the two glove addresses over Ethernet so the Pico stays on Wi-Fi.
   If the port already has an address from another profile, it tells you to run `nmcli device disconnect <port>` first.
3. `discover` should print one left glove. The launchers use it automatically. With several gloves, pass the serial as an argument or set `GLOVE_SN`.

## 3. Pico headset

1. Join the Pico and the laptop to the same Wi-Fi.
2. Start a Pico launcher (below). It prints `Pico page: https://<wifi-ip>:8091/`.
3. In the Pico browser, open that URL (the main page, not `/preview`) and accept the certificate warning.
4. Put both controllers down so the headset switches to hand tracking, then press **Enter AR**.

The launchers find the Wi-Fi address and DexStream on their own. Override with `WIFI_INTERFACE`, a Wi-Fi IP argument, or `DEXSTREAM_BIN`.

## 4. Launchers

| What | Command | Needs |
| --- | --- | --- |
| Joint limit tool | `scripts/launch_vega_joint_limits.sh` | nothing |
| Hand only (fingers, arm fixed) | `scripts/launch_wuji_vega_hand_only.sh` | glove |
| Glove fingers + wrist | `scripts/launch_wuji_vega_glove_wrist.sh` | glove |
| Glove fingers + whole arm (IMU) | `scripts/launch_wuji_vega_glove_arm.sh` | glove |
| **Hand + Pico** (arm from headset, wrist and fingers from glove) | `scripts/launch_wuji_vega_pico_hand.sh` | glove + Pico |
| **Both hands + Pico** (both arms, wrists, and hands) | `scripts/launch_wuji_vega_bimanual_pico.sh` | two gloves + Pico |
| **Wrist trackers** (both arms from two PICO Motion Trackers, GMR-style IK; gloves for fingers) | `scripts/launch_wuji_vega_bimanual_trackers.sh` | two trackers + Pico + XRoboToolkit PC Service (+ two gloves) |
| DexStream only (keeps the Pico page up) | `scripts/launch_dexstream.sh` | Pico |

**Both hands + Pico steps:**

1. Start `scripts/launch_wuji_vega_bimanual_pico.sh`.
2. Open the Pico page and enter AR with the controllers down.
3. Hold both gloves still for one second.
4. Wait for both `L` and `R` status entries to show tracking.
5. Hold both hands where you want to work and press **R** in the MuJoCo window.

Each run logs raw/filtered Pico XYZ, target, achieved pose, workspace clipping,
and joints to `/tmp/vega-pico-hand-<time>.csv`. Translation scale defaults to
`0.6`; set `PICO_SCALE` to tune it.

To restart the sim without dropping the Pico page:

1. Run `scripts/launch_dexstream.sh` in one terminal and leave it.
2. Run `scripts/launch_wuji_vega_bimanual_pico.sh` in another. It sees DexStream already running and starts only the sim.

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No wired interface with a link` | Adapter not powered or cable out; check `cat /sys/class/net/<port>/carrier` is `1`. |
| Glove not found / `ping 192.168.1.100` fails | Rerun `bash scripts/configure_wuji_network.sh`; disconnect any other profile on the port first. |
| Pico page does not load | DexStream not running or wrong IP: run `scripts/launch_dexstream.sh` and use the URL it prints. |
| Status stays `hand waiting` | Headset is not tracking the hand: enter AR, controllers down, hand in view. |
| Arm offset or too high | Hold your hand in your working position and press **R**. |
| `DexStream not found` | Rerun `scripts/setup_sim.sh`, or set `DEXSTREAM_BIN`. |

## Browser end-effector dashboard (no Pico)

Install `requirements-wuji-web.txt`, run `scripts/launch_vega_ee_preview.sh`, and
open http://127.0.0.1:8088 on the laptop. Simulation is the default. The full-body
preview, device status, keyboard jogging and mouse-drag target pads run in the
browser. See [EE_PREVIEW_RUNBOOK.md](EE_PREVIEW_RUNBOOK.md) before connecting gloves
or using the explicit hardware mode.
