# Wuji glove → Wuji Hand 2 in MuJoCo

This first milestone controls one standalone simulated Hand 2 from one wearable
Wuji glove. The glove side is read from the device. A second glove is optional;
choose one serial number when both are connected. The current left-hand mount
on a fixed Vega 1U is documented in [WUJI_HAND_MUJOCO_RUNBOOK.md](WUJI_HAND_MUJOCO_RUNBOOK.md#current-single-arm-simulation-path--october-8-2026).
Arm motion remains a later milestone.

The application connects only to `DeviceType.WujiGlove`. All finger targets go to
MuJoCo. It does not enable or send motor commands to physical robot hands.

## Install

Run from `/home/isaac/Documents/Dexmate`:

```bash
git submodule update --init third_party/wuji-description
.venv/bin/python -m pip install -r requirements-wuji-sim.txt
.venv/bin/python -m pip check
```

The dependency file pins Wuji SDK `2026.9.22`, MuJoCo `3.15.0`, and NumPy `2.4.6`.
The latter two match the existing simulation environment. The model submodule is
pinned to `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` (v2026.9.11). The selected
models are `hand2/hand2_beta2/body/mjcf/{left,right}.xml`, with a fixed wrist,
20 hinge joints, original position actuators, and the upstream 2 ms timestep.
Keep the model directory and its sibling mesh directory together.

The older `wujihandpy` and separate `wuji-retargeting` projects are archived.
This integration uses the current SDK's `RetargetSession`.

## Check the simulation without hardware

```bash
.venv/bin/python -m wuji_teleop run --input synthetic --side right
.venv/bin/python -m wuji_teleop run --input synthetic --side left
```

The synthetic input is an illustrative finger-curl cycle through the same
landmark → retargeter → actuator path as live tracking. It is not a recording of
a real glove. The viewer shows tracking state, input/control rate, frame age,
and minimum landmark confidence. Space pauses/resumes **tracking** while physics
continues to hold the last target. Close the viewer or press Ctrl-C to exit.

For a finite headless check and a JSON report:

```bash
.venv/bin/python -m wuji_teleop run --input synthetic --side right \
  --headless --duration 30 --report /tmp/wuji-right.json
```

## Connect the wearable glove

The documented glove rig uses an acquisition board, the supplied USB-C cable,
a powered USB-C/Ethernet adapter, and an Ethernet cable to the laptop. This is
not a plain USB connection. Both labeled USB-C ends must have matching
orientation; the manufacturer's guide explains the connector labels.

1. Connect the rig and power adapter. Check the Ethernet link lights.
2. Identify the wired interface and confirm carrier:

   ```bash
   ip -brief address
   cat /sys/class/net/enp3s0/carrier
   ```

   `1` means physical link; `0` means the cable/adapter/power connection still
   needs attention. An IP address cannot fix `NO-CARRIER`.
3. Configure the dedicated wired connection:

   ```bash
   bash scripts/configure_wuji_network.sh enp3s0
   ```

   The script uses NetworkManager to create an in-memory profile with
   `192.168.1.50/24`, no gateway, no default route, and a 3-second duplicate-IP
   check. It refuses to replace an existing IPv4 configuration or matching subnet
   route. Wi-Fi remains the internet connection. If NetworkManager requires
   authentication, run the same script with `sudo` from your local terminal.

   Remove the temporary profile when finished:

   ```bash
   nmcli connection delete wuji-glove-sim-enp3s0
   ```
4. Test the factory addresses and discover the glove:

   ```bash
   ping -c 2 -W 1 192.168.1.100  # left glove, if present
   ping -c 2 -W 1 192.168.1.101  # right glove, if present
   .venv/bin/python -m wuji_teleop discover --seconds 3
   ```

Discovery prints each glove's serial, side, firmware, address, valid skeleton
rate, and latest frame age. It returns a nonzero exit code for missing or stale
tracking. IPs may differ if someone changed the factory configuration.

## Run live

```bash
# One discovered glove: side is automatic.
.venv/bin/python -m wuji_teleop run

# Multiple gloves: copy the desired serial from discovery.
.venv/bin/python -m wuji_teleop run --glove-sn YOUR_SERIAL

# Five-minute validation with recorded run metrics.
.venv/bin/python -m wuji_teleop run --glove-sn YOUR_SERIAL \
  --duration 300 --report /tmp/wuji-live-5min.json
```

Keep fingers naturally relaxed while the glove initializes. Then try an open
hand, gradual closing, each finger separately, and thumb–index pinching. The
input rate depends on glove tracking; retargeting consumes fresh frames at up
to 120 Hz. Physics runs independently at 500 Hz and viewer updates at up to
60 Hz. Slow computation lowers throughput rather than queuing old poses.

Malformed, duplicate, and out-of-order frames cannot refresh the watchdog.
Invalid data or data older than 250 ms causes the **target** to hold; physics
still settles toward that target. A transport error shows `invalid` with the
SDK error; silent loss shows `stale`. Recovery resets retargeter filter state
before processing new tracking. If the SDK does not resume its subscription
after unplugging/replugging, exit and run again. Automatic device reconnection
is not implemented in this milestone.

Confidence is displayed for diagnosis; there is no arbitrary confidence cutoff.
The report's frame age includes estimated transport delay from SDK timestamps
and host monotonic time since receipt. Incorrect clock synchronization produces
stale/invalid status and needs to be resolved before live validation.

## Optional wearer calibration

Start with the SDK's **Default** user and built-in tracking model, as recommended
by the current SDK documentation. Each run temporarily selects Default and
restores the previous SDK user on exit. If tracking fits poorly, create a named
user and calibrate under that user; Default cannot store a calibration.

Use the SDK's documented calibration example, pinned to a revision matching the
installed SDK, or the following equivalent Python workflow. Stop the live
viewer first. Replace `YOUR_SERIAL` and the display name before running:

```python
import asyncio
import time
from wuji_sdk import SdkManager, DeviceType

async def calibrate():
    manager = SdkManager.instance()
    previous = manager.current_user()["user_id"]
    serial = "YOUR_SERIAL"
    gloves = [d for d in manager.scan() if d.device_type == DeviceType.WujiGlove]
    if serial not in [d.sn for d in gloves]:
        raise RuntimeError("Selected serial is not a discovered wearable glove")
    # Reuse a saved user_id on subsequent calibrations instead of creating again.
    user = manager.create_user("Isaac")
    print("Save this user_id:", user["user_id"])
    glove = None
    last_print = 0.0
    def feedback(event):
        nonlocal last_print
        if time.monotonic() - last_print >= 0.5:
            print(event, flush=True)  # includes requested pose and constraints
            last_print = time.monotonic()
    try:
        manager.switch_user(user["user_id"])
        glove = manager.connect(sn=serial, device_name="calibration_glove")
        result = await glove.calibrate(timeout_s=900, on_feedback=feedback)
        print(result)
    finally:
        try:
            if glove is not None:
                manager.disconnect("calibration_glove")
        finally:
            manager.switch_user(previous)

asyncio.run(calibrate())
```

Run a saved script with `.venv/bin/python your_calibration_script.py`, follow the
reported pose prompts, then launch with the saved ID:

```bash
.venv/bin/python -m wuji_teleop run --glove-sn YOUR_SERIAL --user-id YOUR_USER_ID
```

## Verification and acceptance

```bash
.venv/bin/python -m unittest discover -s tests -v
# With a desktop display: also exercise clean GUI shutdown in a subprocess.
WUJI_GUI_TEST=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/launch_vega_1u_mujoco.py --convert-only
```

Automated tests cover both model sides, real SDK retargeting, named actuator
mapping after reordering, limits, malformed/stale/duplicate frames, pause/resume,
SDK failure cleanup, and selecting gloves while excluding robot hands.

Live acceptance requires a connected, worn glove: confirm all five fingers
correspond correctly, demonstrate opening/closing and thumb–index pinching,
pause/resume, unplug Ethernet and observe target hold, then reconnect (restart
if necessary) and complete a five-minute run without MuJoCo warnings. A
synthetic run or mocked disconnect does **not** establish live acceptance.

### Recorded verification — October 7, 2026

- All 19 tests passed with `WUJI_GUI_TEST=1`, including the viewer shutdown
  regression test. Both sides use the installed SDK retargeter in integration tests.
- Right synthetic headless run: 30.000 seconds, 3,600 control updates, 120.0 Hz,
  30.000 seconds of simulated physics, and zero MuJoCo warnings.
- Left synthetic viewer run: 6.005 seconds, 719 updates, 119.7 Hz, zero MuJoCo
  warnings, and clean process exit. A rendered Hand 2 image was visually checked.
- Existing Vega1U conversion still loads 29 bodies, 20 joints, and 19 actuators.
  Its existing mesh/inertia conversion warnings remain.
- `pip check` reports no broken requirements.
- Live acceptance is **pending**: `enp3s0` reports `NO-CARRIER` and SDK discovery
  finds no glove. The network helper correctly stops without changing the network
  when carrier is absent. Successful network activation, real gestures, physical
  disconnect/reconnect, and the five-minute live run remain to be checked.

Implementation components:

- `sources.py`: latest-frame buffer, synthetic input, glove discovery/subscription.
- `retarget.py`: persistent SDK retargeter, pause and stale-input handling.
- `simulation.py`: named actuator mapping, target limits, fixed-palm scene, timing.
- `__main__.py`: discovery and run commands.

`MujocoHand(model, data, side)` resolves joints by name and is reused in the
fixed Vega1U scene. `wuji_teleop.vega_hand_sim` welds Vega joints at their
zero pose, mounts the left Hand 2 at `L_ee`, and reuses the same tracking loop.
The standalone launcher remains available for hand-only checks.

## References

- [Glove connection guide](https://docs.wuji.tech/docs/en/wuji-glove/latest/getting-started/)
- [Glove skeletons, confidence, and timestamps](https://docs.wuji.tech/docs/en/wuji-glove/latest/sdk-data-reference/hand-tracking/)
- [Current SDK retargeting and default-user guidance](https://docs.wuji.tech/docs/en/wuji-sdk/latest/retargeting/)
- [SDK calibration example](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/wuji_glove/5.calibration.py)
- [Pinned hand model release](https://github.com/wuji-technology/wuji-description/tree/c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95)
- [Archived retargeting guide](https://docs.wuji.tech/docs/en/wuji-retargeting/latest/)
