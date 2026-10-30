"""Live view of every link the Vega + Wuji teleop needs, and what is missing.

Read-only by default: routes, pings, the Wuji SDK discovery scan, the DexStream
port and the Pico hand-tracking topic. ``--connect`` also opens and closes an
SDK connection to each Wuji device and a dexcontrol connection to the robot.
That catches a device which answers discovery but refuses connections, so run
it before the teleop, not alongside it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


GLOVES = {"left": "192.168.1.100", "right": "192.168.1.101"}
ROBOT = {"Jetson": "192.168.50.20", "robot SoC": "192.168.50.21"}
DEXSTREAM_PORT = 8091
SIDE_LETTER = {"left": "J", "right": "K"}
SIDES = ("left", "right")
REPO = Path(__file__).resolve().parents[1]
MARK = {"ok": "\033[32m✔\033[0m", "warn": "\033[33m!\033[0m", "fail": "\033[31m✖\033[0m"}
HAND_SUBNET_HINT = ("give the hand an address on a subnet the laptop holds only once "
                    "(e.g. 192.168.77.x like the left hand), or remove the duplicate address")


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: str  # "ok", "warn" or "fail"
    detail: str
    hint: str = ""


def _run(command, timeout=3.0):
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def ping(ip):
    try:
        return subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True,
                              timeout=3).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def route_device(ip, output=None):
    """Interface the kernel would use to reach ``ip``, or None."""
    words = (output if output is not None else _run(["ip", "-4", "route", "get", ip])).split()
    return words[words.index("dev") + 1] if "dev" in words[:-1] else None


def is_wireless(interface):
    return (Path("/sys/class/net") / interface / "wireless").exists()


def route_source(ip, output=None):
    """Local source address the kernel would use to reach ``ip``, or None."""
    words = (output if output is not None else _run(["ip", "-4", "route", "get", ip])).split()
    return words[words.index("src") + 1] if "src" in words[:-1] else None


def ambiguous_hand_source(ip, addresses=None, route_output=None):
    """Why connecting to the Wuji hand at ``ip`` is unsafe, or None.

    When the source address for ``ip`` is on more than one interface, the
    Wuji SDK can connect from another address of the hand's interface; the
    hand cannot answer that, and the failed attempt locks it until a 12 V
    power cycle.
    """
    source = route_source(ip, route_output)
    if source is None:
        return None
    holders = sorted({name for name, address in (
        addresses if addresses is not None else interface_addresses()) if address == source})
    if len(holders) < 2:
        return None
    return (f"laptop {source} is on {' and '.join(holders)}; the SDK may connect from the "
            "wrong address, which locks the hand until a 12 V power cycle")


def interface_addresses(output=None):
    """[(interface, address)] from ``ip -4 -o addr``."""
    text = output if output is not None else _run(["ip", "-4", "-o", "addr", "show"])
    pairs = []
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 4 and words[2] == "inet":
            pairs.append((words[1], words[3].split("/")[0]))
    return pairs


def unconfigured_wired_interfaces(addresses):
    """Wired interfaces with a link but no IPv4 address (glove adapter candidates)."""
    configured = {interface for interface, _ in addresses}
    names = []
    for path in sorted(Path("/sys/class/net").iterdir()):
        if (path / "wireless").exists() or not (path / "device").exists():
            continue
        try:
            carrier = (path / "carrier").read_text().strip() == "1"
        except OSError:
            carrier = False
        if carrier and path.name not in configured:
            names.append(path.name)
    return names


def wifi_ssid():
    for line in _run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list",
                      "--rescan", "no"]).splitlines():
        if line.startswith("yes:"):
            return line[4:]
    return None


def network_checks(addresses):
    checks = []
    wireless = [(name, ip) for name, ip in addresses if is_wireless(name)]
    if wireless:
        name, ip = wireless[0]
        checks.append(Check("Network", "Wi-Fi (Pico)", "ok",
                            f"{wifi_ssid() or '?'}  {ip} on {name}"))
    else:
        checks.append(Check("Network", "Wi-Fi (Pico)", "fail", "no Wi-Fi address",
                            "join the same Wi-Fi network as the Pico"))
    candidates = unconfigured_wired_interfaces(addresses)
    for side, ip in GLOVES.items():
        device = route_device(ip)
        if device and not is_wireless(device):
            checks.append(Check("Network", f"{side} glove route", "ok", f"{ip} via {device}"))
        else:
            adapter = " or ".join(candidates) if candidates else "<glove USB Ethernet interface>"
            checks.append(Check(
                "Network", f"{side} glove route", "fail",
                f"{ip} goes via {device or 'nothing'} (glove network not configured)",
                f"bash scripts/configure_wuji_network.sh {adapter}"))
    # A laptop address on two interfaces only matters for a hand reached through
    # it; device_checks flags that hand (ambiguous_hand_source).
    return checks


def device_checks(scanned):
    """Robot, gloves and hands from pings and the Wuji discovery scan."""
    by_sn = {d["sn"]: d for d in scanned}
    hands = {side: [d for d in scanned if d["type"].endswith("WujiHand2") and
                    d["sn"][3:4] == SIDE_LETTER[side]] for side in SIDES}
    targets = list(ROBOT.values()) + list(GLOVES.values()) + [
        d["ip"] for found in hands.values() for d in found]
    with ThreadPoolExecutor(max_workers=8) as pool:
        alive = dict(zip(targets, pool.map(ping, targets)))
    checks = []
    for name, ip in ROBOT.items():
        checks.append(Check("Robot", name, "ok" if alive[ip] else "fail",
                            f"{ip} {'answers' if alive[ip] else 'no reply'}",
                            "" if alive[ip] else "check robot power and the Ethernet "
                            "link to 192.168.50.x"))
    for side, ip in GLOVES.items():
        serial = next((sn for sn, d in by_sn.items() if d["ip"] == ip), None)
        if alive[ip] and serial:
            checks.append(Check("Gloves", f"{side} glove", "ok", f"{serial} at {ip}"))
        elif alive[ip]:
            checks.append(Check("Gloves", f"{side} glove", "warn",
                                f"{ip} answers ping but not Wuji discovery",
                                "glove may be busy or rebooting; wait or power-cycle it"))
        else:
            checks.append(Check("Gloves", f"{side} glove", "fail", f"{ip} no reply",
                                "turn the glove on; check its adapter and the glove route"))
    for side in SIDES:
        found = hands[side]
        if len(found) == 1:
            hand = found[0]
            ok = alive[hand["ip"]]
            unsafe = ok and ambiguous_hand_source(hand["ip"])
            if unsafe:
                checks.append(Check("Hands", f"{side} hand", "fail",
                                    f"{hand['sn']} at {hand['address']}: {unsafe}",
                                    HAND_SUBNET_HINT))
                continue
            checks.append(Check(
                "Hands", f"{side} hand", "ok" if ok else "fail",
                f"{hand['sn']} at {hand['address']}" + ("" if ok else " (no ping reply)"),
                "" if ok else f"no route to {hand['ip']}: check the hand's port"))
        elif not found:
            checks.append(Check("Hands", f"{side} hand", "fail", "not found by Wuji discovery",
                                "check the hand's 12 V power and its cable to the router"))
        else:
            checks.append(Check("Hands", f"{side} hand", "warn",
                                f"{len(found)} {side} hands found: "
                                + ", ".join(d["sn"] for d in found),
                                f"pass --{side}-hand-sn to the teleop"))
    return checks


def wuji_scan():
    import wuji_sdk

    wuji_sdk.set_log_level("error")
    devices = []
    for d in wuji_sdk.SdkManager.instance().scan():
        address = str(d.address)
        devices.append({"sn": str(d.sn), "type": str(d.device_type), "address": address,
                        "ip": address.rsplit(":", 1)[0]})
    return devices


def wuji_scan_isolated(timeout=20.0):
    """`wuji_scan` in a child process, for callers that must keep running.

    wuji_sdk holds the GIL for the whole scan (~3 s), freezing every other
    thread in the calling process: a teleop's arm loop and its glove streams,
    which then miss heartbeats and are dropped by the SDK.
    """
    code = ("import json\nfrom wuji_teleop.connection_monitor import wuji_scan\n"
            "print(json.dumps(wuji_scan()))")
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Wuji discovery took over {timeout:.0f} s") from None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"Wuji discovery failed: {detail}")
    return json.loads(lines[-1])


def connect_checks(scanned):
    """Open and close a real connection to every Wuji device and the robot."""
    import wuji_sdk

    manager = wuji_sdk.SdkManager.instance()
    checks = []
    addresses = interface_addresses()
    for device in scanned:
        unsafe = (device["type"].endswith("WujiHand2") and
                  ambiguous_hand_source(device["ip"], addresses))
        if unsafe:
            checks.append(Check("Connect", device["sn"], "fail", f"not attempted: {unsafe}",
                                HAND_SUBNET_HINT))
            continue
        alias = f"monitor_{device['sn']}"
        started = time.monotonic()
        try:
            manager.connect(sn=device["sn"], device_name=alias,
                            options=wuji_sdk.ConnectOptions(timeout_ms=3000, retry_count=1))
            manager.disconnect(alias)
            checks.append(Check("Connect", device["sn"], "ok",
                                f"connected in {time.monotonic() - started:.1f} s"))
        except Exception as exc:
            checks.append(Check(
                "Connect", device["sn"], "fail",
                f"answers discovery but refuses connections ({str(exc)[:40]}...)",
                "power-cycle this device; close any other program using it"))
    os.environ.setdefault("ROBOT_NAME", "dm/vgce08fed0a4-1u")
    os.environ.setdefault("ZENOH_CONFIG", str(REPO / "dm_vgce08fed0a4-1u.dzcfg"))
    try:
        from dexcontrol.robot import Robot

        started = time.monotonic()
        robot = Robot(auto_shutdown=False)
        try:
            estop = robot.estop.get_status()
        finally:
            robot.shutdown()
        flags = [name for name, key in (("button", "button_pressed"),
                                        ("software", "software_estop_enabled")) if estop[key]]
        checks.append(Check(
            "Connect", "robot (dexcontrol)", "warn" if flags else "ok",
            f"connected in {time.monotonic() - started:.1f} s; E-stop "
            + (" + ".join(flags) + " ON" if flags else "off"),
            "release the E-stop before teleop" if flags else ""))
    except Exception as exc:
        checks.append(Check("Connect", "robot (dexcontrol)", "fail", str(exc)[:80],
                            "check ROBOT_NAME / ZENOH_CONFIG and the robot link"))
    return checks


class PicoFrames:
    """Counts DexStream ``vr/hands`` frames per side on one Zenoh session."""

    def __init__(self):
        import zenoh

        from .vr_pose_source import PicoHandPoseSource

        self.session = zenoh.open(zenoh.Config())
        self.sources = {side: PicoHandPoseSource(side=side, session=self.session)
                        for side in SIDES}
        self._last = {side: (time.monotonic(), 0) for side in SIDES}

    def checks(self, wifi_ip):
        listening = False
        for host in filter(None, (wifi_ip, "127.0.0.1")):
            try:
                with socket.create_connection((host, DEXSTREAM_PORT), timeout=0.5):
                    listening = True
                    break
            except OSError:
                pass
        page = f"https://{wifi_ip or '<wifi-ip>'}:{DEXSTREAM_PORT}/"
        checks = [Check("Pico", "DexStream", "ok" if listening else "fail",
                        f"serving {page}" if listening else f"nothing on port {DEXSTREAM_PORT}",
                        "" if listening else "start it: scripts/launch_dexstream.sh hand "
                        "(the teleop launcher also starts it)")]
        now = time.monotonic()
        for side, source in self.sources.items():
            snapshot = source.read()
            then, count = self._last[side]
            rate = (snapshot.received - count) / max(1e-3, now - then)
            self._last[side] = (now, snapshot.received)
            frame = snapshot.frame
            age = None if frame is None else now - frame.received_at
            if frame is not None and age < 0.5:
                checks.append(Check("Pico", f"{side} hand tracking", "ok",
                                    f"{rate:.0f} Hz, {age * 1000:.0f} ms old"))
            else:
                checks.append(Check(
                    "Pico", f"{side} hand tracking", "warn",
                    "no frames" if frame is None else f"last frame {age:.1f} s ago",
                    f"on the Pico: open {page}, Enter AR, keep the {side} hand in view"))
        return checks

    def close(self):
        for source in self.sources.values():
            source.close()
        # Closing a Zenoh session can stall ~10 s on this laptop; do not wait for it.
        threading.Thread(target=self.session.close, daemon=True).start()


def render(checks, *, title):
    lines = [f"\033[1m{title}\033[0m", ""]
    group = None
    for check in checks:
        if check.group != group:
            group = check.group
            lines.append(f"\033[1m{group}\033[0m")
        lines.append(f"  {MARK[check.status]} {check.name:<22s} {check.detail}")
        if check.hint and check.status != "ok":
            lines.append(f"      → {check.hint}")
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    lines.append("")
    lines.append("All connections OK." if not failed and not warned else
                 f"{len(failed)} missing, {len(warned)} warning(s).")
    return "\n".join(lines)


def collect(pico, *, connect=False):
    addresses = interface_addresses()
    wifi_ip = next((ip for name, ip in addresses if is_wireless(name)), None)
    try:
        scanned = wuji_scan()
        scan_checks = []
    except Exception as exc:
        scanned = []
        scan_checks = [Check("Hands", "Wuji discovery", "fail", str(exc)[:80])]
    checks = network_checks(addresses) + device_checks(scanned) + scan_checks
    if pico is not None:
        checks += pico.checks(wifi_ip)
    if connect:
        checks += connect_checks(scanned)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument("--connect", action="store_true",
                        help="Also open real connections (implies --once; not during teleop)")
    parser.add_argument("--no-pico", action="store_true", help="Skip the Pico topic check")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args(argv)
    once = args.once or args.connect
    pico = None if args.no_pico else PicoFrames()
    try:
        if once:
            if pico is not None:
                time.sleep(1.5)  # let tracking frames arrive before counting them
            checks = collect(pico, connect=args.connect)
            print(render(checks, title="Vega + Wuji connections"))
            return 1 if any(c.status == "fail" for c in checks) else 0
        while True:
            checks = collect(pico)
            title = (f"Vega + Wuji connections   {time.strftime('%H:%M:%S')}   "
                     f"(every {args.interval:g} s, Ctrl-C to quit)")
            sys.stdout.write("\033[H\033[2J" + render(checks, title=title) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        if pico is not None:
            pico.close()


if __name__ == "__main__":
    raise SystemExit(main())
