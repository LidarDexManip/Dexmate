"""Record the laptop side of a human demonstration: both Wuji gloves and PICO.

Logs every glove skeleton frame (21 keypoints in the glove's wrist frame with
per-joint confidence), the fingertip EMF sensors, the fused palm IMU, and every
XRoboToolkit frame (all motion trackers by serial, the headset pose and PICO
hand tracking), each stamped with host UTC time. It never opens a robot or
robot-hand connection.

The XRoboToolkit app on the headset records the stereo passthrough video, head
pose and the same trackers (Data Collection: Tracking + Vision, Record). The
offline pipeline aligns the two recordings by the trackers seen in both.

Before the demonstration the recorder runs a short wrist calibration, because
the trackers sit on the forearms while the glove skeleton is in the hand's own
frame:

1. ``still``: forearms level and forward, palms down, wrists straight, hold still.
2. ``rotate``: keep each wrist straight (hand in line with the forearm) and turn
   the forearms around: palms up and down, point forward, sideways and up.

With ``--stereo HEADSET_IP`` the laptop also records the headset's live stereo
stream instead of the headset recording it: ``stereo_record`` runs as a child
process in Fast-FoundationStereo's environment and writes ``<session>/stereo/``
(raw stream with packet arrival times, left eye images, disparity of every
frame). Before the wrist calibration a ``headshake`` step turns the head, which
times the stream against the head pose (``process_human_demo.sh sync``).

Run it through ``scripts/record_human_demo.sh``. Ctrl-C ends the recording.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np

from . import log as demo_log

SIDES = ("left", "right")
LEFT_GLOVE_ADDRESS = "192.168.1.100:50001"
RIGHT_GLOVE_ADDRESS = "192.168.1.101:50001"
XR_POLL_HZ = 250.0
REPO = Path(__file__).resolve().parents[2]
DEFAULT_STEREO_PYTHON = Path.home() / "Desktop" / "Fast-FoundationStereo" / ".venv" / "bin" / "python"
HEADSHAKE = ("headshake", 5.0, "Look at a textured scene (not a blank wall) and turn your head left and "
                               "right a few times, then nod. This times the camera stream.")
CALIBRATION = (
    ("still", 3.0, "Forearms level and pointing forward, palms down, wrists straight. Hold still."),
    ("rotate", 10.0, "Keep both wrists straight and turn the forearms around: palms up and down, "
                     "point forward, to the side and up."),
)


def _xyzw(q):
    return (float(q.x), float(q.y), float(q.z), float(q.w))


class LoggedGlove:
    """A GloveSource whose every skeleton, EMF and palm IMU frame is logged."""

    def __init__(self, writer, side, address, user_id, *, sdk=None, connect=None):
        from ..sources import GloveSource

        self.writer = writer
        self.side_index = SIDES.index(side)
        self.frames = 0
        self.last_error = None
        logger = self

        class _Source(GloveSource):
            def _on_frame(self, skeleton):
                logger._log_skeleton(skeleton)
                super()._on_frame(skeleton)

            def _on_emf(self, frame):
                logger._log_emf(frame)
                super()._on_emf(frame)

        factory = connect or _Source
        self.source = factory(None, user_id, sdk=sdk, device_name=f"wuji_demo_glove_{side}",
                              address=address)
        try:
            if self.source.side != side:
                raise RuntimeError(f"glove at {address} reports {self.source.side}, expected {side}")
            self.imu_in_wrist = np.asarray(self.source.palm_imu_mount_rotation(), float)
            self._imu = self.source.palm_imu_resource().subscribe_with_callback(
                self._log_imu, on_error=lambda message: setattr(self, "last_error", str(message)))
        except BaseException:
            self.source.close()
            raise

    def _log_skeleton(self, skeleton):
        try:
            joints = list(skeleton.joints)
            row = np.zeros((), demo_log.GLOVE_DTYPE)
            row["t"] = time.time()
            row["stamp_us"] = int(skeleton.header.timestamp_us)
            row["seq"] = int(skeleton.header.seq)
            row["side"] = self.side_index
            row["keypoints"] = np.asarray([tuple(j.pose.position) for j in joints], np.float32)
            row["orientation"] = np.asarray([_xyzw(j.pose.orientation) for j in joints], np.float32)
            row["confidence"] = np.asarray([float(j.confidence) for j in joints], np.float32)
            self.writer.append("glove", row)
            self.frames += 1
        except Exception as exc:  # never raise into the SDK thread
            self.last_error = f"skeleton: {exc}"

    def _log_emf(self, frame):
        try:
            poses = list(frame.poses)
            row = np.zeros((), demo_log.EMF_DTYPE)
            row["t"] = time.time()
            row["side"] = self.side_index
            row["confidence"] = [float(p.confidence) for p in poses]
            row["position"] = [tuple(float(v) for v in p.pose.position) for p in poses]
            row["orientation"] = [_xyzw(p.pose.orientation) for p in poses]
            self.writer.append("emf", row)
        except Exception as exc:
            self.last_error = f"emf: {exc}"

    def _log_imu(self, frame):
        try:
            row = np.zeros((), demo_log.IMU_DTYPE)
            row["t"] = time.time()
            row["stamp_us"] = int(frame.header.timestamp_us)
            row["side"] = self.side_index
            row["quat"] = _xyzw(frame.orientation)
            a, g = frame.linear_acceleration, frame.angular_velocity
            row["acc"] = (a.x, a.y, a.z)
            row["gyro"] = (g.x, g.y, g.z)
            self.writer.append("imu", row)
        except Exception as exc:
            self.last_error = f"imu: {exc}"

    def meta(self):
        return {"serial": self.source.serial, "side": SIDES[self.side_index],
                "address": self.source.address, "firmware": str(self.source.firmware),
                "user_id": getattr(self.source, "user_id", None),
                "imu_in_wrist_rotation": self.imu_in_wrist.tolist()}

    def close(self):
        try:
            self._imu.close()
        finally:
            self.source.close()


class XrPoller:
    """Poll XRoboToolkit and log each new frame (trackers, headset, PICO hands)."""

    def __init__(self, writer, *, sdk=None):
        if sdk is None:
            import xrobotoolkit_sdk as sdk
        self.sdk = sdk
        self.writer = writer
        self.serials = {}  # serial -> index
        self.frames = 0
        self.last_seen = {}  # serial -> host time
        self.headset_seen = None
        self.last_error = None
        self._last_key = None
        self._stop = threading.Event()
        sdk.init()
        self._thread = threading.Thread(target=self._run, name="demo-xr", daemon=True)
        self._thread.start()

    def _hands(self):
        out = np.zeros((2, 26, 7), np.float32)
        active = np.zeros(2, np.uint8)
        for k, name in enumerate(SIDES):
            try:
                active[k] = bool(getattr(self.sdk, f"get_{name}_hand_is_active")())
                if active[k]:
                    out[k] = np.asarray(getattr(self.sdk, f"get_{name}_hand_tracking_state")(),
                                        np.float32)
            except Exception:
                active[k] = 0
        return out, active

    def poll(self):
        sdk = self.sdk
        count = int(sdk.num_motion_data_available())
        motion_ns = int(sdk.get_motion_timestamp_ns()) if count > 0 else 0
        serials = [str(s) for s in sdk.get_motion_tracker_serial_numbers()][:max(count, 0)]
        raw = np.asarray(sdk.get_motion_tracker_pose(), float).reshape(-1, 7)[:max(count, 0)]
        headset = np.asarray(sdk.get_headset_pose(), float)
        key = (motion_ns, raw.tobytes(), headset.tobytes())
        if key == self._last_key:
            return False
        self._last_key = key
        now = time.time()
        row = np.zeros((), demo_log.XR_DTYPE)
        row["t"] = now
        row["motion_ns"] = motion_ns
        try:
            row["sdk_ns"] = int(sdk.get_time_stamp_ns())
        except Exception:
            row["sdk_ns"] = 0
        row["serial"] = -1
        n = min(len(serials), len(raw), demo_log.MAX_TRACKERS)
        row["count"] = n
        for slot in range(n):
            index = self.serials.setdefault(serials[slot], len(self.serials))
            row["serial"][slot] = index
            row["tracker"][slot] = raw[slot]
            self.last_seen[serials[slot]] = now
        if headset.shape == (7,) and np.any(headset[3:]):
            row["headset"] = headset
            self.headset_seen = now
        row["hand"], row["hand_active"] = self._hands()
        self.writer.append("xr", row)
        self.frames += 1
        return True

    def _run(self):
        period = 1.0 / XR_POLL_HZ
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as exc:
                self.last_error = str(exc)
            self._stop.wait(period)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            self.sdk.close()
        except Exception:
            pass


def running_teleop():
    """PIDs of a teleop or rehearsal that would hold the gloves, if any."""
    out = subprocess.run(["pgrep", "-af", r"python.* -m wuji_teleop\.(hardware_teleop|vega_"
                          r"bimanual_sim|vega_arm_sim)"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


class StereoChild:
    """``stereo_record`` in Fast-FoundationStereo's environment, writing <session>/stereo."""

    def __init__(self, session, headset, python, *, checkpoint="20-30-48", size="1280x480",
                 window=True, baseline_m=None):
        self.dir = Path(session) / "stereo"
        self.headset = headset
        cmd = [str(python), "-m", "wuji_teleop.demo.stereo_record", headset, "--out", str(self.dir),
               "--checkpoint", checkpoint, "--size", size] + (["--window"] if window else []) \
            + (["--baseline-m", str(baseline_m)] if baseline_m else [])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        self.log_path = Path(session) / "stereo.log"
        self._log = open(self.log_path, "w")
        # Own session: Ctrl-C reaches only record.py, which then stops the child in order.
        self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.cmd = cmd

    def status(self):
        try:
            return json.loads((self.dir / "status.json").read_text())
        except (OSError, ValueError):
            return None

    def frames(self):
        s = self.status()
        return s["frames"] if s else 0

    def describe(self):
        code = self.proc.poll()
        if code is not None:
            return f"stereo EXITED ({code}; see {self.log_path})"
        s = self.status()
        if not s:
            return "stereo starting (loading the depth model)"
        if not s["connected"]:
            return "stereo waiting for the headset (app in front, headset awake?)"
        return (f"stereo {s['frames']} frames {s['stream_hz']:.0f} Hz, disparity {s['disparity']}"
                + (f" (backlog {s['backlog']})" if s["backlog"] > 5 else ""))

    def stop(self, timeout=600.0):
        """SIGINT, then wait while it finishes the queued disparity."""
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            end = time.time() + timeout
            while self.proc.poll() is None and time.time() < end:
                s = self.status() or {}
                print(f"\r  stereo: finishing, {s.get('disparity', 0)}/{s.get('frames', 0)} frames with "
                      "disparity", end="", flush=True)
                time.sleep(0.5)
            print()
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(10)
        self._log.close()
        return self.proc.returncode


class Recorder:
    def __init__(self, out, gloves, poller, writer, stereo=None):
        self.out, self.gloves, self.poller, self.writer = Path(out), gloves, poller, writer
        self.stereo = stereo
        self.events = []
        self.started = time.time()

    def event(self, label, **extra):
        item = {"t": time.time(), "label": label, **extra}
        self.events.append(item)
        with open(self.out / "events.jsonl", "a") as handle:
            handle.write(json.dumps(item) + "\n")

    def status(self):
        now = time.time()
        parts = []
        for side, glove in self.gloves.items():
            parts.append(f"{side} glove {glove.frames} frames"
                         + (f" ({glove.last_error})" if glove.last_error else ""))
        live = sorted(s for s, t in self.poller.last_seen.items() if now - t < 0.5)
        parts.append(f"trackers live {len(live)} {live}")
        head = self.poller.headset_seen is not None and now - self.poller.headset_seen < 0.5
        parts.append("headset " + ("ok" if head else "MISSING"))
        if self.poller.last_error:
            parts.append(f"xr error: {self.poller.last_error}")
        if self.stereo is not None:
            parts.append(self.stereo.describe())
        line = "; ".join(parts)
        try:  # the stereo live view shows it
            tmp = self.out / "status.txt.tmp"
            tmp.write_text(line)
            os.replace(tmp, self.out / "status.txt")
        except OSError:
            pass
        return line

    def meta(self):
        return {
            "format": "wuji_demo_pc_v1",
            "host": socket.gethostname(),
            "started_utc": self.started,
            "stopped_utc": time.time(),
            "gloves": {side: glove.meta() for side, glove in self.gloves.items()},
            "tracker_serials": sorted(self.poller.serials, key=self.poller.serials.get),
            "events": self.events,
            "counts": dict(self.writer.counts),
            "stereo": ({"headset": self.stereo.headset, "dir": "stereo", "command": self.stereo.cmd}
                       if self.stereo is not None else None),
            "time_base": "host UTC seconds (time.time()); glove stamp_us is SDK time "
                         "synchronized to host UTC",
            "xr_axes": "raw OpenXR (x right, y up, z toward the user), x y z qx qy qz qw",
        }


def _countdown(recorder, label, seconds, text, stop):
    print(f"\n[{label}] {text}")
    recorder.event(f"{label}_start", text=text, seconds=seconds)
    end = time.time() + seconds
    while not stop.is_set() and time.time() < end:
        print(f"\r  {end - time.time():4.1f} s  | {recorder.status()}", end="", flush=True)
        stop.wait(0.2)
    recorder.event(f"{label}_end")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="New session directory "
                        "(default ~/vega-demos/<time>)")
    parser.add_argument("--hands", choices=("both", "left", "right", "none"), default="both")
    parser.add_argument("--left-glove-address", default=LEFT_GLOVE_ADDRESS)
    parser.add_argument("--right-glove-address", default=RIGHT_GLOVE_ADDRESS)
    parser.add_argument("--user-id", default="vega-operator",
                        help="Wuji SDK user for the glove hand models ('' = Default user)")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="No wrist calibration (the processor then needs --calibration-from)")
    g = parser.add_argument_group("live stereo (instead of the headset's own recording)")
    g.add_argument("--stereo", metavar="HEADSET_IP",
                   help="Also record the headset's stereo stream with Fast-FoundationStereo disparity")
    g.add_argument("--stereo-python", type=Path, default=DEFAULT_STEREO_PYTHON,
                   help="Python of the Fast-FoundationStereo environment")
    g.add_argument("--stereo-checkpoint", default="20-30-48",
                   help="Fast-FoundationStereo checkpoint: 20-30-48 keeps up with 30 fps; 23-36-37 is "
                        "more accurate but queues frames (finished after Ctrl-C)")
    g.add_argument("--stereo-baseline-m", type=float,
                   help="Measured stereo baseline (m): the live view then shows depth in metres")
    g.add_argument("--stereo-size", default="1280x480", help="Side-by-side stream size, both eyes")
    g.add_argument("--no-stereo-window", action="store_true",
                   help="No live window (ego camera | depth, with the current step and link status)")
    args = parser.parse_args(argv)
    if args.stereo and not Path(args.stereo_python).exists():
        sys.exit(f"{args.stereo_python} not found; set up ~/Desktop/Fast-FoundationStereo "
                 "(see HUMAN_DEMO_CAPTURE.md) or pass --stereo-python")

    busy = running_teleop()
    if busy:
        sys.exit("A teleop or rehearsal is running and holds the gloves; close it first:\n  "
                 + "\n  ".join(busy))
    out = args.out or Path.home() / "vega-demos" / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out.expanduser().resolve()
    if out.exists():
        sys.exit(f"{out} exists; choose a new session directory")
    pc = out / "pc"
    writer = demo_log.StreamWriter(pc)

    import wuji_sdk

    from ..connection_monitor import ping

    wuji_sdk.set_log_level("warn")
    sides = {"both": SIDES, "none": ()}.get(args.hands, (args.hands,))
    gloves = {}
    poller = None
    stereo = None
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        for side in sides:
            address = args.left_glove_address if side == "left" else args.right_glove_address
            ip = address.rsplit(":", 1)[0]
            if not (ping(ip) or ping(ip)):
                raise RuntimeError(f"{side} glove {address} does not answer ping "
                                   "(off, asleep, or scripts/configure_wuji_network.sh not run?)")
            print(f"Connecting the {side} glove at {address} ...", flush=True)
            gloves[side] = LoggedGlove(writer, side, address, args.user_id, sdk=wuji_sdk)
        poller = XrPoller(writer)
        if args.stereo:
            stereo = StereoChild(out, args.stereo, args.stereo_python, checkpoint=args.stereo_checkpoint,
                                 size=args.stereo_size, window=not args.no_stereo_window,
                                 baseline_m=args.stereo_baseline_m)
        recorder = Recorder(pc, gloves, poller, writer, stereo)
        recorder.event("connected")
        print(f"Recording to {out}")
        if stereo is not None:
            print("On the headset (XRoboToolkit, app in front, headset awake): Head tracking on, Pico "
                  "Motion Tracker tab -> object tracking -> Send. Do NOT press Data Collection Record: "
                  "the laptop records the cameras.")
        else:
            print("On the headset (XRoboToolkit): Head tracking on, Pico Motion Tracker tab -> object "
                  "tracking -> Send, then Data Collection: Tracking + Vision -> Record.")
        print("Waiting for the trackers and headset ...", flush=True)
        while not stop.is_set():
            live = [s for s, t in poller.last_seen.items() if time.time() - t < 0.5]
            head = poller.headset_seen is not None and time.time() - poller.headset_seen < 0.5
            print(f"\r  {recorder.status()}", end="", flush=True)
            if stereo is not None and stereo.proc.poll() is not None:
                raise RuntimeError(f"the stereo recorder exited; see {stereo.log_path}")
            if (len(live) >= len(sides) and head and all(g.frames for g in gloves.values())
                    and (stereo is None or stereo.frames() > 0)):
                break
            stop.wait(0.3)
        print()
        steps = ([HEADSHAKE] if stereo is not None else []) + \
            ([] if args.skip_calibration or not sides else list(CALIBRATION))
        if steps:
            for label, seconds, text in steps:
                if stop.is_set():
                    break
                _countdown(recorder, label, seconds, text, stop)
        if not stop.is_set():
            print("\nDemonstrate now. Ctrl-C stops the recording"
                  + ("." if stereo is not None else " (stop the headset Record first)."))
            recorder.event("demo_start")
            while not stop.is_set():
                print(f"\r  {time.time() - recorder.events[-1]['t']:6.1f} s | {recorder.status()}",
                      end="", flush=True)
                stop.wait(0.5)
            print()
            recorder.event("demo_end")
        if stereo is not None:
            code = stereo.stop()
            recorder.event("stereo_stopped", returncode=code)
        (pc / "meta.json").write_text(json.dumps(recorder.meta(), indent=2))
        print(f"Saved {out}  ({dict(writer.counts)}"
              + (f", stereo {stereo.frames()} frames" if stereo is not None else "") + ")")
        if stereo is not None:
            print(f"Next: scripts/process_human_demo.sh sync {out}")
    finally:
        if stereo is not None and stereo.proc.poll() is None:
            stereo.stop(timeout=60)
        if poller is not None:
            poller.close()
        for glove in gloves.values():
            try:
                glove.close()
            except Exception as exc:
                print(f"glove close: {exc}", file=sys.stderr)
        writer.close()


if __name__ == "__main__":
    main()
