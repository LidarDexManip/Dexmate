"""Vega ego (head) and wrist camera views for the hardware teleop dashboard.

dexsensor on the robot publishes the cameras once enabled there (on the Jetson:
``dexsensor enable head_camera``, ``dexsensor enable right_wrist_camera``). A
child process subscribes and decodes them, so full-rate JPEG decoding never
competes with the teleop's 100 Hz arm loop, and pipes downscaled PPM frames to
the dashboard at ``FRAME_HZ``. A camera counts as detected while its frames
are fresh.
"""

from __future__ import annotations

import collections
import functools
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# (dashboard name, dexsensor topic under the robot namespace, Jetson sensor name)
CAMERAS = (
    ("Ego camera", "sensors/head_camera/left_rgb", "head_camera"),
    ("Right wrist camera", "sensors/right_wrist_camera/rgb", "right_wrist_camera"),
)
VIEW_WIDTH = 480  # pixels; frames are downscaled by an integer stride to at most this
FRAME_HZ = 10.0
STALE_S = 1.5  # a camera without a frame for this long is shown as not detected
RESTART_S = 5.0
_HEADER = struct.Struct("<BHHI")  # camera index, width, height, PPM byte count
REPO = Path(__file__).resolve().parents[1]


def downscale(rgb, width=VIEW_WIDTH):
    step = max(1, -(-rgb.shape[1] // width))
    return np.ascontiguousarray(rgb[::step, ::step, :3], dtype=np.uint8)


def to_ppm(rgb):
    height, width = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (width, height) + rgb.tobytes()


def pump(latest, out, sent, decode):
    """Send each camera's newest frame once; returns how many were sent.

    ``latest`` maps camera index to (arrival count, raw message bytes).
    """
    count = 0
    for index, (arrivals, payload) in list(latest.items()):
        if sent.get(index) == arrivals:
            continue
        sent[index] = arrivals
        try:
            rgb = downscale(np.asarray(decode(bytes(payload))["data"]))
        except Exception as exc:
            print(f"camera {index}: undecodable frame: {exc}", file=sys.stderr, flush=True)
            continue
        ppm = to_ppm(rgb)
        out.write(_HEADER.pack(index, rgb.shape[1], rgb.shape[0], len(ppm)) + ppm)
        count += 1
    out.flush()
    return count


def worker(cameras=CAMERAS, rate=FRAME_HZ):
    """Child process: subscribe (raw bytes) and decode only what is sent."""
    # Frames get a private copy of stdout; anything libraries print goes to stderr.
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from dexcomm import Node
    from dexcomm.codecs import RGBImageCodec

    node = Node(name="wuji_teleop_cameras")
    latest = {}

    def on_message(index, payload, *_):
        # Raw messages carry no timestamp; count arrivals to spot new frames.
        latest[index] = (latest.get(index, (0, None))[0] + 1, getattr(payload, "data", payload))

    subscribers = [node.create_subscriber(  # noqa: F841 (kept alive for the loop)
        topic=topic, callback=functools.partial(on_message, index), buffer_size=1)
        for index, (_, topic, _) in enumerate(cameras)]
    sent = {}
    parent = os.getppid()
    try:
        # Exit with the dashboard even when no frames arrive to hit a broken pipe.
        while os.getppid() == parent:
            pump(latest, out, sent, RGBImageCodec.decode)
            time.sleep(1.0 / rate)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        node.shutdown()


class RobotCameras:
    """Runs the camera worker and keeps each camera's newest frame."""

    def __init__(self, cameras=CAMERAS, command=None):
        self.cameras = cameras
        self.command = command or [sys.executable, "-m", "wuji_teleop.robot_cameras"]
        self._lock = threading.Lock()
        self._frames = {}  # camera index -> (ppm bytes, received monotonic, sequence)
        self._stderr = collections.deque(maxlen=3)
        self._process = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="robot_cameras", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._process = subprocess.Popen(
                    self.command, cwd=REPO, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError as exc:
                self._stderr.append(f"camera feed did not start: {exc}")
                self._stop.wait(RESTART_S)
                continue
            threading.Thread(target=self._read_errors, args=(self._process.stderr,),
                             daemon=True).start()
            self._read_frames(self._process.stdout)
            self._process.wait()
            self._stop.wait(RESTART_S)

    def _read_frames(self, stream):
        sequence = 0
        while True:
            header = stream.read(_HEADER.size)
            if len(header) < _HEADER.size:
                return
            index, _, _, size = _HEADER.unpack(header)
            ppm = stream.read(size)
            if len(ppm) < size:
                return
            sequence += 1
            with self._lock:
                self._frames[index] = (ppm, time.monotonic(), sequence)

    def _read_errors(self, stream):
        for line in stream:
            text = line.decode(errors="replace").strip()
            if text:
                self._stderr.append(text)

    def views(self, now=None):
        """{name: (ppm or None, sequence, status text)} for every camera."""
        now = time.monotonic() if now is None else now
        with self._lock:
            frames = dict(self._frames)
        running = self._process is not None and self._process.poll() is None
        views = {}
        for index, (name, topic, sensor) in enumerate(self.cameras):
            frame = frames.get(index)
            if frame is not None and now - frame[1] < STALE_S:
                views[name] = (frame[0], frame[2], "streaming")
            elif not running and self._stderr:
                views[name] = (None, 0, f"camera feed stopped: {self._stderr[-1][:90]}")
            else:
                when = "" if frame is None else f" for {now - frame[1]:.0f} s"
                views[name] = (None, 0, f"not detected{when} ({topic}); on the Jetson: "
                                        f"dexsensor enable {sensor}")
        return views

    def close(self):
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    worker()
