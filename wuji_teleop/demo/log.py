"""Append-only binary streams for demonstration capture, and their reader.

SDK callbacks only enqueue a row; one background thread writes, so a slow disk
never delays a callback. Each stream is ``<dir>/<name>.bin``, a sequence of
fixed-size records of one numpy dtype; ``load`` drops a record cut short when
the process was killed mid-write.
"""

from __future__ import annotations

from pathlib import Path
import queue
import threading

import numpy as np

GLOVE_DTYPE = np.dtype([
    ("t", "<f8"),                  # host UTC seconds at callback
    ("stamp_us", "<i8"),           # SDK timestamp (host-UTC synchronized)
    ("seq", "<i8"),
    ("side", "u1"),                # 0 left, 1 right
    ("keypoints", "<f4", (21, 3)),  # metres, glove wrist frame, MediaPipe order
    ("orientation", "<f4", (21, 4)),  # per-joint quaternion x y z w
    ("confidence", "<f4", 21),
])
EMF_DTYPE = np.dtype([
    ("t", "<f8"), ("side", "u1"),
    ("confidence", "<f4", 5),       # fingertip sensors thumb..pinky
    ("position", "<f4", (5, 3)),    # metres, EMF transmitter frame
    ("orientation", "<f4", (5, 4)),  # x y z w
])
IMU_DTYPE = np.dtype([
    ("t", "<f8"), ("stamp_us", "<i8"), ("side", "u1"),
    ("quat", "<f8", 4),            # fused palm orientation x y z w (IMU world)
    ("acc", "<f4", 3), ("gyro", "<f4", 3),
])
MAX_TRACKERS = 4
XR_DTYPE = np.dtype([
    ("t", "<f8"),                  # host UTC seconds when this frame was first seen
    ("motion_ns", "<i8"),          # XRoboToolkit motion timestamp
    ("sdk_ns", "<i8"),             # XRoboToolkit get_time_stamp_ns
    ("count", "u1"),
    ("serial", "<i2", MAX_TRACKERS),  # index into meta serial table, -1 empty
    ("tracker", "<f8", (MAX_TRACKERS, 7)),  # raw OpenXR x y z qx qy qz qw
    ("headset", "<f8", 7),
    ("hand", "<f4", (2, 26, 7)),   # PICO hand tracking (left, right), raw OpenXR
    ("hand_active", "u1", 2),
])
STREAMS = {"glove": GLOVE_DTYPE, "emf": EMF_DTYPE, "imu": IMU_DTYPE, "xr": XR_DTYPE}


class StreamWriter:
    """One writer thread for every stream in a directory."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._queue = queue.SimpleQueue()
        self._handles = {}
        self.counts = {name: 0 for name in STREAMS}
        self._thread = threading.Thread(target=self._run, name="demo-log", daemon=True)
        self._thread.start()

    def append(self, name, row):
        """Queue one record (a numpy structured scalar or 1-row array)."""
        self.counts[name] += 1
        self._queue.put((name, np.asarray(row, dtype=STREAMS[name]).tobytes()))

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            name, data = item
            handle = self._handles.get(name)
            if handle is None:
                handle = self._handles[name] = open(self.directory / f"{name}.bin", "ab")
            handle.write(data)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=10)
        for handle in self._handles.values():
            handle.close()


def load(directory, name):
    """Records of one stream, oldest first (empty if it was never written)."""
    dtype = STREAMS[name]
    path = Path(directory) / f"{name}.bin"
    if not path.exists():
        return np.zeros(0, dtype)
    raw = path.read_bytes()
    whole = len(raw) // dtype.itemsize * dtype.itemsize
    return np.frombuffer(raw[:whole], dtype=dtype).copy()
