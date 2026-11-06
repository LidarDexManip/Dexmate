"""Wuji finger telemetry at the hands' native 1 kHz, for finger compliance work.

The SDK state callbacks and the 100 Hz command path only append a row; a
background thread writes fixed-size binary records a few times a second, so a
slow disk never delays a command. Each run writes:

- ``<prefix>.states.bin``: every joint_states frame per hand (positions, the
  hand's own velocity estimate, and effort, which is measured motor current in A);
- ``<prefix>.commands.bin``: every finger command with its pre-slew retargeted
  goal (the operator's nominal target);
- ``<prefix>.compliance.bin`` (only with the hand force regulator on): per
  command, the nominal reference, the compliant reference sent, the observed
  angles it was computed from (compliant - observed is HACo's command-state
  gap), each fingertip's estimated force and its contact activation;
- ``<prefix>.glove.bin``: every glove skeleton frame the retargeter consumed
  (21 keypoints, metres, in the glove's wrist frame), so glove calibration and
  retargeting can be told apart offline.

``load(prefix)`` reads them back as structured arrays, dropping a record cut
short when the process was killed mid-write.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import threading
import time

import numpy as np

SIDES = ("left", "right")
STATE_DTYPE = np.dtype([("t", "<f8"), ("stamp_us", "<i8"), ("side", "u1"),
                        ("q", "<f4", 20), ("v", "<f4", 20), ("effort", "<f4", 20)])
COMMAND_DTYPE = np.dtype([("t", "<f8"), ("side", "u1"),
                          ("target", "<f4", 20), ("goal", "<f4", 20)])
COMPLIANCE_DTYPE = np.dtype([("t", "<f8"), ("side", "u1"), ("nominal", "<f4", 20),
                             ("compliant", "<f4", 20), ("observed", "<f4", 20),
                             ("force", "<f4", (5, 3)), ("weight", "<f4", 5)])
GLOVE_DTYPE = np.dtype([("t", "<f8"), ("stamp_us", "<i8"), ("side", "u1"),
                        ("keypoints", "<f4", (21, 3))])
_STREAMS = (("states", STATE_DTYPE), ("commands", COMMAND_DTYPE),
            ("compliance", COMPLIANCE_DTYPE), ("glove", GLOVE_DTYPE))
_NO_GOAL = np.full(20, np.nan)


class _Stream:
    def __init__(self, path, dtype, max_rows):
        self.path, self.dtype = Path(path), dtype
        self.rows = deque(maxlen=int(max_rows))
        self.dropped = 0
        self.handle = None  # opened on the first write, so unused streams leave no file

    def close(self):
        if self.handle is not None:
            self.handle.close()

    def append(self, row):
        if len(self.rows) == self.rows.maxlen:
            self.dropped += 1
        self.rows.append(row)

    def take(self):
        rows = list(self.rows)
        self.rows.clear()
        return rows

    def write(self, rows):
        if rows:
            if self.handle is None:
                self.handle = self.path.open("wb")
            np.array(rows, dtype=self.dtype).tofile(self.handle)
            self.handle.flush()


class HandLogger:
    """Bounded background writer for finger states and commands."""

    def __init__(self, prefix, *, flush_s=0.25, max_rows=600_000, clock=time.monotonic):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.Lock()
        self._states = _Stream(f"{prefix}.states.bin", STATE_DTYPE, max_rows)
        self._commands = _Stream(f"{prefix}.commands.bin", COMMAND_DTYPE, max_rows // 5)
        self._compliance = _Stream(f"{prefix}.compliance.bin", COMPLIANCE_DTYPE, max_rows // 5)
        self._glove = _Stream(f"{prefix}.glove.bin", GLOVE_DTYPE, max_rows // 5)
        self._all = (self._states, self._commands, self._compliance, self._glove)
        self._flush_s = float(flush_s)
        self._closed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-log-writer", daemon=True)
        self._thread.start()

    @property
    def dropped(self):
        return sum(stream.dropped for stream in self._all)

    def state(self, side, stamp_us, q, v, effort):
        row = (self.clock(), int(stamp_us), SIDES.index(side), q, v, effort)
        with self._lock:
            if not self._closed:
                self._states.append(row)

    def command(self, side, target, goal=None):
        row = (self.clock(), SIDES.index(side), target, _NO_GOAL if goal is None else goal)
        with self._lock:
            if not self._closed:
                self._commands.append(row)

    def compliance(self, side, nominal, compliant, observed, force, weight):
        row = (self.clock(), SIDES.index(side), nominal, compliant, observed, force, weight)
        with self._lock:
            if not self._closed:
                self._compliance.append(row)

    def glove(self, side, stamp_us, keypoints):
        row = (self.clock(), int(stamp_us), SIDES.index(side), keypoints)
        with self._lock:
            if not self._closed:
                self._glove.append(row)

    def _flush(self):
        with self._lock:
            batches = [(stream, stream.take()) for stream in self._all]
        for stream, rows in batches:
            stream.write(rows)

    def _run(self):
        while not self._stop.wait(self._flush_s):
            self._flush()

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._closed = True
        self._flush()
        for stream in self._all:
            stream.close()


def load(prefix):
    """{"states", "commands", "compliance", "glove"} structured arrays from a run's files.

    A stream that was never written (no rows, or a run before it existed) is an
    empty array.
    """
    logs = {}
    for name, dtype in _STREAMS:
        path = Path(f"{prefix}.{name}.bin")
        data = path.read_bytes() if path.exists() else b""
        logs[name] = np.frombuffer(data[:len(data) - len(data) % dtype.itemsize], dtype=dtype)
    return logs
