"""Latest-frame Pico controller wrist input from DexStream's Zenoh JSON topic."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
import time

import numpy as np


def validate_rigid_pose(value) -> np.ndarray:
    """Copy a finite proper homogeneous transform, rejecting malformed poses."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 wrist transform")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError("Wrist transform has an invalid homogeneous bottom row")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3)):
        raise ValueError("Wrist transform rotation is not a proper orthonormal matrix")
    pose = pose.copy()
    pose.setflags(write=False)
    return pose


@dataclass(frozen=True)
class WristPoseFrame:
    pose: np.ndarray
    index_trigger: float
    timestamp_ns: int
    received_at: float  # host monotonic seconds adjusted for sender clock age


@dataclass(frozen=True)
class WristPoseSnapshot:
    frame: WristPoseFrame | None
    revision: int
    received: int
    error: str | None


class PicoControllerPoseSource:
    """Subscribe to the left controller in DexStream's ``vr/controllers``.

    The expected payload is DexStream's robot-frame matrix JSON.  If Zenoh is
    unavailable, construction explains the optional ``eclipse-zenoh`` package
    needed for Pico input.  Tests may inject a fake session without importing it.
    """

    def __init__(self, topic="vr/controllers", *, session=None, zenoh_module=None):
        if session is None:
            if zenoh_module is None:
                try:
                    import zenoh as zenoh_module
                except ImportError as exc:
                    raise RuntimeError(
                        "Pico controller input requires the optional 'eclipse-zenoh' "
                        "package in this Python environment"
                    ) from exc
            session = zenoh_module.open(zenoh_module.Config())
            self._owns_session = True
        else:
            self._owns_session = False
        self.topic = topic
        self._session = session
        self._lock = threading.Lock()
        self._frame = None
        self._revision = 0
        self._received = 0
        self._error = None
        self._last_timestamp_ns = 0
        self._closed = False
        try:
            self._subscriber = session.declare_subscriber(topic, self._on_sample)
        except BaseException:
            if self._owns_session:
                session.close()
            raise

    @staticmethod
    def parse_payload(payload, *, now_ns=None, now_monotonic=None):
        """Parse DexStream JSON into one left wrist frame."""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            value = json.loads(payload)
        elif isinstance(payload, dict):
            value = payload
        else:
            raise ValueError("Expected a JSON object or encoded JSON payload")
        timestamp_ns = int(value["timestamp_ns"])
        if timestamp_ns <= 0:
            raise ValueError("Invalid controller timestamp")
        now_ns = time.time_ns() if now_ns is None else int(now_ns)
        if timestamp_ns > now_ns + 1_000_000_000:
            raise ValueError("Controller timestamp is more than one second in the future")
        left = value["left"]
        pose = validate_rigid_pose(left["wrist"])
        trigger = float(left["index_trigger"])
        if not np.isfinite(trigger) or not 0.0 <= trigger <= 1.0:
            raise ValueError("Left index trigger must be finite and between 0 and 1")
        now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        sender_age = max(0.0, (now_ns - timestamp_ns) / 1e9)
        return WristPoseFrame(pose, trigger, timestamp_ns, now_monotonic - sender_age)

    def _on_sample(self, sample):
        try:
            payload = sample.payload.to_bytes()
            frame = self.parse_payload(payload)
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError,
                OverflowError) as exc:
            with self._lock:
                self._error = f"Invalid Pico controller frame: {exc}"
                self._revision += 1
            return
        with self._lock:
            if frame.timestamp_ns <= self._last_timestamp_ns:
                return  # duplicate/out-of-order data must not refresh the watchdog
            self._last_timestamp_ns = frame.timestamp_ns
            self._frame = frame
            self._error = None
            self._revision += 1
            self._received += 1

    def read(self):
        with self._lock:
            return WristPoseSnapshot(self._frame, self._revision, self._received, self._error)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._subscriber.undeclare()
        finally:
            if self._owns_session:
                self._session.close()
