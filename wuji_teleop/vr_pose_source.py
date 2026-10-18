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


def _decode_json_payload(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return payload
    raise ValueError("Expected a JSON object or encoded JSON payload")


def _validate_timestamp_ns(value, *, now_ns=None, label="controller"):
    timestamp_ns = int(value["timestamp_ns"])
    if timestamp_ns <= 0:
        raise ValueError(f"Invalid {label} timestamp")
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    if timestamp_ns > now_ns + 1_000_000_000:
        raise ValueError(f"{label.capitalize()} timestamp is more than one second in the future")
    return timestamp_ns


def _received_at_monotonic(timestamp_ns, *, now_ns=None, now_monotonic=None):
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
    sender_age = max(0.0, (now_ns - timestamp_ns) / 1e9)
    return now_monotonic - sender_age


class _PicoWristPoseSource:
    """Shared Zenoh subscriber state for DexStream wrist topics."""

    _invalid_frame_prefix: str
    _zenoh_input_label: str

    def __init__(self, topic, *, session=None, zenoh_module=None):
        if session is None:
            if zenoh_module is None:
                try:
                    import zenoh as zenoh_module
                except ImportError as exc:
                    raise RuntimeError(
                        f"Pico {self._zenoh_input_label} input requires the optional "
                        "'eclipse-zenoh' package in this Python environment"
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

    def _on_sample(self, sample):
        try:
            payload = sample.payload.to_bytes()
            frame = self.parse_payload(payload)
        except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError,
                OverflowError) as exc:
            with self._lock:
                self._error = f"{self._invalid_frame_prefix}: {exc}"
                self._revision += 1
            return
        if frame is None:
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


class PicoControllerPoseSource(_PicoWristPoseSource):
    """Subscribe to the left controller in DexStream's ``vr/controllers``.

    The expected payload is DexStream's robot-frame matrix JSON.  If Zenoh is
    unavailable, construction explains the optional ``eclipse-zenoh`` package
    needed for Pico input.  Tests may inject a fake session without importing it.
    """

    _invalid_frame_prefix = "Invalid Pico controller frame"
    _zenoh_input_label = "controller"

    def __init__(self, topic="vr/controllers", *, session=None, zenoh_module=None):
        super().__init__(topic, session=session, zenoh_module=zenoh_module)

    @staticmethod
    def parse_payload(payload, *, now_ns=None, now_monotonic=None):
        """Parse DexStream JSON into one left wrist frame."""
        value = _decode_json_payload(payload)
        timestamp_ns = _validate_timestamp_ns(value, now_ns=now_ns, label="controller")
        left = value["left"]
        pose = validate_rigid_pose(left["wrist"])
        trigger = float(left["index_trigger"])
        if not np.isfinite(trigger) or not 0.0 <= trigger <= 1.0:
            raise ValueError("Left index trigger must be finite and between 0 and 1")
        received_at = _received_at_monotonic(
            timestamp_ns, now_ns=now_ns, now_monotonic=now_monotonic)
        return WristPoseFrame(pose, trigger, timestamp_ns, received_at)


class PicoHandPoseSource(_PicoWristPoseSource):
    """Subscribe to one hand in DexStream's ``vr/hands``.

    Hand payloads omit index triggers; tracked hands are treated as fully pressed.
    A missing selected side means no hand data and does not invalidate prior frames.
    """

    _invalid_frame_prefix = "Invalid Pico hand frame"
    _zenoh_input_label = "hand"

    def __init__(self, topic="vr/hands", *, side="left", session=None, zenoh_module=None):
        if side not in ("left", "right"):
            raise ValueError("Pico hand side must be 'left' or 'right'")
        self.side = side
        super().__init__(topic, session=session, zenoh_module=zenoh_module)

    def parse_payload(self, payload, *, now_ns=None, now_monotonic=None):
        """Parse one selected wrist frame, or return None while it is untracked."""
        value = _decode_json_payload(payload)
        hand = value.get(self.side)
        if hand is None:
            return None
        timestamp_ns = _validate_timestamp_ns(value, now_ns=now_ns, label="hand")
        pose = validate_rigid_pose(hand["wrist"])
        # DexStream sends a wrist at the origin while the hand is not tracked.
        if np.allclose(pose[:3, 3], 0.0, atol=1e-6):
            return None
        received_at = _received_at_monotonic(
            timestamp_ns, now_ns=now_ns, now_monotonic=now_monotonic)
        return WristPoseFrame(pose, 1.0, timestamp_ns, received_at)
