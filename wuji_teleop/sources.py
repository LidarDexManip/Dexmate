"""Latest-frame sources. Only this module communicates with glove hardware."""

from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol

import numpy as np


def validate_keypoints(value) -> np.ndarray:
    """Copy 21 MediaPipe-order landmarks in meters; reject broken skeletons."""
    points = np.array(value, dtype=np.float32, copy=True)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite XYZ landmarks in meters")
    chains = points[1:].reshape(5, 4, 3)
    if np.any(np.linalg.norm(np.diff(chains, axis=1), axis=2) < 1e-5):
        raise ValueError("Skeleton contains collapsed finger segments")
    palm = np.cross(points[5] - points[0], points[17] - points[0])
    if np.linalg.norm(palm) < 1e-6:
        raise ValueError("Skeleton has a degenerate palm")
    points.setflags(write=False)
    return points


@dataclass(frozen=True)
class HandFrame:
    keypoints: np.ndarray
    received_at: float  # host monotonic seconds, adjusted for SDK transport age
    timestamp_us: int  # original SDK timestamp, retained for diagnostics
    sequence: int
    confidence: float | None = None


@dataclass(frozen=True)
class SourceSnapshot:
    frame: HandFrame | None
    revision: int
    received: int
    error: str | None


class LatestFrameBuffer:
    """Callbacks overwrite one slot; neither SDK queues nor history can grow."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._revision = 0
        self._received = 0
        self._error = None
        self._timestamp = None

    def publish(self, points, *, received_at, timestamp_us, sequence, confidence=None):
        try:
            points = validate_keypoints(points)
            frame = HandFrame(points, received_at, timestamp_us, sequence, confidence)
        except (ValueError, TypeError, OverflowError) as exc:
            self.fail(str(exc))
            return
        with self._lock:
            # Duplicate or out-of-order packets must not refresh the watchdog.
            if self._timestamp is not None and timestamp_us <= self._timestamp:
                return
            self._timestamp = timestamp_us
            self._frame = frame
            self._error = None
            self._revision += 1
            self._received += 1

    def fail(self, message):
        with self._lock:
            self._error = message
            self._revision += 1

    def snapshot(self):
        with self._lock:
            return SourceSnapshot(self._frame, self._revision, self._received, self._error)


class HandSource(Protocol):
    side: str

    def read(self) -> SourceSnapshot: ...
    def close(self) -> None: ...


def synthetic_keypoints(seconds: float, side: str) -> np.ndarray:
    """An illustrative curl cycle, not recorded human tracking or calibration data."""
    points = np.zeros((21, 3), dtype=np.float32)
    # Wrist-frame coordinates: fingers extend toward -Z, palm toward +Y.
    bases = [(0.0206, 0.0084, -0.0289), (0.0294, -0.0059, -0.0919),
             (0.0075, -0.0105, -0.0895), (-0.0137, -0.0077, -0.0844),
             (-0.0340, -0.0023, -0.0743)]
    lengths = [(0.0173, 0.0411, 0.0399), (0.0412, 0.0243, 0.0201),
               (0.0479, 0.0245, 0.0287), (0.0511, 0.0271, 0.0174),
               (0.0271, 0.0260, 0.0170)]
    for finger, (base, bones) in enumerate(zip(bases, lengths)):
        curl = 0.45 * (1 - math.cos(2 * math.pi * seconds / 6 - finger * 0.15))
        start = 1 + 4 * finger
        points[start] = base
        for joint, length in enumerate(bones):
            angle = curl * (1 + 0.7 * joint)
            dx = 0.35 * length if finger == 0 else 0.0
            points[start + joint + 1] = points[start + joint] + [
                dx, length * math.sin(angle), -length * math.cos(angle)
            ]
    if side == "left":
        points[:, 0] *= -1
    return points


class SyntheticSource:
    def __init__(self, side="right"):
        self.side = side
        self._start = time.monotonic()
        self._buffer = LatestFrameBuffer()
        self._sequence = 0

    def read(self):
        now = time.monotonic()
        self._sequence += 1
        self._buffer.publish(
            synthetic_keypoints(now - self._start, self.side), received_at=now,
            timestamp_us=time.time_ns() // 1000, sequence=self._sequence, confidence=1.0,
        )
        return self._buffer.snapshot()

    def close(self):
        pass


def scan_gloves(manager, sdk):
    return [device for device in manager.scan() if device.device_type == sdk.DeviceType.WujiGlove]


def select_glove(devices, serial=None):
    matches = [d for d in devices if serial is None or d.sn == serial]
    if not matches:
        raise RuntimeError(
            "No matching Wuji Glove found. Check power, Ethernet link, and the "
            "192.168.1.x interface; see WUJI_TELEOP_RUNBOOK.md."
        )
    if len(matches) != 1:
        raise RuntimeError("Multiple gloves found; select --glove-sn from: " +
                           ", ".join(d.sn for d in matches))
    return matches[0]


class GloveSource:
    """A single glove subscription. Never opens a robot-hand command channel."""

    def __init__(self, serial=None, user_id="", *, sdk=None, device_name=None,
                 address=None):
        if sdk is None:
            import wuji_sdk as sdk
        self._manager = sdk.SdkManager.instance()
        self._buffer = LatestFrameBuffer()
        self._subscription = None
        self._glove = None
        self._previous_user = None
        # The SDK identifies live connections by device_name.  Keep the legacy
        # name for a single glove, while allowing bilateral callers to give
        # each simultaneously connected glove a distinct name.
        self._device_name = device_name or "wuji_sim_glove"
        self._closed = False
        try:
            if address is None:
                device = select_glove(scan_gloves(self._manager, sdk), serial)
                self.serial = device.sn
                self.address = device.address
                connection = {"sn": self.serial}
            else:
                self.address = address
                connection = {"address": address}
            self._previous_user = self._manager.current_user()["user_id"]
            self._manager.switch_user(user_id)
            self._glove = self._manager.connect(
                **connection, device_name=self._device_name)
            if address is not None:
                self.serial = self._glove.serial_number
                if serial is not None and self.serial != serial:
                    raise RuntimeError(
                        f"Glove at {address} has serial {self.serial}, expected {serial}")
            self.side = self._glove.hand_side().get()
            if self.side not in ("left", "right"):
                raise ValueError(f"Unsupported glove side: {self.side!r}")
            self.firmware = self._glove.version().get()
            self._subscription = self._glove.hand_skeleton().subscribe_with_callback(
                self._on_frame, on_error=self._buffer.fail,
            )
        except BaseException:
            self.close()
            raise

    def _on_frame(self, skeleton):
        try:
            timestamp = int(skeleton.header.timestamp_us)
            received_at = time.monotonic()
            # SDK timestamps are synchronized to host UTC. Account for transport
            # delay as well as time spent waiting in our own latest-frame slot.
            transport_age = max(0.0, time.time() - timestamp / 1e6)
            if timestamp <= 0 or timestamp / 1e6 > time.time() + 1.0:
                raise ValueError("Invalid glove timestamp; check SDK clock synchronization")
            self._buffer.publish(
                [j.pose.position for j in skeleton.joints],
                received_at=received_at - transport_age, timestamp_us=timestamp,
                sequence=int(skeleton.header.seq),
                confidence=min(j.confidence for j in skeleton.joints),
            )
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            self._buffer.fail(f"Invalid skeleton: {exc}")

    def read(self):
        return self._buffer.snapshot()

    def palm_imu_resource(self):
        """Return the fused palm IMU resource on this existing glove handle."""
        if self._closed or self._glove is None:
            raise RuntimeError("Glove source is not connected")
        return self._glove.imu_data_palm()

    def palm_imu_mount_rotation(self):
        """Read the SDK's fixed wrist-to-palm-IMU rotation for this glove."""
        if self._closed or self._glove is None:
            raise RuntimeError("Glove source is not connected")
        from .imu_arm_source import _rotation_from_xyzw
        found = threading.Event()
        result = {}
        wrist = f"{self.side[0]}_wrist"
        imu = f"{self.side[0]}_palm_imu_link"

        def on_frame(frame):
            for transform in frame.transforms:
                if transform.parent_frame_id == wrist and transform.child_frame_id == imu:
                    q = transform.rotation
                    result["rotation"] = _rotation_from_xyzw([q.x, q.y, q.z, q.w])
                    found.set()
                    return

        sub = self._manager.tf_static().subscribe_with_callback(on_frame)
        try:
            if not found.wait(3.0):
                raise RuntimeError(f"No {wrist} to {imu} transform from glove SDK")
            return result["rotation"]
        finally:
            sub.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Attempt every cleanup even if a disconnected device raises on close.
        errors = []
        actions = []
        if self._subscription is not None:
            actions.append(self._subscription.close)
        if self._glove is not None:
            actions.append(lambda: self._manager.disconnect(self._device_name))
        if self._previous_user is not None:
            actions.append(lambda: self._manager.switch_user(self._previous_user))
        for action in actions:
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            import warnings
            warnings.warn("Glove cleanup: " + "; ".join(errors), RuntimeWarning)
