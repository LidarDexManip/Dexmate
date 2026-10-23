"""Latest-frame PICO Motion Tracker input from XRoboToolkit object tracking.

The XRoboToolkit PC service relays each tracker as ``[x, y, z, qx, qy, qz, qw]``
in the OpenXR world (x right, y up, +z toward the user, metres), tagged with the
tracker serial number. This module polls ``xrobotoolkit_sdk`` and converts every
pose into the robot frame (x forward, y left, z up). No robot command transport
is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation


# OpenXR (x right, y up, z back) to robot (x forward, y left, z up).
XR_TO_ROBOT = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
TRACKER_STALE_SECONDS = 0.250
SIDES = ("left", "right")


def xr_pose_to_robot(value) -> np.ndarray:
    """Convert one ``[x, y, z, qx, qy, qz, qw]`` OpenXR pose to a robot-frame 4x4.

    Both the position and the body axes are re-expressed in robot axes, so a
    tracker's local x points forward when its OpenXR local -z does.
    """
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite [x, y, z, qx, qy, qz, qw] pose")
    quat = pose[3:]
    norm = float(np.linalg.norm(quat))
    if not math.isclose(norm, 1.0, abs_tol=0.05):
        raise ValueError("Tracker quaternion is not normalized")
    rotation = Rotation.from_quat(quat / norm).as_matrix()
    out = np.eye(4)
    out[:3, :3] = XR_TO_ROBOT @ rotation @ XR_TO_ROBOT.T
    out[:3, 3] = XR_TO_ROBOT @ pose[:3]
    return out


@dataclass(frozen=True)
class TrackerFrame:
    poses: dict  # serial -> robot-frame 4x4
    headset: np.ndarray | None  # robot-frame 4x4, None until the headset streams
    timestamp_ns: int
    received_at: float  # host monotonic seconds when this frame was first seen


@dataclass(frozen=True)
class TrackerSnapshot:
    frame: TrackerFrame | None
    revision: int
    received: int
    error: str | None


def heading_rotation(yaw):
    """Rotation about robot z by ``yaw`` radians."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def operator_heading(headset=None, tracker_poses=()):
    """Yaw of the operator's forward direction in the robot-axis XR world.

    The headset's local x is the gaze direction. Without a headset pose the
    heading is taken perpendicular to the right-to-left wrist line.
    """
    if headset is not None:
        forward = np.asarray(headset, dtype=np.float64)[:3, 0]
        if math.hypot(forward[0], forward[1]) > 0.2:
            return math.atan2(forward[1], forward[0])
    if len(tracker_poses) == 2:
        left, right = (np.asarray(pose, dtype=np.float64)[:3, 3] for pose in tracker_poses)
        across = left - right
        if math.hypot(across[0], across[1]) > 0.05:
            return math.atan2(across[1], across[0]) - math.pi / 2
    raise ValueError("Cannot tell which way the operator faces; look straight ahead")


def assign_tracker_sides(poses, *, yaw=0.0, left_serial=None, right_serial=None):
    """Return ``{side: serial}`` for the two wrist trackers.

    Configured serials win. Otherwise exactly two trackers must stream, and the
    one further to the operator's left becomes the left wrist.
    """
    serials = {"left": left_serial, "right": right_serial}
    if left_serial and left_serial == right_serial:
        raise ValueError("Left and right tracker serials must differ")
    missing = [serial for serial in serials.values() if serial and serial not in poses]
    if missing:
        raise ValueError(f"Tracker {missing[0]} is not streaming; seen: {sorted(poses)}")
    if all(serials.values()):
        return serials
    unassigned = [serial for serial in poses if serial not in serials.values()]
    needed = [side for side in SIDES if not serials[side]]
    if len(unassigned) != len(needed):
        raise ValueError(
            f"Expected {len(needed)} unassigned wrist tracker(s), found {len(unassigned)}; "
            "set the tracker serials")
    if len(needed) == 1:
        serials[needed[0]] = unassigned[0]
        return serials
    heading = heading_rotation(yaw)
    lateral = {serial: float((heading.T @ poses[serial][:3, 3])[1]) for serial in unassigned}
    left, right = sorted(unassigned, key=lambda serial: lateral[serial], reverse=True)
    if lateral[left] - lateral[right] < 0.05:
        raise ValueError("Hold the wrists apart so the left and right trackers can be told apart")
    return {"left": left, "right": right}


class XRoboToolkitTrackerSource:
    """Poll the XRoboToolkit SDK for PICO Motion Tracker poses.

    The SDK keeps returning the last poses after the stream stops, so a frame
    counts as new only when its motion timestamp advances (or, for a stream
    without timestamps, when the poses change). Tests may inject a fake SDK.
    """

    def __init__(self, *, sdk=None):
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk
            except ImportError as exc:
                raise RuntimeError(
                    "Tracker input requires xrobotoolkit_sdk in this Python environment; "
                    "run scripts/install_xrobotoolkit_sdk.sh") from exc
        self._sdk = sdk
        self._frame = None
        self._revision = 0
        self._received = 0
        self._error = None
        self._last_timestamp_ns = 0
        self._last_raw = None
        self._closed = False
        self.stale_duplicates = 0
        sdk.init()

    def read(self):
        try:
            self._poll()
        except (ValueError, TypeError, IndexError) as exc:
            self._error = f"Invalid tracker frame: {exc}"
            self._revision += 1
        return TrackerSnapshot(self._frame, self._revision, self._received, self._error)

    def _poll(self):
        sdk = self._sdk
        count = int(sdk.num_motion_data_available())
        if count <= 0:
            return
        timestamp_ns = int(sdk.get_motion_timestamp_ns())
        serials = [str(serial) for serial in sdk.get_motion_tracker_serial_numbers()][:count]
        raw = np.asarray(sdk.get_motion_tracker_pose(), dtype=np.float64)[:count]
        if len(serials) != count or raw.shape != (count, 7):
            raise ValueError(f"Expected {count} tracker poses with serials")
        if timestamp_ns > 0:
            if timestamp_ns <= self._last_timestamp_ns:
                return  # same or older frame must not refresh the watchdog
        elif self._last_raw is not None and np.array_equal(raw, self._last_raw):
            return
        if "" in serials:
            raise ValueError("Tracker serial numbers are missing")
        live = self._live_entries(serials, raw)
        poses = {serial: xr_pose_to_robot(raw[index]) for serial, index in live.items()}
        headset_raw = np.asarray(sdk.get_headset_pose(), dtype=np.float64)
        headset = (None if headset_raw.shape != (7,) or not np.any(headset_raw[3:])
                   else xr_pose_to_robot(headset_raw))
        self._last_timestamp_ns = max(self._last_timestamp_ns, timestamp_ns)
        self._last_raw = raw.copy()
        self._frame = TrackerFrame(poses, headset, timestamp_ns, time.monotonic())
        self._error = None
        self._revision += 1
        self._received += 1

    def _live_entries(self, serials, raw):
        """``{serial: row}``; a repeated serial keeps the row that is updating.

        The PC service can keep listing trackers that stopped streaming under a
        serial it repeats; those rows never change. ``stale_duplicates`` counts
        the rows dropped.
        """
        def changed(index):
            last = self._last_raw
            return last is None or last.shape != raw.shape or not np.array_equal(
                last[index], raw[index])

        chosen = {}
        for index, serial in enumerate(serials):
            if serial not in chosen or (changed(index) and not changed(chosen[serial])):
                chosen[serial] = index
        self.stale_duplicates = len(serials) - len(chosen)
        return chosen

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._sdk.close()
