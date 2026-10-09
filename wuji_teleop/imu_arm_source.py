"""Experimental, drift-limited arm input from a Wuji palm fused IMU.

The glove does not measure absolute wrist translation. This estimator uses
gravity-compensated acceleration for short relative motion and holds its last
target when input is invalid or stale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np


def _rotation_from_xyzw(value):
    """Convert SDK [x,y,z,w] quaternion to a 3x3 rotation matrix."""
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("IMU quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("IMU quaternion has zero length")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _rotation_from_rotvec(rotvec):
    """Rodrigues rotation matrix for an axis-angle vector."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = np.asarray(rotvec, dtype=np.float64) / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _vector3(value, label):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"IMU {label} must contain three finite values")
    return v


@dataclass(frozen=True)
class ImuArmSnapshot:
    pose: np.ndarray | None
    status: str
    received_at: float | None
    samples: int
    displacement_estimate_m: float
    max_displacement_estimate_m: float
    gesture_steps: int = 0
    displacement_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PalmImuRetargeter:
    """Estimate short relative wrist motion from fused orientation + accel.

    Calibration averages the measured world-frame gravity vector and gyro
    bias for one second. Translation is bounded, damped, and zero-velocity
    corrected when still. It is not a position tracker.
    """

    def __init__(self, *, calibration_seconds=1.0, stale_after=0.25,
                 max_speed=0.20, max_displacement=0.10, velocity_damping=1.8,
                 stationary_accel=0.20, stationary_gyro=0.06,
                 imu_in_wrist_rotation=None, hand_in_ee_rotation=None,
                 motion_mode="relative", gesture_step=0.06,
                 gesture_threshold=0.8, gesture_cooldown=0.45,
                 tilt_gain=0.50, tilt_deadband=0.05, tilt_smoothing=0.45,
                 translation_leak=0.0, translation_limit=None,
                 pointing_axis=None, vertical_to_forward=False,
                 tilt_pivot=None, arc_gain=1.0, max_arc_angle=2.4):
        self.calibration_seconds = float(calibration_seconds)
        self.stale_after = float(stale_after)
        self.max_speed = float(max_speed)
        self.max_displacement = float(max_displacement)
        self.velocity_damping = float(velocity_damping)
        self.stationary_accel = float(stationary_accel)
        self.stationary_gyro = float(stationary_gyro)
        self.motion_mode = motion_mode
        self.gesture_step = float(gesture_step)
        self.gesture_threshold = float(gesture_threshold)
        self.gesture_cooldown = float(gesture_cooldown)
        self.tilt_gain = float(tilt_gain)
        self.tilt_deadband = float(tilt_deadband)
        self.tilt_smoothing = float(tilt_smoothing)
        self.translation_leak = float(translation_leak)
        self.translation_limit = (None if translation_limit is None
                                  else float(translation_limit))
        self.vertical_to_forward = bool(vertical_to_forward)
        self.arc_gain = float(arc_gain)
        self.max_arc_angle = float(max_arc_angle)
        self._tilt_pivot = None if tilt_pivot is None else _vector3(tilt_pivot, "tilt pivot")
        axis = np.array([0.0, 0.0, 1.0] if pointing_axis is None else pointing_axis,
                        dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        self.pointing_axis = axis / axis_norm if axis_norm > 0 else axis
        self.imu_in_wrist_R = (np.eye(3) if imu_in_wrist_rotation is None else
                               np.asarray(imu_in_wrist_rotation, dtype=np.float64).copy())
        self.hand_in_ee_R = (np.eye(3) if hand_in_ee_rotation is None else
                             np.asarray(hand_in_ee_rotation, dtype=np.float64).copy())
        for label, rotation in (("wrist-to-IMU", self.imu_in_wrist_R),
                                ("end-effector-to-hand", self.hand_in_ee_R)):
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                    not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
                raise ValueError(f"Expected a proper {label} rotation")
        if (self.motion_mode not in ("relative", "gesture", "orientation", "tilt") or
                self.gesture_step <= 0 or self.gesture_threshold <= 0 or
                self.gesture_cooldown < 0 or
                self.calibration_seconds <= 0 or self.stale_after <= 0 or
                self.max_speed <= 0 or self.max_displacement <= 0 or
                self.velocity_damping < 0 or self.tilt_gain <= 0 or
                self.tilt_deadband < 0 or not 0 < self.tilt_smoothing <= 1 or
                self.translation_leak < 0 or
                not self.arc_gain > 0 or not self.max_arc_angle > 0 or
                (self.translation_limit is not None and self.translation_limit <= 0) or
                not np.isfinite(self.pointing_axis).all() or
                not np.isclose(np.linalg.norm(self.pointing_axis), 1.0)):
            raise ValueError("Invalid palm IMU retargeter limits")
        self._lock = threading.Lock()
        self._error = None
        self._last_timestamp = None
        self._last_received = None
        self._last_R = None
        self._ref_R = None
        self._gravity_world = None
        self._gyro_bias_sensor = None
        self._cal_start_us = None
        self._cal_accel = []
        self._cal_gyro = []
        self._cal_rotations = []
        self._last_integrate_us = None
        self._velocity = np.zeros(3)
        self._displacement = np.zeros(3)
        self._translation = np.zeros(3)
        self._arc_rotvec = np.zeros(3)
        self._gesture_goal = np.zeros(3)
        self._gesture_candidate_direction = None
        self._gesture_candidate_since_us = None
        self._gesture_idle_since_us = None
        self._gesture_last_trigger_us = None
        self._gesture_armed = True
        self._gesture_steps = 0
        self._max_displacement = 0.0
        self._anchor = None
        self._samples = 0
        self._status = "calibrating"
        self._last_pose = None
        self._recovery_pending = False

    def fail(self, message):
        with self._lock:
            self._error = str(message)
            self._status = "invalid"
            self._velocity[:] = 0
            self._recovery_pending = True
            self._gesture_candidate_direction = None
            self._gesture_candidate_since_us = None
            self._gesture_armed = False

    def set_tilt_pivot(self, pivot):
        """Swing tilt targets around this sim-frame shoulder point.

        With a pivot, tipping the fingers rotates the anchored hand about the
        shoulder, so the target keeps its reach instead of sliding toward
        the body. ``None`` restores straight-line tilt offsets.
        """
        pivot = None if pivot is None else _vector3(pivot, "tilt pivot")
        with self._lock:
            self._tilt_pivot = pivot
            self._arc_rotvec[:] = 0

    def recenter(self, anchor_pose):
        """Restart stationary calibration and anchor at the current sim pose."""
        anchor = np.asarray(anchor_pose, dtype=np.float64)
        if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
            raise ValueError("Expected a finite 4x4 anchor wrist pose")
        with self._lock:
            self._anchor = anchor.copy()
            self._error = None
            self._last_timestamp = None
            self._last_received = None
            self._last_R = None
            self._ref_R = None
            self._gravity_world = None
            self._gyro_bias_sensor = None
            self._cal_start_us = None
            self._cal_accel.clear()
            self._cal_gyro.clear()
            self._cal_rotations.clear()
            self._last_integrate_us = None
            self._velocity[:] = 0
            self._displacement[:] = 0
            self._translation[:] = 0
            self._arc_rotvec[:] = 0
            self._gesture_goal[:] = 0
            self._gesture_candidate_direction = None
            self._gesture_candidate_since_us = None
            self._gesture_idle_since_us = None
            self._gesture_last_trigger_us = None
            self._gesture_armed = True
            self._gesture_steps = 0
            self._max_displacement = 0.0
            self._status = "calibrating"
            self._recovery_pending = False

    def update(self, timestamp_us, quaternion_xyzw, linear_acceleration,
               angular_velocity, *, received_at=None):
        """Consume one SDK fused-IMU sample; duplicate timestamps are ignored."""
        try:
            timestamp_us = int(timestamp_us)
            if timestamp_us <= 0:
                raise ValueError("invalid timestamp")
            R = _rotation_from_xyzw(quaternion_xyzw)
            accel = _vector3(linear_acceleration, "acceleration")
            gyro = _vector3(angular_velocity, "angular velocity")
            received_at = time.monotonic() if received_at is None else float(received_at)
            if not math.isfinite(received_at):
                raise ValueError("invalid receive time")
        except (ValueError, TypeError, OverflowError) as exc:
            self.fail(f"Invalid palm IMU sample: {exc}")
            return False
        with self._lock:
            if self._last_timestamp is not None and timestamp_us <= self._last_timestamp:
                return False
            previous_us = self._last_timestamp
            self._last_timestamp = timestamp_us
            self._last_received = received_at
            self._last_R = R
            self._samples += 1
            self._error = None
            if self._ref_R is None:
                gravity_sample = R @ accel
                # Restart the one-second window if the wearer moves during
                # calibration. The measured magnitude can differ slightly
                # from 9.81 m/s², so use a broad physical range.
                unsettled = (float(np.linalg.norm(gyro)) > 0.15 or
                             not 8.5 <= float(np.linalg.norm(gravity_sample)) <= 10.8 or
                             (self._cal_accel and
                              np.linalg.norm(gravity_sample - np.mean(self._cal_accel, axis=0)) > 0.35))
                if unsettled:
                    self._cal_accel.clear()
                    self._cal_gyro.clear()
                    self._cal_rotations.clear()
                    self._cal_start_us = timestamp_us
                if self._cal_start_us is None:
                    self._cal_start_us = timestamp_us
                self._cal_accel.append(gravity_sample)
                self._cal_gyro.append(gyro.copy())
                self._cal_rotations.append(R)
                self._status = "calibrating"
                if timestamp_us - self._cal_start_us >= self.calibration_seconds * 1e6:
                    self._gravity_world = np.mean(self._cal_accel, axis=0)
                    self._gyro_bias_sensor = np.mean(self._cal_gyro, axis=0)
                    # Calibration pose anchors orientation at the end of the
                    # stationary window; translation remains relative to zero.
                    self._ref_R = self._cal_rotations[-1].copy()
                    self._last_integrate_us = timestamp_us
                    self._status = "ready"
                return True

            if self._recovery_pending:
                # Discard the first post-gap acceleration sample. This avoids
                # integrating over a long missing-data interval on recovery.
                self._recovery_pending = False
                self._last_integrate_us = timestamp_us
                self._status = "tracking"
                return True

            dt = ((timestamp_us - previous_us) / 1e6
                  if previous_us is not None else 0.0)
            if not math.isfinite(dt) or dt <= 0:
                return False
            dt = min(dt, 0.05)
            a_world = R @ accel - self._gravity_world
            gyro_norm = float(np.linalg.norm(gyro - self._gyro_bias_sensor))
            accel_norm = float(np.linalg.norm(a_world))
            if self.motion_mode == "orientation":
                self._last_integrate_us = timestamp_us
                self._status = "tracking"
                self._error = None
                return True
            if self.motion_mode == "tilt":
                # Pointing covers tip direction. A short acceleration nudge
                # covers a pull that keeps the fingers aimed the same way.
                # The nudge leaks back to zero at rest so the arm returns
                # with the glove instead of sticking where the accel left it.
                nudge_limit = (min(0.30, self.max_displacement)
                               if self.translation_limit is None else
                               min(self.translation_limit, self.max_displacement))
                self._integrate_velocity(dt, a_world, accel_norm, gyro_norm,
                                         self._translation, nudge_limit)
                if (self.translation_leak > 0 and
                        accel_norm <= self.stationary_accel and
                        gyro_norm <= self.stationary_gyro):
                    self._translation *= math.exp(-self.translation_leak * dt)
                self._last_integrate_us = timestamp_us
                self._status = "tracking"
                self._error = None
                return True
            if self.motion_mode == "gesture":
                self._update_gesture(timestamp_us, dt, a_world, accel_norm, gyro_norm)
                self._last_integrate_us = timestamp_us
                self._status = "tracking"
                self._error = None
                return True
            self._integrate_velocity(dt, a_world, accel_norm, gyro_norm,
                                     self._displacement, self.max_displacement)
            self._max_displacement = max(
                self._max_displacement, float(np.linalg.norm(self._displacement)))
            self._last_integrate_us = timestamp_us
            self._status = "tracking"
            self._error = None
            return True

    def _update_gesture(self, timestamp_us, dt, a_world, accel_norm, gyro_norm):
        """Turn one deliberate acceleration stroke into one bounded arm step."""
        if accel_norm < 0.25 and gyro_norm < 0.25:
            if self._gesture_idle_since_us is None:
                self._gesture_idle_since_us = timestamp_us
            if timestamp_us - self._gesture_idle_since_us >= 80_000:
                self._gesture_armed = True
        else:
            self._gesture_idle_since_us = None
        cooldown_done = (self._gesture_last_trigger_us is None or
                         timestamp_us - self._gesture_last_trigger_us >=
                         self.gesture_cooldown * 1e6)
        if (self._gesture_armed and cooldown_done and
                accel_norm >= self.gesture_threshold and gyro_norm < 0.5):
            direction = a_world / accel_norm
            if (self._gesture_candidate_direction is None or
                    float(direction @ self._gesture_candidate_direction) < 0.85):
                self._gesture_candidate_direction = direction
                self._gesture_candidate_since_us = timestamp_us
            elif timestamp_us - self._gesture_candidate_since_us >= 20_000:
                self._gesture_goal += self.gesture_step * direction
                distance = float(np.linalg.norm(self._gesture_goal))
                if distance > self.max_displacement:
                    self._gesture_goal *= self.max_displacement / distance
                self._gesture_last_trigger_us = timestamp_us
                self._gesture_armed = False
                self._gesture_steps += 1
                self._gesture_candidate_direction = None
                self._gesture_candidate_since_us = None
        else:
            self._gesture_candidate_direction = None
            self._gesture_candidate_since_us = None
        remaining = self._gesture_goal - self._displacement
        distance = float(np.linalg.norm(remaining))
        if distance > 0:
            self._displacement += remaining * min(1.0, self.max_speed * dt / distance)
        self._max_displacement = max(
            self._max_displacement, float(np.linalg.norm(self._displacement)))

    def _integrate_velocity(self, dt, acceleration, accel_norm, gyro_norm, store, limit):
        """Integrate gravity-free acceleration and hold still when the palm rests."""
        if accel_norm <= self.stationary_accel and gyro_norm <= self.stationary_gyro:
            self._velocity[:] = 0
            return
        self._velocity *= math.exp(-self.velocity_damping * dt)
        self._velocity += acceleration * dt
        speed = float(np.linalg.norm(self._velocity))
        if speed > self.max_speed:
            self._velocity *= self.max_speed / speed
        store += self._velocity * dt
        distance = float(np.linalg.norm(store))
        if distance > limit:
            store *= limit / distance
            self._velocity[:] = 0

    def _tilt_offset(self, current_rotation, anchor_rotation):
        """Turn a change in pointing direction into a bounded arm offset.

        The pointing axis is the simulated finger direction. Tipping the glove
        swings that axis, and the swing becomes the end-effector translation.
        A twist about the fingers does not move the axis, so it stays an
        orientation command. The result is in the simulation frame.
        """
        neutral = anchor_rotation @ self.pointing_axis
        current = current_rotation @ self.pointing_axis
        swing = current - neutral
        distance = float(np.linalg.norm(swing))
        if distance <= self.tilt_deadband:
            return np.zeros(3)
        offset = self.tilt_gain * swing * ((distance - self.tilt_deadband) / distance)
        if self.vertical_to_forward:
            # Palm-down pitch tips the fingers up or down. That swing is the
            # forward/back reach for the shoulder. Side-to-side yaw stays lateral.
            offset = offset.copy()
            offset[0] = offset[0] + offset[2]
            offset[2] *= 0.15
        magnitude = float(np.linalg.norm(offset))
        if magnitude > self.max_displacement:
            offset *= self.max_displacement / magnitude
        return offset

    def _tilt_arc(self, current_rotation, anchor_rotation):
        """Axis-angle swing of the finger direction, scaled for the arm.

        This is the shortest rotation from the calibrated finger direction to
        the current one, so a twist about the fingers gives no swing.
        """
        neutral = anchor_rotation @ self.pointing_axis
        current = current_rotation @ self.pointing_axis
        cross = np.cross(neutral, current)
        angle = math.atan2(float(np.linalg.norm(cross)), float(neutral @ current))
        if angle <= self.tilt_deadband:
            return np.zeros(3)
        axis_norm = float(np.linalg.norm(cross))
        if axis_norm < 1e-9:
            # Fingers flipped straight back: no unique swing axis.
            return self._arc_rotvec.copy()
        angle = min(self.arc_gain * (angle - self.tilt_deadband), self.max_arc_angle)
        return cross / axis_norm * angle

    def snapshot(self, anchor_pose, *, now=None):
        """Return anchored target pose; stale/invalid input returns no update."""
        now = time.monotonic() if now is None else float(now)
        anchor = np.asarray(anchor_pose, dtype=np.float64)
        if anchor.shape != (4, 4) or not np.isfinite(anchor).all():
            raise ValueError("Expected a finite 4x4 anchor wrist pose")
        with self._lock:
            if self._anchor is None:
                self._anchor = anchor.copy()
            displacement_m = float(np.linalg.norm(self._displacement))
            max_displacement_m = self._max_displacement
            if self._error:
                self._velocity[:] = 0
                self._recovery_pending = True
                return ImuArmSnapshot(None, "invalid", self._last_received, self._samples,
                                      displacement_m, max_displacement_m, self._gesture_steps)
            if self._ref_R is None:
                return ImuArmSnapshot(None, "calibrating", self._last_received, self._samples,
                                      displacement_m, max_displacement_m, self._gesture_steps)
            if self._last_received is None or now - self._last_received > self.stale_after:
                self._velocity[:] = 0
                self._recovery_pending = True
                self._gesture_candidate_direction = None
                self._gesture_candidate_since_us = None
                self._gesture_armed = False
                return ImuArmSnapshot(None, "stale", self._last_received, self._samples,
                                      displacement_m, max_displacement_m, self._gesture_steps)
            pose = self._anchor.copy()
            # The SDK quaternion describes the IMU sensor frame, which is
            # rotated relative to its calibrated wrist frame. Convert to the
            # wrist frame before aligning it with the simulated end effector.
            ref_wrist_R = self._ref_R @ self.imu_in_wrist_R.T
            ref_sim_hand_R = self._anchor[:3, :3] @ self.hand_in_ee_R
            imu_world_to_sim = ref_sim_hand_R @ ref_wrist_R.T
            pose[:3, :3] = (imu_world_to_sim @ self._last_R @
                            self.imu_in_wrist_R.T @ self.hand_in_ee_R.T)
            if self.motion_mode == "tilt" and self._tilt_pivot is not None:
                # Swing the anchored hand around the shoulder for lift, while
                # retaining a radial tilt offset so forward/back motion can
                # change reach and recruit the elbow.
                goal = self._tilt_arc(pose[:3, :3], self._anchor[:3, :3])
                self._arc_rotvec += self.tilt_smoothing * (goal - self._arc_rotvec)
                arm = self._anchor[:3, 3] - self._tilt_pivot
                tilt_offset = self._tilt_offset(
                    pose[:3, :3], self._anchor[:3, :3])
                self._displacement = (_rotation_from_rotvec(self._arc_rotvec) @ arm - arm
                                      + tilt_offset + imu_world_to_sim @ self._translation)
                pose[:3, 3] += self._displacement
                displacement_m = float(np.linalg.norm(self._displacement))
                self._max_displacement = max(self._max_displacement, displacement_m)
                max_displacement_m = self._max_displacement
            elif self.motion_mode == "tilt":
                offset = self._tilt_offset(pose[:3, :3], self._anchor[:3, :3])
                offset = offset + imu_world_to_sim @ self._translation
                magnitude = float(np.linalg.norm(offset))
                if magnitude > self.max_displacement:
                    offset *= self.max_displacement / magnitude
                self._displacement += self.tilt_smoothing * (offset - self._displacement)
                pose[:3, 3] += self._displacement
                displacement_m = float(np.linalg.norm(self._displacement))
                self._max_displacement = max(self._max_displacement, displacement_m)
                max_displacement_m = self._max_displacement
            else:
                pose[:3, 3] += imu_world_to_sim @ self._displacement
            self._last_pose = pose.copy()
            displacement_xyz = tuple(float(v) for v in (pose[:3, 3] - self._anchor[:3, 3]))
            return ImuArmSnapshot(pose, self._status, self._last_received, self._samples,
                                  displacement_m, max_displacement_m, self._gesture_steps,
                                  displacement_xyz)


class PalmImuArmSource:
    """Callback adapter for `glove.imu_data_palm()` on an existing connection."""

    def __init__(self, resource, *, estimator=None):
        self.estimator = estimator or PalmImuRetargeter()
        self._subscription = resource.subscribe_with_callback(
            self._on_frame, on_error=self.estimator.fail)
        self._closed = False

    def _on_frame(self, frame):
        try:
            timestamp = int(frame.header.timestamp_us)
            # SDK timestamps are synchronized to host UTC; convert transport
            # age to monotonic time for a proper stale watchdog.
            received = time.monotonic()
            age = max(0.0, time.time() - timestamp / 1e6)
            if timestamp / 1e6 > time.time() + 1.0:
                raise ValueError("timestamp is in the future")
            q = frame.orientation
            a = frame.linear_acceleration
            g = frame.angular_velocity
            self.estimator.update(timestamp, [q.x, q.y, q.z, q.w],
                                  [a.x, a.y, a.z], [g.x, g.y, g.z],
                                  received_at=received - age)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            self.estimator.fail(f"Invalid palm IMU frame: {exc}")

    def read(self, anchor_pose, *, now=None):
        return self.estimator.snapshot(anchor_pose, now=now)

    def recenter(self, anchor_pose):
        self.estimator.recenter(anchor_pose)

    def set_tilt_pivot(self, pivot):
        self.estimator.set_tilt_pivot(pivot)

    def close(self):
        if not self._closed:
            self._closed = True
            self._subscription.close()
