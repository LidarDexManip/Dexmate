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
                 gesture_threshold=0.8, gesture_cooldown=0.45):
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
        if (self.motion_mode not in ("relative", "gesture", "orientation") or
                self.gesture_step <= 0 or self.gesture_threshold <= 0 or
                self.gesture_cooldown < 0 or
                self.calibration_seconds <= 0 or self.stale_after <= 0 or
                self.max_speed <= 0 or self.max_displacement <= 0 or
                self.velocity_damping < 0):
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
                # Wrist-only control does not estimate or command translation.
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
            # Small deadband rejects fused-IMU noise. At rest, zero velocity
            # prevents residual bias from accumulating into a position walk.
            if accel_norm <= self.stationary_accel and gyro_norm <= self.stationary_gyro:
                self._velocity[:] = 0
                a_world[:] = 0
            else:
                self._velocity *= math.exp(-self.velocity_damping * dt)
                self._velocity += a_world * dt
                speed = float(np.linalg.norm(self._velocity))
                if speed > self.max_speed:
                    self._velocity *= self.max_speed / speed
            self._displacement += self._velocity * dt
            distance = float(np.linalg.norm(self._displacement))
            if distance > self.max_displacement:
                self._displacement *= self.max_displacement / distance
                # Prevent continuing to push against the hard position bound.
                self._velocity[:] = 0
                distance = self.max_displacement
            self._max_displacement = max(self._max_displacement, distance)
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
            pose[:3, 3] += imu_world_to_sim @ self._displacement
            self._last_pose = pose.copy()
            return ImuArmSnapshot(pose, self._status, self._last_received, self._samples,
                                  displacement_m, max_displacement_m, self._gesture_steps)


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

    def close(self):
        if not self._closed:
            self._closed = True
            self._subscription.close()
