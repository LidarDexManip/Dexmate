"""Pico hand tracking places Vega's arm; the palm IMU only aims the wrist."""

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from wuji_teleop.imu_arm_source import ImuArmSnapshot
from wuji_teleop.sources import SyntheticSource
from wuji_teleop.vega_arm_sim import (
    PICO_CSV_HEADER, PicoHandArmMapper, run,
)
from wuji_teleop.vr_pose_source import WristPoseFrame, WristPoseSnapshot


def _snapshot(position, received_at, *, count=1, error=None):
    pose = np.eye(4)
    pose[:3, 3] = position
    frame = WristPoseFrame(pose, 1.0, time.time_ns(), received_at)
    return WristPoseSnapshot(frame, count, count, error)


class _FakeHand:
    """Fresh frames at a base position, shifted by ``offset`` after ``shift_after`` s."""

    def __init__(self, offset=(0.0, 0.0, 0.0), shift_after=0.3, stale=False):
        self.base = np.array([0.2, 0.3, 1.1])
        self.offset = np.asarray(offset, dtype=np.float64)
        self.shift_after = shift_after
        self.stale = stale
        self.started = None
        self.count = 0
        self.closed = False

    def read(self):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        self.count += 1
        shifted = now - self.started > self.shift_after
        position = self.base + (self.offset if shifted else 0.0)
        return _snapshot(position, now - 1.0 if self.stale else now, count=self.count)

    def close(self):
        self.closed = True


class _NoFrames:
    def read(self):
        return WristPoseSnapshot(None, 0, 0, None)

    def close(self):
        pass


class _NoPoseImu:
    def __init__(self):
        self.recentered = 0

    def read(self, anchor_pose, *, now=None):
        return ImuArmSnapshot(None, "calibrating", None, 0, 0.0, 0.0)

    def recenter(self, anchor_pose):
        self.recentered += 1


class PicoHandArmMapperTests(unittest.TestCase):
    def test_first_fresh_frame_calibrates_then_tracks_scaled_offset(self):
        mapper = PicoHandArmMapper(scale=2.0)
        ee = np.array([0.5, 0.1, 0.9])
        target = mapper.update(_snapshot([1.0, 2.0, 3.0], 10.0), ee, 10.0)
        np.testing.assert_allclose(target, ee)
        self.assertEqual(mapper.status, "calibrated")
        target = mapper.update(_snapshot([1.05, 2.0, 2.99], 10.1), ee + 1, 10.1)
        np.testing.assert_allclose(target, ee + [0.10, 0.0, -0.02])
        self.assertEqual(mapper.status, "tracking")

    def test_stale_missing_and_invalid_hold_and_keep_origin(self):
        mapper = PicoHandArmMapper()
        ee = np.zeros(3)
        self.assertIsNone(mapper.update(WristPoseSnapshot(None, 0, 0, None), ee, 1.0))
        self.assertEqual(mapper.status, "waiting")
        mapper.update(_snapshot([1.0, 0.0, 0.0], 1.0), ee, 1.0)
        self.assertIsNone(mapper.update(_snapshot([1.2, 0.0, 0.0], 1.0), ee, 1.3))
        self.assertEqual(mapper.status, "holding")
        self.assertIsNone(mapper.update(
            _snapshot([1.2, 0.0, 0.0], 1.3, error="bad"), ee, 1.3))
        self.assertEqual(mapper.status, "invalid")
        target = mapper.update(_snapshot([1.2, 0.0, 0.0], 1.4), ee + 5, 1.4)
        np.testing.assert_allclose(target, [0.2, 0.0, 0.0])
        self.assertEqual(mapper.status, "tracking")

    def test_recenter_recaptures_on_next_fresh_frame(self):
        mapper = PicoHandArmMapper()
        mapper.update(_snapshot([0.0, 0.0, 0.0], 1.0), np.zeros(3), 1.0)
        mapper.recenter()
        target = mapper.update(_snapshot([0.3, 0.0, 0.0], 1.1), np.ones(3), 1.1)
        np.testing.assert_allclose(target, np.ones(3))
        self.assertEqual(mapper.status, "calibrated")

    def test_rejects_non_positive_scale(self):
        with self.assertRaises(ValueError):
            PicoHandArmMapper(scale=0.0)


class PicoHandRunTests(unittest.TestCase):
    def _run(self, hand, duration, log_csv=None):
        output = io.StringIO()
        with redirect_stdout(output):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="pico-hand", pose_source=hand,
                         arm_imu_source=_NoPoseImu(), headless=True,
                         duration=duration, log_csv=log_csv)
        return report, output.getvalue()

    def test_hand_motion_moves_end_effector_and_logs_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pico.csv"
            report, output = self._run(
                _FakeHand(offset=(0.10, 0.0, 0.0), shift_after=0.6), 3.0, path)
            self.assertIn(str(path), output)
            with path.open() as handle:
                lines = handle.read().splitlines()
        self.assertEqual(lines[0], PICO_CSV_HEADER)
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
        self.assertGreater(len(rows), 20)
        self.assertEqual(report.pico_hand["rows_written"], len(rows))
        np.testing.assert_allclose(report.pico_hand["last_target"],
                                   np.add(report.pico_hand["ee_reference"], [0.10, 0, 0]))
        self.assertLess(report.pico_hand["last_ik_error_m"], 0.01)
        settled = [row for row in rows if float(row["t"]) < 0.55][-1]
        before = np.array([float(settled[name]) for name in ("ee_x", "ee_y", "ee_z")])
        moved = np.array(report.pico_hand["last_ee"]) - before
        np.testing.assert_allclose(moved, [0.10, 0.0, 0.0], atol=0.03)
        for row in rows:
            for name in ("j5", "j6", "j7"):
                self.assertEqual(float(row[name]), 0.0)
        self.assertEqual(report.mujoco_warnings, 0)

    def test_stale_hand_frames_hold_the_arm(self):
        baseline, _ = self._run(_NoFrames(), 0.5)
        self.assertEqual(baseline.arm_status, "hand waiting")
        report, _ = self._run(_FakeHand(stale=True), 0.5)
        self.assertEqual(report.arm_ik_updates, 0)
        self.assertEqual(report.arm_status, "hand holding")
        # Only the home servo's gravity sag, same as never receiving a frame.
        np.testing.assert_allclose(report.arm_joint_peak_delta_rad,
                                   baseline.arm_joint_peak_delta_rad, atol=0.005)
        self.assertEqual(report.pico_hand["rows_written"], 0)
        self.assertIsNone(report.pico_hand["last_target"])

    def test_requires_hand_and_imu_sources(self):
        with self.assertRaises(ValueError):
            run(SyntheticSource("left"), finger_input="synthetic", arm_input="pico-hand",
                arm_imu_source=_NoPoseImu(), headless=True, duration=0.1)
        with self.assertRaises(ValueError):
            run(SyntheticSource("left"), finger_input="synthetic", arm_input="pico-hand",
                pose_source=_FakeHand(), headless=True, duration=0.1)


if __name__ == "__main__":
    unittest.main()
