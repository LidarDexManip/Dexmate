"""Bilateral Vega arm + Hand 2 model and control-loop checks."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

from wuji_teleop import vega_bimanual_sim
from wuji_teleop.sources import SyntheticSource
from wuji_teleop.imu_arm_source import ImuArmSnapshot
from wuji_teleop.vega_arm_sim import (
    ARM_JOINT_LIMITS,
    ARM_HOME_QPOS,
    RIGHT_ARM_BIMANUAL_HOME_QPOS,
    RIGHT_ARM_JOINT_LIMITS,
    RIGHT_ARM_JOINT_SIGNS,
    limit_joint_goal,
)
from wuji_teleop.vega_bimanual_sim import (
    SIDES,
    _side_ik,
    arm_joint_names,
    body_names,
    load_bimanual_scene,
    run,
)
from wuji_teleop.vr_pose_source import WristPoseFrame, WristPoseSnapshot
from wuji_teleop.xr_tracker_source import TrackerFrame, TrackerSnapshot


class _MovingPicoHand:
    def __init__(self, side):
        self.side = side
        self.started = None
        self.count = 0

    def read(self):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        self.count += 1
        pose = np.eye(4)
        pose[:3, 3] = [0.3, 0.12 if self.side == "left" else -0.12, 1.0]
        if now - self.started > 0.15:
            pose[2, 3] += 0.04
        frame = WristPoseFrame(pose, 1.0, time.time_ns(), now)
        return WristPoseSnapshot(frame, self.count, self.count, None)


class _MovingTrackers:
    """Two wrist trackers that start on the robot hands, then move forward."""

    def __init__(self, poses):
        self.poses = poses
        self.started = None
        self.count = 0

    def read(self):
        now = time.monotonic()
        if self.started is None:
            self.started = now
        self.count += 1
        poses = {serial: pose.copy() for serial, pose in self.poses.items()}
        if now - self.started > 0.15:
            for pose in poses.values():
                pose[0, 3] += 0.04
        frame = TrackerFrame(poses, None, self.count, now)
        return TrackerSnapshot(frame, self.count, self.count, None)


class _FakeViewer:
    """Viewer stand-in that presses ``key`` once ``press_after`` seconds have passed."""

    def __init__(self, on_key, key, press_after):
        self.on_key, self.key, self.press_after = on_key, key, press_after
        self.started = time.monotonic()
        self.opt = SimpleNamespace(geomgroup=np.ones(6), sitegroup=np.ones(6))
        self.cam = SimpleNamespace(lookat=np.zeros(3), distance=0.0, azimuth=0.0,
                                   elevation=0.0)

    def lock(self):
        return nullcontext()

    def is_running(self):
        return True

    def set_texts(self, texts):
        pass

    def sync(self):
        if self.key is not None and time.monotonic() - self.started >= self.press_after:
            self.on_key(self.key)
            self.key = None


class _NoPoseImu:
    def read(self, anchor_pose, *, now=None):
        return ImuArmSnapshot(None, "calibrating", None, 0, 0.0, 0.0)

    def recenter(self, anchor_pose):
        pass


class BimanualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()

    def test_both_arms_and_hands_have_named_controls_and_limits(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (54, 54, 54))
        for side in SIDES:
            arm = self.arms[side]
            limits = ARM_JOINT_LIMITS if side == "left" else RIGHT_ARM_JOINT_LIMITS
            self.assertEqual(len(arm.actuator_ids), 7)
            self.assertEqual(len(self.hands[side].actuator_ids), 20)
            np.testing.assert_array_equal(
                self.model.jnt_limited[arm.joint_ids].astype(bool),
                np.isfinite(limits).all(axis=1))
            self.assertTrue(np.all(arm.home >= arm.lower))
            self.assertTrue(np.all(arm.home <= arm.upper))

    def test_right_home_and_limits_follow_mirrored_joint_convention(self):
        np.testing.assert_allclose(
            RIGHT_ARM_BIMANUAL_HOME_QPOS, RIGHT_ARM_JOINT_SIGNS * ARM_HOME_QPOS)
        expected_limits = np.sort(
            RIGHT_ARM_JOINT_SIGNS[:, None] * ARM_JOINT_LIMITS, axis=1)
        np.testing.assert_allclose(RIGHT_ARM_JOINT_LIMITS, expected_limits)

        left_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_names("left")["ee"])
        right_ee = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_names("right")["ee"])
        reflection = np.diag([1.0, -1.0, 1.0])
        np.testing.assert_allclose(
            self.data.xpos[right_ee], reflection @ self.data.xpos[left_ee], atol=3e-4)
        np.testing.assert_allclose(
            self.data.xmat[right_ee].reshape(3, 3),
            reflection @ self.data.xmat[left_ee].reshape(3, 3) @ reflection,
            atol=1e-8)

    def test_each_hand_is_attached_to_matching_end_effector_and_points_out(self):
        for side in SIDES:
            names = body_names(side)
            wrist = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, names["hand"])
            ee = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, names["ee"])
            self.assertEqual(self.model.body_parentid[wrist], ee)
            finger_axis = -self.data.xmat[wrist].reshape(3, 3)[:, 2]
            outward_axis = self.data.xmat[ee].reshape(3, 3)[:, 2]
            self.assertGreater(float(finger_axis @ outward_axis), 0.99)

    def test_right_and_left_joint_names_are_distinct(self):
        self.assertTrue(set(arm_joint_names("left")).isdisjoint(
            arm_joint_names("right")))

    def test_both_upper_ik_solvers_reach_one_axis_targets(self):
        for side in SIDES:
            arm = self.arms[side]
            ee = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)["ee"])
            origin = self.data.xpos[ee].copy()
            solver = _side_ik(self.model, self.data, arm, upper_only=True)
            outward = 0.04 if side == "left" else -0.04
            for delta in (np.array([0.04, 0.0, 0.0]),
                          np.array([0.0, 0.0, 0.04]),
                          np.array([0.0, outward, 0.0])):
                with self.subTest(side=side, delta=delta):
                    target = np.eye(4)
                    target[:3, :3] = self.data.xmat[ee].reshape(3, 3)
                    target[:3, 3] = origin + delta
                    goal = solver.solve(target, seed=arm.home)
                    solver.scratch.qpos[:] = self.data.qpos
                    solver.scratch.qpos[arm.qpos_ids] = goal
                    mujoco.mj_forward(self.model, solver.scratch)
                    np.testing.assert_allclose(
                        solver.scratch.xpos[ee] - origin, delta, atol=1e-3)

    def test_complete_joint_goal_change_is_hard_limited(self):
        seed = np.zeros(7)
        bounded = limit_joint_goal(np.ones(7), seed, max_total_step=0.22)
        self.assertAlmostEqual(np.linalg.norm(bounded - seed), 0.22)
        np.testing.assert_allclose(
            limit_joint_goal(np.full(7, 0.01), seed, max_total_step=0.22),
            np.full(7, 0.01))

    def test_short_synthetic_run_moves_both_sides_without_warnings(self):
        sources = {side: SyntheticSource(side) for side in SIDES}
        with redirect_stdout(io.StringIO()):
            report = run(sources, mode="synthetic", headless=True, duration=0.5)
        self.assertEqual(report.mujoco_warnings, 0)
        for side in SIDES:
            side_report = report.sides[side]
            self.assertEqual(side_report.finger_status, "tracking")
            self.assertGreater(side_report.finger_updates, 20)
            self.assertGreater(side_report.arm_updates, 10)
            self.assertGreater(max(side_report.arm_joint_peak_delta_rad), 0.01)

    def test_pico_mode_accepts_independent_left_and_right_streams(self):
        sources = {side: SyntheticSource(side) for side in SIDES}
        poses = {side: _MovingPicoHand(side) for side in SIDES}
        imus = {side: _NoPoseImu() for side in SIDES}
        with redirect_stdout(io.StringIO()):
            report = run(sources, mode="pico", pose_sources=poses,
                         imu_sources=imus, headless=True, duration=0.5)
        self.assertEqual(report.mujoco_warnings, 0)
        for side in SIDES:
            side_report = report.sides[side]
            self.assertEqual(side_report.arm_status, "hand tracking")
            self.assertGreater(side_report.arm_updates, 10)
            self.assertIsNotNone(side_report.last_target)
            self.assertLess(side_report.last_ik_error_m, 0.03)
            self.assertIn(side, report.acceptance["sides"])
            self.assertIsNotNone(
                report.acceptance["sides"][side]["ik_p95_under_2cm"])
        self.assertEqual(
            report.acceptance["thresholds"]["ik_error_p95_m"], 0.02)
        self.assertIn("quantitative_pass", report.acceptance)

    def _trackers_on_hands(self):
        poses = {}
        for side, serial in (("left", "SN-L"), ("right", "SN-R")):
            ee = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)["ee"])
            pose = np.eye(4)
            pose[:3, :3] = self.data.xmat[ee].reshape(3, 3)
            pose[:3, 3] = self.data.xpos[ee]
            poses[serial] = pose
        return poses

    def test_tracker_mode_drives_both_arms_through_gmr_ik(self):
        sources = {side: SyntheticSource(side) for side in SIDES}
        with redirect_stdout(io.StringIO()):
            report = run(sources, mode="tracker",
                         tracker_source=_MovingTrackers(self._trackers_on_hands()),
                         headless=True, duration=0.8)
        self.assertEqual(report.mujoco_warnings, 0)
        for side, serial in (("left", "SN-L"), ("right", "SN-R")):
            side_report = report.sides[side]
            self.assertEqual(side_report.arm_status, "tracker tracking")
            self.assertEqual(side_report.tracker_serial, serial)
            self.assertGreater(side_report.arm_updates, 10)
            self.assertLess(side_report.last_ik_error_m, 0.01)
            self.assertLess(side_report.ik_rotation_error_p95_rad, 0.05)
            self.assertGreater(max(side_report.arm_joint_peak_delta_rad), 0.01)
            self.assertIn(side, report.acceptance["sides"])

    def test_tracker_recalibration_returns_both_arms_to_home(self):
        # The trackers move the arms forward, then R pins that moved pose to home.
        sources = {side: SyntheticSource(side) for side in SIDES}

        def viewer(model, data, on_key):
            return nullcontext(_FakeViewer(on_key, ord("R"), press_after=0.6))

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "run.csv"
            with mock.patch.object(vega_bimanual_sim, "passive_viewer", viewer), \
                    redirect_stdout(io.StringIO()):
                report = run(sources, mode="tracker",
                             tracker_source=_MovingTrackers(self._trackers_on_hands()),
                             duration=1.4, log_csv=str(log))
            with log.open() as handle:
                rows = list(csv.DictReader(handle))
        for side in SIDES:
            self.assertGreater(max(report.sides[side].arm_joint_peak_delta_rad), 0.01)
            last = [row for row in rows if row["side"] == side][-1]
            goal = np.array([float(last[f"j{i}"]) for i in range(1, 8)])
            np.testing.assert_allclose(goal, self.arms[side].home, atol=0.01)

    def test_tracker_mode_requires_a_tracker_source(self):
        sources = {side: SyntheticSource(side) for side in SIDES}
        with self.assertRaisesRegex(ValueError, "tracker source"):
            run(sources, mode="tracker", headless=True, duration=0.1)

    def test_one_glove_finger_input_makes_the_other_hand_synthetic(self):
        glove = mock.Mock(serial="WG-L", firmware="1.0.0", address="192.168.1.100:50001")
        with mock.patch.object(vega_bimanual_sim, "_connect_bilateral_gloves",
                               return_value={"left": glove}) as connect, \
                mock.patch.object(vega_bimanual_sim, "run") as run_sim, \
                mock.patch("wuji_teleop.xr_tracker_source.XRoboToolkitTrackerSource"), \
                redirect_stdout(io.StringIO()):
            code = vega_bimanual_sim.main(
                ["--arm-input", "tracker", "--finger-input", "left-glove",
                 "--headless", "--duration", "0.1", "--log-csv", "/dev/null"])
        self.assertEqual(code, 0)
        self.assertEqual(connect.call_args.kwargs["sides"], ("left",))
        sources = run_sim.call_args.args[0]
        self.assertIs(sources["left"], glove)
        self.assertIsInstance(sources["right"], SyntheticSource)

    def test_one_glove_finger_input_needs_tracker_arms(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            vega_bimanual_sim.main(["--finger-input", "right-glove"])


if __name__ == "__main__":
    unittest.main()
