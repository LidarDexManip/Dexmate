"""Bilateral Vega arm + Hand 2 model and control-loop checks."""

from contextlib import redirect_stdout
import io
import time
import unittest

import mujoco
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
