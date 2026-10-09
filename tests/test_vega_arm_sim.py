"""Focused tests for the optional articulated Vega arm scene."""

from contextlib import redirect_stdout
import io
import math
import unittest

import mujoco
import numpy as np

from wuji_teleop.imu_arm_source import ImuArmSnapshot
from wuji_teleop.vega_arm_sim import (
    ARM_ACTUATOR_NAMES,
    ARM_HOME_QPOS,
    ARM_JOINT_LIMITS,
    ARM_JOINT_NAMES,
    DampedLeastSquaresIK,
    EE_BODY_NAME,
    ELBOW_JOINT_INDEX,
    ELBOW_MAX_FLEXION,
    MujocoArm,
    SHOULDER_BODY_NAME,
    SyntheticWristPoseSource,
    desired_elbow_angle,
    desired_shoulder_pitch,
    glove_arm_ik,
    load_articulated_scene,
    shoulder_target_angle,
    run,
)
from wuji_teleop.sources import SyntheticSource


class _RaisedGlovePose:
    def read(self, anchor_pose, *, now=None):
        pose = anchor_pose.copy()
        pose[0, 3] -= 0.20
        pose[2, 3] += 0.05
        return ImuArmSnapshot(pose, "tracking", now, 1, 0.20, 0.20, gesture_steps=1)

    def recenter(self, anchor_pose):
        pass


class _PivotRecordingGlovePose(_RaisedGlovePose):
    pivot = None

    def set_tilt_pivot(self, pivot):
        self.pivot = np.asarray(pivot).copy()


class _TuckedTwistingGlovePose:
    def read(self, anchor_pose, *, now=None):
        pose = anchor_pose.copy()
        pose[0, 3] = 0.30
        pose[2, 3] -= 0.05
        pose[:3, :3] = anchor_pose[:3, :3] @ _rotation_about([0, 0, 1], 0.6)
        return ImuArmSnapshot(pose, "tracking", now, 1, 0.0, 0.0)


def _rotation_about(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


class VegaArmSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arm, cls.hand = load_articulated_scene()

    def setUp(self):
        mujoco.mj_resetData(self.model, self.data)
        self.arm.initialize()
        self.hand.initialize()

    def test_named_joint_mapping_and_limit_clamping(self):
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in ARM_JOINT_NAMES]
        actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                        for n in ARM_ACTUATOR_NAMES]
        self.assertTrue(np.array_equal(self.model.actuator_trnid[actuator_ids, 0], ids))
        limited = np.ones(7, dtype=bool)
        limited[ELBOW_JOINT_INDEX] = False
        np.testing.assert_array_equal(self.model.jnt_limited[ids].astype(bool), limited)
        np.testing.assert_array_equal(
            self.model.actuator_ctrllimited[actuator_ids].astype(bool), limited)
        self.assertTrue(np.all(np.isinf(ARM_JOINT_LIMITS[ELBOW_JOINT_INDEX])))
        np.testing.assert_allclose(self.model.jnt_range[ids][limited], ARM_JOINT_LIMITS[limited])
        np.testing.assert_allclose(self.model.actuator_ctrlrange[actuator_ids][limited],
                                   ARM_JOINT_LIMITS[limited])
        self.assertTrue(np.all(ARM_HOME_QPOS >= ARM_JOINT_LIMITS[:, 0]))
        self.assertTrue(np.all(ARM_HOME_QPOS <= ARM_JOINT_LIMITS[:, 1]))
        self.arm.set_target(np.full(7, 5.0))
        ctrl = self.data.ctrl[self.arm.actuator_ids]
        np.testing.assert_allclose(ctrl[limited], ARM_JOINT_LIMITS[limited, 1])
        self.assertEqual(ctrl[ELBOW_JOINT_INDEX], 5.0)
        self.arm.set_target(np.full(7, -5.0))
        ctrl = self.data.ctrl[self.arm.actuator_ids]
        np.testing.assert_allclose(ctrl[limited], ARM_JOINT_LIMITS[limited, 0])
        self.assertEqual(ctrl[ELBOW_JOINT_INDEX], -5.0)
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (34, 34, 34))

    def test_bent_neutral_pose_recruits_shoulder_and_elbow_for_wrist_lift(self):
        np.testing.assert_allclose(self.data.qpos[self.arm.qpos_ids], ARM_HOME_QPOS)
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        target_pose = np.eye(4)
        target_pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        target_pose[:3, 3] = self.data.xpos[ee_id]
        target_pose[2, 3] += 0.08
        target = DampedLeastSquaresIK(self.model, self.data, self.arm).solve(target_pose)
        delta = np.abs(target - ARM_HOME_QPOS)
        self.assertGreater(np.max(delta[:3]), 0.02)  # shoulder
        self.assertGreater(np.count_nonzero(delta > 0.01), 2)

    def test_home_pose_puts_elbow_down_and_hand_in_front(self):
        shoulder = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)]
        elbow = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")]
        hand = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)]
        right_hand = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "R_ee")]
        self.assertGreater(shoulder[2] - elbow[2], 0.18)
        self.assertGreater(hand[0], shoulder[0] + 0.35)
        self.assertLess(abs(hand[1]), 0.20)
        self.assertGreater(right_hand[0], 0.45)
        self.assertGreater(right_hand[1], -0.15)

    def test_down_and_in_target_folds_the_elbow_inside_the_joint_limits(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        # In toward the torso and a little below the home hand.
        pose[:3, 3] = [0.30, 0.05, home[2] - 0.05]
        solver = glove_arm_ik(self.model, self.data, self.arm)
        home_reach = np.linalg.norm(home - self.data.xpos[shoulder_id])
        for _ in range(12):
            joints = solver.solve(pose)
            self.data.qpos[self.arm.qpos_ids] = joints
            mujoco.mj_forward(self.model, self.data)
        self.assertTrue(np.all(joints >= ARM_JOINT_LIMITS[:, 0] - 1e-9))
        self.assertTrue(np.all(joints <= ARM_JOINT_LIMITS[:, 1] + 1e-9))
        self.assertLess(joints[ELBOW_JOINT_INDEX], ARM_HOME_QPOS[ELBOW_JOINT_INDEX] - 0.80)
        self.assertGreaterEqual(joints[ELBOW_JOINT_INDEX], ELBOW_MAX_FLEXION - 0.15)
        shoulder = self.data.xpos[shoulder_id]
        elbow = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")]
        self.assertGreater(shoulder[2] - elbow[2], 0.12)
        reach = np.linalg.norm(self.data.xpos[ee_id] - shoulder)
        self.assertLess(reach, home_reach - 0.12)
        self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.10)

    def test_full_reach_raise_swings_the_shoulder_with_the_elbow_held(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        home_rotation = self.data.xmat[ee_id].reshape(3, 3).copy()
        solver = glove_arm_ik(self.model, self.data, self.arm)
        pivot = solver.arc_pivot
        home_elbow = ARM_HOME_QPOS[ELBOW_JOINT_INDEX]
        for degrees in range(0, 101, 10):
            # Raise the hand on an arc around the shoulder, tipping with it.
            swing = _rotation_about([0, -1, 0], math.radians(degrees))
            pose = np.eye(4)
            pose[:3, :3] = swing @ home_rotation
            pose[:3, 3] = pivot + swing @ (home - pivot)
            for _ in range(12):
                joints = solver.solve(pose)
                self.data.qpos[self.arm.qpos_ids] = joints
                mujoco.mj_forward(self.model, self.data)
            self.assertGreaterEqual(joints[ELBOW_JOINT_INDEX], home_elbow - 0.15, degrees)
            self.assertLess(abs(joints[2] - ARM_HOME_QPOS[2]), 0.15, degrees)
            self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.03, degrees)
        self.assertLess(joints[0], ARM_HOME_QPOS[0] - 1.6)
        self.assertGreater(self.data.xpos[ee_id][2], home[2] + 0.9)

    def test_elbow_goal_only_folds_toward_the_body(self):
        home = ARM_HOME_QPOS[ELBOW_JOINT_INDEX]
        home_reach = 0.60
        self.assertAlmostEqual(desired_elbow_angle(home_reach, 0.0), home, places=2)
        for reach in np.linspace(0.30, 0.80, 11):
            goals = [desired_elbow_angle(reach, rise) for rise in np.linspace(-0.4, 1.2, 17)]
            # A higher target never adds elbow flexion.
            self.assertTrue(np.all(np.diff(goals) >= -1e-12), reach)
            self.assertGreaterEqual(desired_elbow_angle(reach, 0.5), home - 1e-9)
        # Reaching out at any height opens the elbow.
        self.assertGreater(desired_elbow_angle(0.75, 1.0), home + 0.3)
        self.assertGreater(desired_elbow_angle(0.75, -0.3), home + 0.3)
        # Down and in toward the body folds it.
        self.assertLess(desired_elbow_angle(0.50, -0.05), home - 0.8)
        self.assertLessEqual(desired_elbow_angle(0.50, -0.05), 0.0)

    def test_pulling_the_hand_back_swings_the_shoulder_back(self):
        self.arm.set_target(ARM_HOME_QPOS)
        self.arm.initialize()
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        pose[:3, 3] = [0.30, 0.05, 0.90]
        solver = glove_arm_ik(self.model, self.data, self.arm)
        joints = solver.solve(pose)
        self.assertGreater(joints[0], ARM_HOME_QPOS[0] + 0.30)
        self.assertLessEqual(joints[0], ARM_JOINT_LIMITS[0, 1] + 1e-9)
        self.data.qpos[self.arm.qpos_ids] = joints
        mujoco.mj_forward(self.model, self.data)
        shoulder = self.data.xpos[shoulder_id]
        elbow = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")]
        self.assertGreater(shoulder[2] - elbow[2], 0.04)
        self.assertLess(joints[ELBOW_JOINT_INDEX], -1.20)
        self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.15)
        self.assertLess(self.data.xpos[ee_id][0], home[0] - 0.25)

    def test_forward_and_back_pitch_the_shoulder(self):
        self.arm.set_target(ARM_HOME_QPOS)
        self.arm.initialize()
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        elbow_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")
        home = self.data.xpos[ee_id].copy()
        rotation = self.data.xmat[ee_id].reshape(3, 3).copy()
        solver = glove_arm_ik(self.model, self.data, self.arm)

        pivot = solver.shoulder_pivot
        offset = home - pivot
        radius = math.hypot(offset[0], offset[2])
        home_angle = math.atan2(offset[0], -offset[2])

        def solve_on_arc(swing):
            angle = home_angle + swing
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [pivot[0] + radius * math.sin(angle), home[1],
                           pivot[2] - radius * math.cos(angle)]
            joints = solver.solve(pose)
            self.data.qpos[self.arm.qpos_ids] = joints
            mujoco.mj_forward(self.model, self.data)
            error = np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3])
            elbow_forward = self.data.xpos[elbow_id][0] - self.data.xpos[shoulder_id][0]
            self.arm.set_target(ARM_HOME_QPOS)
            self.arm.initialize()
            return joints, error, elbow_forward

        forward, forward_error, forward_elbow = solve_on_arc(0.35)
        back, back_error, back_elbow = solve_on_arc(-0.35)
        # On an arc at home reach the shoulder swings 1:1 and the elbow holds.
        self.assertLess(forward[0], ARM_HOME_QPOS[0] - 0.30)
        self.assertGreater(back[0], ARM_HOME_QPOS[0] + 0.30)
        for joints in (forward, back):
            self.assertLess(abs(joints[ELBOW_JOINT_INDEX] - ARM_HOME_QPOS[ELBOW_JOINT_INDEX]), 0.20)
        self.assertLess(forward_error, 0.05)
        self.assertLess(back_error, 0.05)
        self.assertGreater(forward_elbow, back_elbow + 0.10)

    def test_shoulder_pitch_goal_follows_the_target_direction(self):
        home = 1.0
        self.assertAlmostEqual(desired_shoulder_pitch(home, home), ARM_HOME_QPOS[0])
        self.assertAlmostEqual(desired_shoulder_pitch(home + 0.30, home), ARM_HOME_QPOS[0] - 0.30)
        self.assertAlmostEqual(desired_shoulder_pitch(home - 0.30, home), ARM_HOME_QPOS[0] + 0.30)
        self.assertAlmostEqual(desired_shoulder_pitch(home - 3.0, home), ARM_JOINT_LIMITS[0, 1])
        self.assertAlmostEqual(desired_shoulder_pitch(home + 3.0, home), ARM_JOINT_LIMITS[0, 0])
        # A more bent elbow already lifts the hand, so the shoulder backs off.
        home_elbow = ARM_HOME_QPOS[ELBOW_JOINT_INDEX]
        self.assertGreater(desired_shoulder_pitch(home, home, home_elbow - 0.5), ARM_HOME_QPOS[0] + 0.2)
        self.assertLess(desired_shoulder_pitch(home, home, home_elbow + 0.5), ARM_HOME_QPOS[0] - 0.2)
        self.assertAlmostEqual(shoulder_target_angle([0, 0, 1], [0, 0, 0]), 0.0)
        self.assertAlmostEqual(shoulder_target_angle([0, 0, 1], [1, 0, 1]), math.pi / 2)
        # Above and slightly behind the shoulder continues past pi, not to -pi.
        self.assertGreater(shoulder_target_angle([0, 0, 0], [-0.05, 0, 1]), math.pi)

    def test_raising_the_hand_uses_the_shoulder_not_the_wrist(self):
        self.arm.set_target(ARM_HOME_QPOS)
        self.arm.initialize()
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        pose[:3, 3] = [home[0], home[1], home[2] + 0.14]
        solver = glove_arm_ik(self.model, self.data, self.arm)
        joints = solver.solve(pose)
        shoulder_raise = ARM_HOME_QPOS[0] - joints[0]
        wrist_pitch = abs(joints[5] - ARM_HOME_QPOS[5])
        self.assertGreater(shoulder_raise, 0.15)
        self.assertLess(wrist_pitch, 0.35)
        self.assertTrue(np.all(joints >= ARM_JOINT_LIMITS[:, 0] - 1e-9))
        self.assertTrue(np.all(joints <= ARM_JOINT_LIMITS[:, 1] + 1e-9))
        self.assertLessEqual(joints[ELBOW_JOINT_INDEX], 0.0)
        self.data.qpos[self.arm.qpos_ids] = joints
        mujoco.mj_forward(self.model, self.data)
        self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.15)

    def test_zero_target_holds_arm_without_mujoco_warnings(self):
        initial = self.data.qpos[self.arm.qpos_ids].copy()
        for _ in range(400):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertLess(np.max(np.abs(self.data.qpos[self.arm.qpos_ids] - initial)), 0.08)
        self.assertEqual(int(np.sum(self.data.warning.number)), 0)

    def test_reachable_synthetic_target_moves_inside_limits_smoothly(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        origin = np.eye(4)
        origin[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        origin[:3, 3] = self.data.xpos[ee_id]
        target_pose = SyntheticWristPoseSource(origin).read(0.8)
        solver = DampedLeastSquaresIK(self.model, self.data, self.arm)
        live_before = self.data.qpos.copy()
        target = solver.solve(target_pose)
        # Solving is side-effect free; movement occurs through the arm actuator.
        np.testing.assert_array_equal(self.data.qpos, live_before)
        self.assertTrue(np.all(target >= self.arm.lower))
        self.assertTrue(np.all(target <= self.arm.upper))
        self.arm.set_target(target)
        initial_pose = origin[:3, 3].copy()
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
        final_pose = self.data.xpos[ee_id].copy()
        self.assertGreater(np.linalg.norm(final_pose - initial_pose), 0.005)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertEqual(int(np.sum(self.data.warning.number)), 0)

    def test_combined_synthetic_arm_and_fingers_use_separate_rates(self):
        output = io.StringIO()
        with redirect_stdout(output):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         duration=0.25, headless=True)
        self.assertGreaterEqual(report.finger_updates, 20)
        self.assertGreaterEqual(report.arm_ik_updates, 8)
        self.assertGreater(report.simulation_seconds, 0.20)
        self.assertEqual(report.finger_status, "tracking")
        self.assertEqual(report.mujoco_warnings, 0)
        self.assertEqual(len(report.arm_joint_peak_delta_rad), 7)
        self.assertGreater(np.count_nonzero(np.asarray(report.arm_joint_peak_delta_rad) > 0.01), 2)
        self.assertIn('"arm_ik_updates"', output.getvalue())

    def test_glove_arm_ik_moves_shoulder_and_elbow(self):
        with redirect_stdout(io.StringIO()):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="glove-arm", arm_imu_source=_RaisedGlovePose(),
                         duration=0.8, headless=True)
        delta = np.asarray(report.arm_joint_peak_delta_rad)
        self.assertEqual(len(delta), 7)
        self.assertGreater(report.arm_ik_updates, 8)
        self.assertGreater(np.max(delta[:3]), 0.02)
        self.assertGreater(delta[3], 0.02)
        self.assertEqual(report.arm_status, "IK")
        self.assertEqual(report.mujoco_warnings, 0)

    def test_glove_arm_swings_tilt_targets_around_the_shoulder(self):
        source = _PivotRecordingGlovePose()
        with redirect_stdout(io.StringIO()):
            run(SyntheticSource("left"), finger_input="synthetic",
                arm_input="glove-arm", arm_imu_source=source,
                duration=0.05, headless=True)
        self.assertIsNotNone(source.pivot)
        shoulder = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)]
        self.assertLess(np.linalg.norm(source.pivot - shoulder), 0.08)

    def test_glove_arm_motion_is_slew_limited(self):
        with redirect_stdout(io.StringIO()):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="glove-arm", arm_imu_source=_RaisedGlovePose(),
                         duration=0.2, headless=True)
        delta = np.asarray(report.arm_joint_peak_delta_rad)
        self.assertGreater(np.max(delta), 0.01)
        self.assertLess(np.max(delta), 0.40)

    def test_glove_wrist_twists_while_elbow_tucks(self):
        with redirect_stdout(io.StringIO()):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="glove-arm", arm_imu_source=_TuckedTwistingGlovePose(),
                         duration=0.25, headless=True)
        delta = np.asarray(report.arm_joint_peak_delta_rad)
        self.assertGreater(delta[3], 0.05)
        self.assertGreater(delta[4], 0.05)
        self.assertLess(delta[0], 0.04)
        self.assertEqual(report.mujoco_warnings, 0)


if __name__ == "__main__":
    unittest.main()
