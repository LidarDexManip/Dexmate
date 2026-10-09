"""Focused tests for the optional articulated Vega arm scene."""

from contextlib import redirect_stdout
import io
import unittest

import mujoco
import numpy as np

from wuji_teleop.imu_arm_source import ImuArmSnapshot
from wuji_teleop.vega_arm_sim import (
    ARM_ACTUATOR_NAMES,
    ARM_HOME_QPOS,
    ARM_JOINT_NAMES,
    DampedLeastSquaresIK,
    EE_BODY_NAME,
    MujocoArm,
    SHOULDER_BODY_NAME,
    SyntheticWristPoseSource,
    desired_shoulder_pitch,
    load_articulated_scene,
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
        self.assertFalse(np.any(self.model.jnt_limited[ids]))
        past_old_elbow_stop = self.arm.home.copy()
        past_old_elbow_stop[3] = 1.5  # hardware stop used to be 0.244
        self.arm.set_target(past_old_elbow_stop)
        np.testing.assert_allclose(self.data.ctrl[self.arm.actuator_ids], past_old_elbow_stop)
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

    def test_inward_wrist_target_flexes_the_elbow(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        home_rotation = self.data.xmat[ee_id].reshape(3, 3).copy()
        toward_shoulder = self.data.xpos[shoulder_id] - home
        toward_shoulder /= np.linalg.norm(toward_shoulder)
        solver = DampedLeastSquaresIK(self.model, self.data, self.arm, elbow_weight=0.85,
                                       orientation_weight=0.25, max_step=0.14, iterations=16)

        def solve(distance):
            pose = np.eye(4)
            pose[:3, :3] = home_rotation
            pose[:3, 3] = home + distance * toward_shoulder
            joints = solver.solve(pose)
            self.data.qpos[self.arm.qpos_ids] = joints
            mujoco.mj_forward(self.model, self.data)
            error = np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3])
            self.data.qpos[self.arm.qpos_ids] = self.arm.home
            mujoco.mj_forward(self.model, self.data)
            return joints, error

        extended, extended_error = solve(0.0)
        flexed, flexed_error = solve(0.22)
        self.assertLess(flexed[3], -1.6)
        self.assertLess(flexed[3], extended[3] - 0.45)
        self.assertLess(extended_error, 0.05)
        self.assertLess(flexed_error, 0.12)
        flexed_pose = np.eye(4)
        flexed_pose[:3, :3] = home_rotation
        flexed_pose[:3, 3] = home + 0.22 * toward_shoulder
        joints = solver.solve(flexed_pose)
        self.data.qpos[self.arm.qpos_ids] = joints
        mujoco.mj_forward(self.model, self.data)
        shoulder = self.data.xpos[shoulder_id]
        elbow = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")]
        self.assertGreater(shoulder[2] - elbow[2], 0.12)

    def test_hand_in_front_of_the_body_flexes_the_elbow(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        pose[:3, 3] = [0.30, 0.05, 0.90]
        solver = DampedLeastSquaresIK(
            self.model, self.data, self.arm, max_step=0.12, iterations=18,
            orientation_weight=0.10, elbow_weight=0.40, elbow_down_weight=0.15,
            position_weight=2.2, orient_fade_distance=0.20,
            distal_position_scale=0.0, wrist_pitch_scale=0.15, shoulder_weight=1.6)
        joints = solver.solve(pose)
        self.assertGreater(joints[0], ARM_HOME_QPOS[0] + 0.55)
        self.data.qpos[self.arm.qpos_ids] = joints
        mujoco.mj_forward(self.model, self.data)
        shoulder = self.data.xpos[shoulder_id]
        elbow = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")]
        # The wide backward shoulder swing lifts the elbow, but it stays below the shoulder.
        self.assertGreater(shoulder[2] - elbow[2], 0.04)
        self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.10)
        self.assertGreater(np.linalg.norm(home - pose[:3, 3]), 0.20)

    def test_forward_and_back_pitch_the_shoulder(self):
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        elbow_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_arm_l4")
        home = self.data.xpos[ee_id].copy()
        rotation = self.data.xmat[ee_id].reshape(3, 3).copy()
        solver = DampedLeastSquaresIK(
            self.model, self.data, self.arm, max_step=0.12, iterations=18,
            orientation_weight=0.10, elbow_weight=0.40, elbow_down_weight=0.15,
            position_weight=2.2, orient_fade_distance=0.20,
            distal_position_scale=0.0, wrist_pitch_scale=0.15, shoulder_weight=1.6)

        def solve_at(x):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [x, home[1], home[2]]
            return solver.solve(pose)

        def elbow_forward(joints):
            self.data.qpos[self.arm.qpos_ids] = joints
            mujoco.mj_forward(self.model, self.data)
            return (self.data.xpos[elbow_id][0] - self.data.xpos[shoulder_id][0])

        back = solve_at(home[0] - 0.18)
        forward = solve_at(home[0] + 0.15)
        self.assertGreater(back[0], ARM_HOME_QPOS[0] + 0.40)
        self.assertLess(forward[0], ARM_HOME_QPOS[0] - 0.40)
        self.assertGreater(back[0] - forward[0], 0.90)
        self.assertGreater(elbow_forward(forward), elbow_forward(back) + 0.15)

    def test_shoulder_pitch_goal_covers_a_wide_range(self):
        self.assertAlmostEqual(desired_shoulder_pitch(0.58), ARM_HOME_QPOS[0])
        # A 10 cm tilt used to stop joint 1 near 1.0 rad.
        self.assertLess(desired_shoulder_pitch(0.68), 0.85)
        self.assertGreater(desired_shoulder_pitch(0.48), 1.85)
        self.assertLess(desired_shoulder_pitch(1.20), 0.0)
        self.assertGreater(desired_shoulder_pitch(0.10), 2.60)

    def test_raising_the_hand_curls_the_elbow_not_the_wrist(self):
        self.arm.set_target(ARM_HOME_QPOS)
        self.arm.initialize()
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        home = self.data.xpos[ee_id].copy()
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        pose[:3, 3] = [home[0], home[1], home[2] + 0.14]
        solver = DampedLeastSquaresIK(
            self.model, self.data, self.arm, max_step=0.12, iterations=18,
            orientation_weight=0.10, elbow_weight=0.40, elbow_down_weight=0.15,
            position_weight=2.2, orient_fade_distance=0.20,
            distal_position_scale=0.0, wrist_pitch_scale=0.15, shoulder_weight=1.6)
        joints = solver.solve(pose)
        elbow_curl = ARM_HOME_QPOS[3] - joints[3]
        wrist_pitch = abs(joints[5] - ARM_HOME_QPOS[5])
        self.assertGreater(elbow_curl, 0.40)
        self.assertLess(wrist_pitch, 0.35)
        self.assertLess(wrist_pitch, elbow_curl)
        self.data.qpos[self.arm.qpos_ids] = joints
        mujoco.mj_forward(self.model, self.data)
        self.assertLess(np.linalg.norm(self.data.xpos[ee_id] - pose[:3, 3]), 0.08)

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

    def test_glove_arm_motion_is_slew_limited(self):
        with redirect_stdout(io.StringIO()):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="glove-arm", arm_imu_source=_RaisedGlovePose(),
                         duration=0.2, headless=True)
        delta = np.asarray(report.arm_joint_peak_delta_rad)
        self.assertGreater(np.max(delta), 0.01)
        self.assertLess(np.max(delta), 0.40)


if __name__ == "__main__":
    unittest.main()
