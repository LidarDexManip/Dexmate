"""Focused tests for the optional articulated Vega arm scene."""

from contextlib import redirect_stdout
import io
import unittest

import mujoco
import numpy as np

from wuji_teleop.vega_arm_sim import (
    ARM_ACTUATOR_NAMES,
    ARM_HOME_QPOS,
    ARM_JOINT_NAMES,
    DampedLeastSquaresIK,
    EE_BODY_NAME,
    MujocoArm,
    SyntheticWristPoseSource,
    load_articulated_scene,
    run,
)
from wuji_teleop.sources import SyntheticSource


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
        np.testing.assert_allclose(self.arm.lower, self.model.jnt_range[ids, 0])
        np.testing.assert_allclose(self.arm.upper, self.model.jnt_range[ids, 1])
        self.arm.set_target(self.arm.upper + 1.0)
        np.testing.assert_allclose(self.data.ctrl[self.arm.actuator_ids], self.arm.upper)
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (27, 27, 27))

    def test_bent_neutral_pose_recruits_shoulder_and_elbow_for_wrist_lift(self):
        np.testing.assert_allclose(self.data.qpos[self.arm.qpos_ids], ARM_HOME_QPOS)
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        target_pose = np.eye(4)
        target_pose[:3, :3] = self.data.xmat[ee_id].reshape(3, 3)
        target_pose[:3, 3] = self.data.xpos[ee_id]
        target_pose[2, 3] += 0.08
        target = DampedLeastSquaresIK(self.model, self.data, self.arm).solve(target_pose)
        delta = np.abs(target - ARM_HOME_QPOS)
        self.assertGreater(np.max(delta[:3]), 0.03)  # shoulder
        self.assertGreater(delta[3], 0.03)           # elbow
        self.assertGreater(np.count_nonzero(delta > 0.01), 2)

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


if __name__ == "__main__":
    unittest.main()
