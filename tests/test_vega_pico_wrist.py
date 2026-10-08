"""Pico wrist-only mode must physically freeze Vega's shoulder and elbow."""

from contextlib import redirect_stdout
import io
import time
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wuji_teleop.imu_arm_source import ImuArmSnapshot
from wuji_teleop.sources import SyntheticSource
from wuji_teleop.vega_arm_sim import (
    ARM_JOINT_NAMES, EE_BODY_NAME, WRIST_JOINT_NAMES,
    WristOrientationIK, load_articulated_scene, run,
)
from wuji_teleop.vr_pose_source import WristPoseFrame, WristPoseSnapshot


class _MovingPico:
    def __init__(self):
        self.count = 0

    def read(self):
        self.count += 1
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(
            [0, 0, min(self.count, 16) * 0.012]).as_matrix()
        frame = WristPoseFrame(pose, 1.0, time.time_ns(), time.monotonic())
        return WristPoseSnapshot(frame, self.count, self.count, None)


class _MovingGloveImu:
    def __init__(self):
        self.count = 0

    def read(self, anchor_pose, *, now=None):
        self.count += 1
        pose = anchor_pose.copy()
        pose[:3, :3] = Rotation.from_rotvec(
            [0, 0, min(self.count, 16) * 0.012]).as_matrix() @ anchor_pose[:3, :3]
        # Deliberately impossible translation: wrist-only control must ignore it.
        pose[:3, 3] += [100, 100, 100]
        return ImuArmSnapshot(pose, "tracking", now, self.count, 100, 100)

    def recenter(self, anchor_pose):
        pass


class PicoWristOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.wrist, cls.hand = load_articulated_scene(wrist_only=True)

    def setUp(self):
        mujoco.mj_resetData(self.model, self.data)
        self.wrist.initialize()
        self.hand.initialize()

    def test_only_three_wrist_joints_exist_and_rotate_the_hand(self):
        self.assertEqual((self.model.nq, self.model.nv, self.model.nu), (23, 23, 23))
        for name in ARM_JOINT_NAMES[:4]:
            self.assertEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), -1)
        for name in WRIST_JOINT_NAMES:
            self.assertGreaterEqual(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name), 0)
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        start = self.data.xmat[ee].reshape(3, 3).copy()
        target_rotation = Rotation.from_rotvec([0, 0, 0.2]).as_matrix() @ start
        target = WristOrientationIK(self.model, self.data, self.wrist).solve(target_rotation)
        self.assertGreater(np.max(np.abs(target)), 0.05)
        self.wrist.set_target(target)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
        achieved = self.data.xmat[ee].reshape(3, 3)
        angle_error = np.linalg.norm(Rotation.from_matrix(
            target_rotation @ achieved.T).as_rotvec())
        self.assertLess(angle_error, 0.05)
        self.assertEqual(int(np.sum(self.data.warning.number)), 0)

    def test_pico_clutch_updates_only_wrist_scene(self):
        output = io.StringIO()
        with redirect_stdout(output):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="pico-wrist", pose_source=_MovingPico(),
                         headless=True, duration=0.5)
        self.assertGreater(report.arm_ik_updates, 5)
        self.assertEqual(len(report.arm_joint_peak_delta_rad), 3)
        self.assertGreater(np.max(report.arm_joint_peak_delta_rad), 0.01)
        self.assertEqual(report.mujoco_warnings, 0)

    def test_glove_imu_rotation_ignores_translation_and_uses_only_wrist(self):
        output = io.StringIO()
        with redirect_stdout(output):
            report = run(SyntheticSource("left"), finger_input="synthetic",
                         arm_input="glove-wrist", arm_imu_source=_MovingGloveImu(),
                         headless=True, duration=0.5)
        self.assertGreater(report.arm_ik_updates, 5)
        self.assertEqual(len(report.arm_joint_peak_delta_rad), 3)
        self.assertGreater(np.max(report.arm_joint_peak_delta_rad), 0.01)
        self.assertEqual(report.mujoco_warnings, 0)


if __name__ == "__main__":
    unittest.main()
