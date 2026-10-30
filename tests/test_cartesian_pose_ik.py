"""Precision regressions for GUI pose IK, independent of robot hardware."""
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from wuji_teleop.ee_web import cartesian_pose_ik, apply_hand_mounts
from wuji_teleop.vega_bimanual_sim import load_bimanual_scene, body_names
from wuji_teleop.hardware_teleop import reset_pose


class CartesianPoseIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()
        apply_hand_mounts(cls.model, cls.data)

    def pose(self, side, q):
        arm = self.arms[side]
        self.data.qpos[arm.qpos_ids] = q
        mujoco.mj_forward(self.model, self.data)
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)['ee'])
        pose = np.eye(4)
        pose[:3,3] = self.data.xpos[ee]
        pose[:3,:3] = self.data.xmat[ee].reshape(3,3)
        return pose

    def test_same_nonhome_pose_does_not_move_toward_home(self):
        for side in ('left', 'right'):
            arm = self.arms[side]
            q = np.clip(reset_pose(0)[side] + [.2,-.3,.2,-.2,.4,-.2,.1], arm.lower, arm.upper)
            target = self.pose(side, q)
            solved = cartesian_pose_ik(self.model, self.data, arm).solve(target, seed=q)
            np.testing.assert_allclose(solved, q, atol=1e-10)

    def test_small_reachable_moves_preserve_pose_accuracy_and_local_branch(self):
        for side in ('left', 'right'):
            arm = self.arms[side]
            for offset in ([0,.2,.15,0,.3,.1,.2], [.2,-.3,.2,-.2,.4,-.2,.1]):
                seed = np.clip(reset_pose(0)[side] + offset, arm.lower, arm.upper)
                for joint in range(7):
                    with self.subTest(side=side, joint=joint, offset=offset):
                        nearby = seed.copy()
                        nearby[joint] += .01
                        nearby = np.clip(nearby, arm.lower, arm.upper)
                        target = self.pose(side, nearby)  # Known reachable target from FK.
                        self.pose(side, seed)
                        solved = cartesian_pose_ik(self.model, self.data, arm).solve(target, seed=seed)
                        actual = self.pose(side, solved)
                        self.assertLess(np.linalg.norm(actual[:3,3]-target[:3,3]), .0005)
                        angle = Rotation.from_matrix(target[:3,:3] @ actual[:3,:3].T).magnitude()
                        self.assertLess(angle, np.deg2rad(.1))
                        self.assertLess(np.max(np.abs(solved-seed)), .05)
                        self.assertTrue(np.all(solved >= arm.lower) and np.all(solved <= arm.upper))
