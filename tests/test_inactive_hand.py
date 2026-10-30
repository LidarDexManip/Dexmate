import unittest
import mujoco
import numpy as np
from wuji_teleop.inactive_hand import hand_bound, InactiveHandChecker
from wuji_teleop.vega_bimanual_sim import load_bimanual_scene
from wuji_teleop.collision_check import ArmClearanceChecker, join_arms
from wuji_teleop.hardware_teleop import reset_pose


class InactiveHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()

    def test_envelope_contains_random_finger_configurations(self):
        root, radius, geoms = hand_bound(self.model, 'left')
        random = np.random.default_rng(42)
        hand = self.hands['left']
        for _ in range(30):
            self.data.qpos[hand.qpos_ids] = random.uniform(hand.lower, hand.upper)
            mujoco.mj_forward(self.model, self.data)
            center = self.data.xpos[root]
            for g in geoms:
                mesh = self.model.geom_dataid[g]
                first, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                points = self.model.mesh_vert[first:first+count] @ self.data.geom_xmat[g].reshape(3,3).T + self.data.geom_xpos[g]
                self.assertLessEqual(np.linalg.norm(points-center, axis=1).max(), radius+1e-6)

    def test_envelope_collision_is_not_ignored(self):
        checker = InactiveHandChecker(ArmClearanceChecker(self.model, self.arms, self.hands), 'right', ('left',))
        root, radius = checker.envelopes['left']
        checker.envelopes['left'] = (root, radius + 2)
        result = checker.clearance(join_arms(reset_pose(0)))
        self.assertLess(result.distance_m, 0)
        self.assertIn('envelope', result.pair)
        self.assertTrue(all(not any(int(g) in checker.unknown_hand_geoms for g in pair)
                            for pair in checker.checker._pairs))


class ConvexDistanceTests(unittest.TestCase):
    def test_thin_link_is_not_treated_as_its_large_bounding_sphere(self):
        from wuji_teleop.inactive_hand import ConvexPointDistance
        vertices = np.array([[x, y, z] for x in (-.2, .2) for y in (-.01, .01) for z in (-.01, .01)])
        hull = ConvexPointDistance(vertices)
        center, hand_radius = np.array([0., .24, 0.]), .1
        sphere_gap = np.linalg.norm(center) - np.linalg.norm(vertices[0]) - hand_radius
        self.assertLess(sphere_gap, .03)
        self.assertAlmostEqual(hull.distance(center) - hand_radius, .13)
        self.assertEqual(hull.distance([0, 0, 0]), 0)
        self.assertLess(hull.distance([0, .05, 0]) - hand_radius, 0)

    def test_box_face_edge_corner_distances(self):
        from wuji_teleop.inactive_hand import ConvexPointDistance
        vertices = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        hull = ConvexPointDistance(vertices)
        self.assertAlmostEqual(hull.distance([2,0,0]), 1)
        self.assertAlmostEqual(hull.distance([2,2,0]), np.sqrt(2))
        self.assertAlmostEqual(hull.distance([2,2,2]), np.sqrt(3))
