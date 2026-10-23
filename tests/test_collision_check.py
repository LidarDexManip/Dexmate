"""Hand-aware arm clearance checks and collision-free reset planning."""

import unittest

import numpy as np

from wuji_teleop.collision_check import (
    ArmClearanceChecker,
    Clearance,
    PathFollower,
    PlanningError,
    ResetPlanner,
    join_arms,
    joint_distance,
    split_arms,
)
from wuji_teleop.hardware_teleop import reset_pose
from wuji_teleop.vega_bimanual_sim import SIDES, load_bimanual_scene


L_SHAPE = {"left": [0.064, 0.3, 0.0, -1.556, 1.271, 0.0, 0.0],
           "right": [-0.064, -0.3, 0.0, -1.556, -1.271, 0.0, 0.0]}


def _shoulders_down(**rolls):
    """Upper arms vertical, forearms forward; ``left``/``right`` set j3 roll."""
    q = reset_pose(0.0)
    for side, roll in rolls.items():
        q[side][2] = roll
    return join_arms(q)


class _BoxChecker:
    """Joint-space obstacle: clearance is the Chebyshev distance to a box."""

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low), np.asarray(high)

    def set_fingers(self, side, angles):
        pass

    def clearance(self, q):
        outside = np.maximum(self.low - q, q - self.high)
        distance = float(np.max(outside))
        return Clearance(min(distance, 0.10), "box" if distance < 0.10 else None)


class ClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, cls.arms, cls.hands = load_bimanual_scene()
        cls.checker = ArmClearanceChecker(cls.model, cls.arms, cls.hands)
        for side in SIDES:
            cls.checker.set_fingers(side, np.zeros(20))

    def test_reset_and_named_poses_are_clear(self):
        for q in (join_arms(reset_pose()), join_arms(L_SHAPE), _shoulders_down()):
            self.assertGreater(self.checker.clearance(q).distance_m, 0.03)

    def test_hand_reaching_into_the_other_arm_is_detected(self):
        clearance = self.checker.clearance(_shoulders_down(left=0.8))
        self.assertLess(clearance.distance_m, 0.0)
        self.assertRegex(clearance.pair, r"^l_\w+ <-> R_arm_l\d$")

    def test_crossed_forearms_are_detected(self):
        clearance = self.checker.clearance(_shoulders_down(left=0.9, right=-0.9))
        self.assertLess(clearance.distance_m, 0.0)
        self.assertRegex(clearance.pair, r"^L_arm_l\d <-> R_arm_l\d$")

    def test_finger_angles_change_hand_clearance(self):
        q = _shoulders_down(left=0.6)
        opened = self.checker.clearance(q).distance_m
        self.checker.set_fingers("left", np.full(20, 1.2))
        try:
            curled = self.checker.clearance(q).distance_m
        finally:
            self.checker.set_fingers("left", np.zeros(20))
        self.assertNotAlmostEqual(opened, curled, places=3)

    def test_finger_angles_are_validated(self):
        with self.assertRaises(ValueError):
            self.checker.set_fingers("left", np.zeros(19))

    def test_planner_uses_straight_path_when_clear(self):
        lower = np.concatenate([self.arms[side].lower for side in SIDES])
        upper = np.concatenate([self.arms[side].upper for side in SIDES])
        planner = ResetPlanner(self.checker, lower, upper, rng=np.random.default_rng(0))
        path = planner.plan(join_arms(L_SHAPE), join_arms(reset_pose()))
        self.assertEqual(len(path), 2)
        np.testing.assert_allclose(path[-1], join_arms(reset_pose()))

    def test_planner_refuses_a_colliding_reset_pose(self):
        lower = np.concatenate([self.arms[side].lower for side in SIDES])
        upper = np.concatenate([self.arms[side].upper for side in SIDES])
        planner = ResetPlanner(self.checker, lower, upper)
        with self.assertRaisesRegex(PlanningError, "Reset pose itself"):
            planner.plan(join_arms(L_SHAPE), _shoulders_down(left=0.8))

    def test_planner_refuses_a_start_that_already_touches(self):
        lower = np.concatenate([self.arms[side].lower for side in SIDES])
        upper = np.concatenate([self.arms[side].upper for side in SIDES])
        planner = ResetPlanner(self.checker, lower, upper)
        with self.assertRaisesRegex(PlanningError, "already touch"):
            planner.plan(_shoulders_down(left=0.9, right=-0.9), join_arms(reset_pose()))


class PlannerLogicTests(unittest.TestCase):
    def setUp(self):
        self.lower, self.upper = np.full(14, -2.0), np.full(14, 2.0)
        low, high = np.full(14, -5.0), np.full(14, 5.0)
        # A wall across joints 0-1 between start and goal, open above joint 1 = 0.6.
        low[:2], high[:2] = (0.4, -2.0), (0.6, 0.6)
        self.checker = _BoxChecker(low, high)

    def _assert_path_clear(self, planner, path, start, goal):
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)
        for a, b in zip(path, path[1:]):
            for alpha in np.linspace(0, 1, 50):
                q = a + (b - a) * alpha
                self.assertGreaterEqual(self.checker.clearance(q).distance_m,
                                        planner.margin - 1e-9)

    def test_rrt_connect_finds_a_detour_around_a_blocked_straight_path(self):
        start, goal = np.zeros(14), np.zeros(14)
        goal[0] = 1.0
        planner = ResetPlanner(self.checker, self.lower, self.upper,
                               rng=np.random.default_rng(3))
        path = planner.plan(start, goal, time_limit=10.0)
        self.assertGreater(len(path), 2)
        self._assert_path_clear(planner, path, start, goal)

    def test_moving_one_arm_first_avoids_the_other_arm(self):
        # Blocked only while left joint 0 and right joint 0 are both mid-way.
        low, high = np.full(14, -5.0), np.full(14, 5.0)
        low[[0, 7]], high[[0, 7]] = (0.3, 0.3), (0.7, 0.7)
        self.checker = _BoxChecker(low, high)
        start, goal = np.zeros(14), np.zeros(14)
        goal[[0, 7]] = 1.0
        planner = ResetPlanner(self.checker, self.lower, self.upper,
                               rng=np.random.default_rng(0))
        path = planner.plan(start, goal)
        self.assertEqual(len(path), 3)
        self._assert_path_clear(planner, path, start, goal)

    def test_cancel_stops_planning(self):
        start, goal = np.zeros(14), np.zeros(14)
        goal[0] = 1.0
        planner = ResetPlanner(self.checker, self.lower, self.upper)
        with self.assertRaisesRegex(PlanningError, "cancelled"):
            planner.plan(start, goal, cancelled=lambda: True)

    def test_close_start_may_only_back_away(self):
        start = np.zeros(14)
        start[0] = 0.39  # 1 cm from the wall, under the 3 cm margin
        goal = np.zeros(14)
        goal[0] = -0.5
        planner = ResetPlanner(self.checker, self.lower, self.upper)
        self.assertEqual(len(planner.plan(start, goal)), 2)


class PathFollowerTests(unittest.TestCase):
    def test_trapezoid_starts_ends_and_respects_speed(self):
        waypoints = [np.zeros(14), np.r_[0.8, np.zeros(13)], np.r_[0.8, 0.6, np.zeros(12)]]
        follower = PathFollower(waypoints, max_velocity=0.3, max_acceleration=0.6)
        np.testing.assert_allclose(follower.at(0.0), waypoints[0])
        np.testing.assert_allclose(follower.at(follower.duration), waypoints[-1])
        np.testing.assert_allclose(follower.at(follower.duration + 1.0), waypoints[-1])
        dt = 0.01
        samples = [follower.at(t) for t in np.arange(0.0, follower.duration + dt, dt)]
        speeds = [joint_distance(a, b) / dt for a, b in zip(samples, samples[1:])]
        self.assertLessEqual(max(speeds), 0.3 + 1e-6)
        self.assertAlmostEqual(follower.length, 1.4)

    def test_short_moves_never_reach_cruise_speed(self):
        follower = PathFollower([np.zeros(14), np.r_[0.01, np.zeros(13)]])
        self.assertLess(follower.duration, 0.5)
        np.testing.assert_allclose(follower.at(follower.duration)[0], 0.01)

    def test_split_and_join_round_trip(self):
        q = np.arange(14.0)
        np.testing.assert_allclose(join_arms(split_arms(q)), q)


if __name__ == "__main__":
    unittest.main()
