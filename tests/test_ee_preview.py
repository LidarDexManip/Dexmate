"""Preview approval gates and single-arm paths, with no device connections."""
import unittest
from types import SimpleNamespace

import numpy as np

from wuji_teleop.ee_preview import (straight_path, PreviewTeleop, TableChecker, HoldMapping,
                                  EmptySource, _StaticRetargeter)
from wuji_teleop.collision_check import ArmClearanceChecker, PlanningError, join_arms
from wuji_teleop.hardware_teleop import FakeArms, FakeHands, TeleopLimits, reset_pose
from wuji_teleop.vega_bimanual_sim import SIDES, load_bimanual_scene


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()

    def setUp(self):
        self.q = reset_pose(0.0)
        self.io = FakeArms(self.q, lag=1)
        self.hand_io = FakeHands()
        for s in SIDES:
            self.hand_io.connect(s)
        self.fingers = self.hand_io.positions()
        self.checker = TableChecker(ArmClearanceChecker(self.model, self.arms, self.hands))
        self.clock = [10.0]
        self.control = PreviewTeleop(self.io, self.hand_io, HoldMapping(),
            {s: EmptySource() for s in SIDES}, {s: _StaticRetargeter(self.fingers[s]) for s in SIDES},
            self.checker, None, self.q,
            arm_lower={s: self.arms[s].lower for s in SIDES},
            arm_upper={s: self.arms[s].upper for s in SIDES},
            hand_lower={s: self.hands[s].lower for s in SIDES},
            hand_upper={s: self.hands[s].upper for s in SIDES},
            limits=TeleopLimits(tracking_fault=0.15), clock=lambda: self.clock[0])

    def preview(self):
        start = join_arms(self.q)
        goal = start.copy()
        goal[7] += 0.03
        return straight_path(start, goal, self.checker, self.fingers, "right")

    def test_preview_does_not_command_and_execute_moves_only_right(self):
        preview = self.preview()
        np.testing.assert_array_equal(join_arms(self.io.positions()), preview.start)
        self.control.execute(preview)
        for t in np.arange(0, preview.follower.duration + 0.02, 0.01):
            self.clock[0] = 10 + t
            self.control.io_step()
        np.testing.assert_allclose(self.io.positions()["left"], self.q["left"])
        np.testing.assert_allclose(join_arms(self.io.positions()), preview.follower.waypoints[-1])
        self.assertEqual(self.control.state, "paused")

    def test_rejects_changed_pose(self):
        preview = self.preview()
        self.control.measured["right"][0] += 0.03
        with self.assertRaisesRegex(ValueError, "pose changed"):
            self.control.execute(preview)

    def test_rejects_changed_fingers_and_estop(self):
        preview = self.preview()
        self.control.hand_command["right"][0] += 0.03
        with self.assertRaisesRegex(ValueError, "Finger pose"):
            self.control.execute(preview)
        self.control.estop_flags = (False, True)
        with self.assertRaisesRegex(ValueError, "E-stop"):
            self.control.execute(preview)

    def test_pause_cancels_execution(self):
        self.control.execute(self.preview())
        self.control.request_pause()
        self.clock[0] += 5
        self.control.io_step()
        np.testing.assert_array_equal(join_arms(self.io.positions()), join_arms(self.q))

    def test_rejects_inactive_arm_motion(self):
        start = join_arms(self.q)
        goal = start.copy()
        goal[0] += 0.1
        with self.assertRaisesRegex(PlanningError, "inactive arm"):
            straight_path(start, goal, self.checker, self.fingers, "right")

    def test_rejects_collision_midpath(self):
        class Checker:
            def set_fingers(self, side, q):
                pass
            def clearance(self, q):
                return SimpleNamespace(distance_m=0 if 0.04 < q[7] < 0.06 else 0.1, pair="test")
        goal = np.zeros(14)
        goal[7] = 0.1
        with self.assertRaises(PlanningError):
            straight_path(np.zeros(14), goal, Checker(), self.fingers, "right")

    def test_table_plane_rejects_intersection(self):
        self.assertTrue(self.checker.meshes)
        self.checker.height = 10.0
        with self.assertRaisesRegex(PlanningError, "tabletop"):
            self.preview()

    def test_glove_mode_holds_both_arms(self):
        self.control.request_start()
        self.control.teleop_step()
        self.control.io_step()
        np.testing.assert_array_equal(join_arms(self.io.positions()), join_arms(self.q))


if __name__ == "__main__":
    unittest.main()
