"""Joint load tables: fit, lookup, sweep plan and the sweep itself, without hardware."""

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from wuji_teleop.hand_load import (
    JointLoad,
    SweepAbort,
    fit,
    plan_sweeps,
    sweep,
)
from wuji_teleop.hardware_teleop import FakeHands
from wuji_teleop.vega_bimanual_sim import load_bimanual_scene


class _LoadedHands(FakeHands):
    """Fingers follow commands; joint 2 carries 0.1 + 0.2 q A plus 0.05 A against its motion."""

    def __init__(self):
        super().__init__()
        self.connect("left")
        self.enable("left", np.zeros(20))
        self.stop_at = np.inf
        self.previous = np.zeros(20)
        self.sent = []

    def command(self, side, angles):
        self.previous = self._q[side].copy()
        self._q[side] = np.minimum(np.array(angles, dtype=float), self.stop_at)
        self.sent.append(np.array(angles, dtype=float))

    def feedback(self, side):
        q = self._q[side]
        current = np.zeros(20)
        moved = q[2] - self.previous[2]
        current[2] = 0.1 + 0.2 * q[2] + 0.05 * np.sign(moved)
        return q.copy(), current


def _plan(joint=2, lo=0.0, hi=0.6):
    pose = np.zeros(20)
    return [dict(joint=joint, posture="hold", pose=pose, lo=lo, hi=hi)]


class JointLoadTests(unittest.TestCase):
    def test_table_reads_back_and_interpolates_flat_beyond_its_range(self):
        path = Path(tempfile.mkdtemp()) / "hand.json"
        JointLoad({2: ([0.0, 1.0], [0.1, 0.3])}, np.full(20, 0.1)).save(path)
        table = JointLoad.read(path)
        q = np.zeros(20)
        for angle, load in ((0.5, 0.2), (-1.0, 0.1), (2.0, 0.3)):
            q[2] = angle
            self.assertAlmostEqual(table.at(q)[2], load)
        self.assertEqual(table.at(q)[3], 0.0)  # a joint never swept has no load
        self.assertEqual(table.friction[5], 0.1)

    def test_tables_exist_for_both_robot_hands_and_only_those(self):
        left, right = (JointLoad.for_serial(s) for s in ("WH2JA01260722011", "WH2KA01260827009"))
        self.assertEqual(len(left.curves), 20)
        self.assertEqual(len(right.curves), 19)
        self.assertNotIn(9, right.curves)  # right middle.S2 stays disabled
        for table in (left, right):
            q = np.zeros(20)
            q[2] = 1.25
            self.assertGreater(table.at(q)[2], 0.15)  # the load that held the thumb back
            self.assertLess(table.friction.max(), 0.3)
        self.assertGreater(right.friction[0], left.friction[0])
        self.assertIsNone(JointLoad.for_serial("no-such-hand"))
        self.assertIsNone(JointLoad.for_serial(None))


class SweepTests(unittest.TestCase):
    def _run(self, hands, plan=None, **options):
        with redirect_stdout(io.StringIO()):
            return sweep(hands, "left", plan or _plan(), speed=1.0, **options)

    def test_sweep_and_fit_recover_the_load_and_the_friction(self):
        result = self._run(_LoadedHands())
        self.assertIsNone(result.aborted)
        self.assertEqual([leg["direction"] for leg in result.legs], ["to_lo", "up", "down", "back"])
        self.assertTrue(all(leg["reached"] for leg in result.legs))
        table = fit(result.t, result.q, result.current, result.legs)
        angles, loads = table.curves[2]
        self.assertGreaterEqual(len(angles), 3)
        np.testing.assert_allclose(loads, 0.1 + 0.2 * angles, atol=0.03)
        self.assertAlmostEqual(table.friction[2], 0.05, delta=0.02)
        self.assertEqual(sorted(table.curves), [2])

    def test_every_move_is_gradual_and_holds_really_hold(self):
        hands = _LoadedHands()
        hands._q["left"] = np.full(20, 0.3)  # the hand starts away from the sweep's pose
        began = time.monotonic()
        self._run(hands)
        sent = np.array(hands.sent)
        self.assertLess(np.abs(np.diff(sent, axis=0)).max(), 1.0 / 200 * 10)  # 1 rad/s, 200 Hz
        np.testing.assert_allclose(sent[-1], np.zeros(20), atol=1e-6)
        expected = 0.3 + 0.6 * 2 + 2 * 0.3  # to the pose, up and down, the two pauses
        self.assertGreater(time.monotonic() - began, expected * 0.9)

    def test_a_blocked_joint_ends_that_direction_and_the_sweep_continues(self):
        hands = _LoadedHands()
        hands.stop_at = np.full(20, np.inf)
        hands.stop_at[2] = 0.2  # a stop at 0.2 rad, short of the 0.6 asked
        result = self._run(hands)
        self.assertIsNone(result.aborted)
        up = next(leg for leg in result.legs if leg["direction"] == "up")
        self.assertFalse(up["reached"])
        self.assertLess(np.max(hands.sent, axis=0)[2], 0.2 + 0.22 + 0.15)
        np.testing.assert_allclose(hands.sent[-1], np.zeros(20), atol=1e-6)  # home again

    def test_an_abort_returns_home_and_reports_why(self):
        hands = _LoadedHands()
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 20:
                raise SweepAbort("fault 0x2102 on thumb.S3")

        result = self._run(hands, check=check)
        self.assertEqual(result.aborted, "fault 0x2102 on thumb.S3")
        np.testing.assert_allclose(hands.sent[-1], np.zeros(20), atol=1e-6)

    def test_too_much_current_aborts(self):
        class _Hot(_LoadedHands):
            def feedback(self, side):
                q, current = super().feedback(side)
                current[3] = 0.75
                return q, current

        result = self._run(_Hot())
        self.assertIn("thumb.S4", result.aborted)

    def test_lost_feedback_aborts_without_moving_home(self):
        hands = _LoadedHands()
        hands.feedback_age = lambda side: 1.0
        result = self._run(hands, _plan(lo=-0.3))
        self.assertIn("feedback", result.aborted)
        self.assertEqual(len(hands.sent), 1)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, _, cls.hands = load_bimanual_scene()

    def test_ranges_stay_inside_the_limits_and_skip_disabled_joints(self):
        for side, skip in (("left", ()), ("right", (9,))):
            hand = self.hands[side]
            plans = plan_sweeps(self.model, self.data, hand, side, skip=skip)
            joints = [plan["joint"] for plan in plans]
            self.assertEqual(sorted(set(joints)), [j for j in range(20) if j not in skip])
            self.assertEqual(joints.count(2), 2)  # the thumb again in opposition
            for plan in plans:
                self.assertGreaterEqual(plan["lo"], hand.lower[plan["joint"]])
                self.assertLessEqual(plan["hi"], hand.upper[plan["joint"]])
                self.assertLess(plan["lo"], plan["hi"])


if __name__ == "__main__":
    unittest.main()
