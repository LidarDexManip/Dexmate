"""Reset / start / pause control of both Vega arms and Wuji hands, without hardware."""

from contextlib import redirect_stderr, redirect_stdout
import io
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from wuji_teleop.collision_check import ArmClearanceChecker, ResetPlanner, join_arms
from wuji_teleop.hardware_teleop import (
    ARM_RATE_HZ,
    DEVICE_LINKS,
    DexcontrolArms,
    FakeArms,
    FakeHands,
    HAND_MIT_GAINS,
    HandLinkBlocked,
    HardwareTeleop,
    HeldArms,
    HeldWristImu,
    NoArmInput,
    PicoArmMapping,
    SyntheticPicoLinks,
    TRACKER_IK_CONFIG,
    TeleopApp,
    TeleopLimits,
    TrackerArmMapping,
    TrackerLinks,
    WujiHands,
    _HeldMapping,
    _ImuSlot,
    main,
    reset_pose,
)
from wuji_teleop.retarget import copy_finger
from wuji_teleop.sources import HandFrame, SourceSnapshot, SyntheticSource, synthetic_keypoints
from wuji_teleop.vega_bimanual_sim import SIDES, load_bimanual_scene
from wuji_teleop.vr_pose_source import WristPoseFrame, WristPoseSnapshot
from wuji_teleop.xr_tracker_source import TrackerFrame, TrackerSnapshot


L_SHAPE = {"left": np.array([0.064, 0.3, 0.0, -1.556, 1.271, 0.0, 0.0]),
           "right": np.array([-0.064, -0.3, 0.0, -1.556, -1.271, 0.0, 0.0])}
GLOVE_ANGLES = np.full(20, 0.4)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _ScriptedPicoHand:
    """Fresh frames every read; the wrist rises 4 cm after ``rise_after`` seconds."""

    def __init__(self, side, clock, rise_after=0.3):
        self.side, self.clock, self.rise_after = side, clock, rise_after
        self.started = clock()
        self.count = 0

    def read(self):
        self.count += 1
        pose = np.eye(4)
        pose[:3, 3] = [0.3, 0.2 if self.side == "left" else -0.2, 1.0]
        if self.clock() - self.started > self.rise_after:
            pose[2, 3] += 0.04
        frame = WristPoseFrame(pose, 0.0, 1_000_000 + self.count, self.clock())
        return WristPoseSnapshot(frame, self.count, self.count, None)


class _Glove:
    def __init__(self, clock):
        self.clock = clock
        self.count = 0

    def read(self):
        self.count += 1
        frame = HandFrame(np.zeros((21, 3)), self.clock(), self.count, self.count)
        return SourceSnapshot(frame, self.count, self.count, None)


class _DroppedGlove:
    """A glove whose last frame arrived at `last_frame_at` (the SDK dropped it)."""

    serial = "WG1-dropped"

    def __init__(self, last_frame_at):
        self.frame = HandFrame(np.zeros((21, 3)), last_frame_at, 1, 1)
        self.closed = False

    def read(self):
        return SourceSnapshot(self.frame, 1, 1, None)

    def close(self):
        self.closed = True


class _FixedRetargeter:
    def step(self, points):
        return GLOVE_ANGLES.copy()

    def reset(self):
        pass


class _FixedMapping:
    """Mapping stand-in that proposes one fixed goal for the left arm."""

    def __init__(self, goal):
        self.goal = goal
        self.status = {side: "fixed" for side in SIDES}

    def recenter(self, q):
        pass

    def update(self, q, now):
        return {"left": self.goal.copy()}


class HardwareTeleopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()
        cls.lower = {side: cls.arms[side].lower for side in SIDES}
        cls.upper = {side: cls.arms[side].upper for side in SIDES}

    def _teleop(self, start, *, mapping=None, reset_q=None, lag=0.5, **limits):
        self.clock = _Clock()
        self.arms_io = FakeArms(start, lag=lag)
        self.hands_io = FakeHands()
        for side in SIDES:
            self.hands_io.connect(side)
            self.hands_io.enable(side, np.zeros(20))
        if mapping is None:
            mapping = PicoArmMapping(
                self.model, self.data, self.arms,
                {side: _ScriptedPicoHand(side, self.clock) for side in SIDES},
                {side: HeldWristImu() for side in SIDES}, posture=reset_q or reset_pose())
        planner = ResetPlanner(
            ArmClearanceChecker(self.model, self.arms, self.hands),
            np.concatenate([self.lower[side] for side in SIDES]),
            np.concatenate([self.upper[side] for side in SIDES]),
            rng=np.random.default_rng(0))
        with redirect_stdout(io.StringIO()):
            teleop = HardwareTeleop(
                self.arms_io, self.hands_io, mapping,
                {side: _Glove(self.clock) for side in SIDES},
                {side: _FixedRetargeter() for side in SIDES},
                ArmClearanceChecker(self.model, self.arms, self.hands), planner,
                reset_q or reset_pose(),
                arm_lower=self.lower, arm_upper=self.upper,
                hand_lower={side: self.hands[side].lower for side in SIDES},
                hand_upper={side: self.hands[side].upper for side in SIDES},
                limits=TeleopLimits(**limits), clock=self.clock, threaded_planning=False)
        return teleop

    def _run(self, teleop, seconds):
        """Teleop at 40 Hz interleaved with IO at 100 Hz on the fake clock."""
        next_teleop = self.clock.now
        with redirect_stdout(io.StringIO()):
            for _ in range(int(seconds * ARM_RATE_HZ)):
                if self.clock.now >= next_teleop:
                    teleop.teleop_step()
                    next_teleop += 0.025
                teleop.io_step()
                self.clock.now += 1.0 / ARM_RATE_HZ

    def _call(self, method):
        with redirect_stdout(io.StringIO()):
            return method()

    def test_starts_paused_and_streams_the_measured_pose(self):
        teleop = self._teleop(L_SHAPE)
        self.assertEqual(teleop.state, "paused")
        self._run(teleop, 0.2)
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], L_SHAPE[side])
            np.testing.assert_allclose(self.arms_io.positions()[side], L_SHAPE[side])
            np.testing.assert_allclose(self.hands_io.positions()[side], np.zeros(20))

    def test_start_follows_operator_and_pause_freezes_arms_and_hands(self):
        teleop = self._teleop(reset_pose())
        self.assertTrue(self._call(teleop.request_start))
        self._run(teleop, 1.0)
        self.assertEqual(teleop.state, "running")
        moved = max(np.max(np.abs(teleop.command[side] - reset_pose()[side])) for side in SIDES)
        self.assertGreater(moved, 0.01)
        np.testing.assert_allclose(teleop.hand_command["left"], GLOVE_ANGLES, atol=1e-9)

        self.assertTrue(self._call(teleop.request_pause))
        frozen = {side: teleop.command[side].copy() for side in SIDES}
        hands = {side: teleop.hand_command[side].copy() for side in SIDES}
        self._run(teleop, 0.5)
        self.assertEqual(teleop.state, "paused")
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], frozen[side])
            np.testing.assert_allclose(teleop.hand_command[side], hands[side])

    def test_start_reanchors_so_the_arm_does_not_jump(self):
        teleop = self._teleop(reset_pose())
        self._call(teleop.request_start)
        self._run(teleop, 0.1)  # first fresh frame calibrates; the wrist has not risen
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], reset_pose()[side], atol=1e-6)

    def test_hand_speed_is_limited(self):
        teleop = self._teleop(reset_pose(), max_hand_speed=2.0)
        self._call(teleop.request_start)
        self._run(teleop, 0.1)
        self.assertLessEqual(np.max(teleop.hand_command["left"]), 2.0 * 0.1 + 1e-9)

    def test_without_arm_input_the_arms_hold_while_the_fingers_follow(self):
        teleop = self._teleop(reset_pose(), mapping=_HeldMapping())
        self.assertTrue(self._call(teleop.request_start))
        self._run(teleop, 1.0)
        self.assertEqual(teleop.state, "running")
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], reset_pose()[side])
        np.testing.assert_allclose(teleop.hand_command["left"], GLOVE_ANGLES, atol=1e-9)

    def test_a_hand_regulator_bends_what_is_sent_and_its_log_records_both(self):
        teleop = self._teleop(reset_pose())

        class _Regulator:
            resets = 0

            def reset(self):
                self.resets += 1

            def step(self, now, nominal, q_obs, current):
                return SimpleNamespace(compliant=nominal - 0.1, force=np.ones((5, 3)),
                                       weight=np.ones(5))

        regulator = _Regulator()
        teleop.hand_regulators = {"left": regulator}
        commands, compliance = [], []
        teleop.hand_log = SimpleNamespace(command=lambda *row: commands.append(row),
                                          compliance=lambda *row: compliance.append(row))
        sent = []
        self.hands_io.command = lambda side, angles: sent.append((side, np.array(angles)))
        teleop.sync_hand("left", np.full(20, 0.3))
        teleop.hand_io_step()
        left = dict(sent)["left"]
        np.testing.assert_allclose(left, np.full(20, 0.2))           # compliant was sent
        np.testing.assert_allclose(dict(sent)["right"], teleop.hand_command["right"])
        self.assertEqual(regulator.resets, 1)                         # reset after sync_hand
        side, nominal, compliant, observed, force, weight = compliance[0]
        self.assertEqual(side, "left")
        np.testing.assert_allclose(nominal, np.full(20, 0.3))
        np.testing.assert_allclose(commands[0][1], left)              # log holds what was sent
        teleop.hand_io_step()
        self.assertEqual(regulator.resets, 1)

    def test_estop_holds_the_last_compliant_reference_without_regulating(self):
        teleop = self._teleop(reset_pose())
        offsets = iter([0.1, 0.2, 0.3])

        class _Regulator:
            steps = 0

            def reset(self):
                pass

            def step(self, now, nominal, q_obs, current):
                self.steps += 1
                return SimpleNamespace(compliant=nominal - next(offsets),
                                       force=np.ones((5, 3)), weight=np.ones(5))

        regulator = _Regulator()
        teleop.hand_regulators = {"left": regulator}
        sent = []
        self.hands_io.command = lambda side, angles: sent.append(np.array(angles))
        teleop.sync_hand("left", np.full(20, 0.5))
        teleop.hand_io_step()
        teleop.request_estop()
        teleop.hand_io_step()
        teleop.hand_io_step()
        self.assertEqual(regulator.steps, 1)
        left = sent[0::2]  # SIDES order: left, right
        for frozen in left[1:]:
            np.testing.assert_allclose(frozen, np.full(20, 0.4))
        teleop.state = "paused"  # released: regulation resumes from the frozen pose
        teleop.hand_io_step()
        self.assertEqual(regulator.steps, 2)
        np.testing.assert_allclose(sent[-2], np.full(20, 0.3))

    def test_finger_commands_are_logged_with_their_pre_slew_goal(self):
        teleop = self._teleop(reset_pose(), max_hand_speed=2.0)
        rows = []
        teleop.hand_log = SimpleNamespace(command=lambda *row: rows.append(row),
                                          glove=lambda *row: None)
        self._call(teleop.request_start)
        self._run(teleop, 0.1)
        self.assertEqual(len(rows), 2 * 10)  # both hands, every 100 Hz IO step
        side, target, goal = rows[-2]
        self.assertEqual(side, "left")
        np.testing.assert_allclose(target, teleop.hand_command["left"])
        np.testing.assert_allclose(goal, GLOVE_ANGLES, atol=1e-9)
        self.assertLess(np.max(target), np.max(goal))  # still slewing toward the goal

    def test_each_new_glove_frame_is_logged_once_while_running(self):
        teleop = self._teleop(reset_pose())
        rows = []
        teleop.hand_log = SimpleNamespace(command=lambda *row: None,
                                          glove=lambda *row: rows.append(row))
        frame = HandFrame(np.full((21, 3), 0.01), self.clock(), 7, 7)
        teleop.glove_sources["right"] = SimpleNamespace(
            read=lambda: SourceSnapshot(frame, 1, 1, None))
        self._run(teleop, 0.1)
        self.assertEqual(rows, [])  # paused: nothing is retargeted
        self._call(teleop.request_start)
        self._run(teleop, 0.1)
        left = [row for row in rows if row[0] == "left"]
        right = [row for row in rows if row[0] == "right"]
        self.assertGreaterEqual(len(left), 3)  # a fresh frame at every teleop step
        self.assertEqual(len({row[1] for row in left}), len(left))
        self.assertEqual([row[1] for row in right], [7])  # a repeated frame only once
        np.testing.assert_allclose(right[0][2], 0.01)

    def test_low_confidence_glove_finger_copies_its_neighbour_and_says_so(self):
        teleop = self._teleop(reset_pose())
        seen = []
        teleop.retargeters["right"] = SimpleNamespace(
            step=lambda points: seen.append(np.array(points)) or GLOVE_ANGLES.copy(),
            reset=lambda: None)
        points = synthetic_keypoints(1.0, "right")

        def read():
            frame = HandFrame(points, self.clock(), int(self.clock() * 1e6), 1,
                              0.1, (0.95, 0.95, 0.95, 0.1, 0.95))
            return SourceSnapshot(frame, int(self.clock() * 1e6), 1, None)

        teleop.glove_sources["right"] = SimpleNamespace(read=read)
        with redirect_stdout(io.StringIO()):
            self._call(teleop.request_start)
            self._run(teleop, 0.1)
        np.testing.assert_allclose(seen[-1], copy_finger(points, 3, 4))
        self.assertEqual(teleop.status()["finger_note"],
                         {"left": "", "right": "ring copies pinky (ring sensor 0.10)"})
        logged = [m for m in teleop.messages if "ring follows the pinky" in m]
        self.assertEqual(len(logged), 1)
        self.assertIn("Right glove ring sensor confidence 0.10", logged[0])

    def _record_arm_commands(self):
        sent = []
        real = self.arms_io.command

        def command(q, velocity=None):
            sent.append(velocity)
            real(q)

        self.arms_io.command = command
        return sent

    def test_arm_feedforward_is_off_by_default(self):
        teleop = self._teleop(reset_pose())
        sent = self._record_arm_commands()
        self._call(teleop.request_start)
        self._run(teleop, 0.5)
        self.assertTrue(sent)
        self.assertTrue(all(velocity is None for velocity in sent))

    def test_arm_feedforward_sends_the_scaled_rate_of_the_command(self):
        start = reset_pose()
        goal = {side: start[side].copy() for side in SIDES}
        goal["left"][0] += 1.0
        teleop = self._teleop(start, mapping=_FixedMapping(goal["left"]),
                              arm_velocity_feedforward=1.5, proximal_speed=0.5)
        sent = self._record_arm_commands()
        self._call(teleop.request_start)
        self._run(teleop, 0.3)
        self.assertLessEqual(np.abs(sent[0]["left"]).max(), 1e-9)  # no history yet
        np.testing.assert_allclose(sent[-1]["left"][0], 1.5 * 0.5, atol=0.02)  # at the 0.5 rad/s slew
        np.testing.assert_allclose(sent[-1]["left"][1:3], 0.0, atol=1e-9)

    def test_arm_feedforward_is_smoothed_over_steps_in_the_command(self):
        teleop = self._teleop(reset_pose(), arm_velocity_feedforward=1.0)
        q = {side: np.zeros(7) for side in SIDES}
        rates = []
        for tick in range(30):  # a frame every 4 ticks: the raw rate is all or nothing
            if tick % 4 == 0:
                q = {side: q[side] + 0.01 for side in SIDES}
            rates.append(teleop._arm_feedforward(tick * 0.01, "running", q)["left"][0])
        settled = np.array(rates[12:])  # true rate 0.25 rad/s; a per-tick rate would hit 1.0
        self.assertLessEqual(settled.max(), 0.3 + 1e-9)
        self.assertGreaterEqual(settled.min(), 0.2 - 1e-9)

    def test_arm_feedforward_stops_at_once_on_pause_and_estop_and_starts_clean(self):
        start = reset_pose()
        goal = start["left"].copy()
        goal[0] += 1.0
        teleop = self._teleop(start, mapping=_FixedMapping(goal), arm_velocity_feedforward=1.5)
        sent = self._record_arm_commands()
        self._call(teleop.request_start)
        self._run(teleop, 0.2)
        self.assertIsNotNone(sent[-1])
        self._call(teleop.request_pause)
        sent.clear()
        self._run(teleop, 0.1)
        self.assertTrue(all(velocity is None for velocity in sent))
        self.assertEqual(len(teleop._command_history), 0)
        self._call(teleop.request_start)
        sent.clear()
        self._run(teleop, 0.02)
        self.assertLessEqual(np.abs(sent[0]["left"]).max(), 1e-9)  # the old motion is gone
        self._call(teleop.request_estop)
        sent.clear()
        self._run(teleop, 0.1)
        self.assertTrue(all(velocity is None for velocity in sent))

    def test_dexcontrol_arms_send_velocity_only_when_given_one(self):
        calls = []
        arm = SimpleNamespace(
            set_joint_pos=lambda q: calls.append(("pos", q)),
            set_joint_pos_vel=lambda q, v: calls.append(("pos_vel", q, v)))
        arms = object.__new__(DexcontrolArms)
        arms._arms = {side: arm for side in SIDES}
        q = {side: np.zeros(7) for side in SIDES}
        arms.command(q)
        self.assertEqual([call[0] for call in calls], ["pos", "pos"])
        calls.clear()
        arms.command(q, {side: np.full(7, 0.5) for side in SIDES})
        self.assertEqual([call[0] for call in calls], ["pos_vel", "pos_vel"])
        np.testing.assert_allclose(calls[0][2], 0.5)

    def test_proximal_and_wrist_speeds_are_limited_separately(self):
        start = reset_pose()
        goal = start["left"] + 1.0
        teleop = self._teleop(
            start, mapping=_FixedMapping(goal), proximal_speed=0.5, wrist_speed=1.5)
        self._call(teleop.request_start)
        self._run(teleop, 0.1)
        change = teleop.command["left"] - start["left"]
        np.testing.assert_allclose(change[:4], 0.05, atol=1e-9)
        np.testing.assert_allclose(change[4:], 0.15, atol=1e-9)

    def test_arm_io_does_not_wait_for_slow_hand_commands(self):
        class SlowHands(FakeHands):
            def command(self, side, angles):
                time.sleep(0.03)
                super().command(side, angles)

        teleop = self._teleop(reset_pose())
        slow = SlowHands()
        for side in SIDES:
            slow.connect(side)
            slow.enable(side, np.zeros(20))
        teleop.hands_io = slow
        started = time.monotonic()
        teleop.io_step(send_hands=False)
        arm_elapsed = time.monotonic() - started
        started = time.monotonic()
        teleop.hand_io_step()
        hand_elapsed = time.monotonic() - started
        self.assertLess(arm_elapsed, 0.03)
        self.assertGreaterEqual(hand_elapsed, 0.06)

    def test_reset_moves_along_a_clear_path_and_holds_at_the_reset_pose(self):
        teleop = self._teleop(L_SHAPE)
        self.assertTrue(self._call(teleop.request_reset))
        self.assertEqual(teleop.state, "resetting")
        self._run(teleop, 8.0)
        self.assertEqual(teleop.state, "paused")
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], reset_pose()[side])
            np.testing.assert_allclose(self.arms_io.positions()[side], reset_pose()[side],
                                       atol=1e-3)
        self.assertIn("At the reset pose", teleop.messages[0])

    def test_reset_speed_is_limited(self):
        teleop = self._teleop(L_SHAPE, reset_speed=0.2)
        self._call(teleop.request_reset)
        previous = join_arms(teleop.command)
        fastest = 0.0
        with redirect_stdout(io.StringIO()):
            for _ in range(300):
                teleop.io_step()
                self.clock.now += 0.01
                current = join_arms(teleop.command)
                fastest = max(fastest, np.max(np.abs(current - previous)) / 0.01)
                previous = current
        self.assertLessEqual(fastest, 0.2 + 1e-6)

    def test_pause_stops_a_reset_where_it_is(self):
        teleop = self._teleop(L_SHAPE)
        self._call(teleop.request_reset)
        self._run(teleop, 1.5)
        self._call(teleop.request_pause)
        stopped = join_arms(teleop.command)
        self._run(teleop, 1.0)
        self.assertEqual(teleop.state, "paused")
        np.testing.assert_allclose(join_arms(teleop.command), stopped)
        self.assertGreater(np.max(np.abs(stopped - join_arms(L_SHAPE))), 0.05)

    def test_reset_is_refused_when_the_reset_pose_collides(self):
        bad = reset_pose(0.0)
        bad["left"][2] = 0.8  # left hand into the right arm
        teleop = self._teleop(L_SHAPE, reset_q=bad)
        self._call(teleop.request_reset)
        self.assertEqual(teleop.state, "paused")
        self.assertIn("Reset refused", teleop.messages[0])
        self._run(teleop, 0.2)
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], L_SHAPE[side])

    def test_estop_blocks_motion_until_released(self):
        teleop = self._teleop(reset_pose())
        self._call(teleop.request_estop)
        self._run(teleop, 0.05)
        self.assertEqual(teleop.state, "estop")
        self.assertTrue(self.arms_io.software)
        self.assertFalse(self._call(teleop.request_start))
        self.assertFalse(self._call(teleop.request_reset))
        self.assertTrue(self._call(teleop.request_release))
        self._run(teleop, 0.05)
        self.assertEqual(teleop.state, "paused")
        self.assertFalse(self.arms_io.software)
        self.assertTrue(self._call(teleop.request_start))

    def test_hardware_estop_button_stops_a_running_session(self):
        teleop = self._teleop(reset_pose())
        self._call(teleop.request_start)
        self._run(teleop, 0.5)
        self.arms_io.button = True
        self._run(teleop, 0.05)
        self.assertEqual(teleop.state, "estop")
        frozen = join_arms(teleop.command)
        self._run(teleop, 0.5)
        np.testing.assert_allclose(join_arms(teleop.command), frozen)

    def test_guard_holds_a_target_that_drives_a_hand_into_the_other_arm(self):
        into_arm = reset_pose(0.0)["left"].copy()
        into_arm[2] = 0.8
        teleop = self._teleop(reset_pose(0.0), mapping=_FixedMapping(into_arm))
        self._call(teleop.request_start)
        self._run(teleop, 1.0)
        np.testing.assert_allclose(teleop.command["left"], reset_pose(0.0)["left"])
        self.assertTrue(any("Guard: holding" in message for message in teleop.messages))

    def test_guard_off_holds_nothing_and_still_reports_clearance(self):
        into_arm = reset_pose(0.0)["left"].copy()
        into_arm[2] = 0.8
        teleop = self._teleop(reset_pose(0.0), mapping=_FixedMapping(into_arm), teleop_guard="off")
        self._call(teleop.request_start)
        self._run(teleop, 2.0)
        np.testing.assert_allclose(teleop.command["left"], into_arm)
        self.assertFalse(any("Guard: holding" in message for message in teleop.messages))
        self.assertLess(teleop.status()["clearance"].distance_m, 0.0)
        self.assertEqual(teleop.status()["guard"], "off")

    def test_lagging_arm_only_warns_by_default(self):
        teleop = self._teleop(L_SHAPE, lag=0.0)  # the fake arm never moves
        self._call(teleop.request_reset)
        self._run(teleop, 4.0)
        self.assertEqual(teleop.state, "resetting")
        warnings = [m for m in teleop.messages if "behind its command" in m]
        self.assertEqual(len(warnings), 1)

    def test_blocked_arm_faults_and_holds_where_it_is_when_enabled(self):
        teleop = self._teleop(L_SHAPE, lag=0.0, tracking_fault=0.30)
        self._call(teleop.request_reset)
        self._run(teleop, 4.0)
        self.assertEqual(teleop.state, "fault")
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], L_SHAPE[side])
        self.assertTrue(self._call(teleop.request_pause))
        self.assertEqual(teleop.state, "paused")

    def test_csv_log_records_commanded_and_measured_joints(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/log.csv"
            teleop = self._teleop(L_SHAPE)
            teleop._csv = open(path, "w", encoding="utf-8")
            teleop._csv.write("header\n")
            self._run(teleop, 0.5)
            teleop._csv.close()
            with open(path, encoding="utf-8") as handle:
                rows = handle.read().splitlines()[1:]
        self.assertEqual(len(rows), 10)  # 20 Hz for 0.5 s
        fields = rows[0].split(",")
        self.assertEqual(fields[1], "paused")
        self.assertEqual(len(fields), 2 + 10 + 42)

    def test_start_is_ignored_while_resetting(self):
        teleop = self._teleop(L_SHAPE)
        self._call(teleop.request_reset)
        self.assertFalse(self._call(teleop.request_start))
        self.assertEqual(teleop.state, "resetting")

    def test_small_operator_move_from_reset_pose_moves_the_arm_a_little(self):
        # The IK's posture preference must be the reset pose, not the sim home,
        # or a few centimetres of hand motion swings the elbow outward.
        teleop = self._teleop(reset_pose())
        self._call(teleop.request_start)
        self._run(teleop, 3.0)  # the scripted wrist rises 4 cm (2.4 cm robot)
        change = np.max(np.abs(teleop.command["left"] - reset_pose()["left"]))
        self.assertGreater(change, 0.01)
        self.assertLess(change, 0.15)

    def test_mapping_posture_does_not_change_the_shared_scene_arms(self):
        PicoArmMapping(self.model, self.data, self.arms,
                       {side: None for side in SIDES}, {side: None for side in SIDES},
                       posture=reset_pose())
        self.assertFalse(np.allclose(self.arms["left"].home, reset_pose()["left"]))

    def test_reset_pose_keeps_forearms_level(self):
        for back in (0.0, 0.25, 0.5):
            q = reset_pose(back)
            self.assertAlmostEqual(q["left"][0] + q["left"][3], 0.0)
            np.testing.assert_allclose(q["right"][[0, 3]], [-q["left"][0], q["left"][3]])


class _ScriptedTrackers:
    """Wrist trackers on both robot hands; the left one moves after ``move_after`` s."""

    def __init__(self, poses, clock, move_after=0.3, shift=(0.04, 0.0, 0.0)):
        self.poses, self.clock = poses, clock
        self.move_after, self.shift = move_after, np.asarray(shift)
        self.started = clock()
        self.count = 0
        self.stale_duplicates = 0

    def read(self):
        self.count += 1
        poses = {serial: pose.copy() for serial, pose in self.poses.items()}
        if self.clock() - self.started > self.move_after:
            poses["SN-L"][:3, 3] += self.shift
        frame = TrackerFrame(poses, None, self.count, self.clock())
        return TrackerSnapshot(frame, self.count, self.count, None)

    def close(self):
        pass


def _trackers_on_hands(model, arms, q):
    data = mujoco.MjData(model)
    for side in SIDES:
        data.qpos[arms[side].qpos_ids] = q[side]
    mujoco.mj_kinematics(model, data)
    poses = {}
    for side, serial in (("left", "SN-L"), ("right", "SN-R")):
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               "L_ee" if side == "left" else "R_ee")
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[ee].reshape(3, 3)
        pose[:3, 3] = data.xpos[ee]
        poses[serial] = pose
    return poses


class TrackerInputTests(unittest.TestCase):
    """PICO Motion Tracker arm input (the default on hardware)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.arms, cls.hands = load_bimanual_scene()

    def _trackers(self, clock, **kwargs):
        return _ScriptedTrackers(_trackers_on_hands(self.model, self.arms, reset_pose()),
                                 clock, **kwargs)

    def test_start_calibrates_without_a_jump_then_the_left_arm_follows(self):
        harness = HardwareTeleopTests()
        harness.model, harness.data = self.model, self.data
        harness.arms, harness.hands = self.arms, self.hands
        harness.lower = {side: self.arms[side].lower for side in SIDES}
        harness.upper = {side: self.arms[side].upper for side in SIDES}
        clock = _Clock()
        trackers = self._trackers(clock, move_after=0.5)
        mapping = TrackerArmMapping(self.model, trackers, self.arms,
                                    ik_config=TRACKER_IK_CONFIG)
        teleop = harness._teleop(reset_pose(), mapping=mapping)
        harness.clock = clock  # share one clock with the scripted trackers
        teleop.clock = clock
        trackers.started = clock()
        harness._call(teleop.request_start)
        harness._run(teleop, 0.4)  # calibrated, operator still
        for side in SIDES:
            np.testing.assert_allclose(teleop.command[side], reset_pose()[side], atol=0.02)
        self.assertEqual(mapping.status["left"], "tracking")
        harness._run(teleop, 1.5)  # left wrist 4 cm forward
        left = np.max(np.abs(teleop.command["left"] - reset_pose()["left"]))
        right = np.max(np.abs(teleop.command["right"] - reset_pose()["right"]))
        self.assertGreater(left, 0.02)
        self.assertLess(left, 0.4)
        self.assertLess(right, 0.02)
        self.assertEqual(mapping.last_source_revision, trackers.count)
        self.assertEqual(mapping.last_tracker_timestamp_ns, trackers.count)
        self.assertLessEqual(clock() - mapping.last_source_received_at, 0.025)

    def test_tracker_links_show_live_trackers_and_ignored_duplicates(self):
        trackers = self._trackers(time.monotonic)
        trackers.stale_duplicates = 2
        links = TrackerLinks(trackers, ik_config=TRACKER_IK_CONFIG,
                             service_running=lambda: True).statuses()
        self.assertEqual(links["Tracker service"].level, "ok")
        self.assertIn("2 live tracker(s); ignoring 2 frozen duplicate entries",
                      links["Tracker service"].detail)
        self.assertEqual(links["Left tracker"].level, "ok")
        self.assertIn("SN-L", links["Left tracker"].detail)
        self.assertIn("SN-R", links["Right tracker"].detail)

    def test_tracker_service_down_is_red(self):
        links = TrackerLinks(self._trackers(time.monotonic), ik_config=TRACKER_IK_CONFIG,
                             service_running=lambda: False).statuses()
        self.assertEqual(links["Tracker service"].level, "fail")
        self.assertIn("runService.sh", links["Tracker service"].hint)


class _RecordingImu:
    def __init__(self):
        self.anchors = []

    def recenter(self, anchor_pose):
        self.anchors.append(anchor_pose)

    def read(self, anchor_pose, *, now=None):
        return SimpleNamespace(pose=None)


class _BlockedHands(FakeHands):
    def connect(self, side):
        if side == "right":
            raise HandLinkBlocked("laptop 192.168.1.50 is on two interfaces")
        super().connect(side)


class TeleopAppTests(unittest.TestCase):
    """Connections are made and retried in the background; nothing exits."""

    def _app(self, *, robot_failures=0, hands=None, glove_failures=(), arms=FakeArms,
             arm_input=None, limits=None, **options):
        attempts = {"robot": 0}
        self.glove_attempts = []

        def connect_robot():
            attempts["robot"] += 1
            if attempts["robot"] <= robot_failures:
                raise RuntimeError("robot unreachable")
            return arms(reset_pose())

        def connect_glove(side):
            self.glove_attempts.append(side)
            if side in glove_failures:
                raise RuntimeError("Connection timeout")
            return SyntheticSource(side), HeldWristImu()

        with redirect_stdout(io.StringIO()):
            app = TeleopApp(
                connect_robot=connect_robot, hands_io=hands or FakeHands(),
                connect_glove=connect_glove, arm_input=arm_input or SyntheticPicoLinks(),
                retargeters={side: _FixedRetargeter() for side in SIDES},
                reset_q=reset_pose(), limits=limits or TeleopLimits(), retry_s=5.0,
                start_teleop=False, **options)
        return app

    def test_allow_hands_guards_teleop_without_hand_to_hand_but_not_the_reset_path(self):
        app = self._app(limits=TeleopLimits(teleop_guard="allow-hands"))
        self._step(app, 0.0, "robot_step")
        hands_touch = reset_pose(0.0)
        hands_touch["left"][2], hands_touch["right"][2] = 0.25, -0.25
        q = join_arms(hands_touch)
        for checker in (app.teleop.guard_checker, app.planner.checker):
            for side in SIDES:
                checker.set_fingers(side, np.zeros(20))
        self.assertGreater(app.teleop.guard_checker.clearance(q).distance_m, 0.015)
        self.assertLess(app.planner.checker.clearance(q).distance_m, 0.0)
        self.assertEqual(app.status()["guard"], "allow-hands")

    def test_fingers_only_mode_never_connects_the_arms_and_shows_it(self):
        app = self._app(arms=HeldArms, arm_input=NoArmInput())
        self._step(app, 0.0, "robot_step")
        self.assertEqual(app.links["Robot arms"].level, "off")
        self.assertIn("--arm-input none", app.links["Robot arms"].detail)
        self.assertIsNotNone(app.teleop)
        self.assertNotIn("Tracker service", app.links)

    def test_hand_compliance_builds_one_regulator_per_connected_hand(self):
        from wuji_teleop.hand_compliance import ComplianceParams, HandForceRegulator

        app = self._app(hand_sides=("left",), hand_compliance=ComplianceParams())
        self._step(app, 0.0, "robot_step")
        self.assertEqual(list(app.teleop.hand_regulators), ["left"])
        self.assertIsInstance(app.teleop.hand_regulators["left"], HandForceRegulator)
        plain = self._app()
        self._step(plain, 0.0, "robot_step")
        self.assertEqual(plain.teleop.hand_regulators, {})

    def test_hands_option_leaves_the_other_hand_and_glove_alone(self):
        hands = FakeHands()  # both reachable: only the option keeps the right one out
        app = self._app(hands=hands, hand_sides=("left",))
        self._step(app, 0.0, "robot_step", "hands_step", "gloves_step")
        self.assertTrue(hands.is_connected("left"))
        self.assertFalse(hands.is_connected("right"))
        self.assertEqual(self.glove_attempts, ["left"])
        for name in ("Right hand", "Right glove"):
            self.assertEqual(app.links[name].level, "off")
            self.assertIn("--hands left", app.links[name].detail)

    def _step(self, app, now, *steps):
        with redirect_stdout(io.StringIO()):
            for step in steps:
                getattr(app, step)(now)

    def test_robot_failure_is_shown_and_retried_without_exiting(self):
        app = self._app(robot_failures=1)
        self._step(app, 0.0, "robot_step")
        self.assertEqual(app.links["Robot arms"].level, "fail")
        self.assertIn("robot unreachable", app.links["Robot arms"].detail)
        self.assertIsNone(app.teleop)
        with redirect_stdout(io.StringIO()):
            self.assertFalse(app.request("start"))
        self.assertIn("not connected yet", app.status()["messages"][0])
        self._step(app, 1.0, "robot_step")  # before the retry delay: no new attempt
        self.assertIsNone(app.teleop)
        self._step(app, 6.0, "robot_step")
        self.assertEqual(app.links["Robot arms"].level, "ok")
        self.assertEqual(app.status()["state"], "paused")

    def test_glove_failure_leaves_the_app_running_and_retries(self):
        app = self._app(glove_failures={"right"})
        self._step(app, 0.0, "robot_step", "hands_step", "gloves_step", "input_step")
        self.assertEqual(app.links["Left glove"].level, "ok")
        self.assertEqual(app.links["Right glove"].level, "fail")
        self.assertIn("configure_wuji_network.sh", app.links["Right glove"].hint)
        self.assertEqual(app.links["Pico left hand"].level, "ok")
        with redirect_stdout(io.StringIO()):
            self.assertTrue(app.request("start"))
        self.assertTrue(any("Starting without: Right glove" in m
                            for m in app.status()["messages"]))

    def test_silent_glove_is_closed_and_reconnected(self):
        gloves = []

        def connect_glove(side):
            glove = _DroppedGlove(time.monotonic() - 10.0)
            gloves.append(glove)
            return glove, HeldWristImu()

        app = self._app()
        app.connect_glove = connect_glove
        self._step(app, 0.0, "gloves_step")
        self.assertEqual(app.links["Left glove"].level, "fail")
        self.assertIn("reconnecting", app.links["Left glove"].hint)
        self.assertTrue(all(glove.closed for glove in gloves))
        self.assertEqual(app.gloves, {})
        self.assertEqual(app.glove_slots["left"].read().error, "glove not connected")
        self._step(app, 1.0, "gloves_step")  # before the retry delay
        self.assertEqual(len(gloves), 2)
        self._step(app, 6.0, "gloves_step")
        self.assertEqual(len(gloves), 4)

    def test_blocked_hand_is_not_attempted_and_shows_the_fix(self):
        app = self._app(hands=_BlockedHands())
        self._step(app, 0.0, "robot_step", "hands_step")
        self.assertEqual(app.links["Left hand"].level, "ok")
        right = app.links["Right hand"]
        self.assertEqual(right.level, "fail")
        self.assertIn("two interfaces", right.detail)
        self.assertIn("subnet", right.hint)
        self.assertFalse(app.hands_io.is_connected("right"))

    def test_regulators_use_the_load_table_of_the_connected_hand(self):
        from wuji_teleop.hand_compliance import ComplianceParams

        hands = FakeHands()
        app = self._app(hands=hands, hand_compliance=ComplianceParams())
        self._step(app, 0.0, "robot_step", "hands_step")
        regulator = app.teleop.hand_regulators["left"]
        self.assertIsNone(regulator.load_source())
        self.assertIn("no joint load table", app.links["Left hand"].detail)
        hands.serials["left"] = "WH2JA01260722011"
        self.assertEqual(regulator.load_source().friction.shape, (20,))
        self._step(app, 0.0, "hands_step")
        self.assertNotIn("no joint load table", app.links["Left hand"].detail)

    def test_no_load_table_note_without_compliance(self):
        app = self._app()
        self._step(app, 0.0, "robot_step", "hands_step")
        self.assertNotIn("no joint load table", app.links["Left hand"].detail)

    def test_a_hand_running_without_a_faulted_joint_says_so(self):
        hands = FakeHands()
        hands.faults["right"] = {9: 0x2102}
        app = self._app(hands=hands)
        self._step(app, 0.0, "robot_step", "hands_step")
        right = app.links["Right hand"]
        self.assertEqual(right.level, "ok")
        self.assertIn("without middle.S2 (fault 0x2102, left disabled)", right.detail)
        self.assertNotIn("without", app.links["Left hand"].detail)

    def test_hand_connecting_later_joins_without_a_jump(self):
        hands = FakeHands(start={side: np.full(20, 0.5) for side in SIDES},
                          connected={"left"})
        app = self._app(hands=hands)
        self._step(app, 0.0, "robot_step", "hands_step")
        self.assertEqual(app.links["Right hand"].level, "fail")
        self.assertFalse(hands.is_ready("right"))
        hands.reachable.add("right")
        self._step(app, 6.0, "hands_step")
        self.assertTrue(hands.is_ready("right"))
        np.testing.assert_allclose(app.teleop.hand_command["right"], np.full(20, 0.5))
        with redirect_stdout(io.StringIO()):
            app.teleop.io_step()
        np.testing.assert_allclose(hands.positions()["right"], np.full(20, 0.5))

    def test_late_imu_is_recentred_on_the_last_anchor(self):
        slot = _ImuSlot()
        self.assertIsNone(slot.read(np.eye(4)).pose)
        slot.recenter("anchor")
        imu = _RecordingImu()
        slot.attach(imu)
        self.assertEqual(imu.anchors, ["anchor"])

    def test_status_has_every_link_before_the_robot_connects(self):
        app = self._app()
        status = app.status()
        self.assertEqual(status["state"], "connecting")
        self.assertEqual(set(status["links"]), set(DEVICE_LINKS + SyntheticPicoLinks.link_names))


class _FakeWujiHand:
    def __init__(self, handedness, serial_number="WH2JA0001"):
        self.gains, self.info = [], SimpleNamespace(firmware_version="2.13.1")
        self.limits, self.limit_writes = [1.5] * 20, []
        self._handedness = handedness
        self.serial_number = serial_number
        self.diagnostic_codes, self.enabled, self.sent = {}, [], []

    def joint_diagnostics(self):
        def subscribe(callback):  # nid = bus * 5 + node, joints on nodes 1-4
            callback(SimpleNamespace(joints=[
                SimpleNamespace(nid=j // 4 * 5 + j % 4 + 1,
                                error_code_current=self.diagnostic_codes.get(j, 0))
                for j in range(20)]))
            return SimpleNamespace(close=lambda: None)

        return SimpleNamespace(subscribe_with_callback=subscribe)

    def joint_command(self):
        return SimpleNamespace(publish=lambda: SimpleNamespace(send=self.sent.append))

    def enable(self, joints=None):
        self.enabled.append(joints)

    def handedness(self):
        return SimpleNamespace(get=lambda: self._handedness)

    def mit_params(self):
        return SimpleNamespace(set=self.gains.append)

    def joint_states(self):
        def subscribe(callback):
            self.on_state = callback
            return SimpleNamespace(close=lambda: None)

        return SimpleNamespace(subscribe_with_callback=subscribe)

    def effort_limit(self):
        def write(value):
            self.limit_writes.append(value)
            self.limits = list(value) if isinstance(value, list) else [value] * 20

        return SimpleNamespace(get=lambda: list(self.limits), set=write)


class WujiHandsTests(unittest.TestCase):
    def connect_left(self, *, serial_number="WH2JA0001", **kwargs):
        hand = _FakeWujiHand("Handedness.Left", serial_number)
        self.connections = []

        def connect(**options):
            self.connections.append(options)
            return hand

        manager = SimpleNamespace(connect=connect, disconnect=lambda alias: None)
        with patch("wuji_sdk.SdkManager.instance", return_value=manager), \
                patch("wuji_teleop.connection_monitor.ambiguous_hand_source", return_value=None):
            hands = WujiHands({"left": "WH2JA0001"}, **kwargs)
            hands.addresses["left"] = "192.168.77.110:50001"
            hands.connect("left")
        self.assertTrue(hands.is_connected("left"))
        self.hands = hands
        return hand

    def test_discovery_scans_in_a_child_process(self):
        scanned = [
            {"sn": "WH2JA0001", "type": "DeviceType.WujiHand2", "address": "192.168.77.110:50001"},
            {"sn": "WG1KA0002", "type": "DeviceType.WujiGlove", "address": "192.168.1.101:50001"},
        ]
        with patch("wuji_sdk.SdkManager.instance"), \
                patch("wuji_teleop.connection_monitor.wuji_scan_isolated", return_value=scanned):
            hands = WujiHands()
            hands.discover()
        self.assertEqual(hands.serials, {"left": "WH2JA0001", "right": None})
        self.assertEqual(hands.addresses, {"left": "192.168.77.110:50001", "right": None})

    def test_connect_goes_by_discovered_address_with_a_timeout(self):
        self.connect_left()
        (options,) = self.connections
        self.assertEqual(options["address"], "192.168.77.110:50001")
        self.assertNotIn("sn", options)
        self.assertEqual(options["options"].timeout_ms, 3000)

    def test_connect_refuses_a_different_hand_at_the_address(self):
        with self.assertRaisesRegex(RuntimeError, "expected WH2JA0001"):
            self.connect_left(serial_number="WH2JA0009")

    def test_connect_writes_the_default_finger_gains(self):
        self.assertEqual(HAND_MIT_GAINS, (1.5, 0.1))
        self.assertEqual(self.connect_left().gains, [(1.5, 0.1)])

    def test_connect_writes_requested_finger_gains(self):
        self.assertEqual(self.connect_left(mit_gains=(5, 0.2)).gains, [(5.0, 0.2)])

    def test_current_limit_is_written_on_connect_and_restored_on_close(self):
        hand = self.connect_left(current_limit=0.8)
        self.assertEqual(hand.limit_writes, [0.8])
        self.hands.close()
        self.assertEqual(hand.limit_writes, [0.8, [1.5] * 20])

    def test_hand_current_limit_is_left_alone_by_default(self):
        hand = self.connect_left()
        self.hands.close()
        self.assertEqual(hand.limit_writes, [])

    def test_every_joint_state_frame_is_logged(self):
        rows = []
        hand = self.connect_left(log=SimpleNamespace(state=lambda *row: rows.append(row)))
        joints = [SimpleNamespace(position=0.01 * j, velocity=0.5, effort=-0.02)
                  for j in range(20)]
        hand.on_state(SimpleNamespace(num_joints=20, joints=joints,
                                      header=SimpleNamespace(timestamp_us=777)))
        ((side, stamp, q, v, effort),) = rows
        self.assertEqual((side, stamp), ("left", 777))
        np.testing.assert_allclose(q, 0.01 * np.arange(20))
        self.assertEqual((v[0], effort[19]), (0.5, -0.02))

    def test_a_faulted_joint_is_left_disabled_and_holds_its_measured_angle(self):
        hand = self.connect_left()
        hand.diagnostic_codes = {9: 0x2102}  # right middle.S2 on the robot
        joints = [SimpleNamespace(position=0.2, velocity=0.0, effort=0.0) for _ in range(20)]
        hand.on_state(SimpleNamespace(num_joints=20, joints=joints,
                                      header=SimpleNamespace(timestamp_us=1)))
        self.hands.enable("left", np.full(20, 0.2))
        (mask,) = hand.enabled
        self.assertEqual([j for j in range(20) if not mask[j]], [9])
        self.assertEqual(self.hands.faults["left"], {9: 0x2102})
        self.hands.command("left", np.full(20, 0.6))
        sent = np.array([command.position for command in hand.sent[-1]])
        self.assertAlmostEqual(sent[9], 0.2)
        np.testing.assert_allclose(np.delete(sent, 9), 0.6)

    def test_a_listed_joint_stays_disabled_when_its_fault_reads_clear(self):
        hand = self.connect_left()
        with patch.dict("wuji_teleop.hardware_teleop.DISABLED_JOINTS",
                        {"WH2JA0001": {9: 0x2102}}):
            self.hands.enable("left", np.zeros(20))
        (mask,) = hand.enabled
        self.assertEqual([j for j in range(20) if not mask[j]], [9])
        self.assertEqual(self.hands.faults["left"], {9: 0x2102})

    def test_a_healthy_hand_is_enabled_whole(self):
        hand = self.connect_left()
        self.hands.enable("left", np.zeros(20))
        self.assertEqual(hand.enabled, [None])
        self.assertEqual(self.hands.faults["left"], {})

    def test_out_of_range_finger_gains_are_rejected(self):
        for flags in (["--hand-kp", "0"], ["--hand-kp", "20"], ["--hand-kd", "-0.1"],
                      ["--hand-current-limit", "0"], ["--hand-current-limit", "2"],
                      ["--fingertip-force-limit", "0.5"], ["--fingertip-force-limit", "40"],
                      ["--arm-velocity-feedforward", "-0.5"],
                      ["--arm-velocity-feedforward", "3"], ["--teleop-guard", "maybe"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--fake-robot", "--synthetic-input", *flags])


if __name__ == "__main__":
    unittest.main()
