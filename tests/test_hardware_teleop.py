"""Reset / start / pause control of both Vega arms and Wuji hands, without hardware."""

from contextlib import redirect_stdout
import io
import time
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from wuji_teleop.collision_check import ArmClearanceChecker, ResetPlanner, join_arms
from wuji_teleop.hardware_teleop import (
    ARM_RATE_HZ,
    DEVICE_LINKS,
    FakeArms,
    FakeHands,
    HandLinkBlocked,
    HardwareTeleop,
    HeldWristImu,
    PicoArmMapping,
    SyntheticPicoLinks,
    TRACKER_IK_CONFIG,
    TeleopApp,
    TeleopLimits,
    TrackerArmMapping,
    TrackerLinks,
    _ImuSlot,
    reset_pose,
)
from wuji_teleop.sources import HandFrame, SourceSnapshot, SyntheticSource
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
        self.assertEqual(len(fields), 2 + 28)

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

    def _app(self, *, robot_failures=0, hands=None, glove_failures=()):
        attempts = {"robot": 0}

        def connect_robot():
            attempts["robot"] += 1
            if attempts["robot"] <= robot_failures:
                raise RuntimeError("robot unreachable")
            return FakeArms(reset_pose())

        def connect_glove(side):
            if side in glove_failures:
                raise RuntimeError("Connection timeout")
            return SyntheticSource(side), HeldWristImu()

        with redirect_stdout(io.StringIO()):
            app = TeleopApp(
                connect_robot=connect_robot, hands_io=hands or FakeHands(),
                connect_glove=connect_glove, arm_input=SyntheticPicoLinks(),
                retargeters={side: _FixedRetargeter() for side in SIDES},
                reset_q=reset_pose(), limits=TeleopLimits(), retry_s=5.0,
                start_teleop=False)
        return app

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

    def test_blocked_hand_is_not_attempted_and_shows_the_fix(self):
        app = self._app(hands=_BlockedHands())
        self._step(app, 0.0, "robot_step", "hands_step")
        self.assertEqual(app.links["Left hand"].level, "ok")
        right = app.links["Right hand"]
        self.assertEqual(right.level, "fail")
        self.assertIn("two interfaces", right.detail)
        self.assertIn("subnet", right.hint)
        self.assertFalse(app.hands_io.is_connected("right"))

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


if __name__ == "__main__":
    unittest.main()
