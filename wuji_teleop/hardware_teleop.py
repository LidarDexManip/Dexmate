"""Pico + Wuji glove teleoperation of the physical Vega-1U arms and Wuji hands.

MOVES HARDWARE. A dashboard window gives the operator these commands:

* Reset (R): plan a path to the reset pose that keeps both Wuji hands and both
  arms clear of the torso, the head, each other and their own upper arms,
  then move there slowly and hold.
* Start (S): re-anchor relative arm tracking at the robot's current pose,
  then follow the operator.
* Pause (P or Space): freeze both arms and both hands where they are.
* E-stop (Esc): activate the robot's software E-stop and freeze the hands.
  Release (U) clears it and leaves the robot paused.

Arms track relatively, calibrated at the robot's pose on Start, as in
``vega_bimanual_sim``: by default a PICO Motion Tracker on each wrist
(XRoboToolkit) drives all seven joints through GMR-style IK; ``--arm-input
pico-hand`` uses Pico hand tracking for joints 1-4 and the glove palm IMUs for
the wrists. Hands track absolutely: each glove skeleton is retargeted to the
matching Wuji Hand 2's 20 joints. Every teleop target is also checked for
clearance and held if it would bring a hand or arm closer than the guard
distance.
"""

from __future__ import annotations

import argparse
import collections
import copy
from dataclasses import dataclass
import functools
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from .collision_check import (
    ArmClearanceChecker, PathFollower, PlanningError, ResetPlanner, join_arms, split_arms)
from .retarget import TrackingController
from .sources import SourceSnapshot, SyntheticSource
from .vega_arm_sim import (
    HAND_IN_EE_ROTATION, PicoHandArmMapper, RIGHT_ARM_JOINT_SIGNS, WristOrientationIK,
    relative_wrist_rotation)
from .vega_bimanual_sim import (
    LEFT_GLOVE_ADDRESS, RIGHT_GLOVE_ADDRESS, SIDES, _limit_joint_goal, _side_ik,
    arm_joint_names, body_names, load_bimanual_scene)
from .vr_pose_source import WristPoseFrame, WristPoseSnapshot


RESET_SHOULDER_BACK = 0.25  # rad behind vertical; the elbow closes to keep the forearm level
# The tracker hardware path's IK config used by launch_wuji_vega_hardware_pico.sh.
TRACKER_IK_CONFIG = (Path(__file__).with_name("ik_configs")
                     / "xr_trackers_to_vega_1u_decoupled.json")
ARM_RATE_HZ = 100.0
TELEOP_RATE_HZ = 40.0
STATE_COLORS = {
    "paused": "#c58a00", "planning": "#2a6fb0", "resetting": "#2a6fb0",
    "running": "#1d8a3a", "estop": "#b3261e", "fault": "#b3261e",
    "connecting": "#5f6368",
}


def reset_pose(shoulder_back=RESET_SHOULDER_BACK):
    """Upper arms down and slightly back; elbows close so forearms stay level."""
    pitch = math.pi / 2 + float(shoulder_back)
    left = np.array([pitch, 0.0, 0.0, -pitch, 0.0, 0.0, 0.0])
    return {"left": left, "right": RIGHT_ARM_JOINT_SIGNS * left}


def _copy(q):
    return {side: np.array(q[side], dtype=np.float64) for side in SIDES}


def _slew(current, goal, step):
    return {side: current[side] + np.clip(goal[side] - current[side], -step, step)
            for side in SIDES}


class DexcontrolArms:
    """Both Vega arms through dexcontrol; position commands must be streamed."""

    def __init__(self):
        from dexcontrol.robot import Robot

        self._robot = Robot(auto_shutdown=False)
        self._arms = {side: getattr(self._robot, f"{side}_arm") for side in SIDES}

    def positions(self):
        return {side: np.asarray(arm.get_joint_pos(), dtype=np.float64)
                for side, arm in self._arms.items()}

    def command(self, q):
        for side, arm in self._arms.items():
            arm.set_joint_pos(np.asarray(q[side], dtype=np.float64))

    def estop(self):
        """(hardware button pressed, software E-stop enabled)."""
        status = self._robot.estop.get_status()
        return bool(status["button_pressed"]), bool(status["software_estop_enabled"])

    def set_estop(self, active):
        if active:
            self._robot.estop.activate()
        else:
            self._robot.estop.deactivate()

    def health(self):
        """(heartbeat from the robot is live, seconds since the last beat)."""
        beat = self._robot.heartbeat.get_status()
        return bool(beat.get("is_active")), float(beat.get("time_since_last") or 0.0)

    def close(self):
        self._robot.shutdown()


class HandLinkBlocked(RuntimeError):
    """Connecting to this hand now would lock it (ambiguous laptop address)."""


class WujiHands:
    """Both robot Wuji Hand 2 devices, each connected and enabled on its own.

    ``connect`` refuses a hand whose network path is ambiguous (see
    ``connection_monitor.ambiguous_hand_source``): the failed attempt that
    would follow locks the hand until a 12 V power cycle.
    """

    def __init__(self, serials=None):
        import wuji_sdk

        wuji_sdk.set_log_level("warn")
        self._sdk = wuji_sdk
        self._manager = wuji_sdk.SdkManager.instance()
        self.serials = {side: (serials or {}).get(side) for side in SIDES}
        self.addresses = {side: None for side in SIDES}
        self.firmware = {side: None for side in SIDES}
        self._lock = threading.Lock()
        self._latest = {side: None for side in SIDES}
        self._received_at = {side: None for side in SIDES}
        self._hands, self._subscriptions, self._publishers = {}, {}, {}

    def discover(self):
        """Fill in each side's serial and address from the SDK discovery scan."""
        found = {side: [] for side in SIDES}
        for device in self._manager.scan():
            sn = str(device.sn)
            # Wuji serials carry handedness in the 4th character: J left, K right.
            side = {"J": "left", "K": "right"}.get(sn[3:4])
            if device.device_type == self._sdk.DeviceType.WujiHand2 and side:
                found[side].append((sn, str(device.address)))
        for side in SIDES:
            matches = [hand for hand in found[side] if self.serials[side] in (None, hand[0])]
            self.addresses[side] = matches[0][1] if len(matches) == 1 else None
            if len(matches) == 1:
                self.serials[side] = matches[0][0]
        return found

    def connect(self, side):
        from .connection_monitor import ambiguous_hand_source

        if side in self._hands:
            return
        if self.addresses[side] is None:
            raise RuntimeError("not found by Wuji discovery")
        unsafe = ambiguous_hand_source(self.addresses[side].rsplit(":", 1)[0])
        if unsafe:
            raise HandLinkBlocked(unsafe)
        alias = f"vega_{side}_wuji_hand"
        hand = self._manager.connect(sn=self.serials[side], device_name=alias)
        try:
            reported = str(hand.handedness().get()).lower()
            if side not in reported:
                raise RuntimeError(f"{self.serials[side]} reports {reported}, expected {side}")
            subscription = hand.joint_states().subscribe_with_callback(
                functools.partial(self._on_state, side))
        except BaseException:
            self._manager.disconnect(alias)
            raise
        with self._lock:
            self._hands[side] = hand
            self._subscriptions[side] = subscription
            self.firmware[side] = hand.info.firmware_version

    def _on_state(self, side, frame):
        if frame.num_joints == 20:
            with self._lock:
                self._latest[side] = np.array([j.position for j in frame.joints])
                self._received_at[side] = time.monotonic()

    def is_connected(self, side):
        return side in self._hands

    def is_ready(self, side):
        return side in self._publishers

    def feedback_age(self, side):
        with self._lock:
            received = self._received_at[side]
        return None if received is None else time.monotonic() - received

    def positions(self):
        with self._lock:
            return {side: None if q is None else q.copy() for side, q in self._latest.items()}

    def wait_position(self, side, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            q = self.positions()[side]
            if q is not None:
                return q
            time.sleep(0.02)
        raise RuntimeError("connected but no joint feedback")

    def enable(self, side, hold):
        """Enable one hand holding ``hold`` so enabling cannot move a finger."""
        publisher = self._hands[side].joint_command().publish()
        for _ in range(5):
            self._send(publisher, hold)
        self._hands[side].enable()
        self._publishers[side] = publisher

    def _send(self, publisher, angles):
        publisher.send([self._sdk.JointCommand(position=float(p), velocity=0.0, effort=0.0)
                        for p in angles])

    def command(self, side, angles):
        publisher = self._publishers.get(side)
        if publisher is not None:
            self._send(publisher, angles)

    def close(self):
        for side, hand in list(self._hands.items()):
            publisher = self._publishers.pop(side, None)
            if publisher is not None:
                try:
                    hand.disable()
                except Exception:
                    hand.emergency_stop()
                publisher.close()
            if side in self._subscriptions:
                self._subscriptions.pop(side).close()
            self._manager.disconnect(f"vega_{side}_wuji_hand")
        self._hands.clear()


class FakeArms:
    """Kinematic stand-in for ``DexcontrolArms`` (dashboard rehearsal, tests)."""

    def __init__(self, start, lag=0.5):
        self._q = _copy(start)
        self.lag = float(lag)
        self.button, self.software = False, False

    def positions(self):
        return _copy(self._q)

    def command(self, q):
        self._q = {side: self._q[side] + self.lag * (np.asarray(q[side]) - self._q[side])
                   for side in SIDES}

    def estop(self):
        return self.button, self.software

    def set_estop(self, active):
        self.software = bool(active)

    def health(self):
        return True, 0.0

    def close(self):
        pass


class FakeHands:
    """Stand-in for ``WujiHands`` whose fingers reach each command at once."""

    def __init__(self, start=None, connected=SIDES):
        self._q = {side: np.zeros(20) if start is None else np.array(start[side])
                   for side in SIDES}
        self.reachable = set(connected)
        self._connected, self._ready = set(), set()
        self.serials = {side: f"fake-{side}" for side in SIDES}
        self.firmware = {side: "fake" for side in SIDES}
        self.addresses = {side: "fake" for side in SIDES}

    def discover(self):
        pass

    def connect(self, side):
        if side not in self.reachable:
            raise RuntimeError("fake hand unreachable")
        self._connected.add(side)

    def is_connected(self, side):
        return side in self._connected

    def is_ready(self, side):
        return side in self._ready

    def feedback_age(self, side):
        return 0.0 if side in self._connected else None

    def positions(self):
        return {side: self._q[side].copy() if side in self._connected else None
                for side in SIDES}

    def wait_position(self, side, timeout=3.0):
        return self._q[side].copy()

    def enable(self, side, hold):
        self._q[side] = np.array(hold, dtype=np.float64)
        self._ready.add(side)

    def command(self, side, angles):
        if side in self._ready:
            self._q[side] = np.array(angles, dtype=np.float64)

    def close(self):
        self._ready.clear()
        self._connected.clear()


class SyntheticPicoHand:
    """Pico hand-tracking stand-in that moves the wrist on a slow ellipse."""

    def __init__(self, side, amplitude=0.05, period=6.0):
        self.side, self.amplitude, self.period = side, float(amplitude), float(period)
        self._start = time.monotonic()
        self._revision = 0

    def read(self):
        now = time.monotonic()
        phase = 2 * math.pi * (now - self._start) / self.period
        pose = np.eye(4)
        pose[:3, 3] = (0.3 + self.amplitude * math.sin(phase),
                       (0.2 if self.side == "left" else -0.2),
                       1.0 + self.amplitude * math.cos(phase))
        self._revision += 1
        frame = WristPoseFrame(pose, 0.0, time.time_ns(), now)
        return WristPoseSnapshot(frame, self._revision, self._revision, None)

    def close(self):
        pass


class HeldWristImu:
    """Palm-IMU stand-in with no orientation, so the wrist joints hold."""

    def read(self, anchor_pose, *, now=None):
        return SimpleNamespace(pose=None)

    def recenter(self, anchor_pose):
        pass

    def close(self):
        pass


class PicoArmMapping:
    """``vega_bimanual_sim`` Pico-mode arm mapping on commanded joints, not physics.

    The IK's posture, roll and elbow preferences are all built around its arm's
    ``home``; ``posture`` replaces the sim home (the reset pose on hardware) so
    tracking does not drag the arms toward the sim's pose.
    """

    def __init__(self, model, data, arms, pose_sources, imu_sources, *, pico_scale=0.6,
                 posture=None):
        if posture is not None:
            arms = {side: copy.copy(arm) for side, arm in arms.items()}
            for side, arm in arms.items():
                arm.home = np.clip(np.asarray(posture[side], dtype=np.float64),
                                   arm.lower, arm.upper)
                arm.target = arm.home.copy()
        self.model, self.arms = model, arms
        self.pose_sources, self.imu_sources = pose_sources, imu_sources
        names = {side: body_names(side) for side in SIDES}
        self.ik = {side: _side_ik(model, data, arms[side], upper_only=True) for side in SIDES}
        self.wrist_ik = {
            side: WristOrientationIK(
                model, data, arms[side], joint_indices=(4, 5, 6),
                max_total_step=0.20, joint_limit_margin=0.08,
                ee_body_name=names[side]["ee"])
            for side in SIDES
        }
        self.mappers = {side: PicoHandArmMapper(pico_scale) for side in SIDES}
        body = functools.partial(mujoco.mj_name2id, model, mujoco.mjtObj.mjOBJ_BODY)
        self.ee_ids = {side: body(names[side]["ee"]) for side in SIDES}
        self.shoulder_ids = {side: body(names[side]["shoulder"]) for side in SIDES}
        self.scratch = mujoco.MjData(model)
        self.origins = {}
        self.anchors = {}
        self.wrist_references = {}
        # As in the sim's Pico home hold: the IK prefers its own elbow posture,
        # so hold the anchored pose until the operator actually moves.
        self.holding = {side: True for side in SIDES}
        self.status = {side: "not started" for side in SIDES}

    def _pose(self, side, q):
        self.scratch.qpos[self.arms[side].qpos_ids] = q
        mujoco.mj_kinematics(self.model, self.scratch)
        ee = self.ee_ids[side]
        pose = np.eye(4)
        pose[:3, :3] = self.scratch.xmat[ee].reshape(3, 3)
        pose[:3, 3] = self.scratch.xpos[ee]
        return pose, self.scratch.xpos[self.shoulder_ids[side]].copy()

    def recenter(self, q):
        """Pair the next fresh Pico and IMU frames with the arm pose ``q``."""
        for side in SIDES:
            pose, _ = self._pose(side, q[side])
            self.origins[side] = pose
            self.anchors[side] = np.array(q[side], dtype=np.float64)
            self.wrist_references[side] = self.anchors[side][4:].copy()
            self.mappers[side].recenter()
            self.imu_sources[side].recenter(pose)
            self.holding[side] = True
            self.status[side] = "calibrating"

    def _operator_moved(self, side, wrist_rotation):
        mapper = self.mappers[side]
        if (mapper.filtered_hand_position is not None and mapper.hand_origin is not None and
                np.linalg.norm(mapper.filtered_hand_position - mapper.hand_origin) >
                mapper.deadband_m):
            return True
        if wrist_rotation is None:
            return False
        relative = wrist_rotation @ self.origins[side][:3, :3].T
        angle = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        return angle > math.radians(2.0)

    def update(self, q, now):
        """Joint goals per side; a side without fresh tracking is left out (hold)."""
        goals = {}
        for side in SIDES:
            arm = self.arms[side]
            seed = np.asarray(q[side], dtype=np.float64)
            pose, shoulder = self._pose(side, seed)
            target = self.mappers[side].update(
                self.pose_sources[side].read(), pose[:3, 3], now, shoulder)
            self.status[side] = self.mappers[side].status
            if target is None:
                continue
            imu = self.imu_sources[side].read(self.origins[side], now=now)
            if self.holding[side]:
                self.holding[side] = not self._operator_moved(
                    side, None if imu.pose is None else imu.pose[:3, :3])
                if self.holding[side]:
                    self.status[side] = "anchored, move to start"
                    goals[side] = self.anchors[side].copy()
                    continue
            target_pose = pose.copy()
            target_pose[:3, 3] = target
            goal = self.ik[side].solve(target_pose, seed=seed)
            if imu.pose is not None:
                neutral = goal.copy()
                neutral[4:] = self.wrist_references[side]
                neutral_pose, _ = self._pose(side, neutral)
                wrist_target = relative_wrist_rotation(
                    imu.pose[:3, :3], self.origins[side][:3, :3], neutral_pose[:3, :3])
                goal = self.wrist_ik[side].solve(wrist_target, seed=goal)
                # The wrist can move the palm; correct position once more.
                goal = self.ik[side].solve(target_pose, seed=goal)
            goals[side] = np.clip(_limit_joint_goal(goal, seed), arm.lower, arm.upper)
        return goals


class TrackerArmMapping:
    """PICO Motion Trackers (XRoboToolkit) to all seven joints of each arm.

    The ``vega_bimanual_sim`` tracker-mode mapping (GMR-style two-stage IK).
    ``recenter`` pins the operator's two wrist-tracker poses to the robot's
    current arm pose at the next fresh tracker frame; both arms hold until then.
    The IK's posture target is that calibration pose, so standing still holds.
    """

    def __init__(self, model, source, arms, *, ik_config, left_serial=None,
                 right_serial=None, human_height=None):
        from .gmr_arm_retarget import GmrArmRetargeter, TrackerArmMapper

        self.source, self.arms = source, arms
        self.mapper = TrackerArmMapper(
            GmrArmRetargeter(model, {side: arm_joint_names(side) for side in SIDES},
                             ik_config, actual_human_height=human_height),
            left_serial=left_serial, right_serial=right_serial)
        self._anchor = None
        self.status = {side: "not started" for side in SIDES}

    def recenter(self, q):
        self._anchor = _copy(q)
        self.status = {side: "calibrating" for side in SIDES}

    def update(self, q, now):
        snapshot = self.source.read()
        if self._anchor is not None:
            try:
                self.mapper.calibrate(snapshot, self._anchor, now)
            except ValueError as exc:
                self.status = {side: f"waiting: {exc}" for side in SIDES}
                return {}
            self._anchor = None
            self.status = dict(self.mapper.status)
            return {}  # the calibration frame itself holds the pose
        goals = self.mapper.update(snapshot, now)
        self.status = dict(self.mapper.status)
        return {side: np.clip(_limit_joint_goal(goal, q[side]), self.arms[side].lower,
                              self.arms[side].upper)
                for side, goal in goals.items()}


class _HandGoal:
    """``TrackingController`` output adapter: remembers the latest finger target."""

    def __init__(self):
        self.target = None

    def set_target(self, angles):
        self.target = np.array(angles, dtype=np.float64)


@dataclass
class TeleopLimits:
    max_arm_speed: float = 1.0       # rad/s per joint while following
    max_hand_speed: float = 5.0      # rad/s per finger joint
    reset_speed: float = 0.30        # rad/s peak per joint during reset
    reset_acceleration: float = 0.60
    reset_clearance: float = 0.03    # m, kept along the whole reset path
    guard_clearance: float = 0.015   # m, teleop targets closer than this are held
    tracking_lag_warning: float = 0.30  # rad between command and measured: warn once
    tracking_fault: float | None = None  # rad; set to stop and hold instead (off by default)


class HardwareTeleop:
    """Reset / start / pause state machine streaming to both arms and both hands.

    One IO thread owns every hardware call; a slower teleop thread computes
    targets. ``io_step`` and ``teleop_step`` can also be driven directly.
    """

    def __init__(self, arms_io, hands_io, mapping, glove_sources, retargeters,
                 guard_checker, planner, reset_q, *, arm_lower, arm_upper,
                 hand_lower, hand_upper, limits=None, clock=time.monotonic,
                 threaded_planning=True, log_csv=None):
        self.arms_io, self.hands_io = arms_io, hands_io
        self.mapping = mapping
        self.glove_sources, self.retargeters = glove_sources, retargeters
        self.guard_checker, self.planner = guard_checker, planner
        self.reset_q = _copy(reset_q)
        self.arm_lower, self.arm_upper = _copy(arm_lower), _copy(arm_upper)
        self.hand_lower, self.hand_upper = _copy(hand_lower), _copy(hand_upper)
        self.limits = limits or TeleopLimits()
        self.clock = clock
        self.threaded_planning = threaded_planning
        self._lag_warned = False
        self._csv = None
        self._csv_ticks = 0
        if log_csv:
            self._csv = open(log_csv, "w", encoding="utf-8")
            self._csv.write("t,state," + ",".join(
                f"{kind}_{side[0]}{i}" for kind in ("cmd", "meas") for side in SIDES
                for i in range(1, 8)) + "\n")
        self._lock = threading.RLock()
        self.messages = collections.deque(maxlen=8)
        self.state = "paused"
        self.started_once = False
        measured = arms_io.positions()
        self.measured = _copy(measured)
        self.command = {side: np.clip(measured[side], self.arm_lower[side], self.arm_upper[side])
                        for side in SIDES}
        self.goal = _copy(self.command)
        # A hand that is not connected yet is modelled open until sync_hand().
        hands = hands_io.positions()
        self.hand_command = {side: np.clip(np.zeros(20) if hands[side] is None else hands[side],
                                           self.hand_lower[side], self.hand_upper[side])
                             for side in SIDES}
        self.hand_goal = _copy(self.hand_command)
        self.fingers = {side: TrackingController(retargeters[side], _HandGoal())
                        for side in SIDES}
        self.clearance = None
        self.estop_flags = tuple(arms_io.estop())
        self._pending = collections.deque()
        self._release_deadline = None
        self._guard_holding = False
        self._follower = None
        self._follow_start = 0.0
        self._plan_token = None
        self._threads = []
        self._running = False
        self._next_display_check = 0.0
        if any(self.estop_flags):
            self.state = "estop"
            self._log("Robot E-stop is active; release it (U) before Reset or Start.")
        else:
            self._log("Paused, holding the current pose. Reset (R) or Start (S).")

    def _log(self, text):
        self.messages.appendleft(time.strftime("%H:%M:%S ") + text)
        print(text, flush=True)

    def _freeze(self):
        self.goal = _copy(self.command)
        self.hand_goal = _copy(self.hand_command)

    # Operator commands (any thread) -------------------------------------
    def request_start(self):
        with self._lock:
            if self.state != "paused":
                self._log(f"Start ignored while {self.state}.")
                return False
            if any(self.estop_flags):
                self._log("Start refused: the robot E-stop is active.")
                return False
            self.mapping.recenter(self.command)
            # Fresh controllers reset the retargeters on their first glove frame.
            self.fingers = {side: TrackingController(self.retargeters[side], _HandGoal())
                            for side in SIDES}
            self._guard_holding = False
            self._freeze()
            self.state = "running"
            self.started_once = True
            self._log("Running: arms re-anchored at the current pose; hands follow the gloves.")
            return True

    def request_pause(self):
        with self._lock:
            if self.state in ("running", "resetting", "planning", "fault"):
                was = self.state
                self._plan_token = None
                self._follower = None
                self._freeze()
                self.state = "paused"
                detail = {"running": "frozen", "fault": "fault acknowledged"}.get(
                    was, "reset stopped")
                self._log(f"Paused ({detail}).")
                return True
            return False

    def request_reset(self):
        with self._lock:
            if self.state not in ("paused", "running", "fault"):
                self._log(f"Reset ignored while {self.state}.")
                return False
            if any(self.estop_flags):
                self._log("Reset refused: the robot E-stop is active.")
                return False
            self._freeze()
            self.state = "planning"
            token = self._plan_token = object()
            start = join_arms(self.command)
            fingers = _copy(self.hand_command)
            self._log("Planning a collision-free path to the reset pose...")
        if self.threaded_planning:
            threading.Thread(target=self._plan_reset, args=(token, start, fingers),
                             daemon=True).start()
        else:
            self._plan_reset(token, start, fingers)
        return True

    def request_estop(self):
        with self._lock:
            self._pending.append(("estop", True))
            self._plan_token = None
            self._follower = None
            self._freeze()
            self.state = "estop"
            self._log("E-STOP: software E-stop requested; hands frozen.")

    def request_release(self):
        with self._lock:
            if self.state != "estop":
                return False
            self._pending.append(("estop", False))
            self._release_deadline = self.clock() + 2.0
            self._log("Releasing the software E-stop...")
            return True

    def sync_hand(self, side, angles):
        """A hand just connected holding ``angles``: continue from there, no jump."""
        with self._lock:
            q = np.clip(np.asarray(angles, dtype=np.float64),
                        self.hand_lower[side], self.hand_upper[side])
            self.hand_command[side] = q
            self.hand_goal[side] = q.copy()

    # Reset planning ------------------------------------------------------
    def _plan_reset(self, token, start, fingers):
        try:
            checker = self.planner.checker
            for side in SIDES:
                checker.set_fingers(side, fingers[side])
            began = time.monotonic()
            path = self.planner.plan(start, join_arms(self.reset_q),
                                     cancelled=lambda: self._plan_token is not token)
            follower = PathFollower(path, max_velocity=self.limits.reset_speed,
                                    max_acceleration=self.limits.reset_acceleration)
            elapsed = time.monotonic() - began
        except PlanningError as exc:
            with self._lock:
                if self._plan_token is token and self.state == "planning":
                    self.state = "paused"
                    self._log(f"Reset refused: {exc}")
            return
        except Exception as exc:  # keep the IO loop alive; report the failure
            with self._lock:
                if self._plan_token is token:
                    self.state = "paused"
                    self._log(f"Reset planning failed: {exc}")
            return
        with self._lock:
            if self._plan_token is not token or self.state != "planning":
                return
            self._follower = follower
            self._follow_start = self.clock()
            self.state = "resetting"
            route = "direct" if len(path) == 2 else f"{len(path) - 2} via-point(s)"
            self._log(f"Resetting: {route}, {follower.length:.2f} rad over "
                      f"{follower.duration:.1f} s (planned in {elapsed:.2f} s, "
                      f"{self.planner.checks} clearance checks).")

    # IO thread -----------------------------------------------------------
    def io_step(self):
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for name, value in pending:
            if name == "estop":
                self.arms_io.set_estop(value)
        measured = self.arms_io.positions()
        flags = self.arms_io.estop()
        now = self.clock()
        with self._lock:
            self.measured = _copy(measured)
            self.estop_flags = tuple(flags)
            if any(flags) and self.state != "estop":
                self._plan_token = None
                self._follower = None
                self._freeze()
                self.state = "estop"
                self._log("Robot E-stop is active (button or software); holding.")
            elif self._release_deadline is not None and self.state == "estop":
                # The E-stop status topic can lag the release request.
                if not any(flags):
                    self._release_deadline = None
                    self.command = {side: np.clip(measured[side], self.arm_lower[side],
                                                  self.arm_upper[side]) for side in SIDES}
                    self._freeze()
                    self.state = "paused"
                    self._log("E-stop released; paused at the measured pose.")
                elif now > self._release_deadline:
                    self._release_deadline = None
                    self._log("E-stop still active; is the hardware button pressed?")
            if self.state == "resetting":
                self._step_reset(now, measured)
            elif self.state == "running":
                self.command = _slew(self.command, self.goal,
                                     self.limits.max_arm_speed / ARM_RATE_HZ)
                self.hand_command = _slew(self.hand_command, self.hand_goal,
                                          self.limits.max_hand_speed / ARM_RATE_HZ)
                self._check_tracking(measured)
            arm_command = _copy(self.command)
            hand_command = _copy(self.hand_command)
            state = self.state
        self._write_log(now, state, arm_command, measured)
        self.arms_io.command(arm_command)
        for side in SIDES:
            self.hands_io.command(side, hand_command[side])

    def _step_reset(self, now, measured):
        if self._check_tracking(measured):
            return
        elapsed = now - self._follow_start
        self.command = split_arms(self._follower.at(elapsed))
        if elapsed >= self._follower.duration:
            self.command = _copy(self.reset_q)
            self._follower = None
            self._freeze()
            self.state = "paused"
            self._log("At the reset pose. Hold forearms forward, palms down, then Start (S).")

    def _write_log(self, now, state, command, measured):
        """Commanded and measured arm joints at 20 Hz, for diagnosing motion."""
        self._csv_ticks += 1
        if self._csv is None or self._csv_ticks % 5:
            return
        values = [command[side] for side in SIDES] + [measured[side] for side in SIDES]
        self._csv.write(f"{now:.3f},{state}," + ",".join(
            f"{v:.4f}" for q in values for v in q) + "\n")

    def _check_tracking(self, measured):
        error = max(float(np.max(np.abs(measured[side] - self.command[side])))
                    for side in SIDES)
        if error > self.limits.tracking_lag_warning and not self._lag_warned:
            self._lag_warned = True
            self._log(f"Warning: an arm is {error:.2f} rad behind its command "
                      "(slow or blocked?); still running.")
        elif error < self.limits.tracking_lag_warning / 2:
            self._lag_warned = False
        if self.limits.tracking_fault is None or error <= self.limits.tracking_fault:
            return False
        self.command = {side: np.clip(measured[side], self.arm_lower[side],
                                      self.arm_upper[side]) for side in SIDES}
        self._plan_token = None
        self._follower = None
        self._freeze()
        self.state = "fault"
        self._log(f"FAULT: an arm is {error:.2f} rad from its command (blocked?); "
                  "holding where it is. Check the robot, then Reset.")
        return True

    # Teleop thread -------------------------------------------------------
    def teleop_step(self):
        now = self.clock()
        with self._lock:
            state = self.state
            command = _copy(self.command)
            hand_command = _copy(self.hand_command)
            fingers = self.fingers
        if state != "running":
            if now >= self._next_display_check:
                for side in SIDES:
                    self.guard_checker.set_fingers(side, hand_command[side])
                clearance = self.guard_checker.clearance(join_arms(command))
                with self._lock:
                    self.clearance = clearance
                self._next_display_check = now + 0.25
            return
        hand_goal = {}
        for side in SIDES:
            fingers[side].update(self.glove_sources[side].read(), now)
            target = fingers[side].hand.target
            hand_goal[side] = (hand_command[side] if target is None else
                               np.clip(target, self.hand_lower[side], self.hand_upper[side]))
        goals = self.mapping.update(command, now)
        goal = {side: goals.get(side, command[side]) for side in SIDES}
        for side in SIDES:
            self.guard_checker.set_fingers(side, hand_goal[side])
        clearance = self.guard_checker.clearance(join_arms(goal))
        held = False
        if clearance.distance_m < self.limits.guard_clearance:
            current = self.guard_checker.clearance(join_arms(command))
            if clearance.distance_m < current.distance_m:
                goal, hand_goal, held = command, hand_command, True
        with self._lock:
            if self.state != "running" or self.fingers is not fingers:
                return
            self.goal, self.hand_goal = _copy(goal), _copy(hand_goal)
            self.clearance = clearance
            if held and not self._guard_holding:
                self._log(f"Guard: holding, {clearance.distance_m * 100:.1f} cm "
                          f"({clearance.pair}). Move back to continue.")
            self._guard_holding = held

    # Threads and status --------------------------------------------------
    def _loop(self, step, rate):
        period = 1.0 / rate
        next_tick = time.monotonic()
        while self._running:
            try:
                step()
            except Exception as exc:
                with self._lock:
                    if self.state != "estop":
                        self._plan_token = None
                        self._follower = None
                        self._freeze()
                        self.state = "fault"
                    self._log(f"FAULT in {step.__name__}: {exc}")
                time.sleep(0.2)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def start(self):
        self._running = True
        self._threads = [
            threading.Thread(target=self._loop, args=(self.io_step, ARM_RATE_HZ),
                             name="teleop-io", daemon=True),
            threading.Thread(target=self._loop, args=(self.teleop_step, TELEOP_RATE_HZ),
                             name="teleop-map", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self):
        """Freeze, keep streaming the hold briefly, then stop both threads."""
        self.request_pause()
        time.sleep(0.3)
        self._running = False
        for thread in self._threads:
            thread.join(timeout=2.0)
        if self._csv is not None:
            self._csv.close()
            self._csv = None

    def status(self):
        with self._lock:
            return {
                "state": self.state,
                "messages": list(self.messages),
                "command": _copy(self.command),
                "measured": _copy(self.measured),
                "hand_command": _copy(self.hand_command),
                "arm_status": dict(self.mapping.status),
                "finger_status": {side: self.fingers[side].status for side in SIDES},
                "clearance": self.clearance,
                "estop": self.estop_flags,
                "reset_remaining": (None if self._follower is None else max(
                    0.0, self._follower.duration - (self.clock() - self._follow_start))),
            }


@dataclass(frozen=True)
class LinkStatus:
    level: str  # "ok", "wait", "fail" or "off"
    detail: str
    hint: str = ""


DEVICE_LINKS = ("Robot arms", "Left hand", "Right hand", "Left glove", "Right glove")
DOT_COLORS = {"ok": "#1d8a3a", "wait": "#c58a00", "fail": "#b3261e", "off": "#9aa0a6"}


class _GloveSlot:
    """Glove input for the controller, before and after its glove connects."""

    def __init__(self):
        self.source = None

    def read(self):
        source = self.source
        if source is None:
            return SourceSnapshot(None, 0, 0, "glove not connected")
        return source.read()


class _ImuSlot:
    """Palm-IMU input that keeps its anchor so a late glove is recentred."""

    def __init__(self):
        self.source = None
        self.anchor = None

    def attach(self, source):
        if self.anchor is not None:
            source.recenter(self.anchor)
        self.source = source

    def read(self, anchor_pose, *, now=None):
        source = self.source
        return SimpleNamespace(pose=None) if source is None else source.read(anchor_pose, now=now)

    def recenter(self, anchor_pose):
        self.anchor = anchor_pose
        if self.source is not None:
            self.source.recenter(anchor_pose)


class PicoLinks:
    """Arm input from DexStream hand tracking: mapping plus server and hand links."""

    link_names = ("Pico server", "Pico left hand", "Pico right hand")
    NAMES = {"DexStream": "Pico server", "left hand tracking": "Pico left hand",
             "right hand tracking": "Pico right hand"}
    LEVELS = {"ok": "ok", "warn": "wait", "fail": "fail"}

    def __init__(self, pico_scale=0.6):
        from .connection_monitor import PicoFrames

        self.pico_scale = pico_scale
        self._frames = PicoFrames()
        self.sources = self._frames.sources

    def make_mapping(self, model, data, arms, imu_slots, posture):
        return PicoArmMapping(model, data, arms, self.sources, imu_slots,
                              pico_scale=self.pico_scale, posture=posture)

    def statuses(self):
        from .connection_monitor import interface_addresses, is_wireless

        wifi_ip = next((ip for name, ip in interface_addresses() if is_wireless(name)), None)
        return {self.NAMES[check.name]: LinkStatus(self.LEVELS[check.status], check.detail,
                                                   check.hint)
                for check in self._frames.checks(wifi_ip)}

    def close(self):
        self._frames.close()


class SyntheticPicoLinks:
    """Arm input for --synthetic-input: Pico-hand mapping on synthetic wrists."""

    link_names = ("Pico server", "Pico left hand", "Pico right hand")

    def __init__(self):
        self.sources = {side: SyntheticPicoHand(side) for side in SIDES}

    def make_mapping(self, model, data, arms, imu_slots, posture):
        return PicoArmMapping(model, data, arms, self.sources, imu_slots, posture=posture)

    def statuses(self):
        return {name: LinkStatus("ok", "synthetic input") for name in self.link_names}

    def close(self):
        pass


class _LockedSource:
    """Serialises ``read`` so the mapping and the link checks can share one source."""

    def __init__(self, source):
        self._source = source
        self._lock = threading.Lock()

    def read(self):
        with self._lock:
            return self._source.read()

    def __getattr__(self, name):
        return getattr(self._source, name)


class TrackerLinks:
    """Arm input from PICO Motion Trackers via XRoboToolkit: mapping plus links."""

    link_names = ("Tracker service", "Left tracker", "Right tracker")

    def __init__(self, source, *, ik_config, left_serial=None, right_serial=None,
                 human_height=None, service_running=None):
        self.source = _LockedSource(source)
        self.ik_config, self.human_height = ik_config, human_height
        self.configured = {"left": left_serial, "right": right_serial}
        self.service_running = service_running or (lambda: subprocess.run(
            ["pgrep", "-f", "RoboticsServiceProcess"], capture_output=True).returncode == 0)
        self.mapping = None

    def make_mapping(self, model, data, arms, imu_slots, posture):
        self.mapping = TrackerArmMapping(
            model, self.source, arms, ik_config=self.ik_config,
            left_serial=self.configured["left"], right_serial=self.configured["right"],
            human_height=self.human_height)
        return self.mapping

    def statuses(self):
        from .xr_tracker_source import assign_tracker_sides

        snapshot = self.source.read()
        frame = snapshot.frame
        live = frame is not None and time.monotonic() - frame.received_at < 0.5
        out = {}
        if not self.service_running():
            out["Tracker service"] = LinkStatus(
                "fail", "XRoboToolkit PC service not running",
                "start it: cd /opt/apps/roboticsservice && bash runService.sh "
                "(the launcher starts it)")
        elif snapshot.error:
            out["Tracker service"] = LinkStatus("fail", snapshot.error)
        elif not live:
            out["Tracker service"] = LinkStatus(
                "wait", "service running, no live tracker data",
                "on the headset: XRoboToolkit, Motion Tracker tab, Send; trackers on")
        else:
            stale = getattr(self.source, "stale_duplicates", 0)
            out["Tracker service"] = LinkStatus(
                "ok", f"{len(frame.poses)} live tracker(s)"
                + (f"; ignoring {stale} frozen duplicate entr{'y' if stale == 1 else 'ies'}"
                   if stale else ""))
        assigned = None if self.mapping is None else self.mapping.mapper.serials
        provisional = None
        if assigned is None and live:
            try:
                provisional = assign_tracker_sides(
                    frame.poses, left_serial=self.configured["left"],
                    right_serial=self.configured["right"])
            except ValueError as exc:
                problem = str(exc)
        for side in SIDES:
            name = f"{side.capitalize()} tracker"
            if assigned is not None:
                serial = assigned[side]
                if live and serial in frame.poses:
                    out[name] = LinkStatus("ok", f"{serial} tracking")
                else:
                    out[name] = LinkStatus("fail", f"{serial} lost",
                                           "check the tracker is on and in the headset's view")
            elif provisional is not None:
                out[name] = LinkStatus("ok", f"{provisional[side]} live (sides are fixed at Start)")
            elif live:
                out[name] = LinkStatus("wait", problem,
                                       "wear exactly two trackers, or pass the tracker serials")
            else:
                out[name] = LinkStatus("wait", "no live tracker data")
        return out

    def close(self):
        self.source.close()


class TeleopApp:
    """Connects every device in the background; a failure never closes the app.

    Each link has a status (a dot on the dashboard) and failed connections are
    retried. The controls work once the robot arms are connected; a hand or
    glove that connects later joins the running session without a jump.
    """

    def __init__(self, *, connect_robot, hands_io, connect_glove, arm_input, retargeters,
                 reset_q, limits, retry_s=5.0, start_teleop=True, log_csv=None):
        self.connect_robot, self.hands_io = connect_robot, hands_io
        self.connect_glove, self.arm_input = connect_glove, arm_input
        self.retargeters, self.reset_q, self.limits = retargeters, reset_q, limits
        self.retry_s = float(retry_s)
        self.start_teleop = start_teleop
        self.log_csv = log_csv
        self.model, self.data, self.arms, self.hand_models = load_bimanual_scene()
        self.planner = ResetPlanner(
            ArmClearanceChecker(self.model, self.arms, self.hand_models),
            np.concatenate([self.arms[side].lower for side in SIDES]),
            np.concatenate([self.arms[side].upper for side in SIDES]),
            margin=limits.reset_clearance)
        self._lock = threading.Lock()
        self.link_names = DEVICE_LINKS + tuple(arm_input.link_names)
        self.links = {name: LinkStatus("wait", "connecting...") for name in self.link_names}
        self.messages = collections.deque(maxlen=8)
        self.arms_io = None
        self.teleop = None
        self.gloves = {}
        self.glove_slots = {side: _GloveSlot() for side in SIDES}
        self.imu_slots = {side: _ImuSlot() for side in SIDES}
        self._next_try = collections.defaultdict(float)
        self._stop = threading.Event()
        self._threads = []

    def _log(self, text):
        self.messages.appendleft(time.strftime("%H:%M:%S ") + text)
        print(text, flush=True)

    def _set(self, name, level, detail, hint=""):
        with self._lock:
            previous = self.links[name]
            self.links[name] = LinkStatus(level, detail, hint)
        if previous.level != level and level in ("ok", "fail"):
            self._log(f"{name}: {'connected' if level == 'ok' else detail}")

    def _due(self, name, now):
        if now < self._next_try[name]:
            return False
        self._next_try[name] = now + self.retry_s
        return True

    # Connection steps (each runs on its own background thread) ----------
    def robot_step(self, now):
        name = "Robot arms"
        if self.arms_io is None:
            if not self._due(name, now):
                return
            try:
                self.arms_io = self.connect_robot()
            except Exception as exc:
                self._set(name, "fail", f"not connected ({exc})",
                          "check robot power and the Ethernet link to 192.168.50.x; retrying")
                return
            self._build_teleop()
        alive, since = self.arms_io.health()
        button, software = self.arms_io.estop()
        if not alive:
            self._set(name, "fail", f"no heartbeat from the robot for {since:.1f} s",
                      "check the robot and its Ethernet link")
        elif button or software:
            which = " and ".join(n for n, on in (("button", button), ("software", software)) if on)
            self._set(name, "wait", f"connected; E-stop ON ({which})",
                      "release the physical button, or U for the software E-stop")
        else:
            self._set(name, "ok", "connected, heartbeat live, E-stop off")

    def _build_teleop(self):
        mapping = self.arm_input.make_mapping(self.model, self.data, self.arms,
                                              self.imu_slots, self.reset_q)
        teleop = HardwareTeleop(
            self.arms_io, self.hands_io, mapping, self.glove_slots, self.retargeters,
            ArmClearanceChecker(self.model, self.arms, self.hand_models), self.planner,
            self.reset_q,
            arm_lower={side: self.arms[side].lower for side in SIDES},
            arm_upper={side: self.arms[side].upper for side in SIDES},
            hand_lower={side: self.hand_models[side].lower for side in SIDES},
            hand_upper={side: self.hand_models[side].upper for side in SIDES},
            limits=self.limits, log_csv=self.log_csv)
        if self.start_teleop:
            teleop.start()
        self.teleop = teleop

    def hands_step(self, now):
        from .connection_monitor import HAND_SUBNET_HINT

        hands = self.hands_io
        for side in SIDES:
            name = f"{side.capitalize()} hand"
            if not hands.is_connected(side):
                if not self._due(name, now):
                    continue
                try:
                    if hands.addresses[side] is None:
                        hands.discover()
                    hands.connect(side)
                except HandLinkBlocked as exc:
                    self._set(name, "fail", f"{hands.serials[side]} at "
                              f"{hands.addresses[side]} not connected: {exc}", HAND_SUBNET_HINT)
                    continue
                except Exception as exc:
                    found = hands.addresses[side] is not None
                    hands.addresses[side] = None  # rediscover on the next attempt
                    self._set(name, "fail", f"not connected ({exc})",
                              "answers discovery but refuses connections: power-cycle the "
                              "hand (12 V off ~10 s)" if found else
                              "not found: check the hand's 12 V power and its cable")
                    continue
            if not hands.is_ready(side):
                try:
                    hold = hands.wait_position(side)
                    teleop = self.teleop
                    if teleop is not None:
                        teleop.sync_hand(side, hold)  # before enable: streaming starts at hold
                    hands.enable(side, hold)
                except Exception as exc:
                    self._set(name, "fail", f"connected but not enabled ({exc})")
                    continue
            age = hands.feedback_age(side)
            label = f"{hands.serials[side]} (fw {hands.firmware[side]})"
            if age is not None and age < 0.5:
                self._set(name, "ok", f"{label} enabled, feedback live")
            else:
                self._set(name, "fail", f"{label}: no joint feedback for "
                          f"{'?' if age is None else f'{age:.1f}'} s",
                          "check the hand's power and cable")

    def gloves_step(self, now):
        for side in SIDES:
            name = f"{side.capitalize()} glove"
            if side not in self.gloves:
                if not self._due(name, now):
                    continue
                try:
                    glove, imu = self.connect_glove(side)
                except Exception as exc:
                    self._set(name, "fail", f"not connected ({exc})",
                              "turn the glove on; check the glove network: bash "
                              "scripts/configure_wuji_network.sh <glove adapter>; retrying")
                    continue
                self.gloves[side] = (glove, imu)
                self.glove_slots[side].source = glove
                self.imu_slots[side].attach(imu)
            glove = self.gloves[side][0]
            snapshot = glove.read()
            label = getattr(glove, "serial", None) or "synthetic glove"
            frame = snapshot.frame
            if snapshot.error:
                self._set(name, "fail", f"{label}: {snapshot.error}")
            elif frame is None:
                self._set(name, "wait", f"{label} connected, waiting for frames")
            elif time.monotonic() - frame.received_at < 0.3:
                self._set(name, "ok", f"{label} streaming")
            else:
                self._set(name, "fail", f"{label}: no frames for "
                          f"{time.monotonic() - frame.received_at:.1f} s",
                          "glove asleep, off or its link dropped")

    def input_step(self, now):
        for name, status in self.arm_input.statuses().items():
            self._set(name, status.level, status.detail, status.hint)

    # Commands, threads, status -----------------------------------------
    def request(self, name):
        teleop = self.teleop
        if teleop is None:
            self._log("Robot arms not connected yet: " + (
                "use the physical E-stop." if name == "estop" else f"{name} unavailable."))
            return False
        if name == "start":
            with self._lock:
                missing = [link for link, status in self.links.items() if status.level != "ok"]
            if missing:
                self._log("Starting without: " + ", ".join(missing) + " (those parts hold).")
        return getattr(teleop, f"request_{name}")()

    def _loop(self, step, period):
        while not self._stop.is_set():
            try:
                step(time.monotonic())
            except Exception as exc:
                self._log(f"{step.__name__} error: {exc}")
            self._stop.wait(period)

    def start(self):
        for step in (self.robot_step, self.hands_step, self.gloves_step, self.input_step):
            thread = threading.Thread(target=self._loop, args=(step, 0.5),
                                      name=step.__name__, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        self._stop.set()
        if self.teleop is not None:
            self.teleop.stop()
        # Hardware first (hands disabled, robot released), then the inputs.
        closeables = [self.hands_io, self.arms_io]
        closeables += [source for glove, imu in self.gloves.values() for source in (imu, glove)]
        closeables.append(self.arm_input)
        for closeable in closeables:
            if closeable is None:
                continue
            try:
                closeable.close()
            except Exception as exc:
                print(f"Warning: cleanup of {type(closeable).__name__} failed: {exc}",
                      file=sys.stderr)

    def status(self):
        with self._lock:
            links = dict(self.links)
            messages = list(self.messages)
        teleop = self.teleop
        status = teleop.status() if teleop is not None else {
            "state": "connecting", "messages": [], "reset_remaining": None}
        status["links"] = links
        # Newest first by the HH:MM:SS prefix; the stable sort keeps same-second order.
        status["messages"] = sorted(messages + status["messages"], key=lambda m: m[:8],
                                    reverse=True)[:8]
        return status


class Dashboard:
    """Tk window: state banner, command buttons with shortcuts, link dots, status."""

    def __init__(self, app, title="Vega-1U Pico + Wuji teleop"):
        import tkinter as tk

        self.app = app
        self.root = root = tk.Tk()
        root.title(title)
        root.configure(padx=14, pady=12)
        self.banner = tk.Label(root, text="", font=("TkDefaultFont", 22, "bold"),
                               fg="white", width=26, pady=8)
        self.banner.grid(row=0, column=0, columnspan=5, sticky="ew")
        actions = (
            ("Reset  [R]", "reset", "#2a6fb0"),
            ("Start  [S]", "start", "#1d8a3a"),
            ("Pause  [P / Space]", "pause", "#c58a00"),
            ("E-STOP  [Esc]", "estop", "#b3261e"),
            ("Release E-stop  [U]", "release", "#5f6368"),
        )
        for column, (label, name, color) in enumerate(actions):
            # takefocus=0: Space must never "click" whichever button has focus.
            tk.Button(root, text=label, command=functools.partial(app.request, name),
                      bg=color, fg="white", activebackground=color,
                      font=("TkDefaultFont", 13, "bold"), width=16, height=2,
                      takefocus=0).grid(row=1, column=column, padx=4, pady=10)
        links = tk.Frame(root)
        links.grid(row=2, column=0, columnspan=5, sticky="w")
        self.link_rows = {}
        for row, name in enumerate(app.link_names):
            dot = tk.Label(links, text="●", font=("TkDefaultFont", 15), fg=DOT_COLORS["wait"])
            dot.grid(row=row, column=0, padx=(0, 6))
            tk.Label(links, text=name, font=("TkDefaultFont", 11, "bold"), anchor="w",
                     width=15).grid(row=row, column=1, sticky="w")
            detail = tk.Label(links, text="", font=("TkDefaultFont", 10), anchor="w",
                              justify="left", wraplength=780)
            detail.grid(row=row, column=2, sticky="w")
            self.link_rows[name] = (dot, detail)
        self.details = tk.Label(root, text="", font=("TkFixedFont", 11), justify="left",
                                anchor="w")
        self.details.grid(row=3, column=0, columnspan=5, sticky="w", pady=(8, 0))
        self.log = tk.Label(root, text="", font=("TkFixedFont", 10), justify="left",
                            anchor="w", fg="#333333")
        self.log.grid(row=4, column=0, columnspan=5, sticky="w", pady=(8, 0))
        keys = {"r": "reset", "s": "start", "p": "pause", "space": "pause",
                "escape": "estop", "u": "release"}
        root.bind("<KeyPress>", lambda event: (
            app.request(keys[event.keysym.lower()]) if event.keysym.lower() in keys else None))
        root.focus_force()
        root.after(100, self._refresh)

    def _refresh(self):
        status = self.app.status()
        state = status["state"]
        banner = state.upper()
        if status["reset_remaining"] is not None:
            banner += f"  {status['reset_remaining']:.1f} s"
        self.banner.configure(text=banner, bg=STATE_COLORS.get(state, "#5f6368"))
        for name, (dot, detail) in self.link_rows.items():
            link = status["links"][name]
            dot.configure(fg=DOT_COLORS[link.level])
            hint = f"   → {link.hint}" if link.hint and link.level != "ok" else ""
            detail.configure(text=link.detail + hint)
        lines = []
        clearance = status.get("clearance")
        if clearance is not None:
            near = (">=10 cm" if clearance.pair is None
                    else f"{clearance.distance_m * 100:.1f} cm  {clearance.pair}")
            lines.append(f"Closest parts: {near}")
        if "command" in status:
            for side in SIDES:
                q = np.round(status["command"][side], 2)
                lines.append(f"{side.capitalize()}:  Pico {status['arm_status'][side]}  |  "
                             f"glove {status['finger_status'][side]}  |  "
                             f"arm {' '.join(f'{v:+.2f}' for v in q)}")
        self.details.configure(text="\n".join(lines))
        self.log.configure(text="\n".join(status["messages"]))
        self.root.after(100, self._refresh)

    def run(self):
        self.root.mainloop()


class _StaticRetargeter:
    """Retargeter stand-in for --synthetic-input: a relaxed, slightly curled hand."""

    def __init__(self, angles):
        self.angles = np.asarray(angles, dtype=np.float64)

    def step(self, points):
        return self.angles.copy()

    def reset(self):
        pass


def _connect_glove(args, side):
    """Connect one glove and its palm IMU; raises if it is unreachable."""
    import wuji_sdk

    from .imu_arm_source import PalmImuArmSource, PalmImuRetargeter
    from .sources import GloveSource

    wuji_sdk.set_log_level("warn")
    address = args.left_glove_address if side == "left" else args.right_glove_address
    serial = args.left_glove_sn if side == "left" else args.right_glove_sn
    glove = GloveSource(serial, args.user_id, sdk=wuji_sdk,
                        device_name=f"wuji_bimanual_glove_{side}", address=address)
    try:
        if glove.side != side:
            raise RuntimeError(f"glove at {address} reports {glove.side}, expected {side}")
        imu = PalmImuArmSource(
            glove.palm_imu_resource(),
            estimator=PalmImuRetargeter(
                imu_in_wrist_rotation=glove.palm_imu_mount_rotation(),
                hand_in_ee_rotation=HAND_IN_EE_ROTATION, motion_mode="orientation"))
    except BaseException:
        glove.close()
        raise
    return glove, imu


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake-robot", action="store_true",
                        help="Do not connect to the Vega arms or Wuji hands (rehearsal)")
    parser.add_argument("--synthetic-input", action="store_true",
                        help="Replace the arm input and gloves with synthetic input")
    parser.add_argument("--arm-input", choices=("tracker", "pico-hand"), default="tracker",
                        help="tracker: PICO Motion Trackers via XRoboToolkit drive all 7 joints "
                             "(GMR IK); pico-hand: DexStream hand tracking drives joints 1-4 "
                             "and the glove IMUs the wrists")
    parser.add_argument("--ik-config", default=str(TRACKER_IK_CONFIG),
                        help="GMR ik_configs JSON for tracker input")
    parser.add_argument("--left-tracker-sn")
    parser.add_argument("--right-tracker-sn")
    parser.add_argument("--human-height", type=float,
                        help="Operator height in metres for GMR scaling (tracker input)")
    parser.add_argument("--left-hand-sn")
    parser.add_argument("--right-hand-sn")
    parser.add_argument("--left-glove-sn")
    parser.add_argument("--right-glove-sn")
    parser.add_argument("--left-glove-address", default=LEFT_GLOVE_ADDRESS)
    parser.add_argument("--right-glove-address", default=RIGHT_GLOVE_ADDRESS)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--pico-scale", type=float, default=0.6)
    parser.add_argument("--reset-shoulder-back", type=float, default=RESET_SHOULDER_BACK,
                        help="Reset pose: upper arms this far (rad) behind vertical")
    parser.add_argument("--tracking-fault", type=float, default=0.0,
                        help="Stop and hold when an arm is this far (rad) behind its "
                             "command; 0 (default) only warns")
    parser.add_argument("--log-csv", default=time.strftime(
        "/tmp/vega-hardware-teleop-%Y%m%d-%H%M%S.csv"),
        help="Commanded and measured arm joints at 20 Hz")
    defaults = TeleopLimits()
    for name in ("max_arm_speed", "max_hand_speed", "reset_speed",
                 "reset_clearance", "guard_clearance"):
        parser.add_argument("--" + name.replace("_", "-"), type=float,
                            default=getattr(defaults, name))
    args = parser.parse_args(argv)
    if not 0.0 <= args.reset_shoulder_back <= 0.6:
        parser.error("--reset-shoulder-back must be between 0 and 0.6 rad")
    if not 0.0 < args.max_arm_speed <= 2.0 or not 0.0 < args.reset_speed <= 0.6:
        parser.error("arm speeds must be positive and at most 2.0 (teleop) / 0.6 (reset) rad/s")
    if not args.fake_robot:
        for name in ("ROBOT_NAME", "ZENOH_CONFIG"):
            if not os.environ.get(name):
                parser.error(f"{name} must be set for the robot (see the launcher)")
    limits = TeleopLimits(
        max_arm_speed=args.max_arm_speed, max_hand_speed=args.max_hand_speed,
        reset_speed=args.reset_speed, reset_clearance=args.reset_clearance,
        guard_clearance=args.guard_clearance, tracking_fault=args.tracking_fault or None)
    if args.fake_robot:
        connect_robot, hands_io = functools.partial(FakeArms, reset_pose(0.0)), FakeHands()
    else:
        connect_robot = DexcontrolArms
        hands_io = WujiHands({"left": args.left_hand_sn, "right": args.right_hand_sn})
    if args.synthetic_input:
        connect_glove = lambda side: (SyntheticSource(side), HeldWristImu())  # noqa: E731
        arm_input = SyntheticPicoLinks()
        retargeters = {side: _StaticRetargeter(np.full(20, 0.3)) for side in SIDES}
    else:
        from .retarget import WujiRetargeter

        connect_glove = functools.partial(_connect_glove, args)
        if args.arm_input == "tracker":
            from .xr_tracker_source import XRoboToolkitTrackerSource

            arm_input = TrackerLinks(
                XRoboToolkitTrackerSource(), ik_config=args.ik_config,
                left_serial=args.left_tracker_sn, right_serial=args.right_tracker_sn,
                human_height=args.human_height)
        else:
            arm_input = PicoLinks(args.pico_scale)
        retargeters = {side: WujiRetargeter(side) for side in SIDES}
    app = TeleopApp(connect_robot=connect_robot, hands_io=hands_io,
                    connect_glove=connect_glove, arm_input=arm_input, retargeters=retargeters,
                    reset_q=reset_pose(args.reset_shoulder_back), limits=limits,
                    log_csv=args.log_csv)
    print(f"Logging commanded and measured arm joints to {args.log_csv}", flush=True)
    app.start()
    try:
        Dashboard(app).run()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
