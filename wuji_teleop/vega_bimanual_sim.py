"""Bilateral Pico + Wuji glove teleoperation of Vega 1U in MuJoCo.

Both Vega arms and both Wuji Hand 2 models live in one simulation. Pico hand
tracking supplies relative end-effector positions, while each matching glove
supplies wrist orientation and finger landmarks. In tracker mode, a PICO Motion
Tracker on each wrist (XRoboToolkit object tracking) supplies the full 6D wrist
pose through GMR-style IK instead. No robot command transport is opened by this
module.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import queue
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .imu_arm_source import PalmImuArmSource, PalmImuRetargeter
from .retarget import TrackingController, WujiRetargeter
from .simulation import MODEL_DIR, MujocoHand, passive_viewer
from .sources import GloveSource, SourceSnapshot, SyntheticSource
from .vega_arm_sim import (
    ARM_JOINT_LIMITS,
    ARM_JOINT_NAMES,
    ARM_HOME_QPOS,
    HAND_IN_EE_ROTATION,
    PicoHandArmMapper,
    relative_wrist_rotation,
    RIGHT_ARM_BIMANUAL_HOME_QPOS,
    RIGHT_ARM_JOINT_LIMITS,
    SyntheticWristPoseSource,
    DampedLeastSquaresIK,
    WristOrientationIK,
    _joint_node_map,
    _named_ids,
    _remove_joint_limits,
    _resolve_hand_meshes,
    _set_joint_limits,
    build_articulated_mjcf,
)
from .vr_pose_source import PicoHandPoseSource


SIDES = ("left", "right")
SIDE_PREFIX = {"left": "L", "right": "R"}
HAND_PREFIX = {"left": "l", "right": "r"}
BIMANUAL_OUTPUT = (
    Path.home() / ".cache/dexmate-urdf/vega_1u/vega_1u_bimanual_wuji.xml"
)
BIMANUAL_CSV_HEADER = (
    "t,side,target_x,target_y,target_z,ee_x,ee_y,ee_z,"
    "pico_x,pico_y,pico_z,filtered_x,filtered_y,filtered_z,"
    "workspace_clipped,j1,j2,j3,j4,j5,j6,j7"
)
# target is the filtered GMR target the IK tracks; raw_target is before filtering.
TRACKER_CSV_HEADER = (
    "t,side,target_x,target_y,target_z,ee_x,ee_y,ee_z,"
    "tracker_x,tracker_y,tracker_z,raw_target_x,raw_target_y,raw_target_z,"
    "ik_rotation_error_deg,j1,j2,j3,j4,j5,j6,j7"
    "," + ",".join(
        [f"{frame}_r{i}{j}" for frame in ("raw_target", "target", "ee")
         for i in range(3) for j in range(3)]
        + [f"ik_j{i}" for i in range(1, 8)]
        + [f"actual_j{i}" for i in range(1, 8)]
        + [f"elbow_{axis}" for axis in "xyz"]
        + [f"tracker_r{i}{j}" for i in range(3) for j in range(3)])
)


def arm_joint_names(side):
    prefix = SIDE_PREFIX[side]
    return tuple(f"{prefix}_arm_j{i}" for i in range(1, 8))


def arm_actuator_names(side):
    return tuple(f"{name}_sim_position" for name in arm_joint_names(side))


def body_names(side):
    prefix = SIDE_PREFIX[side]
    return {
        "ee": f"{prefix}_ee",
        "shoulder": f"{prefix}_arm_l2",
        "elbow": f"{prefix}_arm_l4",
        "hand": f"wuji_{side}_wrist",
    }


def _attach_right_hand(root):
    """Attach the untouched right Hand 2 subtree to the existing right arm."""
    hand_xml = MODEL_DIR / "right.xml"
    if not hand_xml.is_file():
        raise FileNotFoundError(f"Pinned right Wuji Hand 2 model missing: {hand_xml}")
    hand_root = ET.parse(hand_xml).getroot()
    _resolve_hand_meshes(hand_root)

    asset = root.find("asset")
    hand_asset = hand_root.find("asset")
    if asset is None or hand_asset is None:
        raise ValueError("Vega or right Wuji MJCF is missing its asset section")
    existing = {item.get("name") for item in asset.findall("mesh")}
    for item in hand_asset:
        if item.tag == "mesh" and item.get("name") in existing:
            raise ValueError(f"Conflicting mesh asset name: {item.get('name')}")
        asset.append(item)

    hand_world = hand_root.find("worldbody")
    wrist = hand_world.find("body[@name='r_wrist']") if hand_world is not None else None
    if wrist is None:
        raise ValueError("Wuji right-hand model has no r_wrist body")
    wrist.set("name", body_names("right")["hand"])
    # Match the left articulated scene: Hand 2's local -Z finger direction
    # points away from the Vega end effector and the neutral palm faces down.
    wrist.set("quat", "0 1 0 0")
    joint_default = hand_root.find("./default/joint")
    if joint_default is None:
        raise ValueError("Wuji right-hand model has no joint dynamics defaults")
    for joint in wrist.iter("joint"):
        for key, value in joint_default.attrib.items():
            joint.attrib.setdefault(key, value)
    target = root.find(".//body[@name='R_ee']")
    if target is None:
        raise ValueError("Converted Vega MJCF has no R_ee body")
    target.append(wrist)

    contacts = root.find("contact")
    if contacts is None:
        contacts = ET.SubElement(root, "contact")
    hand_contacts = hand_root.find("contact")
    if hand_contacts is not None:
        for exclusion in hand_contacts:
            if exclusion.get("body1") == "r_wrist":
                exclusion.set("body1", body_names("right")["hand"])
            if exclusion.get("body2") == "r_wrist":
                exclusion.set("body2", body_names("right")["hand"])
            contacts.append(exclusion)

    actuators = root.find("actuator")
    hand_actuators = hand_root.find("actuator")
    if actuators is None or hand_actuators is None:
        raise ValueError("Bilateral model is missing an actuator section")
    for actuator in hand_actuators:
        actuators.append(actuator)


def build_bimanual_mjcf(output=BIMANUAL_OUTPUT):
    """Build both articulated arms with a matching Hand 2 on each wrist."""
    base_path = build_articulated_mjcf()
    root = ET.parse(base_path).getroot()
    base = root.find("./worldbody/body[@name='base_link']")
    actuators = root.find("actuator")
    if base is None or actuators is None:
        raise ValueError("Articulated Vega model is missing base or actuators")

    # Both arms use the official URDF limits; reapply them to the right arm here
    # so the bilateral scene stays tied to RIGHT_ARM_JOINT_LIMITS.
    joints = _joint_node_map(base)
    actuator_by_name = {node.get("name"): node for node in actuators}
    for index, (joint_name, actuator_name) in enumerate(
            zip(arm_joint_names("right"), arm_actuator_names("right"))):
        joint = joints.get(joint_name)
        actuator = actuator_by_name.get(actuator_name)
        if joint is None or actuator is None:
            raise ValueError(f"Bilateral model is missing {joint_name} or {actuator_name}")
        low, high = RIGHT_ARM_JOINT_LIMITS[index]
        if np.isfinite([low, high]).all():
            _set_joint_limits(joint, low, high)
            actuator.set("ctrlrange", f"{low:g} {high:g}")
            actuator.set("ctrllimited", "true")
        else:
            _remove_joint_limits(joint)
            actuator.attrib.pop("ctrlrange", None)
            actuator.set("ctrllimited", "false")

    _attach_right_hand(root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


class SideArm:
    """Named position control for one side of the bilateral Vega scene."""

    def __init__(self, model, data, side):
        if side not in SIDES:
            raise ValueError(f"Unsupported arm side: {side!r}")
        self.model, self.data, self.side = model, data, side
        self.joint_ids = _named_ids(
            model, mujoco.mjtObj.mjOBJ_JOINT, arm_joint_names(side))
        self.actuator_ids = _named_ids(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm_actuator_names(side))
        if not np.array_equal(model.actuator_trnid[self.actuator_ids, 0], self.joint_ids):
            raise ValueError(f"{side.capitalize()} arm actuator mapping is incorrect")
        limits = ARM_JOINT_LIMITS if side == "left" else RIGHT_ARM_JOINT_LIMITS
        limited = np.isfinite(limits).all(axis=1)
        if not np.array_equal(model.jnt_limited[self.joint_ids].astype(bool), limited):
            raise ValueError(f"{side.capitalize()} arm limits do not match software limits")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.lower = limits[:, 0].copy()
        self.upper = limits[:, 1].copy()
        self.home = (ARM_HOME_QPOS if side == "left"
                     else RIGHT_ARM_BIMANUAL_HOME_QPOS).copy()
        if np.any(self.home < self.lower) or np.any(self.home > self.upper):
            raise ValueError(f"{side.capitalize()} arm home is outside its limits")
        self.target = self.home.copy()

    def set_target(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Expected seven finite arm targets")
        self.target = np.clip(target, self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = self.target

    def initialize(self):
        self.data.qpos[self.qpos_ids] = self.target
        self.data.qvel[self.dof_ids] = 0.0
        self.set_target(self.target)


def _side_ik(model, data, arm, *, upper_only):
    names = body_names(arm.side)
    return DampedLeastSquaresIK(
        model, data, arm,
        max_step=0.15,
        iterations=40 if upper_only else 16,
        orientation_weight=0.0 if upper_only else 0.35,
        elbow_weight=0.20 if upper_only else 0.25,
        elbow_down_weight=0.06 if upper_only else 0.12,
        position_weight=4.0 if upper_only else 1.5,
        shoulder_weight=0.0,
        roll_weight=0.04 if upper_only else 0.08,
        active_joints=(0, 1, 2, 3) if upper_only else None,
        elbow_pole_direction=(
            0.0, 0.25 if arm.side == "left" else -0.25, -1.0),
        ee_body_name=names["ee"],
        shoulder_body_name=names["shoulder"],
        elbow_body_name=names["elbow"],
        shoulder_pitch_joint_name=arm_joint_names(arm.side)[0],
    )


def _limit_joint_goal(goal, seed, max_total_step=0.22):
    """Bound the complete per-update goal change after all IK passes."""
    goal = np.asarray(goal, dtype=np.float64).copy()
    seed = np.asarray(seed, dtype=np.float64)
    delta = goal - seed
    magnitude = float(np.linalg.norm(delta))
    if magnitude > max_total_step:
        goal = seed + delta * (max_total_step / magnitude)
    return goal


def load_bimanual_scene():
    path = build_bimanual_mjcf()
    model = mujoco.MjModel.from_xml_path(str(path))
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError(f"Expected 2 ms timestep, got {model.opt.timestep}")
    if (model.nq, model.nv, model.nu) != (54, 54, 54):
        raise ValueError(
            "Expected 14 arm plus 40 hand coordinates/actuators; got "
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    data = mujoco.MjData(model)
    arms = {side: SideArm(model, data, side) for side in SIDES}
    hands = {side: MujocoHand(model, data, side) for side in SIDES}
    for arm in arms.values():
        arm.initialize()
    for hand in hands.values():
        hand.initialize()
    mujoco.mj_forward(model, data)
    return model, data, arms, hands


@dataclass
class SideReport:
    glove_serial: str | None = None
    finger_status: str = "waiting"
    arm_status: str = "waiting"
    received_frames: int = 0
    finger_updates: int = 0
    arm_updates: int = 0
    max_frame_age_ms: float = 0.0
    arm_joint_peak_delta_rad: list[float] = field(default_factory=list)
    last_target: list[float] | None = None
    last_ee: list[float] | None = None
    last_ik_error_m: float | None = None
    ik_error_p95_m: float | None = None
    ik_error_max_m: float | None = None
    wrist_saturation_fraction: float = 0.0
    tracker_serial: str | None = None
    ik_rotation_error_p95_rad: float | None = None
    pico_outliers: int = 0
    workspace_clips: int = 0
    reacquisitions: int = 0
    max_raw_step_m: float = 0.0
    max_target_step_m: float = 0.0


@dataclass
class BimanualRunReport:
    mode: str
    elapsed_seconds: float = 0.0
    simulation_seconds: float = 0.0
    mujoco_warnings: int = 0
    sides: dict[str, SideReport] = field(
        default_factory=lambda: {side: SideReport() for side in SIDES})
    acceptance: dict = field(default_factory=dict)


def run(sources, *, mode="pico", pose_sources=None, imu_sources=None,
        tracker_source=None, tracker_serials=None, ik_config=None, human_height=None,
        duration=None, headless=False, report_path=None, pico_scale=0.6,
        log_csv=None):
    """Run two independent input/control paths in one MuJoCo world."""
    if set(sources) != set(SIDES):
        raise ValueError("Bimanual run requires left and right finger sources")
    if mode not in ("pico", "tracker", "synthetic"):
        raise ValueError("Bimanual mode must be 'pico', 'tracker', or 'synthetic'")
    if mode == "pico" and (set(pose_sources or {}) != set(SIDES) or
                           set(imu_sources or {}) != set(SIDES)):
        raise ValueError("Pico mode requires left and right Pico and palm-IMU sources")
    if mode == "tracker" and tracker_source is None:
        raise ValueError("Tracker mode requires an XRoboToolkit tracker source")

    model, data, arms, hands = load_bimanual_scene()
    tracker_mapper = None
    if mode == "tracker":
        # mink is only needed for tracker mode.
        from .gmr_arm_retarget import DEFAULT_IK_CONFIG, GmrArmRetargeter, TrackerArmMapper

        serials = tracker_serials or {}
        tracker_mapper = TrackerArmMapper(
            GmrArmRetargeter(model, {side: arm_joint_names(side) for side in SIDES},
                             ik_config or DEFAULT_IK_CONFIG,
                             actual_human_height=human_height),
            left_serial=serials.get("left"), right_serial=serials.get("right"))
        print("Tracker mode: the first tracker frame calibrates; press R to recalibrate "
              "while holding both arms like the robot's home pose.", flush=True)
    finger_controllers = {
        side: TrackingController(WujiRetargeter(side), hands[side]) for side in SIDES
    }
    names = {side: body_names(side) for side in SIDES}
    ee_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names[side]["ee"])
        for side in SIDES
    }
    ik = {side: _side_ik(model, data, arms[side], upper_only=(mode == "pico"))
          for side in SIDES}
    wrist_ik = {
        side: WristOrientationIK(
            model, data, arms[side], joint_indices=(4, 5, 6),
            max_total_step=0.20, joint_limit_margin=0.08,
            ee_body_name=names[side]["ee"])
        for side in SIDES
    }
    mappers = ({side: PicoHandArmMapper(pico_scale) for side in SIDES}
               if mode == "pico" else {})
    origins = {}
    wrist_references = {side: arms[side].home[4:].copy() for side in SIDES}
    synthetic_poses = {}
    for side in SIDES:
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[ee_ids[side]].reshape(3, 3)
        pose[:3, 3] = data.xpos[ee_ids[side]]
        origins[side] = pose
        if mode == "synthetic":
            # Opposing lateral phases keep the two paths visibly independent.
            amplitude = (0.035, 0.022 if side == "left" else -0.022, 0.025)
            synthetic_poses[side] = SyntheticWristPoseSource(
                pose, position_amplitude=amplitude)

    joint_slew = {side: arms[side].home.copy() for side in SIDES}
    max_joint_step = 1.5 / 40.0
    last_calibration_message = None

    def calibrate_trackers(snapshot, now):
        """Pin the wrist trackers to the mirrored home pose; report each new outcome.

        Pinning to home rather than the current pose keeps an arm that has
        drifted from carrying that bias into every later calibration.
        """
        nonlocal last_calibration_message
        try:
            yaw = tracker_mapper.calibrate(
                snapshot, {side: arms[side].home.copy() for side in SIDES}, now)
            message = (f"Calibrated trackers {tracker_mapper.serials}; operator heading "
                       f"{math.degrees(yaw):.0f} deg. Both arms return to home. To "
                       "recalibrate, hold forearms forward, palms down, and press R.")
            last_calibration_message = None
        except ValueError as exc:
            message = f"Tracker calibration waiting: {exc}"
            if message == last_calibration_message:
                return
            last_calibration_message = message
        print(message, flush=True)

    def command_arm(side, goal):
        arm = arms[side]
        joint_slew[side][:] = np.clip(
            joint_slew[side] + np.clip(goal - joint_slew[side],
                                      -max_joint_step, max_joint_step),
            arm.lower, arm.upper)
        arm.set_target(joint_slew[side])

    csv_file = None
    if log_csv:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = log_path.open("w", encoding="utf-8")
        csv_file.write((TRACKER_CSV_HEADER if mode == "tracker"
                        else BIMANUAL_CSV_HEADER) + "\n")
        print(f"Logging bilateral {'tracker' if mode == 'tracker' else 'Pico'} "
              f"targets to {log_path}", flush=True)

    report = BimanualRunReport(mode)
    for side in SIDES:
        report.sides[side].glove_serial = getattr(sources[side], "serial", None)
    snapshots = {side: SourceSnapshot(None, 0, 0, None) for side in SIDES}
    arm_status = {side: ("synthetic" if mode == "synthetic" else "waiting")
                  for side in SIDES}
    pico_home_hold = {side: False for side in SIDES}
    targets = {side: None for side in SIDES}
    errors = {side: None for side in SIDES}
    error_samples = {side: [] for side in SIDES}
    rotation_error_samples = {side: [] for side in SIDES}
    wrist_saturated = {side: 0 for side in SIDES}
    arm_start = {side: data.qpos[arms[side].qpos_ids].copy() for side in SIDES}
    peaks = {side: np.zeros(7) for side in SIDES}
    keys = queue.SimpleQueue()
    context = nullcontext(None) if headless else passive_viewer(model, data, keys.put)
    finger_period, arm_period, viewer_period = 1 / 120, 1 / 40, 1 / 30
    started = time.monotonic()
    next_physics = next_finger = next_arm = next_viewer = started
    next_status = started + 2
    recenter_requested = False
    status_line = "Waiting for bilateral tracking"
    try:
        with context as viewer:
            if viewer:
                with viewer.lock():
                    viewer.opt.geomgroup[2] = 0
                    viewer.opt.sitegroup[3] = 0
                    viewer.cam.lookat[:] = (data.xpos[ee_ids["left"]] +
                                            data.xpos[ee_ids["right"]]) / 2
                    viewer.cam.distance = 1.45
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -12
            started = time.monotonic()
            next_physics = next_finger = next_arm = next_viewer = started
            next_status = started + 2
            while duration is None or time.monotonic() - started < duration:
                if viewer and not viewer.is_running():
                    break
                while not keys.empty():
                    key = keys.get()
                    if key == 32:
                        for controller in finger_controllers.values():
                            controller.toggle_pause()
                    elif key in (ord("r"), ord("R")):
                        recenter_requested = True
                now = time.monotonic()
                with viewer.lock() if viewer else nullcontext():
                    if recenter_requested:
                        for side in SIDES:
                            arm = arms[side]
                            joint_slew[side][:] = arm.home
                            wrist_references[side][:] = arm.home[4:]
                            data.qpos[arm.qpos_ids] = arm.home
                            data.qvel[arm.dof_ids] = 0.0
                            arm.set_target(arm.home)
                        mujoco.mj_forward(model, data)
                        for side in SIDES:
                            current = np.eye(4)
                            current[:3, :3] = data.xmat[ee_ids[side]].reshape(3, 3)
                            current[:3, 3] = data.xpos[ee_ids[side]]
                            origins[side] = current
                            if mode == "pico":
                                mappers[side].recenter()
                                imu_sources[side].recenter(current)
                                pico_home_hold[side] = True
                                arm_status[side] = "calibrating"
                        if mode == "tracker":
                            calibrate_trackers(tracker_source.read(), now)
                        recenter_requested = False
                    if now >= next_finger:
                        for side in SIDES:
                            snapshot = sources[side].read()
                            snapshots[side] = snapshot
                            finger_controllers[side].update(snapshot, now)
                            if snapshot.frame is not None:
                                age = max(0.0, now - snapshot.frame.received_at) * 1000
                                report.sides[side].max_frame_age_ms = max(
                                    report.sides[side].max_frame_age_ms, age)
                        next_finger += max(
                            1, int((now - next_finger) / finger_period) + 1) * finger_period
                    if now >= next_arm:
                        tracker_goals = {}
                        if mode == "tracker":
                            tracker_snapshot = tracker_source.read()
                            if tracker_mapper.serials is None:
                                calibrate_trackers(tracker_snapshot, now)
                            tracker_goals = tracker_mapper.update(tracker_snapshot, now)
                        for side in SIDES:
                            arm = arms[side]
                            ee_id = ee_ids[side]
                            if mode == "tracker":
                                arm_status[side] = f"tracker {tracker_mapper.status[side]}"
                                goal = tracker_goals.get(side)
                                if goal is None:
                                    continue
                                ik_goal = goal.copy()
                                goal = np.clip(_limit_joint_goal(goal, arm.target),
                                               arm.lower, arm.upper)
                                command_arm(side, goal)
                                target = tracker_mapper.targets[side]
                                targets[side] = target[:3, 3].copy()
                                errors[side], rotation_error = tracker_mapper.errors[side]
                                error_samples[side].append(errors[side])
                                rotation_error_samples[side].append(rotation_error)
                                margin = 0.05
                                if np.any(goal[4:] <= arm.lower[4:] + margin) or np.any(
                                        goal[4:] >= arm.upper[4:] - margin):
                                    wrist_saturated[side] += 1
                                report.sides[side].arm_updates += 1
                                tracker = tracker_mapper.tracker_positions[side]
                                if csv_file is not None and tracker is not None:
                                    values = (now - started, side, *target[:3, 3],
                                              *data.xpos[ee_id], *tracker,
                                              *tracker_mapper.raw_targets[side][:3, 3],
                                              math.degrees(rotation_error), *goal,
                                              *tracker_mapper.raw_targets[side][:3, :3].ravel(),
                                              *target[:3, :3].ravel(),
                                              *data.xmat[ee_id].ravel(),
                                              *ik_goal, *data.qpos[arm.qpos_ids],
                                              *data.xpos[mujoco.mj_name2id(
                                                  model, mujoco.mjtObj.mjOBJ_BODY,
                                                  body_names(side)["elbow"])],
                                              *tracker_mapper.tracker_rotations[side].ravel())
                                    csv_file.write(
                                        ",".join(str(v) if isinstance(v, str)
                                                 else f"{v:.6f}" for v in values) + "\n")
                                continue
                            if mode == "synthetic":
                                goal = ik[side].solve(
                                    synthetic_poses[side].read(now - started), seed=arm.target)
                                command_arm(side, goal)
                                arm_status[side] = "synthetic"
                                report.sides[side].arm_updates += 1
                                continue

                            scratch = ik[side].scratch
                            # Always map against the commanded arm pose. The IK
                            # scratch state otherwise contains its last trial pose.
                            scratch.qpos[:] = data.qpos
                            scratch.qpos[arm.qpos_ids] = arm.target
                            mujoco.mj_kinematics(model, scratch)
                            target = mappers[side].update(
                                pose_sources[side].read(), scratch.xpos[ee_id], now,
                                scratch.xpos[mujoco.mj_name2id(
                                    model, mujoco.mjtObj.mjOBJ_BODY,
                                    names[side]["shoulder"])])
                            targets[side] = target
                            arm_status[side] = f"hand {mappers[side].status}"
                            imu_snapshot = imu_sources[side].read(origins[side], now=now)
                            wrist_rotation = (None if imu_snapshot.pose is None
                                              else imu_snapshot.pose[:3, :3])
                            if pico_home_hold[side]:
                                position_moved = (
                                    mappers[side].filtered_hand_position is not None and
                                    mappers[side].hand_origin is not None and
                                    np.linalg.norm(
                                        mappers[side].filtered_hand_position -
                                        mappers[side].hand_origin) >
                                    mappers[side].deadband_m)
                                wrist_moved = False
                                if wrist_rotation is not None:
                                    relative = wrist_rotation @ origins[side][:3, :3].T
                                    angle = math.acos(float(np.clip(
                                        (np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
                                    wrist_moved = angle > math.radians(2.0)
                                if position_moved or wrist_moved:
                                    pico_home_hold[side] = False
                            if target is not None:
                                target_pose = np.eye(4)
                                target_pose[:3, :3] = data.xmat[ee_id].reshape(3, 3)
                                target_pose[:3, 3] = target
                                goal = ik[side].solve(target_pose, seed=arm.target)
                                if wrist_rotation is not None:
                                    neutral_goal = goal.copy()
                                    neutral_goal[4:] = wrist_references[side]
                                    scratch.qpos[:] = data.qpos
                                    scratch.qpos[arm.qpos_ids] = neutral_goal
                                    mujoco.mj_kinematics(model, scratch)
                                    wrist_target = relative_wrist_rotation(
                                        wrist_rotation, origins[side][:3, :3],
                                        scratch.xmat[ee_id].reshape(3, 3))
                                    goal = wrist_ik[side].solve(
                                        wrist_target, seed=goal)
                                    # Wrist orientation can move the palm in
                                    # translation; correct position once more
                                    # while preserving the solved wrist joints.
                                    goal = ik[side].solve(target_pose, seed=goal)
                                if pico_home_hold[side]:
                                    goal = arm.home.copy()
                                goal = _limit_joint_goal(goal, arm.target)
                                goal = np.clip(goal, arm.lower, arm.upper)
                                command_arm(side, goal)
                                scratch.qpos[:] = data.qpos
                                scratch.qpos[arm.qpos_ids] = goal
                                mujoco.mj_kinematics(model, scratch)
                                errors[side] = float(np.linalg.norm(
                                    target - scratch.xpos[ee_id]))
                                error_samples[side].append(errors[side])
                                margin = 0.05
                                if np.any(goal[4:] <= arm.lower[4:] + margin) or np.any(
                                        goal[4:] >= arm.upper[4:] - margin):
                                    wrist_saturated[side] += 1
                                report.sides[side].arm_updates += 1
                                if csv_file is not None:
                                    values = (now - started, side, *target,
                                              *data.xpos[ee_id],
                                              *mappers[side].hand_position,
                                              *mappers[side].filtered_hand_position,
                                              int(mappers[side].workspace_clipped), *goal)
                                    csv_file.write(
                                        ",".join(str(v) if isinstance(v, str)
                                                 else f"{v:.6f}" for v in values) + "\n")
                        next_arm += max(
                            1, int((now - next_arm) / arm_period) + 1) * arm_period
                    now = time.monotonic()
                    steps = min(25, max(
                        0, int((now - next_physics) / model.opt.timestep) + 1))
                    if steps:
                        mujoco.mj_step(model, data, nstep=steps)
                        for side in SIDES:
                            peaks[side] = np.maximum(
                                peaks[side],
                                np.abs(data.qpos[arms[side].qpos_ids] - arm_start[side]))
                        next_physics += steps * model.opt.timestep
                        if now - next_physics > 0.050:
                            next_physics = now
                now = time.monotonic()
                if now >= next_status:
                    warnings = int(np.sum(data.warning.number))
                    if warnings:
                        raise RuntimeError(f"MuJoCo raised {warnings} numerical warnings")
                    parts = []
                    for side in SIDES:
                        controller = finger_controllers[side]
                        part = (f"{side[0].upper()}: fingers {controller.status}, "
                                f"arm {arm_status[side]}")
                        if errors[side] is not None:
                            part += f", IK {errors[side] * 100:.1f} cm"
                            if rotation_error_samples[side]:
                                part += (f" "
                                         f"{math.degrees(rotation_error_samples[side][-1]):.1f} deg")
                        parts.append(part)
                    status_line = " | ".join(parts)
                    print(status_line, flush=True)
                    if csv_file is not None:
                        csv_file.flush()
                    next_status = now + 2
                if viewer and now >= next_viewer:
                    viewer.set_texts((None, None,
                                      "Vega bilateral + Wuji Hand 2 — R: home + recalibrate",
                                      status_line))
                    viewer.sync()
                    next_viewer = now + viewer_period
                sleep_for = min(next_physics, next_finger, next_arm) - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        if csv_file is not None:
            csv_file.close()
        report.elapsed_seconds = time.monotonic() - started
        report.simulation_seconds = float(data.time)
        report.mujoco_warnings = int(np.sum(data.warning.number))
        for side in SIDES:
            side_report = report.sides[side]
            snapshot = snapshots[side]
            controller = finger_controllers[side]
            side_report.finger_status = controller.status
            side_report.arm_status = arm_status[side]
            side_report.received_frames = snapshot.received
            side_report.finger_updates = controller.updates
            side_report.arm_joint_peak_delta_rad = peaks[side].tolist()
            side_report.last_target = (None if targets[side] is None
                                       else targets[side].tolist())
            side_report.last_ee = data.xpos[ee_ids[side]].tolist()
            side_report.last_ik_error_m = errors[side]
            samples = np.asarray(error_samples[side], dtype=np.float64)
            if len(samples):
                side_report.ik_error_p95_m = float(np.percentile(samples, 95))
                side_report.ik_error_max_m = float(np.max(samples))
                side_report.wrist_saturation_fraction = (
                    wrist_saturated[side] / len(samples))
            if rotation_error_samples[side]:
                side_report.ik_rotation_error_p95_rad = float(
                    np.percentile(rotation_error_samples[side], 95))
            if mode == "tracker":
                side_report.tracker_serial = (tracker_mapper.serials or {}).get(side)
                side_report.max_target_step_m = tracker_mapper.filters[side].max_step_m
            if mode == "pico":
                mapper = mappers[side]
                side_report.pico_outliers = mapper.outlier_count
                side_report.workspace_clips = mapper.workspace_clip_count
                side_report.reacquisitions = mapper.reacquisition_count
                side_report.max_raw_step_m = mapper.max_raw_step_m
                side_report.max_target_step_m = mapper.max_target_step_m
        if mode in ("pico", "tracker"):
            thresholds = {
                "ik_error_p95_m": 0.02,
                "wrist_saturation_fraction": 0.0,
                "max_target_step_m": 0.03,
            }
            side_gates = {}
            for side in SIDES:
                side_report = report.sides[side]
                side_gates[side] = {
                    "ik_p95_under_2cm": (
                        None if side_report.ik_error_p95_m is None
                        else side_report.ik_error_p95_m < thresholds["ik_error_p95_m"]),
                    "no_wrist_saturation": (
                        None if side_report.arm_updates == 0
                        else side_report.wrist_saturation_fraction ==
                        thresholds["wrist_saturation_fraction"]),
                    "no_large_conditioned_target_step": (
                        None if side_report.arm_updates == 0
                        else side_report.max_target_step_m <=
                        thresholds["max_target_step_m"]),
                }
            gate_values = [value for gates in side_gates.values()
                           for value in gates.values()]
            report.acceptance = {
                "thresholds": thresholds,
                "sides": side_gates,
                "quantitative_pass": bool(gate_values) and all(
                    value is True for value in gate_values),
                "manual_gates_pending": [
                    "10 seconds stationary with no visible jitter",
                    "no visible jump after hand loss and recovery",
                    "correct forward/up/outward response on both arms",
                    "arm holds while either tracked hand is absent",
                ],
            }
        serialized = json.dumps(asdict(report), indent=2)
        print("Run report:\n" + serialized, flush=True)
        if report_path:
            Path(report_path).write_text(serialized + "\n")
    return report


LEFT_GLOVE_ADDRESS = "192.168.1.100:50001"
RIGHT_GLOVE_ADDRESS = "192.168.1.101:50001"


def _connect_bilateral_gloves(left_serial=None, right_serial=None, user_id="",
                              left_address=LEFT_GLOVE_ADDRESS,
                              right_address=RIGHT_GLOVE_ADDRESS, sides=SIDES):
    import wuji_sdk

    wuji_sdk.set_log_level("warn")
    connected = {}
    opened = []
    try:
        endpoints = (
            ("left", left_serial, left_address),
            ("right", right_serial, right_address),
        )
        for side, serial, address in endpoints:
            if side not in sides:
                continue
            source = GloveSource(
                serial, user_id, sdk=wuji_sdk,
                device_name=f"wuji_bimanual_glove_{side}", address=address)
            opened.append(source)
            if source.side != side:
                raise RuntimeError(
                    f"Glove at {address} reports as {source.side}, expected {side}")
            connected[side] = source
        return connected
    except BaseException:
        for source in reversed(opened):
            source.close()
        raise


def _positive_number(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("must be a finite number greater than zero")
    return number


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true",
                        help="Run bilateral synthetic arms/fingers without hardware")
    parser.add_argument("--arm-input", choices=("pico-hand", "tracker"), default="pico-hand",
                        help="pico-hand: DexStream hand tracking plus glove IMU wrists; "
                             "tracker: XRoboToolkit wrist trackers through GMR-style IK")
    parser.add_argument("--finger-input",
                        choices=("glove", "left-glove", "right-glove", "synthetic"),
                        default="glove",
                        help="left-glove/right-glove: one live glove, the other hand "
                             "synthetic (tracker mode)")
    parser.add_argument("--left-tracker-sn")
    parser.add_argument("--right-tracker-sn")
    parser.add_argument("--ik-config", help="GMR ik_configs JSON for tracker mode")
    parser.add_argument("--human-height", type=_positive_number,
                        help="Operator height in metres for GMR scaling (tracker mode)")
    parser.add_argument("--left-glove-sn")
    parser.add_argument("--right-glove-sn")
    parser.add_argument("--left-glove-address", default=LEFT_GLOVE_ADDRESS)
    parser.add_argument("--right-glove-address", default=RIGHT_GLOVE_ADDRESS)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=_positive_number)
    parser.add_argument("--report")
    parser.add_argument("--pico-scale", type=_positive_number, default=0.6)
    parser.add_argument("--log-csv")
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args(argv)
    if args.headless and args.duration is None and not args.convert_only:
        parser.error("--headless requires --duration")
    tracker = args.arm_input == "tracker"
    if args.synthetic and (args.left_glove_sn or args.right_glove_sn or args.user_id):
        parser.error("glove selection applies only to live input")
    if args.synthetic and (tracker or args.finger_input != "glove"):
        parser.error("--synthetic already replaces the arm and finger inputs")
    if args.finger_input != "glove" and not tracker:
        parser.error("pico-hand arm input needs both gloves for wrist IMUs")
    if not tracker and (args.left_tracker_sn or args.right_tracker_sn or
                        args.ik_config or args.human_height):
        parser.error("tracker serials, --ik-config and --human-height need --arm-input tracker")
    if not args.synthetic and not args.log_csv:
        args.log_csv = (f"/tmp/vega-bimanual-{'tracker' if tracker else 'pico'}-"
                        f"{time.strftime('%Y%m%d-%H%M%S')}.csv")

    sources = pose_sources = imu_sources = tracker_source = None
    try:
        if args.convert_only:
            model, _, _, _ = load_bimanual_scene()
            print(f"Loaded bilateral Vega + Wuji Hands: nq={model.nq} "
                  f"nv={model.nv} nu={model.nu}")
            return 0
        if args.synthetic:
            sources = {side: SyntheticSource(side) for side in SIDES}
            mode = "synthetic"
        else:
            if args.finger_input == "synthetic":
                sources = {side: SyntheticSource(side) for side in SIDES}
            else:
                glove_sides = (SIDES if args.finger_input == "glove"
                               else (args.finger_input.removesuffix("-glove"),))
                sources = _connect_bilateral_gloves(
                    args.left_glove_sn, args.right_glove_sn, args.user_id,
                    args.left_glove_address, args.right_glove_address,
                    sides=glove_sides)
                for side in glove_sides:
                    source = sources[side]
                    print(f"{side.capitalize()} glove {source.serial}: "
                          f"firmware={source.firmware}, address={source.address}",
                          flush=True)
                for side in SIDES:
                    sources.setdefault(side, SyntheticSource(side))
            if tracker:
                from .xr_tracker_source import XRoboToolkitTrackerSource

                tracker_source = XRoboToolkitTrackerSource()
                mode = "tracker"
            else:
                pose_sources = {side: PicoHandPoseSource(side=side) for side in SIDES}
                imu_sources = {}
                for side in SIDES:
                    source = sources[side]
                    imu_sources[side] = PalmImuArmSource(
                        source.palm_imu_resource(),
                        estimator=PalmImuRetargeter(
                            imu_in_wrist_rotation=source.palm_imu_mount_rotation(),
                            hand_in_ee_rotation=HAND_IN_EE_ROTATION,
                            motion_mode="orientation"))
                mode = "pico"
        run(sources, mode=mode, pose_sources=pose_sources, imu_sources=imu_sources,
            tracker_source=tracker_source,
            tracker_serials={"left": args.left_tracker_sn, "right": args.right_tracker_sn},
            ik_config=args.ik_config, human_height=args.human_height,
            duration=args.duration, headless=args.headless,
            report_path=args.report, pico_scale=args.pico_scale,
            log_csv=args.log_csv)
        return 0
    except KeyboardInterrupt:
        print("Stopped; all input connections closed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        closeables = [tracker_source] if tracker_source is not None else []
        for mapping in (imu_sources, pose_sources, sources):
            if mapping:
                closeables.extend(mapping.values())
        for closeable in closeables:
            try:
                closeable.close()
            except Exception as exc:
                print(f"Warning: input cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
