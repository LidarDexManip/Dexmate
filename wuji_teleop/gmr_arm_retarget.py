"""GMR-style two-stage IK from two wrist trackers to both Vega arms.

Follows General Motion Retargeting (Ze et al., 2025, arXiv 2510.02252, Section IV)
and its reference code (``general_motion_retargeting/motion_retarget.py``):

1. Key body matching: an ``ik_configs/*.json`` file in GMR's format maps each robot
   body to a human body with a position cost, an orientation cost, a local position
   offset and a rotation offset.
2. Rest-pose alignment: pressing R while the operator holds the robot's home
   pose sets each wrist's rotation offset ``R_off = R_h(t0)^T R_robot(t0)``, applied
   on the right as in GMR, and a position offset pinning that pose to the robot hand.
3. Non-uniform scaling about the human root: ``s_b = human_scale_table[b] * h / h_ref``.
   The robot base is fixed, so GMR's root translation term is dropped.
4./5. Two mink differential-IK stages, ``ik_match_table1`` then ``ik_match_table2``.
   Each repeats while the task error falls by more than 1e-3, at most 10 extra
   solves, warm-started from the previous frame.

Only the arm joints move in the IK; every other joint is frozen. A light posture
task toward the calibration pose settles each 7-DoF arm's spare elbow swivel.
Every step is a QP with hard bounds at the model's joint ranges, which are the
official Vega URDF limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .xr_tracker_source import (
    SIDES,
    TRACKER_STALE_SECONDS,
    assign_tracker_sides,
    heading_rotation,
    operator_heading,
)


DEFAULT_IK_CONFIG = Path(__file__).with_name("ik_configs") / "xr_trackers_to_vega_1u.json"
WRIST_BODIES = {"left": "Left_Wrist", "right": "Right_Wrist"}
HEADSET_BODY = "Headset"


@dataclass(frozen=True)
class KeyBody:
    robot_body: str
    human_body: str
    position_cost: float
    orientation_cost: float
    position_offset: np.ndarray  # metres, in the robot body frame
    rotation_offset: np.ndarray  # 3x3, applied on the right


def _parse_table(table, name):
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{name} must map robot bodies to GMR entries")
    entries = []
    for robot_body, entry in table.items():
        if not isinstance(entry, list) or len(entry) != 5:
            raise ValueError(f"{name}.{robot_body}: expected [human, pos_cost, rot_cost, "
                             "[dx, dy, dz], [w, x, y, z]]")
        human_body, position_cost, orientation_cost, position_offset, quat = entry
        position_offset = np.asarray(position_offset, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        costs = np.array([position_cost, orientation_cost], dtype=np.float64)
        if (not isinstance(human_body, str) or position_offset.shape != (3,) or
                quat.shape != (4,) or not np.isfinite([*costs, *position_offset, *quat]).all()
                or np.any(costs < 0) or not math.isclose(np.linalg.norm(quat), 1.0, abs_tol=1e-3)):
            raise ValueError(f"{name}.{robot_body}: invalid GMR entry {entry}")
        if not np.any(costs):
            continue  # GMR builds no task for an entry with both costs zero
        entries.append(KeyBody(robot_body, human_body, float(position_cost),
                               float(orientation_cost), position_offset,
                               Rotation.from_quat(quat, scalar_first=True).as_matrix()))
    return entries


def load_ik_config(path=DEFAULT_IK_CONFIG):
    """Load and validate a GMR ``ik_configs`` JSON file."""
    config = json.loads(Path(path).read_text())
    height = float(config["human_height_assumption"])
    if not math.isfinite(height) or height <= 0:
        raise ValueError("human_height_assumption must be a positive height in metres")
    scales = config["human_scale_table"]
    if (not isinstance(scales, dict) or
            not all(math.isfinite(float(v)) and float(v) > 0 for v in scales.values())):
        raise ValueError("human_scale_table must map human bodies to positive scales")
    root = config.get("human_root_name")
    if root not in (None, HEADSET_BODY):
        raise ValueError(f"human_root_name must be null or {HEADSET_BODY!r}")
    tables = []
    for index in (1, 2):
        if config.get(f"use_ik_match_table{index}", False):
            tables.append(_parse_table(config[f"ik_match_table{index}"],
                                       f"ik_match_table{index}"))
    if not tables:
        raise ValueError("Enable at least one of use_ik_match_table1/2")
    for table in tables:
        for key_body in table:
            if key_body.human_body not in scales:
                raise ValueError(f"{key_body.human_body} is matched but has no scale")
    return {
        "human_root_name": root,
        "human_height_assumption": height,
        "human_scale_table": {body: float(scale) for body, scale in scales.items()},
        "tables": tables,
        "posture_cost": config.get("posture_cost", 1e-2),
        "decouple_wrist": bool(config.get("decouple_wrist", False)),
        "solver": config.get("solver", "daqp"),
        "damping": float(config.get("damping", 0.5)),
        "elbow_preference": config.get("elbow_preference", {}),
        "tracker_mount_height_m": config.get("tracker_mount_height_m", {}),
    }


class GmrArmRetargeter:
    """Two-stage mink IK for both arms of one MuJoCo model.

    ``arm_joint_names`` maps ``"left"``/``"right"`` to that arm's joint names.
    """

    def __init__(self, model, arm_joint_names, config=DEFAULT_IK_CONFIG, *,
                 actual_human_height=None, max_iter=10, tolerance=1e-3):
        config = config if isinstance(config, dict) else load_ik_config(config)
        self.model = model
        ratio = 1.0
        if actual_human_height is not None:
            actual_human_height = float(actual_human_height)
            if not math.isfinite(actual_human_height) or actual_human_height <= 0:
                raise ValueError("actual_human_height must be a positive height in metres")
            ratio = actual_human_height / config["human_height_assumption"]
        self.scales = {body: scale * ratio for body, scale in config["human_scale_table"].items()}
        self.root_body = config["human_root_name"]
        # Height above the wrist centre with forearms level and palms down.
        # Calibration converts world-down into each tracker's local axes.
        self.tracker_mount_heights = config.get("tracker_mount_height_m", {})
        if (not isinstance(self.tracker_mount_heights, dict) or
                any(side not in SIDES or not math.isfinite(float(height)) or
                    not 0 <= float(height) <= 0.20
                    for side, height in self.tracker_mount_heights.items())):
            raise ValueError("tracker_mount_height_m must map left/right to heights in [0, 0.20] metres")
        self.tracker_center_offsets = {}
        self.tables = config["tables"]
        # Targets come from the first enabled table's offsets for both stages, as in GMR.
        self.key_bodies = {key_body.robot_body: key_body for key_body in self.tables[0]}
        human_side = {human: side for side, human in WRIST_BODIES.items()}
        self.side_of = {}
        for table in self.tables:
            for key_body in table:
                if key_body.human_body not in human_side:
                    raise ValueError(f"Only wrist trackers are tracked, not {key_body.human_body}")
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, key_body.robot_body) < 0:
                    raise ValueError(f"Robot body {key_body.robot_body} is not in the model")
                if key_body.robot_body not in self.key_bodies:
                    raise ValueError(f"{key_body.robot_body} must appear in the first table")
                self.side_of[key_body.robot_body] = human_side[key_body.human_body]
        self.tasks = [[mink.FrameTask(key_body.robot_body, "body",
                                      position_cost=key_body.position_cost,
                                      orientation_cost=key_body.orientation_cost,
                                      lm_damping=1.0)
                       for key_body in table] for table in self.tables]
        self.qpos_ids, self.dof_ids = {}, {}
        for side, names in arm_joint_names.items():
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
            if min(ids) < 0:
                raise ValueError(f"{side.capitalize()} arm joints are missing from the model")
            self.qpos_ids[side] = model.jnt_qposadr[ids]
            self.dof_ids[side] = model.jnt_dofadr[ids]
        posture_cost = np.zeros(model.nv)
        for dofs in self.dof_ids.values():
            posture_cost[dofs] = config["posture_cost"]
        self.posture = mink.PostureTask(model, cost=posture_cost)
        self.decouple_wrist = config["decouple_wrist"]
        self.position_tasks = {body: mink.FrameTask(
            body, "body", position_cost=key.position_cost, orientation_cost=0.0,
            lm_damping=1.0) for body, key in self.key_bodies.items()}
        self.rotation_tasks = {body: mink.FrameTask(
            body, "body", position_cost=0.0, orientation_cost=key.orientation_cost,
            lm_damping=1.0) for body, key in self.key_bodies.items()}
        self.calibration_wrists = None
        self.limits = [mink.ConfigurationLimit(model)]
        self.solver = config["solver"]
        self.damping = config["damping"]
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.configuration = mink.Configuration(model)
        self.elbow_specs = config.get("elbow_preference", {})
        self.elbow_tasks = {}
        self.elbow_rest = {}
        if self.elbow_specs and self.decouple_wrist:
            raise ValueError("elbow_preference requires coupled IK")
        for side, spec in self.elbow_specs.items():
            if side not in self.qpos_ids:
                raise ValueError(f"Unknown elbow side: {side}")
            cost = float(spec["cost"])
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("Elbow cost must be positive and finite")
            offset = float(spec.get("inward_offset_m", 0.0))
            blend_distance = float(spec.get("inward_blend_distance_m", 0.10))
            if not math.isfinite(offset) or offset < 0:
                raise ValueError("inward_offset_m must be finite and nonnegative")
            if not math.isfinite(blend_distance) or blend_distance <= 0:
                raise ValueError("inward_blend_distance_m must be positive and finite")
            for name in (spec["body"], spec["shoulder"]):
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                    raise ValueError(f"Elbow preference body {name} is missing")
            self.elbow_tasks[side] = mink.FrameTask(
                spec["body"], "body", position_cost=cost,
                orientation_cost=0.0, lm_damping=1.0)
        # Joints other than the arms never reach the wrist frames. Park them inside
        # their limits so the frozen DoFs cannot make the joint-limit QP infeasible.
        self.neutral_qpos = model.qpos0.copy()
        for joint in range(model.njnt):
            if model.jnt_limited[joint] and model.jnt_type[joint] in (
                    mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                address = model.jnt_qposadr[joint]
                self.neutral_qpos[address] = np.clip(
                    self.neutral_qpos[address], *model.jnt_range[joint])
        self.heading = np.eye(3)
        self.offsets = None
        # Per-solve diagnostics are intentionally kept in memory so replay tools
        # can inspect the solver without changing the returned joint goals.
        self.last_diagnostics = {"stages": [], "solve_time_s": 0.0}

    @property
    def calibrated(self):
        return self.offsets is not None

    def sync(self, arm_targets):
        """Restart the IK from the given ``{side: 7 joints}`` arm pose."""
        q = self.neutral_qpos.copy()
        for side, target in arm_targets.items():
            q[self.qpos_ids[side]] = target
        self.configuration.update(q)

    def _human_frames(self, human, root):
        """Human poses in the operator's heading frame, relative to the root."""
        origin = np.zeros(3)
        if self.root_body == HEADSET_BODY:
            if root is None:
                raise ValueError("The IK config measures wrists from the headset, "
                                 "but no headset pose is streaming")
            origin = np.asarray(root, dtype=np.float64)[:3, 3]
        frames = {}
        for body, pose in human.items():
            pose = np.asarray(pose, dtype=np.float64)
            frame = np.eye(4)
            frame[:3, :3] = self.heading.T @ pose[:3, :3]
            center = pose[:3, 3]
            if body in self.tracker_center_offsets:
                center = center + pose[:3, :3] @ self.tracker_center_offsets[body]
            frame[:3, 3] = self.heading.T @ (center - origin)
            frames[body] = frame
        return frames

    def calibrate(self, human, arm_targets, *, root=None, yaw=0.0):
        """GMR step 2: pin the operator's current pose to the robot's arm pose.

        ``human`` maps GMR human bodies to robot-axis 4x4 poses; ``yaw`` is the
        operator's heading in that world.
        """
        missing = sorted({key_body.human_body for key_body in self.key_bodies.values()}
                         - set(human))
        if missing:
            raise ValueError(f"Cannot calibrate without {', '.join(missing)}")
        self.sync(arm_targets)
        self.calibration_wrists = {
            side: np.asarray(q)[4:].copy() for side, q in arm_targets.items()}
        self.posture.set_target_from_configuration(self.configuration)
        self.heading = heading_rotation(yaw)
        self.tracker_center_offsets = {
            WRIST_BODIES[side]: np.asarray(human[WRIST_BODIES[side]])[:3, :3].T @
            np.array([0.0, 0.0, -float(height)])
            for side, height in self.tracker_mount_heights.items() if float(height) > 0
        }
        frames = self._human_frames(human, root)
        offsets = {}
        for robot_body, key_body in self.key_bodies.items():
            robot = self.configuration.get_transform_frame_to_world(robot_body, "body").as_matrix()
            frame = frames[key_body.human_body]
            rotation = frame[:3, :3] @ key_body.rotation_offset
            rotation_offset = rotation.T @ robot[:3, :3]
            scaled = (self.scales[key_body.human_body] * frame[:3, 3] +
                      robot[:3, :3] @ key_body.position_offset)
            offsets[robot_body] = (rotation_offset, robot[:3, 3] - scaled)
        self.offsets = offsets
        self.elbow_rest = {}
        for side, spec in self.elbow_specs.items():
            elbow = self.configuration.get_transform_frame_to_world(
                spec["body"], "body").translation().copy()
            shoulder = self.configuration.get_transform_frame_to_world(
                spec["shoulder"], "body").translation().copy()
            body = next(b for b, s in self.side_of.items() if s == side)
            wrist = self.configuration.get_transform_frame_to_world(
                body, "body").translation().copy()
            self.elbow_rest[side] = (elbow, shoulder, wrist)

    def _elbow_preferences(self, targets):
        """Soft calibrated elbow prior, independent of wrist orientation.

        Follow half the wrist translation to allow reaching. Fade near the
        approximate shoulder-elbow-palm reach boundary. This is a weighted
        competing task, not a strict null-space constraint or collision guard.
        """
        active = []
        for body, target in targets.items():
            side = self.side_of[body]
            if side not in self.elbow_tasks:
                continue
            elbow, shoulder, wrist = self.elbow_rest[side]
            reach = np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow)
            distance = np.linalg.norm(target[:3, 3] - shoulder)
            fade = float(np.clip((reach - distance) / max(0.2 * reach, 1e-6), 0, 1))
            task = self.elbow_tasks[side]
            task.set_position_cost(float(self.elbow_specs[side]["cost"]) * fade)
            pose = np.eye(4)
            pose[:3, 3] = elbow + 0.5 * (target[:3, 3] - wrist)
            # Robot x is forward. Retraction from the calibrated palm position
            # activates a smooth, bounded shift toward the torso centreline.
            # Orientation alone and forward reaches never activate this offset.
            spec = self.elbow_specs[side]
            fraction = float(np.clip((wrist[0] - target[0, 3]) /
                                    float(spec.get("inward_blend_distance_m", 0.10)), 0, 1))
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            offset = float(spec.get("inward_offset_m", 0.0)) * blend
            # Clamp at y=0 so the offset itself cannot cross the centreline.
            pose[1, 3] -= np.sign(pose[1, 3]) * min(offset, abs(pose[1, 3]))
            task.set_target(mink.SE3.from_matrix(pose))
            active.append(task)
        return active

    def targets(self, human, root=None):
        """GMR steps 2-3: robot-frame 4x4 targets for each matched body in ``human``."""
        if not self.calibrated:
            raise RuntimeError("Calibrate before retargeting")
        frames = self._human_frames(human, root)
        targets = {}
        for robot_body, key_body in self.key_bodies.items():
            frame = frames.get(key_body.human_body)
            if frame is None:
                continue
            rotation_offset, position_offset = self.offsets[robot_body]
            target = np.eye(4)
            target[:3, :3] = frame[:3, :3] @ key_body.rotation_offset @ rotation_offset
            target[:3, 3] = (self.scales[key_body.human_body] * frame[:3, 3] +
                             target[:3, :3] @ key_body.position_offset + position_offset)
            targets[robot_body] = target
        return targets

    def _error(self, tasks):
        return float(np.linalg.norm(np.concatenate(
            [task.compute_error(self.configuration) for task in tasks])))

    def _solve_stage(self, tasks, constraints, preferences=()):
        dt = self.model.opt.timestep
        error = self._error(tasks)
        initial_error = error
        iterations = 0
        for _ in range(self.max_iter + 1):
            velocity = mink.solve_ik(self.configuration, [*tasks, *preferences, self.posture], dt,
                                     self.solver, damping=self.damping,
                                     limits=self.limits, constraints=constraints)
            self.configuration.integrate_inplace(velocity, dt)
            new_error = self._error(tasks)
            iterations += 1
            if error - new_error <= self.tolerance:
                break
            error = new_error
        self.last_diagnostics["stages"].append({
            "initial_error": float(initial_error),
            "final_error": float(new_error),
            "iterations": iterations,
        })

    def solve(self, targets):
        """GMR steps 4-5 toward ``{robot_body: 4x4}``; return ``{side: 7 joints}``.

        Arms without a target stay frozen at their previous solution.
        """
        sides = {self.side_of[body] for body in targets}
        self.last_diagnostics = {"stages": [], "solve_time_s": 0.0}
        started = time.perf_counter()
        if self.decouple_wrist:
            result = self._solve_decoupled(targets, sides)
            self.last_diagnostics["solve_time_s"] = time.perf_counter() - started
            return result
        free = set()
        for side in sides:
            free.update(int(dof) for dof in self.dof_ids[side])
        frozen = [dof for dof in range(self.model.nv) if dof not in free]
        constraints = [mink.DofFreezingTask(self.model, dof_indices=frozen)] if frozen else []
        preferences = self._elbow_preferences(targets)
        for table, tasks in zip(self.tables, self.tasks):
            active = []
            for key_body, task in zip(table, tasks):
                if key_body.robot_body in targets:
                    task.set_target(mink.SE3.from_matrix(targets[key_body.robot_body]))
                    active.append(task)
            if active:
                self._solve_stage(active, constraints, preferences)
        q = self.configuration.q
        self.last_diagnostics["solve_time_s"] = time.perf_counter() - started
        return {side: q[self.qpos_ids[side]].copy() for side in sides}

    def _solve_decoupled(self, targets, sides):
        """Use a neutral-wrist virtual arm for position, then rotate the wrist.

        Keeping the position solve independent of wrist orientation prevents
        wrist motion (and its palm offset) feeding back into the elbow solver.
        The actual palm may move around the wrist pivot during rotation.
        """
        if self.calibration_wrists is None:
            raise RuntimeError("Calibrate before decoupled wrist solving")
        q = self.configuration.q.copy()
        wrists = {s: q[self.qpos_ids[s][4:]].copy() for s in sides}
        for side in sides:
            q[self.qpos_ids[side][4:]] = self.calibration_wrists[side]
        self.configuration.update(q)
        for task_map, joint_slice in ((self.position_tasks, slice(0, 4)),
                                      (self.rotation_tasks, slice(4, 7))):
            if task_map is self.rotation_tasks:
                q = self.configuration.q.copy()
                for side in sides:
                    q[self.qpos_ids[side][4:]] = wrists[side]
                self.configuration.update(q)
            free = {int(d) for s in sides for d in self.dof_ids[s][joint_slice]}
            constraints = [mink.DofFreezingTask(
                self.model, dof_indices=[d for d in range(self.model.nv) if d not in free])]
            tasks = []
            for body, target in targets.items():
                task = task_map[body]
                stage_target = target.copy()
                if task_map is self.position_tasks:
                    # _error also reads unweighted orientation residuals;
                    # keep those independent of the operator's wrist turn.
                    stage_target[:3, :3] = np.eye(3)
                task.set_target(mink.SE3.from_matrix(stage_target))
                tasks.append(task)
            if tasks:
                self._solve_stage(tasks, constraints)
        return {s: self.configuration.q[self.qpos_ids[s]].copy() for s in sides}

    def errors(self, targets):
        """``{robot_body: (position m, rotation rad)}`` of the current IK solution."""
        out = {}
        for body, target in targets.items():
            pose = self.configuration.get_transform_frame_to_world(body, "body").as_matrix()
            out[body] = (
                float(np.linalg.norm(target[:3, 3] - pose[:3, 3])),
                float(np.linalg.norm(Rotation.from_matrix(
                    target[:3, :3] @ pose[:3, :3].T).as_rotvec())))
        return out


class PoseTargetFilter:
    """Low-pass and rate-limit one 6D target so tracker glitches cannot fling the arm."""

    def __init__(self, tau_s=0.05, max_speed_m_s=1.0, max_angular_speed_rad_s=6.0):
        self.tau_s = float(tau_s)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_angular_speed_rad_s = float(max_angular_speed_rad_s)
        if (not math.isfinite(self.tau_s) or self.tau_s < 0 or
                not self.max_speed_m_s > 0 or not self.max_angular_speed_rad_s > 0):
            raise ValueError("Invalid tracker filter settings")
        self.pose = None
        self.max_step_m = 0.0

    def reset(self, pose=None):
        self.pose = None if pose is None else np.asarray(pose, dtype=np.float64).copy()

    def update(self, pose, dt):
        pose = np.asarray(pose, dtype=np.float64)
        if self.pose is None:
            self.pose = pose.copy()
            return self.pose.copy()
        dt = min(max(float(dt), 1e-3), 0.25)
        alpha = 1.0 if self.tau_s == 0 else 1.0 - math.exp(-dt / self.tau_s)
        step = alpha * (pose[:3, 3] - self.pose[:3, 3])
        length = float(np.linalg.norm(step))
        if length > self.max_speed_m_s * dt:
            step *= self.max_speed_m_s * dt / length
            length = self.max_speed_m_s * dt
        self.max_step_m = max(self.max_step_m, length)
        rotvec = alpha * Rotation.from_matrix(pose[:3, :3] @ self.pose[:3, :3].T).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle > self.max_angular_speed_rad_s * dt:
            rotvec *= self.max_angular_speed_rad_s * dt / angle
        out = np.eye(4)
        out[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ self.pose[:3, :3]
        out[:3, 3] = self.pose[:3, 3] + step
        self.pose = out
        return out.copy()


class TrackerArmMapper:
    """Wrist trackers to calibrated, filtered GMR targets to both arms' joint goals.

    :meth:`calibrate` (the R key) pairs each tracker with a side and pins the
    operator's pose to the robot's current arm pose. A side whose tracker is
    missing, stale, or invalid yields no goal, so that arm holds.
    """

    def __init__(self, retargeter, *, left_serial=None, right_serial=None,
                 stale_after=TRACKER_STALE_SECONDS):
        self.retargeter = retargeter
        self.configured_serials = {"left": left_serial, "right": right_serial}
        self.stale_after = float(stale_after)
        self.serials = None
        self.filters = {side: PoseTargetFilter() for side in SIDES}
        self.status = {side: "waiting" for side in SIDES}
        self.tracker_positions = {side: None for side in SIDES}
        self.tracker_rotations = {side: None for side in SIDES}
        self.raw_targets = {side: None for side in SIDES}
        self.targets = {side: None for side in SIDES}
        self.errors = {side: None for side in SIDES}
        self.goals = {}
        self.last_revision = None
        self.last_frame_time = None
        self.body_of = {side: body for body, side in retargeter.side_of.items()}

    def _fresh_frame(self, snapshot, now):
        if snapshot.error:
            raise ValueError(snapshot.error)
        frame = snapshot.frame
        if frame is None:
            raise ValueError("No tracker data yet; press Send in the headset's Motion Tracker tab")
        if now - frame.received_at > self.stale_after:
            raise ValueError("Tracker data is stale")
        return frame

    def calibrate(self, snapshot, arm_targets, now):
        """Pin the current wrist poses to ``arm_targets``; return the heading yaw."""
        frame = self._fresh_frame(snapshot, now)
        try:
            yaw = operator_heading(frame.headset)
        except ValueError:
            yaw = None
        serials = assign_tracker_sides(
            frame.poses, yaw=0.0 if yaw is None else yaw,
            left_serial=self.configured_serials["left"],
            right_serial=self.configured_serials["right"])
        if yaw is None:
            yaw = operator_heading(None, (frame.poses[serials["left"]],
                                          frame.poses[serials["right"]]))
        human = {WRIST_BODIES[side]: frame.poses[serial] for side, serial in serials.items()}
        self.retargeter.calibrate(human, arm_targets, root=frame.headset, yaw=yaw)
        self.serials = serials
        for side in SIDES:
            self.filters[side].reset()
            self.status[side] = "calibrated"
        self.goals = {}
        self.last_revision = snapshot.revision
        self.last_frame_time = frame.received_at
        return yaw

    def update(self, snapshot, now):
        """Return ``{side: 7 joint goals}`` for every side that should move."""
        if self.serials is None:
            return {}
        try:
            frame = self._fresh_frame(snapshot, now)
        except ValueError as exc:
            state = "holding" if "stale" in str(exc) else "invalid"
            for side in SIDES:
                self.status[side] = state
            return {}
        if snapshot.revision == self.last_revision:
            return dict(self.goals)
        dt = frame.received_at - self.last_frame_time
        self.last_revision = snapshot.revision
        self.last_frame_time = frame.received_at
        human = {}
        for side, serial in self.serials.items():
            pose = frame.poses.get(serial)
            self.tracker_positions[side] = None if pose is None else pose[:3, 3].copy()
            self.tracker_rotations[side] = None if pose is None else pose[:3, :3].copy()
            if pose is None:
                self.status[side] = "lost"
                self.goals.pop(side, None)
            else:
                human[WRIST_BODIES[side]] = pose
        if not human:
            return {}
        try:
            raw = self.retargeter.targets(human, root=frame.headset)
        except ValueError:
            # e.g. a headset-rooted config while the headset pose is missing
            for side in SIDES:
                self.status[side] = "invalid"
            self.goals = {}
            return {}
        filtered = {}
        for body, target in raw.items():
            side = self.retargeter.side_of[body]
            self.raw_targets[side] = target
            filtered[body] = self.targets[side] = self.filters[side].update(target, dt)
        self.goals.update(self.retargeter.solve(filtered))
        for body, error in self.retargeter.errors(filtered).items():
            side = self.retargeter.side_of[body]
            self.errors[side] = error
            self.status[side] = "tracking"
        return dict(self.goals)
