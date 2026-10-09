"""Optional articulated left Vega arm + Wuji Hand 2 MuJoCo simulation.

This module is deliberately separate from :mod:`vega_hand_sim`: the existing
fixed-arm scene and launcher remain unchanged.  It has no robot command
transport.  A wrist target is converted to left-arm joint targets with a small
damped-least-squares MuJoCo IK solver.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import queue
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .vega_hand_sim import HAND_XML, _resolve_hand_meshes, _vega_build_function
from .retarget import TrackingController, WujiRetargeter
from .simulation import MujocoHand, passive_viewer
from .sources import GloveSource, SourceSnapshot, SyntheticSource
from .vr_pose_source import PicoControllerPoseSource
from .imu_arm_source import PalmImuArmSource


ARM_JOINT_NAMES = tuple(f"L_arm_j{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"{name}_sim_position" for name in ARM_JOINT_NAMES)
WRIST_JOINT_NAMES = ARM_JOINT_NAMES[4:]
WRIST_ACTUATOR_NAMES = ARM_ACTUATOR_NAMES[4:]
# Elbow down, forearm in front of the torso. Order is L_arm_j1..j7.
# Joint 4 is elbow flexion, negative when bent.
ARM_HOME_QPOS = np.array([1.40, -0.20, 0.20, -1.40, 0.0, 0.0, 0.0])
RIGHT_ARM_JOINT_NAMES = tuple(f"R_arm_j{i}" for i in range(1, 8))
RIGHT_ARM_ACTUATOR_NAMES = tuple(f"{name}_sim_position" for name in RIGHT_ARM_JOINT_NAMES)
# Mirror of the left home: elbow down, hand in front rather than out to the side.
RIGHT_ARM_HOME_QPOS = np.array([-1.20, 0.20, -0.30, -1.40, 0.0, 0.0, 0.0])
PALM_IMU_ARM_INPUTS = ("glove-imu", "glove-gesture", "glove-wrist", "glove-arm")
GESTURE_ARM_INPUTS = ("glove-gesture",)
EE_BODY_NAME = "L_ee"
SHOULDER_BODY_NAME = "L_arm_l2"
ELBOW_BODY_NAME = "L_arm_l4"
SHOULDER_PITCH_INDEX = 0
ELBOW_JOINT_INDEX = 3
# L_arm_j6 pitches the hand the same way elbow flexion does.
WRIST_PITCH_INDEX = 5
HAND_BODY_NAME = "wuji_left_wrist"
HAND_IN_EE_ROTATION = np.diag([1.0, -1.0, -1.0])
OUTPUT = Path.home() / ".cache/dexmate-urdf/vega_1u/vega_1u_left_arm_wuji.xml"
WRIST_OUTPUT = Path.home() / ".cache/dexmate-urdf/vega_1u/vega_1u_left_wrist_wuji.xml"


def _joint_node_map(body: ET.Element) -> dict[str, ET.Element]:
    return {node.get("name"): node for node in body.iter("joint") if node.get("name")}


def _remove_joint_limits(joint: ET.Element) -> None:
    """Drop the URDF stop so the sim joint can travel past the hardware range."""
    joint.attrib.pop("range", None)
    joint.set("limited", "false")


def build_articulated_mjcf(output: Path | None = None, *, wrist_only=False) -> Path:
    """Build a fixed-base Vega scene with left arm or wrist and hand movable."""
    output = Path(output) if output is not None else (WRIST_OUTPUT if wrist_only else OUTPUT)
    if not HAND_XML.is_file():
        raise FileNotFoundError(f"Pinned Wuji Hand 2 model missing: {HAND_XML}")
    vega_path = Path(_vega_build_function()())
    root = ET.parse(vega_path).getroot()
    hand_root = ET.parse(HAND_XML).getroot()
    world = root.find("worldbody")
    base = world.find("body[@name='base_link']") if world is not None else None
    if base is None:
        raise ValueError("Converted Vega MJCF has no base_link")
    if root.find(".//body[@name='L_ee']") is None:
        raise ValueError("Converted Vega MJCF has no L_ee body")

    # Keep the zero-pose body tree as a visual reference. In wrist-only mode
    # remove the shoulder/elbow joints entirely so they cannot move in physics.
    # The full-arm scene also holds the right arm in front of the torso.
    movable_names = WRIST_JOINT_NAMES if wrist_only else ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
    movable_actuators = WRIST_ACTUATOR_NAMES if wrist_only else ARM_ACTUATOR_NAMES
    for joint in list(base.iter("joint")):
        if joint.get("name") not in movable_names:
            parent = next((node for node in base.iter() if joint in list(node)), None)
            if parent is not None:
                parent.remove(joint)
    for freejoint in list(base.iter("freejoint")):
        parent = next((node for node in base.iter() if freejoint in list(node)), None)
        if parent is not None:
            parent.remove(freejoint)
    base.set("pos", "0 0 0.15")
    for geom in base.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    actuators = root.find("actuator")
    if actuators is not None:
        root.remove(actuators)
    arm_actuators = ET.SubElement(root, "actuator")
    joint_map = _joint_node_map(base)
    if any(name not in joint_map for name in movable_names):
        raise ValueError("Converted Vega is missing one or more movable left arm joints")
    for name, actuator_name in zip(movable_names, movable_actuators):
        # The URDF conversion leaves zero damping/armature on some pivots. Add
        # modest joint dynamics so MuJoCo's position servo remains well behaved.
        joint_map[name].set("armature", "0.03")
        joint_map[name].set("damping", "2.0")
        _remove_joint_limits(joint_map[name])
        ET.SubElement(arm_actuators, "position", name=actuator_name, joint=name,
                      kp="480", kv="48", forcerange="-150 150",
                      ctrllimited="false", forcelimited="true")
    if not wrist_only:
        for name, actuator_name in zip(RIGHT_ARM_JOINT_NAMES, RIGHT_ARM_ACTUATOR_NAMES):
            if name not in joint_map:
                raise ValueError(f"Converted Vega is missing {name}")
            joint_map[name].set("armature", "0.03")
            joint_map[name].set("damping", "2.0")
            _remove_joint_limits(joint_map[name])
            ET.SubElement(arm_actuators, "position", name=actuator_name, joint=name,
                          kp="480", kv="48", forcerange="-150 150",
                          ctrllimited="false", forcelimited="true")

    # Attach the unmodified Hand 2 model to the arm end-effector.
    _resolve_hand_meshes(hand_root)
    asset = root.find("asset")
    hand_asset = hand_root.find("asset")
    if asset is None or hand_asset is None:
        raise ValueError("Vega or Wuji MJCF is missing its asset section")
    existing = {item.get("name") for item in asset.findall("mesh")}
    for item in hand_asset:
        if item.tag == "mesh" and item.get("name") in existing:
            raise ValueError(f"Conflicting mesh asset name: {item.get('name')}")
        asset.append(item)
    hand_world = hand_root.find("worldbody")
    wrist = hand_world.find("body[@name='l_wrist']") if hand_world is not None else None
    if wrist is None:
        raise ValueError("Wuji Hand model has no l_wrist body")
    wrist.set("name", HAND_BODY_NAME)
    # Point fingers out of L_ee and start with the left palm facing down,
    # matching the wearer's neutral palm-down calibration pose.
    wrist.set("quat", "0 1 0 0")
    joint_default = hand_root.find("./default/joint")
    if joint_default is None:
        raise ValueError("Wuji Hand model has no joint dynamics defaults")
    for joint in wrist.iter("joint"):
        for key, value in joint_default.attrib.items():
            joint.attrib.setdefault(key, value)
    root.find(".//body[@name='L_ee']").append(wrist)

    contacts = root.find("contact")
    if contacts is None:
        contacts = ET.SubElement(root, "contact")
    hand_contacts = hand_root.find("contact")
    if hand_contacts is not None:
        for exclusion in hand_contacts:
            if exclusion.get("body1") == "l_wrist":
                exclusion.set("body1", HAND_BODY_NAME)
            if exclusion.get("body2") == "l_wrist":
                exclusion.set("body2", HAND_BODY_NAME)
            contacts.append(exclusion)
    for actuator in hand_root.find("actuator"):
        arm_actuators.append(actuator)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def _named_ids(model, objtype, names):
    ids = np.array([mujoco.mj_name2id(model, objtype, name) for name in names], dtype=int)
    if np.any(ids < 0) or len(np.unique(ids)) != len(names):
        raise ValueError(f"Missing or duplicate named model entries: {names}")
    return ids


class MujocoArm:
    """Named, limited position control for Vega's left arm or distal wrist."""

    def __init__(self, model, data, *, wrist_only=False):
        self.model, self.data = model, data
        joint_names = WRIST_JOINT_NAMES if wrist_only else ARM_JOINT_NAMES
        actuator_names = WRIST_ACTUATOR_NAMES if wrist_only else ARM_ACTUATOR_NAMES
        self.joint_ids = _named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names)
        self.actuator_ids = _named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_names)
        if not np.array_equal(model.actuator_trnid[self.actuator_ids, 0], self.joint_ids):
            raise ValueError("Left arm actuator-to-joint mapping is incorrect")
        if np.any(model.jnt_limited[self.joint_ids]):
            raise ValueError("Left arm joints are unlimited in this scene")
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.lower = np.full(len(self.joint_ids), -np.inf)
        self.upper = np.full(len(self.joint_ids), np.inf)
        neutral = np.zeros(3) if wrist_only else ARM_HOME_QPOS
        self.home = neutral.copy()
        self.target = self.home.copy()
        self.right_qpos_ids = None
        if not wrist_only:
            right_joints = _named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_ARM_JOINT_NAMES)
            right_actuators = _named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ARM_ACTUATOR_NAMES)
            if not np.array_equal(model.actuator_trnid[right_actuators, 0], right_joints):
                raise ValueError("Right arm actuator-to-joint mapping is incorrect")
            self.right_qpos_ids = model.jnt_qposadr[right_joints]
            self.right_dof_ids = model.jnt_dofadr[right_joints]
            self.right_actuator_ids = right_actuators
            self.right_home = RIGHT_ARM_HOME_QPOS.copy()

    def set_target(self, target):
        target = np.asarray(target, dtype=np.float64)
        if target.shape != self.home.shape or not np.isfinite(target).all():
            raise ValueError(f"Expected {len(self.home)} finite arm joint targets")
        self.target = np.clip(target, self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = self.target

    def initialize(self):
        self.data.qpos[self.qpos_ids] = self.target
        self.data.qvel[self.dof_ids] = 0
        self.set_target(self.target)
        if self.right_qpos_ids is not None:
            self.data.qpos[self.right_qpos_ids] = self.right_home
            self.data.qvel[self.right_dof_ids] = 0
            self.data.ctrl[self.right_actuator_ids] = self.right_home
        mujoco.mj_forward(self.model, self.data)


def desired_shoulder_pitch(forward_m):
    """Pitch the upper arm from the shoulder for a forward or back hand target.

    Joint 1 swings the whole arm. It is 1.40 rad at the home hand, about
    0.58 m in front of the robot. Smaller values swing the upper arm forward
    and up, down to -0.40 rad. Larger values swing it back and down, up to
    2.70 rad; past that the elbow rises behind the body. A glove tilt only
    moves the target about 0.10-0.25 m, so the slope is steep: 0.10 m swings
    the shoulder about 0.5-0.7 rad.
    """
    forward = float(np.clip(forward_m, 0.0, 1.40))
    return float(np.interp(forward, [0.33, 0.58, 0.83], [2.70, 1.40, -0.40]))


def desired_elbow_angle(reach_m, lateral_m=None, forward_m=None, height_m=None):
    """Choose elbow flexion from side-to-side position and hand height.

    Joint 4 is negative in flexion. The home pose holds the hand in front of
    the torso, near world (0.58, 0.05, 0.89) m, at -1.40 rad. Raising the hand
    curls the forearm toward the shoulder. Sweeping out to the robot's left
    opens the elbow. Forward and back are handled by the shoulder pitch, so
    ``forward_m`` is accepted and ignored.
    """
    del forward_m
    reach = float(np.clip(reach_m, 0.32, 0.80))
    from_reach = float(np.interp(reach, [0.34, 0.48, 0.61, 0.78], [-2.50, -1.90, -1.40, -0.90]))
    if lateral_m is None:
        return from_reach
    lateral = float(np.clip(lateral_m, -0.40, 1.05))
    # Zero at the in-front home pose, so a pure side move still bends.
    side_delta = float(np.interp(lateral, [-0.25, 0.05, 0.40, 0.70], [-0.55, 0.0, 0.55, 0.90]))
    reach_delta = 0.25 * (from_reach - (-1.40))
    height_delta = 0.0
    if height_m is not None:
        height = float(np.clip(height_m, 0.40, 1.40))
        height_delta = float(np.interp(height, [0.70, 0.89, 1.02, 1.15], [0.25, 0.0, -0.55, -0.90]))
    return float(np.clip(
        -1.40 + side_delta + reach_delta + height_delta, -2.55, -0.40))


class DampedLeastSquaresIK:
    """Local DLS IK with a reach-dependent elbow-flexion task."""

    def __init__(self, model, data, arm: MujocoArm, damping=0.08,
                 max_step=0.08, iterations=12, orientation_weight=0.45,
                 posture_gain=0.08, elbow_weight=0.55, elbow_down_weight=0.55,
                 position_weight=1.0, orient_fade_distance=None,
                 distal_position_scale=1.0, wrist_pitch_scale=1.0,
                 shoulder_weight=0.0):
        self.model, self.data, self.arm = model, data, arm
        self.scratch = mujoco.MjData(model)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        self.shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)
        self.elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ELBOW_BODY_NAME)
        if self.body_id < 0 or self.shoulder_id < 0 or self.elbow_id < 0:
            raise ValueError("End-effector, shoulder, or elbow body is missing")
        self.damping = float(damping)
        self.max_step = float(max_step)
        self.iterations = int(iterations)
        self.orientation_weight = float(orientation_weight)
        self.posture_gain = float(posture_gain)
        self.elbow_weight = float(elbow_weight)
        self.elbow_down_weight = float(elbow_down_weight)
        self.position_weight = float(position_weight)
        self.orient_fade_distance = (None if orient_fade_distance is None
                                     else float(orient_fade_distance))
        self.distal_position_scale = float(distal_position_scale)
        self.wrist_pitch_scale = float(wrist_pitch_scale)
        self.shoulder_weight = float(shoulder_weight)
        if (self.damping <= 0 or self.max_step <= 0 or self.iterations < 1 or
                self.posture_gain < 0 or self.elbow_weight < 0 or self.elbow_down_weight < 0 or
                self.position_weight <= 0 or self.distal_position_scale < 0 or
                self.wrist_pitch_scale < 0 or self.shoulder_weight < 0 or
                (self.orient_fade_distance is not None and self.orient_fade_distance <= 0)):
            raise ValueError("Invalid DLS solver settings")

    @staticmethod
    def _rotation_error(target, current):
        relative = target @ current.T
        angle = math.acos(float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
        if angle < 1e-7:
            return np.zeros(3)
        axis = np.array([relative[2, 1] - relative[1, 2],
                         relative[0, 2] - relative[2, 0],
                         relative[1, 0] - relative[0, 1]]) / (2 * math.sin(angle))
        return axis * angle

    @staticmethod
    def _elbow_down_error(shoulder, elbow, wrist):
        """Pull the elbow under the shoulder-to-wrist line, toward world down."""
        down = np.array([0.0, 0.0, -1.0])
        span = wrist - shoulder
        span_norm = float(np.linalg.norm(span))
        if span_norm < 1e-4:
            return np.zeros(3)
        span_hat = span / span_norm
        offset = elbow - shoulder
        along = float(np.dot(offset, span_hat))
        pole = offset - along * span_hat
        pole_norm = float(np.linalg.norm(pole))
        target_dir = down - float(np.dot(down, span_hat)) * span_hat
        target_norm = float(np.linalg.norm(target_dir))
        if pole_norm < 1e-4 or target_norm < 1e-4:
            return np.zeros(3)
        desired = shoulder + along * span_hat + (target_dir / target_norm) * pole_norm
        return desired - elbow

    def solve(self, target_pose):
        pose = np.asarray(target_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Expected a finite 4x4 homogeneous wrist target")
        # IK runs in its own state so it cannot teleport the displayed arm.
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = self.data.qvel
        q = self.scratch.qpos[self.arm.qpos_ids].copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        elbow_jacp = np.zeros((3, self.model.nv))
        dofs = self.arm.dof_ids
        elbow_row = np.zeros(7)
        elbow_row[ELBOW_JOINT_INDEX] = self.elbow_weight
        shoulder_row = np.zeros(7)
        shoulder_row[SHOULDER_PITCH_INDEX] = self.shoulder_weight
        task_size = 11 if self.shoulder_weight > 0 else 10
        for _ in range(self.iterations):
            self.scratch.qpos[self.arm.qpos_ids] = q
            mujoco.mj_forward(self.model, self.scratch)
            shoulder = self.scratch.xpos[self.shoulder_id].copy()
            elbow = self.scratch.xpos[self.elbow_id].copy()
            current_pos = self.scratch.xpos[self.body_id].copy()
            current_rot = self.scratch.xmat[self.body_id].reshape(3, 3).copy()
            reach = float(np.linalg.norm(pose[:3, 3] - shoulder))
            # World +Y is the robot's left. Use the target, not the moving
            # shoulder, so the elbow goal stays put when the hand is still.
            elbow_goal = desired_elbow_angle(
                reach, float(pose[1, 3]), float(pose[0, 3]), float(pose[2, 3]))
            shoulder_goal = desired_shoulder_pitch(float(pose[0, 3]))
            down_error = self._elbow_down_error(shoulder, elbow, pose[:3, 3])
            pos_error = pose[:3, 3] - current_pos
            ori_weight = self.orientation_weight
            if self.orient_fade_distance is not None:
                # Follow the hand position first. Aim the wrist once it is close.
                closeness = 1.0 - np.linalg.norm(pos_error) / self.orient_fade_distance
                ori_weight *= float(np.clip(closeness, 0.2, 1.0))
            error_parts = [
                self.position_weight * pos_error,
                ori_weight * self._rotation_error(pose[:3, :3], current_rot),
                [self.elbow_weight * (elbow_goal - q[ELBOW_JOINT_INDEX])],
                self.elbow_down_weight * down_error,
            ]
            if self.shoulder_weight > 0:
                error_parts.append(
                    [self.shoulder_weight * (shoulder_goal - q[SHOULDER_PITCH_INDEX])])
            error = np.concatenate(error_parts)
            if np.linalg.norm(error) < 1e-4:
                break
            mujoco.mj_jacBody(self.model, self.scratch, jacp, jacr, self.body_id)
            mujoco.mj_jacBody(self.model, self.scratch, elbow_jacp, None, self.elbow_id)
            pos_jac = jacp[:, dofs].copy()
            ori_jac = jacr[:, dofs].copy()
            # The wrist can aim the palm. It should not curl the hand; that
            # bend belongs to the elbow.
            pos_jac[:, 4:] *= self.distal_position_scale
            ori_jac[:, WRIST_PITCH_INDEX] *= self.wrist_pitch_scale
            jac_rows = [self.position_weight * pos_jac,
                        ori_weight * ori_jac,
                        elbow_row,
                        self.elbow_down_weight * elbow_jacp[:, dofs]]
            if self.shoulder_weight > 0:
                jac_rows.append(shoulder_row)
            jac = np.vstack(jac_rows)
            # Damped least squares. Shoulder pitch carries forward and back.
            # Elbow flexion follows height and side-to-side motion.
            inverse = np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(task_size),
                                      np.eye(task_size))
            pseudoinverse = jac.T @ inverse
            null_projector = np.eye(7) - pseudoinverse @ jac
            posture = self.arm.home.copy()
            posture[ELBOW_JOINT_INDEX] = elbow_goal
            if self.shoulder_weight > 0:
                posture[SHOULDER_PITCH_INDEX] = shoulder_goal
            delta = (pseudoinverse @ error + self.posture_gain *
                     null_projector @ (posture - q))
            magnitude = np.linalg.norm(delta)
            if magnitude > self.max_step:
                delta *= self.max_step / magnitude
            q = np.clip(q + delta, self.arm.lower, self.arm.upper)
        return q


class WristOrientationIK:
    """Rotate only Vega's three distal wrist joints; proximal arm is welded."""

    def __init__(self, model, data, wrist: MujocoArm, *, damping=0.06,
                 max_step=0.12, iterations=16):
        if len(wrist.qpos_ids) != 3:
            raise ValueError("Wrist orientation IK needs exactly three wrist joints")
        self.model, self.data, self.wrist = model, data, wrist
        self.scratch = mujoco.MjData(model)
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.damping = float(damping)
        self.max_step = float(max_step)
        self.iterations = int(iterations)
        if self.body_id < 0 or self.damping <= 0 or self.max_step <= 0 or self.iterations < 1:
            raise ValueError("Invalid wrist orientation IK settings")

    def solve(self, target_rotation):
        target = np.asarray(target_rotation, dtype=np.float64)
        if (target.shape != (3, 3) or not np.isfinite(target).all() or
                not np.allclose(target.T @ target, np.eye(3), atol=2e-3) or
                not np.isclose(np.linalg.det(target), 1.0, atol=2e-3)):
            raise ValueError("Expected a proper 3x3 wrist target rotation")
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = self.data.qvel
        q = self.wrist.target.copy()
        for _ in range(self.iterations):
            self.scratch.qpos[self.wrist.qpos_ids] = q
            mujoco.mj_forward(self.model, self.scratch)
            current = self.scratch.xmat[self.body_id].reshape(3, 3)
            error = Rotation.from_matrix(target @ current.T).as_rotvec()
            if np.linalg.norm(error) < 1e-4:
                break
            mujoco.mj_jacBody(self.model, self.scratch, self.jacp, self.jacr,
                              self.body_id)
            jac = self.jacr[:, self.wrist.dof_ids]
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + self.damping**2 * np.eye(3), error)
            magnitude = float(np.linalg.norm(delta))
            if magnitude > self.max_step:
                delta *= self.max_step / magnitude
            q = np.clip(q + delta, self.wrist.lower, self.wrist.upper)
        return q


class SyntheticWristPoseSource:
    """Deterministic small reachable wrist motion around a captured pose."""

    def __init__(self, origin_pose, position_amplitude=(0.035, 0.025, 0.025),
                 rotation_amplitude=(0.12, 0.10, 0.12), period=8.0):
        self.origin = np.asarray(origin_pose, dtype=np.float64).copy()
        if self.origin.shape != (4, 4) or not np.isfinite(self.origin).all():
            raise ValueError("Expected a finite 4x4 origin pose")
        self.position_amplitude = np.asarray(position_amplitude, dtype=np.float64)
        self.rotation_amplitude = np.asarray(rotation_amplitude, dtype=np.float64)
        self.period = float(period)
        if self.period <= 0:
            raise ValueError("period must be positive")

    @staticmethod
    def _axis_rotation(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c, s, one = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
        return np.array([[c + x*x*one, x*y*one - z*s, x*z*one + y*s],
                         [y*x*one + z*s, c + y*y*one, y*z*one - x*s],
                         [z*x*one - y*s, z*y*one + x*s, c + z*z*one]])

    def read(self, seconds):
        phase = 2 * math.pi * float(seconds) / self.period
        pose = self.origin.copy()
        local_delta = self.position_amplitude * np.array(
            [math.sin(phase), 0.65 * math.sin(phase * 0.5), math.cos(phase)])
        pose[:3, 3] += self.origin[:3, :3] @ local_delta
        angles = self.rotation_amplitude * np.array(
            [math.sin(phase * .7), math.sin(phase), math.cos(phase * .8)])
        rot = self._axis_rotation([1, 0, 0], angles[0])
        rot = rot @ self._axis_rotation([0, 1, 0], angles[1])
        rot = rot @ self._axis_rotation([0, 0, 1], angles[2])
        pose[:3, :3] = self.origin[:3, :3] @ rot
        return pose


def load_articulated_scene(*, wrist_only=False):
    """Load and validate the articulated left-arm or wrist-only scene."""
    path = build_articulated_mjcf(wrist_only=wrist_only)
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.opt.timestep != 0.002:
        raise ValueError(f"Expected 2 ms timestep, got {model.opt.timestep}")
    data = mujoco.MjData(model)
    arm = MujocoArm(model, data, wrist_only=wrist_only)
    hand = MujocoHand(model, data, "left")
    expected = (23, 23, 23) if wrist_only else (34, 34, 34)
    if (model.nq, model.nv, model.nu) != expected:
        detail = "3 wrist + 20 hand" if wrist_only else "14 arm + 20 hand"
        raise ValueError(f"Expected {detail} coordinates/actuators; got "
                         f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    arm.initialize()
    hand.initialize()
    return model, data, arm, hand


@dataclass
class ArmHandRunReport:
    finger_input: str
    side: str = "left"
    elapsed_seconds: float = 0.0
    simulation_seconds: float = 0.0
    received_frames: int = 0
    finger_updates: int = 0
    arm_ik_updates: int = 0
    input_hz: float = 0.0
    finger_hz: float = 0.0
    arm_hz: float = 0.0
    frame_age_ms: float | None = None
    max_frame_age_ms: float = 0.0
    finger_status: str = "waiting"
    arm_status: str = "tracking"
    arm_imu_displacement_estimate_m: float = 0.0
    arm_imu_max_displacement_estimate_m: float = 0.0
    arm_gesture_command_m: float = 0.0
    arm_gesture_max_command_m: float = 0.0
    arm_gesture_steps: int = 0
    arm_joint_peak_delta_rad: list[float] | None = None
    mujoco_warnings: int = 0


def run(source, *, finger_input, arm_input="synthetic", pose_source=None,
        arm_imu_source=None,
        duration=None, headless=False, report_path=None):
    """Run the arm and glove-finger loops together in the articulated scene.

    MuJoCo physics advances at the model's 500 Hz timestep.  Hand tracking is
    sampled at 120 Hz, arm IK at 40 Hz, and the passive viewer syncs at 30 Hz.
    ``source`` supplies finger landmarks; ``pose_source`` optionally supplies
    Pico poses, while ``arm_imu_source`` supplies drift-limited relative arm
    motion or directional stroke commands from the same glove connection.
    ``glove-arm`` solves all seven joints for a glove orientation plus a
    position taken from how far the fingers tip away from calibration.
    ``glove-wrist`` solves only the three wrist joints.
    """
    wrist_only = arm_input in ("pico-wrist", "glove-wrist")
    model, data, arm, _hand = load_articulated_scene(wrist_only=wrist_only)
    hand = _hand
    finger_controller = TrackingController(WujiRetargeter("left"), hand)
    ik = None if wrist_only else DampedLeastSquaresIK(
        model, data, arm,
        max_step=0.12 if arm_input == "glove-arm" else 0.08,
        iterations=18 if arm_input == "glove-arm" else 12,
        orientation_weight=0.10 if arm_input == "glove-arm" else 0.45,
        elbow_weight=0.40 if arm_input == "glove-arm" else 0.55,
        elbow_down_weight=0.15 if arm_input == "glove-arm" else 0.55,
        position_weight=2.2 if arm_input == "glove-arm" else 1.0,
        orient_fade_distance=0.20 if arm_input == "glove-arm" else None,
        distal_position_scale=0.0 if arm_input == "glove-arm" else 1.0,
        wrist_pitch_scale=0.15 if arm_input == "glove-arm" else 1.0,
        shoulder_weight=1.6 if arm_input == "glove-arm" else 0.0)
    # The solver may jump to a distant pose in one tick. The arm command
    # approaches that pose at a limited joint rate so the motion stays steerable.
    joint_slew = arm.home.copy()
    max_joint_rate = 1.15 if arm_input == "glove-arm" else None

    def command_arm(goal):
        if max_joint_rate is None:
            arm.set_target(goal)
            return
        step = max_joint_rate * arm_period
        joint_slew[:] = np.clip(
            joint_slew + np.clip(goal - joint_slew, -step, step),
            arm.lower, arm.upper)
        arm.set_target(joint_slew)
    wrist_ik = WristOrientationIK(model, data, arm) if wrist_only else None
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    origin = np.eye(4)
    origin[:3, :3] = data.xmat[ee].reshape(3, 3)
    origin[:3, 3] = data.xpos[ee]
    wrist_source = SyntheticWristPoseSource(origin) if arm_input == "synthetic" else None
    if arm_input in ("pico-controller", "pico-wrist") and pose_source is None:
        raise ValueError(f"{arm_input} arm input requires a pose source")
    if arm_input in PALM_IMU_ARM_INPUTS and arm_imu_source is None:
        raise ValueError(f"{arm_input} arm input requires a palm IMU source")
    keys = queue.SimpleQueue()
    context = nullcontext(None) if headless else passive_viewer(model, data, keys.put)
    report = ArmHandRunReport(finger_input)
    arm_start_q = data.qpos[arm.qpos_ids].copy()
    arm_peak_delta = np.zeros(len(arm.qpos_ids))
    snapshot = SourceSnapshot(None, 0, 0, None)
    finger_period, arm_period = 1 / 120, 1 / 40
    viewer_period = 1 / 30
    started = time.monotonic()
    next_physics = next_finger = next_arm = next_viewer = started
    next_status = started + 2
    last_received = last_fingers = last_arm = 0
    last_status = started
    arm_status = "synthetic" if arm_input == "synthetic" else "waiting"
    arm_clutched = False
    arm_needs_release = False
    recenter_requested = False
    controller_origin = None
    ee_reference = None
    status_line = f"Waiting for finger tracking | arm {arm_status}"
    try:
        with context as viewer:
            if viewer:
                with viewer.lock():
                    viewer.opt.geomgroup[2] = 0
                    viewer.opt.sitegroup[3] = 0
                    viewer.cam.lookat[:] = data.xpos[ee]
                    viewer.cam.distance = 1.15
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -15
            started = time.monotonic()
            next_physics = next_finger = next_arm = next_viewer = started
            next_status = started + 2
            last_received = last_fingers = last_arm = 0
            last_status = started
            while duration is None or time.monotonic() - started < duration:
                if viewer and not viewer.is_running():
                    break
                while not keys.empty():
                    key = keys.get()
                    if key == 32:
                        finger_controller.toggle_pause()
                    elif key in (ord("r"), ord("R")) and arm_input in PALM_IMU_ARM_INPUTS:
                        recenter_requested = True
                now = time.monotonic()
                guard = viewer.lock() if viewer else nullcontext()
                with guard:
                    if recenter_requested:
                        recenter_pose = np.eye(4)
                        recenter_pose[:3, :3] = data.xmat[ee].reshape(3, 3)
                        recenter_pose[:3, 3] = data.xpos[ee]
                        arm_imu_source.recenter(recenter_pose)
                        origin = recenter_pose
                        joint_slew[:] = data.qpos[arm.qpos_ids]
                        arm.set_target(joint_slew)
                        arm_status = "calibrating"
                        recenter_requested = False
                    if now >= next_finger:
                        snapshot = source.read()
                        finger_controller.update(snapshot, time.monotonic())
                        if snapshot.frame is not None:
                            age = max(0.0, time.monotonic() - snapshot.frame.received_at) * 1000
                            report.frame_age_ms = age
                            report.max_frame_age_ms = max(report.max_frame_age_ms, age)
                        next_finger += max(1, int((now - next_finger) / finger_period) + 1) * finger_period
                    if now >= next_arm:
                        if wrist_source is not None:
                            command_arm(ik.solve(wrist_source.read(now - started)))
                            arm_status = "synthetic"
                            report.arm_ik_updates += 1
                        elif arm_input in PALM_IMU_ARM_INPUTS:
                            imu_snapshot = arm_imu_source.read(origin, now=now)
                            arm_status = imu_snapshot.status
                            if arm_input in GESTURE_ARM_INPUTS:
                                report.arm_gesture_command_m = imu_snapshot.displacement_estimate_m
                                report.arm_gesture_max_command_m = max(
                                    report.arm_gesture_max_command_m,
                                    imu_snapshot.max_displacement_estimate_m)
                                report.arm_gesture_steps = imu_snapshot.gesture_steps
                            else:
                                report.arm_imu_displacement_estimate_m = imu_snapshot.displacement_estimate_m
                                report.arm_imu_max_displacement_estimate_m = max(
                                    report.arm_imu_max_displacement_estimate_m,
                                    imu_snapshot.max_displacement_estimate_m)
                            if imu_snapshot.pose is not None:
                                if wrist_only:
                                    arm.set_target(wrist_ik.solve(imu_snapshot.pose[:3, :3]))
                                    arm_status = "wrist tracking"
                                else:
                                    command_arm(ik.solve(imu_snapshot.pose))
                                    if arm_input == "glove-arm":
                                        arm_status = "IK"
                                report.arm_ik_updates += 1
                        else:
                            arm_snapshot = pose_source.read()
                            frame = arm_snapshot.frame
                            age_s = (float("inf") if frame is None
                                     else max(0.0, now - frame.received_at))
                            if arm_snapshot.error:
                                arm_status = "invalid"
                                if arm_clutched:
                                    arm_clutched = False
                                    controller_origin = ee_reference = None
                                arm_needs_release = True
                            elif frame is None:
                                arm_status = "waiting"
                            elif age_s > 0.250:
                                arm_status = "stale"
                                # Require a new trigger crossing after input recovers.
                                arm_clutched = False
                                controller_origin = ee_reference = None
                                arm_needs_release = True
                            elif frame.index_trigger <= 0.5:
                                arm_clutched = False
                                arm_needs_release = False
                                controller_origin = ee_reference = None
                                arm_status = "holding"
                            elif arm_needs_release:
                                arm_status = "release trigger to rearm"
                            elif frame.index_trigger > 0.5:
                                if not arm_clutched:
                                    controller_origin = frame.pose.copy()
                                    ee_reference = np.eye(4)
                                    ee_reference[:3, :3] = data.xmat[ee].reshape(3, 3)
                                    ee_reference[:3, 3] = data.xpos[ee]
                                    arm_clutched = True
                                    arm_status = "clutched"
                                else:
                                    delta_position = frame.pose[:3, 3] - controller_origin[:3, 3]
                                    delta_rotation = frame.pose[:3, :3] @ controller_origin[:3, :3].T
                                    target_pose = ee_reference.copy()
                                    target_pose[:3, 3] += delta_position
                                    target_pose[:3, :3] = delta_rotation @ ee_reference[:3, :3]
                                    if wrist_only:
                                        arm.set_target(wrist_ik.solve(target_pose[:3, :3]))
                                        arm_status = "wrist tracking"
                                    else:
                                        command_arm(ik.solve(target_pose))
                                        arm_status = "tracking"
                                    report.arm_ik_updates += 1
                        next_arm += max(1, int((now - next_arm) / arm_period) + 1) * arm_period
                    # Catch up short scheduling delays, while bounding work after stalls.
                    now = time.monotonic()
                    steps = min(25, max(0, int((now - next_physics) / model.opt.timestep) + 1))
                    if steps:
                        mujoco.mj_step(model, data, nstep=steps)
                        arm_peak_delta = np.maximum(
                            arm_peak_delta,
                            np.abs(data.qpos[arm.qpos_ids] - arm_start_q))
                        next_physics += steps * model.opt.timestep
                        if now - next_physics > 0.050:
                            next_physics = now
                now = time.monotonic()
                if now >= next_status:
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError("Non-finite MuJoCo state")
                    report.mujoco_warnings = int(np.sum(data.warning.number))
                    if report.mujoco_warnings:
                        raise RuntimeError(f"MuJoCo raised {report.mujoco_warnings} numerical warnings")
                    age = None if snapshot.frame is None else max(0, now - snapshot.frame.received_at) * 1000
                    report.frame_age_ms = age
                    report.max_frame_age_ms = max(report.max_frame_age_ms, age or 0)
                    dt = max(now - last_status, 1e-6)
                    input_hz = (snapshot.received - last_received) / dt
                    finger_hz = (finger_controller.updates - last_fingers) / dt
                    arm_hz = (report.arm_ik_updates - last_arm) / dt
                    age_text = "n/a" if age is None else f"{age:.1f} ms"
                    status_line = (f"left | fingers {finger_controller.status} | arm {arm_status} | "
                                   f"input {input_hz:.1f} Hz | fingers {finger_hz:.1f} Hz | "
                                   f"arm IK {arm_hz:.1f} Hz | age {age_text} | "
                                   f"{'wrist' if wrist_only else 'arm'} joints >3° "
                                   f"{np.count_nonzero(arm_peak_delta > 0.052):d}/{len(arm_peak_delta)}")
                    if arm_input in ("glove-imu", "glove-arm"):
                        label = "tilt Δp" if arm_input == "glove-arm" else "IMU est Δp"
                        status_line += (f" | {label} {report.arm_imu_displacement_estimate_m:.3f} m"
                                        f" (max {report.arm_imu_max_displacement_estimate_m:.3f} m)")
                    elif arm_input in GESTURE_ARM_INPUTS:
                        status_line += (f" | strokes {report.arm_gesture_steps}"
                                        f" | arm command Δp {report.arm_gesture_command_m:.3f} m")
                    print(status_line + (f" | {finger_controller.detail}"
                                         if finger_controller.detail else ""), flush=True)
                    last_received, last_fingers = snapshot.received, finger_controller.updates
                    last_arm, last_status = report.arm_ik_updates, now
                    next_status = now + 2
                if viewer and now >= next_viewer:
                    title = ("Vega left wrist + Wuji Hand 2" if wrist_only else
                             "Vega left arm + Wuji Hand 2")
                    viewer.set_texts((None, None, title, status_line))
                    viewer.sync()
                    next_viewer = now + viewer_period
                wake_at = min(next_physics, next_finger, next_arm)
                sleep_for = wake_at - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        report.elapsed_seconds = time.monotonic() - started
        report.simulation_seconds = float(data.time)
        report.arm_joint_peak_delta_rad = arm_peak_delta.tolist()
        report.received_frames = snapshot.received
        report.finger_updates = finger_controller.updates
        report.input_hz = snapshot.received / max(report.elapsed_seconds, 1e-6)
        report.finger_hz = finger_controller.updates / max(report.elapsed_seconds, 1e-6)
        report.arm_hz = report.arm_ik_updates / max(report.elapsed_seconds, 1e-6)
        report.finger_status = finger_controller.status
        report.arm_status = arm_status
        if snapshot.frame is not None:
            report.frame_age_ms = max(0, time.monotonic() - snapshot.frame.received_at) * 1000
            report.max_frame_age_ms = max(report.max_frame_age_ms, report.frame_age_ms)
        report.mujoco_warnings = int(np.sum(data.warning.number))
        serialized = json.dumps(asdict(report), indent=2)
        print("Run report:\n" + serialized, flush=True)
        if report_path:
            Path(report_path).write_text(serialized + "\n")
    return report


def main(argv=None):
    import argparse

    def positive_seconds(value):
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError("must be a finite number greater than zero")
        return number

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finger-input", choices=("glove", "synthetic"), default="synthetic")
    parser.add_argument("--arm-input", choices=("synthetic", "pico-controller", "pico-wrist", "glove-imu", "glove-gesture", "glove-wrist", "glove-arm"), default="synthetic")
    parser.add_argument("--glove-sn")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=positive_seconds)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    if args.headless and args.duration is None:
        parser.error("--headless requires --duration")
    if args.finger_input == "synthetic" and (args.glove_sn or args.user_id):
        parser.error("--glove-sn and --user-id apply only to glove input")
    if args.arm_input in PALM_IMU_ARM_INPUTS and args.finger_input != "glove":
        parser.error(f"--arm-input {args.arm_input} requires --finger-input glove so both use one glove connection")
    try:
        if args.finger_input == "glove":
            import wuji_sdk
            wuji_sdk.set_log_level("warn")
            source = GloveSource(args.glove_sn, args.user_id, sdk=wuji_sdk)
        else:
            source = SyntheticSource("left")
        pose_source = None
        arm_imu_source = None
        try:
            if args.arm_input in ("pico-controller", "pico-wrist"):
                pose_source = PicoControllerPoseSource()
            elif args.arm_input in PALM_IMU_ARM_INPUTS:
                from .imu_arm_source import PalmImuRetargeter
                mount_rotation = source.palm_imu_mount_rotation()
                gesture = args.arm_input == "glove-gesture"
                wrist = args.arm_input == "glove-wrist"
                whole_arm = args.arm_input == "glove-arm"
                arm_imu_source = PalmImuArmSource(
                    source.palm_imu_resource(),
                    estimator=PalmImuRetargeter(
                        imu_in_wrist_rotation=mount_rotation,
                        hand_in_ee_rotation=HAND_IN_EE_ROTATION,
                        motion_mode=("orientation" if wrist else
                                     "tilt" if whole_arm else
                                     "gesture" if gesture else "relative"),
                        max_displacement=(0.65 if whole_arm else
                                          0.30 if gesture else 0.10),
                        max_speed=0.22 if whole_arm else 0.30 if gesture else 0.20,
                        velocity_damping=3.2 if whole_arm else 1.8,
                        stationary_accel=0.40 if whole_arm else 0.20,
                        gesture_step=0.06,
                        tilt_gain=0.48 if whole_arm else 0.50,
                        tilt_deadband=0.06 if whole_arm else 0.05,
                        tilt_smoothing=0.20 if whole_arm else 0.45,
                        translation_leak=1.6 if whole_arm else 0.0,
                        translation_limit=0.15 if whole_arm else None,
                        vertical_to_forward=whole_arm))
            if source.side != "left":
                raise ValueError(f"Articulated Vega scene requires a left glove, got {source.side}")
            if args.finger_input == "glove":
                print(f"Glove {source.serial}: side={source.side}, firmware={source.firmware}, "
                      f"address={source.address}", flush=True)
            run(source, finger_input=args.finger_input, arm_input=args.arm_input,
                pose_source=pose_source, arm_imu_source=arm_imu_source, duration=args.duration,
                headless=args.headless, report_path=args.report)
        finally:
            for closeable in (arm_imu_source, source, pose_source):
                if closeable is None:
                    continue
                try:
                    closeable.close()
                except Exception as cleanup_error:
                    print(f"Warning: input cleanup failed: {cleanup_error}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("Stopped; glove connection closed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
