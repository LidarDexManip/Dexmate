"""GMR-style wrist tracker retargeting onto both Vega arms."""

import copy
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wuji_teleop.gmr_arm_retarget import (
    DEFAULT_IK_CONFIG,
    GmrArmRetargeter,
    PoseTargetFilter,
    TrackerArmMapper,
    WRIST_BODIES,
    load_ik_config,
)
from wuji_teleop.vega_arm_sim import RIGHT_ARM_JOINT_LIMITS
from wuji_teleop.vega_bimanual_sim import SIDES, arm_joint_names, body_names, load_bimanual_scene
from wuji_teleop.xr_tracker_source import TrackerFrame, TrackerSnapshot, heading_rotation


def _yawed(pose, yaw):
    """``pose`` as seen in a world turned by ``yaw`` about z."""
    turn = np.eye(4)
    turn[:3, :3] = heading_rotation(yaw)
    return turn @ pose


class _Scene:
    model = data = arms = None

    @classmethod
    def load(cls):
        if cls.model is None:
            cls.model, cls.data, cls.arms, _ = load_bimanual_scene()
        return cls


class GmrArmRetargeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = _Scene.load()
        cls.model, cls.arms = scene.model, scene.arms
        cls.names = {side: arm_joint_names(side) for side in SIDES}
        cls.home = {side: cls.arms[side].home.copy() for side in SIDES}

    def make(self, config=DEFAULT_IK_CONFIG, **kwargs):
        retargeter = GmrArmRetargeter(self.model, self.names, config, **kwargs)
        retargeter.sync(self.home)
        return retargeter

    def ee_poses(self, retargeter):
        return {side: retargeter.configuration.get_transform_frame_to_world(
            body_names(side)["ee"], "body").as_matrix() for side in SIDES}

    def rest_pose_human(self, retargeter):
        """An operator whose wrists sit exactly where the robot hands are."""
        return {WRIST_BODIES[side]: pose for side, pose in self.ee_poses(retargeter).items()}

    def fk_pose(self, retargeter, side, arm_qpos):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = retargeter.neutral_qpos
        scratch.qpos[self.arms[side].qpos_ids] = arm_qpos
        mujoco.mj_kinematics(self.model, scratch)
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)["ee"])
        pose = np.eye(4)
        pose[:3, :3] = scratch.xmat[ee].reshape(3, 3)
        pose[:3, 3] = scratch.xpos[ee]
        return pose

    def test_default_config_uses_gmr_format(self):
        config = load_ik_config()
        self.assertEqual([len(table) for table in config["tables"]], [2, 2])
        self.assertEqual({entry.robot_body for entry in config["tables"][0]}, {"L_ee", "R_ee"})
        raw = json.loads(Path(DEFAULT_IK_CONFIG).read_text())
        self.assertEqual(raw["ik_match_table1"]["L_ee"][0], "Left_Wrist")

    def test_tracker_mount_correction_removes_roll_arc_before_filtering(self):
        config = load_ik_config()
        config["tracker_mount_height_m"] = {"left": 0.04, "right": 0.05}
        solver = self.make(config)
        centers = self.ee_poses(solver)
        rest = {}
        offsets = {}
        for index, side in enumerate(SIDES):
            pose = centers[side].copy()
            # Arbitrary mounting axes must not change the corrected centre.
            pose[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.5, index + 0.8]).as_matrix()
            height = config["tracker_mount_height_m"][side]
            pose[:3, 3] += [0, 0, height]
            offsets[side] = pose[:3, :3].T @ np.array([0, 0, -height])
            rest[side] = pose
        mapper = TrackerArmMapper(solver, left_serial="left", right_serial="right")
        snapshot = lambda poses, i: TrackerSnapshot(
            TrackerFrame(poses, None, i, i / 60.0), i, i, None)
        mapper.calibrate(snapshot(rest, 0), self.home, 0.0)
        initial_targets = solver.targets({WRIST_BODIES[s]: p for s, p in rest.items()})
        for i, angle in enumerate(np.linspace(-0.7, 0.7, 40), start=1):
            moved = {}
            for side in SIDES:
                pose = rest[side].copy()
                pose[:3, :3] = Rotation.from_euler("x", angle).as_matrix() @ pose[:3, :3]
                pose[:3, 3] = centers[side][:3, 3] - pose[:3, :3] @ offsets[side]
                moved[side] = pose
            mapper.update(snapshot(moved, i), i / 60.0)
            for side in SIDES:
                body = body_names(side)["ee"]
                np.testing.assert_allclose(mapper.raw_targets[side][:3, 3],
                                           initial_targets[body][:3, 3], atol=1e-12)
                np.testing.assert_allclose(mapper.targets[side][:3, 3],
                                           initial_targets[body][:3, 3], atol=1e-12)
        # Real translation still passes through the mapping after pivot correction.
        before = solver.targets({WRIST_BODIES[s]: p for s, p in moved.items()})
        for pose in moved.values():
            pose[:3, 3] += [0.03, 0, 0]
        after = solver.targets({WRIST_BODIES[s]: p for s, p in moved.items()})
        for body in before:
            np.testing.assert_allclose(after[body][:3, 3] - before[body][:3, 3],
                                       solver.heading.T @ [0.03, 0, 0], atol=1e-12)
            np.testing.assert_allclose(after[body][:3, :3], before[body][:3, :3])
        mapper.update(snapshot({"right": moved["right"]}, 41), 41 / 60.0)
        self.assertEqual(mapper.status["left"], "lost")
        self.assertEqual(mapper.update(snapshot(moved, 42), 2.0), {})
        self.assertEqual(mapper.status["right"], "holding")

    def test_mount_heights_validate_and_zero_preserves_mapping(self):
        for value in (-0.01, 0.21, float("nan"), float("inf")):
            config = load_ik_config()
            config["tracker_mount_height_m"] = {"left": value}
            with self.assertRaisesRegex(ValueError, "tracker_mount_height_m"):
                self.make(config)
        config = load_ik_config()
        config["tracker_mount_height_m"] = {"left": 0, "right": 0}
        a, b = self.make(), self.make(config)
        human = self.rest_pose_human(a)
        for solver in (a, b):
            solver.calibrate(human, self.home)
        for pose in human.values():
            pose[:3, :3] = Rotation.from_euler("y", 0.4).as_matrix() @ pose[:3, :3]
        for body, pose in a.targets(human).items():
            np.testing.assert_array_equal(pose, b.targets(human)[body])

    def test_elbow_preference_reduces_wrist_turn_motion_through_mapper(self):
        from scripts.evaluate_gmr_elbow import evaluate

        root = DEFAULT_IK_CONFIG.parent
        baseline = evaluate(self.model, self.arms,
                            root / "xr_trackers_to_vega_1u_stage_split_soft.json")
        config = load_ik_config(root / "xr_trackers_to_vega_1u_elbow.json")
        # This elbow-only fixture places tracker origins at wrist centres.
        # Mount compensation is tested separately with genuine rotating arcs.
        config["tracker_mount_height_m"] = {}
        candidate = evaluate(self.model, self.arms, config)
        repeated = evaluate(self.model, self.arms, config)
        for phase in ("wrist_x", "wrist_y", "wrist_z"):
            before, after = baseline["summary"][phase], candidate["summary"][phase]
            self.assertLess(after["elbow_max_m"], before["elbow_max_m"] * 0.5)
            self.assertLess(after["position_p95_m"], 0.005)
            # A soft elbow prior trades orientation accuracy for stability.
            self.assertLess(after["orientation_p95_deg"], 15)
        self.assertLess(candidate["summary"]["reach"]["position_p95_m"], 0.002)
        for row, again in zip(candidate["rows"], repeated["rows"]):
            for side in SIDES:
                np.testing.assert_array_equal(row["q"][side], again["q"][side])
                q = np.asarray(row["q"][side])
                self.assertTrue(np.all(q >= self.arms[side].lower - 1e-8))
                self.assertTrue(np.all(q <= self.arms[side].upper + 1e-8))
        loss = [r for r in candidate["rows"] if r["status"]["left"] == "lost"]
        self.assertTrue(loss)
        for row in loss[1:]:
            np.testing.assert_array_equal(row["q"]["left"], loss[0]["q"]["left"])

    def test_elbow_preference_fades_at_unreachable_target_and_recalibrates(self):
        config = DEFAULT_IK_CONFIG.with_name("xr_trackers_to_vega_1u_elbow.json")
        solver = self.make(config)
        solver.calibrate(self.rest_pose_human(solver), self.home)
        target = solver.targets(self.rest_pose_human(solver))
        target["L_ee"][:3, 3] += [3, 0, 0]
        solver._elbow_preferences(target)
        np.testing.assert_allclose(solver.elbow_tasks["left"].cost[:3], 0)
        old = solver.elbow_rest["left"][0].copy()
        changed = {s: q.copy() for s, q in self.home.items()}
        changed["left"][0] += 0.1
        solver.calibrate(self.rest_pose_human(solver), changed)
        self.assertGreater(np.linalg.norm(solver.elbow_rest["left"][0] - old), 0.001)

    def test_inward_offset_blends_bilaterally_only_on_retraction(self):
        config = load_ik_config(DEFAULT_IK_CONFIG.with_name("xr_trackers_to_vega_1u_elbow.json"))
        old_config = copy.deepcopy(config)
        for spec in old_config["elbow_preference"].values():
            spec.pop("inward_offset_m")
        baseline, shifted = self.make(old_config), self.make(config)
        for solver in (baseline, shifted):
            solver.calibrate(self.rest_pose_human(solver), self.home)
        rest = shifted.targets(self.rest_pose_human(shifted))
        for retraction, expected in ((-0.05, 0), (0, 0), (0.05, 0.0075),
                                     (0.10, 0.015), (0.20, 0.015)):
            targets = {b: p.copy() for b, p in rest.items()}
            for pose in targets.values():
                pose[0, 3] -= retraction
                pose[:3, :3] = pose[:3, :3] @ Rotation.from_euler("x", 0.4).as_matrix()
            for solver in (baseline, shifted):
                solver._elbow_preferences(targets)
            for side in SIDES:
                a = baseline.elbow_tasks[side].transform_target_to_world.translation()
                b = shifted.elbow_tasks[side].transform_target_to_world.translation()
                np.testing.assert_allclose(b - a, [0, -np.sign(a[1]) * expected, 0], atol=1e-12)
            if retraction <= 0:
                a, b = baseline.solve(targets), shifted.solve(targets)
                for side in SIDES:
                    np.testing.assert_array_equal(a[side], b[side])
        # Actual inward reaches remain finite and bounded under the soft prior.
        for amount in np.linspace(0, 0.10, 60):
            targets = {b: p.copy() for b, p in rest.items()}
            for pose in targets.values():
                pose[0, 3] -= amount
            goals = shifted.solve(targets)
            for side in SIDES:
                self.assertTrue(np.isfinite(goals[side]).all())
                self.assertTrue(np.all(goals[side] >= self.arms[side].lower - 1e-8))
                self.assertTrue(np.all(goals[side] <= self.arms[side].upper + 1e-8))

    def test_invalid_inward_parameters_are_rejected(self):
        config = load_ik_config(DEFAULT_IK_CONFIG.with_name("xr_trackers_to_vega_1u_elbow.json"))
        for key, value in (("inward_offset_m", -0.01), ("inward_offset_m", float("nan")),
                           ("inward_blend_distance_m", 0),
                           ("inward_blend_distance_m", float("inf"))):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(config)
                bad["elbow_preference"]["left"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.make(bad)

    def test_decoupled_wrist_rotation_does_not_move_upper_arm(self):
        config = Path(DEFAULT_IK_CONFIG).with_name("xr_trackers_to_vega_1u_decoupled.json")
        baseline, rotating = self.make(config), self.make(config)
        for solver in (baseline, rotating):
            solver.calibrate(self.rest_pose_human(solver), self.home)
        poses = self.ee_poses(baseline)
        for angle in np.linspace(-0.8, 0.8, 25):
            fixed, turned = {}, {}
            for side in SIDES:
                body = body_names(side)["ee"]
                pose = poses[side].copy()
                pose[:3, 3] += [0.01 * math.sin(angle), 0, 0.01]
                fixed[body] = pose
                turned[body] = pose.copy()
                turned[body][:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([angle, 0, 0]).as_matrix()
            a, b = baseline.solve(fixed), rotating.solve(turned)
            for side in SIDES:
                np.testing.assert_allclose(a[side][:4], b[side][:4], atol=1e-8)
                self.assertTrue(np.isfinite(b[side]).all())
                self.assertTrue(np.all(b[side] >= self.arms[side].lower - 1e-8))
                self.assertTrue(np.all(b[side] <= self.arms[side].upper + 1e-8))

    def test_invalid_configs_are_rejected(self):
        raw = json.loads(Path(DEFAULT_IK_CONFIG).read_text())
        cases = {
            "root": ("human_root_name", "Pelvis"),
            "scale": ("human_scale_table", {"Left_Wrist": 1.0}),
            "entry": ("ik_match_table1", {"L_ee": ["Left_Wrist", 50, 5]}),
        }
        for name, (key, value) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                bad = copy.deepcopy(raw)
                bad[key] = value
                path = Path(folder) / "bad.json"
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_ik_config(path)

    def test_calibration_pins_rest_pose_to_robot_hands(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        targets = retargeter.targets(human)
        for side, pose in self.ee_poses(retargeter).items():
            np.testing.assert_allclose(targets[body_names(side)["ee"]], pose, atol=1e-12)

    def test_wrist_motion_is_scaled_by_table_and_height(self):
        retargeter = self.make(actual_human_height=1.40)  # 0.8 of the assumed 1.75 m
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        moved = {body: pose.copy() for body, pose in human.items()}
        moved["Left_Wrist"][:3, 3] += [0.10, 0.05, -0.05]
        before, after = retargeter.targets(human), retargeter.targets(moved)
        np.testing.assert_allclose(after["L_ee"][:3, 3] - before["L_ee"][:3, 3],
                                   0.8 * np.array([0.10, 0.05, -0.05]))
        np.testing.assert_allclose(after["R_ee"], before["R_ee"])

    def test_world_rotation_of_wrist_rotates_target_about_same_axis(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        delta = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        moved = {body: pose.copy() for body, pose in human.items()}
        moved["Right_Wrist"][:3, :3] = delta @ human["Right_Wrist"][:3, :3]
        target = retargeter.targets(moved)["R_ee"]
        np.testing.assert_allclose(target[:3, :3], delta @ human["Right_Wrist"][:3, :3],
                                   atol=1e-12)

    def test_operator_heading_does_not_change_targets(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        moved = {body: pose.copy() for body, pose in human.items()}
        moved["Left_Wrist"][:3, 3] += [0.08, 0.0, 0.04]
        moved["Left_Wrist"][:3, :3] = (Rotation.from_euler("x", 20, degrees=True).as_matrix()
                                       @ moved["Left_Wrist"][:3, :3])
        retargeter.calibrate(human, self.home)
        expected = retargeter.targets(moved)
        yaw = 2.0
        retargeter.calibrate({b: _yawed(p, yaw) for b, p in human.items()}, self.home, yaw=yaw)
        turned = retargeter.targets({b: _yawed(p, yaw) for b, p in moved.items()})
        for body in expected:
            np.testing.assert_allclose(turned[body], expected[body], atol=1e-12)

    def test_local_position_offset_follows_wrist_rotation(self):
        raw = json.loads(Path(DEFAULT_IK_CONFIG).read_text())
        for table in ("ik_match_table1", "ik_match_table2"):
            raw[table]["L_ee"][3] = [0.0, 0.0, 0.1]
        retargeter = self.make(load_config_from(raw))
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        delta = Rotation.from_euler("z", 40, degrees=True).as_matrix()
        moved = {body: pose.copy() for body, pose in human.items()}
        moved["Left_Wrist"][:3, :3] = delta @ human["Left_Wrist"][:3, :3]
        robot = self.ee_poses(retargeter)["left"]
        target = retargeter.targets(moved)["L_ee"]
        offset = np.array([0.0, 0.0, 0.1])
        np.testing.assert_allclose(target[:3, 3] - robot[:3, 3],
                                   (target[:3, :3] - robot[:3, :3]) @ offset, atol=1e-12)

    def test_headset_root_cancels_head_and_body_translation(self):
        raw = json.loads(Path(DEFAULT_IK_CONFIG).read_text())
        raw["human_root_name"] = "Headset"
        retargeter = self.make(load_config_from(raw))
        human = self.rest_pose_human(retargeter)
        headset = np.eye(4)
        headset[:3, 3] = [0.0, 0.0, 1.5]
        retargeter.calibrate(human, self.home, root=headset)
        shift = np.array([0.3, -0.2, 0.0])
        walked = {body: pose.copy() for body, pose in human.items()}
        for pose in walked.values():
            pose[:3, 3] += shift
        moved_head = headset.copy()
        moved_head[:3, 3] += shift
        np.testing.assert_allclose(retargeter.targets(walked, root=moved_head)["L_ee"],
                                   retargeter.targets(human, root=headset)["L_ee"], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "headset"):
            retargeter.targets(human)

    def test_two_stage_solve_reaches_reachable_targets_for_both_arms(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        scratch = mujoco.MjData(self.model)
        rng = np.random.default_rng(3)
        goals = {side: self.home[side].copy() for side in SIDES}
        for _ in range(15):
            scratch.qpos[:] = retargeter.neutral_qpos
            for side in SIDES:
                goals[side] = np.clip(goals[side] + rng.normal(0, 0.1, 7),
                                      self.arms[side].lower, self.arms[side].upper)
                scratch.qpos[self.arms[side].qpos_ids] = goals[side]
            mujoco.mj_kinematics(self.model, scratch)
            moved = {}
            for side in SIDES:
                ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                       body_names(side)["ee"])
                pose = np.eye(4)
                pose[:3, :3] = scratch.xmat[ee].reshape(3, 3)
                pose[:3, 3] = scratch.xpos[ee]
                moved[WRIST_BODIES[side]] = pose
            targets = retargeter.targets(moved)
            solution = retargeter.solve(targets)
            self.assertEqual(set(solution), set(SIDES))
            for body, (position, rotation) in retargeter.errors(targets).items():
                self.assertLess(position, 2e-3, body)
                self.assertLess(rotation, math.radians(1.0), body)
        arm_qpos = np.concatenate([retargeter.qpos_ids[side] for side in SIDES])
        other = np.setdiff1d(np.arange(self.model.nq), arm_qpos)
        np.testing.assert_allclose(retargeter.configuration.q[other],
                                   retargeter.neutral_qpos[other])

    def test_arm_without_target_stays_frozen(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        moved = {"Left_Wrist": human["Left_Wrist"].copy()}
        moved["Left_Wrist"][:3, 3] += [0.05, 0.0, 0.05]
        solution = retargeter.solve(retargeter.targets(moved))
        self.assertEqual(set(solution), {"left"})
        np.testing.assert_allclose(
            retargeter.configuration.q[retargeter.qpos_ids["right"]], self.home["right"])
        self.assertFalse(np.allclose(solution["left"], self.home["left"]))

    def test_ik_keeps_the_elbow_out_of_hyperextension(self):
        # From a nearly straight elbow, an unbounded IK bends it back to reach this target.
        retargeter = self.make()
        retargeter.calibrate(self.rest_pose_human(retargeter), self.home)
        straight = self.home["right"].copy()
        straight[3] = 0.1
        retargeter.sync({"left": self.home["left"], "right": straight})
        bent_back = straight.copy()
        bent_back[3] = 0.8
        human = {"Right_Wrist": self.fk_pose(retargeter, "right", bent_back)}
        for _ in range(40):
            elbow = retargeter.solve(retargeter.targets(human))["right"][3]
        self.assertLessEqual(elbow, RIGHT_ARM_JOINT_LIMITS[3, 1] + 1e-6)

    def test_solve_warm_starts_from_previous_frame(self):
        retargeter = self.make()
        human = self.rest_pose_human(retargeter)
        retargeter.calibrate(human, self.home)
        moved = {body: pose.copy() for body, pose in human.items()}
        moved["Left_Wrist"][:3, 3] += [0.06, 0.0, 0.0]
        targets = retargeter.targets(moved)
        first = retargeter.solve(targets)["left"]
        second = retargeter.solve(targets)["left"]
        self.assertLess(np.linalg.norm(second - first), 1e-3)


def load_config_from(raw):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.json"
        path.write_text(json.dumps(raw))
        return load_ik_config(path)


class PoseTargetFilterTests(unittest.TestCase):
    def test_first_pose_passes_and_jumps_are_rate_limited(self):
        target_filter = PoseTargetFilter(tau_s=0.0, max_speed_m_s=1.0,
                                         max_angular_speed_rad_s=2.0)
        start = np.eye(4)
        np.testing.assert_allclose(target_filter.update(start, 0.02), start)
        jump = np.eye(4)
        jump[:3, 3] = [1.0, 0.0, 0.0]
        jump[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        out = target_filter.update(jump, 0.02)
        np.testing.assert_allclose(out[:3, 3], [0.02, 0.0, 0.0])
        self.assertAlmostEqual(
            np.linalg.norm(Rotation.from_matrix(out[:3, :3]).as_rotvec()), 0.04)
        self.assertAlmostEqual(target_filter.max_step_m, 0.02)

    def test_low_pass_converges_to_a_held_target(self):
        target_filter = PoseTargetFilter(tau_s=0.05)
        target_filter.update(np.eye(4), 0.02)
        goal = np.eye(4)
        goal[:3, 3] = [0.01, 0.0, 0.0]
        first = target_filter.update(goal, 0.02)[0, 3]
        self.assertTrue(0.0 < first < 0.01)
        for _ in range(50):
            out = target_filter.update(goal, 0.02)
        np.testing.assert_allclose(out, goal, atol=1e-6)


class TrackerArmMapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene = _Scene.load()
        cls.model, cls.arms = scene.model, scene.arms
        cls.home = {side: cls.arms[side].home.copy() for side in SIDES}

    def setUp(self):
        retargeter = GmrArmRetargeter(
            self.model, {side: arm_joint_names(side) for side in SIDES})
        retargeter.sync(self.home)
        self.ee = {side: retargeter.configuration.get_transform_frame_to_world(
            body_names(side)["ee"], "body").as_matrix() for side in SIDES}
        self.mapper = TrackerArmMapper(retargeter)
        self.revision = 0

    def snapshot(self, poses, *, received_at=None, error=None):
        self.revision += 1
        frame = TrackerFrame(poses, None, self.revision,
                             time.monotonic() if received_at is None else received_at)
        return TrackerSnapshot(frame, self.revision, self.revision, error)

    def test_calibrated_trackers_move_matching_arms(self):
        poses = {"SN-L": self.ee["left"].copy(), "SN-R": self.ee["right"].copy()}
        yaw = self.mapper.calibrate(self.snapshot(poses), self.home, time.monotonic())
        # The mirrored home hands differ by < 0.3 mm, so the wrist line is nearly lateral.
        self.assertAlmostEqual(yaw, 0.0, places=3)
        self.assertEqual(self.mapper.serials, {"left": "SN-L", "right": "SN-R"})
        moved = {serial: pose.copy() for serial, pose in poses.items()}
        moved["SN-L"][:3, 3] += [0.02, 0.0, 0.0]
        goals = self.mapper.update(self.snapshot(moved), time.monotonic())
        self.assertEqual(set(goals), set(SIDES))
        self.assertEqual(self.mapper.status, {"left": "tracking", "right": "tracking"})
        self.assertGreater(self.mapper.targets["left"][0, 3], self.ee["left"][0, 3])

    def test_lost_stale_and_uncalibrated_trackers_hold(self):
        poses = {"SN-L": self.ee["left"].copy(), "SN-R": self.ee["right"].copy()}
        self.assertEqual(self.mapper.update(self.snapshot(poses), time.monotonic()), {})
        self.mapper.calibrate(self.snapshot(poses), self.home, time.monotonic())
        goals = self.mapper.update(self.snapshot({"SN-L": poses["SN-L"]}), time.monotonic())
        self.assertEqual(set(goals), {"left"})
        self.assertEqual(self.mapper.status["right"], "lost")
        old = self.snapshot(poses, received_at=time.monotonic() - 1.0)
        self.assertEqual(self.mapper.update(old, time.monotonic()), {})
        self.assertEqual(self.mapper.status["left"], "holding")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.mapper.calibrate(old, self.home, time.monotonic())


if __name__ == "__main__":
    unittest.main()
