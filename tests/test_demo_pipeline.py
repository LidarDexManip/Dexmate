"""End-to-end check of the human-demonstration pipeline on a synthetic session.

Writes a session in the real on-disk formats (laptop binary logs, the headset's
trackingData text and side-by-side video) from a known ground truth: a right
hand that grasps and lifts a cylinder, wrist trackers on the forearms, palm
IMUs in their own world frame, a headset clock 0.35 s behind the laptop. Then
runs the stages (hands, track, ground, export) and checks them against the
truth. Depth images are rendered from the truth instead of computed by stereo
matching (``test_demo.StereoTests`` covers that).

Needs the retargeting checkout's environment (libigl, its held refinement):

    ~/Desktop/retargeting/.venv-vision/bin/python -m unittest tests.test_demo_pipeline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

HAVE = all(importlib.util.find_spec(m) for m in ("igl", "trimesh", "cv2"))
RETARGETING = Path("~/Desktop/retargeting").expanduser()

A = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # OpenXR -> robot axes
OFFSET = 0.35          # host = headset + OFFSET
LATENCY = 0.008        # PC Service relay delay of each tracker frame
HOST0 = 1.8e9
EYE_W, EYE_H, FX = 320, 240, 200.0
FPS = 30.0
STILL, ROTATE, MOVE, DEMO = (1.0, 4.0), (4.0, 14.0), (14.0, 15.0), (15.0, 19.0)
END = 20.0
X_TRUE = R.from_rotvec([0.2, -0.4, 1.1]).as_matrix()      # tracker <- wrist (straight)
Y_TRUE = R.from_rotvec([0.0, 0.0, 2.3]).as_matrix()        # world <- IMU world (heading only)
IMU_IN_WRIST = R.from_rotvec([0.1, -0.2, 0.05]).as_matrix()
D_TRUE = None  # tracker-local wrist offset, set from the still pose below

R_STILL = {"right": np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).astype(float),
           "left": np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)}
P_STILL = {"right": np.array([0.35, -0.2, 1.0]), "left": np.array([0.35, 0.2, 1.0])}


# A real half-curled right glove skeleton (Wuji SDK, metres, glove wrist frame),
# logged by the teleop on 2026-11-06.
RIGHT_GLOVE_SKELETON = np.array([
    [0.00000, 0.00000, 0.00000],
    [0.01390, 0.00844, -0.02750],
    [0.04176, 0.01380, -0.04803],
    [0.06587, 0.02285, -0.07175],
    [0.07777, 0.03719, -0.09681],
    [0.02948, -0.00587, -0.08130],
    [0.03986, 0.01968, -0.10925],
    [0.04339, 0.04031, -0.10789],
    [0.04157, 0.05260, -0.08761],
    [0.00728, -0.01046, -0.08594],
    [0.01057, 0.02248, -0.11665],
    [0.01115, 0.04512, -0.10409],
    [0.01042, 0.05460, -0.07917],
    [-0.01200, -0.00766, -0.08406],
    [-0.00285, 0.02657, -0.11226],
    [-0.00209, 0.04510, -0.09558],
    [-0.00135, 0.06408, -0.07820],
    [-0.03365, -0.00227, -0.07371],
    [-0.03186, 0.02066, -0.09558],
    [-0.03148, 0.03739, -0.08769],
    [-0.03164, 0.05241, -0.06780]])


def right_hand_keypoints():
    return RIGHT_GLOVE_SKELETON.copy()


def left_hand_keypoints():
    k = right_hand_keypoints().copy()
    k[:, 1] *= -1  # mirror the palmar axis
    return k


class Truth:
    """Ground-truth motion as functions of host time (relative to HOST0)."""

    def __init__(self):
        from wuji_teleop.demo import hands
        self.kp = {"right": right_hand_keypoints(), "left": left_hand_keypoints()}
        import trimesh
        from baselines.mesh_queries import MeshQuery
        kp = self.kp["right"]
        palm = hands.palm_pose(kp[None])[0]
        hv = hands.HandSurface(kp).vertices(kp[None])[0]
        self.radius = 0.025
        self.mesh = trimesh.creation.cylinder(radius=self.radius, height=0.12, sections=48)
        query = MeshQuery(self.mesh.vertices, self.mesh.faces)
        tips = kp[[4, 8, 12, 16, 20]]
        nz = palm[:3, 2] * np.sign(np.mean((tips - kp[0]) @ palm[:3, 2]))
        z = palm[:3, 0]
        x = np.cross(nz, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        base = kp[0] + 0.55 * (kp[9] - kp[0])

        def pose(s):
            T = np.eye(4)
            T[:3, :3] = np.column_stack([x, y, z])
            T[:3, 3] = base + s * nz
            return T

        lo, hi = 0.0, 0.2
        for _ in range(40):
            mid = (lo + hi) / 2
            T = pose(mid)
            d, _ = query.query((hv - T[:3, 3]) @ T[:3, :3])
            lo, hi = (mid, hi) if d.min() < 0.0005 else (lo, mid)
        self.T_wrist_obj = pose(hi)
        # The grasp pose of the right wrist and the object resting on the table there.
        self.grasp_wrist = np.eye(4)
        self.grasp_wrist[:3, :3] = R_STILL["right"] @ R.from_rotvec([0, 0.0, 0]).as_matrix()
        self.grasp_wrist[:3, 3] = [0.45, -0.12, 0.95]
        self.object_rest = self.grasp_wrist @ self.T_wrist_obj
        self.table_z = self.object_rest[2, 3] - self.radius

    def _demo_u(self, t):
        return float(np.clip((t - DEMO[0]) / (DEMO[1] - DEMO[0]), 0, 1))

    def bend(self, side, t):
        """Hand relative to the straight wrist: the right wrist flexes while lifting."""
        u = self._demo_u(t)
        if side != "right" or t < DEMO[0] or u <= 0.4:
            return np.eye(3)
        return R.from_rotvec([0.25 * min(1.0, (u - 0.4) / 0.4), 0, 0]).as_matrix()

    def _rotate(self, side, t):
        u = (t - ROTATE[0]) / (ROTATE[1] - ROTATE[0])
        return R.from_rotvec([1.0 * np.sin(2 * np.pi * u), 0.6 * np.sin(4 * np.pi * u),
                              0.5 * np.sin(6 * np.pi * u)]).as_matrix() @ R_STILL[side]

    def _demo(self, t):
        u = self._demo_u(t)
        T = self.grasp_wrist.copy()
        if u < 0.25:
            T[2, 3] += 0.06 * (0.25 - u) / 0.25
        elif u > 0.4:
            k = min(1.0, (u - 0.4) / 0.4)
            T[2, 3] += 0.12 * k
            T[:3, :3] = R.from_rotvec([0, 0, 0.3 * k]).as_matrix() @ T[:3, :3]
        return T

    def wrist(self, side, t):
        """World <- straight-wrist hand frame (the forearm's continuation)."""
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_STILL[side], P_STILL[side]
        if ROTATE[0] <= t <= ROTATE[1]:
            T[:3, :3] = self._rotate(side, t)
        if side == "right" and t > MOVE[0]:
            target = self._demo(min(t, DEMO[1]))
            if t < MOVE[1]:  # move from the still pose to the approach pose
                k = (t - MOVE[0]) / (MOVE[1] - MOVE[0])
                key = R.from_matrix([T[:3, :3], target[:3, :3]])
                from scipy.spatial.transform import Slerp
                T[:3, :3] = Slerp([0, 1], key)(k).as_matrix()
                T[:3, 3] = (1 - k) * T[:3, 3] + k * target[:3, 3]
            else:
                T = target
        return T

    def hand_pose(self, side, t):
        """World <- glove wrist frame, bend applied."""
        W = self.wrist(side, t)
        W[:3, :3] = W[:3, :3] @ self.bend(side, t)
        return W

    def object(self, t):
        if t >= DEMO[0] and self._demo_u(t) >= 0.25:
            return self.hand_pose("right", min(t, DEMO[1])) @ self.T_wrist_obj
        return self.object_rest

    def tracker(self, side, t):
        """Forearm tracker pose: the straight-wrist relation, independent of bend."""
        W = self.wrist(side, t)
        T = np.eye(4)
        T[:3, :3] = W[:3, :3] @ X_TRUE.T
        T[:3, 3] = W[:3, 3] - T[:3, :3] @ D_TRUE[side]
        return T

    def keypoints_world(self, side, t):
        W = self.hand_pose(side, t)
        return self.kp[side] @ W[:3, :3].T + W[:3, 3]


def to_xr_tracker(T):
    q = R.from_matrix(A.T @ T[:3, :3] @ A).as_quat()
    return np.r_[A.T @ T[:3, 3], q]


def head_pose_world():
    """Robot world <- OpenXR head frame: looking forward (+x) and 35 deg down."""
    look = R.from_rotvec([0, np.deg2rad(35), 0]).as_matrix()  # pitch down about world y
    forward, up = look @ [1.0, 0, 0], look @ [0, 0, 1.0]
    R_head = np.column_stack([np.cross(forward, up), up, -forward])  # x right, y up, z back
    T = np.eye(4)
    T[:3, :3] = R_head
    T[:3, 3] = [0.0, 0.0, 1.45]
    return T


def build_session(root, truth):
    from wuji_teleop.demo import log as demo_log
    global D_TRUE
    pc = root / "pc"
    writer = demo_log.StreamWriter(pc)
    rng = np.random.default_rng(0)
    # Tracker offset consistent with the processor's still-pose model (25 mm up, 20 mm back).
    D_TRUE = {}
    for side in ("left", "right"):
        Rt = R_STILL[side] @ X_TRUE.T
        # The processor takes the hand direction from the skeleton, levelled.
        local = truth.kp[side][9] - truth.kp[side][0]
        forward = R_STILL[side] @ (local / np.linalg.norm(local))
        forward[2] = 0
        forward /= np.linalg.norm(forward)
        D_TRUE[side] = Rt.T @ (-0.025 * np.array([0, 0, 1.0]) + 0.02 * forward)
    serial_of = {"right": "PC2A_R", "left": "PC2A_L"}
    serials = ["PC2A_L", "PC2A_R"]
    # Gloves and IMUs at 100 Hz (SDK stamps are host-synchronized sample times).
    for k, t in enumerate(np.arange(0.2, END - 0.2, 0.01)):
        for s, side in enumerate(("left", "right")):
            row = np.zeros((), demo_log.GLOVE_DTYPE)
            row["t"] = HOST0 + t + 0.004
            row["stamp_us"] = int((HOST0 + t) * 1e6)
            row["seq"], row["side"] = k, s
            row["keypoints"] = truth.kp[side] + rng.normal(0, 0.0003, (21, 3))
            row["orientation"][:, 3] = 1
            row["confidence"] = 0.95
            writer.append("glove", row)
            imu = np.zeros((), demo_log.IMU_DTYPE)
            imu["t"] = HOST0 + t + 0.003
            imu["stamp_us"] = int((HOST0 + t + 0.005) * 1e6)
            imu["side"] = s
            Rw = truth.hand_pose(side, t + 0.005)[:3, :3]
            imu["quat"] = R.from_matrix(Y_TRUE.T @ Rw @ IMU_IN_WRIST).as_quat()
            writer.append("imu", imu)
    # PC Service tracker frames at 90 Hz, received LATENCY after the headset sampled them.
    for t in np.arange(0.3, END - 0.3, 1 / 90):
        row = np.zeros((), demo_log.XR_DTYPE)
        row["t"] = HOST0 + t + LATENCY + abs(rng.normal(0, 0.002))
        row["motion_ns"] = int((HOST0 + t - OFFSET) * 1e9)
        row["count"] = 2
        row["serial"] = [-1] * demo_log.MAX_TRACKERS
        for slot, side in enumerate(("left", "right")):
            row["serial"][slot] = serials.index(serial_of[side])
            row["tracker"][slot] = to_xr_tracker(truth.tracker(side, t))
        row["headset"] = np.r_[0, 1.45, 0, 0, 0, 0, 1]
        writer.append("xr", row)
    writer.close()
    events = [{"t": HOST0 + 0.1, "label": "connected"},
              {"t": HOST0 + STILL[0], "label": "still_start"}, {"t": HOST0 + STILL[1], "label": "still_end"},
              {"t": HOST0 + ROTATE[0], "label": "rotate_start"}, {"t": HOST0 + ROTATE[1], "label": "rotate_end"},
              {"t": HOST0 + DEMO[0], "label": "demo_start"}, {"t": HOST0 + DEMO[1], "label": "demo_end"}]
    meta = {"format": "wuji_demo_pc_v1", "events": events, "tracker_serials": serials,
            "gloves": {side: {"serial": f"WG_{side}", "side": side, "address": "x", "firmware": "1",
                              "user_id": "", "imu_in_wrist_rotation": IMU_IN_WRIST.tolist()}
                       for side in ("left", "right")}}
    (pc / "meta.json").write_text(json.dumps(meta))
    # Headset recording: app lines at 72 Hz, video at 30 fps from host 0.5 s.
    hs = root / "headset"
    hs.mkdir()
    Thead = head_pose_world()
    M = np.eye(4)
    M[:3, :3] = A
    head_xr = np.linalg.inv(M) @ Thead
    head7 = np.r_[head_xr[:3, 3], R.from_matrix(head_xr[:3, :3]).as_quat()]
    left = np.eye(4)
    left[:3, 3] = [-0.032, 0.0, -0.05]
    right = np.eye(4)
    right[:3, 3] = [0.032, 0.0, -0.05]
    video_start = 0.5
    header = {"notice": "first frame", "timeStampNs": int((HOST0 + video_start - OFFSET) * 1e9),
              "cameraExtrinsics": ",".join(map(str, left.ravel())) + "|" + ",".join(map(str, right.ravel())),
              "cameraIntrinsics": f"{FX},{FX},{EYE_W / 2},{EYE_H / 2}"}
    lines = [json.dumps(header)]
    for t in np.arange(0.4, END - 0.4, 1 / 72):
        motion = {"len": 2, "joints": [{"sn": serial_of[side], "p": ",".join(map(str, to_xr_tracker(truth.tracker(side, t))))}
                                        for side in ("left", "right")]}
        # predictTime runs on the headset's display clock (seconds since boot).
        lines.append(json.dumps({"predictTime": (t - OFFSET + 5000.0) * 1e6, "timeStampNs": int((HOST0 + t - OFFSET) * 1e9),
                                 "Head": {"pose": ",".join(map(str, head7)), "status": 3}, "Motion": motion}))
    (hs / "trackingData_20261111_120000.txt").write_text("\n".join(lines) + "\n")
    frames = int((END - 0.5 - video_start) * FPS)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=gray:s={2 * EYE_W}x{EYE_H}:r={FPS}", "-frames:v", str(frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(hs / "CameraRecord_20261111_120000.mp4")],
                   check=True)
    return video_start


def render_frames(session, truth):
    """Depth and RGB of the truth at every demonstration frame, from the timeline's cameras."""
    import cv2
    import trimesh
    from wuji_teleop.demo import hands, stereo
    tl = session.timeline()
    K = tl["K"]
    W, H = (int(v) for v in tl["image_size"])
    session.frames.mkdir(parents=True, exist_ok=True)
    obj_pts = trimesh.sample.sample_surface_even(truth.mesh, 40000, seed=1)[0]
    surface = hands.HandSurface(truth.kp["right"])
    rendered = []
    for i, t_abs in enumerate(tl["frame_utc"]):
        t = t_abs - HOST0
        if not (DEMO[0] - 0.4 <= t <= DEMO[1] + 0.2):
            continue
        clouds = [obj_pts @ truth.object(t)[:3, :3].T + truth.object(t)[:3, 3]]
        for side in ("left", "right"):
            kp = truth.keypoints_world(side, t)
            mesh = trimesh.Trimesh(surface.vertices(kp[None])[0], surface.faces, process=False)
            clouds.append(trimesh.sample.sample_surface(mesh, 40000, seed=2)[0])
        g = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 260), np.linspace(-0.5, 0.4, 260)), -1).reshape(-1, 2)
        clouds.append(np.column_stack([g, np.full(len(g), truth.table_z)]))
        P = np.concatenate(clouds)
        Tcw = np.linalg.inv(tl["T_world_camera"][i])
        c = P @ Tcw[:3, :3].T + Tcw[:3, 3]
        c = c[c[:, 2] > 0.05]
        u = np.rint(c[:, 0] / c[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v = np.rint(c[:, 1] / c[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth = np.full(H * W, np.inf)
        np.minimum.at(depth, v[ok] * W + u[ok], c[ok, 2])
        depth = depth.reshape(H, W)
        depth[~np.isfinite(depth)] = 0
        depth = cv2.morphologyEx(depth.astype(np.float32), cv2.MORPH_CLOSE, np.ones((3, 3)))
        rgb = np.full((H, W, 3), 128, np.uint8)
        stereo.save_frame(session.frames, i, rgb, depth)
        rendered.append(i)
    return rendered


@unittest.skipUnless(HAVE and (RETARGETING / "baselines" / "held_refinement.py").exists(),
                     "needs the retargeting checkout and its .venv-vision")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        if str(RETARGETING) not in sys.path:
            sys.path.insert(0, str(RETARGETING))
        from wuji_teleop.demo import process
        cls.tmp = Path(tempfile.mkdtemp(prefix="demo-pipeline-"))
        cls.truth = Truth()
        build_session(cls.tmp, cls.truth)
        cls.session = process.Session(cls.tmp)
        cls.args = process.main.__globals__["argparse"].Namespace(
            clock_guess=0.0, frame_time="auto", extrinsic="head_from_camera", intrinsics="auto",
            camera_axes="openxr", depth_scale=1.0, left_tracker=None, right_tracker=None,
            mount_height=0.025, mount_back=0.02, hand_orientation="imu", calibration_from=None,
            mesh=None, mesh_scale=None, init=None, refine_init=True, variant="object-only",
            max_nfev=20, penetration_weight=5.0, holds=None, side="both", ungrounded=False, fps=None)
        process.stage_hands(cls.session, cls.args)
        cls.rendered = render_frames(cls.session, cls.truth)

    @classmethod
    def tearDownClass(cls):
        if not __import__("os").environ.get("KEEP_DEMO_TEST"):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_1_clock_and_calibration(self):
        cal = self.session.calibration()
        # The tracker relay delay is indistinguishable from clock offset.
        self.assertAlmostEqual(cal["clock_offset_s"], OFFSET + LATENCY, delta=0.004)
        self.assertEqual(cal["trackers"], {"left": "PC2A_L", "right": "PC2A_R"})
        for side in ("left", "right"):
            X = np.asarray(cal["wrists"][side]["X"])
            self.assertLess(np.rad2deg(R.from_matrix(X.T @ X_TRUE).magnitude()), 1.0)
            np.testing.assert_allclose(cal["wrists"][side]["offset"], D_TRUE[side], atol=0.002)

    def test_2_world_hands(self):
        tl = self.session.timeline()
        errors = []
        for side in ("left", "right"):
            for i in np.flatnonzero(tl[f"valid_{side}"])[::7]:
                truth = self.truth.keypoints_world(side, tl["frame_utc"][i] - HOST0)
                errors.append(np.linalg.norm(tl[f"keypoints_{side}"][i] - truth, axis=1).max())
        # Residual: the tracker relay delay folded into the clock offset (8 ms of motion).
        self.assertLess(np.median(errors), 0.004)
        self.assertLess(np.percentile(errors, 95), 0.012)

    def test_3_track_ground_export(self):
        from wuji_teleop.demo import process
        tl = self.session.timeline()
        first = self.rendered[0]
        T0 = self.truth.object(tl["frame_utc"][first] - HOST0).copy()
        T0[:3, 3] += [0.006, -0.004, 0.003]
        T0[:3, :3] = R.from_rotvec([0.05, 0.0, 0.04]).as_matrix() @ T0[:3, :3]
        mesh_path = self.tmp / "cylinder.obj"
        self.truth.mesh.export(mesh_path)
        (self.session.out / "object_init.json").write_text(json.dumps(
            {"frame": first, "mesh": str(mesh_path), "mesh_scale": 1.0, "T_world_object": T0.tolist()}))
        process.stage_track(self.session, self.args)
        track = np.load(self.session.out / "object_track.npz")
        err = [np.linalg.norm(track["poses"][i, :3, 3] - self.truth.object(tl["frame_utc"][i] - HOST0)[:3, 3])
               for i in self.rendered]
        self.assertLess(np.median(err), 0.005)
        self.assertLess(np.max(err), 0.03)

        process.stage_ground(self.session, self.args)
        report = json.loads((self.session.out / "ground_report.json").read_text())
        grounded = [h for h in report["holds"] if "skipped" not in h]
        self.assertTrue(grounded, report)
        self.assertEqual(grounded[0]["side"], "right")
        g = np.load(self.session.out / "grounded.npz")["object_poses"]
        a, b = grounded[0]["frames"]
        outside = [i for i in self.rendered if i < a or i > b]
        np.testing.assert_allclose(g[outside], track["poses"][outside])
        self.assertLessEqual(grounded[0]["after"]["max_penetration_mm"],
                             grounded[0]["before"]["max_penetration_mm"] + 0.5)

        process.stage_export(self.session, self.args)
        from baselines.input import load_sequence
        records, scale, info = load_sequence(self.session.out / "retarget_input", "right")
        self.assertEqual(info["object_frames"], len(self.rendered))
        self.assertGreater(info["valid_hand_frames"], 0.8 * len(self.rendered))


if __name__ == "__main__":
    unittest.main()
