"""Unit tests for human demonstration capture and processing (wuji_teleop.demo).

The recorder, headset parser, clock alignment, hands and tracker tests run in
this repository's .venv; the stereo and hold tests need OpenCV / libigl and
run in the retargeting checkout's .venv-vision (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from wuji_teleop.demo import (align, depth_align, hands, headset, headset_stream, log as demo_log, object_track, record,
                              stereo_record, stream_sync)

HAVE_CV2 = importlib.util.find_spec("cv2") is not None
HAVE_IGL = importlib.util.find_spec("igl") is not None
HAVE_MPL = importlib.util.find_spec("matplotlib") is not None
RETARGETING = Path("~/Desktop/retargeting").expanduser()

# A real half-curled right glove skeleton (metres, glove wrist frame).
from tests.test_demo_pipeline import RIGHT_GLOVE_SKELETON  # noqa: E402


class LogTests(unittest.TestCase):
    def test_round_trip_drops_a_cut_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = demo_log.StreamWriter(tmp)
            for k in range(3):
                row = np.zeros((), demo_log.IMU_DTYPE)
                row["t"], row["side"], row["quat"] = k, 1, [0, 0, 0, 1]
                writer.append("imu", row)
            writer.close()
            path = Path(tmp) / "imu.bin"
            path.write_bytes(path.read_bytes()[:-5])
            rows = demo_log.load(tmp, "imu")
            self.assertEqual(rows["t"].tolist(), [0.0, 1.0])
            self.assertEqual(len(demo_log.load(tmp, "glove")), 0)


class _Vec:
    def __init__(self, *v):
        self.x, self.y, self.z = v[:3]
        if len(v) > 3:
            self.w = v[3]

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeGloveSource:
    def __init__(self, serial, user_id, *, sdk=None, device_name=None, address=None):
        self.side, self.serial, self.address, self.firmware, self.user_id = "right", "WG1", address, "2.0", ""
        self.closed = False

    def palm_imu_mount_rotation(self):
        return np.eye(3)

    def palm_imu_resource(self):
        return _Obj(subscribe_with_callback=lambda cb, on_error=None: _Obj(close=lambda: None))

    def close(self):
        self.closed = True


class FakeXrSdk:
    def __init__(self):
        self.ts = 0
        self.poses = [[0.1, 1.0, -0.3, 0, 0, 0, 1], [0.0, 1.0, -0.3, 0, 0, 0, 1]]

    def init(self): pass
    def close(self): pass
    def num_motion_data_available(self): return 2
    def get_motion_timestamp_ns(self): return self.ts
    def get_motion_tracker_serial_numbers(self): return ["SN_A", "SN_B"]
    def get_motion_tracker_pose(self): return self.poses
    def get_headset_pose(self): return [0, 1.6, 0, 0, 0, 0, 1]
    def get_time_stamp_ns(self): return 5
    def get_left_hand_is_active(self): return False
    def get_right_hand_is_active(self): return False


class RecordTests(unittest.TestCase):
    def test_glove_logs_every_skeleton_and_emf_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = demo_log.StreamWriter(tmp)
            glove = record.LoggedGlove(writer, "right", "1.2.3.4:5", "", connect=FakeGloveSource)
            joints = [_Obj(pose=_Obj(position=_Vec(*p), orientation=_Vec(0, 0, 0, 1)), confidence=0.9)
                      for p in RIGHT_GLOVE_SKELETON]
            glove._log_skeleton(_Obj(joints=joints, header=_Obj(timestamp_us=1_000_000, seq=7)))
            glove._log_emf(_Obj(poses=[_Obj(confidence=0.8, pose=_Obj(position=(0.1, 0.2, 0.3),
                                                                     orientation=_Vec(0, 0, 0, 1)))] * 5))
            glove.close()
            writer.close()
            g = demo_log.load(tmp, "glove")
            self.assertEqual(len(g), 1)
            np.testing.assert_allclose(g["keypoints"][0], RIGHT_GLOVE_SKELETON, atol=1e-6)
            self.assertEqual((g["seq"][0], g["side"][0], g["stamp_us"][0]), (7, 1, 1_000_000))
            self.assertEqual(len(demo_log.load(tmp, "emf")), 1)
            self.assertEqual(glove.meta()["serial"], "WG1")

    def test_xr_poller_logs_only_new_frames_with_a_serial_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = demo_log.StreamWriter(tmp)
            sdk = FakeXrSdk()
            poller = record.XrPoller.__new__(record.XrPoller)
            poller.__dict__.update(sdk=sdk, writer=writer, serials={}, frames=0, last_seen={},
                                   headset_seen=None, last_error=None, _last_key=None)
            self.assertTrue(poller.poll())
            self.assertFalse(poller.poll())  # same frame again
            sdk.ts = 10
            sdk.poses = [[0.2, 1.0, -0.3, 0, 0, 0, 1], sdk.poses[1]]
            self.assertTrue(poller.poll())
            writer.close()
            rows = demo_log.load(tmp, "xr")
            self.assertEqual(len(rows), 2)
            self.assertEqual(poller.serials, {"SN_A": 0, "SN_B": 1})
            self.assertEqual(rows["serial"][1][:2].tolist(), [0, 1])
            self.assertAlmostEqual(rows["tracker"][1][0][0], 0.2)


def _tracking_file(path, n=5, cut=True):
    header = {"timeStampNs": 1_000_000_000, "cameraIntrinsics": "400,200,320,240",
              "cameraExtrinsics": ",".join(map(str, np.eye(4).ravel())) + "|"
              + ",".join(map(str, np.eye(4).ravel()))}
    lines = [json.dumps(header)]
    for i in range(n):
        lines.append(json.dumps({"timeStampNs": 1_000_000_000 + i * 10_000_000, "predictTime": 7e9 + i * 1e4,
                                 "Head": {"pose": "0,1.6,0,0,0,0,1"},
                                 "Motion": {"len": 1, "joints": [{"sn": "T1", "p": f"{i * 0.01},1,0,0,0,0,1"},
                                                                 {"sn": "STALE", "p": "9,9,9,0,0,0,1"}]}}))
    text = "\n".join(lines) + "\n"
    if cut:
        text += '{"timeStampNs": 12'
    Path(path).write_text(text)


class HeadsetTests(unittest.TestCase):
    def test_reads_lines_and_ignores_stale_tracker_entries_and_a_cut_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tracking_file(Path(tmp) / "trackingData_x.txt")
            rec = headset.read_tracking(Path(tmp) / "trackingData_x.txt")
            self.assertEqual(len(rec.t_ns), 5)
            self.assertEqual(list(rec.trackers), ["T1"])
            self.assertAlmostEqual(rec.trackers["T1"][3, 0], 0.03)
            left, right = rec.extrinsics()
            np.testing.assert_allclose(left, np.eye(4))

    def test_intrinsics_over_a_double_width_frame_are_halved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tracking_file(Path(tmp) / "t.txt")
            rec = headset.read_tracking(Path(tmp) / "t.txt")
            cam = headset.stereo_camera(rec, 1280, 480)
            self.assertEqual((cam.width, cam.height), (640, 480))
            self.assertAlmostEqual(cam.K_left[0, 0], 200)
            # OpenXR camera axes: the OpenCV optical axis (+z) is the head's -z.
            np.testing.assert_allclose(cam.T_head_left[:3, 2], [0, 0, -1])

    def test_frame_time_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _tracking_file(Path(tmp) / "t.txt")
            rec = headset.read_tracking(Path(tmp) / "t.txt")
            t, mode = headset.frame_times_utc(rec, np.array([0.0, 1 / 30]))
            self.assertEqual(mode, "first-image")
            np.testing.assert_allclose(t, [1.0, 1.0 + 1 / 30])
            t, mode = headset.frame_times_utc(rec, rec.predict_us[:2] / 1e6)
            self.assertEqual(mode, "predict")
            np.testing.assert_allclose(t, rec.seconds[:2])


class AlignTests(unittest.TestCase):
    def test_recovers_the_clock_offset_from_shared_trackers(self):
        th = np.arange(0, 20, 1 / 72)
        xyz = np.column_stack([0.2 * np.sin(1.3 * th) + 0.05 * np.sin(5.1 * th), np.cos(0.7 * th), 0 * th])
        raw = np.column_stack([xyz, np.zeros((len(th), 3)), np.ones(len(th))])
        tp = np.arange(0.5, 19.5, 1 / 90)
        xp = np.column_stack([np.interp(tp - 0.4321, th, xyz[:, k]) for k in range(3)])
        pc = np.column_stack([xp, np.zeros((len(tp), 3)), np.ones(len(tp))])
        offset, info = align.tracker_offset(th, {"S": raw}, {"S": (tp, pc)})
        self.assertAlmostEqual(offset, 0.4321, delta=0.0015)
        self.assertLess(info["position_rms_m"], 0.002)


class HandsTests(unittest.TestCase):
    def test_axyb_recovers_mounting_and_imu_world(self):
        X, Y = R.random(random_state=1).as_matrix(), R.random(random_state=2).as_matrix()
        B = R.random(100, random_state=3).as_matrix()
        A = np.array([Y @ b @ X.T for b in B])
        Xs, Ys, _ = hands.solve_axyb(A, B)
        self.assertLess(R.from_matrix(Xs.T @ X).magnitude(), 1e-6)
        self.assertLess(R.from_matrix(Ys.T @ Y).magnitude(), 1e-6)

    def test_capsule_surface_is_closed_and_outward(self):
        import trimesh
        surface = hands.HandSurface(RIGHT_GLOVE_SKELETON)
        v = surface.vertices(RIGHT_GLOVE_SKELETON[None])[0]
        parts = trimesh.Trimesh(v, surface.faces, process=False).split(only_watertight=False)
        self.assertEqual(len(parts), 16)
        self.assertTrue(all(p.is_watertight and p.volume > 0 for p in parts))

    def test_world_hand_places_the_skeleton_with_the_imu_orientation(self):
        times = np.linspace(0, 1, 51)
        X = R.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        Y = R.from_rotvec([0, 0, 1.0]).as_matrix()
        offset = np.array([0.01, 0.0, -0.02])
        T = np.repeat(np.eye(4)[None], len(times), 0)
        T[:, :3, 3] = np.column_stack([times, 0 * times, 1 + 0 * times])
        bend = R.from_rotvec([0.3, 0, 0]).as_matrix()
        R_hand = T[0, :3, :3] @ X @ bend
        imu = R.from_matrix(np.repeat((Y.T @ R_hand)[None], len(times), 0))
        kp = np.repeat(RIGHT_GLOVE_SKELETON[None], len(times), 0)
        cal = hands.WristCalibration(X, Y, offset, np.eye(3), {})
        out = hands.world_hand(times, times, T, times, imu, times, kp, np.ones((len(times), 21)), cal)
        self.assertTrue(out.valid.all())
        expected = RIGHT_GLOVE_SKELETON @ R_hand.T + T[25, :3, 3] + offset
        np.testing.assert_allclose(out.keypoints[25], expected, atol=1e-9)
        self.assertAlmostEqual(out.wrist_angle_deg[25], np.rad2deg(0.3), places=6)

    def test_interpolation_marks_gaps_invalid(self):
        t = np.array([0.0, 0.05, 0.5, 0.55])
        x = np.arange(4.0)[:, None]
        out, valid = hands.interp_array(t, x, np.array([0.02, 0.3, 0.52, 0.7]), max_gap=0.1)
        self.assertEqual(valid.tolist(), [True, False, True, False])


class ObjectTrackTests(unittest.TestCase):
    def setUp(self):
        import trimesh
        self.mesh = trimesh.creation.box(extents=[0.06, 0.04, 0.10])
        self.model = object_track.ObjectModel(self.mesh)

    def test_icp_recovers_a_small_offset(self):
        import trimesh
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec([0.3, 0.2, 0.1]).as_matrix()
        T[:3, 3] = [0.02, -0.01, 0.5]
        pts = trimesh.sample.sample_surface_even(self.mesh, 3000, seed=5)[0] @ T[:3, :3].T + T[:3, 3]
        scene = pts[pts[:, 2] < T[2, 3]]
        D = np.eye(4)
        D[:3, :3] = R.from_rotvec([0.05, -0.03, 0.04]).as_matrix()
        D[:3, 3] = [0.005, -0.004, 0.006]
        fit = object_track.icp(self.model, scene, T @ D)
        self.assertLess(np.linalg.norm(fit.pose[:3, 3] - T[:3, 3]), 5e-4)

    def test_table_is_removed_but_not_an_object_face(self):
        g = np.stack(np.meshgrid(np.linspace(-0.2, 0.2, 60), np.linspace(-0.2, 0.2, 60)), -1).reshape(-1, 2)
        table = np.column_stack([g, np.zeros(len(g))])
        box_top = np.column_stack([np.random.default_rng(0).uniform(-0.03, 0.03, (300, 2)), np.full(300, 0.04)])
        mask = object_track.support_plane_mask(np.concatenate([table, box_top]), self.model.radius)
        self.assertTrue(mask[:len(table)].all())
        self.assertFalse(mask[len(table):].any())
        self.assertFalse(object_track.support_plane_mask(box_top, self.model.radius).any())

    def test_grasp_contacts_counts_regions_near_the_surface(self):
        kp = np.full((21, 3), 1.0)  # far away
        T = np.eye(4)
        kp[8] = [0.0, 0.025, 0.0]   # index tip 5 mm outside the 4 cm face
        kp[4] = [0.0, -0.03, 0.0]   # thumb tip 10 mm outside the other face
        self.assertEqual(object_track.grasp_contacts(self.model, T, kp), 2)


@unittest.skipUnless(HAVE_IGL and (RETARGETING / "baselines").exists(), "needs libigl and retargeting")
class HoldTests(unittest.TestCase):
    def test_hold_reaching_the_end_keeps_its_frames(self):
        from wuji_teleop.demo import ground
        n = 60
        times = np.arange(n) / 30
        d = np.full((n, 6), 0.05)
        d[30:, 1] = d[30:, 2] = 0.002  # index and middle touch from frame 30 to the end
        d[:5] = np.inf                 # no object for the first frames
        poses = np.repeat(np.eye(4)[None], n, 0)
        poses[30:, 2, 3] = np.linspace(0, 0.1, n - 30)
        holds = ground.detect_holds(times, d, poses, poses)
        self.assertEqual(len(holds), 1)
        h = holds[0]
        self.assertEqual((h["start"], h["end"], h["padding_before"], h["padding_after"]), (30, 59, 10, 0))
        self.assertEqual(h["regions"], ["index", "middle"])


@unittest.skipUnless(HAVE_CV2, "needs OpenCV (retargeting .venv-vision)")
class StereoTests(unittest.TestCase):
    def test_depth_of_a_textured_plane(self):
        from wuji_teleop.demo import stereo
        w, h, f, baseline, z = 320, 240, 200.0, 0.064, 0.6
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        left_T, right_T = np.eye(4), np.eye(4)
        right_T[0, 3] = baseline
        cam = headset.StereoCamera(w, h, K, K.copy(), left_T, right_T)
        rng = np.random.default_rng(0)
        texture = (rng.random((h // 4, (w + 64) // 4)) * 255).astype(np.uint8)
        import cv2
        texture = cv2.resize(texture, ((w + 64), h), interpolation=cv2.INTER_NEAREST)
        disparity = f * baseline / z  # 21.3 px
        left = np.dstack([texture[:, 32:32 + w]] * 3)
        # right(u) = left(u + disparity) = texture(u + 32 + disparity)
        M = np.float32([[1, 0, -(32 + disparity)], [0, 1, 0]])
        right = np.dstack([cv2.warpAffine(texture, M, (w, h), flags=cv2.INTER_LINEAR)] * 3)
        sd = stereo.StereoDepth(cam, scale=1.0, min_depth=0.2)
        self.assertAlmostEqual(sd.rect.baseline, baseline, places=6)
        l, r = sd.rectify(left, right)
        depth = sd.depth(l, r)
        inner = depth[40:-40, 80:-40]
        self.assertGreater((inner > 0).mean(), 0.8)
        self.assertAlmostEqual(float(np.median(inner[inner > 0])), z, delta=0.01)


def _h264(width, height, frames):
    """A short H.264 Annex-B stream, packetized as the headset's encoder sends it:
    SPS/PPS alone first, then one picture per packet."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=30",
         "-frames:v", str(frames), "-c:v", "libx264", "-bf", "0", "-g", "5", "-threads", "1",
         "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"], capture_output=True, check=True).stdout
    starts = [i for i in range(len(raw) - 3) if raw[i:i + 3] == b"\x00\x00\x01"]
    cuts = [i - 1 if i and raw[i - 1] == 0 else i for i in starts]
    nals = [raw[a:b] for a, b in zip(cuts, cuts[1:] + [len(raw)])]
    packets, cur = [], b""
    for nal in nals:
        kind = nal[nal.index(b"\x00\x00\x01") + 3] & 0x1F
        if kind in (1, 5):
            if not packets:
                packets.append(cur)
                cur = b""
            packets.append(cur + nal)
            cur = b""
        else:
            cur += nal
    return raw, packets


class FakeHeadset:
    """The XRoboToolkit app's side: takes control messages on its port (parsed as
    the app's C# does) and, on OPEN_CAMERA, connects back and streams packets."""

    def __init__(self, packets, interval=1 / 30):
        self.packets, self.interval = packets, interval
        self.server = socket.create_server(("127.0.0.1", 0))
        self.port = self.server.getsockname()[1]
        self.commands, self.requests = [], []
        self.thread = threading.Thread(target=self._control, daemon=True)
        self.thread.start()

    @staticmethod
    def _read(conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _control(self):
        conn, _ = self.server.accept()
        with conn:
            while (head := self._read(conn, 4)) is not None:
                body = self._read(conn, struct.unpack(">I", head)[0])
                n = struct.unpack_from("<i", body)[0]
                command = body[4:4 + n].decode()
                m = struct.unpack_from("<i", body, 4 + n)[0]
                data = body[8 + n:8 + n + m]
                self.commands.append(command)
                if command == "OPEN_CAMERA":
                    self.assert_magic = data[:3]
                    fields = struct.unpack_from("<7i", data, 3)
                    k = data[31]
                    camera = data[32:32 + k].decode()
                    j = data[32 + k]
                    ip = data[33 + k:33 + k + j].decode()
                    self.requests.append((*fields, camera, ip))
                    threading.Thread(target=self._send, args=(ip, fields[6]), daemon=True).start()
        self.server.close()

    def _send(self, ip, port):
        with socket.create_connection((ip, port)) as out:
            for packet in self.packets:
                out.sendall(struct.pack(">I", len(packet)) + packet)
                time.sleep(self.interval)
            time.sleep(0.5)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class HeadsetStreamTests(unittest.TestCase):
    def test_messages_round_trip(self):
        req = headset_stream.CameraRequest(1280, 480, 15, 12345, "192.168.1.5")
        body = headset_stream.encode_message("OPEN_CAMERA", req.encode())
        self.assertEqual(struct.unpack(">I", body[:4])[0], len(body) - 4)
        command, data = headset_stream.decode_message(body[4:])
        self.assertEqual(command, "OPEN_CAMERA")
        self.assertEqual(headset_stream.CameraRequest.decode(data), req)
        self.assertEqual(headset_stream.nal_types(b"\x00\x00\x00\x01\x67ab\x00\x00\x01\x68c\x00\x00\x01\x65d"),
                         [7, 8, 5])

    def test_stream_from_a_fake_headset(self):
        width, height, frames = 320, 120, 12
        raw, packets = _h264(width, height, frames)
        self.assertEqual(len(packets), frames + 1)
        fake = FakeHeadset(packets)
        got = []
        with tempfile.TemporaryDirectory() as tmp:
            req = headset_stream.CameraRequest(width, height, 30, port=0, ip="127.0.0.1")
            stream = headset_stream.HeadsetStream("127.0.0.1", req, control_port=fake.port,
                                                  record=Path(tmp) / "run.1",
                                                  on_frame=lambda f, t: got.append((f, t)), log=lambda m: None,
                                                  reset_pause=0.05)
            stream.start()
            self.assertTrue(stream.connected.wait(5))
            deadline = time.monotonic() + 10
            while len(got) < frames - 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            # ffmpeg holds at most one picture while the stream runs.
            self.assertGreaterEqual(len(got), frames - 1)
            stream.stop()
            self.assertEqual(len(got), frames)
            fake.thread.join(3)
            self.assertEqual(fake.commands, ["CLOSE_CAMERA", "OPEN_CAMERA", "CLOSE_CAMERA"])
            self.assertEqual(fake.assert_magic, b"\xca\xfe\x01")
            self.assertEqual(fake.requests[0], (width, height, 30, 5242880, 0, 2, req.port, "VR", "127.0.0.1"))
            self.assertEqual(stream.size, (width, height))
            reference = subprocess.run(["ffmpeg", "-v", "error", "-f", "h264", "-i", "pipe:0", "-f", "rawvideo",
                                        "-pix_fmt", "bgr24", "pipe:1"], input=raw, capture_output=True,
                                       check=True).stdout
            reference = np.frombuffer(reference, np.uint8).reshape(frames, height, width, 3)
            for (frame, _), ref in zip(got, reference):
                np.testing.assert_array_equal(frame, ref)
            times = [t for _, t in got]
            self.assertEqual(times, sorted(times))
            self.assertEqual((Path(tmp) / "run.1.h264").read_bytes(), b"".join(packets))
            rows = (Path(tmp) / "run.1.times.csv").read_text().splitlines()
            self.assertEqual(len(rows), len(packets) + 1)
            self.assertEqual(sum(int(r.split(",")[3]) for r in rows[1:]), frames)

    def test_unreachable_headset(self):
        with socket.create_server(("127.0.0.1", 0)) as s:
            port = s.getsockname()[1]
        stream = headset_stream.HeadsetStream("127.0.0.1", headset_stream.CameraRequest(port=0),
                                              control_port=port, log=lambda m: None)
        with self.assertRaises(ConnectionError):
            stream.start(timeout=1)


# --------------------------------------------------------------- live stereo sync

def _head_motion(t):
    """Head orientation (OpenXR) shaking left-right and nodding, as quaternions."""
    yaw = np.deg2rad(25) * np.sin(2 * np.pi * 0.9 * t) * (t < 6)
    pitch = np.deg2rad(10) * np.sin(2 * np.pi * 1.3 * t + 0.4) * (t > 3)
    return R.from_euler("YX", np.column_stack([yaw, pitch])).as_quat()


R_HEAD_CAM = R.from_euler("x", -8, degrees=True).as_matrix() @ headset.XR_CAMERA_TO_CV  # camera tilted down


class StreamSyncTests(unittest.TestCase):
    def test_xr_stamps_map_to_host_time(self):
        rng = np.random.default_rng(1)
        true = 1.8e9 + np.arange(0, 60, 1 / 90)
        stamp = true - 0.21 + 2e-6 * (true - true[0])           # headset clock: 210 ms ahead, 2 ppm
        arrival = true + 0.003 + rng.exponential(0.004, len(true))  # relay delay, only ever late
        mapped, info = stream_sync.xr_host_times(arrival, (stamp * 1e9).astype(np.int64))
        err = mapped - true
        self.assertLess(np.abs(err - 0.003).max(), 0.002)
        self.assertLess(np.std(err), 0.0005)
        self.assertAlmostEqual(info["offset_s"], 0.213, delta=0.002)

    def test_latency_and_camera_rotation_from_head_motion(self):
        rng = np.random.default_rng(2)
        t_head = np.arange(0, 10, 1 / 90)
        head = stream_sync.HeadOrientation(t_head, _head_motion(t_head))
        latency, period = 0.137, 1 / 20
        t_cap = 0.5 + np.arange(150) * period
        t_packet = t_cap + latency + rng.exponential(0.006, len(t_cap))
        # The image sees the camera's rotation between captures, in camera axes.
        rel = head.relative(t_cap[:-1], t_cap[1:])
        image = rel @ R_HEAD_CAM  # head = R_hc @ camera
        image += rng.normal(0, np.deg2rad(0.03), image.shape)
        lags, corr, _ = stream_sync.latency_curve(t_packet[:-1], t_packet[1:], image, head)
        found, peak = stream_sync.best_latency(lags, corr)
        self.assertAlmostEqual(found, latency + 0.006, delta=0.01)
        self.assertGreater(peak, 0.9)
        head_rot = head.relative(t_packet[:-1] - found, t_packet[1:] - found)
        R_hc, resid, n = stream_sync.camera_in_head(image, head_rot)
        self.assertLess(np.rad2deg(R.from_matrix(R_hc.T @ R_HEAD_CAM).magnitude()), 1.0)

    def test_frame_arrival_report_counts_gaps(self):
        t = np.arange(100) * 0.05
        t[50:] += 0.05  # one frame missing
        rep = stream_sync.frame_arrival_report(t + 1.8e9)
        self.assertEqual(rep["gaps_over_1_5_intervals"], 1)
        self.assertAlmostEqual(rep["interval_ms_median"], 50.0, delta=0.1)

    def test_disparity_png_round_trip(self):
        d = np.array([[0.0, 12.34, np.nan], [191.99, 1023.9, 0.5]], np.float32)
        back = stereo_record.decode_disparity(stereo_record.encode_disparity(d))
        self.assertTrue(np.isnan(back[0, 0]) and np.isnan(back[0, 2]))
        np.testing.assert_allclose(back[[0, 1, 1, 1], [1, 0, 1, 2]], [12.34, 191.99, 1023.9, 0.5], atol=1 / 64)

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV")
    def test_pair_rotation_of_a_turning_camera(self):
        import cv2
        K = stereo_record.nominal_K(640, 480)
        rng = np.random.default_rng(3)
        tex = cv2.resize((rng.random((60, 80)) * 255).astype(np.uint8), (640, 480), interpolation=cv2.INTER_CUBIC)
        delta = R.from_euler("yx", [2.0, -1.0], degrees=True)  # camera turn from frame 0 to 1
        H = K @ delta.as_matrix().T @ np.linalg.inv(K)
        img1 = cv2.warpPerspective(tex, H, (640, 480), borderMode=cv2.BORDER_REFLECT)
        rv, n = stream_sync.pair_rotation(tex, img1, K)
        self.assertGreater(n, 30)
        self.assertLess(np.rad2deg(np.linalg.norm(rv - delta.as_rotvec())), 0.05)


@unittest.skipUnless(HAVE_CV2, "needs OpenCV")
class LiveStereoSessionTests(unittest.TestCase):
    """A synthetic --stereo session: a far textured scene seen by a head-mounted camera."""

    def _session(self, root, latency, period=0.05, frames=180):
        import cv2
        rng = np.random.default_rng(4)
        w, h = 640, 480
        K = stereo_record.nominal_K(w, h)
        tex = cv2.resize((rng.random((90, 120)) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
        t_head = 1.8e9 + np.arange(0, 10, 1 / 90)
        q_head = _head_motion(t_head - t_head[0])
        head = stream_sync.HeadOrientation(t_head, q_head)
        stereo = root / "stereo"
        (stereo / "left").mkdir(parents=True)
        (stereo / "disparity").mkdir()
        t_cap = t_head[0] + 0.3 + np.arange(frames) * period
        C_ref = R.from_matrix(head.at([t_cap[0]]).as_matrix()[0] @ R_HEAD_CAM)
        rows = ["frame,packet_utc_ns,decoded_utc_ns"]
        for i, tc in enumerate(t_cap):
            C = R.from_matrix(head.at([tc]).as_matrix()[0] @ R_HEAD_CAM)
            Hm = K @ (C.inv() * C_ref).as_matrix() @ np.linalg.inv(K)
            cv2.imwrite(str(stereo / "left" / f"{i:06d}.jpg"),
                        cv2.warpPerspective(tex, Hm, (w, h), borderMode=cv2.BORDER_REFLECT))
            cv2.imwrite(str(stereo / "disparity" / f"{i:06d}.png"),
                        stereo_record.encode_disparity(np.full((h, w), 20.0)))
            arrival = int(round((tc + latency + rng.exponential(0.005)) * 1e9))
            rows.append(f"{i},{arrival},{arrival + 2_000_000}")
        (stereo / "frames.csv").write_text("\n".join(rows) + "\n")
        (stereo / "meta.json").write_text(json.dumps({
            "eye_size": [w, h], "K_left_nominal": K.tolist(), "disparity_scale": 64.0, "baseline_m": None}))
        pc = root / "pc"
        pc.mkdir()
        xr = np.zeros(len(t_head), demo_log.XR_DTYPE)
        xr["sdk_ns"] = ((t_head - 0.21) * 1e9).astype(np.int64)
        xr["t"] = t_head + 0.003 + rng.exponential(0.003, len(t_head))
        xr["serial"] = -1
        xr["headset"][:, 3:] = q_head
        (pc / "xr.bin").write_bytes(xr.tobytes())
        (pc / "meta.json").write_text(json.dumps({"gloves": {}, "tracker_serials": [],
                                                  "events": [{"t": t_cap[0], "label": "demo_start"}]}))

    def test_sync_recovers_the_stream_latency(self):
        import argparse
        from wuji_teleop.demo import process
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            self._session(root, latency=0.12)
            args = argparse.Namespace(latency=None, baseline_m=0.06, nominal_camera=False, no_plot=not HAVE_MPL,
                                      left_tracker=None, right_tracker=None, calibration_from=None,
                                      mount_height=0.025, mount_back=0.02, hand_orientation="imu")
            process.stage_sync(process.Session(root), args)
            report = json.loads((root / "processed" / "sync_report.json").read_text())
            # The relay delay of the head pose (~3 ms min) and the frames' arrival jitter shift it a little.
            self.assertAlmostEqual(report["latency"]["seconds"], 0.12 + 0.003, delta=0.015)
            self.assertGreater(report["latency"]["peak_correlation"], 0.9)
            R_est = np.asarray(report["camera_in_head"]["R_head_camera"])
            self.assertLess(np.rad2deg(R.from_matrix(R_est.T @ R_HEAD_CAM).magnitude()), 1.5)
            synced = np.load(root / "processed" / "synced.npz")
            self.assertEqual(len(synced["t_capture"]), 180)
            self.assertGreater(synced["head_valid"].mean(), 0.95)
            left, disparity, depth = stream_sync.load_frame(root, 5, synced)
            self.assertEqual(left.shape, (480, 640, 3))
            self.assertAlmostEqual(float(np.nanmedian(depth)), float(synced["K"][0, 0]) * 0.06 / 20.0, places=4)
            if HAVE_MPL:
                self.assertTrue((root / "processed" / "sync_check.png").exists())


class _ConstantDisparity:
    info = {"checkpoint": "test"}

    def __call__(self, left, right):
        return np.full(left.shape[:2], 7.5, np.float32)


@unittest.skipUnless(HAVE_CV2 and shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs OpenCV and ffmpeg")
class StereoRecordTests(unittest.TestCase):
    def test_records_every_frame_with_disparity(self):
        import cv2
        width, height, frames = 320, 120, 12
        raw, packets = _h264(width, height, frames)
        fake = FakeHeadset(packets)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stereo"
            rec = stereo_record.StereoRecorder(out, "127.0.0.1", size=(width, height), fps=30,
                                               control_port=fake.port, model=_ConstantDisparity(),
                                               log=lambda m: None, reset_pause=0.05)
            stop = threading.Event()
            threading.Timer(2.5, stop.set).start()
            rec.run(stop)
            rows = (out / "frames.csv").read_text().splitlines()
            self.assertEqual(len(rows), frames + 1)
            self.assertEqual(len(list((out / "left").glob("*.jpg"))), frames)
            self.assertEqual(len(list((out / "disparity").glob("*.png"))), frames)
            png = cv2.imread(str(out / "disparity" / "000003.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(png.dtype, np.uint16)
            self.assertEqual(int(png[5, 100]), int(7.5 * 64))
            left = cv2.imread(str(out / "left" / "000000.jpg"))
            self.assertEqual(left.shape, (height, width // 2, 3))
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual((meta["frames"], meta["disparity_frames"]), (frames, frames))
            self.assertEqual(meta["eye_size"], [width // 2, height])
            self.assertEqual((out / "stream.h264").read_bytes(), b"".join(packets))
            packet_ns = [int(r.split(",")[1]) for r in rows[1:]]
            self.assertEqual(packet_ns, sorted(packet_ns))
            fake.thread.join(3)
            self.assertEqual(fake.commands, ["CLOSE_CAMERA", "OPEN_CAMERA", "CLOSE_CAMERA"])


# ------------------------------------------------------------------- MANO fitting

HAVE_TORCH = importlib.util.find_spec("torch") is not None


def _fake_mano_params(seed=0):
    """A MANO-shaped model (same topology sizes and kinematic tree) built from a simple hand."""
    rng = np.random.default_rng(seed)
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    # Joints (metres): index, middle, pinky, ring fingers along +x, thumb out to +y.
    bases = {1: (0.09, 0.025), 4: (0.095, 0.0), 7: (0.08, -0.045), 10: (0.09, -0.022), 13: (0.03, 0.035)}
    J = np.zeros((16, 3))
    for b, (x, y) in bases.items():
        step = (0.03, 0.0) if b != 13 else (0.025, 0.02)
        for k in range(3):
            J[b + k] = (x + step[0] * k, y + step[1] * k, 0.0)
    tips = {13: (0.03 + 0.025 * 3, 0.035 + 0.02 * 3), 1: (0.18, 0.025), 4: (0.185, 0.0),
            10: (0.18, -0.022), 7: (0.17, -0.045)}
    V = 778
    v = rng.normal(0, 0.004, (V, 3))
    owner = rng.integers(0, 16, V)
    v += J[owner]
    v[:16] = J  # the regressor reads these
    from wuji_teleop.demo.mano_fit import TIP_VERTICES
    for vert, base in zip(TIP_VERTICES, (13, 1, 4, 10, 7)):
        v[vert] = (*tips[base], 0.0)
        owner[vert] = base + 2
    owner[:16] = np.arange(16)
    weights = np.zeros((V, 16))
    weights[np.arange(V), owner] = 1.0
    reg = np.zeros((16, V))
    reg[np.arange(16), np.arange(16)] = 1.0
    shapedirs = rng.normal(0, 0.002, (V, 3, 10))
    return {"v_template": v, "shapedirs": shapedirs, "posedirs": np.zeros((V, 3, 135)),
            "J_regressor": reg, "weights": weights, "hands_mean": rng.normal(0, 0.1, 45),
            "hands_components": np.eye(45), "kintree_table": np.array([parents, list(range(16))]),
            "f": rng.integers(0, V, (1538, 3))}


@unittest.skipUnless(HAVE_TORCH, "needs torch (Fast-FoundationStereo's environment)")
class ManoFitTests(unittest.TestCase):
    def test_reads_a_chumpy_pickle_without_chumpy(self):
        import pickle
        import sys
        import types
        from wuji_teleop.demo import mano_fit
        fake = types.ModuleType("chumpy.ch")

        class Ch:
            pass
        Ch.__module__ = "chumpy.ch"
        Ch.__qualname__ = "Ch"
        fake.Ch = Ch
        sys.modules["chumpy"] = types.ModuleType("chumpy")
        sys.modules["chumpy.ch"] = fake
        try:
            params = _fake_mano_params()
            raw = dict(params)
            ch = Ch()
            ch.x = params["shapedirs"]
            raw["shapedirs"] = ch
            data = pickle.dumps(raw)
        finally:
            del sys.modules["chumpy"], sys.modules["chumpy.ch"]
        out = mano_fit.read_mano_pickle(data)
        np.testing.assert_array_equal(out["shapedirs"], params["shapedirs"])
        self.assertEqual(out["kintree_table"].shape, (2, 16))

    def test_fit_recovers_keypoints_of_a_moving_hand(self):
        import torch
        from wuji_teleop.demo import mano_fit
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layer = mano_fit.ManoLayer(_fake_mano_params(), device=device)
        rng = np.random.default_rng(5)
        n = 40
        t = np.linspace(0, 1, n)[:, None]
        pose = torch.as_tensor(0.4 * np.sin(2 * np.pi * t + rng.uniform(0, 6, 45)) * rng.uniform(0, 1, 45),
                               dtype=torch.float32, device=device)
        go = torch.as_tensor(np.column_stack([0.3 * np.sin(3 * t[:, 0]), 0.2 * t[:, 0], 1.0 + 0 * t[:, 0]]),
                             dtype=torch.float32, device=device)
        tr = torch.as_tensor(np.column_stack([0.4 + 0.1 * t[:, 0], 0.1 * t[:, 0], 1.2 + 0 * t[:, 0]]),
                             dtype=torch.float32, device=device)
        betas = torch.as_tensor(rng.normal(0, 1, (1, 10)), dtype=torch.float32, device=device)
        target = layer.joints21(betas, go, pose, tr).cpu().numpy()
        target += rng.normal(0, 0.001, target.shape)
        valid = np.ones(n, bool)
        valid[17] = False
        fit = mano_fit.fit_hand(layer, target, valid, iters=600, w_twist=0.0, w_hinge=0.0, w_pose=0.0)
        self.assertTrue(np.isnan(fit["joints21"][17]).all())
        err = np.linalg.norm(fit["joints21"][valid] - target[valid], axis=-1) * 1000
        self.assertLess(float(np.median(err)), 3.0)
        self.assertEqual(fit["vertices"].shape, (n, 778, 3))

    def test_lost_finger_follows_its_neighbours(self):
        import torch
        from wuji_teleop.demo import mano_fit
        device = "cuda" if torch.cuda.is_available() else "cpu"
        layer = mano_fit.ManoLayer(_fake_mano_params(), device=device)
        n = 30
        t = np.linspace(0, 1, n)[:, None]
        curl = np.zeros((n, 15, 3))
        curl[:, :12, 1] = 0.9 * np.sin(np.pi * t)            # the four fingers bend together
        curl[:, list(mano_fit.POSE_SLOTS["pinky"]), 1] *= 0.6
        ring, mid, pinky = (list(mano_fit.POSE_SLOTS[f]) for f in ("ring", "middle", "pinky"))
        curl[:, ring] = (curl[:, mid] + curl[:, pinky]) / 2
        pose = torch.as_tensor(curl.reshape(n, 45), dtype=torch.float32, device=device)
        zeros = torch.zeros((n, 3), dtype=torch.float32, device=device)
        target = layer.joints21(torch.zeros((1, 10), device=device), zeros, pose, zeros + 0.3).cpu().numpy()
        truth = target.copy()
        conf = np.full((n, 21), 0.95)
        lost = list(mano_fit.FINGER_MP["ring"])
        target[:, lost] = target[:1, lost]                       # the glove's ring stays put
        conf[:, lost] = 0.3
        valid = np.ones(n, bool)
        self.assertEqual(mano_fit.lost_fingers(conf, valid), ("ring",))
        kw = dict(iters=600, w_twist=0.0, w_hinge=0.0)
        stand_in = mano_fit.fit_hand(layer, target, valid, conf, stand_in=("ring",), **kw)
        plain = mano_fit.fit_hand(layer, target, valid, conf, **kw)
        err = lambda fit: float(np.median(np.linalg.norm(fit["joints21"][:, lost] - truth[:, lost], axis=-1)))  # noqa: E731
        self.assertLess(err(stand_in), 0.004)
        self.assertLess(err(stand_in), err(plain) / 2)

    def test_depth_correction_moves_mano_parameters_exactly(self):
        import torch
        from wuji_teleop.demo import depth_align, mano_fit
        layer = mano_fit.ManoLayer(_fake_mano_params(), device="cpu")
        rng = np.random.default_rng(3)
        n = 5
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        betas, go, pose, tr = rng.normal(0, 1, (1, 10)), rng.normal(0, 0.5, (n, 3)), rng.normal(0, 0.3, (n, 45)), \
            rng.normal(0, 0.2, (n, 3))
        verts, joints = layer.forward(t(betas), t(go), t(pose), t(tr))
        fit = {"valid": np.ones(n, bool), "global_orient": go.copy(), "transl": tr.copy(),
               "vertices": verts.numpy().copy(), "joints21": layer.joints21(t(betas), t(go), t(pose), t(tr)).numpy()}
        keypoints = fit["joints21"].astype(float).copy()
        Rs = R.from_rotvec(rng.normal(0, 0.4, (n, 3))).as_matrix()
        shifts = rng.normal(0, 0.05, (n, 3))
        depth_align.apply_correction(fit, Rs, shifts)
        again, _ = layer.forward(t(betas), t(fit["global_orient"]), t(pose), t(fit["transl"]))
        np.testing.assert_allclose(again.numpy(), fit["vertices"], atol=1e-5)
        np.testing.assert_allclose(depth_align.corrected_keypoints(keypoints, fit), fit["joints21"], atol=1e-5)


def _plane_hand(nx=12, ny=10, size=(0.08, 0.06)):
    """A flat 'hand' mesh in z = 0 facing -z, wrist vertex first."""
    x, y = np.meshgrid(np.linspace(0, size[0], nx), np.linspace(-size[1] / 2, size[1] / 2, ny), indexing="ij")
    V = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    F = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = i * ny + j, (i + 1) * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1
            F += [(a, b, c), (b, d, c)]
    F = np.array(F)
    if np.cross(V[F[0, 1]] - V[F[0, 0]], V[F[0, 2]] - V[F[0, 0]])[2] > 0:
        F = F[:, ::-1]
    return V, F


@unittest.skipUnless(HAVE_TORCH and HAVE_CV2, "needs torch and OpenCV (Fast-FoundationStereo's environment)")
class DepthAlignTests(unittest.TestCase):
    FB = 34.0
    K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])

    def _session(self, root, n, true_pose, tracker_z=None):
        """Camera = world. Table at 0.6 m; the true hand (pose (R, t) of the plane) drawn into the depth."""
        import cv2
        V, F = _plane_hand()
        Rt, tt = true_pose
        dense = np.stack(np.meshgrid(np.linspace(0, 0.08, 160), np.linspace(-0.03, 0.03, 120), indexing="ij"), -1)
        dense = np.column_stack([dense.reshape(-1, 2), np.zeros(dense.size // 2)]) @ Rt.T + tt
        Z = np.full((480, 640), 0.6)
        u = np.round(self.K[0, 0] * dense[:, 0] / dense[:, 2] + self.K[0, 2]).astype(int)
        v = np.round(self.K[1, 1] * dense[:, 1] / dense[:, 2] + self.K[1, 2]).astype(int)
        np.minimum.at(Z, (v, u), dense[:, 2])
        trackers = np.full((n, 4, 4), np.nan)
        if tracker_z is not None:
            Z[230:251, 90:111] = tracker_z - depth_align.TRACKER_SURFACE_M     # a puck at pixel (100, 240)
            trackers[:] = np.eye(4)
            trackers[:, :3, 3] = ((100 - 320) / 400 * tracker_z, 0.0, tracker_z)
        (root / "stereo" / "disparity").mkdir(parents=True)
        png = np.round(self.FB / Z * 64).astype(np.uint16)
        for f in range(n):
            cv2.imwrite(str(root / "stereo" / "disparity" / f"{f:06d}.png"), png)
        synced = {"frame": np.arange(n), "t_capture": np.arange(n) / 30.0, "K": self.K,
                  "image_size": np.array([640, 480]), "T_world_camera": np.repeat(np.eye(4)[None], n, 0),
                  "has_disparity": np.ones(n, bool), "tracker_left": trackers, "disparity_scale": 64.0,
                  "baseline_m": np.nan}
        return synced, V, F

    def test_finds_the_hand_in_the_depth(self):
        true_R = R.from_euler("xyz", [0, 0, 20], degrees=True).as_matrix()
        true_t = np.array([-0.04, 0.0, 0.4])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synced, V, F = self._session(root, 12, (true_R, true_t))
            # The fit is off by 25 degrees about the wrist and (4, -3, 1.5) cm.
            fit_R = R.from_euler("z", -5, degrees=True).as_matrix()
            fit_t = true_t + np.array([0.04, -0.03, 0.015])
            verts = V @ fit_R.T + fit_t
            joints = np.repeat(verts[:1], 21, 0)
            fit = {"valid": np.ones(12, bool), "vertices": np.repeat(verts[None], 12, 0),
                   "joints21": np.repeat(joints[None], 12, 0), "faces": F,
                   "global_orient": np.zeros((12, 3)), "transl": np.zeros((12, 3))}
            c = depth_align.align_hand(root, synced, fit, self.FB, depth_align.initial_frames(synced, fit, frame_step=1),
                                       side="left", device="cuda" if __import__("torch").cuda.is_available() else "cpu")
            self.assertFalse(c["refused"])
            self.assertLess(c["misfit_after"], 0.1)
            moved = (verts - verts[0]) @ c["E"].T + verts[0] + c["d"]
            truth = V @ true_R.T + true_t
            self.assertLess(float(np.median(np.linalg.norm(moved - truth, axis=1))), 0.006)

    def test_estimates_fb_from_a_tracker_in_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            synced, _, _ = self._session(Path(tmp), 25, (np.eye(3), np.array([0.0, 0.0, 0.4])), tracker_z=0.3)
            est = depth_align.tracker_fb(Path(tmp), synced, step=1)
            self.assertEqual(est["used"], ["left"])
            self.assertAlmostEqual(est["fb"], self.FB, delta=0.3)


@unittest.skipUnless(importlib.util.find_spec("viser") is not None, "needs viser (retargeting .venv-vision)")
class HandsVideoTests(unittest.TestCase):
    def test_video_camera_looks_at_the_hands(self):
        from wuji_teleop.demo import hands_view
        center = np.array([0.9, 1.3, -0.5])
        eye = hands_view.video_eye(center, camera_position=[0.85, 1.35, -0.15], azimuth_deg=70,
                                   elevation_deg=35, distance=0.6)
        self.assertAlmostEqual(float(np.linalg.norm(eye - center)), 0.6, places=6)
        self.assertAlmostEqual(float(np.degrees(np.arcsin((eye - center)[2] / 0.6))), 35.0, places=6)
        wxyz = hands_view.look_at(eye, center)
        Rwc = R.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()       # OpenCV camera: z forward, y down
        in_camera = Rwc.T @ (center - eye)
        np.testing.assert_allclose(in_camera, [0, 0, 0.6], atol=1e-9)
        self.assertLess((Rwc[:, 1] @ [0, 0, 1]), 0)                # image down is world down


if __name__ == "__main__":
    unittest.main()
