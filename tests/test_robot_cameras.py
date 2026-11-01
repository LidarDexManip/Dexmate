"""Robot camera feed for the hardware dashboard, without a robot or a display."""

import io
import struct
import sys
import time
import unittest
import unittest.mock

import numpy as np

from wuji_teleop.robot_cameras import CAMERAS, RobotCameras, downscale, pump, to_ppm


def _decode(payload):
    return {"data": np.frombuffer(payload, np.uint8).reshape(1200, 1920, 3)}


class PumpTests(unittest.TestCase):
    def test_each_new_frame_is_sent_once_downscaled(self):
        frame = np.zeros((1200, 1920, 3), np.uint8)
        frame[:, :, 1] = 200
        latest, sent, out = {1: (1, frame.tobytes())}, {}, io.BytesIO()
        self.assertEqual(pump(latest, out, sent, _decode), 1)
        self.assertEqual(pump(latest, out, sent, _decode), 0)  # same frame: not resent
        latest[1] = (2, frame.tobytes())
        self.assertEqual(pump(latest, out, sent, _decode), 1)
        data = out.getvalue()
        index, width, height, size = struct.unpack("<BHHI", data[:9])
        self.assertEqual((index, width, height), (1, 480, 300))
        self.assertTrue(data[9:9 + size].startswith(b"P6 480 300 255\n"))

    def test_an_undecodable_frame_is_skipped(self):
        def broken(payload):
            raise ValueError("bad jpeg")

        out = io.BytesIO()
        with unittest.mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(pump({0: (1, b"x")}, out, {}, broken), 0)
        self.assertEqual(out.getvalue(), b"")

    def test_ppm_matches_the_image(self):
        rgb = downscale(np.full((10, 960, 3), 7, np.uint8))
        self.assertEqual(rgb.shape, (5, 480, 3))
        ppm = to_ppm(rgb)
        self.assertEqual(len(ppm), len(b"P6 480 5 255\n") + 480 * 5 * 3)


# Child process standing in for the worker: one 2x1 frame for the wrist camera.
_ONE_FRAME = (
    "import struct, sys\n"
    "ppm = b'P6 2 1 255\\n' + bytes(6)\n"
    "sys.stdout.buffer.write(struct.pack('<BHHI', 1, 2, 1, len(ppm)) + ppm)\n"
    "sys.stdout.buffer.flush()\n"
    "import time; time.sleep(30)\n")


class RobotCamerasTests(unittest.TestCase):
    def _wait(self, cameras, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            views = cameras.views()
            if predicate(views):
                return views
            time.sleep(0.05)
        self.fail(f"timed out: {views}")

    def test_streaming_camera_is_shown_and_the_other_is_not_detected(self):
        cameras = RobotCameras(command=[sys.executable, "-c", _ONE_FRAME])
        try:
            views = self._wait(cameras, lambda v: v["Right wrist camera"][0] is not None)
        finally:
            cameras.close()
        ppm, sequence, text = views["Right wrist camera"]
        self.assertEqual((ppm[:11], sequence, text), (b"P6 2 1 255\n", 1, "streaming"))
        ego = views["Ego camera"]
        self.assertIsNone(ego[0])
        self.assertIn("dexsensor enable head_camera", ego[2])
        stale = cameras.views(now=time.monotonic() + 5)["Right wrist camera"]
        self.assertIsNone(stale[0])
        self.assertIn("not detected for", stale[2])

    def test_a_worker_that_fails_reports_why(self):
        cameras = RobotCameras(command=[sys.executable, "-c",
                                        "import sys; sys.exit('no ZENOH_CONFIG')"])
        try:
            views = self._wait(cameras, lambda v: "stopped" in v["Ego camera"][2])
        finally:
            cameras.close()
        self.assertIn("no ZENOH_CONFIG", views["Ego camera"][2])

    def test_cameras_are_the_ego_and_right_wrist_views(self):
        self.assertEqual([name for name, _, _ in CAMERAS], ["Ego camera", "Right wrist camera"])


if __name__ == "__main__":
    unittest.main()
