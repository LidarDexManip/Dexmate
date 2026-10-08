"""Fake-payload tests for the optional Pico/DexStream wrist source."""

import json
import time
import unittest
from types import SimpleNamespace as NS

import numpy as np

from wuji_teleop.vr_pose_source import PicoControllerPoseSource, validate_rigid_pose


class FakeSession:
    def declare_subscriber(self, topic, callback):
        self.topic = topic
        self.callback = callback
        self.subscriber = NS(undeclare=lambda: setattr(self, "undeclared", True))
        return self.subscriber

    def close(self):
        self.closed = True


class FakeSample:
    def __init__(self, payload):
        self.payload = NS(to_bytes=lambda: payload if isinstance(payload, bytes)
                          else json.dumps(payload).encode("utf-8"))


def payload(timestamp_ns=None, pose=None, trigger=0.0):
    return {
        "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
        "left": {"wrist": (np.eye(4) if pose is None else pose).tolist(),
                 "index_trigger": trigger},
        "right": {"wrist": np.eye(4).tolist(), "index_trigger": 0.0},
    }


class PicoSourceTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.source = PicoControllerPoseSource(session=self.session)

    def tearDown(self):
        self.source.close()

    def test_robot_frame_left_pose_and_trigger_are_decoded(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.2, -0.1, 0.4]
        frame = self.source.parse_payload(payload(pose=pose, trigger=0.75))
        np.testing.assert_allclose(frame.pose, pose)
        self.assertEqual(frame.index_trigger, 0.75)
        self.session.callback(FakeSample(payload(pose=pose, trigger=0.75)))
        snapshot = self.source.read()
        self.assertEqual(self.session.topic, "vr/controllers")
        self.assertEqual(snapshot.received, 1)
        self.assertEqual(snapshot.frame.index_trigger, 0.75)

    def test_rejects_non_rigid_pose_and_invalid_trigger(self):
        for pose in (np.zeros((4, 4)), np.full((4, 4), np.nan)):
            with self.subTest(pose=pose.shape), self.assertRaises(ValueError):
                validate_rigid_pose(pose)
        bad_rotation = np.eye(4)
        bad_rotation[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "rotation"):
            validate_rigid_pose(bad_rotation)
        self.session.callback(FakeSample(payload(trigger=1.5)))
        self.assertIn("trigger", self.source.read().error)

    def test_duplicate_out_of_order_does_not_refresh_and_old_timestamp_is_stale(self):
        stamp = time.time_ns()
        self.session.callback(FakeSample(payload(timestamp_ns=stamp)))
        first = self.source.read().frame
        self.session.callback(FakeSample(payload(timestamp_ns=stamp)))
        self.session.callback(FakeSample(payload(timestamp_ns=stamp - 1)))
        self.assertEqual(self.source.read().received, 1)
        self.assertEqual(self.source.read().frame.received_at, first.received_at)

        old_session = FakeSession()
        old_source = PicoControllerPoseSource(session=old_session)
        old = time.time_ns() - 1_000_000_000
        old_session.callback(FakeSample(payload(timestamp_ns=old, trigger=0.8)))
        snapshot = old_source.read()
        self.assertEqual(snapshot.received, 1)
        self.assertGreater(time.monotonic() - snapshot.frame.received_at, 0.9)
        old_source.close()

    def test_close_undeclares_subscriber(self):
        self.source.close()
        self.assertTrue(self.session.undeclared)

    def test_live_constructor_passes_local_zenoh_config(self):
        session = FakeSession()
        config = object()
        captured = []
        module = NS(Config=lambda: config,
                    open=lambda value: (captured.append(value), session)[1])
        source = PicoControllerPoseSource(zenoh_module=module)
        self.assertEqual(captured, [config])
        source.close()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
