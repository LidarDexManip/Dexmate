"""A glove finger with a low-confidence sensor copies its neighbour (ring <- pinky)."""

import unittest
from unittest.mock import Mock

import numpy as np

from wuji_teleop.retarget import FingerStandIn, TrackingController, copy_finger
from wuji_teleop.sources import HandFrame, LatestFrameBuffer, synthetic_keypoints

GOOD = (0.95, 0.95, 0.95, 0.95, 0.95)
RING_LOW = (0.95, 0.95, 0.95, 0.1, 0.95)
RING = slice(13, 17)


def frame(t, confidence, points=None):
    points = synthetic_keypoints(t, "right") if points is None else points
    return HandFrame(points, t, int(t * 1e6), int(t * 1e3), min(confidence), confidence)


def feed(stand_in, start, stop, confidence, rate=100):
    for t in np.arange(start, stop, 1 / rate):
        points = stand_in.apply(frame(t, confidence))
    return points


class CopyFingerTests(unittest.TestCase):
    def test_ring_takes_pinky_bone_directions_and_keeps_its_base_and_lengths(self):
        points = synthetic_keypoints(1.7, "right")
        out = copy_finger(points, 3, 4)
        ring, pinky = np.diff(out[RING], axis=0), np.diff(points[17:21], axis=0)
        unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)  # noqa: E731
        np.testing.assert_allclose(unit(ring), unit(pinky), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(ring, axis=1),
                                   np.linalg.norm(np.diff(points[RING], axis=0), axis=1),
                                   rtol=1e-5)
        np.testing.assert_array_equal(out[13], points[13])
        others = np.r_[0:14, 17:21]
        np.testing.assert_array_equal(out[others], points[others])
        self.assertFalse(np.allclose(out[RING], points[RING]))


class FingerStandInTests(unittest.TestCase):
    def test_low_ring_confidence_copies_the_pinky_from_the_first_frame(self):
        stand_in = FingerStandIn()
        f = frame(1.0, RING_LOW)
        out = stand_in.apply(f)
        np.testing.assert_allclose(out, copy_finger(f.keypoints, 3, 4))
        self.assertEqual(stand_in.copying, {3})
        self.assertEqual(stand_in.take_changes(),
                         ["ring sensor confidence 0.10: the ring follows the pinky."])
        self.assertEqual(stand_in.take_changes(), [])
        self.assertEqual(stand_in.note, "ring copies pinky (ring sensor 0.10)")

    def test_healthy_glove_and_missing_confidence_pass_through_unchanged(self):
        stand_in = FingerStandIn()
        f = frame(1.0, GOOD)
        self.assertIs(stand_in.apply(f), f.keypoints)
        f = HandFrame(synthetic_keypoints(1.0, "right"), 1.0, 1, 1)  # e.g. synthetic input
        self.assertIs(stand_in.apply(f), f.keypoints)
        self.assertEqual((stand_in.copying, stand_in.note, stand_in.changes),
                         (frozenset(), "", []))

    def test_brief_dip_or_spike_does_not_switch(self):
        stand_in = FingerStandIn()
        feed(stand_in, 0, 1, GOOD)
        feed(stand_in, 1, 1.03, RING_LOW)  # three bad frames
        self.assertEqual(stand_in.copying, frozenset())
        feed(stand_in, 1.03, 3, RING_LOW)
        self.assertEqual(stand_in.copying, {3})
        feed(stand_in, 3, 3.05, GOOD)  # a short good burst, as the failed sensor shows
        feed(stand_in, 3.05, 5, RING_LOW)
        self.assertEqual(stand_in.copying, {3})

    def test_returns_to_its_own_sensor_after_two_seconds_back_up(self):
        stand_in = FingerStandIn()
        feed(stand_in, 0, 1, RING_LOW)
        stand_in.take_changes()
        feed(stand_in, 1, 2.5, GOOD)
        self.assertEqual(stand_in.copying, {3})  # recovered, but not for 2 s yet
        f = frame(4.0, GOOD)
        feed(stand_in, 2.5, 4.0, GOOD)
        self.assertIs(stand_in.apply(f), f.keypoints)
        self.assertEqual(stand_in.copying, frozenset())
        self.assertEqual(stand_in.note, "")
        [message] = stand_in.take_changes()
        self.assertRegex(message, r"^ring sensor confidence back to 0\.9\d: "
                                  r"the ring follows its own sensor again\.$")

    def test_a_low_donor_is_not_copied(self):
        stand_in = FingerStandIn()
        both_low = (0.95, 0.95, 0.95, 0.1, 0.2)
        f = frame(1.0, both_low)
        self.assertIs(stand_in.apply(f), f.keypoints)
        self.assertEqual(stand_in.note, "ring sensor low (0.10), pinky too")
        feed(stand_in, 1, 2, RING_LOW)  # the pinky recovers: copy it now
        self.assertEqual(stand_in.copying, {3})
        feed(stand_in, 2, 3, both_low)
        self.assertEqual(stand_in.copying, frozenset())
        # The smoothed pinky confidence, reported as it crosses the 0.6 threshold.
        self.assertRegex(stand_in.take_changes()[-1], r"^pinky sensor low too \(0\.[56]\d\): "
                                                      r"the ring follows its own low sensor\.$")

    def test_tracking_controller_retargets_the_rebuilt_points(self):
        retargeter = Mock()
        retargeter.step.return_value = np.zeros(20)
        controller = TrackingController(retargeter, Mock(), stand_in=FingerStandIn())
        buffer = LatestFrameBuffer()
        points = synthetic_keypoints(1.0, "right")
        buffer.publish(points, received_at=1.0, timestamp_us=1, sequence=1,
                       confidence=0.1, finger_confidence=RING_LOW)
        self.assertTrue(controller.update(buffer.snapshot(), 1.0))
        np.testing.assert_allclose(retargeter.step.call_args.args[0],
                                   copy_finger(points, 3, 4))


if __name__ == "__main__":
    unittest.main()
