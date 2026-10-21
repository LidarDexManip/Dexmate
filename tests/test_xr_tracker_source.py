"""XRoboToolkit wrist tracker parsing, frame conversion and side assignment."""

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from wuji_teleop.xr_tracker_source import (
    XR_TO_ROBOT,
    XRoboToolkitTrackerSource,
    assign_tracker_sides,
    heading_rotation,
    operator_heading,
    xr_pose_to_robot,
)


def _xr_pose(position, rotation=None):
    quat = (Rotation.identity() if rotation is None else rotation).as_quat()
    return [*position, *quat]


def _robot_pose(position, yaw=0.0):
    pose = np.eye(4)
    pose[:3, :3] = heading_rotation(yaw)
    pose[:3, 3] = position
    return pose


class _FakeSdk:
    def __init__(self):
        self.timestamp = 0
        self.serials = []
        self.poses = []
        self.headset = [0.0] * 7
        self.initialized = self.closed = False

    def init(self):
        self.initialized = True

    def close(self):
        self.closed = True

    def num_motion_data_available(self):
        return len(self.poses)

    def get_motion_timestamp_ns(self):
        return self.timestamp

    def get_motion_tracker_serial_numbers(self):
        return list(self.serials)

    def get_motion_tracker_pose(self):
        return [list(pose) for pose in self.poses]

    def get_headset_pose(self):
        return list(self.headset)


class FrameConversionTests(unittest.TestCase):
    def test_xr_axes_map_to_robot_axes(self):
        # OpenXR forward is -z, right is +x and up is +y.
        np.testing.assert_allclose(XR_TO_ROBOT @ [0, 0, -1], [1, 0, 0])
        np.testing.assert_allclose(XR_TO_ROBOT @ [1, 0, 0], [0, -1, 0])
        np.testing.assert_allclose(XR_TO_ROBOT @ [0, 1, 0], [0, 0, 1])
        self.assertAlmostEqual(np.linalg.det(XR_TO_ROBOT), 1.0)

    def test_pose_conversion_rotates_position_and_body_axes(self):
        # A tracker turned 90 deg left (about OpenXR up) looks along robot +y.
        turn = Rotation.from_euler("y", 90, degrees=True)
        pose = xr_pose_to_robot(_xr_pose([0.1, 1.2, -0.4], turn))
        np.testing.assert_allclose(pose[:3, 3], [0.4, -0.1, 1.2])
        np.testing.assert_allclose(pose[:3, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(pose[:3, 2], [0, 0, 1], atol=1e-12)

    def test_unnormalized_or_zero_quaternions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "normalized"):
            xr_pose_to_robot([0, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            xr_pose_to_robot([0, 0, math.nan, 0, 0, 0, 1])


class SideAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.poses = {"A": _robot_pose([0.3, 0.2, 1.0]), "B": _robot_pose([0.3, -0.2, 1.0])}

    def test_operator_left_tracker_becomes_left_wrist(self):
        self.assertEqual(assign_tracker_sides(self.poses), {"left": "A", "right": "B"})
        # Facing robot -x swaps which tracker is on the operator's left.
        self.assertEqual(assign_tracker_sides(self.poses, yaw=math.pi),
                         {"left": "B", "right": "A"})

    def test_configured_serials_win_and_must_stream(self):
        self.assertEqual(assign_tracker_sides(self.poses, left_serial="B"),
                         {"left": "B", "right": "A"})
        with self.assertRaisesRegex(ValueError, "not streaming"):
            assign_tracker_sides(self.poses, left_serial="C")
        with self.assertRaisesRegex(ValueError, "must differ"):
            assign_tracker_sides(self.poses, left_serial="A", right_serial="A")

    def test_ambiguous_trackers_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unassigned"):
            assign_tracker_sides({"A": self.poses["A"]})
        close = {"A": _robot_pose([0.3, 0.01, 1.0]), "B": _robot_pose([0.3, -0.01, 1.0])}
        with self.assertRaisesRegex(ValueError, "apart"):
            assign_tracker_sides(close)

    def test_heading_comes_from_headset_or_wrist_line(self):
        self.assertAlmostEqual(operator_heading(_robot_pose([0, 0, 1.6], yaw=0.4)), 0.4)
        self.assertAlmostEqual(
            operator_heading(None, (self.poses["A"], self.poses["B"])), 0.0)
        with self.assertRaisesRegex(ValueError, "faces"):
            operator_heading(None, ())


class TrackerSourceTests(unittest.TestCase):
    def setUp(self):
        self.sdk = _FakeSdk()
        self.source = XRoboToolkitTrackerSource(sdk=self.sdk)

    def test_new_timestamps_publish_frames_and_repeats_do_not(self):
        self.assertTrue(self.sdk.initialized)
        self.assertIsNone(self.source.read().frame)
        self.sdk.serials = ["L", "R"]
        self.sdk.poses = [_xr_pose([-0.2, 1.0, -0.3]), _xr_pose([0.2, 1.0, -0.3])]
        self.sdk.headset = _xr_pose([0.0, 1.6, 0.0])
        self.sdk.timestamp = 100
        first = self.source.read()
        np.testing.assert_allclose(first.frame.poses["L"][:3, 3], [0.3, 0.2, 1.0])
        np.testing.assert_allclose(first.frame.headset[:3, 3], [0.0, 0.0, 1.6])
        repeat = self.source.read()
        self.assertEqual(repeat.revision, first.revision)
        self.assertEqual(repeat.received, 1)
        self.sdk.timestamp = 200
        self.assertEqual(self.source.read().received, 2)
        self.source.close()
        self.assertTrue(self.sdk.closed)

    def test_streams_without_timestamps_publish_only_changed_poses(self):
        self.sdk.serials = ["L"]
        self.sdk.poses = [_xr_pose([0.0, 1.0, -0.3])]
        self.assertEqual(self.source.read().received, 1)
        self.assertEqual(self.source.read().received, 1)
        self.sdk.poses = [_xr_pose([0.0, 1.0, -0.31])]
        self.assertEqual(self.source.read().received, 2)

    def test_bad_frames_report_errors_and_keep_the_last_good_frame(self):
        self.sdk.serials = ["L"]
        self.sdk.poses = [_xr_pose([0.0, 1.0, -0.3])]
        self.sdk.timestamp = 1
        good = self.source.read()
        self.sdk.poses = [[0.0, 1.0, -0.3, 0.0, 0.0, 0.0, 0.0]]
        self.sdk.timestamp = 2
        bad = self.source.read()
        self.assertIn("normalized", bad.error)
        self.assertIs(bad.frame, good.frame)
        self.sdk.serials = ["L", "L"]
        self.sdk.poses = [_xr_pose([0, 1, 0]), _xr_pose([0, 1, 0])]
        self.sdk.timestamp = 3
        self.assertIn("repeated", self.source.read().error)


if __name__ == "__main__":
    unittest.main()
