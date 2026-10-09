import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from wuji_teleop.imu_arm_source import PalmImuRetargeter


def _calibrated():
    source = PalmImuRetargeter()
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0.01, 0, 0],
                      received_at=t / 1e6)
    return source


def _anchor():
    pose = np.eye(4)
    pose[:3, 3] = [0.4, -0.1, 0.8]
    return pose


def test_stationary_rest_does_not_walk():
    source = _calibrated()
    for i in range(102, 502):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0.01, 0, 0],
                      received_at=t / 1e6)
    snapshot = source.snapshot(_anchor(), now=5.02)
    assert snapshot.status == "tracking"
    np.testing.assert_allclose(snapshot.pose[:3, 3], _anchor()[:3, 3], atol=1e-9)
    np.testing.assert_allclose(snapshot.pose[:3, :3], np.eye(3), atol=1e-9)


def test_short_acceleration_moves_and_position_stays_bounded_after_stop():
    source = _calibrated()
    for i in range(102, 152):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [2.0, 0, 9.716], [0.01, 0, 0],
                      received_at=t / 1e6)
    moving = source.snapshot(_anchor(), now=1.52)
    assert np.linalg.norm(moving.pose[:3, 3] - _anchor()[:3, 3]) > 0
    for i in range(152, 252):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0.01, 0, 0],
                      received_at=t / 1e6)
    stopped = source.snapshot(_anchor(), now=2.52)
    np.testing.assert_allclose(stopped.pose, moving.pose, atol=1e-9)


def test_stale_and_invalid_samples_hold_without_pose_updates():
    source = _calibrated()
    good = source.snapshot(_anchor(), now=1.01)
    assert good.pose is not None
    stale = source.snapshot(_anchor(), now=1.30)
    assert stale.status == "stale"
    assert stale.pose is None
    source.update(1_020_001, [0, 0, 0, 0], [0, 0, 9.716], [0, 0, 0],
                  received_at=1.02)
    invalid = source.snapshot(_anchor(), now=1.02)
    assert invalid.status == "invalid"
    assert invalid.pose is None


def test_speed_and_translation_limits_are_enforced():
    source = _calibrated()
    for i in range(102, 1102):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [100, 0, 9.716], [0.5, 0, 0],
                      received_at=t / 1e6)
        assert np.linalg.norm(source._velocity) <= source.max_speed + 1e-12
        assert np.linalg.norm(source._displacement) <= source.max_displacement + 1e-12
    target = source.snapshot(_anchor(), now=11.02)
    assert np.linalg.norm(target.pose[:3, 3] - _anchor()[:3, 3]) <= 0.10 + 1e-12


def test_stale_recovery_discards_first_acceleration_sample():
    source = _calibrated()
    source.snapshot(_anchor(), now=1.01)
    stale = source.snapshot(_anchor(), now=1.30)
    assert stale.pose is None
    source.update(1_310_001, [0, 0, 0, 1], [100, 0, 9.716], [0.5, 0, 0],
                  received_at=1.31)
    recovered = source.snapshot(_anchor(), now=1.31)
    np.testing.assert_allclose(recovered.pose[:3, 3], _anchor()[:3, 3], atol=1e-9)


def test_recenter_restarts_calibration_and_uses_sim_frame_mapping():
    source = PalmImuRetargeter()
    # IMU fused-world frame starts with a 90 degree yaw relative to the sim.
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    qz = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
    sensor_gravity = rz.T @ np.array([0, 0, 9.716])
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, qz, sensor_gravity, [0, 0, 0], received_at=t / 1e6)
    anchor = _anchor()
    anchor[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    initial = source.snapshot(anchor, now=1.01)
    np.testing.assert_allclose(initial.pose[:3, :3], anchor[:3, :3], atol=1e-9)
    # A 90-degree sensor-world rotation maps through startup alignment into sim.
    q180 = [0, 0, 1, 0]
    source.update(1_010_001, q180, sensor_gravity, [0, 0, 0], received_at=1.01)
    moved = source.snapshot(anchor, now=1.01)
    r180 = np.diag([-1.0, -1.0, 1.0])
    expected = anchor[:3, :3] @ rz.T @ r180
    np.testing.assert_allclose(moved.pose[:3, :3], expected, atol=1e-9)
    new_anchor = _anchor()
    new_anchor[:3, :3] = np.eye(3)
    source.recenter(new_anchor)
    assert source.snapshot(new_anchor, now=1.02).status == "calibrating"
    assert np.linalg.norm(source._displacement) == 0


def test_nonidentity_imu_mount_rotation_maps_sensor_to_wrist_axes():
    # The actual glove reports a substantial static wrist-to-IMU rotation.
    mount = Rotation.from_quat([0, np.sqrt(0.5), np.sqrt(0.5), 0]).as_matrix()
    source = PalmImuRetargeter(imu_in_wrist_rotation=mount)
    ref_q = Rotation.from_matrix(mount).as_quat()
    gravity_sensor = mount.T @ [0, 0, 9.716]
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, ref_q, gravity_sensor, [0, 0, 0], received_at=t / 1e6)
    anchor = _anchor()
    np.testing.assert_allclose(source.snapshot(anchor, now=1.01).pose[:3, :3],
                               np.eye(3), atol=1e-9)
    wrist_roll = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    sensor_R = wrist_roll @ mount
    source.update(1_010_001, Rotation.from_matrix(sensor_R).as_quat(),
                  sensor_R.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.01)
    np.testing.assert_allclose(source.snapshot(anchor, now=1.01).pose[:3, :3],
                               wrist_roll, atol=1e-9)


def test_palm_down_hand_mount_preserves_upward_translation():
    mount = Rotation.from_quat([0, np.sqrt(0.5), np.sqrt(0.5), 0]).as_matrix()
    hand_mount = np.diag([1.0, -1.0, -1.0])
    source = PalmImuRetargeter(imu_in_wrist_rotation=mount,
                               hand_in_ee_rotation=hand_mount)
    ref_q = Rotation.from_matrix(mount).as_quat()
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, ref_q, mount.T @ [0, 0, 9.716],
                      [0, 0, 0], received_at=t / 1e6)
    anchor = _anchor()
    anchor[:3, :3] = hand_mount
    np.testing.assert_allclose(source.snapshot(anchor, now=1.01).pose[:3, :3],
                               hand_mount, atol=1e-9)
    for i in range(101, 111):
        t = i * 10_000
        source.update(t + 1, ref_q, mount.T @ [0, 0, 11.716],
                      [0, 0, 0], received_at=t / 1e6)
    target = source.snapshot(anchor, now=1.11)
    assert target.pose[2, 3] > anchor[2, 3]


def test_directional_strokes_accumulate_arm_commands_without_double_trigger():
    source = PalmImuRetargeter(motion_mode="gesture", max_displacement=0.30,
                               max_speed=0.30)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    # Start moving up, then decelerate. The braking pulse is not another step.
    for i in range(101, 111):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 11.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(111, 126):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 7.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(126, 181):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    first = source.snapshot(_anchor(), now=1.81)
    assert first.gesture_steps == 1
    np.testing.assert_allclose(first.pose[:3, 3] - _anchor()[:3, 3],
                               [0, 0, 0.06], atol=1e-6)
    # A separate sideways stroke moves the target again across the workspace.
    for i in range(181, 191):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [2, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(191, 221):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    second = source.snapshot(_anchor(), now=2.21)
    assert second.gesture_steps == 2
    np.testing.assert_allclose(second.pose[:3, 3] - _anchor()[:3, 3],
                               [0.06, 0, 0.06], atol=1e-6)


def test_rotating_glove_does_not_command_directional_stroke():
    source = PalmImuRetargeter(motion_mode="gesture")
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(101, 121):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [2, 0, 9.716], [0.8, 0, 0],
                      received_at=t / 1e6)
    assert source.snapshot(_anchor(), now=1.21).gesture_steps == 0


def test_tilt_tracks_finger_direction_and_ignores_twist():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               tilt_gain=0.55, tilt_deadband=0.04,
                               max_displacement=0.40)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    anchor = _anchor()
    # Same end-effector orientation as the bent Vega home pose: fingers point +X.
    anchor[:3, :3] = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])
    rest = source.snapshot(anchor, now=1.01)
    np.testing.assert_allclose(rest.pose[:3, 3], anchor[:3, 3], atol=1e-8)

    world_up_tip = Rotation.from_euler("y", -35, degrees=True).as_matrix()
    tip = anchor[:3, :3].T @ world_up_tip @ anchor[:3, :3]
    source.update(1_020_001, Rotation.from_matrix(tip).as_quat(),
                  tip.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.02)
    raised = source.snapshot(anchor, now=1.02)
    delta = raised.pose[:3, 3] - anchor[:3, 3]
    assert delta[2] > 0.15
    assert abs(delta[1]) < 0.05
    assert np.linalg.norm(delta) <= 0.40 + 1e-9

    twist = anchor[:3, :3].T @ Rotation.from_euler("x", 40, degrees=True).as_matrix() @ anchor[:3, :3]
    # Twist about the finger axis (+X in this home pose) must not translate.
    source.update(1_030_001, Rotation.from_matrix(twist).as_quat(),
                  twist.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.03)
    twisted = source.snapshot(anchor, now=1.03)
    np.testing.assert_allclose(twisted.pose[:3, 3], anchor[:3, 3], atol=1e-6)


def test_tilt_mode_integrates_translation_without_a_pointing_change():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               max_speed=0.45, velocity_damping=2.6,
                               max_displacement=0.45)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(101, 161):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 12.5], [0, 0, 0],
                      received_at=t / 1e6)
    moved = source.snapshot(_anchor(), now=1.61)
    assert moved.pose[2, 3] > _anchor()[2, 3] + 0.04
    for i in range(161, 221):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    held = source.snapshot(_anchor(), now=2.21)
    np.testing.assert_allclose(held.pose[:3, 3], moved.pose[:3, 3], atol=1e-6)


def test_pivot_tilt_changes_reach_for_elbow_control():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               tilt_gain=0.55, tilt_deadband=0.04,
                               vertical_to_forward=True)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    anchor = _anchor()
    source.set_tilt_pivot([0.0, -0.1, 0.3])
    rest = source.snapshot(anchor, now=1.01)
    current_rotation = Rotation.from_euler("y", -35, degrees=True).as_matrix()
    source.update(1_010_001, Rotation.from_matrix(current_rotation).as_quat(),
                  current_rotation.T @ [0, 0, 9.716], [0, 0, 0],
                  received_at=1.01)
    tilted = source.snapshot(anchor, now=1.01)
    pivot = np.array([0.0, -0.1, 0.3])
    rest_reach = np.linalg.norm(rest.pose[:3, 3] - pivot)
    tilted_reach = np.linalg.norm(tilted.pose[:3, 3] - pivot)
    assert tilted_reach < rest_reach - 0.005


def test_tilt_translation_leaks_back_when_the_palm_rests():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               max_speed=0.45, velocity_damping=2.6,
                               max_displacement=0.45, translation_leak=1.6)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    for i in range(101, 161):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 12.5], [0, 0, 0],
                      received_at=t / 1e6)
    moved = source.snapshot(_anchor(), now=1.61)
    assert moved.pose[2, 3] > _anchor()[2, 3] + 0.04
    for i in range(161, 261):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    rested = source.snapshot(_anchor(), now=2.61)
    assert rested.pose[2, 3] < moved.pose[2, 3] - 0.01
    assert rested.pose[2, 3] > _anchor()[2, 3]


def test_vertical_tilt_becomes_a_forward_reach():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               tilt_gain=0.55, tilt_deadband=0.04,
                               max_displacement=0.40, vertical_to_forward=True)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    anchor = _anchor()
    anchor[:3, :3] = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])
    up = anchor[:3, :3].T @ Rotation.from_euler("y", -35, degrees=True).as_matrix() @ anchor[:3, :3]
    source.update(1_020_001, Rotation.from_matrix(up).as_quat(),
                  up.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.02)
    raised = source.snapshot(anchor, now=1.02)
    delta = raised.pose[:3, 3] - anchor[:3, 3]
    assert delta[0] > 0.15
    assert abs(delta[2]) < delta[0]

    down = anchor[:3, :3].T @ Rotation.from_euler("y", 35, degrees=True).as_matrix() @ anchor[:3, :3]
    source.update(1_030_001, Rotation.from_matrix(down).as_quat(),
                  down.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.03)
    lowered = source.snapshot(anchor, now=1.03)
    assert lowered.pose[0, 3] < anchor[0, 3] - 0.15


def test_tilt_with_pivot_swings_the_hand_around_the_shoulder():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               tilt_deadband=0.04, max_displacement=0.40,
                               arc_gain=1.2, vertical_to_forward=True)
    pivot = np.array([0.0, -0.1, 1.2])
    source.set_tilt_pivot(pivot)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    anchor = _anchor()
    # Fingers point +X, as in the Vega home pose.
    anchor[:3, :3] = np.array([[0.0, 0.0, 1.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0]])
    arm = anchor[:3, 3] - pivot
    up = anchor[:3, :3].T @ Rotation.from_euler("y", -80, degrees=True).as_matrix() @ anchor[:3, :3]
    source.update(1_020_001, Rotation.from_matrix(up).as_quat(),
                  up.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.02)
    raised = source.snapshot(anchor, now=1.02).pose[:3, 3]
    # The shoulder arc carries the lift, while the forward/back component
    # changes reach so the downstream IK can recruit the elbow.
    assert abs(np.linalg.norm(raised - pivot) - np.linalg.norm(arm)) > 0.01
    assert raised[2] > pivot[2] + 0.2
    assert np.linalg.norm(raised - anchor[:3, 3]) > 0.40
    swung = np.degrees(np.arccos((raised - pivot) @ arm / (arm @ arm)))
    assert abs(swung - 1.2 * (80 - np.degrees(0.04))) < 2.0

    twist = anchor[:3, :3].T @ Rotation.from_euler("x", 40, degrees=True).as_matrix() @ anchor[:3, :3]
    source.update(1_030_001, Rotation.from_matrix(twist).as_quat(),
                  twist.T @ [0, 0, 9.716], [0, 0, 0], received_at=1.03)
    np.testing.assert_allclose(source.snapshot(anchor, now=1.03).pose[:3, 3],
                               anchor[:3, 3], atol=1e-6)


def test_small_tilt_stays_inside_the_deadband():
    source = PalmImuRetargeter(motion_mode="tilt", tilt_smoothing=1.0,
                               tilt_deadband=0.04)
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    small = Rotation.from_euler("y", 2, degrees=True)
    source.update(1_020_001, small.as_quat(), small.as_matrix().T @ [0, 0, 9.716],
                  [0, 0, 0], received_at=1.02)
    held = source.snapshot(_anchor(), now=1.02)
    np.testing.assert_allclose(held.pose[:3, 3], _anchor()[:3, 3], atol=1e-8)


def test_orientation_only_mode_never_integrates_translation():
    source = PalmImuRetargeter(motion_mode="orientation")
    for i in range(101):
        t = i * 10_000
        source.update(t + 1, [0, 0, 0, 1], [0, 0, 9.716], [0, 0, 0],
                      received_at=t / 1e6)
    q = Rotation.from_euler("z", 20, degrees=True).as_quat()
    for i in range(101, 151):
        t = i * 10_000
        source.update(t + 1, q, [5, 0, 9.716], [0, 0, 0.5],
                      received_at=t / 1e6)
    snapshot = source.snapshot(_anchor(), now=1.51)
    assert snapshot.status == "tracking"
    np.testing.assert_allclose(snapshot.pose[:3, 3], _anchor()[:3, 3], atol=1e-9)
    np.testing.assert_allclose(snapshot.pose[:3, :3],
                               Rotation.from_quat(q).as_matrix(), atol=1e-9)


class ImuArmSourceTests(unittest.TestCase):
    def test_stationary_rest(self):
        test_stationary_rest_does_not_walk()

    def test_short_acceleration(self):
        test_short_acceleration_moves_and_position_stays_bounded_after_stop()

    def test_stale_and_invalid(self):
        test_stale_and_invalid_samples_hold_without_pose_updates()

    def test_limits(self):
        test_speed_and_translation_limits_are_enforced()

    def test_recovery(self):
        test_stale_recovery_discards_first_acceleration_sample()

    def test_recenter_and_frame_mapping(self):
        test_recenter_restarts_calibration_and_uses_sim_frame_mapping()

    def test_nonidentity_mount_rotation(self):
        test_nonidentity_imu_mount_rotation_maps_sensor_to_wrist_axes()

    def test_palm_down_upward_translation(self):
        test_palm_down_hand_mount_preserves_upward_translation()

    def test_directional_strokes(self):
        test_directional_strokes_accumulate_arm_commands_without_double_trigger()

    def test_rotation_does_not_nudge_arm(self):
        test_rotating_glove_does_not_command_directional_stroke()

    def test_orientation_only(self):
        test_orientation_only_mode_never_integrates_translation()

    def test_tilt_tracking(self):
        test_tilt_tracks_finger_direction_and_ignores_twist()

    def test_tilt_arc_around_shoulder(self):
        test_tilt_with_pivot_swings_the_hand_around_the_shoulder()

    def test_vertical_tilt_reaches_forward(self):
        test_vertical_tilt_becomes_a_forward_reach()

    def test_tilt_deadband(self):
        test_small_tilt_stays_inside_the_deadband()

    def test_tilt_translation(self):
        test_tilt_mode_integrates_translation_without_a_pointing_change()

    def test_tilt_translation_leaks(self):
        test_tilt_translation_leaks_back_when_the_palm_rests()
