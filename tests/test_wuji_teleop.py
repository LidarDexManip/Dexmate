"""Run with .venv/bin/python -m unittest discover -s tests -v."""

import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from wuji_teleop.__main__ import main
from wuji_teleop.retarget import TrackingController, WujiRetargeter
from wuji_teleop.simulation import MujocoHand, load_hand
from wuji_teleop.sources import (
    GloveSource, LatestFrameBuffer, select_glove, synthetic_keypoints, validate_keypoints,
)


class BufferTests(unittest.TestCase):
    def test_latest_only_and_owned_copy(self):
        buffer = LatestFrameBuffer()
        points = synthetic_keypoints(0, "right")
        for i in range(100):
            buffer.publish(points, received_at=i, timestamp_us=i, sequence=i)
        points[:] = 0
        snapshot = buffer.snapshot()
        self.assertEqual(snapshot.received, 100)
        self.assertEqual(snapshot.frame.sequence, 99)
        self.assertGreater(np.linalg.norm(snapshot.frame.keypoints), 0)
        self.assertFalse(snapshot.frame.keypoints.flags.writeable)

    def test_duplicate_out_of_order_and_invalid_do_not_refresh_watchdog(self):
        buffer = LatestFrameBuffer()
        points = synthetic_keypoints(0, "right")
        buffer.publish(points, received_at=1, timestamp_us=100, sequence=1)
        for timestamp in (100, 99):
            buffer.publish(points, received_at=2, timestamp_us=timestamp, sequence=2)
        self.assertEqual(buffer.snapshot().received, 1)
        self.assertEqual(buffer.snapshot().frame.received_at, 1)
        buffer.publish(np.zeros((21, 3)), received_at=3, timestamp_us=101, sequence=3)
        self.assertIsNotNone(buffer.snapshot().error)
        self.assertEqual(buffer.snapshot().frame.received_at, 1)
        buffer.publish(points, received_at=4, timestamp_us=102, sequence=4)
        self.assertIsNone(buffer.snapshot().error)

    def test_reject_malformed_and_degenerate_frames(self):
        for points in (np.zeros((20, 3)), np.zeros((21, 3)),
                       np.full((21, 3), np.nan), np.full((21, 3), np.inf)):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                validate_keypoints(points)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.buffer = LatestFrameBuffer()
        self.hand = Mock()
        self.retargeter = Mock()
        self.retargeter.step.return_value = np.zeros(20)
        self.controller = TrackingController(self.retargeter, self.hand)

    def publish(self, now):
        self.buffer.publish(synthetic_keypoints(now, "right"), received_at=now,
                            timestamp_us=int(now * 1e6), sequence=int(now * 1e6))
        return self.buffer.snapshot()

    def test_loss_holds_and_recovery_resets(self):
        snapshot = self.publish(1)
        self.assertTrue(self.controller.update(snapshot, 1))
        self.assertFalse(self.controller.update(snapshot, 1.1))  # no reprocessing
        self.assertFalse(self.controller.update(snapshot, 1.3))
        self.assertEqual(self.controller.status, "stale")
        self.hand.set_target.assert_called_once()
        self.assertTrue(self.controller.update(self.publish(2), 2))
        self.assertEqual(self.retargeter.reset.call_count, 2)

    def test_pause_resume_and_invalid_input_hold(self):
        self.controller.update(self.publish(1), 1)
        self.controller.toggle_pause()
        self.controller.update(self.publish(2), 2)
        self.assertEqual(self.controller.status, "paused")
        self.hand.set_target.assert_called_once()
        self.controller.toggle_pause()
        self.controller.update(self.publish(3), 3)
        self.assertEqual(self.retargeter.reset.call_count, 2)
        self.buffer.fail("tracking error")
        self.controller.update(self.buffer.snapshot(), 3.01)
        self.assertEqual(self.controller.status, "invalid")
        self.assertEqual(self.hand.set_target.call_count, 2)
        self.controller.update(self.publish(4), 4)
        self.assertEqual(self.retargeter.reset.call_count, 3)

    def test_solver_failure_waits_for_new_frame(self):
        self.retargeter.step.side_effect = ValueError("bad pose")
        snapshot = self.publish(1)
        self.controller.update(snapshot, 1)
        self.assertEqual(self.controller.status, "invalid")
        self.hand.set_target.assert_not_called()
        self.controller.update(snapshot, 1.01)
        self.assertEqual(self.retargeter.step.call_count, 1)


def fake_sdk(devices=None, side="left", subscribe_error=None):
    devices = devices if devices is not None else [NS(sn="glove1", address="test", device_type="glove")]
    subscription = Mock()
    resource = Mock()
    resource.subscribe_with_callback.return_value = subscription
    resource.subscribe_with_callback.side_effect = subscribe_error
    glove = Mock()
    glove.hand_side.return_value.get.return_value = side
    glove.version.return_value.get.return_value = "test-firmware"
    glove.hand_skeleton.return_value = resource
    manager = Mock()
    manager.scan.return_value = devices
    manager.current_user.return_value = {"user_id": "previous"}
    manager.connect.return_value = glove
    sdk = NS(SdkManager=NS(instance=lambda: manager), DeviceType=NS(WujiGlove="glove"))
    return sdk, manager, glove, subscription


class GloveTests(unittest.TestCase):
    def test_multiple_gloves_require_serial(self):
        devices = [NS(sn="a"), NS(sn="b")]
        with self.assertRaisesRegex(RuntimeError, "Multiple gloves"):
            select_glove(devices)
        self.assertEqual(select_glove(devices, "b").sn, "b")
        with self.assertRaisesRegex(RuntimeError, "No matching"):
            select_glove(devices, "missing")

    def test_robot_hand_is_never_selected(self):
        sdk, manager, _, _ = fake_sdk([NS(sn="robot", address="test", device_type="hand")])
        with self.assertRaisesRegex(RuntimeError, "No matching"):
            GloveSource(sdk=sdk)
        manager.connect.assert_not_called()

    def test_callback_side_timestamp_and_cleanup(self):
        sdk, manager, glove, subscription = fake_sdk()
        source = GloveSource(sdk=sdk)
        self.assertEqual(source.side, "left")
        callback = glove.hand_skeleton().subscribe_with_callback.call_args.args[0]
        points = synthetic_keypoints(0, "left")
        skeleton = NS(header=NS(timestamp_us=time.time_ns() // 1000, seq=1),
                      joints=[NS(pose=NS(position=p), confidence=0.95) for p in points])
        callback(skeleton)
        self.assertEqual(source.read().received, 1)
        self.assertEqual(source.read().frame.keypoints.shape, (21, 3))
        skeleton.header.timestamp_us -= 100
        callback(skeleton)
        self.assertEqual(source.read().received, 1)
        error_callback = glove.hand_skeleton().subscribe_with_callback.call_args.kwargs["on_error"]
        error_callback("disconnected")
        self.assertEqual(source.read().error, "disconnected")
        source.close()
        source.close()
        subscription.close.assert_called_once()
        manager.disconnect.assert_called_once_with("wuji_sim_glove")
        self.assertEqual(manager.switch_user.call_args_list[0].args, ("",))
        self.assertEqual(manager.switch_user.call_args_list[-1].args, ("previous",))
        glove.enable.assert_not_called()

    def test_delayed_transport_frame_is_already_stale(self):
        sdk, _, glove, _ = fake_sdk()
        source = GloveSource(sdk=sdk)
        try:
            callback = glove.hand_skeleton().subscribe_with_callback.call_args.args[0]
            skeleton = NS(header=NS(timestamp_us=int((time.time() - 2) * 1e6), seq=1),
                          joints=[NS(pose=NS(position=p), confidence=1)
                                  for p in synthetic_keypoints(0, "left")])
            callback(skeleton)
            self.assertGreater(time.monotonic() - source.read().frame.received_at, 1.9)
        finally:
            source.close()

    def test_partial_connection_cleanup(self):
        sdk, manager, _, _ = fake_sdk(subscribe_error=RuntimeError("subscribe failed"))
        with self.assertRaisesRegex(RuntimeError, "subscribe failed"):
            GloveSource(sdk=sdk)
        manager.disconnect.assert_called_once()
        manager.switch_user.assert_called_with("previous")

    def test_cleanup_continues_when_subscription_close_fails(self):
        sdk, manager, _, subscription = fake_sdk()
        source = GloveSource(sdk=sdk)
        subscription.close.side_effect = RuntimeError("link gone")
        with self.assertWarns(RuntimeWarning):
            source.close()
        manager.disconnect.assert_called_once()
        manager.switch_user.assert_called_with("previous")


class ModelIntegrationTests(unittest.TestCase):
    def test_both_sides_retarget_and_step_with_contacts(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                model, data, hand = load_hand(side)
                retargeter = WujiRetargeter(side)
                targets = []
                for i in range(240):
                    target = retargeter.step(synthetic_keypoints(i / 40, side))
                    hand.set_target(target)
                    targets.append(hand.target.copy())
                    mujoco.mj_step(model, data, nstep=12)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertEqual(int(np.sum(data.warning.number)), 0)
                self.assertGreater(np.ptp(np.array(targets), axis=0).max(), 0.3)
                self.assertTrue(np.all(data.ctrl >= hand.lower))
                self.assertTrue(np.all(data.ctrl <= hand.upper))

    def test_left_hand_mount_points_out_of_fixed_vega_arm(self):
        from wuji_teleop.vega_hand_sim import load_vega_hand

        model, data, hand = load_vega_hand("left")
        wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wuji_left_wrist")
        ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L_ee")
        self.assertEqual(model.body_parentid[wrist], ee)
        self.assertEqual((model.nq, model.nv, model.nu), (20, 20, 20))
        # The hand's local -Z finger axis must point away from the end effector.
        wrist_finger_axis = -data.xmat[wrist].reshape(3, 3)[:, 2]
        ee_outward_axis = data.xmat[ee].reshape(3, 3)[:, 2]
        self.assertGreater(np.dot(wrist_finger_axis, ee_outward_axis), 0.99)
        self.assertEqual(len(hand.actuator_ids), 20)
        standalone, _, _ = load_hand("left")
        np.testing.assert_allclose(model.dof_armature, standalone.dof_armature)
        self.assertEqual(data.ncon, 0)
        for _ in range(500):
            mujoco.mj_step(model, data)
        self.assertEqual(int(np.sum(data.warning.number)), 0)
        self.assertLess(np.max(np.abs(data.qvel)), 0.01)

    def test_named_mapping_survives_actuator_reorder(self):
        from wuji_teleop.simulation import MODEL_DIR
        import xml.etree.ElementTree as ET
        root = ET.parse(MODEL_DIR / "right.xml").getroot()
        root.find("compiler").set("meshdir", str((MODEL_DIR / "../meshes/right").resolve()))
        actuator = root.find("actuator")
        actuator[:] = list(reversed(list(actuator)))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        hand = MujocoHand(model, data, "right")
        target = np.linspace(0.01, 0.20, 20)
        hand.set_target(target)
        np.testing.assert_allclose(data.ctrl[::-1], target)

    def test_limits_and_nonfinite_commands(self):
        _, data, hand = load_hand("right")
        hand.set_target(np.full(20, 1000))
        np.testing.assert_allclose(data.ctrl, hand.upper)
        before = data.ctrl.copy()
        for bad in (np.full(20, np.nan), np.zeros(19)):
            with self.assertRaises(ValueError):
                hand.set_target(bad)
            np.testing.assert_array_equal(before, data.ctrl)
        hand.set_target(np.full(20, -1000))
        np.testing.assert_allclose(data.ctrl, hand.lower)

    def test_missing_actuator_and_wrong_transmission_fail(self):
        model, data, _ = load_hand("right")
        model.actuator_trnid[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "mapping"):
            MujocoHand(model, data, "right")
        with self.assertRaisesRegex(ValueError, "20 named"):
            MujocoHand(model, data, "left")


class CliTests(unittest.TestCase):
    def test_headless_requires_duration_before_hardware_access(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["run", "--headless"])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_duration(self):
        for value in ("0", "-1", "nan", "inf"):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                main(["run", "--duration", value])

    @unittest.skipUnless(os.environ.get("WUJI_GUI_TEST") == "1", "opt-in graphical viewer test")
    def test_viewer_shutdown_does_not_race_glfw(self):
        result = subprocess.run(
            [sys.executable, "-m", "wuji_teleop", "run", "--input", "synthetic", "--duration", "1"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
