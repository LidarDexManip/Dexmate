"""Layout and model discovery of the camera-view layer inspector (no window)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _session(root, n=3, w=64, h=48):
    root = Path(root)
    (root / "stereo" / "left").mkdir(parents=True)
    (root / "stereo" / "disparity").mkdir(parents=True)
    (root / "processed").mkdir()
    K = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    cam = np.eye(4)
    cam[:3, :3] = np.diag([1.0, -1.0, -1.0])  # camera at the origin looking down -z
    z = {"frame": np.arange(n), "t_capture": np.arange(n) / 30.0, "K": K, "image_size": np.array([w, h]),
         "baseline_m": np.array(0.064), "disparity_scale": np.array(64.0),
         "T_world_camera": np.repeat(cam[None], n, 0), "has_disparity": np.ones(n, bool)}
    for side, x in (("left", -0.05), ("right", 0.05)):
        tr = np.repeat(np.eye(4)[None], n, 0)
        tr[:, :3, 3] = [x, 0.0, -0.4]
        kp = np.zeros((n, 21, 3))
        kp[:] = [x, 0.0, -0.4]  # keypoint 0 is the wrist, as in synced.npz
        kp[:, 1:, 1] += np.linspace(0.01, 0.08, 20)
        z.update({f"tracker_{side}": tr, f"tracker_valid_{side}": np.ones(n, bool), f"wrist_{side}": tr.copy(),
                  f"hand_valid_{side}": np.ones(n, bool), f"keypoints_{side}": kp})
    np.savez(root / "processed" / "synced.npz", **z)
    V = np.array([[-0.01, 0, -0.4], [0.01, 0, -0.4], [0, 0.02, -0.4]], np.float32)
    np.savez(root / "processed" / "mano_left.npz", vertices=np.repeat(V[None], n, 0), faces=np.array([[0, 1, 2]]),
             valid=np.ones(n, bool), joints21=np.zeros((n, 21, 3), np.float32),
             depth_R=np.repeat(np.eye(3)[None], n, 0), depth_t=np.zeros((n, 3)))
    np.savez(root / "processed" / "sync_check.npz", other=np.zeros(1))  # not a model
    for i in range(n):
        cv2.imwrite(str(root / "stereo" / "left" / f"{i:06d}.jpg"), np.full((h, w, 3), 90, np.uint8))
        cv2.imwrite(str(root / "stereo" / "disparity" / f"{i:06d}.png"), np.full((h, w), 64 * 8, np.uint16))
    return root


@unittest.skipIf(cv2 is None, "needs OpenCV")
class LayerViewTests(unittest.TestCase):
    def test_finds_models_and_lays_panels_side_by_side(self):
        from wuji_teleop.demo import layer_view as lv

        with tempfile.TemporaryDirectory() as tmp:
            scene = lv.Scene(_session(tmp))
            self.assertEqual(list(scene.models), ["mano"])
            self.assertEqual(set(scene.models["mano"]), {"left"})
            on = {k: True for k, _ in lv.LAYERS}
            both = lv.compose(scene, 1, on, list(lv.SIDES), "mano", 400, 150)
            self.assertEqual(both.shape, (150, 400, 3))
            # Two 4:3 panels side by side fit the width, not the height.
            scale = (400 - lv.GAP_PX) / (2 * 64)
            gap_col = int((400 - (2 * int(64 * scale) + lv.GAP_PX)) / 2 + int(64 * scale) + lv.GAP_PX // 2)
            self.assertTrue((both[:, gap_col] == 24).all(), "panels overlap")
            off = dict(on, depth=False)
            single = lv.compose(scene, 1, off, ["right"], "none", 400, 150)
            self.assertEqual(single.shape, (150, 400, 3))

    def test_hand_follows_the_depth_correction(self):
        from wuji_teleop.demo import layer_view as lv

        with tempfile.TemporaryDirectory() as tmp:
            root = _session(tmp)
            d = dict(np.load(root / "processed" / "mano_left.npz"))
            d["depth_t"][:] = [0.0, 0.01, 0.0]
            np.savez(root / "processed" / "mano_left.npz", **d)
            scene = lv.Scene(root)
            h = scene.hand(0, "left", scene.model("mano"))
            np.testing.assert_allclose(h["wrist"] - h["wrist_before"], [0, 0.01, 0])
            np.testing.assert_allclose(h["keypoints"][0], h["wrist"])
            raw = scene.hand(0, "left", scene.model("none"))
            np.testing.assert_allclose(raw["wrist"], raw["wrist_before"])


if __name__ == "__main__":
    unittest.main()
