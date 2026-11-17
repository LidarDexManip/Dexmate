"""Depth from the headset's side-by-side stereo passthrough video.

Each video frame is left|right. The pair is rectified from the recorded
intrinsics and the left-to-right extrinsic, matched with semi-global block
matching plus a weighted-least-squares filter (OpenCV contrib), and converted
to metric depth in the rectified left camera. The depth stage saves, per video
frame, the rectified left image (JPEG) and depth (16-bit PNG, millimetres) at
``scale`` of the eye resolution, which the tracker and viewer reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Rectified:
    K: np.ndarray            # 3x3 rectified left intrinsics (at the stored scale)
    width: int
    height: int
    baseline: float          # metres
    T_left_rect: np.ndarray  # original left camera <- rectified left camera


class StereoDepth:
    def __init__(self, camera, *, scale=0.5, min_depth=0.15, max_depth=2.0, wls=True):
        w, h = camera.width, camera.height
        T_rl = np.linalg.inv(camera.T_left_right)  # right <- left
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            camera.K_left, np.zeros(5), camera.K_right, np.zeros(5), (w, h),
            np.ascontiguousarray(T_rl[:3, :3]), np.ascontiguousarray(T_rl[:3, 3:4]),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.maps = [cv2.initUndistortRectifyMap(K, np.zeros(5), R, P, (w, h), cv2.CV_32FC1)
                     for K, R, P in ((camera.K_left, R1, P1), (camera.K_right, R2, P2))]
        self.scale = float(scale)
        sw, sh = int(round(w * scale)), int(round(h * scale))
        K = P1[:3, :3] * np.array([[scale], [scale], [1.0]])
        baseline = float(-P2[0, 3] / P2[0, 0])
        if baseline <= 0:
            raise ValueError(f"Rectified baseline {baseline:.4f} m is not positive; check the "
                             "extrinsic convention (headset.stereo_camera)")
        T = np.eye(4)
        T[:3, :3] = R1.T
        self.rect = Rectified(K, sw, sh, baseline, T)
        self.min_depth, self.max_depth = min_depth, max_depth
        fb = K[0, 0] * baseline
        disparities = int(np.ceil(fb / min_depth / 16.0) * 16)
        self.num_disparities = int(np.clip(disparities, 32, 256))
        block = 5
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.num_disparities, blockSize=block,
            P1=8 * 3 * block * block, P2=32 * 3 * block * block, disp12MaxDiff=1,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.wls = None
        if wls and hasattr(cv2, "ximgproc"):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls.setLambda(8000.0)
            self.wls.setSigmaColor(1.2)

    def rectify(self, left, right):
        out = []
        for image, (mx, my) in zip((left, right), self.maps):
            r = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
            if self.scale != 1.0:
                r = cv2.resize(r, (self.rect.width, self.rect.height), interpolation=cv2.INTER_AREA)
            out.append(r)
        return out

    def depth(self, left_rect, right_rect):
        """Metric depth (H,W float32, 0 invalid) in the rectified left camera."""
        gl = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)
        disp_l = self.left_matcher.compute(gl, gr)
        if self.wls is not None:
            disp_r = self.right_matcher.compute(gr, gl)
            disp = self.wls.filter(disp_l, gl, disparity_map_right=disp_r)
            conf = self.wls.getConfidenceMap()
        else:
            disp, conf = disp_l, None
        disp = disp.astype(np.float32) / 16.0
        depth = np.zeros_like(disp)
        good = disp > 0.5
        depth[good] = self.rect.K[0, 0] * self.rect.baseline / disp[good]
        good &= (depth >= self.min_depth) & (depth <= self.max_depth)
        if conf is not None:
            good &= conf > 80
        depth[~good] = 0
        return depth


def split_side_by_side(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    w = rgb.shape[1] // 2
    return rgb[:, :w], rgb[:, w:2 * w]


def save_frame(directory, index, rgb, depth):
    directory = Path(directory)
    cv2.imwrite(str(directory / f"{index:06d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(directory / f"{index:06d}.png"), mm)


def load_frame(directory, index):
    """(rgb uint8, depth float32 metres) saved by the depth stage."""
    directory = Path(directory)
    bgr = cv2.imread(str(directory / f"{index:06d}.jpg"), cv2.IMREAD_COLOR)
    mm = cv2.imread(str(directory / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or mm is None:
        raise FileNotFoundError(f"frame {index} missing in {directory}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mm.astype(np.float32) / 1000.0


from .object_track import backproject  # noqa: E402,F401  (re-exported)
