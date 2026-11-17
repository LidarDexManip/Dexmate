"""Read an XRoboToolkit headset recording (PICO 4 Ultra Enterprise).

The app's Data Collection (Tracking + Vision) writes to ``/sdcard/Download``:

- ``CameraRecord_<stamp>.mp4``: the stereo passthrough, left|right side by side
  (``PXRCapture_RenderMode_3D``) at the chosen resolution, e.g. 2160x810.
- ``trackingData_<stamp>.txt``: line 1 is JSON with ``timeStampNs`` (headset UTC
  when the first image was taken), ``cameraExtrinsics`` (two 4x4 row-major
  matrices, left|right, comma separated) and ``cameraIntrinsics``; every later
  line is one app frame: ``timeStampNs``, ``predictTime`` (microseconds, the
  clock the video frames are stamped in), ``Head.pose``, ``Motion.joints``
  (tracker ``sn`` and ``p``), and optionally ``Hand``. Poses are raw OpenXR
  ``x,y,z,qx,qy,qz,qw`` (x right, y up, z toward the user).

The camera conventions are not documented by PICO; every assumption is a named
option here, and ``inspect`` reports the values a recording actually holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from ..xr_tracker_source import XR_TO_ROBOT

# OpenXR camera (x right, y up, looking along -z) to OpenCV camera axes.
XR_CAMERA_TO_CV = np.diag([1.0, -1.0, -1.0])


def parse_pose(text):
    v = np.array([float(x) for x in str(text).split(",")], float)
    if v.shape != (7,):
        raise ValueError(f"pose needs 7 values, got {len(v)}")
    return v


def xr_matrix(pose7):
    """4x4 OpenXR pose (no axis change) from x y z qx qy qz qw."""
    out = np.eye(4)
    q = np.asarray(pose7[3:], float)
    out[:3, :3] = Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()
    out[:3, 3] = pose7[:3]
    return out


ROBOT_FROM_XR = np.eye(4)
ROBOT_FROM_XR[:3, :3] = XR_TO_ROBOT


def to_robot_world(T_xr):
    """Re-express a pose whose *reference* frame is the OpenXR world in robot axes.

    Only the world axes change; the body frame stays the device's own (for the
    head, the OpenXR head frame), so camera extrinsics in the head frame still
    apply on the right.
    """
    return ROBOT_FROM_XR @ np.asarray(T_xr, float)


@dataclass
class HeadsetRecording:
    header: dict
    t_ns: np.ndarray            # (N,) headset UTC ns per app frame
    predict_us: np.ndarray      # (N,)
    head: np.ndarray            # (N, 7) raw OpenXR, NaN when missing
    trackers: dict = field(default_factory=dict)   # serial -> (N, 7) raw, NaN when missing
    hands: dict = field(default_factory=dict)      # side -> (N, 26, 7) raw, NaN when inactive

    @property
    def seconds(self):
        return self.t_ns.astype(np.float64) / 1e9

    def extrinsics(self):
        """(left, right) 4x4 as written by the app (row-major)."""
        text = str(self.header.get("cameraExtrinsics", ""))
        parts = [p for p in text.split("|") if p.strip()]
        if len(parts) != 2:
            raise ValueError("cameraExtrinsics must hold a left and a right matrix")
        out = []
        for part in parts:
            v = np.array([float(x) for x in part.split(",")], float)
            if v.shape != (16,):
                raise ValueError("each camera extrinsic must have 16 values")
            out.append(v.reshape(4, 4))
        return out

    def intrinsics(self):
        text = str(self.header.get("cameraIntrinsics", ""))
        return np.array([float(x) for x in text.split(",") if x.strip()], float)


def read_tracking(path):
    """Parse ``trackingData_<stamp>.txt``."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = json.loads(lines[0])
    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a line cut short when the recording stopped
    n = len(rows)
    t_ns = np.array([int(r.get("timeStampNs", 0)) for r in rows], np.int64)
    predict = np.array([float(r.get("predictTime", np.nan)) for r in rows], float)
    head = np.full((n, 7), np.nan)
    trackers = {}
    hands = {}
    for i, r in enumerate(rows):
        pose = (r.get("Head") or {}).get("pose")
        if pose:
            head[i] = parse_pose(pose)
        motion = r.get("Motion") or {}
        joints = motion.get("joints") or []
        for j in joints[:int(motion.get("len", len(joints)))]:
            sn = str(j.get("sn", "")).strip()
            if sn and j.get("p"):
                trackers.setdefault(sn, np.full((n, 7), np.nan))[i] = parse_pose(j["p"])
        for side, key in (("left", "leftHand"), ("right", "rightHand")):
            hand = (r.get("Hand") or {}).get(key) or {}
            locations = hand.get("HandJointLocations") or []
            if hand.get("isActive") and len(locations) >= 26:
                arr = hands.setdefault(side, np.full((n, 26, 7), np.nan))
                arr[i] = [parse_pose(x["p"]) for x in locations[:26]]
    if n and np.any(np.diff(t_ns) <= 0):
        keep = np.r_[True, np.diff(t_ns) > 0]
        t_ns, predict, head = t_ns[keep], predict[keep], head[keep]
        trackers = {k: v[keep] for k, v in trackers.items()}
        hands = {k: v[keep] for k, v in hands.items()}
    return HeadsetRecording(header, t_ns, predict, head, trackers, hands)


def video_pts(path):
    """Presentation timestamps (s) of every video frame, in presentation order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    pts = np.array(sorted(float(x) for x in out.split() if x and x != "N/A"), float)
    if not len(pts):
        raise ValueError(f"No video frames in {path}")
    return pts


def video_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def frame_times_utc(recording, pts, mode="auto"):
    """Headset UTC seconds of each video frame.

    ``first-image``: the header's ``timeStampNs`` is the first frame; later
    frames follow by PTS. ``predict``: PTS are ``predictTime`` microseconds and
    are mapped to UTC by the app lines' (predictTime, timeStampNs) pairs.
    ``auto`` picks ``predict`` when the PTS lie inside the predictTime range.
    """
    pts = np.asarray(pts, float)
    predict_s = recording.predict_us / 1e6
    good = np.isfinite(predict_s)
    if mode == "auto":
        inside = good.any() and predict_s[good].min() - 1 <= pts[0] <= predict_s[good].max() + 1
        mode = "predict" if inside else "first-image"
    if mode == "predict":
        fit = np.polyfit(predict_s[good], recording.seconds[good], 1)
        return np.polyval(fit, pts), mode
    if mode == "first-image":
        return recording.header["timeStampNs"] / 1e9 + (pts - pts[0]), mode
    raise ValueError(f"unknown frame time mode {mode}")


@dataclass
class StereoCamera:
    """Per-eye pinhole cameras of the side-by-side video, in the head frame."""
    width: int                  # per eye
    height: int
    K_left: np.ndarray
    K_right: np.ndarray
    T_head_left: np.ndarray     # OpenXR head frame <- OpenCV left camera
    T_head_right: np.ndarray

    @property
    def T_left_right(self):
        return np.linalg.inv(self.T_head_left) @ self.T_head_right


def stereo_camera(recording, video_width, video_height, *, extrinsic="head_from_camera",
                  intrinsics="auto", camera_axes="openxr"):
    """Interpret the header's camera parameters.

    ``extrinsic``: ``head_from_camera`` (each matrix maps camera to head
    coordinates) or ``camera_from_head`` (inverted). ``camera_axes``: the
    extrinsic camera frame uses OpenXR axes (looking along -z) or ``opencv``.
    ``intrinsics``: ``per-eye`` (fx fy cx cy of one eye image), ``full`` (of
    the whole side-by-side frame), or ``auto`` (``full`` when fx is about
    twice fy, as for one focal length over a double-width frame).
    """
    left, right = recording.extrinsics()
    if extrinsic == "camera_from_head":
        left, right = np.linalg.inv(left), np.linalg.inv(right)
    elif extrinsic != "head_from_camera":
        raise ValueError(extrinsic)
    axes = np.eye(4)
    if camera_axes == "openxr":
        axes[:3, :3] = XR_CAMERA_TO_CV.T  # OpenCV camera -> OpenXR camera
    elif camera_axes != "opencv":
        raise ValueError(camera_axes)
    values = recording.intrinsics()
    if len(values) < 4:
        raise ValueError("cameraIntrinsics needs at least fx, fy, cx, cy")
    fx, fy, cx, cy = values[:4]
    eye_w = video_width // 2
    if intrinsics == "auto":
        intrinsics = "full" if abs(fx / fy - 2.0) < 0.25 else "per-eye"
    if intrinsics == "full":
        fx, cx = fx / 2.0, cx / 2.0 if cx > eye_w else cx
    elif intrinsics != "per-eye":
        raise ValueError(intrinsics)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return StereoCamera(eye_w, video_height, K, K.copy(), left @ axes, right @ axes)


def find_recording(directory):
    """(tracking txt, video mp4) in a pulled headset folder; newest pair wins."""
    directory = Path(directory)
    texts = sorted(directory.glob("trackingData_*.txt"))
    videos = sorted(directory.glob("CameraRecord_*.mp4"))
    if not texts:
        raise ValueError(f"No trackingData_*.txt in {directory}")
    if not videos:
        raise ValueError(f"No CameraRecord_*.mp4 in {directory}")
    return texts[-1], videos[-1]


def inspect(directory):
    """A plain report of what a pulled headset recording contains."""
    text, video = find_recording(directory)
    rec = read_tracking(text)
    w, h = video_size(video)
    pts = video_pts(video)
    times, mode = frame_times_utc(rec, pts)
    s = rec.seconds
    lines = [f"tracking {text.name}: {len(s)} app frames over {s[-1] - s[0]:.1f} s "
             f"({(len(s) - 1) / max(s[-1] - s[0], 1e-9):.1f} Hz)",
             f"  head pose in {np.isfinite(rec.head[:, 0]).mean():.0%} of frames",
             f"  trackers: " + ", ".join(f"{k} ({np.isfinite(v[:, 0]).mean():.0%})"
                                         for k, v in rec.trackers.items()),
             f"  hands: " + (", ".join(f"{k} ({np.isfinite(v[:, 0, 0]).mean():.0%})"
                                       for k, v in rec.hands.items()) or "none"),
             f"  intrinsics: {rec.intrinsics().tolist()}"]
    try:
        for name, m in zip(("left", "right"), rec.extrinsics()):
            lines.append(f"  extrinsic {name}: t={np.round(m[:3, 3], 4).tolist()} "
                         f"rot={np.round(Rotation.from_matrix(m[:3, :3]).as_rotvec(), 3).tolist()}")
    except ValueError as exc:
        lines.append(f"  extrinsics: {exc}")
    lines.append(f"video {video.name}: {w}x{h}, {len(pts)} frames, PTS {pts[0]:.3f}..{pts[-1]:.3f} s "
                 f"({(len(pts) - 1) / max(pts[-1] - pts[0], 1e-9):.1f} fps), time mode {mode}, "
                 f"frames span UTC {times[0] - s[0]:+.3f}..{times[-1] - s[-1]:+.3f} s "
                 "relative to the tracking lines")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(inspect(sys.argv[1]))
