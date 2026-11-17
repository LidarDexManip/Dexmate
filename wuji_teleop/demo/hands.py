"""World-frame human hands from wrist trackers, palm IMUs and glove skeletons.

The PICO trackers sit on the back of each forearm (watch-style), so they give
the forearm, not the hand. The glove skeleton is in the hand's own wrist frame
and its fused palm IMU gives the hand's orientation in an IMU world frame. A
short calibration with the wrists held straight relates the two:

    R_world_tracker(t) @ X = Y @ R_imuworld_wrist(t)     (wrist straight)

``X`` is the tracker-to-hand mounting rotation and ``Y`` the IMU world in the
PICO world, solved together (AX = YB). Afterwards the hand's orientation is
``Y @ R_imuworld_wrist(t)`` whatever the wrist does, and the wrist joint centre
is the tracker position plus a fixed tracker-local offset.

``HandSurface`` builds a closed capsule surface (finger bones plus a palm slab)
of fixed topology from the 21 keypoints, standing in for a MANO skin in the
held-segment refinement's penetration and contact terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..xr_tracker_source import xr_pose_to_robot
from . import log as demo_log

SIDES = ("left", "right")
FINGER_BONES = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)]
BONE_RADII = [0.0110, 0.0100, 0.0090,   # thumb
              0.0095, 0.0085, 0.0075,   # index
              0.0095, 0.0085, 0.0075,   # middle
              0.0090, 0.0080, 0.0072,   # ring
              0.0085, 0.0075, 0.0068]   # pinky
PALM_RING = [0, 1, 5, 9, 13, 17]
PALM_HALF_THICKNESS = 0.012
TIPS = (4, 8, 12, 16, 20)


# --------------------------------------------------------------------- streams

def glove_series(pc_dir, side):
    """(t, keypoints (N,21,3), confidence (N,21)) on host UTC from the SDK stamp."""
    g = demo_log.load(pc_dir, "glove")
    g = g[g["side"] == SIDES.index(side)]
    t = g["stamp_us"].astype(np.float64) / 1e6
    order = np.argsort(t, kind="stable")
    keep = np.r_[True, np.diff(t[order]) > 0]
    g = g[order][keep]
    return g["stamp_us"].astype(np.float64) / 1e6, g["keypoints"].astype(np.float64), \
        g["confidence"].astype(np.float64)


def imu_series(pc_dir, side):
    """(t, Rotation of the palm IMU in its world frame)."""
    m = demo_log.load(pc_dir, "imu")
    m = m[m["side"] == SIDES.index(side)]
    t = m["stamp_us"].astype(np.float64) / 1e6
    order = np.argsort(t, kind="stable")
    t, q = t[order], m["quat"][order]
    keep = np.r_[True, np.diff(t) > 0] & (np.linalg.norm(q, axis=1) > 0.5)
    return t[keep], Rotation.from_quat(q[keep])


def tracker_poses(raw7):
    """(N,4,4) robot-axis world poses from raw OpenXR rows (NaN rows stay NaN)."""
    out = np.full((len(raw7), 4, 4), np.nan)
    for i, row in enumerate(raw7):
        if np.isfinite(row).all() and np.linalg.norm(row[3:]) > 0.5:
            out[i] = xr_pose_to_robot(row / np.r_[1, 1, 1, [np.linalg.norm(row[3:])] * 4])
    return out


def _gaps(t_src, t_query):
    """Seconds between each query and its nearest source sample on either side."""
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    span = t_src[i] - t_src[i - 1]
    outside = (t_query < t_src[0]) | (t_query > t_src[-1])
    return np.where(outside, np.inf, span)


def interp_array(t_src, x_src, t_query, max_gap=0.1):
    """Linear interpolation of (N,...) samples; invalid where the bracket is wider than max_gap."""
    flat = x_src.reshape(len(x_src), -1)
    good = np.isfinite(flat).all(axis=1)
    t, flat = t_src[good], flat[good]
    if len(t) < 2:
        return np.full((len(t_query),) + x_src.shape[1:], np.nan), np.zeros(len(t_query), bool)
    out = np.column_stack([np.interp(t_query, t, flat[:, k]) for k in range(flat.shape[1])])
    valid = _gaps(t, t_query) <= max_gap
    out[~valid] = np.nan
    return out.reshape((len(t_query),) + x_src.shape[1:]), valid


def interp_rotation(t_src, rot_src, t_query, max_gap=0.1):
    """Slerp; returns (matrices (M,3,3) NaN where invalid, valid)."""
    out = np.full((len(t_query), 3, 3), np.nan)
    if len(t_src) < 2:
        return out, np.zeros(len(t_query), bool)
    valid = _gaps(t_src, t_query) <= max_gap
    if valid.any():
        out[valid] = Slerp(t_src, rot_src)(t_query[valid]).as_matrix()
    return out, valid


def interp_poses(t_src, T_src, t_query, max_gap=0.1):
    good = np.isfinite(T_src.reshape(len(T_src), -1)).all(axis=1)
    t, T = t_src[good], T_src[good]
    out = np.full((len(t_query), 4, 4), np.nan)
    if len(t) < 2:
        return out, np.zeros(len(t_query), bool)
    R, valid = interp_rotation(t, Rotation.from_matrix(T[:, :3, :3]), t_query, max_gap)
    p, valid_p = interp_array(t, T[:, :3, 3], t_query, max_gap)
    valid &= valid_p
    out[valid, :3, :3] = R[valid]
    out[valid, :3, 3] = p[valid]
    out[valid, 3] = [0, 0, 0, 1]
    return out, valid


# ----------------------------------------------------------------- calibration

def project_so3(M):
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def solve_axyb(A, B):
    """Rotations X, Y minimizing sum ||A_i X - Y B_i||^2 (needs rotation about 2+ axes)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    eye = np.eye(3)
    rows = [np.hstack([np.kron(eye, a), -np.kron(b.T, eye)]) for a, b in zip(A, B)]
    _, s, vt = np.linalg.svd(np.vstack(rows))
    v = vt[-1]
    X = v[:9].reshape(3, 3, order="F")
    Y = v[9:].reshape(3, 3, order="F")
    scale = np.sign(np.linalg.det(X)) * abs(np.linalg.det(X)) ** (-1 / 3)
    X, Y = project_so3(X * scale), project_so3(Y * scale)
    return X, Y, {"singular_ratio": float(s[-1] / max(s[-2], 1e-12))}


def rotation_spread_deg(R):
    """How far the rotations spread (max angle from their mean), for calibration quality."""
    mean = Rotation.from_matrix(R).mean()
    return float(np.rad2deg(np.max((mean.inv() * Rotation.from_matrix(R)).magnitude())))


@dataclass
class WristCalibration:
    X: np.ndarray            # tracker <- wrist (neutral) rotation
    Y: np.ndarray            # PICO world <- IMU world rotation
    offset: np.ndarray       # tracker-local vector from tracker to wrist joint centre
    imu_in_wrist: np.ndarray  # wrist <- palm IMU (SDK tf_static)
    report: dict

    def to_json(self):
        return {"X": self.X.tolist(), "Y": self.Y.tolist(), "offset": self.offset.tolist(),
                "imu_in_wrist": self.imu_in_wrist.tolist(), "report": self.report}

    @classmethod
    def from_json(cls, data):
        return cls(np.asarray(data["X"]), np.asarray(data["Y"]), np.asarray(data["offset"]),
                   np.asarray(data["imu_in_wrist"]), data.get("report", {}))


def wrist_in_imu_world(imu_R, imu_in_wrist):
    """R_imuworld_wrist from R_imuworld_imu and the SDK's wrist <- IMU rotation."""
    return imu_R @ np.asarray(imu_in_wrist).T


def calibrate_wrist(t_tracker, T_tracker, t_imu, imu_rot, imu_in_wrist, forward_local,
                    still, rotate, *, mount_height=0.025, mount_back=0.02, step=0.05):
    """Solve X, Y over the ``rotate`` (and ``still``) windows; offset from ``still``.

    ``still`` and ``rotate`` are (start, end) host-UTC windows. ``forward_local``
    is the hand's wrist-to-middle-knuckle direction in the glove wrist frame.
    The wrist joint centre is taken ``mount_height`` below the tracker and
    ``mount_back`` ahead of it along the hand, with the forearm level and palm
    down during ``still``.
    """
    samples = np.concatenate([np.arange(*rotate, step), np.arange(*still, step)])
    A, valid_a = interp_poses(t_tracker, T_tracker, samples, 0.1)
    Bimu, valid_b = interp_rotation(t_imu, imu_rot, samples, 0.1)
    valid = valid_a & valid_b
    if valid.sum() < 20:
        raise ValueError("Calibration windows lack tracker or palm IMU samples")
    A, Bimu = A[valid, :3, :3], Bimu[valid]
    B = np.array([wrist_in_imu_world(b, imu_in_wrist) for b in Bimu])
    spread = rotation_spread_deg(A)
    if spread < 30:
        raise ValueError(f"The forearms turned only {spread:.0f} deg during calibration; "
                         "turn them further (palms up/down, point sideways and up)")
    X, Y, info = solve_axyb(A, B)
    residual = np.rad2deg(Rotation.from_matrix(
        np.einsum("nij,jk->nik", A, X).transpose(0, 2, 1) @ (Y @ B)).magnitude())
    # Wrist offset from the still window.
    s = np.arange(*still, step)
    Ts, vs = interp_poses(t_tracker, T_tracker, s, 0.1)
    Bs, vb = interp_rotation(t_imu, imu_rot, s, 0.1)
    keep = vs & vb
    if keep.sum() < 5:
        raise ValueError("The still window lacks tracker or palm IMU samples")
    forward = np.array([Y @ wrist_in_imu_world(b, imu_in_wrist) @ forward_local for b in Bs[keep]])
    forward[:, 2] = 0
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)
    d_world = -mount_height * np.array([0, 0, 1.0]) + mount_back * forward
    offset = np.mean([T[:3, :3].T @ d for T, d in zip(Ts[keep], d_world)], axis=0)
    report = {"samples": int(valid.sum()), "forearm_rotation_spread_deg": spread,
              "axyb_residual_deg_median": float(np.median(residual)),
              "axyb_residual_deg_p90": float(np.percentile(residual, 90)),
              "mount_height_m": mount_height, "mount_back_m": mount_back, **info}
    return WristCalibration(X, Y, offset, np.asarray(imu_in_wrist), report)


def twist_angle(R_rel, axis):
    """Signed rotation (rad) of each relative rotation about ``axis`` (swing-twist)."""
    q = Rotation.from_matrix(R_rel).as_quat()  # x y z w
    return 2 * np.arctan2(q[:, :3] @ axis, q[:, 3])


# ------------------------------------------------------------------ world hand

@dataclass
class WorldHand:
    keypoints: np.ndarray   # (N,21,3) world
    wrist: np.ndarray       # (N,4,4) world <- glove wrist frame
    valid: np.ndarray       # (N,)
    confidence: np.ndarray  # (N,21) glove joint confidence
    wrist_angle_deg: np.ndarray  # (N,) hand relative to the calibrated straight wrist
    twist_deg: np.ndarray   # (N,) twist about the forearm axis (should stay near 0)


def world_hand(times, t_tracker, T_tracker, t_imu, imu_rot, t_glove, keypoints, confidence,
               calibration, *, orientation="imu", max_gap=0.1, min_confidence=0.0):
    """World-frame keypoints at ``times``.

    ``orientation='imu'`` uses the palm IMU for the hand; ``'tracker'`` assumes
    a straight wrist (hand rigid with the forearm).
    """
    times = np.asarray(times, float)
    A, va = interp_poses(t_tracker, T_tracker, times, max_gap)
    kp, vk = interp_array(t_glove, keypoints, times, max_gap)
    conf, _ = interp_array(t_glove, confidence, times, max_gap)
    valid = va & vk
    R_hand = np.full((len(times), 3, 3), np.nan)
    if orientation == "imu":
        Bimu, vb = interp_rotation(t_imu, imu_rot, times, max_gap)
        valid &= vb
        R_hand[valid] = calibration.Y @ (Bimu[valid] @ calibration.imu_in_wrist.T)
    elif orientation == "tracker":
        R_hand[valid] = A[valid, :3, :3] @ calibration.X
    else:
        raise ValueError(orientation)
    if min_confidence > 0:
        valid &= np.nan_to_num(conf).min(axis=1) >= min_confidence
    wrist = np.full((len(times), 4, 4), np.nan)
    wrist[valid, :3, :3] = R_hand[valid]
    wrist[valid, :3, 3] = A[valid, :3, 3] + np.einsum("nij,j->ni", A[valid, :3, :3],
                                                       calibration.offset)
    wrist[valid, 3] = [0, 0, 0, 1]
    local = kp - kp[:, :1]
    world = np.full_like(kp, np.nan)
    world[valid] = np.einsum("nij,nkj->nki", wrist[valid, :3, :3], local[valid]) \
        + wrist[valid, None, :3, 3]
    angle = np.full(len(times), np.nan)
    twist = np.full(len(times), np.nan)
    if valid.any():
        rel = np.einsum("ji,njk->nik", calibration.X, A[valid, :3, :3].transpose(0, 2, 1)
                        @ R_hand[valid])
        angle[valid] = np.rad2deg(Rotation.from_matrix(rel).magnitude())
        forward = np.nanmedian(local[valid][:, 9], axis=0)
        forward /= np.linalg.norm(forward)
        twist[valid] = np.rad2deg(twist_angle(rel, forward))
    return WorldHand(world, wrist, valid, conf, angle, twist)


def palm_pose(keypoints):
    """(N,4,4) anatomical palm frames (retargeting's ``palm_frame``), NaN where invalid."""
    k = np.asarray(keypoints, float)
    out = np.full((len(k), 4, 4), np.nan)
    x = k[:, 5] - k[:, 17]
    y = k[:, 9] - k[:, 0]
    nx = np.linalg.norm(x, axis=1)
    good = np.isfinite(k).all(axis=(1, 2)) & (nx > 0.005)
    x = x / np.where(nx > 0, nx, 1)[:, None]
    y = y - x * np.einsum("ij,ij->i", x, y)[:, None]
    ny = np.linalg.norm(y, axis=1)
    good &= ny > 0.005
    y = y / np.where(ny > 0, ny, 1)[:, None]
    z = np.cross(x, y)
    out[good, :3, :3] = np.stack([x, y, z], axis=-1)[good]
    out[good, :3, 3] = k[good, 0]
    out[good, 3] = [0, 0, 0, 1]
    return out


# --------------------------------------------------------------- hand surface

def _capsule_template(segments=12, cap_rings=3):
    """Unit capsule: (end index, unit offset) per vertex, and outward faces."""
    ends, offsets = [0], [(0.0, 0.0, -1.0)]
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = []
    for j in range(1, cap_rings + 1):
        phi = j * (np.pi / 2) / cap_rings
        rings.append((0, np.sin(phi), -np.cos(phi)))
    for j in range(cap_rings, 0, -1):
        phi = j * (np.pi / 2) / cap_rings
        rings.append((1, np.sin(phi), np.cos(phi)))
    for end, radial, axial in rings:
        for th in theta:
            ends.append(end)
            offsets.append((radial * np.cos(th), radial * np.sin(th), axial))
    ends.append(1)
    offsets.append((0.0, 0.0, 1.0))
    faces = []
    first = 1
    for k in range(segments):
        faces.append((0, first + (k + 1) % segments, first + k))
    for r in range(len(rings) - 1):
        a, b = first + r * segments, first + (r + 1) * segments
        for k in range(segments):
            k2 = (k + 1) % segments
            faces += [(a + k, a + k2, b + k2), (a + k, b + k2, b + k)]
    last = first + (len(rings) - 1) * segments
    top = len(offsets) - 1
    for k in range(segments):
        faces.append((top, last + k, last + (k + 1) % segments))
    return np.array(ends), np.array(offsets), np.array(faces)


def _signed_volume(vertices, faces):
    tri = vertices[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6)


class HandSurface:
    """Closed capsule hand surface with the same faces in every frame."""

    def __init__(self, sample_keypoints, segments=12, cap_rings=3):
        self.ends, self.offsets, cap_faces = _capsule_template(segments, cap_rings)
        n = len(self.offsets)
        probe = np.zeros((n, 3))
        probe[:] = self.offsets * 0.01 + np.where(self.ends[:, None] == 1, [0, 0, 0.05], 0)
        if _signed_volume(probe, cap_faces) < 0:
            cap_faces = cap_faces[:, ::-1]
        faces = [cap_faces + b * n for b in range(len(FINGER_BONES))]
        base = n * len(FINGER_BONES)
        m = len(PALM_RING)
        palm = []
        for k in range(1, m - 1):
            palm.append((base, base + k, base + k + 1))            # top (+normal)
            palm.append((base + m, base + m + k + 1, base + m + k))  # bottom
        for k in range(m):
            k2 = (k + 1) % m
            palm += [(base + k, base + m + k, base + m + k2), (base + k, base + m + k2, base + k2)]
        palm = np.array(palm)
        sample = self._vertices(np.asarray(sample_keypoints, float)[None])[0]
        if _signed_volume(sample[base:], palm - base) < 0:
            palm = palm[:, ::-1]
        self.faces = np.vstack(faces + [palm]).astype(np.int64)
        self.vertex_count = base + 2 * m
        self.bone_of_vertex = np.r_[np.repeat(np.arange(len(FINGER_BONES)), n), np.full(2 * m, -1)]

    def _vertices(self, keypoints):
        k = np.asarray(keypoints, float)
        out = []
        for (a, b), r in zip(FINGER_BONES, BONE_RADII):
            p0, p1 = k[:, a], k[:, b].copy()
            axis = p1 - p0
            length = np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-6)
            axis = axis / length
            if b in TIPS:  # keep the fingertip surface at the tip keypoint
                p1 = p1 - axis * np.minimum(r, 0.5 * length)
            helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
            e1 = np.cross(axis, helper)
            e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
            e2 = np.cross(axis, e1)
            R = np.stack([e1, e2, axis], axis=-1)
            ends = np.where(self.ends[None, :, None] == 0, p0[:, None], p1[:, None])
            out.append(ends + r * np.einsum("nij,vj->nvi", R, self.offsets))
        ring = k[:, PALM_RING]
        x = k[:, 5] - k[:, 17]
        y = k[:, 9] - k[:, 0]
        normal = np.cross(x, y)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        out.append(ring + PALM_HALF_THICKNESS * normal[:, None])
        out.append(ring - PALM_HALF_THICKNESS * normal[:, None])
        return np.concatenate(out, axis=1)

    def vertices(self, keypoints):
        """(N, V, 3) surface vertices for (N, 21, 3) keypoints (NaN stays NaN)."""
        return self._vertices(keypoints)
