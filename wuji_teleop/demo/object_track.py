"""Rigid object tracking in the headset depth, guided by the hands.

Each frame fits the object mesh to the depth points near its predicted pose by
robust point-to-plane ICP (scene points matched to front-facing model samples).
Depth points on the hands are removed first, using the world-frame hand
capsules. The prediction uses the hand: while the object is within grasp
distance of a hand it is carried by that hand's wrist motion, otherwise it
keeps its last pose. A frame whose fit has too few inliers or too large an
error is marked invalid and keeps the prediction, so a hand that occludes the
object never drags it onto the fingers.

This is the measurement the held-segment refinement (``ground``) then corrects;
FoundationPose could replace it through retargeting's client later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation



def backproject(depth, K, stride=1):
    """(M,3) camera-frame points of valid depth pixels (OpenCV axes) and their (v,u)."""
    d = depth[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    z = d[v, u]
    v, u = v * stride, u * stride
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.column_stack([x, y, z]), np.column_stack([v, u])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _exp(xi):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(xi[3:]).as_matrix()
    T[:3, 3] = xi[:3]
    return T


class ObjectModel:
    def __init__(self, mesh, samples=4000, seed=0):
        import trimesh
        points, faces = trimesh.sample.sample_surface_even(mesh, samples, seed=seed)
        self.points = np.asarray(points, float)
        self.normals = np.asarray(mesh.face_normals[faces], float)
        self.tree = cKDTree(self.points)
        self.radius = float(np.linalg.norm(self.points - self.points.mean(0), axis=1).max())
        self.center = self.points.mean(0)


@dataclass
class FitResult:
    pose: np.ndarray      # camera <- object
    inliers: int
    rms: float
    expected: int


def icp(model, scene, pose, *, iterations=20, max_distance=(0.02, 0.008), huber=0.004,
        camera_origin=np.zeros(3)):
    """Point-to-plane ICP of camera-frame ``scene`` points to the model; returns FitResult."""
    pose = np.asarray(pose, float).copy()
    if len(scene) < 10:
        return FitResult(pose, 0, np.inf, 0)
    rms, count = np.inf, 0
    for it in range(iterations):
        limit = np.interp(it, [0, iterations - 1], max_distance)
        R, t = pose[:3, :3], pose[:3, 3]
        local = (scene - t) @ R  # scene in the object frame
        dist, idx = model.tree.query(local, distance_upper_bound=limit)
        ok = np.isfinite(dist)
        if ok.sum() < 10:
            return FitResult(pose, int(ok.sum()), np.inf, 0)
        p, q, n = local[ok], model.points[idx[ok]], model.normals[idx[ok]]
        view = (camera_origin - t) @ R  # camera position in the object frame
        front = np.einsum("ij,ij->i", n, view - q) > 0
        p, q, n = p[front], q[front], n[front]
        if len(p) < 10:
            return FitResult(pose, len(p), np.inf, 0)
        r = np.einsum("ij,ij->i", p - q, n)
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        # Small object-frame motion delta applied to the model: q' = q + w x q + v.
        J = np.column_stack([n, np.cross(q, n)])
        H = (J * w[:, None]).T @ J + 1e-6 * np.eye(6)
        g = (J * w[:, None]).T @ r
        xi = np.linalg.solve(H, g)
        pose = pose @ _exp(np.r_[xi[:3], xi[3:]])
        rms, count = float(np.sqrt(np.mean(r ** 2))), len(p)
        if np.linalg.norm(xi) < 1e-5:
            break
    # Expected visible samples: front-facing model points inside the fit.
    view = (camera_origin - pose[:3, 3]) @ pose[:3, :3]
    expected = int((np.einsum("ij,ij->i", model.normals, view - model.points) > 0).sum())
    return FitResult(pose, count, rms, expected)


def depth_support(model, pose_cam, depth, K, tolerance=0.01):
    """(fraction of unoccluded front-facing model samples that lie on the depth, their count).

    A sample counts as visible when it projects inside the image onto valid
    depth that is not clearly nearer than it (nearer depth means an occluder,
    such as the hand, which neither supports nor refutes the pose).
    """
    R, t = pose_cam[:3, :3], pose_cam[:3, 3]
    view = -t @ R
    front = np.einsum("ij,ij->i", model.normals, view - model.points) > 0
    p = model.points[front] @ R.T + t
    p = p[p[:, 2] > 0.05]
    if not len(p):
        return 0.0, 0
    u = np.rint(p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.rint(p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(p))
    z[inside] = depth[v[inside], u[inside]]
    seen = inside & (z > 0) & (z > p[:, 2] - tolerance)
    if not seen.any():
        return 0.0, 0
    on = np.abs(z[seen] - p[seen, 2]) < tolerance
    return float(on.mean()), int(seen.sum())


def support_plane_mask(world_points, object_radius, *, up=(0.0, 0.0, 1.0), tolerance=0.006,
                       max_tilt_deg=15.0, iterations=150, seed=0):
    """True for points on a level supporting surface (table) wider than the object.

    A thin object resting on a table fits the table's points about as well as
    its own, so the table is removed before fitting. Only a plane within
    ``max_tilt_deg`` of level whose inliers extend well beyond the object counts,
    so a flat face of the object itself is never removed.
    """
    p = np.asarray(world_points, float)
    out = np.zeros(len(p), bool)
    if len(p) < 200:
        return out
    rng = np.random.default_rng(seed)
    up = np.asarray(up, float)
    best, best_count = None, 0
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n @ up) < np.cos(np.deg2rad(max_tilt_deg)):
            continue
        inliers = np.abs((p - a) @ n) < tolerance
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = (a, n * np.sign(n @ up)), count
    if best is None or best_count < 0.2 * len(p):
        return out
    a, n = best
    height = (p - a) @ n
    on = np.abs(height) < tolerance
    extent = np.ptp(p[on], axis=0)
    if np.linalg.norm(extent) < 2 * object_radius + 0.03:
        return out
    return height < tolerance  # on the plane or below it


GRASP_POINTS = {"thumb": (3, 4), "index": (7, 8), "middle": (11, 12), "ring": (15, 16),
                "pinky": (19, 20), "palm": (0, 5, 9, 13, 17)}


def grasp_contacts(model, pose_world, keypoints, distance=0.02):
    """How many hand regions (fingers by their last two joints, palm) touch the object.

    A region touches when one of its keypoints is within ``distance`` of the
    object surface (keypoints are bone centres, about a finger radius inside
    the skin).
    """
    local = (np.asarray(keypoints, float) - pose_world[:3, 3]) @ pose_world[:3, :3]
    d, _ = model.tree.query(local)
    return sum(bool(np.min(d[list(ids)]) < distance) for ids in GRASP_POINTS.values())


def hand_distance(points, keypoints, radius=0.012):
    """Distance from each point to the nearest finger bone or palm segment, minus ``radius``."""
    from .hands import FINGER_BONES
    segments = FINGER_BONES + [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (5, 17)]
    best = np.full(len(points), np.inf)
    for a, b in segments:
        p0, p1 = keypoints[a], keypoints[b]
        d = p1 - p0
        t = np.clip((points - p0) @ d / max(d @ d, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (p0 + t[:, None] * d), axis=1))
    return best - radius


@dataclass
class TrackResult:
    poses: np.ndarray      # (N,4,4) world <- object
    valid: np.ndarray      # (N,) fit accepted
    rms: np.ndarray
    inlier_fraction: np.ndarray
    carried_by: np.ndarray  # (N,) -1 none, 0 left, 1 right


def track(model, frames, T_world_camera, K, hands_world, wrists_world, init_pose_world, *,
          stride=2, crop_margin=0.04, hand_clearance=0.015,
          min_support=0.5, min_visible=30, max_rms=0.006, progress=None):
    """Track through ``frames`` (iterable of (index, depth)), returning TrackResult.

    ``T_world_camera`` (N,4,4) rectified left camera poses, ``hands_world``
    {side: (N,21,3)}, ``wrists_world`` {side: (N,4,4)} (NaN when invalid).
    """
    n = len(T_world_camera)
    poses = np.full((n, 4, 4), np.nan)
    valid = np.zeros(n, bool)
    rms = np.full(n, np.nan)
    frac = np.zeros(n)
    carried = np.full(n, -1, int)
    current = np.asarray(init_pose_world, float).copy()
    previous_index = None
    sides = sorted(hands_world)
    for index, depth in frames:
        Twc = T_world_camera[index]
        if not np.isfinite(Twc).all():
            continue
        # Hand-carried prediction: the hand touches the object at two or more places.
        # The object's previous pose stays a second hypothesis, so a release is seen.
        hypotheses = [current]
        carrier = -1
        if previous_index is not None:
            for k, side in enumerate(sides):
                kp_prev = hands_world[side][previous_index]
                W0, W1 = wrists_world[side][previous_index], wrists_world[side][index]
                if not (np.isfinite(kp_prev).all() and np.isfinite(W0).all() and np.isfinite(W1).all()):
                    continue
                if grasp_contacts(model, current, kp_prev) >= 2:
                    hypotheses = [W1 @ np.linalg.inv(W0) @ current, current]
                    carrier = k
                    break
        Tcw = np.linalg.inv(Twc)
        all_points, _ = backproject(depth, K, stride)
        best = None
        for h, guess in enumerate(hypotheses):
            scene = all_points
            if len(scene):
                pred_cam = Tcw @ guess
                center_cam = pred_cam[:3, :3] @ model.center + pred_cam[:3, 3]
                scene = scene[np.linalg.norm(scene - center_cam, axis=1) < model.radius + crop_margin]
            if len(scene):
                world_pts = scene @ Twc[:3, :3].T + Twc[:3, 3]
                keep = ~support_plane_mask(world_pts, model.radius)
                for side in sides:
                    kp = hands_world[side][index]
                    if np.isfinite(kp).all():
                        keep &= hand_distance(world_pts, kp) > hand_clearance
                scene = scene[keep]
            fit = icp(model, scene, Tcw @ guess)
            f, seen = depth_support(model, fit.pose, depth, K)
            accepted = bool(np.isfinite(fit.rms) and fit.rms <= max_rms and f >= min_support
                            and seen >= min_visible)
            score = (accepted, f - (fit.rms if np.isfinite(fit.rms) else 1.0) * 50)
            if best is None or score > best[0]:
                best = (score, h, fit, f, accepted)
        _, h, fit, f, accepted = best
        if accepted:
            current = Twc @ fit.pose
            carried[index] = carrier if h == 0 else -1
        else:
            current = hypotheses[0]  # nothing fits (occluded): carry on with the hand
            carried[index] = carrier
        poses[index] = current
        valid[index] = accepted
        rms[index] = fit.rms
        frac[index] = f
        previous_index = index
        if progress is not None:
            progress(index, accepted, fit)
    return TrackResult(poses, valid, rms, frac, carried)


def initial_hypotheses(center_world, count=24, seed=0):
    """Poses at ``center_world`` with orientations spread over SO(3)."""
    rotations = Rotation.create_group("O") if count == 24 else Rotation.random(count, random_state=seed)
    out = []
    for r in rotations:
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = center_world
        out.append(T)
    return out


def initialize(model, depth, T_world_camera, K, center_world, hands=(), *, stride=2):
    """Best ICP fit over orientation hypotheses around a world point (object origin there)."""
    Tcw = np.linalg.inv(T_world_camera)
    scene, _ = backproject(depth, K, stride)
    c_cam = Tcw[:3, :3] @ center_world + Tcw[:3, 3]
    scene = scene[np.linalg.norm(scene - c_cam, axis=1) < model.radius + 0.05]
    world = scene @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    off_table = ~support_plane_mask(world, model.radius)
    scene, world = scene[off_table], world[off_table]
    for kp in hands:
        if np.isfinite(kp).all():
            scene = scene[hand_distance(world, kp) > 0.015]
            world = scene @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    best = None
    for T in initial_hypotheses(center_world):
        T = T.copy()
        T[:3, 3] = center_world - T[:3, :3] @ model.center
        fit = icp(model, scene, Tcw @ T, iterations=30, max_distance=(0.03, 0.006))
        support, seen = depth_support(model, fit.pose, depth, K)
        score = support - fit.rms * 50
        if np.isfinite(fit.rms) and seen >= 30 and (best is None or score > best[0]):
            best = (score, T_world_camera @ fit.pose, fit)
    if best is None:
        raise ValueError("No depth points near the given object position")
    return best[1], best[2]
