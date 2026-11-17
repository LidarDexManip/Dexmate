"""Ground the tracked object to the hands with retargeting's held-segment refinement.

For every interval where a hand holds the object, retargeting's
``baselines.held_refinement.Problem`` jointly solves small rigid corrections of
the object (``object-only``), or of the object and the hand root
(``object-hand-root``), over the whole interval: bidirectional hand/object
penetration, contact-patch proximity on the fingers and palm that touch it,
correction size and smoothness, a stable-grasp object-to-palm prior when the
grasp is steady, and drift of the projection in the headset camera from the
tracked estimate. Corrections are bounded (object 30 mm / 20 deg, hand root
10 mm / 10 deg) and taper to zero over padding frames, so the motion outside
the interval and the timing are unchanged.

Differences from retargeting's own experiment, which assumes a MANO skin and a
static camera: the hand surface is ``hands.HandSurface`` (capsules from the
glove keypoints), the camera moves with the head (one camera pose per frame),
and held intervals and contact regions are detected from the hand-object
distances instead of annotated (an annotation file can still override them).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def retargeting_dir():
    return Path(os.environ.get("RETARGETING_DIR", "~/Desktop/retargeting")).expanduser().resolve()


def held_refinement():
    root = retargeting_dir()
    if not (root / "baselines" / "held_refinement.py").exists():
        raise RuntimeError(f"retargeting checkout not found at {root} (set RETARGETING_DIR)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from baselines import held_refinement as module
    return module


def bone_regions(surface):
    """Region index per hand-surface vertex (0..4 fingers thumb..pinky, 5 palm)."""
    bone = surface.bone_of_vertex
    return np.where(bone < 0, 5, bone // 3)


def contact_distances(surface, vertices, object_poses, mesh_query):
    """(N, 6) minimum signed distance of each region's surface to the object (m)."""
    regions = bone_regions(surface)
    out = np.full((len(vertices), len(REGIONS)), np.inf)
    for i, (v, T) in enumerate(zip(vertices, object_poses)):
        if not (np.isfinite(v).all() and np.isfinite(T).all()):
            continue
        local = (v - T[:3, 3]) @ T[:3, :3]
        d, _ = mesh_query.query(local)
        for r in range(len(REGIONS)):
            out[i, r] = d[regions == r].min()
    return out


def _runs(mask):
    mask = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    return list(zip(edges[::2], edges[1::2]))  # [start, end)


def _majority(mask, width=5):
    k = np.ones(width) / width
    return np.convolve(np.asarray(mask, float), k, mode="same") > 0.5


def detect_holds(times, distances, object_poses, hand_poses, *, contact=0.012, touch=0.004,
                 min_regions=2, min_duration=0.3, min_motion=0.02, padding=10,
                 stable_translation=0.01, stable_rotation_deg=10.0):
    """Held intervals for one hand.

    ``distances`` (N,6) per-region gaps, ``hand_poses`` (N,4,4) palm frames.
    A frame is held when at least ``min_regions`` regions are within
    ``contact``; runs shorter than ``min_duration`` or in which the object moves
    less than ``min_motion`` are dropped. The contact patches the refinement
    pulls onto the surface are only the regions whose median gap over the hold
    is within ``touch`` (a region merely near the object would be dragged into
    it). Returns dicts with ``start``/``end`` (inclusive frame indices),
    ``regions`` and ``stable_grasp``.
    """
    touching = distances < contact
    held = _majority(touching.sum(axis=1) >= min_regions)
    present = np.isfinite(distances).all(axis=1)
    out = []
    n = len(times)
    for a, b in _runs(held & present):
        if times[b - 1] - times[a] < min_duration:
            continue
        # Padding uses only frames that have both the hand and the object.
        before = 0
        while before < padding and a - before - 1 >= 0 and present[a - before - 1]:
            before += 1
        after = 0
        while after < padding and b + after < n and present[b + after]:
            after += 1
        centers = object_poses[a:b, :3, 3]
        if np.nanmax(np.linalg.norm(centers - centers[0], axis=1)) < min_motion:
            continue
        median_gap = np.median(distances[a:b], axis=0)
        regions = [REGIONS[r] for r in np.flatnonzero(median_gap <= touch)]
        if not regions:
            regions = [REGIONS[int(np.argmin(median_gap))]]
        rel = np.array([np.linalg.inv(P) @ O for P, O in zip(hand_poses[a:b], object_poses[a:b])])
        spread_t = float(np.nanmax(np.linalg.norm(rel[:, :3, 3] - np.median(rel[:, :3, 3], 0), axis=1)))
        mean = Rotation.from_matrix(rel[:, :3, :3]).mean()
        spread_r = float(np.rad2deg(np.max((mean.inv() * Rotation.from_matrix(rel[:, :3, :3])).magnitude())))
        out.append({"start": int(a), "end": int(b - 1), "regions": regions,
                    "stable_grasp": bool(spread_t < stable_translation and spread_r < stable_rotation_deg),
                    "relative_spread_mm": spread_t * 1000, "relative_spread_deg": spread_r,
                    "padding_before": before, "padding_after": after})
    return out


def _moving_problem_class(module):
    class MovingCameraProblem(module.Problem):
        """retargeting's Problem with one camera pose per frame (the headset moves).

        ``penetration_weight`` scales retargeting's penetration residual block,
        which is averaged over every surface vertex and so lets a few deeply
        penetrating vertices through when contact patches pull the other way.
        """

        def __init__(self, *args, cameras, penetration_weight=1.0, **kwargs):
            self.cameras = np.asarray(cameras, float)
            self.penetration_weight = float(penetration_weight)
            super().__init__(*args, **kwargs)
            self.reference_pixels = []
            for i in range(self.n):
                obj = module.transform_points(self.data["object_pose"][i], self.samples[:12])
                pixels, _ = module.project(np.concatenate([self.data["hand"][i], obj]),
                                           self.cameras[i], self.K[i])
                self.reference_pixels.append(pixels)

        def _penetration_rows(self, i):
            return len(self.vertices[i]) + len(self.samples)

        def frame_residual(self, x, i, details=False):
            self.T = self.cameras[i]
            out = super().frame_residual(x, i, details)
            if not details and self.penetration_weight != 1.0:
                out = out.copy()
                out[:self._penetration_rows(i)] *= self.penetration_weight
            return out

        def evaluate_frame(self, x, i):
            cache = self._frame_cache.get(i)
            if cache is not None and np.array_equal(cache[0], x):
                return cache[1], cache[2]
            self.T = self.cameras[i]
            r, J = super().evaluate_frame(x, i)
            if self.penetration_weight != 1.0:
                rows = self._penetration_rows(i)
                r, J = r.copy(), J.copy()
                r[:rows] *= self.penetration_weight
                J[:rows] *= self.penetration_weight
                self._frame_cache[i] = (x.copy(), r, J)
            return r, J

    return MovingCameraProblem


def camera_for_frame(T_camera_world, points, min_depth=0.1):
    """The frame's camera, moved back along its axis if any point is too close or behind."""
    T = np.asarray(T_camera_world, float).copy()
    z = points @ T[2, :3] + T[2, 3]
    if z.min() < min_depth:
        T[2, 3] += min_depth - z.min()
    return T


def refine_hold(hold, side, times, keypoints, vertices, faces, object_poses, mesh,
                T_world_camera, K_normalized, *, variant="object-only", max_nfev=50,
                penetration_weight=5.0):
    """Solve one held interval; returns (corrected object poses, corrected keypoints, report)."""
    module = held_refinement()
    before = int(hold.get("padding_before", hold.get("padding", 10)))
    after = int(hold.get("padding_after", hold.get("padding", 10)))
    a = max(0, hold["start"] - before)
    b = min(len(times), hold["end"] + after + 1)
    sel = slice(a, b)
    n = b - a
    lo, hi = hold["start"] - a, hold["end"] - a
    # Corrections ramp in and out over the padding; with no padding at a
    # sequence end they stay free there (nothing after them to match).
    taper = np.ones(n)
    if lo > 0:
        taper[:lo + 1] = np.linspace(0, 1, lo + 1)
    if hi < n - 1:
        taper[hi:] = np.linspace(1, 0, n - hi)
    data = {"hand": keypoints[sel].copy(), "object_pose": object_poses[sel].copy(),
            "timestamps": np.asarray(times[sel], float), "valid": np.ones(n, bool),
            "frame_ids": np.arange(a, b)}
    if not (np.isfinite(data["hand"]).all() and np.isfinite(vertices[sel]).all()
            and np.isfinite(data["object_pose"]).all()):
        raise ValueError(f"{side} hold {hold['start']}-{hold['end']}: hand or object missing in "
                         "the interval or its padding")
    held = np.zeros(n, bool)
    held[lo:hi + 1] = True
    samples = module.surface_samples(mesh, 12, 0)
    cameras = []
    for i in range(n):
        Tcw = np.linalg.inv(T_world_camera[a + i])
        pts = np.concatenate([data["hand"][i], module.transform_points(data["object_pose"][i], samples)])
        cameras.append(camera_for_frame(Tcw, pts))
    Problem = _moving_problem_class(module)
    problem = Problem(data, vertices[sel], faces, mesh, cameras[0], K_normalized[sel], held,
                      hold["regions"], hold.get("stable_grasp", False), taper, cameras=cameras,
                      penetration_weight=penetration_weight)
    dof = {"object-only": 6, "object-hand-root": 12}[variant]
    x, report = problem.solve(dof, max_nfev)
    poses, joints = [], []
    for i, row in enumerate(x):
        pose, _, j = problem.geometry(row, i)
        poses.append(pose)
        joints.append(j)
    before = [problem.frame_residual(np.zeros(dof), i, True) for i in range(n)]
    after = [problem.frame_residual(row, i, True) for i, row in enumerate(x)]

    def summary(rows):
        held_rows = [r for r, h in zip(rows, held) if h]
        return {"max_penetration_mm": max(r["max_penetration_mm"] for r in held_rows),
                "mean_contact_gap_mm": float(np.mean([v for r in held_rows for v in r["contact_gap_mm"].values()])),
                "mean_relative_position_from_median_mm": float(np.mean(
                    [r["relative_position_from_phase_median_mm"] for r in held_rows])),
                "mean_reprojection_drift_normalized": float(np.mean(
                    [r["reprojection_drift_normalized"] for r in held_rows]))}

    report.update(side=side, variant=variant, hold=hold, frames=[a, b - 1],
                  before=summary(before), after=summary(after),
                  max_object_translation_mm=max(r["object_translation_mm"] for r in after),
                  max_object_rotation_deg=max(r["object_rotation_deg"] for r in after),
                  max_hand_translation_mm=max(r["hand_translation_mm"] for r in after))
    return a, np.asarray(poses), np.asarray(joints), report


def ground(times, hands, surfaces, object_poses, mesh, T_world_camera, K_normalized, *,
           variant="object-only", max_nfev=50, holds=None, penetration_weight=5.0, log=print):
    """Ground the whole sequence; returns (object poses, {side: keypoints}, report).

    ``hands`` {side: (N,21,3)}, ``surfaces`` {side: (HandSurface, (N,V,3))}.
    ``holds`` overrides detection: a list of dicts with ``side``, ``start``,
    ``end``, ``regions``, optional ``stable_grasp`` and ``padding``.
    """
    module = held_refinement()
    query = module.MeshQuery(mesh.vertices, mesh.faces)
    poses = np.asarray(object_poses, float).copy()
    keypoints = {s: np.asarray(k, float).copy() for s, k in hands.items()}
    from ..demo.hands import palm_pose
    if holds is None:
        holds = []
        for side, (surface, vertices) in surfaces.items():
            distances = contact_distances(surface, vertices, poses, query)
            palms = palm_pose(hands[side])
            for hold in detect_holds(times, distances, poses, palms):
                holds.append({"side": side, **hold})
    holds = sorted(holds, key=lambda h: h["start"])
    reports = []
    for hold in holds:
        side = hold["side"]
        hold = {"stable_grasp": False, **hold}
        surface, vertices = surfaces[side]
        log(f"grounding {side} hold, frames {hold['start']}-{hold['end']}, regions "
            f"{hold['regions']}, stable grasp {hold['stable_grasp']}")
        try:
            a, new_poses, new_joints, report = refine_hold(
                hold, side, times, keypoints[side], vertices, surface.faces, poses, mesh,
                T_world_camera, K_normalized, variant=variant, max_nfev=max_nfev,
                penetration_weight=penetration_weight)
        except ValueError as exc:
            log(f"  skipped: {exc}")
            reports.append({"side": side, "hold": hold, "skipped": str(exc)})
            continue
        poses[a:a + len(new_poses)] = new_poses
        if variant == "object-hand-root":
            keypoints[side][a:a + len(new_joints)] = new_joints
        log(f"  penetration {report['before']['max_penetration_mm']:.1f} -> "
            f"{report['after']['max_penetration_mm']:.1f} mm, contact gap "
            f"{report['before']['mean_contact_gap_mm']:.1f} -> {report['after']['mean_contact_gap_mm']:.1f} mm, "
            f"object moved up to {report['max_object_translation_mm']:.1f} mm / "
            f"{report['max_object_rotation_deg']:.1f} deg")
        reports.append(report)
    return poses, keypoints, {"variant": variant, "holds": reports}


def load_holds(path):
    data = json.loads(Path(path).read_text())
    return data["holds"] if isinstance(data, dict) else data
