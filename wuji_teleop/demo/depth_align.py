"""Align the fitted MANO hands to the stereo depth at the start of a take.

The hands are placed in the world by the wrist trackers and the palm IMUs
(through the wrist calibration), not by the camera, so a tracker's own position
error or a calibration error shows as a hand beside or turned against its
image. On 2026-11-12 the right tracker read about 5 cm left of where the
camera saw it, and the right hand was turned about 45 degrees.

This module finds, per hand, one rigid correction (a rotation about the wrist
and a shift, in the world) that best lays the MANO surface on the measured
depth over the first seconds of the take, and applies it to every frame:

- depth comes from the recorded disparity PNGs, ``Z = f*B / d``;
- each MANO vertex that faces the camera is projected into the left image and
  compared with the depth at its pixel; errors are truncated at 1.5 cm, so
  background, the other hand and stereo holes cost the same and do not pull;
- a coarse-to-fine grid search (rotation about the wrist, then shift) runs on
  the GPU, with a weak prior towards no correction.

``f*B`` (focal length in pixels times baseline in metres) is the stereo
baseline measured in pixels. Without a measured baseline it is estimated from
a wrist tracker seen in the depth: its tracked distance times the disparity at
its projection. A tracker whose projection misses its puck reads scattered
values and is not used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRUNCATE_M = 0.015
WINDOW_S = 3.0
# Weak prior: a 10 cm shift or a 30 degree turn each cost 2% of a fully wrong fit.
PRIOR_SHIFT_M = 0.10
PRIOR_ANGLE_DEG = 30.0
PRIOR_WEIGHT = 0.02
# A correction beyond these is refused (the hand stays as fitted): with the hand out
# of view, a flat hand fits the empty table 20 cm away (2026-11-12, takes 1-2).
MAX_SHIFT_M = 0.15
MAX_TURN_DEG = 45.0
MAX_MISFIT = 0.85
IN_VIEW = 0.5  # a frame counts for the initial window with this share of the hand in the image
TRACKER_SURFACE_M = 0.005  # the puck's top is nearer the camera than its tracked centre


def disparity_path(session_root, frame):
    return Path(session_root) / "stereo" / "disparity" / f"{int(frame):06d}.png"


def read_depth(session_root, frame, fb, disparity_scale=64.0):
    """Depth in metres (0 = no match) from a frame's disparity PNG, or None."""
    import cv2

    png = cv2.imread(str(disparity_path(session_root, frame)), cv2.IMREAD_UNCHANGED)
    if png is None:
        return None
    Z = np.zeros(png.shape, np.float32)
    ok = png > 0
    Z[ok] = fb * disparity_scale / png[ok]
    return Z


def _project(K, Pc):
    return K[0, 0] * Pc[..., 0] / Pc[..., 2] + K[0, 2], K[1, 1] * Pc[..., 1] / Pc[..., 2] + K[1, 2]


def tracker_fb(session_root, synced, *, step=5, half=4, min_samples=20, max_spread=0.08,
               disparity_scale=64.0):
    """Estimate f*B (px*m) from the wrist trackers seen in the disparity.

    Per sampled frame: the tracker's distance from the camera (minus
    ``TRACKER_SURFACE_M``) times the median disparity in a (2 half + 1)^2 patch
    at its projection. A tracker counts when it gives at least ``min_samples``
    frames and their interquartile range is within ``max_spread`` of the median.
    """
    import cv2

    z = synced
    K = np.asarray(z["K"], float)
    per = {}
    for side in ("left", "right"):
        key = f"tracker_{side}"
        if key not in z:
            continue
        values = []
        for i in range(0, len(z["frame"]), step):
            Twc, T = z["T_world_camera"][i], z[key][i]
            if not (np.isfinite(Twc).all() and np.isfinite(T).all()):
                continue
            Tcw = np.linalg.inv(Twc)
            p = Tcw[:3, :3] @ T[:3, 3] + Tcw[:3, 3]
            if p[2] < 0.05:
                continue
            u, v = (int(round(a)) for a in _project(K, p))
            png = cv2.imread(str(disparity_path(session_root, z["frame"][i])), cv2.IMREAD_UNCHANGED)
            if png is None or not (half <= u < png.shape[1] - half and half <= v < png.shape[0] - half):
                continue
            patch = png[v - half:v + half + 1, u - half:u + half + 1].astype(float) / disparity_scale
            patch = patch[patch > 0]
            if len(patch) < 0.75 * (2 * half + 1) ** 2:
                continue
            values.append((p[2] - TRACKER_SURFACE_M) * np.median(patch))
        values = np.asarray(values)
        if len(values):
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            per[side] = {"samples": int(len(values)), "median": float(med), "spread": float((q3 - q1) / med),
                         "used": bool(len(values) >= min_samples and (q3 - q1) / med <= max_spread)}
    used = [s for s, r in per.items() if r["used"]]
    fb = float(np.median([per[s]["median"] for s in used])) if used else None
    return {"fb": fb, "trackers": per, "used": used}


def _rotations(max_deg, step_deg, device):
    import torch
    from scipy.spatial.transform import Rotation

    a = np.deg2rad(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    grid = np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    return grid, torch.as_tensor(Rotation.from_rotvec(grid).as_matrix(), dtype=torch.float32, device=device)


def _shifts(max_m, step_m):
    a = np.arange(-max_m, max_m + 1e-9, step_m)
    return np.stack(np.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)


def _frame_bases(synced, mano, i, rotation_frame, shift_frame, side):
    """Per-frame Q (the rotation is Q E Q^T about the wrist) and S (the shift is S d)."""
    from scipy.spatial.transform import Rotation

    Q = np.eye(3) if rotation_frame == "world" else Rotation.from_rotvec(mano["global_orient"][i]).as_matrix()
    S = np.eye(3) if shift_frame == "world" else np.asarray(synced[f"tracker_{side}"][i][:3, :3], float)
    return Q, S


class _Window:
    """The frames one correction is judged on, on the GPU.

    A correction is a rotation E about the wrist and a shift d, expressed in a
    frame per ``rotation_frame`` ("world", or "hand": MANO's wrist frame, which
    turns with the hand) and ``shift_frame`` ("world", or "tracker").
    """

    def __init__(self, session_root, synced, mano, idx, fb, *, max_vertices=None, device="cuda",
                 rotation_frame="world", shift_frame="world", side=None):
        import torch

        self.torch, self.device = torch, device
        K = np.asarray(synced["K"], float)
        self.K = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        self.frames = []
        rng = np.random.default_rng(0)
        for i in idx:
            Twc, V = synced["T_world_camera"][i], mano["vertices"][i]
            if not (np.isfinite(Twc).all() and np.isfinite(V).all()):
                continue
            Z = read_depth(session_root, synced["frame"][i], fb, float(synced.get("disparity_scale", 64.0)))
            if Z is None:
                continue
            Tcw = np.linalg.inv(Twc)
            wrist = mano["joints21"][i][0]
            Q, S = _frame_bases(synced, mano, i, rotation_frame, shift_frame, side)
            if not (np.isfinite(Q).all() and np.isfinite(S).all()):
                continue
            t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)  # noqa: E731
            self.frames.append({"V": V.astype(np.float32), "wrist": wrist, "Z": t(Z), "Rcw": t(Tcw[:3, :3]),
                                "tcw": t(Tcw[:3, 3]), "faces": mano["faces"], "Q": Q, "S": S,
                                "Q_t": t(Q), "S_t": t(S)})
        self.max_vertices, self.rng = max_vertices, rng
        self.set_visibility(np.eye(3), np.zeros(3))

    def set_visibility(self, R, t):
        """Keep the vertices that face the camera with the correction (R, t) applied."""
        torch = self.torch
        for f in self.frames:
            Rw, tw = f["Q"] @ R @ f["Q"].T, f["S"] @ t
            V = (f["V"] - f["wrist"]) @ Rw.T + f["wrist"] + tw
            Rcw, tcw = f["Rcw"].cpu().numpy(), f["tcw"].cpu().numpy()
            Vc = V @ Rcw.T + tcw
            F = f["faces"]
            fn = np.cross(Vc[F[:, 1]] - Vc[F[:, 0]], Vc[F[:, 2]] - Vc[F[:, 0]])
            n = np.zeros_like(Vc)
            for k in range(3):
                np.add.at(n, F[:, k], fn)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            keep = np.flatnonzero((n * -Vc / np.linalg.norm(Vc, axis=1, keepdims=True)).sum(1) > 0.3)
            if self.max_vertices and len(keep) > self.max_vertices:
                keep = self.rng.choice(keep, self.max_vertices, replace=False)
            f["rel"] = torch.as_tensor(((f["V"][keep] - f["wrist"]) @ f["Q"]).astype(np.float32),
                                       device=self.device)  # Q^T rel
            f["wrist_t"] = torch.as_tensor(f["wrist"].astype(np.float32), device=self.device)

    def cost(self, Rs, ts, chunk=1 << 26):
        """Mean truncated depth misfit (0 = every vertex on the depth, 1 = none) per (R, t) pair.

        ``Rs`` (C,3,3) torch and ``ts`` (C,3) torch: every rotation with every shift.
        Returns a (len(Rs), len(ts)) array.
        """
        torch = self.torch
        fx, fy, cx, cy = self.K
        out = torch.zeros((len(Rs), len(ts)), device=self.device)
        tau2 = TRUNCATE_M ** 2
        for f in self.frames:
            H, W = f["Z"].shape
            n = len(f["rel"])
            if n == 0:
                continue
            rot = torch.einsum("rij,vj->rvi", Rs, f["rel"]) @ f["Q_t"].T + f["wrist_t"]  # (R,V,3) world
            rot_c = rot @ f["Rcw"].T + f["tcw"]                                      # camera
            sh_c = ts @ (f["Rcw"] @ f["S_t"]).T                                      # (T,3)
            step = max(1, chunk // (n * len(ts)))
            for r0 in range(0, len(Rs), step):
                P = rot_c[r0:r0 + step, None] + sh_c[None, :, None]                 # (r,T,V,3)
                z = P[..., 2].clamp_min(1e-3)
                u = torch.round(fx * P[..., 0] / z + cx).long()
                v = torch.round(fy * P[..., 1] / z + cy).long()
                inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                zo = f["Z"][v.clamp(0, H - 1), u.clamp(0, W - 1)]
                r2 = torch.where(inside & (zo > 0), torch.clamp((zo - z) ** 2, max=tau2),
                                 torch.full_like(z, tau2))
                out[r0:r0 + step] += r2.mean(-1) / tau2
        return (out / max(len(self.frames), 1)).cpu().numpy()


IDENTITY_PRIOR = (np.eye(3), np.zeros(3), PRIOR_WEIGHT, PRIOR_ANGLE_DEG, PRIOR_SHIFT_M)
# While tracking: a 10 degree turn or a 3 cm shift from the previous window costs 3% of a fully wrong fit.
TRACK_PRIOR = (0.03, 10.0, 0.03)
TRACK_WINDOW_S = 0.5  # 1 s was 2% worse on held-out frames (take2, 2026-11-16); 0.25 s overfits
TRACK_STEP_S = 0.25


def _prior(Rc, tc, anchors):
    """Prior cost of each (rotation, shift) pair: squared distance to each anchor (R, t, weight, deg, m)."""
    total = np.zeros((len(Rc), len(tc)))
    for R_a, t_a, weight, deg, metres in anchors:
        cos = (np.einsum("ij,rij->r", R_a, Rc) - 1) / 2
        angle = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))[:, None]
        dist = np.linalg.norm(tc - t_a, axis=1)[None, :]
        total += weight * ((angle / deg) ** 2 + (dist / metres) ** 2)
    return total


def _search(window, rotvecs, Rs, shifts, base_R, base_t, anchors=(IDENTITY_PRIOR,)):
    """Best (R, t) among base_R composed with each rotation and base_t plus each shift."""
    import torch

    Rc = Rs @ torch.as_tensor(base_R, dtype=torch.float32, device=Rs.device)
    tc = (base_t + shifts).astype(np.float32)
    total = window.cost(Rc, torch.as_tensor(tc, device=Rs.device)) + _prior(Rc.cpu().numpy(), tc, anchors)
    r, s = np.unravel_index(np.argmin(total), total.shape)
    return Rc[r].cpu().numpy().astype(float), base_t + shifts[s], float(total[r, s])


def in_view(synced, mano):
    """(N,) share of each frame's MANO vertices that project into the left image."""
    K = np.asarray(synced["K"], float)
    W, H = (int(v) for v in synced["image_size"])
    share = np.zeros(len(mano["valid"]))
    for i in np.flatnonzero(np.asarray(mano["valid"], bool)):
        Twc = synced["T_world_camera"][i]
        if not np.isfinite(Twc).all():
            continue
        Tcw = np.linalg.inv(Twc)
        P = mano["vertices"][i] @ Tcw[:3, :3].T + Tcw[:3, 3]
        front = P[:, 2] > 0.05
        u, v = _project(K, P[front])
        share[i] = ((u >= 0) & (u < W) & (v >= 0) & (v < H)).sum() / len(P)
    return share


def initial_frames(synced, mano, window_s=WINDOW_S, frame_step=3):
    """Every ``frame_step``-th frame in the first ``window_s`` s with this hand in view and depth."""
    valid = (np.asarray(mano["valid"], bool) & np.asarray(synced["has_disparity"], bool)
             & (in_view(synced, mano) >= IN_VIEW))
    t_rel = synced["t_capture"] - synced["t_capture"][0]
    first = np.flatnonzero(valid)
    if not len(first):
        return first
    return first[t_rel[first] < t_rel[first[0]] + window_s][::frame_step]


def align_hand(session_root, synced, mano, fb, idx, *, side, rotation_frame="world", shift_frame="world",
               device="cuda", log=None):
    """One rigid correction (E about the wrist, then d) that lays the hand on the depth in frames ``idx``.

    Returns a dict with ``E`` (3x3), ``d`` (m), the frames they are expressed in,
    and ``misfit_before``/``misfit_after`` (0-1) over ``idx``, or None when
    fewer than 3 of the frames are usable.
    """
    from scipy.spatial.transform import Rotation

    if len(idx) == 0:
        return None
    kw = dict(device=device, rotation_frame=rotation_frame, shift_frame=shift_frame, side=side)
    coarse = _Window(session_root, synced, mano, idx[::3], fb, max_vertices=150, **kw)
    if len(coarse.frames) < 3:
        return None
    # Coarse: turns up to 60 degrees about each axis, shifts up to 12 cm; then finer around the best.
    rv, Rs = _rotations(60, 15, device)
    E, d, _ = _search(coarse, rv, Rs, _shifts(0.12, 0.015), np.eye(3), np.zeros(3))
    fine = _Window(session_root, synced, mano, idx, fb, **kw)
    for max_deg, step_deg, max_m, step_m in ((10, 5, 0.015, 0.005), (4, 2, 0.006, 0.002), (1, 1, 0.002, 0.001)):
        fine.set_visibility(E, d)
        rv, Rs = _rotations(max_deg, step_deg, device)
        E, d, _ = _search(fine, rv, Rs, _shifts(max_m, step_m), E, d)
    before, after = _misfits(fine, E, d, device)
    angle = float(np.rad2deg(Rotation.from_matrix(E).magnitude()))
    if log:
        log(f"    depth alignment over {len(fine.frames)} frames: shift {np.round(d * 100, 1).tolist()} cm "
            f"({shift_frame} frame), turn {angle:.0f} deg ({rotation_frame} frame); "
            f"depth misfit {before:.2f} -> {after:.2f}")
    problems = [text for bad, text in (
        (np.linalg.norm(d) > MAX_SHIFT_M, f"shift {np.linalg.norm(d) * 100:.0f} cm > {MAX_SHIFT_M * 100:.0f}"),
        (angle > MAX_TURN_DEG, f"turn {angle:.0f} deg > {MAX_TURN_DEG:.0f}"),
        (after > MAX_MISFIT, f"misfit {after:.2f} > {MAX_MISFIT}")) if bad]
    if problems and log:
        log(f"    refused ({', '.join(problems)}): the hand may be out of view; it stays as fitted")
    return {"E": E, "d": d, "angle_deg": angle, "rotation_frame": rotation_frame, "shift_frame": shift_frame,
            "misfit_before": before, "misfit_after": after, "frames": [int(synced["frame"][i]) for i in idx],
            "fb": fb, "refused": problems}


def _misfits(window, E, d, device):
    window.set_visibility(np.eye(3), np.zeros(3))
    before = float(window.cost(*_as_torch(np.eye(3)[None], np.zeros((1, 3)), device))[0, 0])
    window.set_visibility(E, d)
    after = float(window.cost(*_as_torch(E[None], d[None], device))[0, 0])
    return before, after


def _as_torch(Rs, ts, device):
    import torch

    return (torch.as_tensor(np.asarray(Rs, np.float32), device=device),
            torch.as_tensor(np.asarray(ts, np.float32), device=device))


def misfit_over_take(session_root, synced, mano, fb, correction, *, side, window_s=WINDOW_S, frame_step=6,
                     device="cuda"):
    """Depth misfit before/after one correction in consecutive windows: does it hold over the take?"""
    valid = np.asarray(mano["valid"], bool) & np.asarray(synced["has_disparity"], bool)
    t_rel = synced["t_capture"] - synced["t_capture"][0]
    rows = []
    for w0 in np.arange(0.0, float(t_rel[-1]), window_s):
        idx = np.flatnonzero(valid & (t_rel >= w0) & (t_rel < w0 + window_s))[::frame_step]
        win = _Window(session_root, synced, mano, idx, fb, device=device, side=side,
                      rotation_frame=correction["rotation_frame"], shift_frame=correction["shift_frame"])
        if len(win.frames) < 3:
            continue
        b, a = _misfits(win, correction["E"], correction["d"], device)
        rows.append({"start_s": float(w0), "before": b, "after": a})
    return rows


def per_frame(fit, correction, synced, side):
    """(N,3,3) world rotations about the wrist and (N,3) world shifts of a correction, per frame."""
    N = len(fit["valid"])
    R, t = np.full((N, 3, 3), np.nan), np.full((N, 3), np.nan)
    for i in np.flatnonzero(np.asarray(fit["valid"], bool) & np.isfinite(fit["global_orient"]).all(1)):
        Q, S = _frame_bases(synced, fit, i, correction["rotation_frame"], correction["shift_frame"], side)
        R[i], t[i] = Q @ correction["E"] @ Q.T, S @ correction["d"]
    return R, t


def inflated(mano, metres):
    """A copy of a MANO fit whose vertices are pushed ``metres`` out along their normals.

    For judging the fit against the depth of a gloved hand: the glove's surface
    sits a few mm outside the hand. Only ``vertices`` change.
    """
    if not metres:
        return mano
    V = np.asarray(mano["vertices"], np.float64)
    F = np.asarray(mano["faces"], np.int64)
    fn = np.cross(V[:, F[:, 1]] - V[:, F[:, 0]], V[:, F[:, 2]] - V[:, F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, (slice(None), F[:, k]), fn)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    out = dict(mano)
    out["vertices"] = (V + metres * n).astype(np.float32)
    return out


def track_hand(session_root, synced, mano, fb, *, side, initial=None, window_s=TRACK_WINDOW_S,
               step_s=TRACK_STEP_S, frame_step=3, rotation_frame="world", shift_frame="world",
               device="cuda", log=None):
    """A correction per ``step_s`` (judged over ``window_s``), carried from the initial alignment.

    Each window searches near the previous window's correction, with a prior
    towards it (``TRACK_PRIOR``) and the weak one towards no correction. The
    correction is a rotation about the wrist in ``rotation_frame`` ("world" or
    "hand") and a shift in ``shift_frame`` ("world" or "tracker": a corrected
    wrist offset that moves with the forearm). Returns window centres (s, from
    the take's first frame), their (R, t) in those frames and misfits.
    """
    valid = np.asarray(mano["valid"], bool) & np.asarray(synced["has_disparity"], bool)
    t_rel = synced["t_capture"] - synced["t_capture"][0]
    if initial is None:
        initial = align_hand(session_root, synced, mano, fb, initial_frames(synced, mano), side=side,
                             rotation_frame=rotation_frame, shift_frame=shift_frame, device=device, log=log)
    if initial is None or initial["refused"]:
        return None
    R, t = initial["E"], initial["d"]
    rows = []
    usable = np.flatnonzero(valid)
    for w0 in np.arange(t_rel[usable[0]], t_rel[usable[-1]], step_s):
        idx = np.flatnonzero(valid & (t_rel >= w0) & (t_rel < w0 + window_s))[::frame_step]
        win = _Window(session_root, synced, mano, idx, fb, max_vertices=200, device=device, side=side,
                      rotation_frame=rotation_frame, shift_frame=shift_frame)
        if len(win.frames) < 3:
            continue
        anchors = ((R, t) + TRACK_PRIOR, IDENTITY_PRIOR)
        for max_deg, step_deg, max_m, step_m in ((20, 10, 0.05, 0.0125), (6, 3, 0.012, 0.004),
                                                 (2, 1, 0.004, 0.002)):
            win.set_visibility(R, t)
            rv, Rs = _rotations(max_deg, step_deg, device)
            R, t, _ = _search(win, rv, Rs, _shifts(max_m, step_m), R, t, anchors)
        before, after = _misfits(win, R, t, device)
        rows.append({"t": float(t_rel[idx].mean()), "R": R, "t_shift": t, "before": before, "after": after})
    if log and rows:
        b = np.mean([r["before"] for r in rows])
        a = np.mean([r["after"] for r in rows])
        shifts = np.array([np.linalg.norm(r["t_shift"]) for r in rows]) * 100
        log(f"    tracked over {len(rows)} windows: depth misfit {b:.2f} -> {a:.2f} (mean); "
            f"shift {shifts.min():.0f}-{shifts.max():.0f} cm")
    return {"initial": initial, "windows": rows, "window_s": window_s, "step_s": step_s, "fb": fb,
            "rotation_frame": rotation_frame, "shift_frame": shift_frame, "side": side}


def tracked_per_frame(fit, synced, track):
    """(N,3,3), (N,3): the tracked corrections at every frame (held at the ends), in the world frame.

    Corrections tracked in the hand or tracker frame are interpolated in that
    frame, then turned into each frame's world rotation (Q R Q^T about the
    wrist) and shift (S t).
    """
    from scipy.spatial.transform import Rotation, Slerp

    rows = track["windows"]
    t_rel = synced["t_capture"] - synced["t_capture"][0]
    times = np.array([r["t"] for r in rows])
    if len(rows) == 1:
        times = np.array([times[0], times[0] + 1e-3])
        rows = rows * 2
    q = np.clip(t_rel, times[0], times[-1])
    R = Slerp(times, Rotation.from_matrix(np.array([r["R"] for r in rows])))(q).as_matrix()
    t = np.stack([np.interp(q, times, [r["t_shift"][k] for r in rows]) for k in range(3)], 1)
    rotation_frame = track.get("rotation_frame", "world")
    shift_frame = track.get("shift_frame", "world")
    if rotation_frame == "world" and shift_frame == "world":
        return R, t
    for i in range(len(q)):
        Q, S = _frame_bases(synced, fit, i, rotation_frame, shift_frame, track.get("side"))
        if np.isfinite(Q).all():
            R[i] = Q @ R[i] @ Q.T
        t[i] = S @ t[i] if np.isfinite(S).all() else np.nan
    return R, t


def apply_correction(fit, R, t):
    """The MANO fit moved per frame by R_i (about the wrist), then t_i, in place.

    MANO turns about its root joint, so the parameters change exactly as
    global_orient <- R_i global_orient, transl <- transl + t_i. R and t are kept
    as ``depth_R``/``depth_t`` (for ``corrected_keypoints``).
    """
    from scipy.spatial.transform import Rotation

    R, t = np.asarray(R, float), np.asarray(t, float)
    ok = np.isfinite(t).all(1) & np.isfinite(fit["transl"]).all(1)
    wrist = fit["joints21"][:, 0].astype(float)
    for key in ("vertices", "joints21"):
        X = fit[key]
        X[ok] = (np.einsum("nij,nvj->nvi", R[ok], X[ok] - wrist[ok, None]) + wrist[ok, None]
                 + t[ok, None]).astype(X.dtype)
    fit["global_orient"][ok] = (Rotation.from_matrix(R[ok]) * Rotation.from_rotvec(fit["global_orient"][ok])
                                ).as_rotvec().astype(fit["global_orient"].dtype)
    fit["transl"][ok] = (fit["transl"][ok] + t[ok]).astype(fit["transl"].dtype)
    fit["depth_R"], fit["depth_t"] = R.astype(np.float32), t.astype(np.float32)
    return fit


def corrected_keypoints(keypoints, fit):
    """Glove keypoints (N,21,3) moved like the MANO hand by its stored depth correction."""
    if "depth_R" not in fit:
        return keypoints
    R, t = np.asarray(fit["depth_R"], float), np.asarray(fit["depth_t"], float)
    pivot = np.asarray(fit["joints21"], float)[:, 0] - t                    # the wrist before correction
    return np.einsum("nij,nvj->nvi", R, keypoints - pivot[:, None]) + pivot[:, None] + t[:, None]
