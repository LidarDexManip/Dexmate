"""MANO hands fitted to the recorded wrist trackers and glove skeletons.

Input is a synced live-stereo session (``process_human_demo.sh sync``): at each
camera frame, ``synced.npz`` holds each hand's 21 keypoints in the world
(MediaPipe order). They come from the wrist tracker (position), the palm IMU
(orientation, through the wrist calibration) and the glove skeleton (fingers).
This module fits the MANO hand model to them, per hand:

- one shape ``betas`` (10) for the whole take;
- per frame, the global orientation, translation and the 15 finger joint
  rotations (axis-angle, relative to MANO's mean hand).

A glove finger whose sensor reads low confidence through the take (the right
glove's ring, straight in every frame of 2026-11-12) is not fitted to: its
joints bend as the mean of its neighbours' (``lost_fingers``, ``--no-stand-in``).
Hiding the healthy left ring this way put its tip 14-16 mm (median) from the
glove's, against 46-52 mm with no stand-in.

The loss is the confidence-weighted keypoint error. Priors keep the fit
anatomical: finger bones may not twist about their own axis (keypoints do not
show twist), the middle and end joints of the fingers bend as hinges, poses
stay near the mean hand, and consecutive frames stay close. Everything runs at
once on the GPU (PyTorch).

MANO's model files are licensed; download ``mano_v1_2.zip`` ("Models & Code")
from https://mano.is.tue.mpg.de and pass ``--mano`` (the zip, or a directory
holding ``MANO_RIGHT.pkl``/``MANO_LEFT.pkl``). The pickles are read without the
unmaintained ``chumpy`` package.

The hands are then moved onto the stereo depth (``depth_align``, ``--depth-align``):
the trackers and the wrist calibration place them only to within about 5-15 cm
and 35 degrees, and the error changes through a take, so by default the
correction is tracked in 0.5 s windows from an alignment over the first 3 s.

Runs in Fast-FoundationStereo's environment (torch): ``scripts/fit_mano.sh SESSION``.
Writes ``processed/mano_<side>.npz``, ``processed/mano_report.json`` and
``processed/mano_overlay.mp4`` (the left camera with the MANO hands and the glove
keypoints drawn in).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile

import numpy as np

SIDES = ("left", "right")
# MANO joint order: wrist; index 1-3, middle 4-6, pinky 7-9, ring 10-12, thumb 13-15.
# Fingertip vertices (manopth convention): thumb, index, middle, ring, pinky.
TIP_VERTICES = (745, 317, 444, 556, 673)
# MediaPipe index -> MANO joint (0..15) or fingertip (16 + k).
MP_FROM_MANO = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
# Child of each MANO joint (for the bone direction it rotates), -1 = tip vertex k.
_CHILD = {1: 2, 2: 3, 3: 17, 4: 5, 5: 6, 6: 18, 7: 8, 8: 9, 9: 20, 10: 11, 11: 12, 12: 19,
          13: 14, 14: 15, 15: 16}
HINGE_JOINTS = (2, 3, 5, 6, 8, 9, 11, 12)  # PIP and DIP of the four fingers
PALM_MP = (0, 5, 9, 13, 17)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Per finger: its three rotations in hand_pose (15 x 3) and its MediaPipe keypoints past the base.
POSE_SLOTS = {"index": (0, 1, 2), "middle": (3, 4, 5), "pinky": (6, 7, 8), "ring": (9, 10, 11),
              "thumb": (12, 13, 14)}
FINGER_MP = {f: tuple(range(2 + 4 * i, 5 + 4 * i)) for i, f in enumerate(FINGERS)}
# A finger the glove lost bends with these neighbours (the thumb has none).
NEIGHBOURS = {"index": ("middle",), "middle": ("index", "ring"), "ring": ("middle", "pinky"),
              "pinky": ("ring",)}
LOST_CONFIDENCE = 0.6  # healthy glove fingers read 0.84-0.97 (median); a failed sensor 0.3


# ---------------------------------------------------------------- model files

class _Stub:
    """Stands in for chumpy objects while unpickling; keeps their state."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def _array(value):
    if isinstance(value, _Stub):
        state = value.__dict__
        if "a" in state and "idxs" in state:  # chumpy.reordering.Select: a.ravel()[idxs]
            out = _array(state["a"]).ravel()[np.asarray(state["idxs"]).ravel()]
            shape = state.get("preferred_shape")
            return out.reshape(shape) if shape is not None else out
        for key in ("x", "r", "_state"):
            if key in value.__dict__:
                return _array(value.__dict__[key])
        raise ValueError(f"cannot read a chumpy object with fields {sorted(value.__dict__)}")
    if hasattr(value, "toarray"):  # scipy sparse
        return np.asarray(value.toarray())
    return np.asarray(value)


def read_mano_pickle(data: bytes) -> dict:
    raw = _Unpickler(io.BytesIO(data), encoding="latin1").load()
    out = {k: _array(raw[k]).astype(np.float64) for k in
           ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "hands_mean", "hands_components")}
    out["kintree_table"] = _array(raw["kintree_table"]).astype(np.int64)
    out["f"] = _array(raw["f"]).astype(np.int64)
    return out


def find_mano(source, side):
    """MANO_<SIDE>.pkl as a dict, from a mano_v1_2.zip or a directory (searched)."""
    name = f"MANO_{side.upper()}.pkl"
    source = Path(source).expanduser()
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as z:
            hits = [n for n in z.namelist() if n.endswith("/" + name) or n == name]
            if not hits:
                raise FileNotFoundError(f"{name} not in {source}")
            return read_mano_pickle(z.read(hits[0]))
    if source.is_file():
        return read_mano_pickle(source.read_bytes())
    hits = sorted(source.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {source}")
    return read_mano_pickle(hits[0].read_bytes())


def default_mano_source():
    for candidate in (Path.home() / "Downloads" / "mano_v1_2.zip", Path.home() / "Desktop" / "mano_v1_2",
                      Path.home() / "Downloads" / "mano_v1_2", Path.home() / "mano_v1_2.zip"):
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------- the model

class ManoLayer:
    """MANO forward kinematics and skinning in torch (axis-angle, flat_hand_mean=False)."""

    def __init__(self, params, device="cuda", dtype=None):
        import torch

        self.torch = torch
        dtype = dtype or torch.float32
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
        self.v_template = t(params["v_template"])                      # (V,3)
        self.shapedirs = t(params["shapedirs"])[..., :10]              # (V,3,10)
        self.posedirs = t(params["posedirs"]).reshape(-1, 135)         # (V*3,135)
        self.J_regressor = t(params["J_regressor"])                    # (16,V)
        self.weights = t(params["weights"])                            # (V,16)
        self.hands_mean = t(params["hands_mean"])                      # (45,)
        self.parents = [int(p) for p in params["kintree_table"][0]]
        self.parents[0] = -1
        self.faces = np.asarray(params["f"], np.int64)
        self.device, self.dtype = device, dtype
        rest = self.joints21(*(torch.zeros((1, n), dtype=dtype, device=device) for n in (10, 3, 45, 3)))
        self.rest21 = rest[0].detach().cpu().numpy()

    @staticmethod
    def rodrigues(aa):
        torch = __import__("torch")
        angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis = aa / angle
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(aa.shape[:-1] + (3, 3))
        eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
        s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
        return eye + s * K + (1 - c) * (K @ K)

    def forward(self, betas, global_orient, hand_pose, transl):
        """(B,10), (B,3), (B,45), (B,3) -> vertices (B,V,3), joints (B,16,3)."""
        torch = self.torch
        B = hand_pose.shape[0]
        betas = betas.expand(B, -1) if betas.shape[0] == 1 else betas
        v_shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        pose = torch.cat([global_orient, hand_pose + self.hands_mean], -1).reshape(B, 16, 3)
        R = self.rodrigues(pose)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        feature = (R[:, 1:] - eye).reshape(B, 135)
        v_posed = v_shaped + (feature @ self.posedirs.T).reshape(B, -1, 3)
        rel = J.clone()
        rel[:, 1:] = J[:, 1:] - J[:, self.parents[1:]]
        chain = [None] * 16
        for i in range(16):
            local = torch.cat([torch.cat([R[:, i], rel[:, i, :, None]], -1),
                               torch.tensor([0, 0, 0, 1.0], dtype=R.dtype, device=R.device).expand(B, 1, 4)], 1)
            chain[i] = local if i == 0 else chain[self.parents[i]] @ local
        T = torch.stack(chain, 1)                                       # (B,16,4,4)
        joints = T[:, :, :3, 3]
        skin_t = T[:, :, :3, 3] - torch.einsum("bjik,bjk->bji", T[:, :, :3, :3], J)
        Rv = torch.einsum("vj,bjik->bvik", self.weights, T[:, :, :3, :3])
        tv = torch.einsum("vj,bji->bvi", self.weights, skin_t)
        verts = torch.einsum("bvik,bvk->bvi", Rv, v_posed) + tv
        return verts + transl[:, None], joints + transl[:, None]

    def joints21(self, betas, global_orient, hand_pose, transl, verts_joints=None):
        torch = self.torch
        verts, joints = verts_joints or self.forward(betas, global_orient, hand_pose, transl)
        all21 = torch.cat([joints, verts[:, list(TIP_VERTICES)]], 1)   # 16 joints + 5 tips
        return all21[:, list(MP_FROM_MANO)]


# ------------------------------------------------------------------- fitting

def _kabsch(src, dst):
    """R, t with dst ~ R src + t."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - cd).T @ (src - cs))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    R = u @ D @ vt
    return R, cd - R @ cs


def lost_fingers(confidence, valid, threshold=LOST_CONFIDENCE):
    """Fingers whose glove keypoints are not measured in this take, judged by their confidence.

    A glove finger whose fingertip sensor fails keeps a made-up pose (the right
    glove's ring stayed straight through every take of 2026-11-12, at confidence
    about 0.3, while healthy fingers read 0.84-0.97). The thumb is never judged
    lost: no neighbour can stand in for it.
    """
    if confidence is None:
        return ()
    c = np.asarray(confidence, float)[np.asarray(valid, bool)]
    if not len(c):
        return ()
    return tuple(f for f in NEIGHBOURS if np.nanmedian(c[:, list(FINGER_MP[f])]) < threshold)


def fit_hand(layer, keypoints, valid, confidence=None, *, iters=800, lr=0.02, w_pose=1e-3, w_twist=1.0,
             w_hinge=0.3, w_smooth=0.05, w_shape=1e-3, stand_in=(), w_stand_in=1.0, log=None):
    """Fit MANO to (N,21,3) world keypoints (metres). Returns numpy params and per-frame error.

    ``stand_in`` names fingers whose glove keypoints are not to be trusted (see
    ``lost_fingers``): their keypoints past the base get no weight, and their
    joint rotations follow the mean of their neighbours' (``NEIGHBOURS``,
    relative to MANO's mean hand, so each keeps its own rest splay), with weight
    ``w_stand_in``. Their ``error_mm`` is measured against the glove anyway.
    """
    import torch
    from scipy.spatial.transform import Rotation

    dev, dt = layer.device, layer.dtype
    N = len(keypoints)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        raise ValueError("fewer than 2 frames with a valid hand")
    K = np.nan_to_num(np.asarray(keypoints, float)[idx])
    conf = np.ones((len(idx), 21)) if confidence is None else np.clip(np.nan_to_num(np.asarray(confidence)[idx]), 0.1, 1.0)
    # Initial wrist pose: align MANO's mean-hand palm to the observed palm, per frame.
    rest = layer.rest21
    go0, tr0 = np.zeros((len(idx), 3)), np.zeros((len(idx), 3))
    root = rest[0]
    for n, k in enumerate(K):
        R, t = _kabsch(rest[list(PALM_MP)] - root, k[list(PALM_MP)])
        go0[n] = Rotation.from_matrix(R).as_rotvec()
        tr0[n] = t + R @ root - root  # MANO rotates about its root joint
    target = torch.as_tensor(K, dtype=dt, device=dev)
    for f in stand_in:
        conf[:, list(FINGER_MP[f])] = 0.0
    weight = torch.as_tensor(conf, dtype=dt, device=dev)
    ties = [(list(POSE_SLOTS[f]), [list(POSE_SLOTS[n]) for n in NEIGHBOURS[f] if n not in stand_in]
             or [list(POSE_SLOTS[n]) for n in NEIGHBOURS[f]]) for f in stand_in]
    betas = torch.zeros((1, 10), dtype=dt, device=dev, requires_grad=True)
    go = torch.as_tensor(go0, dtype=dt, device=dev).requires_grad_(True)
    tr = torch.as_tensor(tr0, dtype=dt, device=dev).requires_grad_(True)
    pose = torch.zeros((len(idx), 45), dtype=dt, device=dev, requires_grad=True)
    # Bone directions and the palm normal of the mean hand, in MANO's (global-aligned) joint frames.
    r16 = np.concatenate([rest[[MP_FROM_MANO.index(j) for j in range(16)]],
                          rest[[MP_FROM_MANO.index(16 + k) for k in range(5)]]])
    bone = np.zeros((15, 3))
    for j in range(1, 16):
        child = _CHILD[j]
        d = r16[child] - r16[j]
        bone[j - 1] = d / np.linalg.norm(d)
    normal = np.cross(rest[5] - rest[0], rest[17] - rest[0])
    normal /= np.linalg.norm(normal)
    bone_t = torch.as_tensor(bone, dtype=dt, device=dev)
    normal_t = torch.as_tensor(normal, dtype=dt, device=dev)
    hinge = torch.as_tensor([j - 1 for j in HINGE_JOINTS], device=dev)
    # Consecutive valid frames (no smoothness across gaps).
    pairs = np.flatnonzero(np.diff(idx) == 1)
    a_idx = torch.as_tensor(pairs, device=dev)

    def loss_fn(stage):
        j21 = layer.joints21(betas, go, pose, tr)
        err = ((j21 - target) * 100.0).norm(dim=-1)                   # cm
        data = (weight * err ** 2).mean()
        if stage == 0:
            return data, err
        full = (pose + layer.hands_mean).reshape(-1, 15, 3)
        twist = (full * bone_t).sum(-1)                              # along the bone
        bend_sideways = (full[:, hinge] * normal_t).sum(-1)           # hinge joints: no sideways bend
        reg = (w_pose * (pose ** 2).sum(-1).mean() + w_twist * (twist ** 2).sum(-1).mean()
               + w_hinge * (bend_sideways ** 2).sum(-1).mean() + w_shape * (betas ** 2).sum())
        p3 = pose.reshape(-1, 15, 3)
        for own, donors in ties:
            reg = reg + w_stand_in * ((p3[:, own] - sum(p3[:, d] for d in donors) / len(donors)) ** 2
                                      ).sum(-1).mean()
        if len(pairs):
            reg = reg + w_smooth * (((pose[a_idx + 1] - pose[a_idx]) ** 2).sum(-1).mean()
                                    + ((go[a_idx + 1] - go[a_idx]) ** 2).sum(-1).mean()
                                    + (((tr[a_idx + 1] - tr[a_idx]) * 100.0) ** 2).sum(-1).mean())
        return data + reg, err

    # Stage 0: wrist pose and shape on all keypoints with the mean finger pose; stage 1: everything.
    for stage, params, steps in ((0, [betas, go, tr], iters // 4), (1, [betas, go, tr, pose], iters)):
        opt = torch.optim.Adam(params, lr=lr)
        for i in range(steps):
            opt.zero_grad()
            loss, err = loss_fn(stage)
            loss.backward()
            opt.step()
            if log and (i % 200 == 0 or i == steps - 1):
                log(f"    stage {stage} step {i:4d}: loss {loss.item():.3f}, keypoint error median "
                    f"{err.median().item() * 10:.1f} mm")
    with torch.no_grad():
        verts, joints = layer.forward(betas, go, pose, tr)
        j21 = layer.joints21(betas, go, pose, tr, (verts, joints))
        err_mm = ((j21 - target).norm(dim=-1) * 1000.0).cpu().numpy()
    out = {k: np.full((N,) + shape, np.nan, np.float32) for k, shape in
           (("global_orient", (3,)), ("hand_pose", (45,)), ("transl", (3,)), ("vertices", (778, 3)),
            ("joints21", (21, 3)), ("error_mm", (21,)))}
    out["global_orient"][idx] = go.detach().cpu().numpy()
    out["hand_pose"][idx] = pose.detach().cpu().numpy()
    out["transl"][idx] = tr.detach().cpu().numpy()
    out["vertices"][idx] = verts.cpu().numpy()
    out["joints21"][idx] = j21.cpu().numpy()
    out["error_mm"][idx] = err_mm
    out["betas"] = betas.detach().cpu().numpy()[0]
    out["valid"] = np.asarray(valid, bool)
    return out


# -------------------------------------------------------------------- video

OVERLAY_COLORS = {"left": (255, 150, 40), "right": (40, 150, 255)}  # BGR
_BONES21 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
            (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_mesh(img, vertices, faces, T_camera_world, K, color, alpha=0.6):
    """Shade a mesh (world coordinates) into a BGR image in place, far triangles first."""
    import cv2

    P = vertices @ T_camera_world[:3, :3].T + T_camera_world[:3, 3]
    if not np.isfinite(P).all() or (P[:, 2] <= 0.05).any():
        return False
    uv = (P @ K.T)[:, :2] / P[:, 2:3]
    tri = P[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    view = -tri.mean(1)
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    shade = np.clip(np.abs((normal * view).sum(1)), 0.25, 1.0)
    pix = np.round(uv[faces]).astype(np.int32)
    layer = img.copy()
    for f in np.argsort(-tri[:, :, 2].mean(1)):
        cv2.fillConvexPoly(layer, pix[f], tuple(int(c * shade[f]) for c in color), cv2.LINE_AA)
    img[:] = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    return True


def render_overlay(session, out=None, keypoints=True, log=print):
    """``processed/mano_overlay.mp4``: the left image with the MANO hands (and glove keypoints)."""
    import cv2

    session = Path(session).expanduser().resolve()
    z = np.load(session / "processed" / "synced.npz")
    from .depth_align import corrected_keypoints

    mano = {s: dict(np.load(session / "processed" / f"mano_{s}.npz")) for s in SIDES
            if (session / "processed" / f"mano_{s}.npz").exists()}
    glove = {s: corrected_keypoints(z[f"keypoints_{s}"], m) for s, m in mano.items() if f"keypoints_{s}" in z}
    K, T = z["K"], z["T_world_camera"]
    fps = float(1.0 / np.median(np.diff(z["t_capture"])))
    out = Path(out) if out else session / "processed" / "mano_overlay.mp4"
    writer = None
    for i, f in enumerate(z["frame"]):
        img = cv2.imread(str(session / "stereo" / "left" / f"{int(f):06d}.jpg"))
        if img is None:
            continue
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        if np.isfinite(T[i]).all():
            Tcw = np.linalg.inv(T[i])
            for side, m in mano.items():
                if m["valid"][i]:
                    draw_mesh(img, m["vertices"][i], m["faces"], Tcw, K, OVERLAY_COLORS[side])
                kp = glove[side][i] if side in glove else None
                if keypoints and kp is not None and np.isfinite(kp).all():
                    P = kp @ Tcw[:3, :3].T + Tcw[:3, 3]
                    if (P[:, 2] > 0.05).all():
                        uv = np.round((P @ K.T)[:, :2] / P[:, 2:3]).astype(int)
                        for a, b in _BONES21:
                            cv2.line(img, tuple(uv[a]), tuple(uv[b]), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {int(f)}  t {z['t_capture'][i] - z['t_capture'][0]:.2f} s", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)
    if writer is not None:
        writer.release()
        _to_h264(out)
        log(f"  video: {out}")
    return out


def _to_h264(path):
    """Re-encode OpenCV's mp4v as H.264 so browsers and most players open it (needs ffmpeg)."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_suffix(".h264.mp4")
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-movflags", "+faststart", str(tmp)], capture_output=True)
    if done.returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------- CLI

def session_fb(session, synced, baseline_m=None, log=print):
    """f*B (px*m) for a session: from a baseline (argument, else sync's), else from the wrist trackers."""
    from . import depth_align

    K = np.asarray(synced["K"], float)
    if baseline_m is None and np.isfinite(float(synced["baseline_m"])):
        baseline_m = float(synced["baseline_m"])
    if baseline_m:
        return float(K[0, 0] * baseline_m), {"source": "baseline", "baseline_m": float(baseline_m)}
    est = depth_align.tracker_fb(session, synced)
    for side, r in est["trackers"].items():
        log(f"  f*B from the {side} tracker: {r['median']:.1f} px*m over {r['samples']} frames, "
            f"spread {r['spread']:.0%}" + ("" if r["used"] else " (not used: its projection misses its puck)"))
    if est["fb"] is None:
        return None, {"source": "none", **est}
    log(f"  f*B {est['fb']:.1f} px*m (baseline {est['fb'] / K[0, 0] * 1000:.0f} mm at fx {K[0, 0]:.0f} px)")
    return est["fb"], {"source": "wrist trackers", **est}


def depth_correct(session, synced, fit, fb, side, mode, *, inflate_m=0.0, rotation_frame="world",
                  shift_frame="world", device="cuda", log=print):
    """Move a MANO fit onto the stereo depth (``depth_align``); returns the report entry.

    The alignment judges a copy of the hand inflated by ``inflate_m`` (the glove's
    bulk); the correction found is applied to the fit itself.
    """
    from . import depth_align

    z = dict(synced)
    probe = depth_align.inflated(fit, inflate_m)
    frames = dict(rotation_frame=rotation_frame, shift_frame=shift_frame)
    initial = depth_align.align_hand(session, z, probe, fb, depth_align.initial_frames(z, fit), side=side,
                                     device=device, log=log, **frames)
    if initial is None:
        log(f"  {side}: never in view with depth; depth alignment skipped")
        return {"mode": "skipped", "reason": "never in view with depth"}
    entry = {"mode": mode, "inflate_m": inflate_m, **frames,
             "initial": {"turn_deg": initial["angle_deg"], "shift_m": initial["d"].tolist(),
                                       "misfit_before": initial["misfit_before"],
                                       "misfit_after": initial["misfit_after"],
                                       "start_frame": initial["frames"][0]}}
    if initial["refused"]:
        return {**entry, "mode": "refused", "reason": "; ".join(initial["refused"])}
    if mode == "initial":
        one = {"windows": [{"t": 0.0, "R": initial["E"], "t_shift": initial["d"]}], "side": side, **frames}
        R, t = depth_align.tracked_per_frame(fit, z, one)
    else:
        track = depth_align.track_hand(session, z, probe, fb, side=side, initial=initial, device=device, log=log,
                                       **frames)
        R, t = depth_align.tracked_per_frame(fit, z, track)
        w = track["windows"]
        entry["tracked"] = {"windows": len(w), "window_s": track["window_s"], "step_s": track["step_s"],
                            "misfit_before_mean": float(np.mean([r["before"] for r in w])),
                            "misfit_after_mean": float(np.mean([r["after"] for r in w])),
                            "shift_cm_range": [float(min(np.linalg.norm(r["t_shift"]) for r in w) * 100),
                                               float(max(np.linalg.norm(r["t_shift"]) for r in w) * 100)]}
    depth_align.apply_correction(fit, R, t)
    return entry


def fit_session(session, mano_source, *, device="cuda", iters=800, video=True, stand_in=True,
                depth="track", baseline_m=None, inflate_m=0.0, rotation_frame="world", shift_frame="world",
                log=print):
    session = Path(session).expanduser().resolve()
    synced = dict(np.load(session / "processed" / "synced.npz"))
    report = {"mano": str(mano_source), "hands": {}}
    fb = None
    if depth != "off" and "has_disparity" in synced and synced["has_disparity"].any():
        fb, report["depth_fb"] = session_fb(session, synced, baseline_m, log=log)
        if fb is None:
            log("  no f*B (no baseline, no tracker seen in the depth): depth alignment skipped; "
                "pass --baseline-m")
        else:
            report["depth_fb"]["fb"] = fb
    for side in SIDES:
        key = f"keypoints_{side}"
        if key not in synced:
            log(f"{side}: no world-frame hand in synced.npz (wrist calibration missing?); skipped")
            continue
        kp, valid = synced[key], synced[f"hand_valid_{side}"]
        conf = synced[f"glove_confidence_{side}"] if f"glove_confidence_{side}" in synced else None
        log(f"{side}: fitting MANO to {int(valid.sum())} of {len(valid)} frames ...")
        layer = ManoLayer(find_mano(mano_source, side), device=device)
        t0 = time.time()
        lost = lost_fingers(conf, valid) if stand_in else ()
        for f in lost:
            c = np.nanmedian(np.asarray(conf)[valid][:, list(FINGER_MP[f])])
            log(f"  {side} {f}: glove confidence {c:.2f}, so the {f} follows the "
                f"{' and '.join(NEIGHBOURS[f])} instead of the glove")
        fit = fit_hand(layer, kp, valid, conf, iters=iters, stand_in=lost, log=log)
        fit["faces"] = layer.faces
        e = fit["error_mm"][valid]
        measured = [i for i in range(21) if not any(i in FINGER_MP[f] for f in lost)]
        tips = e[:, [i for i in (4, 8, 12, 16, 20) if i in measured]]
        depth_entry = depth_correct(session, synced, fit, fb, side, depth, inflate_m=inflate_m,
                                    rotation_frame=rotation_frame, shift_frame=shift_frame,
                                    device=device, log=log) if fb else None
        report["hands"][side] = {
            "frames": int(len(valid)), "fitted": int(valid.sum()), "seconds": round(time.time() - t0, 1),
            "betas": fit["betas"].tolist(),
            "stand_in": {f: list(NEIGHBOURS[f]) for f in lost},
            "depth_alignment": depth_entry,
            "keypoint_error_mm": {"median": float(np.median(e[:, measured])),
                                  "p95": float(np.percentile(e[:, measured], 95)),
                                  "fingertips_median": float(np.median(tips)),
                                  "note": "over the measured keypoints (stand-in fingers left out)"},
            "per_keypoint_median_mm": np.median(e, axis=0).round(1).tolist()}
        np.savez_compressed(session / "processed" / f"mano_{side}.npz", frame=synced["frame"], t_capture=synced["t_capture"], **fit)
        log(f"  {side}: keypoint error median {np.median(e[:, measured]):.1f} mm, "
            f"p95 {np.percentile(e[:, measured], 95):.1f} mm, "
            f"fingertips {np.median(tips):.1f} mm ({time.time() - t0:.0f} s)")
    report["frame"] = ("world: the PICO world in robot axes (x forward, y left, z up, metres), the same as "
                       "synced.npz; hand_pose is relative to MANO's mean hand (flat_hand_mean=False); with a "
                       "depth alignment, the hands are moved onto the stereo depth (depth_R about the wrist, then "
                       "depth_t, per frame), and error_mm is against the glove before that move")
    (session / "processed" / "mano_report.json").write_text(json.dumps(report, indent=2))
    if video and report["hands"]:
        render_overlay(session, log=log)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--mano", type=Path, help="mano_v1_2.zip or a directory with MANO_*.pkl "
                        "(default: ~/Downloads/mano_v1_2.zip if present)")
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-video", action="store_true", help="skip processed/mano_overlay.mp4")
    parser.add_argument("--depth-align", choices=("track", "initial", "off"), default="track",
                        help="move the hands onto the stereo depth: 'track' follows them in 0.5 s windows from "
                             "an alignment over the first 3 s, 'initial' applies that first alignment to the "
                             "whole take (default: track)")
    parser.add_argument("--baseline-m", type=float,
                        help="stereo baseline for depth (default: sync's, else estimated from the trackers)")
    parser.add_argument("--inflate-m", type=float, default=0.0,
                        help="judge the depth alignment on the hand pushed out this far along its surface "
                             "normals (the glove's bulk; 0.004 tried on 2026-11-16)")
    parser.add_argument("--rotation-frame", choices=("world", "hand"), default="world",
                        help="frame of the tracked rotation about the wrist (hand: turns with the hand)")
    parser.add_argument("--shift-frame", choices=("world", "tracker"), default="world",
                        help="frame of the tracked shift (tracker: a corrected wrist offset that moves "
                             "with the forearm)")
    parser.add_argument("--no-stand-in", action="store_true",
                        help="fit every glove finger, even one whose sensor reads low confidence all take")
    args = parser.parse_args(argv)
    source = args.mano or default_mano_source()
    if source is None:
        sys.exit("MANO model not found. Download mano_v1_2.zip ('Models & Code') from "
                 "https://mano.is.tue.mpg.de (free registration, license) to ~/Downloads, or pass --mano.")
    for session in args.sessions:
        print(f"=== {session}")
        fit_session(session, source, device=args.device, iters=args.iters, video=not args.no_video,
                    stand_in=not args.no_stand_in, depth=args.depth_align, baseline_m=args.baseline_m,
                    inflate_m=args.inflate_m, rotation_frame=args.rotation_frame, shift_frame=args.shift_frame)


if __name__ == "__main__":
    main()
