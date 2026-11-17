"""Export a processed demonstration as a retargeting input bundle.

Writes ``processed/retarget_input/`` in the format ``~/Desktop/retargeting``'s
``baselines.input.load_sequence`` reads (see its INSTALL.md, "Supply a
reconstruction bundle"): ``all_hand_meshes.npz`` with both hands' 21 joints in
the same order (wrist, thumb..pinky, four each), ``layout_camera_frame_optimized.json``
with one object pose per frame, ``mesh/object.obj``, ``source.mp4`` (the
rectified left headset view, one frame per pose) and ``gravity.json``.

The bundle's "camera frame" is the gravity-aligned PICO world (z up), shifted
up so every point has positive z as the loader requires; ``T_export_world`` in
``import_provenance.json`` records the shift. TopoRetarget registers each hand
to its robot palm with one rigid transform, so the frame choice does not
change the retargeted motion.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")


def _fill(values, valid):
    """Invalid rows take the nearest valid row (they stay marked invalid)."""
    out = np.array(values, float)
    good = np.flatnonzero(valid)
    if not len(good):
        return np.zeros_like(out)
    idx = good[np.clip(np.searchsorted(good, np.arange(len(out))), 0, len(good) - 1)]
    out[~valid] = out[idx[~valid]]
    return out


def export(session, side="both", *, use_grounded=True, fps=None):
    from .process import load_mesh
    tl = session.timeline()
    obj = json.loads((session.out / "object.json").read_text())
    track = dict(np.load(session.out / "object_track.npz"))
    grounded_path = session.out / "grounded.npz"
    grounded = dict(np.load(grounded_path)) if use_grounded and grounded_path.exists() else None
    poses = grounded["object_poses"] if grounded is not None else track["poses"]
    n = len(tl["frame_utc"])
    have = np.array([(session.frames / f"{i:06d}.jpg").exists() for i in range(n)])
    have &= np.isfinite(poses.reshape(n, -1)).all(axis=1)
    if not have.any():
        raise SystemExit("No frames with both a depth image and an object pose; run depth and track")
    a = int(np.flatnonzero(have)[0])
    b = int(np.flatnonzero(have)[-1]) + 1
    sel = np.arange(a, b)
    if not have[sel].all():
        missing = sel[~have[sel]]
        raise SystemExit(f"Frames {missing[:5].tolist()}... lack depth images or object poses; "
                         "the bundle needs a contiguous range")
    joints, valid = {}, {}
    for s in SIDES:
        key = f"keypoints_{s}"
        if key in tl:
            kp = (grounded or {}).get(key, tl[key])[sel]
            ok = tl[f"valid_{s}"][sel] & np.isfinite(kp).all(axis=(1, 2))
        else:
            kp, ok = np.zeros((len(sel), 21, 3)), np.zeros(len(sel), bool)
        joints[s], valid[s] = kp, ok
    obj_poses = poses[sel]
    points = np.concatenate([obj_poses[:, :3, 3]] + [joints[s][valid[s]].reshape(-1, 3) for s in SIDES])
    shift = 0.3 - float(np.nanmin(points[:, 2]))
    T_export_world = np.eye(4)
    T_export_world[2, 3] = shift
    out = session.out / "retarget_input"
    if out.exists():
        shutil.rmtree(out)
    (out / "mesh").mkdir(parents=True)
    hands = {}
    for s in SIDES:
        hands[f"{s}_joints"] = _fill(joints[s], valid[s]) + [0, 0, shift]
        hands[f"{s}_valid"] = valid[s]
    np.savez_compressed(out / "all_hand_meshes.npz", **hands)
    W, H = (float(v) for v in tl["image_size"])
    K = tl["K"]
    intr = {"fx_norm": K[0, 0] / W, "fy_norm": K[1, 1] / H, "cx_norm": K[0, 2] / W, "cy_norm": K[1, 2] / H}
    object_valid = track["valid"][sel] | (track["carried_by"][sel] >= 0)
    records = []
    for k, (P, ok) in enumerate(zip(obj_poses, object_valid)):
        q = Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]]
        records.append({"frame_idx": k, "tracking_valid": bool(ok), "intrinsics_normalized": intr,
                        "local_to_scene": {"quat_wxyz_camera_frame": q.tolist(),
                                           "translation_camera_frame": (P[:3, 3] + [0, 0, shift]).tolist()}})
    (out / "layout_camera_frame_optimized.json").write_text(json.dumps(
        {"frame": "camera_frame", "translation_scale_optimization": {"mesh_scale": 1.0},
         "objects": records}))
    load_mesh(obj["mesh"], obj["mesh_scale"]).export(out / "mesh" / "object.obj")
    (out / "gravity.json").write_text(json.dumps({"vec3d": [0.0, 0.0, 1.0]}))
    rate = fps or 1.0 / float(np.median(np.diff(tl["frame_utc"][sel])))
    listing = out / "frames.txt"
    listing.write_text("".join(f"file '{session.frames / f'{i:06d}.jpg'}'\nduration {1 / rate:.6f}\n"
                               for i in sel))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-r", f"{rate:.6f}", "-fps_mode", "cfr", "-frames:v", str(len(sel)),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out / "source.mp4")], check=True)
    listing.unlink()
    counts = {s: int((valid[s] & object_valid).sum()) for s in SIDES}
    (out / "import_provenance.json").write_text(json.dumps({
        "source": str(session.root), "frames": [a, b - 1], "fps": rate,
        "object_poses": "grounded" if grounded is not None else "depth-tracked",
        "T_export_world": T_export_world.tolist(),
        "valid_frames_with_object": counts,
        "frame": "gravity-aligned PICO world in robot axes (x forward, y left, z up), shifted up",
        "notes": ["Hands: wrist trackers + palm IMU + glove skeleton (wuji_teleop.demo.hands).",
                  "Object: depth ICP in the headset stereo depth, carried by the hand while occluded, "
                  "then held-segment refinement (wuji_teleop.demo.ground).",
                  "tracking_valid is true where the object was fitted to depth or carried by a hand."]},
        indent=2))
    print(f"Wrote {out} (frames {a}-{b - 1}, {rate:.1f} fps)")
    holding = set(SIDES)
    report_path = session.out / "ground_report.json"
    if report_path.exists():
        found = {h.get("side", h.get("hold", {}).get("side"))
                 for h in json.loads(report_path.read_text())["holds"] if "skipped" not in h}
        holding = found or holding
    for s in SIDES:
        if counts[s] and s in holding:
            print(f"  {s}: {counts[s]} frames with hand and object. Retarget with:\n"
                  f"    cd ~/Desktop/retargeting && MUJOCO_GL=egl ./retarget.sh --method toporetarget "
                  f"--hand {s} --input {out} --continuity-experiment limited "
                  f"--output outputs/{session.root.name}_{s}")
    return out
