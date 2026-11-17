"""Offline processing of a recorded human demonstration (retargeting's .venv-vision).

Session layout::

    <session>/pc/          laptop log (record.py)
    <session>/headset/     CameraRecord_*.mp4 + trackingData_*.txt (``pull``)
    <session>/processed/   everything below

Stages (each reruns on its own; later stages read earlier outputs):

- ``pull``: copy the newest headset recording off the PICO with adb.
- ``inspect``: report what the headset files hold (camera conventions, rates).
- ``hands``: align clocks, calibrate the wrists, world-frame hands and head
  camera poses at every video frame -> ``timeline.npz``, ``calibration.json``.
- ``depth``: rectify the stereo pairs and compute depth -> ``frames/``.
- ``track``: track the object mesh in the depth from an initial pose
  (``object_init.json``, set in the viewer) -> ``object_track.npz``.
- ``ground``: held-segment refinement against the hands -> ``grounded.npz``.
- ``export``: retargeting input bundles per hand -> ``retarget/<side>/``.
- ``view``: the viser world viewer.

A session recorded with ``record.py --stereo`` (the live stereo stream with
Fast-FoundationStereo disparity, no headset recording) has ``<session>/stereo/``
instead of ``headset/``; its stage is ``sync`` (``stream_sync``): stream latency
from head motion, then every stream at each frame's capture time ->
``synced.npz``, ``sync_report.json``, ``sync_check.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from . import align, hands as demo_hands, headset
from .log import load as load_stream
from ..xr_tracker_source import assign_tracker_sides, operator_heading, xr_pose_to_robot

SIDES = ("left", "right")


class Session:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.pc = self.root / "pc"
        self.headset = self.root / "headset"
        self.out = self.root / "processed"
        self.frames = self.out / "frames"

    def meta(self):
        return json.loads((self.pc / "meta.json").read_text())

    def events(self):
        return {e["label"]: e["t"] for e in self.meta().get("events", [])}

    def timeline(self):
        return dict(np.load(self.out / "timeline.npz", allow_pickle=False))

    def calibration(self):
        return json.loads((self.out / "calibration.json").read_text())

    def write_json(self, name, data):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / name).write_text(json.dumps(data, indent=2, default=float))


# ----------------------------------------------------------------------- pull

def pull(session, serial=None):
    adb = ["adb"] + (["-s", serial] if serial else [])
    listing = subprocess.run(adb + ["shell", "ls", "-t", "/sdcard/Download/"], capture_output=True,
                             text=True, check=True).stdout.split()
    text = next((f for f in listing if f.startswith("trackingData_") and f.endswith(".txt")), None)
    video = next((f for f in listing if f.startswith("CameraRecord_") and f.endswith(".mp4")), None)
    if not text or not video:
        raise SystemExit("No trackingData_*.txt / CameraRecord_*.mp4 in the headset's /sdcard/Download")
    session.headset.mkdir(parents=True, exist_ok=True)
    for name in (text, video):
        target = session.headset / name
        if target.exists():
            print(f"{name} already pulled")
            continue
        subprocess.run(adb + ["pull", f"/sdcard/Download/{name}", str(target)], check=True)
    print(f"Pulled {text} and {video} into {session.headset}")


# ---------------------------------------------------------------------- hands

def _side_trackers(rec, offset, events, left_sn=None, right_sn=None):
    """{side: serial} for the wrist trackers, by position during the still pose."""
    t = rec.seconds + offset
    at = events.get("still_end", events.get("demo_start", t[len(t) // 2]))
    i = int(np.clip(np.searchsorted(t, at), 0, len(t) - 1))
    poses = {}
    for serial, raw in rec.trackers.items():
        good = np.flatnonzero(np.isfinite(raw[:, 0]))
        if len(good):
            j = good[np.argmin(np.abs(good - i))]
            poses[serial] = xr_pose_to_robot(raw[j])
    head_raw = rec.head[i] if np.isfinite(rec.head[i]).all() else None
    yaw = operator_heading(xr_pose_to_robot(head_raw) if head_raw is not None else None,
                           tuple(poses.values())[:2])
    if len(poses) > 2 and not (left_sn and right_sn):
        raise SystemExit(f"{len(poses)} trackers recorded ({sorted(poses)}); give --left-tracker "
                         "and --right-tracker")
    return assign_tracker_sides(poses, yaw=yaw, left_serial=left_sn, right_serial=right_sn)


def stage_hands(session, args):
    meta = session.meta()
    events = session.events()
    text, video = headset.find_recording(session.headset)
    rec = headset.read_tracking(text)
    pc_trackers = align.pc_tracker_series(session.pc, meta["tracker_serials"])
    offset, sync = align.tracker_offset(rec.seconds, rec.trackers, pc_trackers,
                                        guess=args.clock_guess)
    print(f"clock: host = headset + {offset:+.4f} s (tracker {sync['serial']}, speed correlation "
          f"{sync['speed_correlation']:.3f}, position rms {sync['position_rms_m'] * 1000:.1f} mm)")
    if sync["position_rms_m"] > 0.01:
        print("  warning: the two tracker streams differ by more than 1 cm after alignment")
    width, height = headset.video_size(video)
    pts = headset.video_pts(video)
    frame_headset, time_mode = headset.frame_times_utc(rec, pts, args.frame_time)
    frame_utc = frame_headset + offset
    camera = headset.stereo_camera(rec, width, height, extrinsic=args.extrinsic,
                                   intrinsics=args.intrinsics, camera_axes=args.camera_axes)
    from .stereo import StereoDepth
    rect = StereoDepth(camera, scale=args.depth_scale).rect
    # Head and camera poses at every video frame (headset clock).
    head = np.array([headset.to_robot_world(headset.xr_matrix(r)) if np.isfinite(r).all()
                     else np.full((4, 4), np.nan) for r in rec.head])
    T_world_head, head_valid = demo_hands.interp_poses(rec.seconds, head, frame_headset, 0.1)
    T_world_camera = np.full_like(T_world_head, np.nan)
    T_world_camera[head_valid] = T_world_head[head_valid] @ camera.T_head_left @ rect.T_left_rect
    sides = _side_trackers(rec, offset, events, args.left_tracker, args.right_tracker)
    print(f"trackers: left {sides['left']}, right {sides['right']}")
    out = {"frame_utc": frame_utc, "frame_headset": frame_headset, "T_world_head": T_world_head,
           "T_world_camera": T_world_camera, "K": rect.K, "image_size": np.array([rect.width, rect.height]),
           "baseline": np.array(rect.baseline)}
    calibration = {"clock_offset_s": offset, "sync": sync, "frame_time_mode": time_mode,
                   "camera": {"extrinsic": args.extrinsic, "intrinsics": args.intrinsics,
                              "camera_axes": args.camera_axes, "eye_size": [camera.width, camera.height],
                              "K_left": camera.K_left.tolist(), "baseline_m": rect.baseline,
                              "T_head_left": camera.T_head_left.tolist()},
                   "trackers": sides, "wrists": {}}
    for side in SIDES:
        glove_meta = meta["gloves"].get(side)
        if glove_meta is None:
            continue
        t_glove, kp, conf = demo_hands.glove_series(session.pc, side)
        t_imu, imu_rot = demo_hands.imu_series(session.pc, side)
        raw = rec.trackers[sides[side]]
        T_tracker = demo_hands.tracker_poses(raw)
        t_tracker = rec.seconds + offset
        imu_in_wrist = np.asarray(glove_meta["imu_in_wrist_rotation"])
        forward = np.nanmedian(kp[:, 9] - kp[:, 0], axis=0)
        forward /= np.linalg.norm(forward)
        if args.calibration_from:
            cal = demo_hands.WristCalibration.from_json(
                json.loads(Path(args.calibration_from).read_text())["wrists"][side])
        else:
            cal = demo_hands.calibrate_wrist(
                t_tracker, T_tracker, t_imu, imu_rot, imu_in_wrist, forward,
                (events["still_start"], events["still_end"]),
                (events["rotate_start"], events["rotate_end"]),
                mount_height=args.mount_height, mount_back=args.mount_back)
        r = cal.report
        print(f"{side} wrist calibration: forearms turned {r['forearm_rotation_spread_deg']:.0f} deg, "
              f"residual median {r['axyb_residual_deg_median']:.1f} deg, 90% {r['axyb_residual_deg_p90']:.1f} deg")
        hand = demo_hands.world_hand(frame_utc, t_tracker, T_tracker, t_imu, imu_rot, t_glove, kp,
                                     conf, cal, orientation=args.hand_orientation)
        print(f"  {side} hand valid in {hand.valid.mean():.0%} of {len(frame_utc)} frames; wrist "
              f"bend median {np.nanmedian(hand.wrist_angle_deg):.0f} deg, twist about the forearm "
              f"median {np.nanmedian(np.abs(hand.twist_deg)):.0f} deg (a growing twist means palm-IMU heading drift)")
        tracker_frames, _ = demo_hands.interp_poses(t_tracker, T_tracker, frame_utc, 0.1)
        out.update({f"keypoints_{side}": hand.keypoints, f"wrist_{side}": hand.wrist,
                    f"valid_{side}": hand.valid, f"confidence_{side}": hand.confidence,
                    f"wrist_angle_{side}": hand.wrist_angle_deg, f"twist_{side}": hand.twist_deg,
                    f"tracker_{side}": tracker_frames})
        calibration["wrists"][side] = cal.to_json()
    session.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(session.out / "timeline.npz", **out)
    session.write_json("calibration.json", calibration)
    print(f"Wrote {session.out / 'timeline.npz'} ({len(frame_utc)} frames, "
          f"{(len(frame_utc) - 1) / max(frame_utc[-1] - frame_utc[0], 1e-9):.1f} fps)")


# ---------------------------------------------------------------------- depth

def _camera_from_calibration(session):
    cal = session.calibration()["camera"]
    text, video = headset.find_recording(session.headset)
    rec = headset.read_tracking(text)
    width, height = headset.video_size(video)
    return headset.stereo_camera(rec, width, height, extrinsic=cal["extrinsic"],
                                 intrinsics=cal["intrinsics"], camera_axes=cal["camera_axes"]), video


def stage_depth(session, args):
    import cv2
    from .stereo import StereoDepth, save_frame, split_side_by_side
    camera, video = _camera_from_calibration(session)
    timeline = session.timeline()
    scale = float(timeline["image_size"][0]) / camera.width
    stereo = StereoDepth(camera, scale=scale)
    session.frames.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = len(timeline["frame_utc"])
    start, end = args.start or 0, min(args.end or n, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    t0 = time.monotonic()
    for i in range(start, end):
        ok, frame = cap.read()
        if not ok:
            print(f"video ended at frame {i}")
            break
        left, right = stereo.rectify(*split_side_by_side(frame))
        save_frame(session.frames, i, left, stereo.depth(left, right))
        if (i - start) % 30 == 0:
            rate = (i - start + 1) / max(time.monotonic() - t0, 1e-9)
            print(f"\rdepth {i + 1}/{end}  ({rate:.1f} frames/s)", end="", flush=True)
    print(f"\nSaved rectified left images and depth to {session.frames}")


# ---------------------------------------------------------------------- track

def load_mesh(path, scale=1.0):
    import trimesh
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.apply_scale(scale)
    return mesh


def stage_track(session, args):
    from . import object_track
    from .stereo import load_frame
    init = json.loads((Path(args.init) if args.init else session.out / "object_init.json").read_text())
    mesh_path = Path(args.mesh or init["mesh"]).expanduser()
    scale = float(args.mesh_scale if args.mesh_scale is not None else init.get("mesh_scale", 1.0))
    mesh = load_mesh(mesh_path, scale)
    model = object_track.ObjectModel(mesh)
    tl = session.timeline()
    n = len(tl["frame_utc"])
    frame0 = int(init["frame"])
    K = tl["K"]
    hands = {s: tl[f"keypoints_{s}"] for s in SIDES if f"keypoints_{s}" in tl}
    wrists = {s: tl[f"wrist_{s}"] for s in SIDES if f"wrist_{s}" in tl}
    _, depth0 = load_frame(session.frames, frame0)
    T0 = np.asarray(init["T_world_object"], float)
    if args.refine_init:
        center = T0[:3, :3] @ model.center + T0[:3, 3]
        T0, fit = object_track.initialize(model, depth0, tl["T_world_camera"][frame0], K, center,
                                          hands=[h[frame0] for h in hands.values()])
        print(f"initial pose refined: rms {fit.rms * 1000:.1f} mm")
    available = [i for i in range(n) if (session.frames / f"{i:06d}.png").exists()]

    def frames(indices):
        for i in indices:
            yield i, load_frame(session.frames, i)[1]

    def progress(i, ok, fit):
        if i % 30 == 0:
            print(f"\rtrack frame {i}: {'fit' if ok else 'held'} rms "
                  f"{fit.rms * 1000 if np.isfinite(fit.rms) else float('nan'):.1f} mm", end="", flush=True)

    forward = [i for i in available if i >= frame0]
    backward = [i for i in available if i <= frame0][::-1]
    a = object_track.track(model, frames(forward), tl["T_world_camera"], K, hands, wrists, T0,
                           progress=progress)
    b = object_track.track(model, frames(backward), tl["T_world_camera"], K, hands, wrists, T0,
                           progress=progress)
    print()
    pick = np.zeros(n, bool)
    pick[backward] = True
    poses = np.where(pick[:, None, None], b.poses, a.poses)
    out = {k: np.where(pick if v.ndim == 1 else pick[:, None, None], getattr(b, k), getattr(a, k))
           for k, v in (("valid", a.valid), ("rms", a.rms), ("inlier_fraction", a.inlier_fraction),
                        ("carried_by", a.carried_by))}
    np.savez_compressed(session.out / "object_track.npz", poses=poses, sides=np.array(sorted(hands)),
                        **out)
    session.write_json("object.json", {"mesh": str(mesh_path.resolve()), "mesh_scale": scale,
                                       "init_frame": frame0, "T_world_object_init": T0.tolist()})
    tracked = np.isfinite(poses[:, 0, 0])
    print(f"object: fitted to depth in {out['valid'][tracked].mean():.0%} of {tracked.sum()} frames, "
          f"carried by a hand in {(out['carried_by'][tracked] >= 0).mean():.0%}")


# --------------------------------------------------------------------- ground

def hand_surfaces(tl):
    out = {}
    for side in SIDES:
        key = f"keypoints_{side}"
        if key not in tl:
            continue
        kp = tl[key]
        good = np.flatnonzero(tl[f"valid_{side}"])
        if not len(good):
            continue
        surface = demo_hands.HandSurface(kp[good[0]])
        out[side] = (surface, surface.vertices(kp))
    return out


def stage_ground(session, args):
    from . import ground
    tl = session.timeline()
    obj = json.loads((session.out / "object.json").read_text())
    mesh = load_mesh(obj["mesh"], obj["mesh_scale"])
    track = dict(np.load(session.out / "object_track.npz"))
    W, H = tl["image_size"]
    K = tl["K"]
    Kn = np.tile([K[0, 0] / W, K[1, 1] / H, K[0, 2] / W, K[1, 2] / H], (len(tl["frame_utc"]), 1))
    hands = {s: tl[f"keypoints_{s}"] for s in SIDES if f"keypoints_{s}" in tl}
    holds = ground.load_holds(args.holds) if args.holds else None
    times = tl["frame_utc"] - tl["frame_utc"][0]
    poses, keypoints, report = ground.ground(times, hands, hand_surfaces(tl), track["poses"], mesh,
                                             tl["T_world_camera"], Kn, variant=args.variant,
                                             max_nfev=args.max_nfev, holds=holds,
                                             penetration_weight=args.penetration_weight)
    np.savez_compressed(session.out / "grounded.npz", object_poses=poses,
                        **{f"keypoints_{s}": k for s, k in keypoints.items()})
    session.write_json("ground_report.json", report)
    print(f"{len(report['holds'])} hold(s) grounded; report in {session.out / 'ground_report.json'}")


# --------------------------------------------------------------------- export

def stage_sync(session, args):
    from . import stream_sync
    stream_sync.sync_session(session, args)


def stage_export(session, args):
    from . import export
    export.export(session, args.side, use_grounded=not args.ungrounded, fps=args.fps)


# ----------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("pull", "inspect", "hands", "depth", "track", "ground",
                                          "export", "view", "all", "sync"))
    parser.add_argument("session", type=Path)
    g = parser.add_argument_group("pull")
    g.add_argument("--adb-serial")
    g = parser.add_argument_group("hands")
    g.add_argument("--clock-guess", type=float, default=0.0,
                   help="Approximate host minus headset clock (s) if it exceeds 30 s")
    g.add_argument("--frame-time", choices=("auto", "predict", "first-image"), default="auto")
    g.add_argument("--extrinsic", choices=("head_from_camera", "camera_from_head"),
                   default="head_from_camera")
    g.add_argument("--intrinsics", choices=("auto", "per-eye", "full"), default="auto")
    g.add_argument("--camera-axes", choices=("openxr", "opencv"), default="openxr")
    g.add_argument("--depth-scale", type=float, default=0.5,
                   help="Depth and stored image resolution as a fraction of one eye")
    g.add_argument("--left-tracker")
    g.add_argument("--right-tracker")
    g.add_argument("--mount-height", type=float, default=0.025,
                   help="Tracker centre above the wrist joint centre (m), palms down")
    g.add_argument("--mount-back", type=float, default=0.02,
                   help="Tracker centre behind the wrist joint centre along the forearm (m)")
    g.add_argument("--hand-orientation", choices=("imu", "tracker"), default="imu")
    g.add_argument("--calibration-from", help="Reuse the wrist calibration of another session (hands: "
                   "instead of this one; sync: for a hand whose own calibration fails; a calibration.json "
                   "or sync_report.json)")
    g = parser.add_argument_group("sync (live stereo sessions; also uses the hands options for "
                                  "trackers and the wrist calibration)")
    g.add_argument("--latency", type=float, help="Stream latency in seconds instead of the estimate "
                   "from head motion")
    g.add_argument("--baseline-m", type=float, help="Stereo baseline (m), for depth from disparity")
    g.add_argument("--nominal-camera", action="store_true",
                   help="Use the nominal camera rotation in the head frame, not the estimate")
    g.add_argument("--camera-offset", type=float, nargs=3, metavar=("RIGHT", "UP", "BACK"),
                   help="Left camera position in the OpenXR head frame (m; x right, y up, z back). "
                        "Default 0 0 0 (camera at the head origin); fitted on 2026-11-16: "
                        "-0.023 -0.011 -0.074 with the factory --baseline-m 0.064")
    g.add_argument("--no-plot", action="store_true")
    g = parser.add_argument_group("depth")
    g.add_argument("--start", type=int)
    g.add_argument("--end", type=int)
    g = parser.add_argument_group("track")
    g.add_argument("--mesh", help="Object mesh (OBJ/PLY/STL); default from object_init.json")
    g.add_argument("--mesh-scale", type=float, help="0.001 for a mesh in millimetres")
    g.add_argument("--init", help="Initial pose JSON (default processed/object_init.json)")
    g.add_argument("--no-refine-init", dest="refine_init", action="store_false")
    g = parser.add_argument_group("ground")
    g.add_argument("--variant", choices=("object-only", "object-hand-root"), default="object-only")
    g.add_argument("--max-nfev", type=int, default=50)
    g.add_argument("--penetration-weight", type=float, default=5.0)
    g.add_argument("--holds", help="JSON list of holds (side, start, end, regions) to use "
                   "instead of detection")
    g = parser.add_argument_group("export")
    g.add_argument("--side", choices=("left", "right", "both"), default="both")
    g.add_argument("--ungrounded", action="store_true")
    g.add_argument("--fps", type=float)
    g = parser.add_argument_group("view")
    g.add_argument("--port", type=int, default=8120)
    g.add_argument("--frame", type=int, help="Frame to open at (default: first with depth)")
    g.add_argument("--video", action="store_true",
                   help="live-stereo session: render the 3D view beside the camera image to "
                        "processed/hands_view.mp4 (headless Chrome as the viser client) instead of serving")
    g.add_argument("--video-azimuth", type=float, default=30.0,
                   help="video viewpoint, degrees about the vertical from the head camera's side (default 30)")
    g.add_argument("--video-elevation", type=float, default=55.0, help="degrees above the hands (default 55)")
    g.add_argument("--video-distance", type=float, default=0.65, help="metres from the hands (default 0.65)")
    g.add_argument("--video-step", type=int, default=1, help="render every Nth frame (default 1)")
    args = parser.parse_args(argv)
    session = Session(args.session)
    if args.stage == "pull":
        pull(session, args.adb_serial)
    elif args.stage == "inspect":
        print(headset.inspect(session.headset))
    elif args.stage == "view" and (session.root / "stereo").exists() and (session.out / "synced.npz").exists():
        from . import hands_view  # a live-stereo session: MANO hands / glove keypoints, camera, depth
        if args.video:
            hands_view.record_video(session, port=args.port + 1, baseline_m=args.baseline_m,
                                    azimuth_deg=args.video_azimuth, elevation_deg=args.video_elevation,
                                    distance=args.video_distance, step=args.video_step)
        else:
            hands_view.serve(session, args.port, initial_frame=args.frame, baseline_m=args.baseline_m)
    elif args.stage == "view":
        from . import viewer
        viewer.serve(session, args.port, mesh=args.mesh,
                     mesh_scale=args.mesh_scale if args.mesh_scale is not None else 1.0,
                     initial_frame=args.frame)
    else:
        stages = ["hands", "depth", "track", "ground", "export"] if args.stage == "all" else [args.stage]
        for stage in stages:
            print(f"=== {stage} ===", flush=True)
            globals()[f"stage_{stage}"](session, args)


if __name__ == "__main__":
    main()
