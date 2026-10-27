#!/usr/bin/env python3
"""Compare elbow priors through the live TrackerArmMapper conditioning path."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wuji_teleop.gmr_arm_retarget import GmrArmRetargeter, TrackerArmMapper
from wuji_teleop.vega_bimanual_sim import arm_joint_names, body_names, load_bimanual_scene
from wuji_teleop.xr_tracker_source import TrackerFrame, TrackerSnapshot

SIDES = ("left", "right")


def evaluate(model, arms, config, frames=120):
    solver = GmrArmRetargeter(model, {s: arm_joint_names(s) for s in SIDES}, config)
    home = {s: arms[s].home.copy() for s in SIDES}
    solver.sync(home)
    rest = {s: solver.configuration.get_transform_frame_to_world(
        body_names(s)["ee"], "body").as_matrix() for s in SIDES}
    # Generate an actual tracker mounted above each virtual wrist centre.
    # This keeps the intended centre trajectory constant across mount settings.
    mount_offsets = {s: rest[s][:3, :3].T @ np.array(
        [0, 0, float(solver.tracker_mount_heights.get(s, 0))]) for s in SIDES}
    tracker_rest = {s: p.copy() for s, p in rest.items()}
    for s, pose in tracker_rest.items():
        pose[:3, 3] += pose[:3, :3] @ mount_offsets[s]
    mapper = TrackerArmMapper(solver, left_serial="left", right_serial="right")
    initial = TrackerSnapshot(TrackerFrame(tracker_rest, None, 0, 0.0), 0, 0, None)
    mapper.calibrate(initial, home, 0.0)
    elbow_rest = {s: solver.configuration.get_transform_frame_to_world(
        body_names(s)["elbow"], "body").translation().copy() for s in SIDES}
    rows = []
    revision = 0
    previous = home
    for phase in ("hold", "wrist_x", "wrist_y", "wrist_z", "reach", "loss_recovery"):
        for i in range(frames):
            revision += 1
            now = revision / 60.0
            # Smooth out-and-back movement, with zero endpoint velocity.
            wave = np.sin(np.pi * i / (frames - 1)) ** 2
            poses = {s: rest[s].copy() for s in SIDES}
            for side in SIDES:
                sign = 1 if side == "left" else -1
                if phase.startswith("wrist_"):
                    axis = "xyz".index(phase[-1])
                    rot = np.zeros(3)
                    rot[axis] = sign * 0.6 * wave
                    poses[side][:3, :3] = rest[side][:3, :3] @ Rotation.from_rotvec(rot).as_matrix()
                if phase == "reach":
                    poses[side][:3, 3] += wave * np.array([0.06, sign * 0.03, 0.04])
                if phase == "loss_recovery":
                    poses[side][:3, 3] += wave * np.array([0.02, 0, 0.02])
                poses[side][:3, 3] += poses[side][:3, :3] @ mount_offsets[side]
            if phase == "loss_recovery" and frames // 3 <= i < 2 * frames // 3:
                poses.pop("left")
            snapshot = TrackerSnapshot(TrackerFrame(poses, None, revision, now),
                                       revision, revision, None)
            goals = mapper.update(snapshot, now)
            held = {s: goals.get(s, previous[s]).copy() for s in SIDES}
            rows.append({"phase": phase, "time_s": now,
                         "q": {s: held[s].tolist() for s in SIDES},
                         "status": dict(mapper.status),
                         "errors": {s: list(mapper.errors[s]) for s in goals},
                         "elbow_m": {s: float(np.linalg.norm(
                             solver.configuration.get_transform_frame_to_world(
                                 body_names(s)["elbow"], "body").translation() - elbow_rest[s]))
                             for s in SIDES},
                         "solve_time_s": solver.last_diagnostics["solve_time_s"]})
            previous = held
    summary = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        part = [r for r in rows if r["phase"] == phase]
        errors = np.array([r["errors"][s] for r in part for s in r["errors"]])
        q = np.array([[r["q"][s] for s in SIDES] for r in part])
        jerk = np.diff(q, n=3, axis=0) * 60 ** 3
        margin = min(float(np.min(np.minimum(q[:, k] - arms[s].lower,
                                             arms[s].upper - q[:, k]))) for k, s in enumerate(SIDES))
        summary[phase] = {"position_p95_m": float(np.percentile(errors[:, 0], 95)),
                          "orientation_p95_deg": float(np.degrees(np.percentile(errors[:, 1], 95))),
                          "elbow_max_m": max(r["elbow_m"][s] for r in part for s in SIDES),
                          "jerk_max_rad_s3": float(np.abs(jerk).max()),
                          "limit_margin_rad": margin,
                          "solve_p95_ms": float(np.percentile([r["solve_time_s"] for r in part], 95) * 1000)}
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--output", default="/tmp/gmr-elbow-comparison.json")
    args = parser.parse_args()
    model, _, arms, _ = load_bimanual_scene()
    report = {p: evaluate(model, arms, p) for p in args.configs}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({p: r["summary"] for p, r in report.items()}, indent=2))


if __name__ == "__main__":
    main()
