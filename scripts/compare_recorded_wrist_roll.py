#!/usr/bin/env python3
"""Compare configurations on a selected raw-tracker CSV interval.

Recalibrates at the interval's first frame, so this is a controlled replay,
not a reconstruction of the original session's unknown calibration state.
The fixed-centre probe retains measured rotations but removes translation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from wuji_teleop.gmr_arm_retarget import GmrArmRetargeter, TrackerArmMapper
from wuji_teleop.vega_bimanual_sim import arm_joint_names, body_names, load_bimanual_scene
from wuji_teleop.xr_tracker_source import TrackerFrame, TrackerSnapshot

SIDES = ("left", "right")


def replay(model, arms, config, frames, fixed_center):
    solver = GmrArmRetargeter(model, {s: arm_joint_names(s) for s in SIDES}, config)
    mapper = TrackerArmMapper(solver, left_serial="left", right_serial="right")
    home = {s: arms[s].home.copy() for s in SIDES}
    t0, rest = frames[0]
    mapper.calibrate(TrackerSnapshot(TrackerFrame(rest, None, 0, t0), 0, 0, None), home, t0)
    # Offset used by the corrected tracker-centre mapping, in tracker-local axes.
    offsets = {s: rest[s][:3, :3].T @ np.array(
        [0, 0, -float(solver.tracker_mount_heights.get(s, 0))]) for s in SIDES}
    elbow0 = {s: solver.configuration.get_transform_frame_to_world(
        body_names(s)["elbow"], "body").translation().copy() for s in SIDES}
    q = {s: home[s].copy() for s in SIDES}
    rows = []
    for revision, (t, raw) in enumerate(frames[1:], 1):
        poses = {s: p.copy() for s, p in raw.items()}
        if fixed_center:
            for s, p in poses.items():
                p[:3, 3] = rest[s][:3, 3] + rest[s][:3, :3] @ offsets[s] - p[:3, :3] @ offsets[s]
        goals = mapper.update(TrackerSnapshot(TrackerFrame(poses, None, revision, t),
                                             revision, revision, None), t)
        q.update(goals)
        rows.append({"time": t, "q": {s: a.tolist() for s, a in q.items()},
                     "status": dict(mapper.status),
                     "error": {s: mapper.errors[s] for s in goals},
                     "elbow_m": {s: float(np.linalg.norm(solver.configuration.get_transform_frame_to_world(
                         body_names(s)["elbow"], "body").translation() - elbow0[s])) for s in SIDES}})
    metrics = {}
    for s in SIDES:
        joint = np.array([r["q"][s] for r in rows])
        errors = np.array([r["error"][s] for r in rows if s in r["error"]])
        metrics[s] = {"wrist_range_deg_j5_j6_j7": np.degrees(np.ptp(joint[:, 4:], axis=0)).tolist(),
                      "bend_rms_deg_j6_j7": float(np.degrees(np.sqrt(np.mean((joint[:, 5:] - home[s][5:]) ** 2)))),
                      "elbow_max_m": max(r["elbow_m"][s] for r in rows),
                      "jump_rejections": mapper.jump_rejections[s],
                      "position_p95_m": float(np.percentile(errors[:, 0], 95)) if len(errors) else None,
                      "orientation_p95_deg": float(np.degrees(np.percentile(errors[:, 1], 95))) if len(errors) else None}
    return {"metrics": metrics, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("configs", nargs="+")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--output", default="/tmp/gmr-recorded-roll-comparison.json")
    args = p.parse_args()
    grouped = {}
    with open(args.csv) as stream:
        for row in csv.DictReader(stream):
            if any(v is None for v in row.values()):
                continue
            t = float(row["t"])
            if not args.start <= t <= args.end:
                continue
            pose = np.eye(4)
            pose[:3, 3] = [float(row[f"tracker_{a}"]) for a in "xyz"]
            pose[:3, :3] = np.array([float(row[f"tracker_r{i}{j}"]) for i in range(3) for j in range(3)]).reshape(3, 3)
            grouped.setdefault(t, {})[row["side"]] = pose
    frames = [(t, ps) for t, ps in sorted(grouped.items()) if set(ps) == set(SIDES)]
    if len(frames) < 2:
        p.error("Need at least two complete bilateral frames")
    model, _, arms, _ = load_bimanual_scene()
    report = {config: {mode: replay(model, arms, config, frames, mode == "fixed_center")
                       for mode in ("recorded", "fixed_center")} for config in args.configs}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({c: {mode: result["metrics"] for mode, result in modes.items()}
                      for c, modes in report.items()}, indent=2))


if __name__ == "__main__":
    main()
