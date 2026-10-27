#!/usr/bin/env python3
"""Deterministic GMR replay benchmark for the two wrist-tracker IK configs."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wuji_teleop.gmr_arm_retarget import (
    GmrArmRetargeter, WRIST_BODIES, load_ik_config,
)
from wuji_teleop.vega_bimanual_sim import arm_joint_names, body_names, load_bimanual_scene


def pose_from_fk(model, arms, side, q):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[arms[side].qpos_ids] = q
    mujoco.mj_kinematics(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)["ee"])
    out = np.eye(4)
    out[:3, :3] = data.xmat[bid].reshape(3, 3)
    out[:3, 3] = data.xpos[bid]
    return out


def run(config_path, steps=180):
    model, _, arms, _ = load_bimanual_scene()
    sides = ("left", "right")
    names = {s: arm_joint_names(s) for s in sides}
    retargeter = GmrArmRetargeter(model, names, config_path)
    home = {s: arms[s].home.copy() for s in sides}
    retargeter.sync(home)
    rest = {WRIST_BODIES[s]: pose_from_fk(model, arms, s, home[s]) for s in sides}
    retargeter.calibrate(rest, home)
    previous = {s: home[s].copy() for s in sides}
    previous_velocity = {s: np.zeros(7) for s in sides}
    records = []
    for i in range(steps):
        t = i / 60.0
        human = {}
        for s in sides:
            pose = rest[WRIST_BODIES[s]].copy()
            sign = 1.0 if s == "left" else -1.0
            # Reach, wrist-only rotation, close approach, then a bounded spike.
            pose[:3, 3] += [0.04 * math.sin(1.2 * t), sign * 0.025 * math.sin(0.8 * t),
                            0.03 * math.sin(0.7 * t)]
            angle = (0.65 * math.sin(1.5 * t) if 1.0 < t < 2.0 else 0.0)
            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec([angle, 0, 0]).as_matrix()
            if i == steps // 2:
                pose[:3, 3] += [0.025, -sign * 0.015, 0.02]
            human[WRIST_BODIES[s]] = pose
        targets = retargeter.targets(human)
        started = time.perf_counter()
        goals = retargeter.solve(targets)
        elapsed = time.perf_counter() - started
        qvel = {s: (goals[s] - previous[s]) / (1 / 60.0) for s in sides}
        qacc = {s: (qvel[s] - previous_velocity[s]) / (1 / 60.0) for s in sides}
        qjerk = {s: (qacc[s] - (records[-1]["acc"][s] if records else np.zeros(7))) / (1 / 60.0)
                 for s in sides}
        errors = retargeter.errors(targets)
        records.append({"q": {s: goals[s].tolist() for s in sides},
                        "vel": {s: qvel[s].tolist() for s in sides},
                        "acc": {s: qacc[s].tolist() for s in sides},
                        "jerk": {s: qjerk[s].tolist() for s in sides},
                        "errors": {b: list(v) for b, v in errors.items()},
                        "solve_time_s": elapsed,
                        "diagnostics": retargeter.last_diagnostics})
        previous, previous_velocity = goals, qvel
    all_pos = [v for r in records for v in r["errors"].values()]
    pos = np.array([v[0] for v in all_pos])
    ori = np.array([v[1] for v in all_pos])
    vel = np.array([abs(x) for r in records for s in sides for x in r["vel"][s]])
    acc = np.array([abs(x) for r in records for s in sides for x in r["acc"][s]])
    jerk = np.array([abs(x) for r in records for s in sides for x in r["jerk"][s]])
    margins = np.array([min(float(np.min(np.asarray(r["q"][s]) - arms[s].lower)),
                            float(np.min(arms[s].upper - np.asarray(r["q"][s]))))
                        for r in records for s in sides])
    return {"config": str(config_path), "frames": steps,
            "metrics": {"position_error_m": {"p50": float(np.percentile(pos, 50)), "p95": float(np.percentile(pos, 95)), "max": float(pos.max())},
                        "orientation_error_rad": {"p50": float(np.percentile(ori, 50)), "p95": float(np.percentile(ori, 95)), "max": float(ori.max())},
                        "joint_velocity_rad_s_max": float(vel.max()),
                        "joint_acceleration_rad_s2_max": float(acc.max()),
                        "joint_jerk_rad_s3_max": float(jerk.max()),
                        "joint_limit_margin_rad_min": float(margins.min()),
                        "solve_time_ms": {"p50": float(np.percentile([r["solve_time_s"] for r in records], 50) * 1000),
                                          "p95": float(np.percentile([r["solve_time_s"] for r in records], 95) * 1000),
                                          "max": float(max(r["solve_time_s"] for r in records) * 1000)}},
            "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/gmr-baseline.json")
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "wuji_teleop" / "ik_configs"
    configs = [root / "xr_trackers_to_vega_1u.json",
               root / "xr_trackers_to_vega_1u_decoupled.json",
               root / "xr_trackers_to_vega_1u_stage_split.json",
               root / "xr_trackers_to_vega_1u_stage_split_soft.json"]
    report = {"runs": [run(p, args.steps) for p in configs]}
    Path(args.output).write_text(json.dumps(report, indent=2))
    for item in report["runs"]:
        print(item["config"])
        print(json.dumps(item["metrics"], indent=2))


if __name__ == "__main__":
    main()
