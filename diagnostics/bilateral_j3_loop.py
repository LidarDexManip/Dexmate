#!/usr/bin/env python3
"""Run a guarded bilateral J2/J3 up/down loop.

The default is a dry run. The right J2/J3 start is mirrored from the live
left J2/J3 pose; every other joint stays at its live starting position.
Motion requires --run and an exact confirmation.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def wait_for_motion(arm, handle, target, emit, timeout):
    deadline = time.monotonic() + timeout
    try:
        while not handle.is_done:
            emit("motion", {
                "state": handle.state,
                "message": handle.message,
                "position_rad": arm.get_joint_pos(),
                "target_minus_actual_rad": target - arm.get_joint_pos(),
            })
            if time.monotonic() >= deadline:
                handle.cancel()
                raise TimeoutError("J3 waypoint exceeded its deadline")
            time.sleep(0.2)
        if handle.state != "finished":
            raise RuntimeError(f"Motion ended as {handle.state}: {handle.message}")
    finally:
        if not handle.is_done:
            handle.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="New JSONL output file")
    parser.add_argument("--run", action="store_true",
                        help="Publish motion after the confirmation prompt")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of up/down cycles; default: 1")
    parser.add_argument("--forever", action="store_true",
                        help="Continue until Ctrl-C")
    parser.add_argument("--offset", type=float, default=0.5,
                        help="J2/J3 offset in radians; default: 0.5")
    parser.add_argument("--velocity-scale", type=float, default=0.05,
                        help="SDK velocity scale; default: 0.05")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Maximum seconds allowed per waypoint")
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if not 0 < args.offset <= 0.5:
        parser.error("--offset must be greater than 0 and at most 0.5")
    if not 0 < args.velocity_scale <= 0.1:
        parser.error("--velocity-scale must be greater than 0 and at most 0.1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    import numpy as np
    from dexcontrol.robot import Robot

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    with args.log.open("x") as log:
        def emit(event, data):
            record = {"utc": datetime.now(timezone.utc).isoformat(),
                      "event": event, "data": data}
            line = json.dumps(record, default=encode)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        robot = Robot()
        try:
            arms = {"left": robot.left_arm, "right": robot.right_arm}
            status = robot.get_component_status(show=False)
            emit("components", status)
            states = status.get("states", {})
            for name in arms:
                state = states.get(f"{name}_arm", {})
                if state.get("connection") != 2 or state.get("operation") != 2:
                    raise RuntimeError(f"{name} arm is not connected and enabled")

            current_positions = {
                name: arm.get_joint_pos() for name, arm in arms.items()
            }
            plans = {}
            for name, arm in arms.items():
                current = current_positions[name]
                limits = arm.joint_pos_limit
                if limits is None or not np.isfinite(current).all() or not np.isfinite(limits).all():
                    raise RuntimeError(f"Invalid position or limits for {name} arm")
                base = current.copy()
                if name == "right":
                    # J2 and J3 are mirrored across the left/right arms.
                    base[1] = -current_positions["left"][1]
                    base[2] = -current_positions["left"][2]
                    base = np.clip(base, limits[:, 0], limits[:, 1])
                offset = np.zeros_like(base)
                offset[[1, 2]] = args.offset
                targets = [
                    np.clip(base + offset, limits[:, 0], limits[:, 1]),
                    np.clip(base - offset, limits[:, 0], limits[:, 1]),
                ]
                arm._ensure_target_publisher()
                receiver_present = arm._target_publisher.has_subscribers()
                emit("plan", {
                    "arm": name,
                    "joints": arm.joint_name,
                    "start_rad": current,
                    "base_rad": base,
                    "j2_j3_offset_rad": args.offset,
                    "targets_rad": targets,
                    "limits_rad": limits,
                    "receiver_present": receiver_present,
                    "subscriber_sessions": arm._target_publisher.subscriber_count(),
                })
                if not receiver_present:
                    raise RuntimeError(f"No motion target receiver for {name} arm")
                plans[name] = (arm, base, targets)

            if not args.run:
                print("Dry run only: no motion command was published.", flush=True)
                return

            print(
                "WARNING: both arms will move only at J2/J3 between the printed targets. "
                "No collision checking is available; keep the area clear and keep "
                "the e-stop ready.", flush=True)
            if input("Type RUN BOTH J2 J3 LOOP to continue: ") != "RUN BOTH J2 J3 LOOP":
                emit("cancelled_by_operator", {})
                return

            right_arm, right_base, _ = plans["right"]
            if not np.allclose(right_base, current_positions["right"]):
                emit("mirror_alignment_command", {
                    "arm": "right",
                    "target_rad": right_base,
                    "velocity_scale": args.velocity_scale,
                })
                alignment_handle = right_arm.move_to_joint_pos(
                    right_base, velocity_scale=args.velocity_scale)
                wait_for_motion(
                    right_arm, alignment_handle, right_base, emit, args.timeout)
                emit("mirror_alignment_finished", {
                    "position_rad": right_arm.get_joint_pos(),
                })

            total = 0
            while args.forever or total < args.cycles:
                for waypoint in range(2):
                    handles = []
                    for name, (arm, _, targets) in plans.items():
                        target = targets[waypoint]
                        emit("waypoint_command", {
                            "arm": name,
                            "cycle": total + 1,
                            "waypoint": waypoint,
                            "target_rad": target,
                            "velocity_scale": args.velocity_scale,
                        })
                        handles.append((name, arm, target,
                                        arm.move_to_joint_pos(
                                            target,
                                            velocity_scale=args.velocity_scale)))
                    for name, arm, target, handle in handles:
                        wait_for_motion(arm, handle, target, emit, args.timeout)
                        emit("waypoint_finished", {
                            "arm": name,
                            "cycle": total + 1,
                            "waypoint": waypoint,
                            "position_rad": arm.get_joint_pos(),
                        })
                total += 1

            # Restore both arms to the exact joint targets captured before alignment.
            for name, (arm, _, _) in plans.items():
                start_target = current_positions[name]
                emit("restore_command", {
                    "arm": name,
                    "target_rad": start_target,
                    "velocity_scale": args.velocity_scale,
                })
                restore_handle = arm.move_to_joint_pos(
                    start_target, velocity_scale=args.velocity_scale)
                wait_for_motion(
                    arm, restore_handle, start_target, emit, args.timeout)
                emit("restore_finished", {
                    "arm": name,
                    "position_rad": arm.get_joint_pos(),
                })
            emit("loop_finished", {"cycles": total})
        except BaseException as error:
            emit("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            robot.shutdown()
            emit("shutdown", "completed")


if __name__ == "__main__":
    main()