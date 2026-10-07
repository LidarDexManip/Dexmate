#!/usr/bin/env python3
"""Run a bounded, operator-confirmed right-arm joint waypoint loop.

The default is a dry run: it prints clipped targets without publishing motion.
Use --run only after checking the path and restoring the robot motion plugin.
This script has no collision checking and is limited to small offsets from the
live starting pose.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def wait_for_motion(arm, handle, target, emit, timeout, interval=0.25):
    deadline = time.monotonic() + timeout
    try:
        while not handle.is_done:
            actual = arm.get_joint_pos()
            emit("motion", {
                "state": handle.state,
                "message": handle.message,
                "position_rad": actual,
                "target_minus_actual_rad": target - actual,
                "timestamp_ns": arm.get_timestamp_ns(),
            })
            if time.monotonic() >= deadline:
                handle.wait(timeout=0)
                raise TimeoutError("Waypoint exceeded its deadline; cancelling.")
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
        if handle.state != "finished":
            raise RuntimeError(f"Motion ended as {handle.state}: {handle.message}")
    finally:
        if not handle.is_done:
            handle.cancel()
            emit("cancel_requested", {"state": handle.state})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="New JSONL output file")
    parser.add_argument("--run", action="store_true",
                        help="Publish motion after the confirmation prompt")
    parser.add_argument("--cycles", type=int, default=1,
                        help="Number of back-and-forth cycles; default: 1")
    parser.add_argument("--forever", action="store_true",
                        help="Continue until Ctrl-C; overrides --cycles")
    parser.add_argument("--velocity-scale", type=float, default=0.1,
                        help="SDK velocity scale, limited to 0.1 or less")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Maximum seconds allowed for each waypoint")
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
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
            arm = robot.right_arm
            status = robot.get_component_status(show=False)
            emit("components", status)
            state = status.get("states", {}).get("right_arm", {})
            if state.get("connection") != 2 or state.get("operation") != 2:
                raise RuntimeError("Right arm is not connected and enabled")

            current = arm.get_joint_pos()
            limits = arm.joint_pos_limit
            if limits is None:
                raise RuntimeError("Joint limits unavailable; cannot validate targets")
            if not np.isfinite(current).all() or not np.isfinite(limits).all():
                raise RuntimeError("Invalid joint positions or limits")

            # Small, deliberately conservative offsets for J3, J5, and J7.
            # Other joints remain at their live starting positions.
            offsets = np.zeros_like(current)
            offsets[[2, 4, 6]] = [0.20, 0.15, 0.20]
            targets = [np.clip(current + offsets, limits[:, 0], limits[:, 1]),
                       np.clip(current - offsets, limits[:, 0], limits[:, 1])]
            emit("plan", {
                "joints": arm.joint_name,
                "start_rad": current,
                "offset_rad": offsets,
                "targets_rad": targets,
                "limits_rad": limits,
                "velocity_scale": args.velocity_scale,
                "cycles": "forever" if args.forever else args.cycles,
                "dry_run": not args.run,
            })

            arm._ensure_target_publisher()
            receiver_present = arm._target_publisher.has_subscribers()
            emit("motion_target_receiver", {
                "topic": arm._target_publisher.topic,
                "has_subscribers": receiver_present,
                "subscriber_sessions": arm._target_publisher.subscriber_count(),
            })

            if not args.run:
                print("Dry run only: no motion command was published.", flush=True)
                return
            if not receiver_present:
                raise RuntimeError(
                    "No receiver for motion targets; motion plugin must be restored first")

            print(
                "WARNING: right arm will move between the printed targets. "
                "No collision checking is available. Keep the area clear, "
                "watch the arm, and keep e-stop ready.", flush=True)
            if input("Type RUN RIGHT ARM LOOP to continue: ") != "RUN RIGHT ARM LOOP":
                emit("cancelled_by_operator", {})
                return

            total = 0
            while args.forever or total < args.cycles:
                for index, target in enumerate(targets):
                    emit("waypoint_command", {
                        "cycle": total + 1,
                        "waypoint": index,
                        "target_rad": target,
                        "velocity_scale": args.velocity_scale,
                        "timeout_seconds": args.timeout,
                    })
                    handle = arm.move_to_joint_pos(
                        target, velocity_scale=args.velocity_scale)
                    wait_for_motion(arm, handle, target, emit, args.timeout)
                    emit("waypoint_finished", {
                        "cycle": total + 1,
                        "waypoint": index,
                        "position_rad": arm.get_joint_pos(),
                    })
                total += 1
            emit("loop_finished", {"cycles": total})
        except BaseException as error:
            emit("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            robot.shutdown()
            emit("shutdown", "completed")


if __name__ == "__main__":
    main()