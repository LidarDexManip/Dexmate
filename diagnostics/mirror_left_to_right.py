#!/usr/bin/env python3
"""Preview or move the right arm to the mirrored live left-arm pose."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True,
                        help="New JSONL output file")
    parser.add_argument("--run", action="store_true",
                        help="Move the right arm after confirmation")
    parser.add_argument("--velocity-scale", type=float, default=0.05,
                        help="SDK velocity scale; default: 0.05")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Maximum seconds to wait for motion")
    args = parser.parse_args()

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
            left = robot.left_arm
            right = robot.right_arm
            status = robot.get_component_status(show=False)
            emit("components", status)
            states = status.get("states", {})
            for name in ("left_arm", "right_arm"):
                state = states.get(name, {})
                if state.get("connection") != 2 or state.get("operation") != 2:
                    raise RuntimeError(f"{name} is not connected and enabled")

            left_position = left.get_joint_pos()
            right_position = right.get_joint_pos()
            right_limits = right.joint_pos_limit
            if right_limits is None:
                raise RuntimeError("Right-arm joint limits are unavailable")
            if not np.isfinite(left_position).all() or not np.isfinite(right_limits).all():
                raise RuntimeError("Invalid live position or right-arm limits")

            # Confirmed by the SDK's left/right predefined pose pairs.
            mirror_signs = np.array([-1, -1, -1, 1, -1, -1, -1])
            target = left_position * mirror_signs
            clipped_target = np.clip(target, right_limits[:, 0], right_limits[:, 1])
            if not np.allclose(target, clipped_target):
                raise RuntimeError(
                    "Mirrored pose exceeds right-arm limits; refusing to clip a full pose")

            right._ensure_target_publisher()
            receiver_present = right._target_publisher.has_subscribers()
            emit("plan", {
                "left_position_rad": left_position,
                "mirror_signs": mirror_signs,
                "right_start_rad": right_position,
                "right_target_rad": target,
                "right_limits_rad": right_limits,
                "receiver_present": receiver_present,
                "subscriber_sessions": right._target_publisher.subscriber_count(),
                "velocity_scale": args.velocity_scale,
                "dry_run": not args.run,
            })

            if not receiver_present:
                raise RuntimeError("No right-arm motion target receiver")
            if not args.run:
                print("Dry run only: no motion command was published.", flush=True)
                return

            print(
                "WARNING: the right arm will move to the mirrored full left-arm pose. "
                "No collision checking is available; clear the workspace and keep "
                "the e-stop ready.", flush=True)
            if input("Type RUN MIRRORED RIGHT POSE to continue: ") != "RUN MIRRORED RIGHT POSE":
                emit("cancelled_by_operator", {})
                return

            emit("motion_command", {
                "target_rad": target,
                "velocity_scale": args.velocity_scale,
                "timeout_seconds": args.timeout,
            })
            handle = right.move_to_joint_pos(target, velocity_scale=args.velocity_scale)
            try:
                handle.wait(timeout=args.timeout)
            finally:
                if not handle.is_done:
                    handle.cancel()
                    emit("cancel_requested", {"state": handle.state})
            if handle.state != "finished":
                raise RuntimeError(f"Motion ended as {handle.state}: {handle.message}")
            emit("motion_finished", {
                "state": handle.state,
                "position_rad": right.get_joint_pos(),
            })
        except BaseException as error:
            emit("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            robot.shutdown()
            emit("shutdown", "completed")


if __name__ == "__main__":
    main()