#!/usr/bin/env python3
"""Inspect the right arm; optionally monitor one operator-confirmed folded move.

Default: reads only, followed by SDK shutdown. No dance playback.
Run on the robot using its configured dexcontrol Python environment.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def monitor(arm, handle, target, emit, timeout, interval=0.25):
    deadline = time.monotonic() + timeout
    try:
        while True:
            actual = arm.get_joint_pos()
            emit("motion", {
                "state": handle.state,
                "message": handle.message,
                "position_rad": actual,
                "target_minus_actual_rad": target - actual,
                "timestamp_ns": arm.get_timestamp_ns(),
            })
            if handle.is_done:
                if handle.state != "finished":
                    raise RuntimeError(f"Motion ended as {handle.state}: {handle.message}")
                return
            if time.monotonic() >= deadline:
                # Preserve the SDK's missing-plugin diagnosis if no status
                # response arrived. This does not send another motion command.
                handle.wait(timeout=0)
                raise TimeoutError("Right-arm test exceeded its deadline; cancelling.")
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
    finally:
        # A timeout from wait() alone does not cancel motion.
        if not handle.is_done:
            handle.cancel()
            emit("cancel_requested", {"state": handle.state})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-folded", action="store_true",
                        help="Offer an interactive, right-arm-only motion test")
    parser.add_argument("--log", type=Path, required=True,
                        help="New JSONL output file (existing files are preserved)")
    args = parser.parse_args()

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
            emit("versions", robot.get_version_info(show=False))
            status = robot.get_component_status(show=False)
            emit("components", status)
            arm = robot.right_arm
            current = arm.get_joint_pos()
            requested = arm.get_predefined_pose("folded")
            limits = arm.joint_pos_limit
            if limits is None:
                raise RuntimeError("Joint limits unavailable; cannot validate target")
            target = np.clip(requested, limits[:, 0], limits[:, 1])
            emit("right_arm", {
                "joints": arm.joint_name,
                "position_rad": current,
                "requested_folded_rad": requested,
                "effective_folded_rad": target,
                "target_minus_actual_deg": np.rad2deg(target - current),
                "limits_rad": limits,
            })
            try:
                emit("joint_errors", arm.get_joint_err())
            except ValueError as error:
                emit("joint_errors_unavailable", str(error))
            emit("brake_release_status", arm.get_brake_status())
            # Inspect the installed SDK's publisher matching information without
            # publishing a target. Pin this helper to the inspected 0.5.0 API.
            arm._ensure_target_publisher()
            time.sleep(1)
            receiver_present = arm._target_publisher.has_subscribers()
            emit("motion_target_receiver", {
                "topic": arm._target_publisher.topic,
                "has_subscribers": receiver_present,
                "subscriber_sessions": arm._target_publisher.subscriber_count(),
            })
            for _ in range(5):
                emit("state", arm.get_state())
                time.sleep(0.2)

            if not args.test_folded:
                return

            if not receiver_present:
                raise RuntimeError("No receiver for motion targets; inspect the robot-server motion plugin before testing")

            print("\nTEST: right arm only, folded target printed above, velocity_scale=0.1, "
                  "20-second deadline. SDK joint-limit clipping is shown explicitly.\n"
                  "This path has no collision checking. The elbow target is at its limit.\n"
                  "An operator must watch the robot with e-stop ready. Check the entire "
                  "arm/wrist path for the body, cables, and other objects.\n"
                  "If anything obstructs movement, use e-stop; do not push the arm.", flush=True)
            if input("Type TEST RIGHT ARM only when the path is clear and you are ready: ") != "TEST RIGHT ARM":
                emit("cancelled_by_operator", {})
                return

            # Recheck state after the operator's prompt, before issuing a target.
            status = robot.get_component_status(show=False)
            emit("components_before_test", status)
            state = status.get("states", {}).get("right_arm", {})
            if state.get("connection") != 2 or state.get("operation") != 2:
                raise RuntimeError("Right arm is not connected and enabled")
            actual = arm.get_joint_pos()
            if not arm._target_publisher.has_subscribers():
                raise RuntimeError("Motion-target receiver disappeared; test cancelled before publishing")
            if not np.isfinite(actual).all() or not np.isfinite(target).all():
                raise RuntimeError("Invalid joint positions")
            emit("test_command", {"target_rad": target, "velocity_scale": 0.1,
                                  "timeout_seconds": 20})
            handle = arm.move_to_joint_pos(target, velocity_scale=0.1)
            monitor(arm, handle, target, emit, timeout=20)
            emit("test_finished", {"target_minus_actual_rad": target - arm.get_joint_pos()})
        except BaseException as error:
            emit("failure", {"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            robot.shutdown()
            emit("shutdown", "completed")


if __name__ == "__main__":
    main()
