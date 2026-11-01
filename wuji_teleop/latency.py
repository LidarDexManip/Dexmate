"""Bounded wrist-latency telemetry and report generation.

The control loops only append small in-memory records.  JSON serialization is
deferred until shutdown, so a slow disk cannot delay a hardware command.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time

import numpy as np


SIDES = ("left", "right")
WRIST_JOINTS = (4, 5, 6)


class AsyncCsvLogger:
    """Bounded background text writer used by the 100 Hz hardware path."""

    def __init__(self, path, header, *, max_rows=4096):
        self.path = Path(path)
        self._rows = deque(maxlen=int(max_rows))
        self._condition = threading.Condition()
        self._closed = False
        self.dropped = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write(header)
        self._thread = threading.Thread(target=self._run, name="teleop-csv-writer",
                                        daemon=True)
        self._thread.start()

    def write(self, row):
        with self._condition:
            if self._closed:
                return
            if len(self._rows) == self._rows.maxlen:
                self._rows.popleft()
                self.dropped += 1
            self._rows.append(str(row))
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._rows or self._closed)
                rows = list(self._rows)
                self._rows.clear()
                done = self._closed and not rows
            if rows:
                self._handle.writelines(rows)
            if done:
                self._handle.close()
                return

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()


@dataclass(frozen=True)
class LatencySample:
    """One 100 Hz arm-I/O observation with its latest tracker mapping metadata."""

    timestamp: float
    state: str
    mapping_revision: int | None
    tracker_timestamp_ns: int | None
    source_received_at: float | None
    mapping_started_at: float | None
    mapping_finished_at: float | None
    tracker_orientation_delta_deg: dict[str, list[float]]
    mapped_orientation_delta_deg: dict[str, list[float]]
    command_sent_at: float
    measured_received_at: float
    arm_read_s: float
    arm_write_s: float
    io_duration_s: float
    goal: dict[str, list[float]]
    command: dict[str, list[float]]
    measured: dict[str, list[float]]


def _summary(values):
    values = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(values):
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": int(len(values)),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "max_ms": float(np.max(values) * 1000.0),
    }


def _velocity_lag(samples, side, joint, max_lag_s=1.5):
    if len(samples) < 4:
        return None
    timestamps = np.asarray([sample.timestamp for sample in samples])
    command = np.asarray([sample.command[side][joint] for sample in samples])
    measured = np.asarray([sample.measured[side][joint] for sample in samples])
    dt = float(np.median(np.diff(timestamps)))
    if not math.isfinite(dt) or dt <= 0 or np.ptp(command) < 0.03:
        return None
    command_velocity = np.diff(command)
    measured_velocity = np.diff(measured)
    max_lag = min(int(max_lag_s / dt), len(command_velocity) // 3)
    best_correlation, best_lag = -math.inf, 0
    for lag in range(max_lag + 1):
        left = command_velocity[:len(command_velocity) - lag or None]
        right = measured_velocity[lag:]
        if len(left) < 3 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if math.isfinite(correlation) and correlation > best_correlation:
            best_correlation, best_lag = correlation, lag
    if not math.isfinite(best_correlation):
        return None
    return {
        "lag_ms": float(best_lag * dt * 1000.0),
        "correlation": best_correlation,
        "command_range_rad": float(np.ptp(command)),
    }


class LatencyRecorder:
    """Thread-safe, bounded collector shared by tracker mapping and hardware I/O."""

    def __init__(self, path=None, *, max_samples=60_000, configuration=None,
                 clock=time.monotonic):
        self.path = None if path is None else Path(path)
        self.max_samples = int(max_samples)
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.configuration = dict(configuration or {})
        self.clock = clock
        self._lock = threading.Lock()
        self._samples = deque(maxlen=self.max_samples)
        self._mapping = None
        self._mapping_events = deque(maxlen=max(1, self.max_samples // 2))
        self._hand_writes = deque(maxlen=self.max_samples)
        self._deadline_misses = deque(maxlen=self.max_samples)
        self._dropped = 0

    def record_mapping(self, *, revision, tracker_timestamp_ns, source_received_at,
                       started_at, finished_at, goal,
                       tracker_orientation_delta_deg=None,
                       mapped_orientation_delta_deg=None):
        event = {
            "revision": None if revision is None else int(revision),
            "tracker_timestamp_ns": (None if tracker_timestamp_ns is None else
                                     int(tracker_timestamp_ns)),
            "source_received_at": source_received_at,
            "started_at": float(started_at),
            "finished_at": float(finished_at),
            "goal": {side: np.asarray(values, dtype=float).tolist()
                     for side, values in goal.items()},
            "tracker_orientation_delta_deg": dict(tracker_orientation_delta_deg or {}),
            "mapped_orientation_delta_deg": dict(mapped_orientation_delta_deg or {}),
        }
        with self._lock:
            previous = self._mapping
            self._mapping = event
            if (event["revision"] is not None and
                    event["source_received_at"] is not None and
                    (previous is None or event["revision"] != previous["revision"])):
                self._mapping_events.append(event)

    def record_io(self, *, state, goal, command, measured, command_sent_at,
                  measured_received_at, arm_read_s, arm_write_s, io_duration_s):
        with self._lock:
            mapping = self._mapping
            if len(self._samples) == self._samples.maxlen:
                self._dropped += 1
            self._samples.append(LatencySample(
                timestamp=float(command_sent_at), state=str(state),
                mapping_revision=None if mapping is None else mapping["revision"],
                tracker_timestamp_ns=None if mapping is None else mapping["tracker_timestamp_ns"],
                source_received_at=None if mapping is None else mapping["source_received_at"],
                mapping_started_at=None if mapping is None else mapping["started_at"],
                mapping_finished_at=None if mapping is None else mapping["finished_at"],
                tracker_orientation_delta_deg=(
                    {} if mapping is None else mapping["tracker_orientation_delta_deg"]),
                mapped_orientation_delta_deg=(
                    {} if mapping is None else mapping["mapped_orientation_delta_deg"]),
                command_sent_at=float(command_sent_at),
                measured_received_at=float(measured_received_at),
                arm_read_s=float(arm_read_s), arm_write_s=float(arm_write_s),
                io_duration_s=float(io_duration_s),
                goal={side: np.asarray(values, dtype=float).tolist()
                      for side, values in goal.items()},
                command={side: np.asarray(values, dtype=float).tolist()
                         for side, values in command.items()},
                measured={side: np.asarray(values, dtype=float).tolist()
                          for side, values in measured.items()}))

    def record_hand_write(self, duration_s):
        with self._lock:
            self._hand_writes.append(float(duration_s))

    def record_deadline_miss(self, overrun_s):
        if overrun_s <= 0:
            return
        with self._lock:
            self._deadline_misses.append(float(overrun_s))

    def report(self):
        with self._lock:
            samples = list(self._samples)
            mappings = list(self._mapping_events)
            hand_writes = list(self._hand_writes)
            deadline_misses = list(self._deadline_misses)
            dropped = self._dropped

        intervals = []
        source_times = [event["source_received_at"] for event in mappings
                        if event["source_received_at"] is not None]
        for previous, current in zip(source_times, source_times[1:]):
            if current > previous:
                intervals.append(current - previous)
        median_interval = float(np.median(intervals)) if intervals else None
        gap_threshold = None if median_interval is None else max(0.05, 2.5 * median_interval)
        frame_gaps = [] if gap_threshold is None else [
            interval for interval in intervals if interval > gap_threshold]
        mapping_durations = [event["finished_at"] - event["started_at"]
                             for event in mappings]
        first_command = {}
        for sample in samples:
            revision = sample.mapping_revision
            fresh = (sample.source_received_at is not None and
                     0.0 <= sample.command_sent_at - sample.source_received_at <= 0.5)
            if (sample.state == "running" and fresh and revision is not None and
                    revision not in first_command and
                    sample.mapping_finished_at is not None and
                    sample.command_sent_at >= sample.mapping_finished_at):
                first_command[revision] = sample
        goal_to_command, source_to_command = [], []
        for event in mappings:
            sample = first_command.get(event["revision"])
            if sample is None or event["revision"] is None:
                continue
            goal_to_command.append(sample.command_sent_at - event["finished_at"])
            if event["source_received_at"] is not None:
                source_to_command.append(sample.command_sent_at - event["source_received_at"])

        running = [sample for sample in samples if sample.state == "running"]
        wrist_lag = {
            side: {f"j{joint + 1}": _velocity_lag(running, side, joint)
                   for joint in WRIST_JOINTS}
            for side in SIDES
        }
        wrist_lag_summary = {
            side: _summary([
                entry["lag_ms"] / 1000.0 for entry in joints.values()
                if entry is not None])
            for side, joints in wrist_lag.items()
        }
        io_count = len(samples)
        missed = len(deadline_misses)
        report = {
            "schema_version": 1,
            "generated_at_unix_s": time.time(),
            "configuration": self.configuration,
            "buffer": {"samples": io_count, "dropped_oldest": dropped,
                       "capacity": self.max_samples},
            "stages": {
                "tracker_frame_interval": _summary(intervals),
                "mapping": _summary(mapping_durations),
                "goal_to_first_command": _summary(goal_to_command),
                "source_receive_to_command": _summary(source_to_command),
                "arm_read": _summary([sample.arm_read_s for sample in samples]),
                "arm_write": _summary([sample.arm_write_s for sample in samples]),
                "hand_write": _summary(hand_writes),
                "io_loop": _summary([sample.io_duration_s for sample in samples]),
                "deadline_overrun": _summary(deadline_misses),
            },
            "io_deadlines": {
                "ticks": io_count,
                "misses": missed,
                "miss_fraction": None if not io_count else missed / io_count,
            },
            "tracker_frames": {
                "count": len(source_times),
                "gap_threshold_ms": (None if gap_threshold is None else
                                     gap_threshold * 1000.0),
                "gaps": len(frame_gaps),
                "largest_gap_ms": (None if not frame_gaps else
                                   max(frame_gaps) * 1000.0),
            },
            "wrist_command_to_measured": wrist_lag,
            "wrist_command_to_measured_summary": wrist_lag_summary,
            "acceptance": {
                "source_receive_to_command_p95_below_75_ms": (
                    _summary(source_to_command)["p95_ms"] is not None and
                    _summary(source_to_command)["p95_ms"] < 75.0),
                "command_to_measured_median_below_300_ms": None,
                "external_end_to_end_required": True,
            },
            "samples": [asdict(sample) for sample in samples],
        }
        lags = [entry["lag_ms"] for side in wrist_lag.values() for entry in side.values()
                if entry is not None]
        if lags:
            median_lag = float(np.median(lags))
            report["acceptance"]["command_to_measured_median_below_300_ms"] = median_lag < 300.0
            report["acceptance"]["command_to_measured_median_ms"] = median_lag
        return report

    def close(self):
        if self.path is None:
            return None
        report = self.report()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return report


def format_report(report):
    """Return a compact terminal summary for an existing latency JSON report."""
    stages = report.get("stages", {})
    ranked = [(values.get("p95_ms"), name) for name, values in stages.items()
              if isinstance(values, dict) and values.get("p95_ms") is not None]
    lines = []
    if ranked:
        value, name = max(ranked)
        lines.append(f"Slowest p95 stage: {name} ({value:.1f} ms)")
    for side in SIDES:
        entries = report.get("wrist_command_to_measured", {}).get(side, {})
        values = [entry["lag_ms"] for entry in entries.values() if entry is not None]
        if values:
            lines.append(f"{side.capitalize()} wrist command→measured median: "
                         f"{np.median(values):.1f} ms")
        else:
            lines.append(f"{side.capitalize()} wrist: insufficient motion for lag estimate")
    deadlines = report.get("io_deadlines", {})
    if deadlines.get("ticks"):
        lines.append(f"100 Hz deadline misses: {deadlines.get('misses', 0)}/"
                     f"{deadlines['ticks']} ({100 * deadlines.get('miss_fraction', 0):.1f}%)")
    return "\n".join(lines)
