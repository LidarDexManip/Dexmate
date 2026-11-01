"""Bounded wrist-latency collection and report analysis."""

import json
import tempfile
import unittest

import numpy as np

from wuji_teleop.latency import LatencyRecorder, format_report


class LatencyRecorderTests(unittest.TestCase):
    def _record(self, recorder, count=120, delay_ticks=3):
        commands = np.sin(np.linspace(0, 8, count))
        measured = np.concatenate([np.zeros(delay_ticks), commands[:-delay_ticks]])
        for index in range(count):
            t = index * 0.01
            goal = {side: np.full(7, commands[index]) for side in ("left", "right")}
            recorder.record_mapping(
                revision=index, tracker_timestamp_ns=index * 10_000_000,
                source_received_at=t - 0.004, started_at=t - 0.003,
                finished_at=t - 0.001, goal=goal)
            command = {side: np.full(7, commands[index]) for side in ("left", "right")}
            actual = {side: np.full(7, measured[index]) for side in ("left", "right")}
            recorder.record_io(
                state="running", goal=goal, command=command, measured=actual,
                command_sent_at=t, measured_received_at=t - 0.0005,
                arm_read_s=0.0004, arm_write_s=0.0003, io_duration_s=0.001)

    def test_report_estimates_stages_and_wrist_lag(self):
        recorder = LatencyRecorder(max_samples=500)
        self._record(recorder)
        report = recorder.report()
        self.assertAlmostEqual(report["stages"]["mapping"]["p95_ms"], 2.0)
        self.assertAlmostEqual(
            report["stages"]["source_receive_to_command"]["p95_ms"], 4.0)
        lag = report["wrist_command_to_measured"]["left"]["j5"]
        self.assertAlmostEqual(lag["lag_ms"], 30.0, delta=10.0)
        self.assertGreater(lag["correlation"], 0.9)
        self.assertAlmostEqual(
            report["wrist_command_to_measured_summary"]["left"]["p50_ms"],
            30.0, delta=10.0)

    def test_buffer_drops_oldest_instead_of_growing(self):
        recorder = LatencyRecorder(max_samples=5)
        self._record(recorder, count=12)
        report = recorder.report()
        self.assertEqual(report["buffer"]["samples"], 5)
        self.assertEqual(report["buffer"]["dropped_oldest"], 7)

    def test_close_writes_json_and_formatter_compares_wrists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/latency.json"
            recorder = LatencyRecorder(path, max_samples=200)
            self._record(recorder)
            recorder.close()
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
        summary = format_report(report)
        self.assertIn("Slowest p95 stage", summary)
        self.assertIn("Left wrist", summary)
        self.assertIn("Right wrist", summary)


if __name__ == "__main__":
    unittest.main()
