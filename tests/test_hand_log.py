"""Finger telemetry log: background writes, read-back, and a killed writer."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wuji_teleop.hand_log import STATE_DTYPE, HandLogger, load


class HandLoggerTests(unittest.TestCase):
    def setUp(self):
        self.prefix = Path(tempfile.mkdtemp()) / "run"

    def test_states_and_commands_read_back(self):
        times = iter([1.0, 2.0, 3.0])
        log = HandLogger(self.prefix, clock=lambda: next(times))
        q, v, effort = np.arange(20) * 0.1, np.full(20, 0.5), np.full(20, -0.02)
        log.state("right", 123456, q, v, effort)
        log.command("left", np.full(20, 0.3), np.full(20, 0.4))
        log.command("right", np.zeros(20))
        log.close()
        logs = load(self.prefix)
        (state,) = logs["states"]
        self.assertEqual((state["t"], state["stamp_us"], state["side"]), (1.0, 123456, 1))
        np.testing.assert_allclose(state["q"], q, atol=1e-6)
        np.testing.assert_allclose(state["effort"], effort, atol=1e-6)
        first, second = logs["commands"]
        np.testing.assert_allclose(first["goal"], 0.4, atol=1e-6)
        self.assertEqual(second["side"], 1)
        self.assertTrue(np.isnan(second["goal"]).all())  # no goal given

    def test_compliance_rows_read_back_and_older_logs_have_none(self):
        log = HandLogger(self.prefix, clock=lambda: 5.0)
        force = np.arange(15, dtype=float).reshape(5, 3)
        log.compliance("left", np.full(20, 0.5), np.full(20, 0.4), np.full(20, 0.3), force,
                       np.array([0, 1, 0, 0, 0.5]))
        log.close()
        (row,) = load(self.prefix)["compliance"]
        self.assertEqual((row["t"], row["side"]), (5.0, 0))
        np.testing.assert_allclose(row["compliant"] - row["observed"], 0.1, atol=1e-6)
        np.testing.assert_allclose(row["force"], force)
        self.assertEqual(row["weight"][1], 1.0)
        plain = HandLogger(Path(tempfile.mkdtemp()) / "run")
        plain.command("left", np.zeros(20))
        plain.close()
        self.assertEqual(len(load(plain.prefix)["compliance"]), 0)
        self.assertFalse(Path(f"{plain.prefix}.compliance.bin").exists())

    def test_glove_keypoints_read_back(self):
        log = HandLogger(self.prefix, clock=lambda: 2.5)
        keypoints = np.arange(63, dtype=float).reshape(21, 3) * 1e-3
        log.glove("right", 123, keypoints)
        log.close()
        (row,) = load(self.prefix)["glove"]
        self.assertEqual((row["t"], row["stamp_us"], row["side"]), (2.5, 123, 1))
        np.testing.assert_allclose(row["keypoints"], keypoints, atol=1e-7)

    def test_rows_after_close_are_ignored_and_close_is_idempotent(self):
        log = HandLogger(self.prefix)
        log.close()
        log.state("left", 0, np.zeros(20), np.zeros(20), np.zeros(20))
        log.close()
        self.assertEqual(len(load(self.prefix)["states"]), 0)

    def test_a_record_cut_short_by_a_kill_is_dropped(self):
        log = HandLogger(self.prefix)
        for stamp in (1, 2):
            log.state("left", stamp, np.zeros(20), np.zeros(20), np.zeros(20))
        log.close()
        path = Path(f"{self.prefix}.states.bin")
        path.write_bytes(path.read_bytes()[:STATE_DTYPE.itemsize + 10])
        self.assertEqual(list(load(self.prefix)["states"]["stamp_us"]), [1])


if __name__ == "__main__":
    unittest.main()
