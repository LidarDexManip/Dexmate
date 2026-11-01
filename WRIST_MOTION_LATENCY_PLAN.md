# Wrist motion latency

Status: active, simulation-first project.

## Goal

Measure and remove the roughly one-second perceived delay in hardware wrist
motion without weakening stale-frame holds, collision checking, joint limits,
pause or E-stop behavior. The default path is PICO Motion Trackers through
XRoboToolkit and GMR to Vega joints 5–7.

## Current baseline

Existing command/measured CSV logs indicate about 0.20–0.30 seconds of
command-to-measured wrist lag. They did not include tracker arrival, mapping,
IK or command-publication timestamps, so they could not locate upstream delay.

The implementation records those stages in a bounded in-memory buffer and
writes `--latency-report` JSON at shutdown. Use:

```bash
.venv/bin/python scripts/analyze_wrist_latency.py /tmp/vega-hardware-latency-*.json
```

`--proximal-speed` and `--wrist-speed` are separate. Both default to the old
1.0 rad/s behavior; evaluate wrist values of 1.0, 1.5 and 2.0 in fake-robot and
MuJoCo runs before requesting a physical Vega trial.

## Hardware observations and follow-up

The first saved hardware CSV had 1,003 running samples. Mapping/IK p95 was
about 1.9 ms and arm-I/O p95 about 0.16 ms. Source-receive-to-command p95 was
about 107 ms, but the shutdown JSON was absent, so deadline misses and the
report's command-to-measured estimate remain unconfirmed.

The operator also observed that rotating the left wrist causes about 45 degrees
of unintended wrist bend, while the right wrist behaves correctly. The existing
CSV contains joint goals, commands and feedback but not tracker orientation, so
it cannot identify whether the cause is tracker orientation mapping or physical
joint following. Compare tracker motion, mapped targets, and joints 5-7 during
isolated slow rotation before changing any left-side sign or gain. The
calibration-relative orientation diagnostic was removed after a possible
`KeyError: 'left'` path was identified; reintroduce it only with a tested,
defensive side/body lookup or a focused test.

## Acceptance

- Source-receive to command p95 below 75 ms.
- Command-to-measured wrist median below 300 ms.
- Externally measured end-to-end median below 350 ms and p95 below 500 ms.
- No overshoot, stale-frame motion, joint-limit, clearance, pause or E-stop regression.

## Validation order

1. Unit tests and deterministic tracker replay.
2. `--fake-robot --synthetic-input` dashboard rehearsal.
3. MuJoCo with live trackers.
4. Physical Vega only with explicit approval and direct E-stop access.

Do not modify submodules, firmware, saved network profiles or physical robot
settings for this project.
