# Hardware wrist-bend mismatch

Status: investigate next session. No corrective change has been made yet.

## Observed behavior

- In the hardware tracker teleoperation path, bending the **right wrist** moves
  in the wrong direction (described as bending toward the left).
- Right-wrist twist/rotation feels correct.
- The issue has been observed on hardware; the corresponding simulation motion
  looked smooth.
- Hardware teleoperation is reported working again. The earlier
  `FAULT in teleop_step: 'left'` was a separate software fault; do not assume it
  explains this wrist-direction mismatch.

## Goal

Identify whether the discrepancy is in tracker-frame interpretation, the
tracker-to-robot/GMR orientation mapping, right-arm joint conventions, or
physical joint following. Correct only the demonstrated cause while preserving
joint limits, filtering, collision checking, pause, and E-stop behavior.

## Next-session diagnostic sequence

1. Begin in MuJoCo with the same tracker setup and record a slow, isolated
   right-wrist bend, return to neutral, then a slow twist. Use the left wrist
   only as a comparison; do not change signs or gains yet.
2. Repeat on hardware only with the operator present, clear workspace, low
   motion amplitude, and direct E-stop access. Keep the arm at a safe pose and
   stop if the motion is unexpected.
3. For each motion, capture synchronized values for the right tracker pose,
   calibration-relative tracker rotation, mapped wrist target, joints 5–7
   commanded values, and measured joint feedback. Record the corresponding
   left-side values as a control.
4. Compare bend and twist independently. Determine whether the unexpected
   direction first appears in the tracker delta, mapped target, command, or
   measured motion before editing the mapping.
5. Make the smallest isolated correction, then replay the same motions in
   simulation and run the relevant tracker/GMR and hardware-teleop tests.

## Notes and guardrails

- Current hardware CSV records joint goals, commands, and feedback, but the
  calibration-relative wrist-orientation diagnostic was removed after a
  possible `KeyError: 'left'` path was identified. Reintroduce diagnostics only
  with a tested, defensive side/body lookup or a focused test.
- Existing tracker-to-robot calibration and right-side configuration should be
  inspected before changing joint signs, axis permutations, or speed limits.
- Do not use physical hardware to discover a sign convention by trial and
  error. Establish the expected direction in simulation first.
- Do not alter firmware, submodules, saved network profiles, or physical robot
  settings as part of the initial diagnosis.
