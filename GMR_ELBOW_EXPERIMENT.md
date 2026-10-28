# Soft elbow preference experiment

The optional `elbow_preference` config adds a position task for each elbow to
both coupled IK stages. Calibration records the elbow, shoulder and palm
positions. The preferred elbow follows half of the palm target translation;
wrist target rotation does not change that preference. The cost fades to zero
over the final 20% of the approximate shoulder-elbow-palm reach. Recalibration
replaces these references. Missing-side tasks are omitted and its joints stay
frozen. Existing configs retain their previous behavior.

The experimental config now has per-side `inward_offset_m: 0.015` and
`inward_blend_distance_m: 0.10`, with the user's elbow cost of 9 preserved.
The offset activates when the palm target retracts in robot x relative to its
calibrated position: zero at neutral or farther forward, half (7.5 mm) at
5 cm retraction, and the full 15 mm at 10 cm or more. Smoothstep blending avoids
an abrupt onset. The preferred elbow shifts in y toward the torso centreline,
clamped so the offset itself does not cross it. This is a retraction-based
approximation of proximity to the body, not a torso-distance measurement.
Omitting `inward_offset_m` or setting it to zero restores the previous target.
This changes a preferred target, not a hard elbow allowance or guaranteed
physical displacement, and does not guarantee less wrist bending in every pose.

## Tracker mounting correction

`tracker_mount_height_m` contains the left/right distance from tracker origin
down to wrist centre, in metres. Zero disables correction. The experimental
config uses 0.03 m per side as an explicit trial estimate for the operator's
watch-style mounting, not a measured offset. With nonzero heights, calibrate with
forearms horizontal, palms down, and the trackers directly above the wrist
centres. Calibration derives each local offset from that orientation; no fixed
tracker axis is assumed. Each incoming centre is computed as `p + R @ offset`
before scaling, target filtering and IK. Press R after changing the mounting or
calibration pose. Restart to load changed configuration values.

This corrects the positional arc of a tracker mounted above a rotating wrist.
It does not alter the requested orientation or lock wrist bend joints. Offsets
along the forearm or sideways are not estimated by this height-only model. If
the tracker is not directly above the wrist centre, a full measured local
translation is needed instead. Existing general-motion logs cannot reliably
identify a wrist pivot without a deliberate stationary-centre calibration.

## Position-jump hold and measured roll alignment (2026-10-28)

The elbow config now enables `tracker_jump_threshold_m: 0.10`. A tracker
translation more than 10 cm from that side's last accepted translation is
rejected before target filtering or IK. That arm holds; the other may continue.
Repeated rejected samples cannot move the accepted reference. Returning near
the last accepted position resumes tracking; R explicitly accepts a new origin.
This also applies on reacquisition. A persistent tracking-origin change requires
R, and slow accumulated drift below the threshold is not detected. Configs
omitting the field retain the prior behavior.

The optional `xr_trackers_to_vega_1u_roll_aligned.json` adds per-tracker local
roll axes measured from the 439–459 s interval of the October 28 recording.
The leading singular vector of the successive local rotation vectors captured
97.8% (left) and 97.5% (right) of rotational variation. Axes use the converted
tracker-local coordinates from `xr_pose_to_robot`, not raw OpenXR coordinates.
They describe this particular mounting and must be re-estimated if the tracker
mounting changes materially.

At calibration, the measured roll axis is expressed in robot space and aligned
by the shortest rotation to the robot's joint-5 axis. Subsequent orientation
deltas are conjugated by that alignment about the calibrated robot orientation.
This preserves rotation angles and all three orientation degrees of freedom;
it remaps bend axes rather than suppressing intentional bend. Position targets,
mount correction, elbow costs and inward allowance are unchanged. The alignment
is fixed for that calibration, not a dynamically measured human forearm axis.

Comparison command (raw recording is local, not committed):

```bash
.venv/bin/python scripts/compare_recorded_wrist_roll.py \
  /tmp/gmr-wrist-demo-20261028.csv \
  wuji_teleop/ik_configs/xr_trackers_to_vega_1u_elbow.json \
  wuji_teleop/ik_configs/xr_trackers_to_vega_1u_roll_aligned.json \
  --start 439 --end 459 --output /tmp/gmr-recorded-roll-comparison.json
```

This recalibrates both variants at the first selected frame. It is a controlled
comparison, not an exact reproduction of the session's unknown calibration.
Recorded-input joint-6/7 RMS deviation from neutral dropped from 13.86 to 6.78
degrees on the left and 7.64 to 7.27 on the right. Both variants rejected 18 right
position samples. With translation synthetically removed while retaining the
measured rotations, the values were 11.54 to 6.37 and 7.27 to 6.71 degrees. This
is in-sample evidence, not independent validation or a claim of zero bend.

For the live experiment use the roll-aligned config with the existing tracker
launcher and `FINGER_INPUT=synthetic`. Hold forearms level, palms down, then R.
The launcher default and hardware launchers are unchanged. Set
`tracker_roll_axis_local` to an empty object to disable the alignment.

This is a competing weighted cost, not a strict null-space task. It does not
guarantee a fixed elbow, human elbow tracking, or collision avoidance. It uses
an inferred posture because there is no elbow tracker. The reach estimate is
approximate and does not describe the complete feasible workspace.

## Reproduce the comparison

```bash
.venv/bin/python scripts/evaluate_gmr_elbow.py \
  wuji_teleop/ik_configs/xr_trackers_to_vega_1u_stage_split_soft.json \
  wuji_teleop/ik_configs/xr_trackers_to_vega_1u_elbow.json \
  --output /tmp/gmr-elbow-comparison.json
```

This comparison uses `TrackerArmMapper`, including target filtering, explicit
timestamps, calibration, bilateral wrist rotations around three axes, a reach,
and one missing tracker followed by recovery. Joint goals are repeatable;
wall-clock timings are not. Derivatives describe IK goals, not physical robot
motion. Sequential segments preserve the warm start and filter history.

On the generated 0.6-rad wrist turns, maximum elbow displacement from calibration
fell from 12.39 to 4.25 cm (x), 6.08 to 1.76 cm (y), and 4.01 to 0.008 cm (z).
For x rotation, p95 palm position error increased from 0.70 to 2.32 mm, and p95
orientation error increased from 4.37 to 11.52 degrees. Reach position p95 was
0.61 mm. This warrants a simulation experiment, not a claim that the problem is
solved. Some segments have higher jerk; the largest x-rotation spike decreased.

The older `replay_gmr_baseline.py` goes directly to the solver, bypasses the
mapper, and introduces abrupt orientation changes. Its jerk values should not
be interpreted as evidence of live tracking quality. Neither benchmark yet
provides the full roadmap baseline or collision-clearance acceptance.

## Live simulation only

```bash
FINGER_INPUT=synthetic \
IK_CONFIG="$PWD/wuji_teleop/ik_configs/xr_trackers_to_vega_1u_elbow.json" \
scripts/launch_wuji_vega_bimanual_trackers.sh
```

Hold the desired neutral elbow posture with forearms forward and palms down,
then press R. Keep wrist positions steady and slowly rotate each wrist. Then
reach forward, sideways and upward. Assess elbow stability and whether the
orientation compromise is acceptable. The launcher default remains unchanged.
