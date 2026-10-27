# GMR IK improvement roadmap

Status: starting design note; no implementation is implied by this file.

## Goal

Improve the quality, naturalness, and robustness of the Vega-1U bimanual wrist-tracker retargeting while preserving the existing safety behavior and real-time control rate.

The immediate target is the MuJoCo `wuji_teleop` tracker mode. Physical hardware validation comes only after simulation acceptance and only when explicitly requested.

## Source

- Paper: [Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking](https://arxiv.org/abs/2510.02252), Araujo et al., 2025.
- Reference implementation: [YanjieZe/GMR](https://github.com/YanjieZe/GMR).
- Most relevant paper section: Section IV, "General Motion Retargeting."

## Important starting point

Dexmate already implements a substantial wrist-tracker adaptation of Section IV in `wuji_teleop/gmr_arm_retarget.py`:

1. Configurable human-to-robot key-body matching.
2. Cartesian rest-pose position and orientation alignment during calibration.
3. Per-key-body scaling, optionally adjusted by operator height.
4. Two sequential Mink differential-IK stages.
5. Joint-limit constraints, convergence stopping, a ten-iteration budget, and previous-frame warm starts.
6. A calibration-pose posture task to resolve the redundant elbow swivel.
7. Online additions that the offline paper does not address directly: pose filtering, target rate limiting, stale/lost tracker holding, and optional position/wrist decoupling.

This means the next session should not begin by replacing the solver. It should begin by measuring the current implementation and testing focused changes against the baseline.

## Paper-to-Dexmate mapping

| GMR idea | Current Dexmate state | Gap or adaptation |
| --- | --- | --- |
| Key-body map with position/orientation weights | Both wrist trackers map to `L_ee` and `R_ee` through JSON tables | GMR uses many whole-body key bodies; Dexmate observes only the wrists for this mode |
| Rest-pose Cartesian alignment | Pressing **R** computes per-wrist position and rotation offsets | Appropriate for online teleoperation; calibration repeatability still needs measurement |
| Non-uniform local scaling | Per-wrist scale plus optional operator-height ratio | Both wrist scales default to `1.0`; no reach- or shoulder-based calibration exists |
| Stage 1: rotations plus end-effector positions | Wrist position and rotation tasks run for both end effectors | Because wrists are the only key bodies, stage 1 and stage 2 have the same task structure |
| Stage 2: all key-body positions and rotations | Same wrist tasks run again with higher rotation weight | The second stage may add little beyond extra iterations unless its objective is made meaningfully different |
| Joint bounds | Mink `ConfigurationLimit` uses Vega URDF limits | The paper notes that tighter practical ranges can prevent non-human motion; Dexmate has no configurable comfort margin in this solver |
| Sequential warm start | Each frame starts from the previous IK result | Implemented and unit-tested |
| Ground-height correction | Not applied | Correct to omit for fixed-base arms; this paper step is for floating-base whole-body motion sequences |
| Artifact evaluation | End-effector residuals, target-step limits, saturation, and MuJoCo warnings are reported | Missing solve time, stage convergence, joint velocity/acceleration/jerk, limit margin, calibration drift, and repeatable recorded inputs |

## What the paper suggests for this robot

The most transferable lesson is not merely "use two IK calls." Retargeting quality depends on the target construction, scaling, constraints, initialization, and evaluation as a system.

For a fixed-base, wrist-only teleoperation setup:

- Rest alignment, scaling, warm starts, and bounded differential IK transfer directly.
- Whole-body grounding and foot-contact corrections do not apply.
- A large whole-body key-body map cannot be reproduced from two wrist trackers. Extra human constraints require an additional measured signal or a clearly labeled inferred prior.
- The arm's redundant elbow swivel must be chosen by a posture or naturalness objective because wrist pose alone does not determine it.
- Abrupt velocity spikes and self-intersection matter to Dexmate even though the paper primarily evaluates offline motion used to train tracking policies.

## Recommended first milestone: measurable GMR baseline

Build a deterministic, simulation-only replay and comparison harness before changing IK behavior.

### Deliverables

1. A small recorded or generated sequence format containing timestamps, headset pose, left/right tracker poses, and calibration frame.
2. A replay path that drives `TrackerArmMapper` without XR hardware and emits the existing run report plus the metrics below.
3. A baseline fixture covering:
   - neutral hold;
   - forward, lateral, and vertical reach;
   - wrist-only rotation;
   - near-workspace-boundary reach;
   - bilateral crossing/close approach;
   - one-frame spike, dropped frame, and reacquisition.
4. A comparison command that runs the same sequence against multiple IK configs and produces JSON or CSV results.

### Add these metrics

- Position and orientation error, p50/p95/max, per side.
- Solver time, p50/p95/max, and missed real-time budgets.
- Iterations and error reduction for each IK stage.
- Per-joint peak velocity, acceleration, and jerk.
- Minimum distance to each configured joint limit.
- Target rate-limit/filter interventions.
- Hold and reacquisition discontinuity.
- Minimum robot self-clearance if the existing collision checker can report it without changing behavior.

### Baseline acceptance

- Existing GMR and bimanual tests remain green.
- Replaying the same fixture produces deterministic joint goals and metrics.
- The harness needs no tracker, glove, Pico, network change, or physical robot.
- The current solver meets the existing tracker-mode gates: position-error p95 below `0.02 m`, zero ordinary-motion wrist saturation, maximum conditioned target step at or below `0.03 m`, and zero MuJoCo numerical warnings.

## Experiments, in priority order

Run one controlled change at a time against the replay baseline.

### Experiment A: make the two stages distinct

Test a stage-1 objective that establishes a stable gross arm pose, followed by a stage-2 objective that refines the full wrist pose. Candidate variants:

- Stage 1: wrist position with low or zero wrist-orientation weight; stage 2: wrist position plus full orientation.
- Stage 1: position plus only the orientation components that materially affect elbow/forearm placement; stage 2: full orientation.
- Compare both against the current paper-shaped wrist configuration and the existing decoupled-wrist configuration.

Do not assume the paper's exact whole-body stage split remains optimal when the only observed key bodies are end effectors.

Success means lower joint jerk and fewer large elbow changes without materially worsening p95 position/orientation error or solve time.

### Experiment B: practical joint-limit margins

Add optional per-joint comfort ranges inside the URDF hard limits, as suggested by the paper's observation that ranges sometimes need tightening to avoid non-human movement.

- Hard URDF limits remain the final safety boundary.
- Prefer a soft limit-avoidance cost or velocity damper before globally clipping the goal.
- Start with shoulder and elbow margins; measure wrist reach loss separately.

Success means improved minimum limit margin and fewer saturated frames with no meaningful loss of reachable workspace.

### Experiment C: elbow/null-space naturalness

Replace the single fixed calibration-posture preference with an optional arm-specific redundant-DoF objective. Possible versions, from simplest to most complex:

1. Weighted posture cost with stronger penalties on visually undesirable shoulder/elbow changes.
2. A preferred elbow plane or elbow-out task computed from shoulder-to-wrist geometry.
3. A slowly adapting posture target that preserves continuity while following large reaches.

The objective must weaken near unreachable targets so wrist tracking and hard limits remain authoritative.

Success means fewer elbow flips and lower joint jerk across repeated reaches, with unchanged hold behavior when a tracker is lost.

### Experiment D: operator-specific scaling

Compare the current height scalar with a short reach calibration. A practical calibration could sample neutral, forward, lateral, and upward reaches and estimate bounded per-axis or per-side scale factors.

Avoid unconstrained anisotropic fitting: it can distort motion or magnify tracking noise. Store calibration values in an explicit session/config object, not as hidden global state.

Success means more usable robot workspace, fewer unreachable targets, and consistent directional motion across operators.

### Experiment E: collision-aware retargeting

The paper identifies self-intersection as a damaging artifact, but its Section IV equations do not provide a complete collision-avoidance solution. Treat this as a Dexmate extension, not a faithful paper reproduction.

Start by measuring clearance and rejecting regressions. Only then evaluate a soft collision cost or velocity constraint compatible with Mink. Keep the existing downstream hardware clearance hold as an independent safety layer.

Success means greater minimum clearance in close/crossed-arm fixtures without oscillation or loss of ordinary reach accuracy.

## Suggested first coding slice

Keep the initial change small and reviewable:

1. Add per-stage solve diagnostics to `GmrArmRetargeter` without changing its returned goals.
2. Add an in-memory deterministic trajectory fixture to `tests/test_gmr_arm_retarget.py`.
3. Add a replay/benchmark script under `scripts/` that writes a machine-readable report under `/tmp`.
4. Record the baseline for both:
   - `wuji_teleop/ik_configs/xr_trackers_to_vega_1u.json`
   - `wuji_teleop/ik_configs/xr_trackers_to_vega_1u_decoupled.json`
5. Only after the baseline is stable, add one experimental config for Experiment A.

Likely files:

- `wuji_teleop/gmr_arm_retarget.py`
- `wuji_teleop/vega_bimanual_sim.py` only if report plumbing is reusable there
- `wuji_teleop/ik_configs/`
- `tests/test_gmr_arm_retarget.py`
- a new focused test/benchmark script under `scripts/`

Do not modify the URDF or any submodule for the first milestone.

## Verification ladder

1. Focused unit test:

   ```bash
   .venv/bin/python -m unittest tests.test_gmr_arm_retarget
   ```

2. Bimanual integration tests:

   ```bash
   .venv/bin/python -m unittest tests.test_vega_bimanual_sim
   ```

3. Repository verification suite from `AGENTS.md`.
4. Headless deterministic replay comparison.
5. MuJoCo viewer protocol with synthetic inputs.
6. Live trackers in MuJoCo, using repeated scripted operator motions.
7. Hardware only after simulation gates pass and the user explicitly requests it.

## Decision rules

- Favor measured improvement over closer textual resemblance to the paper.
- Keep paper-faithful behavior and Dexmate-specific extensions distinguishable in code and config names.
- Change one objective, constraint, or scale rule per experiment.
- Preserve lost/stale tracker hold, calibration behavior, joint hard limits, target filtering, and the downstream collision guard.
- Reject an approach that improves wrist residual while causing substantially worse jerk, limit proximity, clearance, latency, or reacquisition jumps.
- Do not tune from a single pose or a single operator.

## Questions to answer with data

- Does the current second stage measurably improve the wrist-only solution?
- Is coupled or decoupled wrist solving better for real tracker trajectories, not just synthetic FK targets?
- Which joints create most of the visible unnaturalness: shoulder redundancy, elbow posture, or wrist saturation?
- Is operator height enough, or does a reach calibration materially reduce clipping?
- How much latency and jitter come from tracker conditioning versus the IK itself?
- Are close-arm failures best handled inside IK or by the existing clearance hold?

## Resume checklist for the next session

1. Read this file and Section IV of the paper.
2. Inspect `wuji_teleop/gmr_arm_retarget.py` and the two XR tracker IK configs.
3. Run the focused GMR and bimanual tests before editing.
4. Implement the measurable-baseline milestone first.
5. Save baseline artifacts and exact commands.
6. Begin Experiment A only after baseline replay is deterministic.

## Working hypothesis

The best near-term gain is likely to come from making the stage objectives arm-specific and measurably different, then improving the redundant elbow/null-space objective. Reimplementing GMR wholesale is unlikely to help because its core Section IV pipeline is already present; the present limitation is sparse wrist-only observation plus a stage configuration that does not exploit the richer key-body structure used in the paper.
