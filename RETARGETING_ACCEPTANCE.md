# Pico bilateral retargeting acceptance

Use this protocol before commanding physical Vega hardware. It validates only
the MuJoCo path: Pico hands drive arm joints 1–4, glove IMUs drive joints 5–7,
and glove skeletons drive the fingers.

## Quantitative gates

The bilateral run report at `/tmp/vega-bimanual-pico-live.json` evaluates these
for each arm:

- 95th-percentile IK residual is below `0.02 m`.
- Wrist saturation fraction is zero during ordinary neutral motion.
- No conditioned target update exceeds `0.03 m`.
- MuJoCo reports zero numerical warnings.

The report's `acceptance.quantitative_pass` is true only when both arms have
data and all three per-arm gates pass. Workspace clipping and rejected Pico
outliers are recorded for diagnosis; occasional clipping or rejection is safer
than forwarding an unreachable or implausible target.

## Viewer protocol

1. Connect both gloves and confirm `.100` and `.101` ping through their assigned
   Ethernet interfaces.
2. Open the DexStream Pico page, enter AR with controllers down, and confirm
   both hands are tracked.
3. Run `scripts/launch_wuji_vega_bimanual_pico.sh`, assume a comfortable pose,
   and press **R** once.
4. Hold both hands still for 10 seconds. Neither simulated arm should visibly
   jitter.
5. Move one hand at a time: forward/back, up/down, then outward/inward. The
   matching arm must move in the same direction without moving the other arm.
6. Rotate each wrist through an ordinary neutral range. The hand position
   should remain stable and the wrist should not pin against a joint limit.
7. Hide one tracked hand for at least one second. Its arm must hold. Reveal it
   again; it must reacquire at the held pose without a jump.
8. Close the viewer normally and inspect the printed report plus
   `/tmp/vega-bimanual-pico-live.json`.

Record the operator observations for steps 4–7 alongside the report. Physical
robot testing remains out of scope until all quantitative and viewer gates pass.
