# Contact-aware Wuji grasping

Status: deferred until the wrist-latency project passes acceptance.

## Captured problem

Glove teleoperation currently retargets finger position without using robot-hand
contact or impedance feedback. This makes it difficult to establish and retain
a grasp without excessive closure or object slip.

## Future constraints

- Preserve immediate operator opening, pause, E-stop, collision and hard-limit behavior.
- Begin with measured joint position/velocity/effort and available fingertip tactile data.
- Keep existing Wuji Hand 2 MIT gains (`kp=1.5`, `kd=0.1`) until a separate,
  simulation-first grasp controller is reviewed.
- Prefer a deterministic contact-hold reflex before adaptive or learned control.
- Do not mix grasping changes into the wrist-latency implementation.

## Research starting points

- [DOGlove: Dexterous Manipulation with a Low-Cost Open-Source Haptic Force Feedback Glove](https://roboticsproceedings.org/rss21/p104.html)
- [Shear-Based Grasp Control for Multifingered Underactuated Tactile Robotic Hands](https://ieeexplore.ieee.org/document/10972061/)
- [PP-Tac: Paper Picking Using Omnidirectional Tactile Feedback](https://roboticsproceedings.org/rss21/p056.html)
- [DexTeleop-0: Force-Aware Bimanual Dexterous Teleoperation](https://arxiv.org/abs/2606.23431)
- [CARD: Contact-Anchored Retargeting for Dexterous Manipulation](https://arxiv.org/abs/2610.36539)
