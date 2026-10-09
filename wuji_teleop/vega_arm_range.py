"""Open the articulated Vega arm scene with one slider per arm joint.

No glove or headset is needed. Both arms start at their home poses. Open the
viewer's Control panel on the right and drag a slider to command that joint.
Sliders span -pi to pi rad because the arm joints have no limits in this
scene. Simulation only; nothing is sent to a robot.
"""

from __future__ import annotations

import math

import mujoco
import mujoco.viewer

from .vega_arm_sim import ARM_ACTUATOR_NAMES, RIGHT_ARM_ACTUATOR_NAMES, load_articulated_scene


def main():
    model, data, arm, hand = load_articulated_scene()
    # Unlimited actuators get a -1..1 slider in the viewer. Give them a full turn.
    for name in ARM_ACTUATOR_NAMES + RIGHT_ARM_ACTUATOR_NAMES:
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        model.actuator_ctrllimited[actuator] = 1
        model.actuator_ctrlrange[actuator] = (-math.pi, math.pi)
    arm.initialize()
    hand.initialize()
    print("Open the Control panel on the right and drag the L_arm_j1..j7 sliders.", flush=True)
    print("j1 shoulder pitch (home 1.40), j4 elbow (home -1.40, negative bends).", flush=True)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
