"""Jog the articulated Vega left arm joint by joint. No glove needed.

A small control window has one row per left-arm joint: - and + buttons that
repeat while held, a slider spanning that joint's sim limit (the official URDF
range), the measured
angle, and the original URDF range for reference. The MuJoCo window shows the
arm. Both arms start at home. Simulation only; nothing is sent to a robot.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
import tkinter as tk
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .simulation import passive_viewer
from .vega_arm_sim import ARM_JOINT_NAMES, SHOULDER_BODY_NAME, load_articulated_scene
from .vega_hand_sim import _vega_build_function

JOG_STEPS = (0.01, 0.05, 0.10)


def _urdf_ranges():
    """Hardware ranges from the unmodified converted Vega model, if present."""
    try:
        root = ET.parse(Path(_vega_build_function()())).getroot()
    except (OSError, ET.ParseError, ValueError):
        return {}
    ranges = {}
    for joint in root.iter("joint"):
        name, span = joint.get("name"), joint.get("range")
        if name in ARM_JOINT_NAMES and span:
            low, high = (float(value) for value in span.split())
            ranges[name] = (low, high)
    return ranges


def main():
    model, data, arm, hand = load_articulated_scene()
    arm.initialize()
    hand.initialize()
    target = arm.home.copy()
    urdf = _urdf_ranges()
    # Unlimited sim joints (the elbow) get the URDF range, or ±pi, as a slider span.
    lower, upper = arm.lower.copy(), arm.upper.copy()
    for index, name in enumerate(ARM_JOINT_NAMES):
        if not (np.isfinite(lower[index]) and np.isfinite(upper[index])):
            lower[index], upper[index] = urdf.get(name, (-math.pi, math.pi))
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHOULDER_BODY_NAME)

    with passive_viewer(model, data, lambda _key: None) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[2] = 0
            viewer.cam.lookat[:] = data.xpos[shoulder]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15

        root = tk.Tk()
        root.title("Vega left arm jog")
        step = tk.DoubleVar(value=JOG_STEPS[1])
        values = [tk.DoubleVar(value=float(q)) for q in target]
        readouts = []

        def apply(index):
            target[index] = float(np.clip(values[index].get(), lower[index], upper[index]))
            values[index].set(round(target[index], 4))
            arm.set_target(target)

        def jog(index, direction):
            values[index].set(round(values[index].get() + direction * step.get(), 4))
            apply(index)

        def home():
            for index, value in enumerate(arm.home):
                values[index].set(float(value))
                apply(index)

        def print_angles():
            measured = data.qpos[arm.qpos_ids]
            print("Measured angles (rad):", flush=True)
            for name, value in zip(ARM_JOINT_NAMES, measured):
                print(f"  {name}: {value:+.3f}  ({math.degrees(value):+.1f} deg)", flush=True)

        header = ("Joint", "", "Command (rad, sim limits)", "", "Measured", "URDF range")
        for column, text in enumerate(header):
            tk.Label(root, text=text, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=column, padx=4, pady=(6, 2))
        for index, name in enumerate(ARM_JOINT_NAMES):
            row = index + 1
            tk.Label(root, text=name).grid(row=row, column=0, padx=6, sticky="w")
            tk.Button(root, text="-", width=3, repeatdelay=300, repeatinterval=60,
                      command=lambda i=index: jog(i, -1)).grid(row=row, column=1)
            tk.Scale(root, variable=values[index], from_=lower[index], to=upper[index],
                     resolution=0.01, orient="horizontal", length=320,
                     command=lambda _value, i=index: apply(i)).grid(row=row, column=2)
            tk.Button(root, text="+", width=3, repeatdelay=300, repeatinterval=60,
                      command=lambda i=index: jog(i, 1)).grid(row=row, column=3)
            readout = tk.Label(root, width=20, anchor="w", font=("TkFixedFont", 9))
            readout.grid(row=row, column=4, padx=6)
            readouts.append(readout)
            span = urdf.get(name)
            tk.Label(root, text="-" if span is None else f"[{span[0]:+.3f}, {span[1]:+.3f}]",
                     font=("TkFixedFont", 9)).grid(row=row, column=5, padx=6)

        controls = tk.Frame(root)
        controls.grid(row=len(ARM_JOINT_NAMES) + 1, column=0, columnspan=6, pady=8)
        tk.Label(controls, text="Jog step (rad):").pack(side="left")
        for size in JOG_STEPS:
            tk.Radiobutton(controls, text=f"{size:g}", variable=step,
                           value=size).pack(side="left")
        tk.Button(controls, text="Home", command=home).pack(side="left", padx=10)
        tk.Button(controls, text="Print angles", command=print_angles).pack(side="left")

        last = time.monotonic()

        def tick():
            nonlocal last
            if not viewer.is_running():
                root.destroy()
                return
            now = time.monotonic()
            steps = min(int((now - last) / model.opt.timestep), 50)
            last += steps * model.opt.timestep
            with viewer.lock():
                for _ in range(steps):
                    mujoco.mj_step(model, data)
            viewer.sync()
            for readout, value in zip(readouts, data.qpos[arm.qpos_ids]):
                readout.config(text=f"{value:+.3f} ({math.degrees(value):+6.1f} deg)")
            root.after(16, tick)

        print("Use the 'Vega left arm jog' window. Hold - or + to jog a joint.", flush=True)
        print("j1 shoulder pitch, j4 elbow (negative bends). Sliders span the URDF "
              "limits.", flush=True)
        root.after(16, tick)
        root.mainloop()


if __name__ == "__main__":
    main()
