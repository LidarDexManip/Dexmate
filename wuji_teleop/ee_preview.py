"""Single-arm Cartesian preview GUI. Simulation by default; no Pico required."""
import argparse
from dataclasses import dataclass
import os
import subprocess
import time
import tkinter as tk
from tkinter import messagebox

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .ee_dashboard import ComponentPanel, KEY_AXES, jog_target, drag_target
from .collision_check import Clearance, ArmClearanceChecker, PathFollower, PlanningError, join_arms, split_arms
from .hardware_teleop import (HardwareTeleop, TeleopLimits, FakeArms, FakeHands,
                             DexcontrolArms, WujiHands, reset_pose, _StaticRetargeter)
from .retarget import WujiRetargeter
from .simulation import passive_viewer
from .sources import GloveSource, SourceSnapshot
from .vega_bimanual_sim import SIDES, load_bimanual_scene, body_names, _side_ik


class EmptySource:
    def read(self):
        return SourceSnapshot(None, 0, 0, None)


class HoldMapping:
    status = {s: "GUI hold" for s in SIDES}

    def recenter(self, q):
        pass

    def update(self, q, now):
        return q


@dataclass
class Preview:
    start: np.ndarray
    fingers: dict
    follower: PathFollower


def straight_path(start, goal, checker, fingers, side, margin=0.03):
    """Reject unsafe paths; never move the inactive arm to find a detour."""
    start, goal = np.asarray(start), np.asarray(goal)
    other = slice(7, 14) if side == "left" else slice(0, 7)
    if start.shape != (14,) or goal.shape != (14,) or not np.isfinite([start, goal]).all():
        raise PlanningError("Expected finite 14-joint poses")
    if not np.allclose(start[other], goal[other], atol=1e-9):
        raise PlanningError("The inactive arm must remain fixed")
    for s in SIDES:
        checker.set_fingers(s, fingers[s])
    count = max(2, int(np.ceil(np.max(np.abs(goal - start)) / 0.01)) + 1)
    for index, q in enumerate(np.linspace(start, goal, count)):
        clearance = checker.clearance(q)
        if clearance.distance_m < margin:
            location = "start pose" if index == 0 else f"{index / (count - 1):.0%} along path"
            raise PlanningError(f"Path rejected: {clearance.distance_m * 100:.1f} cm at "
                                f"{clearance.pair} ({location}); needs {margin * 100:.1f} cm")
    return Preview(start.copy(), {s: fingers[s].copy() for s in SIDES},
                   PathFollower([start.copy(), goal.copy()], max_velocity=0.15,
                                max_acceleration=0.3))


class TableChecker:
    """Conservative infinite tabletop plane, evaluated on robot mesh vertices."""
    def __init__(self, checker, height=None):
        self.checker, self.height = checker, height
        self.meshes = []
        model = checker.model
        for geom in range(model.ngeom):
            if geom in getattr(checker, "unknown_hand_geoms", ()):
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and body.startswith(
                    ("L_arm_", "R_arm_", "l_", "r_", "wuji_left_wrist", "wuji_right_wrist")):
                mesh = model.geom_dataid[geom]
                first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                self.meshes.append((geom, model.mesh_vert[first:first + count]))

    def set_fingers(self, side, angles):
        self.checker.set_fingers(side, angles)

    def clearance(self, q):
        result = self.checker.clearance(q)
        if self.height is None:
            return result
        data = self.checker.data
        for geom, vertices in self.meshes:
            z = vertices @ data.geom_xmat[geom].reshape(3, 3)[2] + data.geom_xpos[geom, 2]
            gap = float(np.min(z)) - self.height
            if gap < result.distance_m:
                result = Clearance(gap, "arm/hand and tabletop plane")
        return result


class PreviewTeleop(HardwareTeleop):
    required_hand_sides = SIDES

    def execute(self, preview):
        with self._lock:
            if self.state != "paused" or any(self.estop_flags):
                raise ValueError("Pause and clear E-stop before executing")
            if not self.arms_io.health()[0]:
                raise ValueError("Robot heartbeat is not live")
            for q in (join_arms(self.command), join_arms(self.measured)):
                if np.max(np.abs(q - preview.start)) > 0.02:
                    raise ValueError("Robot pose changed; preview again")
            for s in SIDES:
                if np.max(np.abs(self.hand_command[s] - preview.fingers[s])) > 0.02:
                    raise ValueError("Finger pose changed; preview again")
                if s not in self.required_hand_sides:
                    continue
                age = self.hands_io.feedback_age(s)
                if age is None or age > 0.5:
                    raise ValueError("Hand feedback is stale")
                measured = self.hands_io.positions()[s]
                if measured is None or np.max(np.abs(measured - preview.fingers[s])) > 0.05:
                    raise ValueError("Measured fingers changed; preview again")
            self.reset_q = {s: q.copy() for s, q in split_arms(preview.follower.waypoints[-1]).items()}
            self._follower = preview.follower
            self._follow_start = self.clock()
            self.state = "resetting"
            self._log("Executing reviewed path; fingers held. Space pauses; Esc E-stop.")

    def request_reset(self):
        return False

    def io_step(self):
        if not self.arms_io.health()[0]:
            self.request_pause()
        if self.state in ("running", "resetting"):
            positions = self.hands_io.positions()
            for side in self.required_hand_sides:
                age = self.hands_io.feedback_age(side)
                if age is None or age > 0.5 or positions[side] is None:
                    self.request_pause()
                elif self.state == "resetting" and np.max(np.abs(
                        positions[side] - self.hand_command[side])) > 0.05:
                    self.request_pause()
        super().io_step()

    def _step_reset(self, now, measured):
        if self._check_tracking(measured):
            return
        elapsed = now - self._follow_start
        self.command = {s: q.copy() for s, q in split_arms(self._follower.at(elapsed)).items()}
        if elapsed >= self._follower.duration:
            self._follower = None
            self._freeze()
            self.state = "paused"
            self._log("Target reached. Preview another target or enable glove control.")


class PreviewGUI:
    def __init__(self, root, viewer, model, data, arms, hands, controller, side, glove, hardware):
        self.root, self.viewer = root, viewer
        self.model, self.data, self.arms, self.hands = model, data, arms, hands
        self.controller, self.side, self.glove, self.hardware = controller, side, glove, hardware
        self.preview, self.play_started, self.reviewed = None, None, False
        self.checker = TableChecker(ArmClearanceChecker(model, arms, hands))
        self.ik = _side_ik(model, data, arms[side], upper_only=False)
        self.ik.iterations = 120
        self.ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)["ee"])
        root.title(f"Vega {side} end effector — {'HARDWARE' if hardware else 'SIMULATION'}")
        layout = tk.Frame(root)
        layout.pack(fill="both", expand=True)
        controls = tk.Frame(layout)
        controls.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        sidebar = tk.Frame(layout)
        sidebar.pack(side="right", fill="both", padx=8, pady=8)
        ComponentPanel(sidebar, self.component_status).pack(fill="x")
        tk.Label(controls, text="Robot frame: x forward, y left, z up. Metres and degrees.").pack()
        tk.Label(controls, text="Table height: 0.762 m (30 in). Plane spans all x/y; other obstacles are not modelled.").pack()
        table_row = tk.Frame(controls)
        table_row.pack()
        tk.Label(table_row, text="Robot base origin height above floor (m):").pack(side="left")
        self.base_height = tk.StringVar(value="0.15")
        self.table_confirmed = tk.BooleanVar(value=False)
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.model_base_height = float(data.xpos[base, 2])
        tk.Entry(table_row, textvariable=self.base_height, width=9).pack(side="left")
        self.base_height.trace_add("write", self.table_changed)
        tk.Scale(controls, variable=self.base_height, from_=0, to=0.6, resolution=0.005,
                 orient="horizontal", label="Adjust base height to align the tabletop", length=450).pack()
        tk.Checkbutton(controls, text="Base height verified against the real robot (required for hardware)",
                       variable=self.table_confirmed).pack()
        tk.Label(controls, text="Initial 0.15 m is a model assumption; visually align before measuring.", fg="#a03000").pack()
        self.table_changed()
        self.values = []
        fields = tk.Frame(controls)
        fields.pack()
        for name in ("X", "Y", "Z", "Roll", "Pitch", "Yaw"):
            column = tk.Frame(fields)
            column.pack(side="left", padx=4)
            tk.Label(column, text=name).pack()
            value = tk.StringVar(value="0")
            tk.Entry(column, textvariable=value, width=9).pack()
            value.trace_add("write", self.invalidate)
            self.values.append(value)
        for label, callback in (("Use current pose", self.current_pose), ("Preview path", self.make_preview),
                ("Execute reviewed path", self.execute), ("Enable glove (arm holds)", self.glove_mode),
                ("Pause / hold (Space)", self.pause), ("Software E-stop (Esc)", controller.request_estop),
                ("Release software E-stop", controller.request_release)):
            tk.Button(controls, text=label, command=lambda f=callback: self.action(f)).pack(fill="x", padx=8, pady=2)
        self.input_notice = tk.StringVar(value="Keyboard and drag edit the preview only.")
        tk.Label(sidebar, text="TARGET CONTROL", font=("TkDefaultFont", 10, "bold")).pack(pady=(10, 2))
        tk.Label(sidebar, text="Click this panel, then use arrows for X/Y; PgUp/PgDn for Z.\n"
                 "I/K roll · J/L pitch · U/O yaw · Shift for fine steps.", justify="left").pack()
        self.drag_origin = np.zeros(3)
        self.canvases = {}
        self.auto_job = None
        for plane, title in (("top", "Top view — drag X/Y"), ("side", "Side view — drag X/Z")):
            tk.Label(sidebar, text=title).pack()
            canvas = tk.Canvas(sidebar, width=260, height=170, bg="#18212d", highlightthickness=1, takefocus=True)
            canvas.pack()
            canvas.bind("<Button-1>", lambda event, p=plane: self.drag(event, p))
            canvas.bind("<B1-Motion>", lambda event, p=plane: self.drag(event, p))
            canvas.bind("<ButtonRelease-1>", lambda event: self.schedule_preview())
            self.canvases[plane] = canvas
        tk.Label(sidebar, textvariable=self.input_notice, wraplength=340, justify="left").pack(pady=4)
        tk.Button(sidebar, text="Full-body view", command=self.overview).pack(fill="x")
        tk.Label(controls, text="Full Vega-1U URDF + Wuji hands; fixed base/lift/torso/head.").pack()
        self.status = tk.StringVar()
        tk.Label(controls, textvariable=self.status, justify="left", wraplength=650).pack(padx=8, pady=8)
        root.bind("<space>", lambda event: self.pause())
        root.bind("<Escape>", lambda event: controller.request_estop())
        root.bind("<KeyPress>", self.keyboard)
        self.current_pose()
        self.overview()
        root.after(30, self.tick)

    def component_status(self):
        status = self.controller.status()
        result = {}
        if not self.hardware:
            result["Arm controller"] = ("off", "SIMULATED — no robot commands")
            for side in SIDES:
                result[f"{side.capitalize()} hand feedback"] = ("off", "simulated")
        else:
            alive, age = self.controller.arms_io.health()
            result["Arm controller"] = ("ok" if alive else "fail", f"{status['state']}; heartbeat age {age:.1f}s")
            for side in SIDES:
                age = self.controller.hands_io.feedback_age(side)
                result[f"{side.capitalize()} hand feedback"] = (
                    "ok" if age is not None and age < 0.5 else "fail",
                    "live" if age is not None and age < 0.5 else "stale / disconnected")
        if self.glove:
            snap = self.glove.read()
            fresh = snap.frame is not None and time.monotonic() - snap.frame.received_at < 0.25
            result["Glove stream"] = ("fail" if snap.error else "ok" if fresh else "wait",
                f"{self.side}: " + (snap.error or ("streaming" if fresh else "waiting / stale")))
        return result

    def overview(self):
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [0, 0, 0.65]
            self.viewer.cam.distance = 2.8
            self.viewer.cam.azimuth, self.viewer.cam.elevation = 135, -15
            self.viewer.opt.geomgroup[1] = 1
            self.viewer.opt.geomgroup[2] = 0

    def edit_target(self, values):
        if self.controller.status()["state"] != "paused":
            self.input_notice.set("Pause before editing an end-effector target.")
            return False
        for variable, value in zip(self.values, values):
            variable.set(f"{value:.5f}")
        self.input_notice.set("Target edited — preview pending; no robot motion.")
        return True

    def keyboard(self, event):
        # Text entry keeps its normal arrows and letters. Space/Esc remain stop controls.
        if isinstance(event.widget, (tk.Entry, tk.Scale)):
            return
        key = event.keysym if event.keysym in KEY_AXES else event.keysym.lower()
        if key not in KEY_AXES:
            return
        try:
            values = jog_target([float(v.get()) for v in self.values], key, bool(event.state & 1))
            if self.edit_target(values):
                self.schedule_preview()
        except ValueError as exc:
            self.input_notice.set(str(exc))
        return "break"

    def drag(self, event, plane):
        event.widget.focus_set()
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
        try:
            values = drag_target([float(v.get()) for v in self.values], self.drag_origin,
                                 plane, event.x, event.y)
            self.edit_target(values)
        except ValueError as exc:
            self.input_notice.set(str(exc))

    def schedule_preview(self):
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
        def run():
            self.auto_job = None
            try:
                self.make_preview()
                self.input_notice.set("Playing checked preview. Execute remains a separate action.")
            except Exception as exc:
                self.input_notice.set(f"Preview rejected: {exc}")
        self.auto_job = self.root.after(200, run)

    def draw_targets(self):
        try:
            xyz = np.array([float(v.get()) for v in self.values[:3]])
            if not np.isfinite(xyz).all():
                return
        except ValueError:
            return
        delta = (xyz - self.drag_origin) * 400
        for plane, canvas in self.canvases.items():
            canvas.delete("all")
            canvas.create_line(130, 0, 130, 170, fill="#566477")
            canvas.create_line(0, 85, 260, 85, fill="#566477")
            x, y = ((130 - delta[1], 85 - delta[0]) if plane == "top"
                    else (130 + delta[0], 85 - delta[2]))
            canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#43d9c2", outline="white")
            canvas.create_text(8, 10, text="+X ↑ / +Y ←" if plane == "top" else "+Z ↑ / +X →",
                               fill="white", anchor="w")

    def action(self, callback):
        try:
            callback()
        except Exception as exc:
            messagebox.showerror("Action refused", str(exc))

    def invalidate(self, *args):
        self.preview, self.reviewed, self.play_started = None, False, None

    def display(self, q, fingers):
        for s in SIDES:
            self.data.qpos[self.arms[s].qpos_ids] = q[s]
            self.data.qpos[self.hands[s].qpos_ids] = fingers[s]
        mujoco.mj_forward(self.model, self.data)

    def current_pose(self):
        status = self.controller.status()
        with self.viewer.lock():
            self.display(status["measured"], status["hand_command"])
            xyz = self.data.xpos[self.ee].copy()
            rpy = Rotation.from_matrix(self.data.xmat[self.ee].reshape(3, 3)).as_euler("xyz", degrees=True)
        self.drag_origin = xyz.copy()
        for variable, value in zip(self.values, np.r_[xyz, rpy]):
            variable.set(f"{value:.5f}")

    def table_changed(self, *args):
        self.invalidate()
        self.table_confirmed.set(False)
        self.controller.request_pause()
        try:
            value = float(self.base_height.get())
            if np.isfinite(value) and value >= 0:
                self.checker.height = self.model_base_height + 0.762 - value
        except ValueError:
            pass

    def configure_table(self):
        base_text = self.base_height.get().strip()
        if self.hardware and (not base_text or not self.table_confirmed.get()):
            raise ValueError("Verify the robot base height against the real robot, then check the confirmation box")
        base_height = float(base_text) if base_text else None
        if base_height is not None and (not np.isfinite(base_height) or base_height < 0):
            raise ValueError("Base origin height must be finite and nonnegative")
        self.checker.height = None if base_height is None else self.model_base_height + 0.762 - base_height
        # Runtime finger guards use the same tabletop plane as the preview.
        with self.controller._lock:
            self.controller.guard_checker.height = self.checker.height

    def make_preview(self):
        self.invalidate()
        status = self.controller.status()
        if status["state"] != "paused":
            raise ValueError("Pause before previewing")
        self.configure_table()
        numbers = np.array([float(v.get()) for v in self.values])
        if not np.isfinite(numbers).all():
            raise ValueError("Enter finite coordinates")
        target = np.eye(4)
        target[:3, 3] = numbers[:3]
        target[:3, :3] = Rotation.from_euler("xyz", numbers[3:], degrees=True).as_matrix()
        with self.viewer.lock():
            self.display(status["command"], status["hand_command"])
            q = self.ik.solve(target, seed=status["command"][self.side])
            goal = {s: status["command"][s].copy() for s in SIDES}
            goal[self.side] = q
            self.display(goal, status["hand_command"])
            error = np.linalg.norm(self.data.xpos[self.ee] - target[:3, 3])
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ self.data.xmat[self.ee].reshape(3, 3)).magnitude()
        if error > 0.01 or rotation_error > np.deg2rad(5):
            raise ValueError(f"IK not reached: {error * 1000:.1f} mm, {np.rad2deg(rotation_error):.1f} degrees")
        if np.any(q < self.arms[self.side].lower) or np.any(q > self.arms[self.side].upper):
            raise ValueError("IK exceeds joint limits")
        self.preview = straight_path(join_arms(status["command"]), join_arms(goal),
                                     self.checker, status["hand_command"], self.side)
        self.play_started = time.monotonic()

    def execute(self):
        if self.preview is None or not self.reviewed:
            raise ValueError("Preview and watch the full playback first")
        if self.hardware and not messagebox.askyesno("Move physical robot", "Execute the displayed path? "
                "Confirm the entire path is clear of the table, people and external obstacles. Fingers hold during this move."):
            return
        self.controller.execute(self.preview)
        self.invalidate()

    def glove_mode(self):
        if self.glove is None:
            raise ValueError("Launch with --glove to connect the selected glove")
        self.configure_table()
        snapshot = self.glove.read()
        if snapshot.error or snapshot.frame is None or time.monotonic() - snapshot.frame.received_at > 0.25:
            raise ValueError(snapshot.error or "No fresh glove frames")
        status = self.controller.status()
        for side in SIDES:
            self.checker.set_fingers(side, status["hand_command"][side])
        if self.checker.clearance(join_arms(status["command"])).distance_m < 0.015:
            raise ValueError("Current pose is too close to the table or robot; resolve the clearance first")
        self.invalidate()
        self.controller.request_start()

    def pause(self):
        self.controller.request_pause()
        self.invalidate()

    def tick(self):
        if not self.viewer.is_running():
            self.root.destroy()
            return
        status = self.controller.status()
        q, fingers, label = status["measured"], status["hand_command"], "Live measured pose"
        if self.preview is not None:
            if status["state"] != "paused":
                self.invalidate()
            else:
                elapsed = time.monotonic() - self.play_started
                q, fingers = split_arms(self.preview.follower.at(elapsed)), self.preview.fingers
                self.reviewed = elapsed >= self.preview.follower.duration
                label = "PREVIEW complete — ready to execute" if self.reviewed else "PREVIEW playback (no robot motion)"
        with self.viewer.lock():
            self.display(q, fingers)
            self.viewer.user_scn.ngeom = 0
            if self.checker.height is not None:
                mujoco.mjv_initGeom(self.viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([1.0, 1.0, 0.005]), np.array([0.0, 0.0, self.checker.height - 0.005]),
                    np.eye(3).ravel(), np.array([0.55, 0.4, 0.25, 0.35], dtype=np.float32))
                self.viewer.user_scn.ngeom = 1
            try:
                xyz = np.array([float(v.get()) for v in self.values[:3]])
                if np.isfinite(xyz).all():
                    index = self.viewer.user_scn.ngeom
                    mujoco.mjv_initGeom(self.viewer.user_scn.geoms[index], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, 0.018), xyz, np.eye(3).ravel(),
                        np.array([0.1, 0.9, 0.75, 0.8], dtype=np.float32))
                    self.viewer.user_scn.ngeom += 1
            except ValueError:
                pass
        self.viewer.sync()
        self.draw_targets()
        glove_status = "not connected"
        if self.glove is not None:
            snap = self.glove.read()
            glove_status = snap.error or ("streaming" if snap.frame is not None and
                time.monotonic() - snap.frame.received_at < 0.25 else "waiting / stale")
        self.status.set(f"{status['state'].upper()} | {label}\n{self.side} glove: {glove_status}\n" +
                        (status["messages"][0] if status["messages"] else ""))
        self.root.after(30, self.tick)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=SIDES, default="right")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--glove", action="store_true")
    args = parser.parse_args(argv)
    if args.hardware and not args.glove:
        parser.error("Hardware requires --glove")
    if args.hardware and not all(os.environ.get(k) for k in ("ROBOT_NAME", "ZENOH_CONFIG")):
        parser.error("Use scripts/launch_vega_ee_preview.sh for hardware configuration")
    chooser = tk.Tk()
    chooser.title("Choose arm — Cartesian preview")
    choice = tk.StringVar(value=args.side)
    tk.Label(chooser, text="Choose one arm for this session. The other arm holds.").pack(padx=20, pady=10)
    for side in ("right", "left"):
        tk.Radiobutton(chooser, text=side.capitalize(), variable=choice, value=side).pack(anchor="w", padx=20)
    ComponentPanel(chooser).pack(padx=10, pady=10)
    accepted = []
    def select():
        accepted.append(choice.get())
        chooser.destroy()
    tk.Button(chooser, text="Open preview", command=select).pack(pady=10)
    chooser.mainloop()
    if not accepted:
        return 0
    args.side = accepted[0]
    if args.hardware or args.glove:
        other = subprocess.run(["pgrep", "-af", "python.*-m wuji_teleop"], capture_output=True, text=True)
        conflicts = [line for line in other.stdout.splitlines() if line.split()[0] != str(os.getpid())]
        if conflicts:
            raise RuntimeError("Stop the other teleop/simulation before connecting devices: " + "; ".join(conflicts))
    model, data, arms, hands = load_bimanual_scene()
    glove = arms_io = hands_io = controller = None
    try:
        if args.glove:
            glove = GloveSource(None, "", device_name=f"ee_preview_{args.side}",
                                address=f"192.168.1.{100 if args.side == 'left' else 101}:50001")
            if glove.side != args.side:
                raise ValueError("Glove handedness mismatch")
        arms_io = DexcontrolArms() if args.hardware else FakeArms(reset_pose(0.0))
        hands_io = WujiHands() if args.hardware else FakeHands()
        hands_io.discover()
        # Both hands are measured for collision geometry; only selected hand is enabled.
        for s in SIDES:
            hands_io.connect(s)
            hands_io.wait_position(s)
        hand_start = hands_io.positions()
        measured_arms = arms_io.positions()
        for side in SIDES:
            q = measured_arms[side]
            if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < arms[side].lower) or np.any(q > arms[side].upper):
                raise ValueError(f"{side} measured arm pose is outside the model limits")
        hands_io.enable(args.side, hand_start[args.side])
        sources = {s: glove if s == args.side and glove else EmptySource() for s in SIDES}
        retargeters = {s: WujiRetargeter(s) if s == args.side and glove else
                       _StaticRetargeter(hand_start[s]) for s in SIDES}
        controller = PreviewTeleop(arms_io, hands_io, HoldMapping(), sources, retargeters,
            TableChecker(ArmClearanceChecker(model, arms, hands)), None, reset_pose(),
            arm_lower={s: arms[s].lower for s in SIDES}, arm_upper={s: arms[s].upper for s in SIDES},
            hand_lower={s: hands[s].lower for s in SIDES}, hand_upper={s: hands[s].upper for s in SIDES},
            limits=TeleopLimits(tracking_fault=0.15))
        controller.start()
        with passive_viewer(model, data, lambda key: controller.request_pause() if key == 32
                            else controller.request_estop() if key == 256 else None) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.8]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 135, -15
            root = tk.Tk()
            PreviewGUI(root, viewer, model, data, arms, hands, controller, args.side, glove, args.hardware)
            root.mainloop()
    finally:
        if controller:
            controller.stop()
        for resource in (hands_io, arms_io, glove):
            if resource is not None:
                resource.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
