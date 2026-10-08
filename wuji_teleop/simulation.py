"""MuJoCo hand controller and standalone runner; no device control APIs."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np

from .retarget import TrackingController, WujiRetargeter
from .sources import HandSource, SourceSnapshot


DESCRIPTION_COMMIT = "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95"
MODEL_DIR = (Path(__file__).resolve().parents[1] / "third_party/wuji-description"
             / "hand2/hand2_beta2/body/mjcf")


def joint_names(side):
    prefix = {"left": "l", "right": "r"}[side]
    names = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
    for finger in ("index_finger", "middle_finger", "ring_finger", "pinky"):
        names.extend(f"{finger}_{joint}" for joint in ("mcp_flex", "mcp_abd", "pip", "dip"))
    return [f"{prefix}_{name}" for name in names]


class MujocoHand:
    """Resolve by name, so this also works inside a larger Vega1U model."""

    def __init__(self, model, data, side):
        self.model, self.data = model, data
        names = joint_names(side)
        prefix = {"left": "l", "right": "r"}[side]
        actuators = [f"{prefix}_{finger}J{j}" for finger in ("TH", "FF", "MF", "RF", "LF")
                     for j in range(4)]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuators]
        if min(joints + ids) < 0 or len(set(ids)) != 20:
            raise ValueError("Model must expose all 20 named Hand 2 joints and actuators")
        self.actuator_ids = np.array(ids)
        self.joint_ids = np.array(joints)
        self.qpos_ids = model.jnt_qposadr[joints]
        if not np.array_equal(model.actuator_trnid[ids, 0], joints):
            raise ValueError("Hand 2 actuator-to-joint mapping does not match firmware order")
        if not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Hand 2 joints must be scalar hinges")
        if not (np.all(model.jnt_limited[joints]) and np.all(model.actuator_ctrllimited[ids])):
            raise ValueError("Hand 2 must have joint and actuator control limits")
        # A position actuator has affine bias -kp*q -kv*qdot, unit transmission.
        if not (np.all(model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
                and np.allclose(model.actuator_biasprm[ids, 1], -model.actuator_gainprm[ids, 0])
                and np.allclose(model.actuator_gear[ids, 0], 1)):
            raise ValueError("Expected original Hand 2 position actuators")
        self.lower = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[ids, 0])
        self.upper = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[ids, 1])
        if np.any(self.lower >= self.upper):
            raise ValueError("Inconsistent Hand 2 joint/actuator limits")
        self.target = np.clip(np.zeros(20), self.lower, self.upper)

    def set_target(self, angles):
        values = np.asarray(angles, dtype=np.float64)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("Expected 20 finite target angles in radians")
        self.target = np.clip(values, self.lower, self.upper)
        self.data.ctrl[self.actuator_ids] = self.target

    def initialize(self):
        self.set_target(self.target)
        self.data.qpos[self.qpos_ids] = self.target
        self.data.qvel[self.model.jnt_dofadr[self.joint_ids]] = 0
        mujoco.mj_forward(self.model, self.data)


def load_hand(side):
    path = MODEL_DIR / f"{side}.xml"
    if not path.is_file():
        raise FileNotFoundError(
            "Hand models missing. Run: git submodule update --init third_party/wuji-description"
        )
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != (20, 20, 20):
        raise ValueError("Standalone Hand 2 must have a fixed palm and exactly 20 joints")
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError("Expected the pinned model's 2 ms timestep")
    model.vis.headlight.ambient[:] = 0.3
    model.vis.headlight.diffuse[:] = 0.8
    data = mujoco.MjData(model)
    hand = MujocoHand(model, data, side)
    hand.initialize()
    return model, data, hand


@dataclass
class RunReport:
    source: str
    side: str
    elapsed_seconds: float = 0.0
    simulation_seconds: float = 0.0
    received_frames: int = 0
    control_updates: int = 0
    input_hz: float = 0.0
    control_hz: float = 0.0
    frame_age_ms: float | None = None
    max_frame_age_ms: float = 0.0
    final_status: str = "waiting"
    mujoco_warnings: int = 0


@contextmanager
def passive_viewer(model, data, callback):
    from mujoco import viewer as mjviewer
    before = set(threading.enumerate())
    handle = mjviewer.launch_passive(model, data, key_callback=callback)
    # MuJoCo 3.15's close() only signals its daemon render thread. Python's
    # atexit glfw.terminate otherwise races that thread and segfaults on Linux.
    # The pinned viewer starts this named thread but exposes no public join().
    render_threads = [t for t in threading.enumerate()
                      if t not in before and t.name.endswith("(_launch_internal)")]
    try:
        yield handle
    finally:
        handle.close()
        for thread in render_threads:
            thread.join()


def run(source: HandSource, *, headless=False, duration=None, report_path=None,
        scene_loader=load_hand, title="Wuji Hand 2 — Space: pause/resume",
        camera_body=None, camera_distance=None, render_hz=60):
    model, data, hand = scene_loader(source.side)
    controller = TrackingController(WujiRetargeter(source.side), hand)
    keys = queue.SimpleQueue()
    context = nullcontext(None)
    if not headless:
        context = passive_viewer(model, data, keys.put)
    report = RunReport(type(source).__name__, source.side)
    snapshot = SourceSnapshot(None, 0, 0, None)
    started = time.monotonic()
    next_physics = next_control = next_render = started
    next_status = started + 2
    period = 1 / 120
    last_received = last_updates = 0
    last_status = started
    display_text = "Waiting for tracking"
    try:
        with context as viewer:
            if viewer:
                with viewer.lock():
                    viewer.opt.geomgroup[2] = 0  # Hide duplicate collision meshes.
                    viewer.opt.sitegroup[3] = 0
                    viewer.cam.lookat[:] = model.stat.center
                    viewer.cam.distance = model.stat.extent * 1.6
                    if camera_body is not None:
                        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, camera_body)
                        if body_id < 0:
                            raise ValueError(f"Camera body not found: {camera_body}")
                        viewer.cam.lookat[:] = data.xpos[body_id]
                    if camera_distance is not None:
                        viewer.cam.distance = camera_distance
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -15
            # Viewer creation can take hundreds of milliseconds; start the run's
            # clocks after it is ready, so startup is not counted as missed work.
            started = time.monotonic()
            next_physics = next_control = next_render = started
            last_status = started
            next_status = started + 2
            while duration is None or time.monotonic() - started < duration:
                if viewer and not viewer.is_running():
                    break
                while not keys.empty():
                    if keys.get() == 32:
                        controller.toggle_pause()
                now = time.monotonic()
                with viewer.lock() if viewer else nullcontext():
                    if now >= next_control:
                        snapshot = source.read()
                        controller.update(snapshot, time.monotonic())
                        if snapshot.frame is not None:
                            age = max(0, time.monotonic() - snapshot.frame.received_at) * 1000
                            report.frame_age_ms = age
                            report.max_frame_age_ms = max(report.max_frame_age_ms, age)
                        missed = max(1, int((now - next_control) / period) + 1)
                        next_control += missed * period
                    # Catch up short scheduling delays; cap work after a long OS stall.
                    now = time.monotonic()
                    steps = min(25, max(0, int((now - next_physics) / model.opt.timestep) + 1))
                    if steps:
                        # Restore targets if a viewer slider changed data.ctrl.
                        hand.set_target(hand.target)
                        mujoco.mj_step(model, data, nstep=steps)
                        next_physics += steps * model.opt.timestep
                        if now - next_physics > 0.050:
                            next_physics = now
                if now >= next_status:
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                        raise RuntimeError("Non-finite MuJoCo state")
                    report.mujoco_warnings = int(np.sum(data.warning.number))
                    if report.mujoco_warnings:
                        raise RuntimeError(f"MuJoCo raised {report.mujoco_warnings} numerical warnings")
                    age = None if snapshot.frame is None else max(0, now - snapshot.frame.received_at) * 1000
                    report.frame_age_ms = age
                    report.max_frame_age_ms = max(report.max_frame_age_ms, age or 0)
                    dt = max(now - last_status, 1e-6)
                    input_hz = (snapshot.received - last_received) / dt
                    control_hz = (controller.updates - last_updates) / dt
                    age_text = "n/a" if age is None else f"{age:.1f} ms"
                    confidence = snapshot.frame.confidence if snapshot.frame else None
                    confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
                    display_text = (f"{source.side} | {controller.status} | input {input_hz:.1f} Hz | "
                                    f"control {control_hz:.1f} Hz | age {age_text} | confidence {confidence_text}")
                    print(display_text + (f" | {controller.detail}" if controller.detail else ""), flush=True)
                    last_received, last_updates = snapshot.received, controller.updates
                    last_status = now
                    next_status = now + 2
                if viewer and now >= next_render:
                    viewer.set_texts((None, None, title, display_text))
                    viewer.sync()
                    next_render = now + 1 / render_hz
                sleep_for = min(next_physics, next_control) - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        report.elapsed_seconds = time.monotonic() - started
        report.simulation_seconds = float(data.time)
        report.received_frames = snapshot.received
        report.control_updates = controller.updates
        report.input_hz = snapshot.received / max(report.elapsed_seconds, 1e-6)
        report.control_hz = controller.updates / max(report.elapsed_seconds, 1e-6)
        report.final_status = controller.status
        if snapshot.frame is not None:
            report.frame_age_ms = max(0, time.monotonic() - snapshot.frame.received_at) * 1000
            report.max_frame_age_ms = max(report.max_frame_age_ms, report.frame_age_ms)
        report.mujoco_warnings = int(np.sum(data.warning.number))
        serialized = json.dumps(asdict(report), indent=2)
        print("Run report:\n" + serialized, flush=True)
        if report_path:
            Path(report_path).write_text(serialized + "\n")
    return report
