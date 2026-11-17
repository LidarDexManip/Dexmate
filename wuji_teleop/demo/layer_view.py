"""Camera-view inspector of a synced live-stereo session, with layers switched on and off.

``scripts/view_demo_layers.sh SESSION`` (``.venv-vision``). Per camera frame it
shows the left image and, beside it (never on top), the stereo depth. Each layer
is a check box (or its number key) and is drawn on both panels:

1. **Trackers**: each wrist tracker's position and axes (x red, y green, z blue, 5 cm).
2. **Wrist**: the wrist joint centre, joined to its tracker by the mounting offset
   (yellow). After a depth fit a grey ring marks where it was before.
3. **Hand orientation**: the hand's axes at the wrist (palm IMU through the wrist
   calibration, then the depth fit; thick arrows, 4 cm).
4. **Depth**: the depth panel, near bright and far dark (``--near``/``--far``, metres; outside or no depth: dark blue).
5. **Finger skeleton**: the glove's 21 keypoints, placed by the wrist pose.
6. **Hand model**: the fitted hand surface of the chosen model.

Hand models are the ``processed/<name>_<side>.npz`` files with ``vertices`` and
``faces`` (``mano`` from ``fit_mano.sh``, other models from experiments). The
wrist, orientation and skeleton follow the chosen model's depth correction
(``depth_R``/``depth_t``); "none" shows them as synced, before any depth fit.

Keys: Space play/pause, Left/Right one frame, Shift+Left/Right one second,
1-6 the layers, L/R the left/right hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .depth_align import corrected_keypoints, read_depth
from .mano_fit import OVERLAY_COLORS, _BONES21, draw_mesh

SIDES = ("left", "right")
LAYERS = (("trackers", "1 Trackers"), ("wrist", "2 Wrist"), ("orientation", "3 Hand orientation"),
          ("depth", "4 Depth"), ("skeleton", "5 Finger skeleton"), ("model", "6 Hand model"))
AXIS_COLORS = ((60, 60, 255), (60, 220, 60), (255, 120, 40))  # BGR x, y, z
GAP_PX = 8


def find_models(processed):
    """{name: {side: path}} for every processed/<name>_<side>.npz that holds a surface."""
    models = {}
    for path in sorted(Path(processed).glob("*_*.npz")):
        name, _, side = path.stem.rpartition("_")
        if side not in SIDES or not name:
            continue
        with np.load(path) as d:
            if "vertices" in d.files and "faces" in d.files:
                models.setdefault(name, {})[side] = path
    return models


def load_model(paths):
    out = {}
    for side, path in paths.items():
        d = dict(np.load(path))
        out[side] = {k: d[k] for k in ("vertices", "faces", "valid", "joints21", "depth_R", "depth_t") if k in d}
    return out


class Scene:
    """Everything one frame needs, in the world frame; drawing is per panel scale."""

    def __init__(self, session, near=0.2, far=1.0):
        self.root = Path(session).expanduser().resolve()
        self.z = dict(np.load(self.root / "processed" / "synced.npz"))
        self.K = np.asarray(self.z["K"], float)
        self.n = len(self.z["frame"])
        self.t = self.z["t_capture"] - self.z["t_capture"][0]
        baseline = float(self.z["baseline_m"])
        self.fb = float(self.K[0, 0] * baseline) if np.isfinite(baseline) else None
        self.disparity_scale = float(self.z.get("disparity_scale", 64.0))
        self.near, self.far = near, far
        self.models = find_models(self.root / "processed")
        self._loaded = {}

    def model(self, name):
        if name not in self.models:
            return {}
        if name not in self._loaded:
            self._loaded[name] = load_model(self.models[name])
        return self._loaded[name]

    def image(self, i):
        import cv2

        return cv2.imread(str(self.root / "stereo" / "left" / f"{int(self.z['frame'][i]):06d}.jpg"))

    def depth_image(self, i, shape):
        import cv2

        Z = read_depth(self.root, self.z["frame"][i], self.fb, self.disparity_scale) if self.fb else None
        if Z is None:
            img = np.zeros((*shape[:2], 3), np.uint8)
            cv2.putText(img, "no depth", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return img
        v = np.clip((Z - self.near) / (self.far - self.near), 0, 1)
        img = cv2.cvtColor((235 - 185 * v).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # near bright, far dark
        img[(Z <= 0) | (Z > self.far) | (Z < self.near)] = (60, 20, 20)  # no depth or out of range: dark blue
        return img

    def hand(self, i, side, model):
        """World tracker pose, wrist before/after the depth fit, hand rotation, keypoints, mesh."""
        z = self.z
        out = {"tracker": z[f"tracker_{side}"][i] if z[f"tracker_valid_{side}"][i] else None}
        if not z[f"hand_valid_{side}"][i]:
            return out
        W = z[f"wrist_{side}"][i]
        R_fix, t_fix = np.eye(3), np.zeros(3)
        fit = model.get(side)
        if fit is not None and "depth_R" in fit and np.isfinite(fit["depth_t"][i]).all():
            R_fix, t_fix = np.asarray(fit["depth_R"][i], float), np.asarray(fit["depth_t"][i], float)
        out["wrist_before"] = W[:3, 3]
        out["wrist"] = W[:3, 3] + t_fix
        out["R_hand"] = R_fix @ W[:3, :3]
        kp = z[f"keypoints_{side}"][i]
        out["keypoints"] = (R_fix @ (kp - W[:3, 3]).T).T + W[:3, 3] + t_fix
        if fit is not None and fit["valid"][i]:
            out["mesh"] = (fit["vertices"][i], fit["faces"])
        return out


def _uv(P_world, Tcw, K):
    P = np.atleast_2d(P_world) @ Tcw[:3, :3].T + Tcw[:3, 3]
    if (P[:, 2] <= 0.05).any() or not np.isfinite(P).all():
        return None
    uv = (P @ K.T)[:, :2] / P[:, 2:3]
    return np.round(uv).astype(int)


def draw_layers(img, scene, i, layers, sides, model_name, scale):
    """Draw the switched-on layers into one panel (already resized by ``scale``)."""
    import cv2

    T = scene.z["T_world_camera"][i]
    if not np.isfinite(T).all():
        return
    Tcw = np.linalg.inv(T)
    K = scene.K.copy()
    K[:2] *= scale
    model = scene.model(model_name)
    lw = max(1, int(round(scale)))
    for side in sides:
        h = scene.hand(i, side, model)
        col = OVERLAY_COLORS[side]
        if layers["model"] and "mesh" in h:
            draw_mesh(img, h["mesh"][0], h["mesh"][1], Tcw, K, col, alpha=0.5)
        if layers["skeleton"] and "keypoints" in h:
            uv = _uv(h["keypoints"], Tcw, K)
            if uv is not None:
                for a, b in _BONES21:
                    cv2.line(img, tuple(uv[a]), tuple(uv[b]), (255, 255, 255), lw, cv2.LINE_AA)
                for p in uv:
                    cv2.circle(img, tuple(p), lw + 1, col, -1, cv2.LINE_AA)
        if layers["trackers"] and h.get("tracker") is not None:
            Tt = h["tracker"]
            pts = np.vstack([Tt[:3, 3], Tt[:3, 3] + 0.05 * Tt[:3, :3].T])
            uv = _uv(pts, Tcw, K)
            if uv is not None:
                for k in range(3):
                    cv2.line(img, tuple(uv[0]), tuple(uv[k + 1]), AXIS_COLORS[k], lw, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[0]), 3 * lw, (255, 255, 255), lw, cv2.LINE_AA)
        if layers["wrist"] and "wrist" in h:
            if not np.allclose(h["wrist"], h["wrist_before"]):
                uv = _uv(h["wrist_before"], Tcw, K)
                if uv is not None:
                    cv2.circle(img, tuple(uv[0]), 4 * lw, (150, 150, 150), lw, cv2.LINE_AA)
            uv = _uv(h["wrist"], Tcw, K)
            if uv is not None:
                if h.get("tracker") is not None:
                    ut = _uv(h["tracker"][:3, 3], Tcw, K)
                    if ut is not None:
                        cv2.line(img, tuple(ut[0]), tuple(uv[0]), (0, 230, 255), lw, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[0]), 4 * lw, (0, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(img, tuple(uv[0]), 4 * lw, (0, 0, 0), 1, cv2.LINE_AA)
        if layers["orientation"] and "R_hand" in h:
            pts = np.vstack([h["wrist"], h["wrist"] + 0.04 * h["R_hand"].T])
            uv = _uv(pts, Tcw, K)
            if uv is not None:
                for k in range(3):
                    cv2.arrowedLine(img, tuple(uv[0]), tuple(uv[k + 1]), AXIS_COLORS[k], 2 * lw,
                                    cv2.LINE_AA, tipLength=0.25)


def compose(scene, i, layers, sides, model_name, width, height):
    """The panels for frame i, side by side and scaled to fit width x height without overlap."""
    import cv2

    rgb = scene.image(i)
    if rgb is None:
        rgb = np.zeros((int(scene.z["image_size"][1]), int(scene.z["image_size"][0]), 3), np.uint8)
    h0, w0 = rgb.shape[:2]
    panels = [rgb]
    if layers["depth"]:
        panels.append(scene.depth_image(i, rgb.shape))
    n = len(panels)
    scale = max(min((width - GAP_PX * (n - 1)) / (n * w0), height / h0), 0.1)
    pw, ph = int(w0 * scale), int(h0 * scale)
    canvas = np.full((max(height, ph), max(width, n * pw + GAP_PX * (n - 1)), 3), 24, np.uint8)
    x0 = (canvas.shape[1] - (n * pw + GAP_PX * (n - 1))) // 2
    y0 = (canvas.shape[0] - ph) // 2
    for k, p in enumerate(panels):
        p = cv2.resize(p, (pw, ph), interpolation=cv2.INTER_LINEAR)
        draw_layers(p, scene, i, layers, sides, model_name, scale)
        x = x0 + k * (pw + GAP_PX)
        canvas[y0:y0 + ph, x:x + pw] = p
    return canvas


def run(args):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    scene = Scene(args.session, near=args.near, far=args.far)
    root = tk.Tk()
    root.title(f"Demo layers: {scene.root.name}")
    root.geometry(args.geometry)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")  # check boxes with a visible tick
    if not args.no_topmost:
        root.attributes("-topmost", True)

    bar = ttk.Frame(root, padding=4)
    bar.pack(side="top", fill="x")
    layer_vars = {key: tk.BooleanVar(value=key not in args.off) for key, _ in LAYERS}
    for key, label in LAYERS:
        ttk.Checkbutton(bar, text=label, variable=layer_vars[key], command=lambda: redraw()).pack(side="left", padx=4)
    ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
    side_vars = {s: tk.BooleanVar(value=True) for s in SIDES}
    for s in SIDES:
        ttk.Checkbutton(bar, text=s.capitalize(), variable=side_vars[s], command=lambda: redraw()).pack(side="left")
    ttk.Label(bar, text="  Model:").pack(side="left")
    names = list(scene.models) + ["none"]
    model_var = tk.StringVar(value=args.model if args.model in names else names[0])
    box = ttk.Combobox(bar, textvariable=model_var, values=names, width=14, state="readonly")
    box.pack(side="left")
    box.bind("<<ComboboxSelected>>", lambda _e: (root.focus_set(), redraw()))

    # The bottom rows are packed before the image area so it never pushes them out of the window.
    legend = ttk.Label(root, padding=(6, 0, 6, 4), foreground="#555",
                       text="Axes: x red, y green, z blue.  Tracker: thin axes + white ring.  Wrist: yellow dot, "
                            "yellow line to its tracker, grey ring before the depth fit.  Hand orientation: "
                            "thick arrows.  Left hand blue, right hand orange.")
    legend.pack(side="bottom", fill="x")
    bottom = ttk.Frame(root, padding=4)
    bottom.pack(side="bottom", fill="x")
    # A canvas does not grow to its image (a label would, and resize itself in a loop).
    canvas = tk.Canvas(root, background="#181818", highlightthickness=0, width=200, height=150)
    canvas.pack(side="top", fill="both", expand=True)
    image_item = canvas.create_image(0, 0, anchor="nw")
    state = {"i": max(0, min(args.start, scene.n - 1)), "playing": False, "photo": None}
    play_btn = ttk.Button(bottom, text="Play", width=6)
    play_btn.pack(side="left")
    slider = ttk.Scale(bottom, from_=0, to=scene.n - 1, orient="horizontal",
                       command=lambda v: goto(int(float(v)), from_slider=True))
    slider.pack(side="left", fill="x", expand=True, padx=8)
    info = ttk.Label(bottom, width=46, anchor="w")
    info.pack(side="left")

    def redraw():
        i = state["i"]
        w, h = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 150)
        layers = {k: v.get() for k, v in layer_vars.items()}
        sides = [s for s in SIDES if side_vars[s].get()]
        img = compose(scene, i, layers, sides, model_var.get(), w, h)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(img[:, :, ::-1]))
        canvas.itemconfigure(image_item, image=state["photo"])
        info.configure(text=f"frame {i} / {scene.n - 1}   stereo #{int(scene.z['frame'][i])}   t {scene.t[i]:.2f} s")

    def goto(i, from_slider=False):
        state["i"] = int(np.clip(i, 0, scene.n - 1))
        if not from_slider:
            slider.set(state["i"])
        redraw()

    period_ms = int(round(1000 * float(np.median(np.diff(scene.z["t_capture"])))))

    def tick():
        if not state["playing"]:
            return
        if state["i"] >= scene.n - 1:
            toggle_play()
            return
        goto(state["i"] + 1)
        root.after(period_ms, tick)

    def toggle_play(_e=None):
        state["playing"] = not state["playing"]
        play_btn.configure(text="Pause" if state["playing"] else "Play")
        if state["playing"]:
            tick()

    play_btn.configure(command=toggle_play)
    fps = max(1, int(round(1000 / max(period_ms, 1))))
    root.bind("<space>", toggle_play)
    root.bind("<Left>", lambda _e: goto(state["i"] - 1))
    root.bind("<Right>", lambda _e: goto(state["i"] + 1))
    root.bind("<Shift-Left>", lambda _e: goto(state["i"] - fps))
    root.bind("<Shift-Right>", lambda _e: goto(state["i"] + fps))
    for n, (key, _) in enumerate(LAYERS, start=1):
        root.bind(str(n), lambda _e, k=key: (layer_vars[k].set(not layer_vars[k].get()), redraw()))
    root.bind("l", lambda _e: (side_vars["left"].set(not side_vars["left"].get()), redraw()))
    root.bind("r", lambda _e: (side_vars["right"].set(not side_vars["right"].get()), redraw()))
    pending = {"id": None}

    def on_resize(_e):
        if pending["id"]:
            root.after_cancel(pending["id"])
        pending["id"] = root.after(60, redraw)

    canvas.bind("<Configure>", on_resize)
    root.after(100, lambda: goto(state["i"]))
    root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session")
    parser.add_argument("--model", default="mano", help="hand model shown first (processed/<model>_<side>.npz)")
    parser.add_argument("--start", type=int, default=0, help="first frame (index into synced.npz)")
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=1.0)
    parser.add_argument("--off", nargs="*", default=[], choices=[k for k, _ in LAYERS],
                        help="layers switched off at start")
    parser.add_argument("--geometry", default="1500x700")
    parser.add_argument("--no-topmost", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
