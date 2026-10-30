"""Read-only component status and Cartesian target input helpers."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import tkinter as tk

import numpy as np

from .connection_monitor import ping, route_device

ENDPOINTS = {
    "Jetson": "192.168.50.20", "Robot SoC": "192.168.50.21",
    "Left glove": "192.168.1.100", "Right glove": "192.168.1.101",
    "Left Wuji hand": "192.168.77.110", "Right Wuji hand": "192.168.77.111",
}
COLORS = {"ok": "#22863a", "wait": "#b7791f", "fail": "#b3261e", "off": "#666666"}
KEY_AXES = {"Up": (0, 1), "Down": (0, -1), "Left": (1, 1), "Right": (1, -1),
            "Prior": (2, 1), "Next": (2, -1), "i": (3, 1), "k": (3, -1),
            "j": (4, 1), "l": (4, -1), "u": (5, 1), "o": (5, -1)}


def jog_target(values, key, fine=False):
    """5 mm / 2 degree steps, or one fifth that with Shift held."""
    result = np.asarray(values, dtype=float).copy()
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Enter six finite target coordinates")
    axis, direction = KEY_AXES[key]
    result[axis] += direction * (0.005 if axis < 3 else 2.0) * (0.2 if fine else 1.0)
    return result


def drag_target(values, origin, plane, x, y, width=260, height=170, scale=400):
    """Top view: screen right is -Y, up is +X. Side: right +X, up +Z."""
    result = np.asarray(values, dtype=float).copy()
    a, b = (float(x) - width / 2) / scale, (height / 2 - float(y)) / scale
    if plane == "top":
        result[0], result[1] = origin[0] + b, origin[1] - a
    elif plane == "side":
        result[0], result[2] = origin[0] + a, origin[2] + b
    else:
        raise ValueError("Unknown drag plane")
    return result


class ComponentPanel(tk.LabelFrame):
    """Pings only: reachability is never reported as a working SDK stream."""
    def __init__(self, parent, live_status=None):
        super().__init__(parent, text="Components — read-only checks", padx=8, pady=6)
        self.live_status = live_status
        self.results = {}
        self.stop = threading.Event()
        self.rows = {}
        for name in (*ENDPOINTS, "Arm controller", "Left hand feedback", "Right hand feedback",
                     "Glove stream", "Pico / trackers"):
            row = tk.Label(self, text=f"● {name}: checking…", anchor="w", justify="left", wraplength=340)
            row.pack(fill="x", pady=2)
            self.rows[name] = row
        self.refresh_job = None
        self.bind("<Destroy>", self.destroyed)
        threading.Thread(target=self.poll, daemon=True, name="ee-network-status").start()
        self.refresh_job = self.after(100, self.refresh)

    def destroyed(self, event):
        if event.widget is self:
            self.stop.set()
            if self.refresh_job is not None:
                self.after_cancel(self.refresh_job)
                self.refresh_job = None

    def poll(self):
        def check(item):
            name, ip = item
            alive, route = ping(ip), route_device(ip)
            return name, ("ok" if alive else "fail", f"{ip} {'reachable' if alive else 'no reply'} ({route or 'no route'})")
        with ThreadPoolExecutor(max_workers=6) as pool:
            while not self.stop.is_set():
                try:
                    self.results = dict(pool.map(check, ENDPOINTS.items()))
                except Exception as exc:
                    self.results = {name: ("wait", f"check failed: {exc}") for name in ENDPOINTS}
                self.stop.wait(3)

    def refresh(self):
        statuses = dict(self.results)
        statuses.update({"Arm controller": ("off", "not connected"),
                         "Left hand feedback": ("off", "not connected"),
                         "Right hand feedback": ("off", "not connected"),
                         "Glove stream": ("off", "not connected"),
                         "Pico / trackers": ("off", "not used in this workflow")})
        if self.live_status:
            try:
                statuses.update(self.live_status())
            except Exception as exc:
                statuses["Arm controller"] = ("fail", str(exc))
        for name, (level, text) in statuses.items():
            self.rows[name].config(text=f"● {name}: {text}", fg=COLORS[level])
        if not self.stop.is_set():
            self.refresh_job = self.after(400, self.refresh)
