"""3D viewer of a synced live-stereo session: MANO hands (or glove skeletons) in the world.

``process_human_demo.sh view SESSION`` opens it for a session recorded with
``record.py --stereo`` (the headset-recording sessions keep ``viewer``). It
shows, per camera frame:

- each hand's MANO mesh from ``processed/mano_<side>.npz`` (``fit_mano.sh``),
  else the 21-keypoint skeleton from ``synced.npz``;
- the wrist trackers;
- the head camera frustum with the left image;
- the depth point cloud (the scene and held objects, e.g. a pipette), with a
  baseline (``--baseline-m`` here or in ``sync``) or the f*B that ``fit_mano``
  estimated from the trackers.

A hand that never enters the camera image is hidden at first ("Hands out of
view"): it is placed by its tracker and wrist calibration alone, unchecked.
With a depth alignment (``fit_mano --depth-align``), the glove keypoints are
drawn moved with their MANO hand; the trackers stay where they were measured.

``--video`` renders the same scene to ``processed/hands_view.mp4`` instead: a
headless Chrome is the viser client, each frame is rendered from a fixed 3D
viewpoint (``--video-azimuth``/``--video-elevation``/``--video-distance``; the
frustum without its image) and placed beside the left camera image with the
MANO hands drawn in.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .depth_align import corrected_keypoints, in_view
from .viewer import BONES, SIDE_COLORS, _wxyz

SIDES = ("left", "right")
OVERLAY_BGR = {"left": (255, 150, 40), "right": (40, 150, 255)}


def _cloud(session_root, frame, K, baseline, T_world_camera, stride):
    import cv2

    root = Path(session_root)
    png = cv2.imread(str(root / "stereo" / "disparity" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(root / "stereo" / "left" / f"{frame:06d}.jpg"))
    if png is None or img is None:
        return None, None
    d = png[::stride, ::stride].astype(np.float32) / 64.0
    v, u = np.mgrid[0:png.shape[0]:stride, 0:png.shape[1]:stride]
    ok = png[::stride, ::stride] > 0
    z = K[0, 0] * baseline / d[ok]
    keep = (z > 0.1) & (z < 2.5)
    z = z[keep]
    uu, vv = u[ok][keep], v[ok][keep]
    pts = np.column_stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z])
    world = pts @ T_world_camera[:3, :3].T + T_world_camera[:3, 3]
    return world.astype(np.float32), img[vv, uu][:, ::-1]


class HandsScene:
    """The session's data and the viser scene for one frame at a time."""

    def __init__(self, session, server, baseline_m=None, initial_frame=None):
        root = session.root
        self.root, self.server = root, server
        z = dict(np.load(session.out / "synced.npz"))
        self.z, self.n, self.K = z, len(z["frame"]), z["K"]
        self.W, self.H = (int(v) for v in z["image_size"])
        baseline = baseline_m or (float(z["baseline_m"]) if np.isfinite(z["baseline_m"]) else None)
        report = session.out / "mano_report.json"
        if baseline is None and report.exists():
            fb = (json.loads(report.read_text()).get("depth_fb") or {}).get("fb")
            baseline = fb / float(self.K[0, 0]) if fb else None
        self.baseline = baseline
        self.mano = {}
        for side in SIDES:
            path = session.out / f"mano_{side}.npz"
            if path.exists():
                self.mano[side] = dict(np.load(path))
                if f"keypoints_{side}" in z:
                    z[f"keypoints_{side}"] = corrected_keypoints(z[f"keypoints_{side}"], self.mano[side])
        self.sides = [s for s in SIDES if s in self.mano or f"keypoints_{s}" in z]
        self.seen = {s: bool((in_view(z, self.mano[s]) > 0.2).any()) if s in self.mano else True
                     for s in self.sides}
        self.t_rel = z["t_capture"] - z["t_capture"][0]

        server.scene.set_up_direction("+z")
        gui = server.gui
        self.frame = gui.add_slider("Frame", min=0, max=self.n - 1, step=1, initial_value=int(initial_frame or 0))
        self.play = gui.add_checkbox("Play", False)
        self.speed = gui.add_dropdown("Speed", options=["0.25x", "0.5x", "1x", "2x"], initial_value="1x")
        with gui.add_folder("Layers"):
            self.show_mano = gui.add_checkbox("MANO hands", True, visible=bool(self.mano))
            self.show_skel = gui.add_checkbox("Glove keypoints", not self.mano)
            self.show_unseen = gui.add_checkbox("Hands out of view", False, visible=not all(self.seen.values()))
            self.show_trackers = gui.add_checkbox("Wrist trackers", True)
            self.show_camera = gui.add_checkbox("Camera image", True)
            self.show_cloud = gui.add_checkbox("Depth point cloud", baseline is not None, visible=baseline is not None)
            self.stride = gui.add_slider("Point cloud stride", min=1, max=8, step=1, initial_value=3,
                                         visible=baseline is not None)
        self.info = gui.add_markdown("")
        self.dirty = True
        for toggle in (self.show_mano, self.show_skel, self.show_unseen, self.show_trackers, self.show_camera,
                       self.show_cloud, self.stride):
            toggle.on_update(lambda _: setattr(self, "dirty", True))
        self.handles = {}
        self.cloud = server.scene.add_point_cloud("/cloud", np.zeros((1, 3), np.float32), (0, 0, 0),
                                                  point_size=0.002, visible=False)
        self.fov = 2 * np.arctan(self.H / 2 / self.K[1, 1])

    def _put(self, name, factory):
        if name in self.handles:
            self.handles[name].remove()
        self.handles[name] = factory()

    def visible_side(self, side):
        return self.seen[side] or self.show_unseen.value

    def center(self):
        """Median wrist position of the hands shown at first (for the camera)."""
        pts = [(self.mano[s]["joints21"] if s in self.mano else self.z[f"keypoints_{s}"])[:, 0]
               for s in self.sides if self.seen[s]] or [np.zeros((1, 3))]
        return np.nanmedian(np.concatenate(pts), axis=0)

    def update(self, i):
        import cv2

        z, scene = self.z, self.server.scene
        lines = [f"**Frame {int(z['frame'][i])}** ({i + 1}/{self.n}), t = {self.t_rel[i]:.2f} s"]
        for side in self.sides:
            color = SIDE_COLORS[side]
            shown = self.visible_side(side)
            m = self.mano.get(side)
            ok_mano = m is not None and bool(m["valid"][i]) and np.isfinite(m["vertices"][i]).all()
            self._put(f"/hands/{side}/mano", lambda: scene.add_mesh_simple(
                f"/hands/{side}/mano", m["vertices"][i] if ok_mano else np.zeros((3, 3)),
                m["faces"] if ok_mano else np.array([[0, 1, 2]]), color=color, opacity=0.9,
                visible=ok_mano and shown and self.show_mano.value))
            kp = z.get(f"keypoints_{side}")
            ok_kp = kp is not None and bool(z[f"hand_valid_{side}"][i]) and np.isfinite(kp[i]).all()
            seg = np.array([[kp[i][a], kp[i][b]] for a, b in BONES]) if ok_kp else np.zeros((1, 2, 3))
            self._put(f"/hands/{side}/keypoints", lambda: scene.add_line_segments(
                f"/hands/{side}/keypoints", seg, color, thickness=0.003,
                visible=ok_kp and shown and self.show_skel.value))
            T = z.get(f"tracker_{side}")
            if T is not None and np.isfinite(T[i]).all():
                self._put(f"/trackers/{side}", lambda: scene.add_frame(
                    f"/trackers/{side}", axes_length=0.04, axes_radius=0.003, wxyz=_wxyz(T[i][:3, :3]),
                    position=T[i][:3, 3], visible=shown and self.show_trackers.value))
            status = "MANO" if ok_mano else ("keypoints" if ok_kp else "missing")
            if ok_mano:
                status += f", fit error median {np.nanmedian(m['error_mm'][i]):.1f} mm"
            if not self.seen[side]:
                status += " (never in the camera view: placed by its tracker alone)"
            lines.append(f"- {side} hand: {status}")
        Twc = z["T_world_camera"][i]
        image = cv2.imread(str(self.root / "stereo" / "left" / f"{int(z['frame'][i]):06d}.jpg"))
        if np.isfinite(Twc).all():
            self._put("/camera", lambda: scene.add_camera_frustum(
                "/camera", fov=self.fov, aspect=self.W / self.H, scale=0.06, thickness=0.0015,
                image=image[:, :, ::-1] if (image is not None and self.show_camera.value) else None,
                format="jpeg", jpeg_quality=70, wxyz=_wxyz(Twc[:3, :3]), position=Twc[:3, 3]))
        if self.baseline and self.show_cloud.value and np.isfinite(Twc).all():
            pts, colors = _cloud(self.root, int(z["frame"][i]), self.K, self.baseline, Twc, int(self.stride.value))
            if pts is not None and len(pts):
                self.cloud.points, self.cloud.colors = pts, colors
                self.cloud.visible = True
        else:
            self.cloud.visible = False
        self.info.content = "\n".join(lines)
        return image

    def overlay(self, i, image):
        """The left camera image with the shown MANO hands drawn in (BGR)."""
        import cv2

        out = image.copy()
        Twc = self.z["T_world_camera"][i]
        if not np.isfinite(Twc).all():
            return out
        Tcw = np.linalg.inv(Twc)
        layer = out.copy()
        for side, m in self.mano.items():
            if not (self.visible_side(side) and m["valid"][i] and np.isfinite(m["vertices"][i]).all()):
                continue
            P = m["vertices"][i] @ Tcw[:3, :3].T + Tcw[:3, 3]
            if (P[:, 2] < 0.05).any():
                continue
            uv = np.round((P @ self.K.T)[:, :2] / P[:, 2:3]).astype(np.int32)
            for F in m["faces"][np.argsort(-P[m["faces"]].mean(1)[:, 2])]:
                cv2.fillConvexPoly(layer, uv[F], OVERLAY_BGR[side], cv2.LINE_AA)
        return cv2.addWeighted(layer, 0.45, out, 0.55, 0)


def serve(session, port=8120, initial_frame=None, baseline_m=None):
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=port)
    scene = HandsScene(session, server, baseline_m, initial_frame)
    center = scene.center()
    if np.isfinite(center).all():
        server.initial_camera.look_at = tuple(center)
        server.initial_camera.position = tuple(center + np.array([-0.45, 0.3, 0.4]))
        server.initial_camera.up = (0.0, 0.0, 1.0)
    print(f"Hands viewer: http://localhost:{port}" + ("" if scene.mano else "  (no MANO fit yet: glove keypoints)"),
          flush=True)
    previous = None
    dt = float(np.median(np.diff(scene.t_rel))) if scene.n > 1 else 1 / 30
    while True:
        tick = time.monotonic()
        i = int(scene.frame.value)
        if i != previous or scene.dirty:
            scene.dirty = False
            scene.update(i)
            previous = i
        if scene.play.value:
            scene.frame.value = (i + 1) % scene.n
            time.sleep(max(0.0, dt / float(scene.speed.value.rstrip("x")) - (time.monotonic() - tick)))
        else:
            time.sleep(0.03)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """wxyz of a viser (OpenCV: x right, y down, z forward) camera at ``eye`` looking at ``target``."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    xyzw = Rotation.from_matrix(np.column_stack([right, down, f])).as_quat()
    return np.array([xyzw[3], *xyzw[:3]])


def video_eye(center, camera_position, azimuth_deg, elevation_deg, distance):
    """A viewpoint turned ``azimuth_deg`` about the vertical from the head camera's side of the hands."""
    away = np.asarray(camera_position, float) - center
    heading = np.arctan2(away[1], away[0]) if np.hypot(away[0], away[1]) > 0.05 else np.pi
    a, e = heading + np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    return center + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _chrome():
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return name
    raise SystemExit("--video needs google-chrome or chromium as the headless viser client")


def record_video(session, out=None, *, port=8121, baseline_m=None, size=(960, 720), azimuth_deg=30.0,
                 elevation_deg=55.0, distance=0.65, step=1, log=print):
    """Render every ``step``-th frame from a fixed 3D view beside the camera image: ``hands_view.mp4``."""
    import cv2
    import viser

    out = Path(out) if out else session.out / "hands_view.mp4"
    server = viser.ViserServer(host="127.0.0.1", port=port)
    scene = HandsScene(session, server, baseline_m)
    scene.show_skel.value = False
    scene.show_camera.value = False  # the camera image is the right half; on the frustum it hides the hands
    width, height = size
    profile = tempfile.mkdtemp(prefix="hands-view-chrome-")
    chrome = subprocess.Popen(
        [_chrome(), "--headless=new", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         f"--user-data-dir={profile}", f"--window-size={width + 400},{height + 200}", "--mute-audio",
         f"http://127.0.0.1:{port}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    writer = None
    try:
        deadline = time.monotonic() + 60
        while not server.get_clients():
            if time.monotonic() > deadline or chrome.poll() is not None:
                raise RuntimeError("the headless Chrome client did not connect")
            time.sleep(0.2)
        client = next(iter(server.get_clients().values()))
        center = scene.center()
        cam = np.nanmedian(scene.z["T_world_camera"][:, :3, 3], axis=0)
        eye = video_eye(center, cam, azimuth_deg, elevation_deg, distance)
        wxyz = look_at(eye, center)
        fps = 1.0 / float(np.median(np.diff(scene.t_rel))) / step
        tmp = out.with_suffix(".tmp.mp4")
        frames = range(0, scene.n, step)
        t0 = time.monotonic()
        for k, i in enumerate(frames):
            image = scene.update(i)
            server.flush()
            time.sleep(0.05)  # let the browser take the new meshes and image before rendering
            render = client.get_render(height, width, wxyz=wxyz, position=eye, fov=np.deg2rad(50),
                                       transport_format="jpeg", timeout=30)
            left = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            if image is None:
                image = np.zeros((scene.H, scene.W, 3), np.uint8)
            right = cv2.resize(scene.overlay(i, image), (round(scene.W * height / scene.H), height))
            canvas = np.hstack([left, right])
            label = f"{session.root.name}  frame {int(scene.z['frame'][i])}  t {scene.t_rel[i]:5.2f} s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(canvas, (4, 4), (20 + tw, 16 + th), (255, 255, 255), -1)
            cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(canvas, "left camera + MANO", (width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if log and (k % 100 == 0 or k == len(frames) - 1):
                log(f"  rendered {k + 1}/{len(frames)} frames ({time.monotonic() - t0:.0f} s)")
    finally:
        if writer is not None:
            writer.release()
        chrome.terminate()
        try:
            chrome.wait(10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
        server.stop()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    if log:
        log(f"  video: {out}")
    return out
