"""viser world viewer for a processed demonstration.

Shows, in the PICO world (robot axes: x forward, y left, z up), the two hands
(skeleton and capsule surface), the object (depth-tracked and grounded), the
head camera with its rectified left image, and that frame's depth as a
coloured point cloud. A timeline slider, play and per-layer toggles.

Placing the object: give ``--mesh`` (first time), pick a frame where the object
is clearly visible, press "Object to view centre", drag the gizmo until the
mesh sits on the point cloud, then "Save initial pose". ``process.py track``
refines that pose against the depth before tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .hands import FINGER_BONES, HandSurface

SIDE_COLORS = {"left": (70, 140, 255), "right": (255, 140, 60)}
BONES = FINGER_BONES + [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]


def _wxyz(R):
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def serve(session, port=8120, mesh=None, mesh_scale=1.0, initial_frame=None):
    import viser
    from .process import hand_surfaces, load_mesh
    from .stereo import backproject, load_frame

    tl = session.timeline()
    n = len(tl["frame_utc"])
    t_rel = tl["frame_utc"] - tl["frame_utc"][0]
    K = tl["K"]
    W, H = (int(v) for v in tl["image_size"])
    sides = [s for s in ("left", "right") if f"keypoints_{s}" in tl]
    surfaces = hand_surfaces(tl)
    obj_meta_path = session.out / "object.json"
    init_path = session.out / "object_init.json"
    obj_meta = json.loads(obj_meta_path.read_text()) if obj_meta_path.exists() else None
    if mesh is None and obj_meta is not None:
        mesh, mesh_scale = obj_meta["mesh"], obj_meta["mesh_scale"]
    if mesh is None and init_path.exists():
        init = json.loads(init_path.read_text())
        mesh, mesh_scale = init["mesh"], init.get("mesh_scale", 1.0)
    object_mesh = load_mesh(mesh, mesh_scale) if mesh else None
    track = dict(np.load(session.out / "object_track.npz")) if (session.out / "object_track.npz").exists() else None
    grounded = dict(np.load(session.out / "grounded.npz")) if (session.out / "grounded.npz").exists() else None
    holds = []
    if (session.out / "ground_report.json").exists():
        for h in json.loads((session.out / "ground_report.json").read_text())["holds"]:
            hold = h.get("hold", h)
            holds.append((h.get("side", hold.get("side")), hold["start"], hold["end"]))

    server = viser.ViserServer(host="127.0.0.1", port=port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4, height=4, cell_size=0.1, position=(0, 0, 0))
    gui = server.gui
    with_depth = [i for i in range(n) if (session.frames / f"{i:06d}.png").exists()]
    start_frame = initial_frame if initial_frame is not None else (with_depth[0] if with_depth else 0)
    frame = gui.add_slider("Frame", min=0, max=n - 1, step=1, initial_value=int(start_frame))
    play = gui.add_checkbox("Play", False)
    speed = gui.add_dropdown("Speed", options=["0.25x", "0.5x", "1x", "2x"], initial_value="1x")
    prev_b, next_b = gui.add_button("Previous"), gui.add_button("Next")
    with gui.add_folder("Layers"):
        show_hands = gui.add_checkbox("Hand skeletons", True)
        show_surface = gui.add_checkbox("Hand surfaces", True)
        show_tracked = gui.add_checkbox("Object: depth-tracked", grounded is None)
        show_grounded = gui.add_checkbox("Object: grounded", grounded is not None,
                                         visible=grounded is not None)
        show_cloud = gui.add_checkbox("Depth point cloud", True)
        show_camera = gui.add_checkbox("Head camera image", True)
        show_trackers = gui.add_checkbox("Wrist trackers", False)
        cloud_stride = gui.add_slider("Point cloud stride", min=1, max=8, step=1, initial_value=3)
    info = gui.add_markdown("")
    dirty = [True]
    for toggle in (show_hands, show_surface, show_tracked, show_grounded, show_cloud, show_camera,
                   show_trackers, cloud_stride):
        toggle.on_update(lambda _: dirty.__setitem__(0, True))
    place = None
    if object_mesh is not None:
        with gui.add_folder("Place object", expand_by_default=track is None):
            to_view = gui.add_button("Object to view centre")
            save = gui.add_button("Save initial pose")
            status = gui.add_markdown("")
        start = np.eye(4)
        if init_path.exists():
            start = np.asarray(json.loads(init_path.read_text())["T_world_object"], float)
        place = server.scene.add_transform_controls("/place", scale=0.15, wxyz=_wxyz(start[:3, :3]),
                                                    position=start[:3, 3], visible=track is None)
        place_mesh = server.scene.add_mesh_simple("/place/mesh", object_mesh.vertices, object_mesh.faces,
                                                  color=(120, 220, 120), opacity=0.7, visible=track is None)

        @to_view.on_click
        def _(_):
            i = int(frame.value)
            Twc = tl["T_world_camera"][i]
            try:
                _, depth = load_frame(session.frames, i)
                patch = depth[H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
                z = float(np.median(patch[patch > 0])) if (patch > 0).any() else 0.5
            except FileNotFoundError:
                z = 0.5
            center = Twc[:3, :3] @ np.array([0, 0, z]) + Twc[:3, 3]
            place.position = tuple(center - object_mesh.vertices.mean(0))
            place.visible = place_mesh.visible = True

        @save.on_click
        def _(_):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(np.roll(np.asarray(place.wxyz), -1)).as_matrix()
            T[:3, 3] = place.position
            session.out.mkdir(parents=True, exist_ok=True)
            init_path.write_text(json.dumps({"frame": int(frame.value), "mesh": str(Path(mesh).resolve()),
                                             "mesh_scale": float(mesh_scale),
                                             "T_world_object": T.tolist()}, indent=2))
            status.content = f"Saved frame {int(frame.value)} to `{init_path}`"

    @prev_b.on_click
    def _(_):
        play.value = False
        frame.value = max(0, int(frame.value) - 1)

    @next_b.on_click
    def _(_):
        play.value = False
        frame.value = min(n - 1, int(frame.value) + 1)

    handles = {}

    def put_mesh(name, vertices, faces, color, opacity, visible):
        if name in handles:
            handles[name].remove()
        handles[name] = server.scene.add_mesh_simple(name, vertices, faces, color=color,
                                                     opacity=opacity, visible=visible)

    if object_mesh is not None:
        for name, color, opacity in (("/object/tracked", (170, 170, 170), 0.45),
                                     ("/object/grounded", (230, 80, 160), 1.0)):
            handles[name] = server.scene.add_mesh_simple(name, object_mesh.vertices, object_mesh.faces,
                                                         color=color, opacity=opacity, visible=False)
    cloud = server.scene.add_point_cloud("/cloud", np.zeros((1, 3)), (0, 0, 0), point_size=0.0018)
    frustum = None
    fov = 2 * np.arctan(H / 2 / K[1, 1])

    def update(i):
        nonlocal frustum
        lines = []
        for side in sides:
            kp = (grounded or {}).get(f"keypoints_{side}", tl[f"keypoints_{side}"])[i]
            ok = bool(tl[f"valid_{side}"][i]) and np.isfinite(kp).all()
            seg = np.array([[kp[a], kp[b]] for a, b in BONES]) if ok else np.zeros((1, 2, 3))
            lines.append(side)
            name = f"/hands/{side}/skeleton"
            if name in handles:
                handles[name].remove()
            handles[name] = server.scene.add_line_segments(
                name, seg, SIDE_COLORS[side], thickness=0.004, visible=ok and show_hands.value)
            if side in surfaces:
                surface, verts = surfaces[side]
                v = verts[i]
                if grounded is not None and f"keypoints_{side}" in grounded:
                    v = surface.vertices(grounded[f"keypoints_{side}"][i:i + 1])[0]
                put_mesh(f"/hands/{side}/surface", v if ok else np.zeros((3, 3)), surface.faces if ok else
                         np.array([[0, 1, 2]]), SIDE_COLORS[side], 0.5, ok and show_surface.value)
            T = tl[f"tracker_{side}"][i]
            name = f"/trackers/{side}"
            if np.isfinite(T).all():
                server.scene.add_frame(name, axes_length=0.05, axes_radius=0.003, wxyz=_wxyz(T[:3, :3]),
                                       position=T[:3, 3], visible=show_trackers.value)
        for name, source, toggle in (("/object/tracked", track and track["poses"], show_tracked),
                                     ("/object/grounded", grounded and grounded["object_poses"], show_grounded)):
            if name not in handles:
                continue
            P = None if source is None else source[i]
            if P is not None and np.isfinite(P).all():
                handles[name].wxyz = _wxyz(P[:3, :3])
                handles[name].position = P[:3, 3]
                handles[name].visible = toggle.value
            else:
                handles[name].visible = False
        Twc = tl["T_world_camera"][i]
        image = None
        try:
            image, depth = load_frame(session.frames, i)
        except FileNotFoundError:
            depth = None
        if depth is not None and np.isfinite(Twc).all() and show_cloud.value:
            pts, vu = backproject(depth, K, int(cloud_stride.value))
            world = pts @ Twc[:3, :3].T + Twc[:3, 3]
            cloud.points = world.astype(np.float32)
            cloud.colors = image[vu[:, 0], vu[:, 1]]
            cloud.visible = True
        else:
            cloud.visible = False
        if frustum is not None:
            frustum.remove()
            frustum = None
        if np.isfinite(Twc).all():
            frustum = server.scene.add_camera_frustum(
                "/head_camera", fov=fov, aspect=W / H, scale=0.06, thickness=0.0015,
                image=image if show_camera.value else None,
                format="jpeg", jpeg_quality=70, wxyz=_wxyz(Twc[:3, :3]), position=Twc[:3, 3])
        text = [f"**Frame {i}/{n - 1}**, t = {t_rel[i]:.2f} s"]
        for side in sides:
            text.append(f"- {side} hand: {'tracked' if tl[f'valid_{side}'][i] else 'missing'}, wrist bend "
                        f"{tl[f'wrist_angle_{side}'][i]:.0f} deg")
        if track is not None:
            carried = int(track["carried_by"][i])
            text.append(f"- object: {'fitted to depth' if track['valid'][i] else 'not fitted (predicted)'}"
                        + (f", carried by the {track['sides'][carried]} hand" if carried >= 0 else ""))
        inside = [f"{s} {a}-{b}" for s, a, b in holds if a <= i <= b]
        if inside:
            text.append(f"- grounded hold: {', '.join(inside)}")
        info.content = "\n".join(text)

    # Start every client (and "Reset View") looking at the hands.
    if sides:
        pts = np.concatenate([tl[f"keypoints_{s}"][int(frame.value)] for s in sides])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if not len(pts):
            pts = np.concatenate([tl[f"keypoints_{s}"].reshape(-1, 3) for s in sides])
        center = np.nanmedian(pts, axis=0)
        if np.isfinite(center).all():
            server.initial_camera.look_at = tuple(center)
            server.initial_camera.position = tuple(center + np.array([-0.55, 0.35, 0.45]))
            server.initial_camera.up = (0.0, 0.0, 1.0)

    print(f"Demonstration viewer: http://localhost:{port}", flush=True)
    previous = None
    dt = float(np.median(np.diff(t_rel))) if n > 1 else 1 / 30
    while True:
        tick = time.monotonic()
        i = int(frame.value)
        if i != previous or dirty[0]:
            dirty[0] = False
            update(i)
            previous = i
        if play.value:
            frame.value = (i + 1) % n
            factor = float(speed.value.rstrip("x"))
            time.sleep(max(0.0, dt / factor - (time.monotonic() - tick)))
        else:
            time.sleep(0.03)
