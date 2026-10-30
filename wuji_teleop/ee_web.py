"""Loopback-only browser dashboard for preview-first Vega arm control."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import struct
import subprocess
import time
import zlib

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from aiohttp import web
from scipy.spatial.transform import Rotation

from .ee_dashboard import ENDPOINTS
from .inactive_hand import InactiveHandChecker
from .connection_monitor import ping, route_device
from .ee_preview import (PreviewTeleop, TableChecker, HoldMapping, EmptySource, straight_path)
from .hardware_teleop import (FakeArms, FakeHands, DexcontrolArms, WujiHands,
                             TeleopLimits, reset_pose, _StaticRetargeter)
from .retarget import WujiRetargeter
from .sources import GloveSource
from .vega_bimanual_sim import SIDES, load_bimanual_scene, body_names, _side_ik
from .collision_check import ArmClearanceChecker, join_arms, split_arms


def png_bytes(rgb):
    def chunk(kind, body):
        return struct.pack('!I', len(body)) + kind + body + struct.pack('!I', zlib.crc32(kind + body) & 0xffffffff)
    height, width, _ = rgb.shape
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 2)) + chunk(b'IEND', b''))


# Installation-specific mount: right palm faces robot +Y with forearm forward.
WEB_HAND_MOUNT_ROLL_DEG = {'left': 0.0, 'right': 90.0}


def apply_hand_mounts(model, data):
    """Rotate hand roots about the EE's local +Z axis; arm coordinates stay unchanged."""
    for side, degrees in WEB_HAND_MOUNT_ROLL_DEG.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)['hand'])
        if body < 0:
            raise ValueError(f'Missing {side} hand mount')
        # Original mount is Rx(pi), MuJoCo quaternions use w,x,y,z.
        rotation = Rotation.from_euler('z', degrees, degrees=True) * Rotation.from_euler('x', 180, degrees=True)
        x, y, z, w = rotation.as_quat()
        model.body_quat[body] = [w, x, y, z]
    mujoco.mj_setConst(model, mujoco.MjData(model))
    mujoco.mj_forward(model, data)


def cartesian_pose_ik(model, data, arm):
    """Seven-joint DLS pose solve, seeded locally without teleop posture tasks."""
    solver = _side_ik(model, data, arm, upper_only=False)
    solver.iterations = 160
    solver.max_step = 0.1
    solver.damping = 0.02
    solver.position_weight = 1.0
    solver.orientation_weight = 0.5
    solver.elbow_weight = 0.0
    solver.elbow_down_weight = 0.0
    solver.roll_weight = 0.0
    solver.posture_gain = 0.0
    return solver


class Session:
    def __init__(self, side='right', hardware=False, use_glove=False):
        self.side, self.hardware, self.use_glove = side, hardware, use_glove
        self.model, self.data, self.arms, self.hands = load_bimanual_scene()
        apply_hand_mounts(self.model, self.data)
        self.glove = self.arms_io = self.hands_io = self.control = self.renderer = None
        self.preview = None
        self.preview_id = None
        self.started = 0
        self.network = {}
        self.startup_errors = {}
        self.inactive_hands = tuple(s for s in SIDES if s != side) if hardware else ()
        self.base_height = 0.15
        self.table_verified = False
        self.target = np.zeros(6)
        self.last_seen = time.monotonic()
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, .65]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.8, 135, -15
        self.option = mujoco.MjvOption()
        self.option.geomgroup[1] = 0
        self.option.geomgroup[2] = 1
        self.option.geomgroup[3] = 0
        # Vega visuals are group 2; Wuji's opaque hand meshes are group 1.
        # Keep hand rendering separate from Vega collision geometry.
        for geom in range(self.model.ngeom):
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    int(self.model.geom_bodyid[geom])) or ''
            if body.startswith(('l_', 'r_', 'wuji_left_', 'wuji_right_')):
                self.model.geom_group[geom] = 0 if self.model.geom_group[geom] == 1 else 3
        self.model.vis.headlight.ambient[:] = .4
        self.model.vis.headlight.diffuse[:] = .7
        try:
            if use_glove:
                self.glove = GloveSource(None, '', device_name=f'ee_web_{side}',
                    address=f'192.168.1.{100 if side == "left" else 101}:50001')
                if self.glove.side != side:
                    raise ValueError('Glove handedness mismatch')
            self.arms_io = DexcontrolArms() if hardware else FakeArms(reset_pose(0))
            self.hands_io = WujiHands() if hardware else FakeHands()
            try:
                self.hands_io.discover()
            except Exception as exc:
                if not hardware:
                    raise
                self.startup_errors['Hand discovery'] = str(exc)
            for name in ((side,) if hardware else SIDES):
                try:
                    self.hands_io.connect(name)
                    self.hands_io.wait_position(name)
                except Exception as exc:
                    if not hardware:
                        raise
                    self.startup_errors[f'{name.title()} hand feedback'] = str(exc)
            hand_start = self.hands_io.positions()
            # Never display a guessed finger pose. Unused hands get a conservative
            # envelope; missing selected-hand feedback blocks all control.
            for name in SIDES:
                if hand_start[name] is None:
                    if name not in self.inactive_hands:
                        self.startup_errors.setdefault(f'{name.title()} hand feedback', 'No measured joint feedback')
                    hand_start[name] = np.zeros(20)
                    for geom in range(self.model.ngeom):
                        body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                                int(self.model.geom_bodyid[geom])) or ''
                        if body.startswith((name[0] + '_', 'wuji_' + name + '_')):
                            self.model.geom_group[geom] = 4
            self.option.geomgroup[4] = 0
            measured = self.arms_io.positions()
            for name in SIDES:
                q = measured[name]
                if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < self.arms[name].lower) or np.any(q > self.arms[name].upper):
                    raise ValueError(f'{name} arm is outside model limits')
            if not self.startup_errors:
                self.hands_io.enable(side, hand_start[side])
            def checker():
                raw = ArmClearanceChecker(self.model, self.arms, self.hands)
                return TableChecker(InactiveHandChecker(raw, side, self.inactive_hands))
            self.checker = checker()
            self.control = PreviewTeleop(self.arms_io, self.hands_io, HoldMapping(),
                {s: self.glove if s == side and self.glove else EmptySource() for s in SIDES},
                {s: WujiRetargeter(s) if s == side and self.glove else _StaticRetargeter(hand_start[s]) for s in SIDES},
                checker(), None, reset_pose(),
                arm_lower={s: self.arms[s].lower for s in SIDES}, arm_upper={s: self.arms[s].upper for s in SIDES},
                hand_lower={s: self.hands[s].lower for s in SIDES}, hand_upper={s: self.hands[s].upper for s in SIDES},
                limits=TeleopLimits(tracking_fault=.15))
            self.control.required_hand_sides = (side,) if hardware else SIDES
            self.select_side(side)
            self.set_table(.15, False)
            if not self.startup_errors:
                self.control.start()
            else:
                self.control._log('READ-ONLY arm feedback; motion disabled: ' + self.block_reason())
            self.model.vis.global_.offheight = 540
            self.model.vis.global_.offwidth = 720
            self.renderer = mujoco.Renderer(self.model, height=540, width=720)
        except BaseException:
            self.close()
            raise

    def block_reason(self):
        return '; '.join(f'{name}: {detail}' for name, detail in self.startup_errors.items())

    def require_control(self):
        if self.startup_errors:
            raise ValueError('Motion disabled — ' + self.block_reason() +
                             '. Restore the missing links, then restart the dashboard.')

    def refresh_readonly(self):
        if self.startup_errors:
            measured = self.arms_io.positions()
            if any(np.asarray(measured[s]).shape != (7,) or not np.isfinite(measured[s]).all() for s in SIDES):
                raise ValueError('Invalid arm feedback; cannot display current pose')
            with self.control._lock:
                self.control.measured = {s: np.array(measured[s], copy=True) for s in SIDES}
                self.control.command = {s: np.array(measured[s], copy=True) for s in SIDES}
                self.control.estop_flags = tuple(self.arms_io.estop())
                for side, q in self.hands_io.positions().items():
                    if q is not None:
                        self.control.hand_command[side] = np.array(q, copy=True)

    def invalidate(self):
        self.preview = self.preview_id = None

    def select_side(self, side):
        if side not in SIDES:
            raise ValueError('Choose left or right')
        if self.control.state != 'paused' and hasattr(self, 'ik'):
            raise ValueError('Pause before changing arms')
        if side != self.side and (self.hardware or self.glove):
            raise ValueError('Restart with --side ' + side + ' to switch connected devices')
        self.side = side
        self.ik = cartesian_pose_ik(self.model, self.data, self.arms[side])
        self.ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_names(side)['ee'])
        self.current_pose()

    def display(self, q, fingers):
        for s in SIDES:
            self.data.qpos[self.arms[s].qpos_ids] = q[s]
            self.data.qpos[self.hands[s].qpos_ids] = fingers[s]
        mujoco.mj_forward(self.model, self.data)

    def current_pose(self):
        self.refresh_readonly()
        state = self.control.status()
        self.display(state['measured'], state['hand_command'])
        self.target = np.r_[self.data.xpos[self.ee], Rotation.from_matrix(
            self.data.xmat[self.ee].reshape(3, 3)).as_euler('xyz', degrees=True)]
        self.invalidate()

    def set_target(self, target):
        if self.control.state != 'paused':
            raise ValueError('Pause before editing a target')
        target = np.asarray(target, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError('Enter six finite target coordinates')
        self.target = target.copy()
        self.invalidate()

    def set_table(self, height, verified):
        height = float(height)
        if not np.isfinite(height) or not 0 <= height <= 2:
            raise ValueError('Base height must be between 0 and 2 metres')
        self.control.request_pause()
        self.invalidate()
        self.base_height, self.table_verified = height, bool(verified)
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.checker.height = float(self.data.xpos[base, 2]) + .762 - height
        self.control.guard_checker.height = self.checker.height

    def plan(self):
        self.require_control()
        self.invalidate()
        state = self.control.status()
        if state['state'] != 'paused':
            raise ValueError('Pause before previewing')
        if self.hardware and not self.table_verified:
            raise ValueError('Verify the measured base height before a hardware preview')
        pose = np.eye(4)
        pose[:3, 3] = self.target[:3]
        pose[:3, :3] = Rotation.from_euler('xyz', self.target[3:], degrees=True).as_matrix()
        self.display(state['command'], state['hand_command'])
        goal = {s: state['command'][s].copy() for s in SIDES}
        goal[self.side] = self.ik.solve(pose, seed=goal[self.side])
        self.display(goal, state['hand_command'])
        error = np.linalg.norm(self.data.xpos[self.ee] - pose[:3, 3])
        angle = Rotation.from_matrix(pose[:3, :3].T @ self.data.xmat[self.ee].reshape(3, 3)).magnitude()
        if error > .01 or angle > np.deg2rad(5):
            raise ValueError(f'IK did not converge: position error {error*1000:.1f} mm / orientation error {np.rad2deg(angle):.1f} degrees')
        q = goal[self.side]
        if np.any(q < self.arms[self.side].lower) or np.any(q > self.arms[self.side].upper):
            raise ValueError('Joint limits exceeded')
        self.preview = straight_path(join_arms(state['command']), join_arms(goal), self.checker,
                                     state['hand_command'], self.side)
        self.preview_id = secrets.token_hex(12)
        self.started = time.monotonic()

    def execute(self, preview_id):
        self.require_control()
        if self.preview is None or preview_id != self.preview_id:
            raise ValueError('Preview changed; preview again')
        if time.monotonic() - self.started < self.preview.follower.duration:
            raise ValueError('Watch the preview playback first')
        self.control.execute(self.preview)
        self.invalidate()

    def glove_start(self):
        self.require_control()
        if not self.glove:
            raise ValueError('Restart with --glove to use the physical glove')
        if self.hardware and not self.table_verified:
            raise ValueError('Verify the base height first')
        snap = self.glove.read()
        if snap.error or snap.frame is None or time.monotonic() - snap.frame.received_at > .25:
            raise ValueError(snap.error or 'No fresh glove frames')
        state = self.control.status()
        for s in SIDES:
            self.checker.set_fingers(s, state['hand_command'][s])
        if self.checker.clearance(join_arms(state['command'])).distance_m < .015:
            raise ValueError('Resolve current robot/table clearance first')
        self.invalidate()
        if not self.control.request_start():
            raise ValueError('Glove start refused; pause and check E-stop')

    def status(self):
        self.refresh_readonly()
        state = self.control.status()
        if state['state'] != 'paused':
            self.invalidate()
        components = dict(self.network)
        components['Arm controller'] = ['off', 'SIMULATED — no robot connection']
        if self.hardware:
            alive, age = self.arms_io.health()
            components['Arm controller'] = ['ok' if alive else 'fail', f"{state['state']} · heartbeat {age:.1f}s"]
        for s in SIDES:
            age = self.hands_io.feedback_age(s)
            components[f'{s.title()} hand feedback'] = (['off', 'simulated'] if not self.hardware else
                ['ok' if age is not None and age < .5 else 'fail', 'live' if age is not None and age < .5 else 'stale'])
        components['Glove stream'] = ['off', 'not connected · launch with --glove']
        if self.glove:
            snap = self.glove.read()
            fresh = snap.frame is not None and time.monotonic() - snap.frame.received_at < .25
            components['Glove stream'] = ['fail' if snap.error else 'ok' if fresh else 'wait',
                f'{self.side}: ' + (snap.error or ('streaming' if fresh else 'waiting / stale'))]
        for side in self.inactive_hands:
            components[f'{side.title()} hand feedback'] = ['off', 'unused — conservative hand envelope; arm held']
            for name in (f'{side.title()} Wuji hand', f'{side.title()} glove'):
                detail = components.get(name, ['off', 'not checked'])[1]
                components[name] = ['off', f'unused for {self.side}-only control · ' + detail]
        components['Pico / trackers'] = ['off', 'not used']
        for name, detail in self.startup_errors.items():
            components[name] = ['fail', detail + ' · pose unknown; motion blocked']
        if self.startup_errors:
            components['Arm controller'] = ['wait', 'READ-ONLY measured arm feedback; no commands sent']
        return dict(blocked=bool(self.startup_errors), block_reason=self.block_reason(),
            side=self.side, hardware=self.hardware, state='read-only' if self.startup_errors else state['state'], target=self.target.tolist(),
            base_height=self.base_height, table_verified=self.table_verified, components=components,
            preview_id=self.preview_id, ready=bool(self.preview and time.monotonic()-self.started >= self.preview.follower.duration),
            duration=self.preview.follower.duration if self.preview else 0, messages=state['messages'],
            side_locked=bool(self.hardware or self.glove))

    def frame(self):
        self.refresh_readonly()
        state = self.control.status()
        q, fingers = state['measured'], state['hand_command']
        if self.preview and state['state'] == 'paused':
            q, fingers = split_arms(self.preview.follower.at(time.monotonic()-self.started)), self.preview.fingers
        self.display(q, fingers)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
        scene = self.renderer.scene
        cameras = scene.camera
        forward = np.mean([c.forward for c in cameras], axis=0)
        forward /= np.linalg.norm(forward)
        up = np.mean([c.up for c in cameras], axis=0)
        up /= np.linalg.norm(up)
        near = float(np.mean([c.frustum_near for c in cameras]))
        self.frame_camera = dict(position=np.mean([c.pos for c in cameras], axis=0).tolist(),
            forward=forward.tolist(), up=up.tolist(), width=720, height=540,
            tan_y=float(np.mean([c.frustum_top-c.frustum_bottom for c in cameras])) / (2*near),
            shift_x=float(np.mean([c.frustum_center for c in cameras]))/near,
            shift_y=float(np.mean([c.frustum_top+c.frustum_bottom for c in cameras]))/(2*near))
        # Thin axes show the displayed EE; the browser's larger gizmo edits the target.
        position = self.data.xpos[self.ee]
        rotation = self.data.xmat[self.ee].reshape(3, 3)
        for axis, color in enumerate(([1., .2, .2, 1.], [.2, 1., .3, 1.], [.25, .55, 1., 1.])):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.ones(3), position,
                               np.eye(3).ravel(), np.array(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .003,
                                position, position + .075 * rotation[:, axis])
            scene.ngeom += 1
        for kind, size, pos, color in (
                (mujoco.mjtGeom.mjGEOM_BOX, [1, 1, .005], [0, 0, self.checker.height-.005], [.55, .4, .25, .3]),
                (mujoco.mjtGeom.mjGEOM_SPHERE, [.018]*3, self.target[:3], [.1, .9, .75, .8])):
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size), np.array(pos),
                               np.eye(3).ravel(), np.array(color, dtype=np.float32))
            scene.ngeom += 1
        for side, (root, radius) in self.checker.checker.envelopes.items():
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, radius), self.data.xpos[root], np.eye(3).ravel(),
                np.array([1., .65, .15, .15], dtype=np.float32))
            scene.ngeom += 1
        return png_bytes(self.renderer.render())

    def close(self):
        if self.control:
            self.control.stop()
        for resource in (self.renderer, self.hands_io, self.arms_io, self.glove):
            if resource:
                resource.close()


def create_app(session, port=8088):
    token = secrets.token_urlsafe(32)
    hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}

    @web.middleware
    async def guard(request, handler):
        if request.host not in hosts:
            raise web.HTTPForbidden(text='Invalid host')
        origin = request.headers.get('Origin')
        if origin and origin not in {f'http://{host}' for host in hosts}:
            raise web.HTTPForbidden(text='Invalid origin')
        if request.path.startswith('/api/') and not secrets.compare_digest(request.headers.get('X-Session-Key', ''), token):
            raise web.HTTPForbidden(text='Invalid session')
        try:
            response = await handler(request)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            response = web.json_response({'error': str(exc)}, status=400)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'"
        return response

    app = web.Application(middlewares=[guard], client_max_size=16384)

    async def index(request):
        html = (Path(__file__).parent/'web'/'ee_dashboard.html').read_text()
        return web.Response(text=html.replace('__SESSION_KEY__', token), content_type='text/html')

    async def state(request):
        session.last_seen = time.monotonic()
        return web.json_response(session.status())

    async def frame(request):
        image = session.frame()
        return web.Response(body=image, content_type='image/png',
                            headers={'X-View-Projection': json.dumps(session.frame_camera, separators=(',', ':'))})

    async def action(request):
        body = await request.json()
        name = request.match_info['name']
        if name == 'target':
            session.set_target(body['target'])
        elif name == 'preview':
            if session.control.state != 'paused':
                raise ValueError('Pause before previewing')
            session.set_table(body['base_height'], body.get('table_verified', False))
            session.set_target(body['target'])
            session.plan()
        elif name == 'execute':
            session.execute(body.get('preview_id'))
        elif name == 'pause':
            session.control.request_pause(); session.invalidate()
        elif name == 'estop':
            if session.startup_errors:
                session.arms_io.set_estop(True)
            else:
                session.control.request_estop()
            session.invalidate()
        elif name == 'release':
            session.require_control()
            session.control.request_release(); session.invalidate()
        elif name == 'glove':
            session.glove_start()
        elif name == 'current':
            session.current_pose()
        elif name == 'side':
            session.select_side(body['side'])
        elif name == 'table':
            session.set_table(body['height'], body.get('verified', False))
        elif name == 'camera':
            values = np.asarray([body['azimuth'], body['elevation'], body['distance']], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Invalid camera')
            session.camera.azimuth = values[0] % 360
            session.camera.elevation = float(np.clip(values[1], -85, 85))
            session.camera.distance = float(np.clip(values[2], .5, 5))
        else:
            raise web.HTTPNotFound()
        return web.json_response(session.status())

    async def background(app):
        def checks():
            def check(item):
                name, ip = item
                ok, route = ping(ip), route_device(ip)
                return name, ['ok' if ok else 'fail', f'{ip} · {"reachable" if ok else "no reply"} · {route or "no route"}']
            with ThreadPoolExecutor(max_workers=6) as pool:
                return dict(pool.map(check, ENDPOINTS.items()))
        async def monitor():
            while True:
                session.network = await asyncio.to_thread(checks)
                await asyncio.sleep(3)
        async def watchdog():
            while True:
                if session.hardware and time.monotonic() - session.last_seen > 1.5:
                    session.control.request_pause(); session.invalidate()
                await asyncio.sleep(.2)
        tasks = [asyncio.create_task(monitor()), asyncio.create_task(watchdog())]
        yield
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        session.close()

    app.cleanup_ctx.append(background)
    app.router.add_get('/', index)
    app.router.add_get('/api/state', state)
    app.router.add_get('/api/frame', frame)
    app.router.add_post('/api/{name}', action)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=SIDES, default='right')
    parser.add_argument('--glove', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args(argv)
    if args.hardware and (not args.glove or not all(os.environ.get(k) for k in ('ROBOT_NAME', 'ZENOH_CONFIG'))):
        parser.error('Hardware requires --glove and the launcher robot configuration')
    if args.hardware or args.glove:
        result = subprocess.run(['pgrep', '-af', 'python.*-m wuji_teleop'], capture_output=True, text=True)
        conflicts = [line for line in result.stdout.splitlines() if line.split()[0] != str(os.getpid())]
        if conflicts:
            parser.error('Stop other teleop/simulation sessions before connecting devices: ' + '; '.join(conflicts))
    session = Session(args.side, args.hardware, args.glove)
    print(f'Dashboard: http://127.0.0.1:{args.port} ({"HARDWARE" if args.hardware else "SIMULATION ONLY"})', flush=True)
    web.run_app(create_app(session, args.port), host='127.0.0.1', port=args.port, handle_signals=True)


if __name__ == '__main__':
    main()
