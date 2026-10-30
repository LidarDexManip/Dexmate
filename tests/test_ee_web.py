import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import re
import json
import unittest
from unittest.mock import patch
import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from wuji_teleop.ee_web import Session, create_app


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [patch('wuji_teleop.ee_web.ping', return_value=False),
                        patch('wuji_teleop.ee_web.route_device', return_value=None)]
        for p in self.patches:
            p.start()
        self.session = Session()
        self.client = TestClient(TestServer(create_app(self.session, port=18089), port=18089))
        await self.client.start_server()
        text = await (await self.client.get('/')).text()
        self.key = re.search("const key='([^']+)'", text)[1]
        self.headers = {'X-Session-Key': self.key}

    async def asyncTearDown(self):
        await self.client.close()
        for p in self.patches:
            p.stop()

    async def post(self, name, body):
        return await self.client.post('/api/'+name, json=body, headers=self.headers)

    async def test_rejects_cross_origin_and_missing_token(self):
        self.assertEqual((await self.client.post('/api/execute', json={})).status, 403)
        headers = dict(self.headers, Origin='https://untrusted.example')
        response = await self.client.post('/api/pause', json={}, headers=headers)
        self.assertEqual(response.status, 403)

    async def test_render_and_switch_arm_without_motion(self):
        initial = self.session.arms_io.positions()
        response = await self.client.get('/api/frame', headers=self.headers)
        self.assertEqual(response.status, 200)
        self.assertTrue((await response.read()).startswith(b'\x89PNG'))
        camera = json.loads(response.headers['X-View-Projection'])
        self.assertEqual((camera['width'], camera['height']), (720, 540))
        self.assertGreater(camera['tan_y'], 0)
        point = np.asarray(self.session.camera.lookat)
        offset = point - camera['position']
        forward, up = np.array(camera['forward']), np.array(camera['up'])
        right = np.cross(forward, up)
        depth = offset @ forward
        self.assertGreater(depth, 0)
        self.assertAlmostEqual(float(offset @ right / depth), camera['shift_x'], places=5)
        self.assertAlmostEqual(float(offset @ up / depth), camera['shift_y'], places=5)
        self.assertEqual((await self.post('side', {'side': 'left'})).status, 200)
        self.assertEqual(self.session.side, 'left')
        for side in initial:
            np.testing.assert_array_equal(initial[side], self.session.arms_io.positions()[side])

    async def test_preview_gate_and_invalidation(self):
        target = self.session.target.copy()
        target[0] += .01
        # Lower the test plane so this test exercises approval, not table contact.
        body = {'target': target.tolist(), 'base_height': .25, 'table_verified': False}
        response = await self.post('preview', body)
        self.assertEqual(response.status, 200, await response.text())
        state = await response.json()
        self.assertEqual((await self.post('execute', {'preview_id':state['preview_id']})).status, 400)
        self.assertEqual((await self.post('target', {'target':self.session.target.tolist()})).status, 200)
        self.assertEqual((await self.post('execute', {'preview_id':state['preview_id']})).status, 400)

    async def test_rotated_hand_clearance_is_checked(self):
        response = await self.post('preview', {'target': self.session.target.tolist(),
                                  'base_height': .15, 'table_verified': False})
        self.assertEqual(response.status, 400)
        self.assertIn('tabletop', (await response.json())['error'])

    async def test_bad_target_and_table_rejected(self):
        self.assertEqual((await self.post('target', {'target':[1,2]})).status, 400)
        self.assertEqual((await self.post('table', {'height':-1})).status, 400)
        self.assertEqual((await self.post('preview', {})).status, 400)


if __name__ == '__main__':
    unittest.main()


class MissingHandTests(unittest.TestCase):
    def test_missing_selected_hand_reads_arms_but_never_commands_or_enables(self):
        from unittest.mock import Mock
        from wuji_teleop.hardware_teleop import FakeArms, FakeHands, reset_pose
        arms = FakeArms(reset_pose(0))
        arms.command = Mock(wraps=arms.command)
        hands = FakeHands(connected=('left',))
        hands.enable = Mock(wraps=hands.enable)
        with patch('wuji_teleop.ee_web.DexcontrolArms', return_value=arms), \
             patch('wuji_teleop.ee_web.WujiHands', return_value=hands):
            session = Session(hardware=True)
        try:
            status = session.status()
            self.assertTrue(status['blocked'])
            self.assertEqual(status['state'], 'read-only')
            self.assertIn('Right hand feedback', status['block_reason'])
            self.assertEqual(status['components']['Left hand feedback'][0], 'off')
            arms._q['right'][0] += .05
            session.status()
            np.testing.assert_array_equal(session.control.measured['right'], arms._q['right'])
            for action in (session.plan, session.glove_start, lambda: session.execute('anything')):
                with self.assertRaisesRegex(ValueError, 'Motion disabled'):
                    action()
            self.assertFalse(session.control._running)
            arms.command.assert_not_called()
            hands.enable.assert_not_called()
            self.assertTrue(session.frame().startswith(b'\x89PNG'))
        finally:
            session.close()
        arms.command.assert_not_called()
        hands.enable.assert_not_called()


class HandMountTests(unittest.TestCase):
    def test_right_palm_faces_left_without_rotating_arm_or_fingers_axis(self):
        import mujoco
        from wuji_teleop.ee_web import apply_hand_mounts
        from wuji_teleop.vega_bimanual_sim import load_bimanual_scene, body_names
        from wuji_teleop.hardware_teleop import reset_pose
        m, d, arms, hands = load_bimanual_scene()
        for side, q in reset_pose(0).items():
            d.qpos[arms[side].qpos_ids] = q
        mujoco.mj_forward(m, d)
        ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_names('right')['ee'])
        wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_names('right')['hand'])
        left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_names('left')['hand'])
        q_before = d.qpos.copy()
        ee_before = d.xmat[ee].copy()
        left_before = d.xmat[left].copy()
        finger_before = -d.xmat[wrist].reshape(3,3)[:,2].copy()
        apply_hand_mounts(m, d)
        np.testing.assert_allclose(d.qpos, q_before)
        np.testing.assert_allclose(d.xmat[ee], ee_before, atol=1e-10)
        np.testing.assert_allclose(d.xmat[left], left_before, atol=1e-10)
        rotation = d.xmat[wrist].reshape(3,3)
        np.testing.assert_allclose(rotation[:,1], [0,1,0], atol=2e-5)
        np.testing.assert_allclose(-rotation[:,2], finger_before, atol=1e-10)


class SingleHandTests(unittest.TestCase):
    def test_right_only_connects_enables_and_executes_without_left_feedback(self):
        from unittest.mock import Mock
        from wuji_teleop.hardware_teleop import FakeArms, FakeHands, reset_pose
        arms = FakeArms(reset_pose(0), lag=1)
        hands = FakeHands(connected=('right',))
        hands.connect = Mock(wraps=hands.connect)
        hands.enable = Mock(wraps=hands.enable)
        with patch('wuji_teleop.ee_web.DexcontrolArms', return_value=arms), \
             patch('wuji_teleop.ee_web.WujiHands', return_value=hands), \
             patch('wuji_teleop.ee_web.PreviewTeleop.start'):
            session = Session(hardware=True)
        try:
            self.assertFalse(session.status()['blocked'])
            hands.connect.assert_called_once_with('right')
            self.assertEqual(hands.enable.call_count, 1)
            self.assertEqual(hands.enable.call_args.args[0], 'right')
            self.assertEqual(session.control.required_hand_sides, ('right',))
            self.assertEqual(session.status()['components']['Left hand feedback'][0], 'off')
            session.set_table(.25, True)
            target = session.target.copy()
            target[0] += .01
            session.set_target(target)
            session.plan()
            session.started -= session.preview.follower.duration + 1
            session.execute(session.preview_id)
            session.control.io_step()
            self.assertNotEqual(session.control.state, 'paused')
            np.testing.assert_array_equal(arms.positions()['left'], reset_pose(0)['left'])
            self.assertIsNone(hands.positions()['left'])
        finally:
            session.close()
