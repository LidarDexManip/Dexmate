"""HACo-style fingertip force regulation, without hardware."""

import unittest

import numpy as np

from wuji_teleop.hand_compliance import (
    ComplianceParams,
    FingertipJacobians,
    HandForceRegulator,
)
from wuji_teleop.hand_load import JointLoad
from wuji_teleop.vega_bimanual_sim import load_bimanual_scene

KP = 2.85  # measured firmware stiffness for the written kp 1.5, A/rad
PARAMS = ComplianceParams(friction=0.0, spread_friction=0.0)  # frictionless simulated joints
LOWER, UPPER = np.full(20, -1.0), np.full(20, 2.0)
_J = np.array([[0.05, 0.0, 0.03, 0.01],
               [0.0, 0.04, 0.0, 0.0],
               [0.06, 0.0, 0.04, 0.02]])


def _fixed_jacobians(q):
    return [_J.copy() for _ in range(5)]


def _current_for(force, params):
    """Joint currents that make finger 1 (index) push with ``force``."""
    tau = _J.T @ force
    current = np.zeros(20)
    deadband = np.array([params.friction, params.spread_friction, params.friction,
                         params.friction])
    current[4:8] = np.sign(tau) * (np.abs(tau) + deadband)
    return current


class RegulatorUnitTests(unittest.TestCase):
    def regulator(self, **params):
        return HandForceRegulator(_fixed_jacobians, LOWER, UPPER, ComplianceParams(**params))

    def test_currents_inside_the_friction_deadband_pass_the_nominal_through(self):
        reg = self.regulator()
        q = np.full(20, 0.3)
        for k in range(50):
            step = reg.step(0.01 * k, q, q, np.full(20, 0.1))
        np.testing.assert_array_equal(step.compliant, q)
        np.testing.assert_array_equal(step.weight, 0.0)

    def test_force_estimate_recovers_the_fingertip_force(self):
        params = ComplianceParams(mu=1e-9)
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params)
        force = np.array([3.0, -1.0, 6.0])
        estimate, _ = reg.estimate_forces(np.zeros(20), _current_for(force, params))
        np.testing.assert_allclose(estimate[1], force, atol=1e-3)
        np.testing.assert_allclose(estimate[[0, 2, 3, 4]], 0.0)

    def test_force_over_the_limit_while_still_bends_the_reference_back(self):
        params = ComplianceParams()
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params)
        q = np.full(20, 0.5)
        current = _current_for(np.array([0.0, 0.0, 20.0]), params)
        for k in range(30):
            step = reg.step(0.01 * k, q, q, current)
        self.assertGreater(step.weight[1], 0.99)
        retreat = _J @ (step.compliant[4:8] - q[4:8])
        self.assertLess(retreat[2], -1e-3)  # moved against the force it applies
        np.testing.assert_array_equal(step.compliant[:4], q[:4])  # other fingers untouched

    def test_a_finger_moving_along_its_force_is_not_taken_for_contact(self):
        params = ComplianceParams()
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params)
        current = _current_for(np.array([0.0, 0.0, 20.0]), params)
        direction = np.linalg.pinv(_J) @ np.array([0.0, 0.0, 1.0])
        q = np.full(20, 0.2)
        for k in range(30):  # fingertip moving at ~0.1 m/s along the force: free motion
            q[4:8] += direction * 0.1 * 0.01
            step = reg.step(0.01 * k, q, q, current)
        np.testing.assert_array_equal(step.weight, 0.0)

    def test_a_finger_pushed_back_against_its_force_is_in_contact(self):
        params = ComplianceParams()
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params)
        current = _current_for(np.array([0.0, 0.0, 20.0]), params)
        direction = np.linalg.pinv(_J) @ np.array([0.0, 0.0, 1.0])
        q = np.full(20, 0.2)
        for k in range(30):  # something pushes the fingertip back at ~0.1 m/s
            q[4:8] -= direction * 0.1 * 0.01
            step = reg.step(0.01 * k, np.full(20, 0.2), q, current)
        self.assertGreater(step.weight[1], 0.99)

    def _load(self):
        """Index S3 carries 0.4 A of its own at any angle; index S1 has 0.3 A of friction."""
        friction = np.zeros(20)
        friction[4] = 0.3
        return JointLoad({6: ([0.0, 2.0], [0.4, 0.4])}, friction)

    def test_a_joints_own_load_is_not_read_as_force(self):
        q, current = np.full(20, 0.2), np.zeros(20)
        current[6] = 0.4
        bare = HandForceRegulator(_fixed_jacobians, LOWER, UPPER)
        self.assertGreater(np.linalg.norm(bare.estimate_forces(q, current)[0][1]), 1.0)
        loaded = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, load_source=self._load)
        np.testing.assert_array_equal(loaded.estimate_forces(q, current)[0], 0.0)

    def test_force_beyond_a_joints_own_load_is_still_read(self):
        params = ComplianceParams(mu=1e-9)
        force = np.array([0.0, 0.0, 10.0])
        current = _current_for(force, params)
        table = JointLoad({6: ([0.0, 2.0], [0.4, 0.4])}, np.zeros(20))
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params, load_source=lambda: table)
        current[6] += 0.4
        np.testing.assert_allclose(reg.estimate_forces(np.full(20, 0.2), current)[0][1], force,
                                   atol=1e-3)

    def test_a_joints_swept_friction_widens_only_its_own_deadband(self):
        q, current = np.full(20, 0.2), np.zeros(20)
        current[[4, 7]] = 0.3  # within index S1's friction + margin, beyond index S4's default
        table = self._load()
        table.curves.clear()
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, load_source=lambda: table)
        reg.estimate_forces(q, current)
        self.assertAlmostEqual(reg._deadband[4], 0.35)
        self.assertAlmostEqual(reg._deadband[7], ComplianceParams().friction)

    def test_the_table_follows_the_hand_once_its_serial_is_known(self):
        source = [None]
        table = self._load()
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, load_source=lambda: source[0])
        q, current = np.full(20, 0.2), np.zeros(20)
        current[6] = 0.4
        self.assertGreater(np.linalg.norm(reg.estimate_forces(q, current)[0][1]), 1.0)
        source[0] = table
        np.testing.assert_array_equal(reg.estimate_forces(q, current)[0], 0.0)
        source[0] = None
        self.assertGreater(np.linalg.norm(reg.estimate_forces(q, current)[0][1]), 1.0)
        self.assertEqual(reg._deadband[4], ComplianceParams().friction)

    def test_steps_and_joint_limits_are_respected(self):
        params = ComplianceParams(step_limit=0.01)
        reg = HandForceRegulator(_fixed_jacobians, LOWER, UPPER, params)
        q = np.full(20, 1.99)
        current = _current_for(np.array([0.0, 0.0, -200.0]), params)  # wants to move up
        previous = q.copy()
        for k in range(40):
            step = reg.step(0.01 * k, q, q, current)
            self.assertLessEqual(np.abs(step.compliant - previous).max(), 0.01 + 1e-12)
            self.assertTrue(np.all(step.compliant <= UPPER) and np.all(step.compliant >= LOWER))
            previous = step.compliant


class FingertipContactTests(unittest.TestCase):
    """The curled index fingertip closes onto a rigid plane and sticks there; the
    hand's joint stiffness balances the contact force (linearized equilibrium). The
    simulated joints have no friction, so neither does the force estimate."""

    @classmethod
    def setUpClass(cls):
        cls.model, _, _, hands = load_bimanual_scene()
        cls.hand = hands["left"]

    def press(self, close, regulate=True, seconds=4.0, open_at=None):
        jac = FingertipJacobians(self.model, self.hand, "left")
        reg = HandForceRegulator(jac, self.hand.lower, self.hand.upper, PARAMS)
        q0 = np.full(20, 0.1)
        q0[4:8] = (0.5, 0.0, 0.7, 0.5)
        jac(q0)
        tip0 = jac.data.site_xpos[jac.tips[1]].copy()
        normal = jac(q0)[1] @ np.array(close)
        normal /= np.linalg.norm(normal)
        plane = normal @ tip0 + 0.012
        target, stuck, rows = q0.copy(), None, []
        for k in range(int(seconds / 0.01)):
            t = k * 0.01
            ramp = min(t, 1.0) if open_at is None or t < open_at else max(0.0, 1 - (t - open_at))
            q_nom = q0.copy()
            q_nom[4:8] += ramp * np.array(close)
            J = jac(target)[1]
            tip = jac.data.site_xpos[jac.tips[1]].copy()
            if stuck is None and normal @ tip > plane:
                stuck = tip - normal * (normal @ tip - plane)
            force = np.zeros(3) if stuck is None else np.linalg.solve(J @ J.T / KP, tip - stuck)
            q_obs = target.copy()
            if force @ normal > 0:
                q_obs[4:8] -= J.T @ force / KP
            else:
                force[:] = 0.0
                if stuck is not None and normal @ tip < plane:
                    stuck = None
            current = KP * (target - q_obs)
            if regulate:
                step = reg.step(t, q_nom, q_obs, current)
                target = step.compliant
            else:
                target = q_nom
            rows.append((t, np.linalg.norm(force), np.abs(target - q_nom).max()))
        return np.array(rows)

    def test_regulated_press_settles_near_the_limit_without_oscillating(self):
        for close in ((0.3, 0.0, 0.24, 0.18), (1.0, 0.0, 0.8, 0.6)):  # 0.25 and 1 rad past
            free = self.press(close, regulate=False)
            rows = self.press(close)
            late = rows[rows[:, 0] > 3.0]
            self.assertLess(late[:, 1].mean(), 1.5 * PARAMS.force_limit)
            self.assertLess(late[:, 1].mean(), 0.5 * free[free[:, 0] > 3.0, 1].mean())
            self.assertLess(np.abs(np.diff(rows[rows[:, 0] > 2.5, 1])).max(), 0.5)
        self.assertGreater(free[free[:, 0] > 3.0, 1].mean(), 20 * PARAMS.force_limit)

    def test_every_finger_joint_moves_its_fingertip(self):
        for J in FingertipJacobians(self.model, self.hand, "left")(np.full(20, 0.3)):
            self.assertGreater(np.linalg.norm(J, axis=0).min(), 0.015)  # DIP lever ~25 mm

    def test_a_straight_finger_does_not_inflate_force_along_its_axis(self):
        jac = FingertipJacobians(self.model, self.hand, "left")
        reg = HandForceRegulator(jac, self.hand.lower, self.hand.upper, PARAMS)
        q = np.zeros(20)
        q[6] = 0.05
        J = jac(q)[1]
        normal = np.linalg.svd(J)[0][:, 0]
        torque = np.zeros(20)
        torque[4:8] = J.T @ (5.0 * normal) + (0.0, 0.0, 0.05, 0.0)  # plus a push mid-finger
        force = reg.estimate_forces(q, torque)[0][1]
        self.assertLess(np.linalg.norm(force), 6.0)

    def test_a_light_press_under_the_limit_is_left_alone(self):
        rows = self.press((0.12, 0.0, 0.1, 0.08))
        self.assertLess(rows[:, 2].max(), 1e-9)

    def test_opening_releases_the_correction(self):
        rows = self.press((0.3, 0.0, 0.24, 0.18), open_at=2.0, seconds=3.5)
        self.assertGreater(rows[(rows[:, 0] > 1.5) & (rows[:, 0] < 2.0), 2].max(), 0.05)
        self.assertLess(rows[-1, 2], 1e-3)


if __name__ == "__main__":
    unittest.main()
