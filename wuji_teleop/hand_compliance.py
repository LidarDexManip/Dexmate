"""HACo-style hand force regulation on current-sensed Wuji fingers.

Hand branch of Ye et al., "HACo: Learning Haptic Active Compliance for
Force-Aware Dexterous Manipulation" (2026), Fig. 3: nominal hand references from
retargeting become controller-executable compliant references. These hands have
no tactile sensors, so each fingertip force is estimated from motor current
through the finger's contact Jacobian; the rest follows the paper:

    tau_i  = joint current less its own load, beyond a friction deadband (finger i's 4 joints)
    f_i    = (J_i J_i^T + mu I)^-1 J_i tau_i      force the fingertip applies
    fbar_i = f_i * min(1, F_lim / |f_i|)          clipped, same direction
    d_i    = (fbar_i - f_i) / K                   virtual contact stiffness K
    dq*    = argmin sum_i w_i |J_i dq - d_i|^2 + lam_r |dq|^2 + lam_s |dq - dq_prev|^2
    q_cmp  = q_nom + dq*   (joint limits; |dq* - dq_prev| <= step limit)

Forces use the Wuji SDK's placeholder torque constant (1 N*m per A), so they are
uncalibrated newtons until per-joint torque constants are identified. The
activation w_i rises smoothly once finger i is loaded (|f_i| > force_on) while
nearly still or pushed back, and falls once |f_i| < force_off: current-based
force is polluted by motion (dynamics and the firmware's noisy damping term),
unlike HACo's tactile force, so the fingertip's motion gates the onset.

The correction is proportional: a steady force keeps about 1 / (1 + G) of its
excess over F_lim, G ~ kp_eff / (K (|J|^2 + lam_r)) with fingertip levers |J| of
5-10 cm. In rigid contact the per-step loop gain is about kp_eff / (K lam_s),
which the defaults keep near 0.5 for kp_eff 2.85 A/rad (the written kp 1.5);
scale K or lam_s with a higher kp. Joint torques cannot reveal force along a
nearly straight finger, so mu damps the estimate there. One addition
to the paper's objective: lam_r grows by (1 - w_i) lam_release as a finger's
activation falls, so a weak pull toward nominal in contact (tight force limit)
does not mean a slow release afterwards. Only fingertip forces are regulated, as
in HACo: load on other finger segments is bounded by the hand's current limit.

A joint's own load (cover, tendons) reads as force too, so ``load_source`` gives the
connected hand's ``hand_load.JointLoad``: its load at the measured angle is subtracted
and its friction, plus a margin for sticking, widens that joint's deadband.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SPREAD_JOINTS = (1, 5, 9, 13, 17)  # S2 of each finger
# Fingertip ends (sites on the distal links). The *_tip_sensor_frame bodies sit on
# the DIP axis, where the DIP joint cannot move them.
_TIP_SITES = ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip",
              "pinky_tip")


@dataclass(frozen=True)
class ComplianceParams:
    # From two left-hand sessions (2026-11-02): stalled grasps at the 0.8 A cap
    # read 5-13 N (median), spoon presses 3-5 N.
    force_limit: float = 4.0       # F_lim, uncalibrated N per fingertip
    force_on: float = 2.5          # activation rises above this ...
    force_off: float = 1.5         # ... and falls below this
    stiffness: float = 6.0         # K, N/m (virtual, not an object stiffness)
    lam_r: float = 0.001           # m^2, penalizes unneeded joint motion in contact
    lam_release: float = 0.25      # m^2, extra pull to nominal as w falls (release ~50 ms)
    lam_s: float = 1.0             # m^2, temporal smoothing toward dq_prev
    mu: float = 1e-4               # m^2, damps force along tip directions the joints
                                   # barely move (a straight finger's axis), < ~10 mm/rad
    step_limit: float = 0.06       # rad per update: retreats at 6 rad/s, faster than the
                                   # glove reference may close (max_hand_speed 5 rad/s)
    friction: float = 0.15         # A deadband on flexion joints (S1, S3, S4)
    spread_friction: float = 0.30  # A deadband on spread joints (S2)
    friction_margin: float = 0.05  # A above a joint's swept friction, for sticking
    still_speed: float = 0.02      # m/s, fingertip speed along the force below which
                                   # (or pushed back) a loaded finger activates
    free_speed: float = 0.05       # m/s, moving this fast along its force ends contact
    activation_tau: float = 0.03   # s, activation smoothing
    release_hold: float = 0.1      # s below force_off before the activation drops


class FingertipJacobians:
    """Fingertip position Jacobians (3 x 4 per finger) from a MuJoCo hand model.

    Uses its own ``MjData`` so it never races the teleop's clearance checks.
    """

    def __init__(self, model, hand, side):
        import mujoco

        self._mujoco = mujoco
        self.model, self.data = model, mujoco.MjData(model)
        self.qpos_ids = np.asarray(hand.qpos_ids)
        dofs = model.jnt_dofadr[np.asarray(hand.joint_ids)]
        self.dofs = [dofs[4 * i:4 * i + 4] for i in range(5)]
        prefix = {"left": "l", "right": "r"}[side]
        self.tips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_{name}")
                     for name in _TIP_SITES]
        if min(self.tips) < 0:
            raise ValueError("hand model lacks fingertip sites")
        self._jacp = np.zeros((3, model.nv))

    def __call__(self, q):
        mj = self._mujoco
        self.data.qpos[self.qpos_ids] = q
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)
        out = []
        for tip, dofs in zip(self.tips, self.dofs):
            mj.mj_jacSite(self.model, self.data, self._jacp, None, tip)
            out.append(self._jacp[:, dofs].copy())
        return out


@dataclass
class RegulatorStep:
    compliant: np.ndarray   # (20,) q_cmp sent to the hand
    force: np.ndarray       # (5, 3) estimated fingertip force, uncalibrated N
    weight: np.ndarray      # (5,) contact activation w_i


class HandForceRegulator:
    """One hand's regulator; call ``step`` once per command update."""

    def __init__(self, jacobians, lower, upper, params=None, load_source=None):
        self.jacobians = jacobians
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.params = params or ComplianceParams()
        deadband = np.full(20, self.params.friction)
        deadband[list(SPREAD_JOINTS)] = self.params.spread_friction
        self._base_deadband = self._deadband = deadband
        self.load_source = load_source  # () -> JointLoad or None, asked at every estimate
        self._load = None
        self.reset()

    def reset(self):
        self.dq = np.zeros(20)
        self.weight = np.zeros(5)
        self._loaded = np.zeros(5, bool)
        self._below_since = np.full(5, np.nan)
        self._last = None  # (time, q)

    def _follow_load(self):
        """A hand's serial, so its table, may only be known once it connects."""
        load = None if self.load_source is None else self.load_source()
        if load is not self._load:
            self._load = load
            self._deadband = self._base_deadband if load is None else np.maximum(
                self._base_deadband, load.friction + self.params.friction_margin)

    def estimate_forces(self, q, current, jacobians=None):
        p = self.params
        self._follow_load()
        if self._load is not None:
            current = current - self._load.at(q)
        tau = np.sign(current) * np.maximum(np.abs(current) - self._deadband, 0.0)
        jacs = self.jacobians(q) if jacobians is None else jacobians
        forces = np.zeros((5, 3))
        for i, J in enumerate(jacs):
            forces[i] = np.linalg.solve(J @ J.T + p.mu * np.eye(3), J @ tau[4 * i:4 * i + 4])
        return forces, jacs

    def _update_weights(self, now, q, forces, magnitude, jacs):
        """A fingertip moving along the force it applies is unobstructed: in free
        motion a finger accelerates along its force, in contact it cannot (the
        joints may still move in the finger's null space, and the regulator's
        own retreat runs against the force). So onset needs that speed near zero
        or negative (pushed back against its force), and a clear forward speed
        ends contact."""
        p = self.params
        along, dt = np.full(5, np.inf), 0.0  # no speed yet: stillness unconfirmed
        if self._last is not None:
            dt = max(now - self._last[0], 1e-4)
            qdot = (q - self._last[1]) / dt
            for i, J in enumerate(jacs):
                direction = forces[i] / max(magnitude[i], 1e-9)
                along[i] = direction @ (J @ qdot[4 * i:4 * i + 4])
        self._last = (now, q.copy())
        # A release by force must last release_hold: the correction's own brief
        # dips below force_off are not one.
        below = magnitude < p.force_off
        self._below_since = np.where(below, np.where(np.isnan(self._below_since), now,
                                                     self._below_since), np.nan)
        released = below & (now - np.nan_to_num(self._below_since, nan=now) >= p.release_hold)
        released |= np.isfinite(along) & (along > p.free_speed)
        self._loaded = np.where(self._loaded, ~released,
                                (magnitude > p.force_on) & (along < p.still_speed))
        alpha = 1.0 - np.exp(-dt / p.activation_tau) if dt else 0.0
        self.weight += alpha * (self._loaded.astype(float) - self.weight)

    def step(self, now, q_nom, q_obs, current):
        p = self.params
        q_nom, q_obs = np.asarray(q_nom, float), np.asarray(q_obs, float)
        forces, jacs = self.estimate_forces(q_obs, np.asarray(current, float))
        magnitude = np.linalg.norm(forces, axis=1)
        self._update_weights(now, q_obs, forces, magnitude, jacs)
        scale = np.minimum(1.0, p.force_limit / np.maximum(magnitude, 1e-9))
        dq = np.zeros(20)
        for i, J in enumerate(jacs):
            s = slice(4 * i, 4 * i + 4)
            d = (scale[i] - 1.0) * forces[i] / p.stiffness
            lam_r = p.lam_r + (1.0 - self.weight[i]) * p.lam_release
            lhs = self.weight[i] * J.T @ J + (lam_r + p.lam_s) * np.eye(4)
            dq[s] = np.linalg.solve(lhs, self.weight[i] * J.T @ d + p.lam_s * self.dq[s])
        dq = self.dq + np.clip(dq - self.dq, -p.step_limit, p.step_limit)
        compliant = np.clip(q_nom + dq, self.lower, self.upper)
        self.dq = compliant - q_nom
        return RegulatorStep(compliant, forces, self.weight.copy())
