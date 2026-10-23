"""Hand-aware clearance checks and collision-free reset paths for both Vega arms.

The Vega links in the bimanual MuJoCo scene are visual-only, so MuJoCo
contacts cannot be used. Instead this measures mesh distances (MuJoCo uses
each mesh's convex hull, which over-approximates the part) between:

* each arm's upper arm, elbow, forearm and Wuji hand and the torso/head,
* the two arms from the shoulder link outward, including both hands,
* each Wuji hand and its own shoulder and upper arm.

Both hands are part of every check, at the finger angles last given to
:meth:`ArmClearanceChecker.set_fingers`.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time

import mujoco
import numpy as np

from .vega_bimanual_sim import HAND_PREFIX, SIDE_PREFIX, SIDES, body_names


TORSO_BODIES = (
    "base_link", "lift_link", "torso_flip_link", "head_l1", "head_l2", "head_l3")
REPORT_DISTANCE_M = 0.10


class PlanningError(RuntimeError):
    """No path to the requested pose keeps the required clearance."""


@dataclass(frozen=True)
class Clearance:
    distance_m: float  # REPORT_DISTANCE_M means "at least this far"
    pair: str | None


def split_arms(q):
    q = np.asarray(q, dtype=np.float64)
    return {"left": q[:7], "right": q[7:]}


def join_arms(arm_q):
    return np.concatenate([np.asarray(arm_q[side], dtype=np.float64) for side in SIDES])


def joint_distance(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


class ArmClearanceChecker:
    """Minimum mesh distance between the body parts that can collide."""

    def __init__(self, model, arms, hands):
        self.model = model
        self.data = mujoco.MjData(model)
        self.arms = arms
        self.hands = hands
        self._fromto = np.zeros(6)
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in range(model.nbody)]
        # Each body carries a visual and a collision copy of the same part;
        # one mesh per body is enough, so keep the lighter one.
        geom_of = {}
        for geom in range(model.ngeom):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            body = int(model.geom_bodyid[geom])
            vertices = model.mesh_vertnum[model.geom_dataid[geom]]
            if (body not in geom_of or vertices <
                    model.mesh_vertnum[model.geom_dataid[geom_of[body]]]):
                geom_of[body] = geom

        def geoms(predicate):
            return [g for b, g in sorted(geom_of.items()) if predicate(names[b])]

        def link(side, low, high):
            prefix = f"{SIDE_PREFIX[side]}_arm_l"
            return geoms(lambda n: n.startswith(prefix) and n[len(prefix):].isdigit()
                         and low <= int(n[len(prefix):]) <= high)

        def hand(side):
            wrist = body_names(side)["hand"]
            return geoms(lambda n: n == wrist or n.startswith(HAND_PREFIX[side] + "_"))

        torso = geoms(lambda n: n in TORSO_BODIES)
        if len(torso) != len(TORSO_BODIES):
            raise ValueError("Bimanual scene is missing torso or head meshes")
        pairs = []
        for side in SIDES:
            if not hand(side) or len(link(side, 1, 8)) != 8:
                raise ValueError(f"Bimanual scene is missing {side} arm or hand meshes")
            pairs += itertools.product(link(side, 3, 8) + hand(side), torso)
            pairs += itertools.product(hand(side), link(side, 1, 4))
        pairs += itertools.product(link("left", 2, 8) + hand("left"),
                                   link("right", 2, 8) + hand("right"))
        self._pairs = np.array(pairs, dtype=np.int64)
        self._bound = (model.geom_rbound[self._pairs[:, 0]] +
                       model.geom_rbound[self._pairs[:, 1]])
        self._names = names
        mujoco.mj_forward(model, self.data)

    def set_fingers(self, side, angles):
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (20,) or not np.isfinite(angles).all():
            raise ValueError("Expected 20 finite Wuji Hand 2 joint angles")
        hand = self.hands[side]
        self.data.qpos[hand.qpos_ids] = np.clip(angles, hand.lower, hand.upper)

    def clearance(self, q):
        """Smallest distance (m) over all checked pairs for the 14 arm joints."""
        for side, values in split_arms(q).items():
            self.data.qpos[self.arms[side].qpos_ids] = values
        mujoco.mj_kinematics(self.model, self.data)
        centers = self.data.geom_xpos
        # Bounding spheres give a lower bound; visit pairs nearest-first and
        # stop once no remaining pair can beat the best exact distance.
        lower = (np.linalg.norm(centers[self._pairs[:, 0]] - centers[self._pairs[:, 1]],
                                axis=1) - self._bound)
        candidates = np.flatnonzero(lower < REPORT_DISTANCE_M)
        best, best_pair = REPORT_DISTANCE_M, None
        for index in candidates[np.argsort(lower[candidates])]:
            if lower[index] >= best:
                break
            g1, g2 = self._pairs[index]
            distance = mujoco.mj_geomDistance(
                self.model, self.data, int(g1), int(g2), best, self._fromto)
            if distance < best:
                best, best_pair = float(distance), index
        if best_pair is None:
            return Clearance(best, None)
        g1, g2 = self._pairs[best_pair]
        pair = (f"{self._names[self.model.geom_bodyid[g1]]} <-> "
                f"{self._names[self.model.geom_bodyid[g2]]}")
        return Clearance(best, pair)


class ResetPlanner:
    """Joint-space path to a target pose that keeps ``margin`` clearance.

    Tries the straight path, then moving one arm at a time, then RRT-Connect
    with shortcutting. A start pose that is already closer than ``margin`` may
    only move within ``escape_radius`` of the start without getting closer.
    """

    def __init__(self, checker, lower, upper, *, margin=0.03, resolution=0.03,
                 step=0.20, escape_radius=0.35, rng=None):
        self.checker = checker
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        if self.lower.shape != (14,) or self.upper.shape != (14,):
            raise ValueError("Expected 14 lower and upper arm joint limits")
        self.margin = float(margin)
        self.resolution = float(resolution)
        self.step = float(step)
        self.escape_radius = float(escape_radius)
        self.rng = np.random.default_rng() if rng is None else rng
        self.checks = 0

    def _clearance(self, q):
        self.checks += 1
        return self.checker.clearance(q)

    def plan(self, start, goal, *, time_limit=5.0, cancelled=lambda: False):
        start = np.clip(np.asarray(start, dtype=np.float64), self.lower, self.upper)
        goal = np.asarray(goal, dtype=np.float64)
        if np.any(goal < self.lower) or np.any(goal > self.upper):
            raise PlanningError("Reset pose is outside the arm joint limits")
        self.checks = 0
        at_goal = self._clearance(goal)
        if at_goal.distance_m < self.margin:
            raise PlanningError(
                f"Reset pose itself is only {at_goal.distance_m * 100:.1f} cm clear "
                f"({at_goal.pair})")
        at_start = self._clearance(start)
        if at_start.distance_m <= 0.0:
            raise PlanningError(
                f"Arms already touch in the model ({at_start.pair}); move them apart first")
        self._start = start
        self._start_floor = min(self.margin, at_start.distance_m - 0.002)

        def mid(first):
            q = start.copy()
            sl = slice(0, 7) if first == "left" else slice(7, 14)
            q[sl] = goal[sl]
            return q

        for path in ([start, goal], [start, mid("left"), goal], [start, mid("right"), goal]):
            if all(self._edge_valid(a, b) for a, b in zip(path, path[1:])):
                return path
        path = self._rrt_connect(start, goal, time.monotonic() + time_limit, cancelled)
        return self._shortcut(path, iterations=80, cancelled=cancelled)

    def _valid(self, q):
        distance = self._clearance(q).distance_m
        if distance >= self.margin:
            return True
        return (joint_distance(q, self._start) <= self.escape_radius and
                distance >= self._start_floor)

    def _edge_valid(self, a, b):
        count = max(1, math.ceil(joint_distance(a, b) / self.resolution))
        return all(self._valid(a + (b - a) * (i / count)) for i in range(1, count + 1))

    def _sample(self, start, goal):
        if self.rng.random() < 0.15:
            return self.rng.uniform(self.lower, self.upper)
        low = np.maximum(self.lower, np.minimum(start, goal) - 0.6)
        high = np.minimum(self.upper, np.maximum(start, goal) + 0.6)
        return self.rng.uniform(low, high)

    def _rrt_connect(self, start, goal, deadline, cancelled):
        trees = ([start], [-1]), ([goal], [-1])

        def nearest(tree, q):
            nodes = np.asarray(tree[0])
            return int(np.argmin(np.max(np.abs(nodes - q), axis=1)))

        def steer(a, b):
            distance = joint_distance(a, b)
            return b if distance <= self.step else a + (b - a) * (self.step / distance)

        def extend(tree, q):
            index = nearest(tree, q)
            new = steer(tree[0][index], q)
            if not self._edge_valid(tree[0][index], new):
                return None
            tree[0].append(new)
            tree[1].append(index)
            return len(tree[0]) - 1

        def branch(tree, index):
            path = []
            while index != -1:
                path.append(tree[0][index])
                index = tree[1][index]
            return path

        while time.monotonic() < deadline:
            if cancelled():
                raise PlanningError("Reset planning cancelled")
            grow, other = trees
            added = extend(grow, self._sample(start, goal))
            if added is not None:
                target = grow[0][added]
                while True:
                    reached = extend(other, target)
                    if reached is None:
                        break
                    if joint_distance(other[0][reached], target) < 1e-9:
                        a, b = branch(grow, added), branch(other, reached)
                        path = a[::-1] + b[1:] if grow is trees[0] else b[::-1] + a[1:]
                        return path
            trees = trees[::-1]
        raise PlanningError(
            f"No collision-free reset path found within the time limit "
            f"({self.checks} clearance checks)")

    def _shortcut(self, path, *, iterations, cancelled):
        path = list(path)
        for _ in range(iterations):
            if len(path) <= 2 or cancelled():
                break
            i, j = sorted(self.rng.choice(len(path), size=2, replace=False))
            if j - i > 1 and self._edge_valid(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path


class PathFollower:
    """Trapezoidal timing along a joint-space polyline (max-joint arc length)."""

    def __init__(self, waypoints, *, max_velocity=0.30, max_acceleration=0.60):
        self.waypoints = [np.asarray(w, dtype=np.float64) for w in waypoints]
        if len(self.waypoints) < 2:
            self.waypoints = self.waypoints * 2
        lengths = [joint_distance(a, b) for a, b in zip(self.waypoints, self.waypoints[1:])]
        self.cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        self.length = float(self.cumulative[-1])
        v, a = float(max_velocity), float(max_acceleration)
        if v <= 0 or a <= 0:
            raise ValueError("Velocity and acceleration limits must be positive")
        ramp = v / a
        if a * ramp * ramp >= self.length:  # never reaches cruise speed
            ramp = math.sqrt(self.length / a)
            v = a * ramp
        self._v, self._a, self._ramp = v, a, ramp
        cruise = 0.0 if v == 0 else (self.length - a * ramp * ramp) / v
        self.duration = 2 * ramp + max(0.0, cruise)

    def _arc(self, t):
        t = min(max(t, 0.0), self.duration)
        a, v, ramp = self._a, self._v, self._ramp
        if t < ramp:
            return 0.5 * a * t * t
        if t > self.duration - ramp:
            left = self.duration - t
            return self.length - 0.5 * a * left * left
        return 0.5 * a * ramp * ramp + v * (t - ramp)

    def at(self, t):
        s = self._arc(t)
        index = int(np.clip(np.searchsorted(self.cumulative, s, side="right") - 1,
                            0, len(self.waypoints) - 2))
        span = self.cumulative[index + 1] - self.cumulative[index]
        alpha = 0.0 if span <= 0 else (s - self.cumulative[index]) / span
        a, b = self.waypoints[index], self.waypoints[index + 1]
        return a + (b - a) * min(1.0, max(0.0, alpha))
