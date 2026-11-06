"""Each Wuji finger joint's own load: what its motor carries with nothing touching the hand.

The cover and tendons load a joint more as it closes (thumb S3 about 0.27 A at 1.25 rad on
both hands), and friction differs per joint and per hand (right thumb S1 about twice the
left's). ``HandForceRegulator`` would read both as fingertip force and hold back a thumb
that touches nothing, so it subtracts this table.

``python -m wuji_teleop.hand_load left`` MOVES THE PHYSICAL HAND (only when asked; nothing
may touch it): each joint goes once up and down its range at 0.5 rad/s while the others
hold. The mean current of the two directions at an angle is the load there, half their
difference is friction. Tables are per hand, in ``wuji_teleop/hand_load/<serial>.json``.
``--plan`` only discovers the hand and prints the sweeps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TABLE_DIR = Path(__file__).parent / "hand_load"
_FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
_LABELS = ("thumb", "index", "middle", "ring", "pinky")
_HOLD = np.array([0.3, -0.2, 0.2, 0.1] + [0.0] * 16)  # fingers straight, thumb clear of them
_THUMB_OPPOSED, _THUMB_BENT = (0.3, -0.8, 0.5, 0.2), (0.3, -0.2, 0.8, 0.2)
# Range the gloves asked of each joint over the 2026-11 runs (0.5-99.5 percentile, both hands).
_REQUESTED = np.array([(-0.33, 0.89), (-1.47, 0.60), (0.00, 1.55), (-0.04, 1.51),
                       (-0.17, 1.57), (-0.70, 0.40), (0.02, 1.90), (0.00, 1.51),
                       (0.00, 1.57), (-0.64, 0.40), (-0.06, 1.66), (0.05, 1.22),
                       (0.01, 1.57), (-0.70, 0.38), (0.00, 1.72), (0.04, 1.03),
                       (-0.09, 1.55), (-0.68, 0.61), (-0.10, 1.94), (0.00, 0.99)])
_LIMIT_MARGIN = 0.05       # rad kept from the joint limits
_CLEARANCE = 0.005         # m kept from the rest of the hand (or the start gap less 0.5 mm)
SPEED = 0.5                # rad/s
RATE_HZ = 200.0
CURRENT_LIMIT = 0.8        # A per joint while sweeping
MAX_CURRENT, OVER_S = 0.7, 0.2       # A, s: abort
BLOCKED, BLOCKED_S = 0.22, 0.1       # rad of lag, s: this direction ends here
MAX_TEMP_C, FEEDBACK_TIMEOUT_S = 70.0, 0.1
_PAUSE_S = 0.3
_BIN = 0.1                 # rad, width of a fitted angle bin
_MIN_SAMPLES = 10          # per bin and direction


def joint_label(joint):
    return f"{_LABELS[joint // 4]}.S{joint % 4 + 1}"


class JointLoad:
    """One hand's load curves ({joint: (angles, A)}; none for a joint never swept) and friction."""

    def __init__(self, curves, friction):
        self.curves = {int(j): (np.asarray(a, float), np.asarray(v, float))
                       for j, (a, v) in curves.items()}
        self.friction = np.asarray(friction, float)

    def at(self, q):
        """Each joint's own load in A at angles ``q``; flat beyond the swept range."""
        load = np.zeros(20)
        for joint, (angles, values) in self.curves.items():
            load[joint] = np.interp(q[joint], angles, values)
        return load

    def save(self, path):
        data = {"friction": np.round(self.friction, 3).tolist(),
                "curves": {str(j): [np.round(a, 3).tolist(), np.round(v, 3).tolist()]
                           for j, (a, v) in self.curves.items()}}
        Path(path).write_text(json.dumps(data, indent=1) + "\n")

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["curves"], data["friction"])

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def for_serial(serial):
        """The table of the hand with this serial, or None."""
        path = TABLE_DIR / f"{serial}.json"
        return JointLoad.read(path) if serial and path.exists() else None


def plan_sweeps(model, data, hand, side, skip=()):
    """The sweeps for one hand, each joint's requested range cut where its finger would come
    close to the rest of the hand: {joint, posture, pose, lo, hi}."""
    import mujoco

    body = mujoco.mjtObj.mjOBJ_BODY
    prefix = side[0]

    def subtree(root):
        out, todo = [], [root]
        while todo:
            current = todo.pop()
            out.append(current)
            todo += [b for b in range(model.nbody)
                     if model.body_parentid[b] == current and b != current]
        return out

    def geoms(bodies):
        bodies = set(bodies)
        return [g for g in range(model.ngeom)
                if model.geom_bodyid[g] in bodies and model.geom_contype[g]]

    trees = [subtree(mujoco.mj_name2id(model, body, f"{prefix}_{f}_proximal")) for f in _FINGERS]
    whole_hand = subtree(mujoco.mj_name2id(model, body, f"wuji_{side}_wrist"))
    outer = [geoms(mujoco.mj_name2id(model, body, f"{prefix}_{f}_{link}")
                   for link in ("middle", "distal", "tip_sensor_frame")) for f in _FINGERS]
    rest = [geoms(set(whole_hand) - set(tree)) for tree in trees]
    whole = [geoms(tree) for tree in trees]
    others = [geoms(b for k, tree in enumerate(trees) if k != i for b in tree) for i in range(5)]
    fromto = np.zeros(6)

    def clearance(q, finger):
        data.qpos[hand.qpos_ids] = q
        mujoco.mj_kinematics(model, data)
        pairs = [(a, b) for a in outer[finger] for b in rest[finger]]
        pairs += [(a, b) for a in whole[finger] for b in others[finger]]
        return min(mujoco.mj_geomDistance(model, data, a, b, 0.05, fromto) for a, b in pairs)

    def trimmed(pose, joint):
        lo = max(_REQUESTED[joint, 0], hand.lower[joint] + _LIMIT_MARGIN)
        hi = min(_REQUESTED[joint, 1], hand.upper[joint] - _LIMIT_MARGIN)
        center = float(np.clip(pose[joint], lo, hi))
        q = pose.copy()
        q[joint] = center
        floor = min(_CLEARANCE, clearance(q, joint // 4) - 0.0005)
        ends = []
        for end in (lo, hi):
            last = center
            for value in np.linspace(center, end, 40):
                q[joint] = value
                if clearance(q, joint // 4) < floor:
                    break
                last = float(value)
            ends.append(last)
        return ends

    opposed, bent = _HOLD.copy(), _HOLD.copy()
    opposed[:4], bent[:4] = _THUMB_OPPOSED, _THUMB_BENT
    jobs = [(j, "hold", _HOLD) for j in range(20)]
    jobs += [(j, "thumb opposed", opposed) for j in (0, 2, 3)] + [(1, "thumb bent", bent)]
    sweeps = []
    for joint, posture, pose in jobs:
        if joint not in skip:
            lo, hi = trimmed(pose, joint)
            sweeps.append(dict(joint=joint, posture=posture, pose=pose.copy(), lo=lo, hi=hi))
    return sweeps


class SweepAbort(RuntimeError):
    pass


@dataclass
class SweepResult:
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    current: list = field(default_factory=list)
    legs: list = field(default_factory=list)  # {segment, joint, direction, start, end, reached}
    aborted: str | None = None


class _Mover:
    def __init__(self, hands, side, result, rate_hz, check):
        self.hands, self.side, self.result = hands, side, result
        self.period, self.check = 1.0 / rate_hz, check
        self.target = np.array(hands.positions()[side], dtype=float)
        self._over_since = None

    def tick(self, target, checked=True):
        """Send ``target``, record the feedback, return the measured angles."""
        self.hands.command(self.side, target)
        self.target = np.array(target, dtype=float)
        q, current = self.hands.feedback(self.side)
        now = time.monotonic()
        self.result.t.append(now)
        self.result.q.append(q)
        self.result.current.append(current)
        if checked:
            age = self.hands.feedback_age(self.side)
            if age is None or age > FEEDBACK_TIMEOUT_S:
                raise SweepAbort("joint feedback stopped")
            if self.check is not None:
                self.check()
            worst = int(np.argmax(np.abs(current)))
            if abs(current[worst]) > MAX_CURRENT:
                self._over_since = self._over_since or now
                if now - self._over_since > OVER_S:
                    raise SweepAbort(f"{current[worst]:+.2f} A on {joint_label(worst)}")
            else:
                self._over_since = None
        return q

    def _ramp(self, step, duration):
        """Call ``step(fraction)`` at the rate until the fraction reaches 1; False stops it."""
        begin, ticks = time.monotonic(), 0
        while True:
            fraction = min((time.monotonic() - begin) / max(duration, 1e-3), 1.0)
            if step(fraction) is False:
                return False
            if fraction >= 1.0:
                return True
            ticks += 1
            time.sleep(max(0.0, begin + ticks * self.period - time.monotonic()))

    def move(self, goal, speed, checked=True):
        start = self.target.copy()

        def step(fraction):
            self.tick(start + (goal - start) * fraction, checked)

        self._ramp(step, np.max(np.abs(goal - start)) / speed)

    def hold(self, goal, seconds):
        def step(_):
            self.tick(goal)

        self._ramp(step, seconds)

    def leg(self, joint, goal, speed):
        """Move one joint; False if it lagged so far that this direction ended early."""
        start, over_since = self.target.copy(), [None]

        def step(fraction):
            target = start.copy()
            target[joint] = start[joint] + (goal - start[joint]) * fraction
            q = self.tick(target)
            if abs(target[joint] - q[joint]) <= BLOCKED:
                over_since[0] = None
                return True
            now = time.monotonic()
            over_since[0] = over_since[0] or now
            if now - over_since[0] > BLOCKED_S:
                target[joint] = q[joint]  # release the push at once
                self.tick(target)
                return False
            return True

        return self._ramp(step, abs(goal - start[joint]) / speed)


def sweep(hands, side, sweeps, *, speed=SPEED, rate_hz=RATE_HZ, check=None):
    """Run the sweeps on a connected, enabled hand and return to the first sweep's pose.

    ``check()`` may raise SweepAbort (a new fault, say). An abort also returns to that pose,
    unless the feedback is gone, and is reported in ``aborted``.
    """
    result = SweepResult()
    mover = _Mover(hands, side, result, rate_hz, check)
    home = sweeps[0]["pose"]
    try:
        mover.move(home, speed)
        for n, plan in enumerate(sweeps):
            joint = plan["joint"]
            print(f"{n + 1}/{len(sweeps)} {joint_label(joint)} ({plan['posture']}) "
                  f"{plan['lo']:+.2f}..{plan['hi']:+.2f}", flush=True)
            mover.move(plan["pose"], speed)
            mover.hold(plan["pose"], _PAUSE_S)
            for goal, direction in ((plan["lo"], "to_lo"), (plan["hi"], "up"),
                                    (plan["lo"], "down"), (plan["pose"][joint], "back")):
                begin = time.monotonic()
                reached = mover.leg(joint, goal, speed)
                result.legs.append(dict(segment=n, joint=joint, direction=direction, start=begin,
                                        end=time.monotonic(), reached=reached))
                if not reached:
                    print(f"   {direction}: stopped at {mover.target[joint]:+.2f} rad", flush=True)
            mover.hold(plan["pose"], _PAUSE_S)
        mover.move(home, speed)
    except SweepAbort as exc:
        result.aborted = str(exc)
        if "feedback" not in result.aborted:
            mover.move(home, speed, checked=False)
    return result


def fit(t, q, current, legs):
    """A ``JointLoad`` from recorded sweeps. Sweeps of one joint in several postures merge."""
    t, q, current = np.asarray(t), np.asarray(q, float), np.asarray(current, float)
    rows = {}
    for segment in sorted({leg["segment"] for leg in legs}):
        ours = {leg["direction"]: leg for leg in legs if leg["segment"] == segment}
        joint = ours["up"]["joint"]
        data = {}
        for direction in ("up", "down"):
            leg = ours[direction]
            window = (t >= leg["start"] + 0.1) & (t <= leg["end"] - 0.05)
            data[direction] = (q[window, joint], current[window, joint])
        if not (len(data["up"][0]) and len(data["down"][0])):
            continue
        low = np.floor(min(a.min() for a, _ in data.values()) / _BIN) * _BIN
        high = np.ceil(max(a.max() for a, _ in data.values()) / _BIN) * _BIN
        edges = np.arange(low, high + _BIN / 2, _BIN)
        for a, b in zip(edges[:-1], edges[1:]):
            up = (data["up"][0] >= a) & (data["up"][0] < b)
            down = (data["down"][0] >= a) & (data["down"][0] < b)
            if up.sum() >= _MIN_SAMPLES and down.sum() >= _MIN_SAMPLES:
                rising, falling = np.median(data["up"][1][up]), np.median(data["down"][1][down])
                rows.setdefault(joint, []).append((round((a + b) / 2, 2), (rising + falling) / 2,
                                                   (rising - falling) / 2))
    curves, friction = {}, np.zeros(20)
    for joint, values in rows.items():
        values = np.array(values)
        angles = np.unique(values[:, 0])
        curves[joint] = (angles, [values[values[:, 0] == a, 1].mean() for a in angles])
        friction[joint] = np.median(values[:, 2])
    return JointLoad(curves, friction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("side", choices=("left", "right"))
    parser.add_argument("--plan", action="store_true", help="Print the sweeps; do not connect")
    args = parser.parse_args(argv)
    if subprocess.run(["pgrep", "-f", "wuji_teleop[.]hardware_teleop"],
                      capture_output=True).returncode == 0:
        sys.exit("The teleop is running; close it first.")

    from .hardware_teleop import DISABLED_JOINTS, WujiHands
    from .vega_bimanual_sim import load_bimanual_scene

    hands = WujiHands(current_limit=CURRENT_LIMIT)
    hands.discover()
    serial = hands.serials[args.side]
    if serial is None or hands.addresses[args.side] is None:
        sys.exit(f"{args.side} hand not found by Wuji discovery")
    model, data, _, models = load_bimanual_scene()
    sweeps = plan_sweeps(model, data, models[args.side], args.side,
                         skip=DISABLED_JOINTS.get(serial, {}))
    for plan in sweeps:
        print(f"  {joint_label(plan['joint']):10} {plan['posture']:14} "
              f"{plan['lo']:+.2f}..{plan['hi']:+.2f}")
    seconds = sum(2 * (p["hi"] - p["lo"]) + 2 * abs(p["pose"][p["joint"]] - p["lo"])
                  for p in sweeps) / SPEED + len(sweeps)
    print(f"{args.side} {serial}: {len(sweeps)} sweeps, about {seconds / 60:.1f} min")
    if args.plan:
        return 0

    problems = []

    def on_diagnostics(frame):
        for entry in frame.joints:
            if entry.mcu_temp_c_fb > MAX_TEMP_C:
                problems.append(f"MCU temperature {entry.mcu_temp_c_fb:.1f} C")
            joint = hands.joint_index(entry.nid)
            if entry.error_code_current and joint not in hands.faults[args.side]:
                problems.append(f"fault {int(entry.error_code_current):#06x} on "
                                f"{joint_label(joint)}")

    def check():
        if problems:
            raise SweepAbort(problems[0])

    watch = None
    try:
        hands.connect(args.side)
        hands.enable(args.side, hands.wait_position(args.side))
        faulted = hands.faults[args.side]
        if faulted:
            print("left disabled:", ", ".join(joint_label(j) for j in sorted(faulted)))
        watch = hands.watch_diagnostics(args.side, on_diagnostics)
        result = sweep(hands, args.side, [p for p in sweeps if p["joint"] not in faulted],
                       check=check)
    finally:
        if watch is not None:
            watch.close()
        hands.close()
    if result.aborted:
        print(f"Aborted: {result.aborted}; table not written", file=sys.stderr)
        return 1
    TABLE_DIR.mkdir(exist_ok=True)
    path = TABLE_DIR / f"{serial}.json"
    fit(result.t, result.q, result.current, result.legs).save(path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
