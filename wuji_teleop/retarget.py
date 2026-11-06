"""Retargeting and input watchdog, independent of the viewer and hardware."""

import math

import numpy as np

from .sources import SourceSnapshot, validate_keypoints

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Glove finger -> the neighbour it copies while its own sensor reads low
# confidence. The right glove's ring sensor failed on 2026-11-06 (no reading in
# about half its frames, the rest near the palm): the skeleton curled the ring
# into the palm, and through the retargeter also bent its neighbours.
FINGER_STAND_INS = {"ring": "pinky"}


def copy_finger(points, finger, donor):
    """Rebuild a finger's three bones along the donor finger's bone directions.

    Keeps the finger's own base point (fixed in the glove's hand model) and its
    bone lengths, so the result is a plausible finger of the right size.
    """
    points = np.array(points, dtype=np.float32)
    base, source = 1 + 4 * finger, 1 + 4 * donor
    lengths = np.linalg.norm(np.diff(points[base:base + 4], axis=0), axis=1)
    bones = np.diff(points[source:source + 4], axis=0)
    donor_lengths = np.linalg.norm(bones, axis=1)
    lengths = np.where(lengths > 1e-3, lengths, donor_lengths)
    points[base + 1:base + 4] = points[base] + np.cumsum(
        bones * (lengths / donor_lengths)[:, None], axis=0)
    return points


class FingerStandIn:
    """Swap in a neighbour for a glove finger whose sensor reads low confidence.

    Each fingertip sensor's confidence (``frame.finger_confidence``, from the
    glove's EMF poses) is smoothed with time constant ``tau``. Once
    it falls below ``low``, the finger is rebuilt from its donor before
    retargeting (the SDK retargeter couples fingers, so a bad finger would also
    bend its neighbours). It returns to its own sensor once its confidence has
    stayed at ``high`` or above for ``recover_s``. A donor that is low itself is
    not copied. ``changes`` collects operator messages for the caller to drain.
    """

    def __init__(self, pairs=None, *, low=0.6, high=0.8, tau=0.3, recover_s=2.0):
        pairs = FINGER_STAND_INS if pairs is None else pairs
        self.pairs = {FINGERS.index(f): FINGERS.index(d) for f, d in pairs.items()}
        self.low, self.high, self.tau, self.recover_s = low, high, tau, recover_s
        self.confidence = None  # smoothed, per finger
        self.weak = set()  # fingers whose own sensor is judged low
        self.copying = frozenset()  # weak fingers rebuilt from their donor
        self.note = ""
        self.changes = []
        self._recovering = {}  # finger -> time its confidence first reached `high`
        self._last = None

    def apply(self, frame):
        values = frame.finger_confidence
        if values is None:
            return frame.keypoints
        values = np.asarray(values, dtype=np.float64)
        now = frame.received_at
        if self.confidence is None:
            self.confidence = values.copy()
        else:
            dt = min(max(now - self._last, 0.0), 1.0)
            self.confidence += (1 - math.exp(-dt / self.tau)) * (values - self.confidence)
        self._last = now
        for finger in self.pairs:
            level = self.confidence[finger]
            if finger not in self.weak:
                if level < self.low:
                    self.weak.add(finger)
            elif level < self.high:
                self._recovering.pop(finger, None)
            elif now - self._recovering.setdefault(finger, now) >= self.recover_s:
                self.weak.discard(finger)
                del self._recovering[finger]
        copying = frozenset(f for f in self.weak
                            if self.confidence[self.pairs[f]] >= self.low)
        self._report(copying)
        points = frame.keypoints
        for finger in sorted(copying):
            points = copy_finger(points, finger, self.pairs[finger])
        return points

    def _report(self, copying):
        def level(f):
            return f"{self.confidence[f]:.2f}"

        for f in sorted(copying - self.copying):
            self.changes.append(f"{FINGERS[f]} sensor confidence {level(f)}: the "
                                f"{FINGERS[f]} follows the {FINGERS[self.pairs[f]]}.")
        for f in sorted(self.copying - copying):
            self.changes.append(
                f"{FINGERS[f]} sensor confidence back to {level(f)}: the {FINGERS[f]} "
                "follows its own sensor again." if f not in self.weak else
                f"{FINGERS[self.pairs[f]]} sensor low too ({level(self.pairs[f])}): "
                f"the {FINGERS[f]} follows its own low sensor.")
        self.copying = copying
        self.note = "; ".join(
            f"{FINGERS[f]} copies {FINGERS[self.pairs[f]]} ({FINGERS[f]} sensor {level(f)})"
            if f in copying else f"{FINGERS[f]} sensor low ({level(f)}), "
            f"{FINGERS[self.pairs[f]]} too" for f in sorted(self.weak))

    def take_changes(self):
        changes, self.changes = self.changes, []
        return changes


class WujiRetargeter:
    def __init__(self, side):
        from wuji_sdk import Handedness, HandModel, RetargetSession
        handedness = {"left": Handedness.Left, "right": Handedness.Right}[side]
        self._session = RetargetSession.for_hand(HandModel.WujiHand2, side=handedness)

    def step(self, points):
        result = np.asarray(self._session.step(validate_keypoints(points)), dtype=np.float64)
        if result.shape != (20,) or not np.isfinite(result).all():
            raise ValueError("Retargeter did not return 20 finite joint angles")
        return result

    def reset(self):
        self._session.reset()


class TrackingController:
    """Hold targets on loss/pause; reset filter state before accepting new data."""

    def __init__(self, retargeter, hand, stale_after=0.250, stand_in=None):
        self.retargeter = retargeter
        self.hand = hand
        self.stale_after = stale_after
        self.stand_in = stand_in  # optional FingerStandIn, applied before retargeting
        self.paused = False
        self.status = "waiting"
        self.detail = "Waiting for the first valid frame"
        self.updates = 0
        self._revision = -1
        self._reset_needed = True

    def toggle_pause(self):
        self.paused = not self.paused
        self._reset_needed = True
        self.status = "paused" if self.paused else "waiting"

    def update(self, snapshot: SourceSnapshot, now: float):
        if self.paused:
            self.status, self.detail = "paused", "Space to resume"
            return False
        frame = snapshot.frame
        if snapshot.error:
            self.status, self.detail = "invalid", snapshot.error
        elif frame is None:
            self.status, self.detail = "waiting", "Waiting for the first valid frame"
        elif now - frame.received_at > self.stale_after:
            self.status, self.detail = "stale", "Holding last target; check glove connection"
        else:
            if snapshot.revision == self._revision:
                return False
            self._revision = snapshot.revision
            try:
                if self._reset_needed:
                    self.retargeter.reset()
                points = (frame.keypoints if self.stand_in is None
                          else self.stand_in.apply(frame))
                self.hand.set_target(self.retargeter.step(points))
            except Exception as exc:
                self.status, self.detail = "invalid", f"Retargeting failed: {exc}"
                self._reset_needed = True
                return False
            self._reset_needed = False
            self.status, self.detail = "tracking", ""
            self.updates += 1
            return True
        self._reset_needed = True
        return False
