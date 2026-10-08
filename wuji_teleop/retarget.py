"""Retargeting and input watchdog, independent of the viewer and hardware."""

import numpy as np

from .sources import SourceSnapshot, validate_keypoints


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

    def __init__(self, retargeter, hand, stale_after=0.250):
        self.retargeter = retargeter
        self.hand = hand
        self.stale_after = stale_after
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
                self.hand.set_target(self.retargeter.step(frame.keypoints))
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
