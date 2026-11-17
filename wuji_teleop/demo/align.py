"""Put the headset recording and the laptop log on one clock (host UTC).

The headset writes its trackers in its own UTC clock; the laptop logs the same
trackers as relayed by the XRoboToolkit PC Service, stamped on arrival in host
UTC. Matching the two tracker trajectories gives the offset, network delay
included, so every stream lines up with the gloves' host-synchronized time.
"""

from __future__ import annotations

import numpy as np

from . import log as demo_log


def pc_tracker_series(pc_dir, serial_table):
    """{serial: (t_host, raw7)} from the laptop's XRoboToolkit log."""
    xr = demo_log.load(pc_dir, "xr")
    out = {}
    for index, serial in enumerate(serial_table):
        times, poses = [], []
        for row in xr:
            slots = np.flatnonzero(row["serial"][:row["count"]] == index)
            if len(slots):
                times.append(row["t"])
                poses.append(row["tracker"][slots[0]])
        if times:
            out[serial] = (np.asarray(times), np.asarray(poses))
    return out


def _speed_on_grid(t, xyz, grid):
    good = np.isfinite(xyz).all(axis=1)
    t, xyz = t[good], xyz[good]
    if len(t) < 10:
        return None
    p = np.column_stack([np.interp(grid, t, xyz[:, k]) for k in range(3)])
    inside = (grid >= t[0]) & (grid <= t[-1])
    v = np.linalg.norm(np.gradient(p, grid, axis=0), axis=1)
    v[~inside] = np.nan
    return v


def tracker_offset(headset_t, headset_trackers, pc_trackers, *, search=30.0, step=0.002,
                   guess=0.0):
    """Seconds to add to headset UTC to get host UTC, and a quality report.

    Both clocks are UTC, so the offset is searched within +-``search`` s of
    ``guess`` by cross-correlating tracker speed, then refined by minimizing
    the position difference around the best lag.
    """
    shared = [s for s in headset_trackers if s in pc_trackers]
    if not shared:
        raise ValueError(f"No tracker serial in both recordings (headset {sorted(headset_trackers)}, "
                         f"laptop {sorted(pc_trackers)})")
    coarse = []
    for serial in shared:
        th, ph = headset_t, headset_trackers[serial][:, :3]
        tp, pp = pc_trackers[serial][0], pc_trackers[serial][1][:, :3]
        lags = np.arange(-search, search + step, step * 5)
        grid = np.arange(th[0], th[-1], step * 5)
        vh = _speed_on_grid(th, ph, grid)
        best, best_score = None, -np.inf
        for lag in lags:
            vp = _speed_on_grid(tp - guess - lag, pp, grid)
            if vh is None or vp is None:
                continue
            both = np.isfinite(vh) & np.isfinite(vp)
            if both.sum() < 200:
                continue
            a, b = vh[both] - vh[both].mean(), vp[both] - vp[both].mean()
            score = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            if score > best_score:
                best, best_score = guess + lag, score
        if best is not None:
            coarse.append((best_score, best, serial))
    if not coarse:
        raise ValueError("Tracker trajectories do not overlap; were both recordings running together?")
    score, offset, serial = max(coarse)
    # Refine on positions within +-25 ms at 1 ms.
    th, ph = headset_t, headset_trackers[serial][:, :3]
    tp, pp = pc_trackers[serial][0], pc_trackers[serial][1][:, :3]
    best_err, best_off = np.inf, offset
    for off in offset + np.arange(-0.025, 0.0251, 0.001):
        q = th + off
        good = np.isfinite(ph).all(axis=1) & (q >= tp[0]) & (q <= tp[-1])
        if good.sum() < 50:
            continue
        interp = np.column_stack([np.interp(q[good], tp, pp[:, k]) for k in range(3)])
        err = float(np.sqrt(np.mean(np.sum((interp - ph[good]) ** 2, axis=1))))
        if err < best_err:
            best_err, best_off = err, off
    return float(best_off), {"serial": serial, "speed_correlation": float(score),
                             "position_rms_m": float(best_err),
                             "per_serial_coarse": {s: float(o) for _, o, s in coarse}}
