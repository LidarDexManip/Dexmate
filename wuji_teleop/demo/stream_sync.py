"""Put a live stereo recording on one clock with the gloves and the trackers.

``record.py --stereo`` leaves three kinds of time stamps in a session:

- **Gloves and palm IMUs**: the Wuji SDK stamp, already host UTC.
- **Trackers and head pose**: XRoboToolkit stamps each frame on the headset
  (``sdk_ns``, headset clock; about 0.21 s ahead of this laptop on
  2026-11-12). The laptop also logs when each frame arrived. Arrivals are only
  ever late, so the lower envelope of (arrival - stamp), fitted as a line to
  follow clock drift, maps stamps to host UTC with the stamps' low jitter.
- **Stereo frames**: only the arrival of each frame's packet here; the headset
  sends no capture time. The capture time is the arrival minus a latency
  ``L`` (exposure, encoding, Wi-Fi), taken as constant.

``L`` is found from head motion. Between consecutive frames, optical flow on
the left image gives the camera's rotation; the head pose gives the head's
rotation over the same interval, shifted by ``L``. The ``L`` whose head
rotation correlates best with the image rotation wins. Shaking the head at the
start of a recording (``record.py`` asks for it) makes the peak sharp. The
same pairs give the camera's rotation in the head frame (hand-eye, rotation
only), which checks the camera axes and serves for the camera pose.

Writes ``processed/synced.npz`` (every stream sampled at each frame's capture
time), ``processed/sync_report.json`` and ``processed/sync_check.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from . import hands as demo_hands
from . import log as demo_log
from .headset import XR_CAMERA_TO_CV, to_robot_world, xr_matrix
from ..xr_tracker_source import assign_tracker_sides, operator_heading, xr_pose_to_robot

SIDES = ("left", "right")
LATENCY_SEARCH = (-0.1, 0.6)  # seconds
MIN_PAIR_ROTATION_DEG = 0.3   # pairs used for the camera-in-head rotation
DISPARITY_SCALE = 64.0


# ---------------------------------------------------------------------- clocks

def read_frames(stereo_dir):
    """(frame, packet arrival s, decoded s) from ``frames.csv``, host UTC."""
    rows = np.loadtxt(Path(stereo_dir) / "frames.csv", delimiter=",", skiprows=1, dtype=np.int64,
                      ndmin=2)
    return rows[:, 0], rows[:, 1] / 1e9, rows[:, 2] / 1e9


def xr_host_times(arrival, sdk_ns, window=5.0):
    """Host UTC of each XRoboToolkit frame from its headset stamp.

    The lower envelope of (arrival - stamp) over ``window``-second bins is
    fitted with a line (offset and drift). Rows without a stamp keep their
    arrival time.
    """
    arrival = np.asarray(arrival, float)
    stamp = np.asarray(sdk_ns, np.int64) / 1e9
    have = np.asarray(sdk_ns) > 0
    out = arrival.copy()
    if have.sum() < 10:
        return out, {"method": "arrival (no headset stamps)"}
    s, d = stamp[have], arrival[have] - stamp[have]
    s0 = float(s[0])
    bins = np.floor((s - s0) / window).astype(int)
    xs, ys = [], []
    for b in np.unique(bins):
        k = np.flatnonzero(bins == b)
        j = k[np.argmin(d[k])]
        xs.append(s[j] - s0)
        ys.append(d[j])
    xs, ys = np.asarray(xs), np.asarray(ys)
    slope, offset = (np.polyfit(xs, ys, 1) if len(xs) >= 3 else (0.0, float(ys.min())))
    out[have] = s + offset + slope * (s - s0)
    late = arrival[have] - out[have]
    return out, {"method": "headset stamp + lower envelope of arrival - stamp",
                 "offset_s": float(offset), "drift_ppm": float(slope * 1e6),
                 "arrival_after_mapped_ms_median": float(np.median(late) * 1e3),
                 "arrival_after_mapped_ms_p95": float(np.percentile(late, 95) * 1e3)}


def frame_arrival_report(t):
    """Rate and arrival jitter of the frames, against a regular frame clock."""
    if len(t) < 3:
        return {"frames": int(len(t))}
    d = np.diff(t)
    period = float(np.median(d))
    i = np.arange(len(t))
    r = t - period * i
    # Lower-envelope line through per-second minima (frames are only ever late).
    win = max(int(round(1.0 / period)), 2)
    idx = np.array([k + np.argmin(r[k:k + win]) for k in range(0, len(r), win)])
    if len(idx) >= 3:
        slope, icpt = np.polyfit(i[idx], r[idx], 1)
    else:
        slope, icpt = 0.0, float(r.min())
    late = r - (icpt + slope * i)
    return {"frames": int(len(t)), "fps": float(1.0 / (period + slope)),
            "interval_ms_median": period * 1e3, "interval_ms_p95": float(np.percentile(d, 95) * 1e3),
            "interval_ms_max": float(d.max() * 1e3),
            "late_vs_regular_clock_ms_median": float(np.median(late) * 1e3),
            "late_vs_regular_clock_ms_p95": float(np.percentile(late, 95) * 1e3),
            "late_vs_regular_clock_ms_max": float(late.max() * 1e3),
            "gaps_over_1_5_intervals": int((d > 1.5 * period).sum())}


# ------------------------------------------------------------- image rotation

def _kabsch(a, b):
    """R minimizing sum |b - R a|^2 (rows are vectors)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ D @ vt


def _bearings(points, K):
    p = np.column_stack([points, np.ones(len(points))]) @ np.linalg.inv(K).T
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def pair_rotation(gray0, gray1, K, min_points=30):
    """Camera rotation from frame 0 to frame 1 in frame 0's axes (rotvec), and the
    number of tracks used; None if too few. Assumes the scene is far compared
    with the camera's motion between frames (a turning head)."""
    import cv2

    p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < min_points:
        return None, 0
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - p0, axis=2)[:, 0] < 1.0)
    if good.sum() < min_points:
        return None, int(good.sum())
    b0, b1 = _bearings(p0[good, 0], K), _bearings(p1[good, 0], K)
    keep = np.ones(len(b0), bool)
    for _ in range(3):  # trim tracks that move unlike the rest (hands, near objects)
        R = _kabsch(b0[keep], b1[keep])
        err = np.linalg.norm(b1 - b0 @ R.T, axis=1)
        keep = err <= max(np.percentile(err, 70), 1e-4)
        if keep.sum() < min_points:
            return None, int(keep.sum())
    R = _kabsch(b0[keep], b1[keep])
    # Tracks that do not agree on one rotation (lost in fast motion, or a near
    # scene) make the estimate unreliable: more than 1 px median misfit.
    misfit_px = np.median(np.linalg.norm(b1[keep] - b0[keep] @ R.T, axis=1)) * K[0, 0]
    if misfit_px > 1.0:
        return None, int(keep.sum())
    # b1 = C1^T C0 b0, so the camera's own rotation C0^T C1 is R^T.
    return Rotation.from_matrix(R.T).as_rotvec(), int(keep.sum())


def image_rotations(left_dir, frames, K, *, half_resolution=True, progress=None):
    """(P,3) rotvecs between frames[i] and frames[i+1] (NaN when unknown), (P,) tracks."""
    import cv2

    flag = cv2.IMREAD_REDUCED_GRAYSCALE_2 if half_resolution else cv2.IMREAD_GRAYSCALE
    Ks = K.copy()
    if half_resolution:
        Ks[:2] /= 2.0
    out = np.full((max(len(frames) - 1, 0), 3), np.nan)
    tracks = np.zeros(len(out), int)
    prev = None
    for n, f in enumerate(frames):
        img = cv2.imread(str(Path(left_dir) / f"{int(f):06d}.jpg"), flag)
        if prev is not None and img is not None:
            rv, count = pair_rotation(prev, img, Ks)
            tracks[n - 1] = count
            if rv is not None:
                out[n - 1] = rv
        prev = img
        if progress and n % 200 == 0:
            progress(n, len(frames))
    return out, tracks


# --------------------------------------------------------------------- latency

class HeadOrientation:
    """Slerp of the head's OpenXR orientation over host time, with gap checks."""

    def __init__(self, t, quat_xyzw, max_gap=0.1):
        q = np.asarray(quat_xyzw, float)
        good = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
        t, q = np.asarray(t, float)[good], q[good]
        order = np.argsort(t, kind="stable")
        t, q = t[order], q[order]
        keep = np.r_[True, np.diff(t) > 0]
        self.t, self.max_gap = t[keep], max_gap
        self.slerp = Slerp(self.t, Rotation.from_quat(q[keep]))

    def valid(self, times):
        return demo_hands._gaps(self.t, np.asarray(times, float)) <= self.max_gap

    def at(self, times):
        times = np.clip(np.asarray(times, float), self.t[0], self.t[-1])
        return self.slerp(times)

    def relative(self, t0, t1):
        """Rotvec of the head's rotation from t0 to t1 in the head frame at t0."""
        return (self.at(t0).inv() * self.at(t1)).as_rotvec()


def latency_curve(t0, t1, image_rotvec, head, search=LATENCY_SEARCH, step=0.002):
    """Pearson correlation of image and head rotation angles for each candidate latency."""
    lags = np.arange(search[0], search[1] + step / 2, step)
    angle_img = np.linalg.norm(image_rotvec, axis=1)
    ok_img = np.isfinite(angle_img)
    corr = np.full(len(lags), np.nan)
    used = np.zeros(len(lags), int)
    for k, lag in enumerate(lags):
        a, b = t0 - lag, t1 - lag
        ok = ok_img & head.valid(a) & head.valid(b)
        if ok.sum() < 20:
            continue
        angle_head = np.linalg.norm(head.relative(a[ok], b[ok]), axis=1)
        x, y = angle_img[ok], angle_head
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        corr[k] = float(np.corrcoef(x, y)[0, 1])
        used[k] = int(ok.sum())
    return lags, corr, used


def best_latency(lags, corr):
    """Peak of the curve, refined with a parabola through its neighbours."""
    if not np.isfinite(corr).any():
        return None, None
    k = int(np.nanargmax(corr))
    lag = float(lags[k])
    if 0 < k < len(lags) - 1 and np.isfinite(corr[k - 1:k + 2]).all():
        y0, y1, y2 = corr[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag += float(0.5 * (y0 - y2) / den * (lags[1] - lags[0]))
    return lag, float(corr[k])


def camera_in_head(image_rotvec, head_rotvec):
    """R_head_camera from paired rotations (head = R camera), the residual (deg), pairs used."""
    ok = np.isfinite(image_rotvec).all(axis=1) & np.isfinite(head_rotvec).all(axis=1)
    big = ok & (np.rad2deg(np.linalg.norm(image_rotvec, axis=1)) > MIN_PAIR_ROTATION_DEG)
    if big.sum() < 10:
        return None, None, int(big.sum())
    use = big.copy()
    for _ in range(3):  # drop pairs the fit does not explain (failed flow, timing outliers)
        R = _kabsch(image_rotvec[use], head_rotvec[use])
        resid = np.rad2deg(np.linalg.norm(head_rotvec - image_rotvec @ R.T, axis=1))
        use = big & (resid <= max(3.0 * np.median(resid[use]), 0.1))
        if use.sum() < 10:
            return None, None, int(use.sum())
    R = _kabsch(image_rotvec[use], head_rotvec[use])
    resid = np.rad2deg(np.linalg.norm(head_rotvec[use] - image_rotvec[use] @ R.T, axis=1))
    return R, float(np.median(resid)), int(use.sum())


# ---------------------------------------------------------------------- streams

def xr_streams(pc_dir):
    """Host times of every XRoboToolkit row, the raw head poses, and the clock report."""
    xr = demo_log.load(pc_dir, "xr")
    if not len(xr):
        raise SystemExit(f"No XRoboToolkit log in {pc_dir}")
    t, report = xr_host_times(xr["t"], xr["sdk_ns"])
    return xr, t, report


def strictly_increasing(t, *arrays):
    """Sort by time and keep the last row of equal times (rows can share a headset stamp)."""
    t = np.asarray(t, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.r_[np.diff(t) > 0, True]
    return (t[keep],) + tuple(np.asarray(a)[order][keep] for a in arrays)


def tracker_series(xr, t, serial_index):
    rows, poses = [], []
    for n, row in enumerate(xr):
        slots = np.flatnonzero(row["serial"][:row["count"]] == serial_index)
        if len(slots):
            rows.append(n)
            poses.append(row["tracker"][slots[0]])
    if not rows:
        return np.zeros(0), np.zeros((0, 7))
    return strictly_increasing(t[np.asarray(rows)], np.asarray(poses))


def rotation_speed(t, quat_xyzw, grid):
    """Rotation speed (rad/s) on a time grid, NaN outside the samples."""
    head = HeadOrientation(t, quat_xyzw)
    r = head.at(grid)
    w = np.r_[(r[:-1].inv() * r[1:]).magnitude() / np.diff(grid), np.nan]
    w[~head.valid(grid)] = np.nan
    return w


def match_trackers_to_gloves(trackers, imus, step=0.02):
    """{side: serial} pairing each glove with the tracker that turns with its palm IMU.

    ``trackers`` is {serial: (t, raw7)}, ``imus`` {side: (t, Rotation)}. A forearm
    tracker and the palm IMU on the same arm rotate together most of the time,
    so their rotation speeds correlate (about 0.8 against 0.2 on 2026-11-12).
    Returns the pairing and the correlation table; {} when it is not clear.
    """
    serials = [s for s, (ts, _) in trackers.items() if len(ts) > 20]
    sides = [side for side in SIDES if side in imus and len(imus[side][0]) > 20]
    if not serials or not sides:
        return {}, {}
    lo = max([trackers[s][0][0] for s in serials] + [imus[side][0][0] for side in sides])
    hi = min([trackers[s][0][-1] for s in serials] + [imus[side][0][-1] for side in sides])
    if hi - lo < 5.0:
        return {}, {}
    grid = np.arange(lo, hi, step)
    speed_imu = {side: rotation_speed(imus[side][0], imus[side][1].as_quat(), grid) for side in sides}
    table = {}
    for serial in serials:
        ts, raw = trackers[serial]
        w = rotation_speed(ts, raw[:, 3:], grid)
        for side in sides:
            ok = np.isfinite(w) & np.isfinite(speed_imu[side])
            table[(serial, side)] = float(np.corrcoef(w[ok], speed_imu[side][ok])[0, 1]) if ok.sum() > 100 else np.nan
    import itertools
    best, best_score, runner_up = {}, -np.inf, -np.inf
    for chosen in itertools.permutations(serials, len(sides)):
        score = np.nansum([table[(serial, side)] for serial, side in zip(chosen, sides)])
        if score > best_score:
            runner_up, best_score, best = best_score, score, dict(zip(sides, chosen))
        elif score > runner_up:
            runner_up = score
    weakest = min(table[(best[side], side)] for side in sides)
    if not np.isfinite(weakest) or weakest < 0.4 or best_score - runner_up < 0.3:
        return {}, table
    return best, table


def assign_sides(trackers, head_t, head_raw, at, left=None, right=None):
    """{side: serial} by position at time ``at`` relative to the operator's heading."""
    poses = {}
    for serial, (ts, raw) in trackers.items():
        if len(ts):
            j = int(np.argmin(np.abs(ts - at)))
            if np.isfinite(raw[j]).all() and np.linalg.norm(raw[j][3:]) > 0.5:
                poses[serial] = xr_pose_to_robot(raw[j] / np.r_[1, 1, 1, [np.linalg.norm(raw[j][3:])] * 4])
    head = None
    if len(head_t):
        j = int(np.argmin(np.abs(head_t - at)))
        q = head_raw[j][3:]
        if np.linalg.norm(q) > 0.5:
            head = xr_pose_to_robot(head_raw[j] / np.r_[1, 1, 1, [np.linalg.norm(q)] * 4])
    if not poses:
        return {}
    if len(poses) == 1 and not (left or right):
        return {}
    if len(poses) > 2 and not (left and right):
        raise SystemExit(f"{len(poses)} trackers recorded ({sorted(poses)}); give --left-tracker and "
                         "--right-tracker")
    yaw = operator_heading(head, tuple(poses.values())[:2])
    return assign_tracker_sides(poses, yaw=yaw, left_serial=left, right_serial=right)


def _rate(t):
    t = np.unique(np.asarray(t, float))
    return float(1.0 / np.median(np.diff(t))) if len(t) > 2 else None


def _gap_stats(t_src, t_query):
    if len(t_src) < 2:
        return {"valid_frames": 0.0}
    g = demo_hands._gaps(np.unique(t_src), t_query)
    fin = g[np.isfinite(g)]
    return {"valid_frames": float(np.mean(g <= 0.1)),
            "bracket_ms_median": float(np.median(fin) * 1e3) if len(fin) else None,
            "bracket_ms_max": float(fin.max() * 1e3) if len(fin) else None}


# ------------------------------------------------------------------------- plot

def plot_check(path, t_mid, angle_img, angle_head, lags, corr, latency, peak, period):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    head_c, img_c = "#2a78d6", "#eb6834"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), gridspec_kw={"height_ratios": [3, 2]},
                                   facecolor=surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    rate_img = np.rad2deg(angle_img) / period
    rate_head = np.rad2deg(angle_head) / period
    ok = np.isfinite(rate_img) & np.isfinite(rate_head)
    t0 = t_mid[ok][0] if ok.any() else 0.0
    # Show the 20 s with the most motion.
    tt = t_mid - t0
    if ok.any() and tt[ok][-1] > 20:
        energy = np.convolve(np.nan_to_num(rate_head), np.ones(int(20 / period)), "same")
        c = tt[int(np.argmax(energy))]
        lo, hi = max(c - 10, 0), c + 10
    else:
        lo, hi = 0, tt[ok][-1] if ok.any() else 1
    sel = ok & (tt >= lo) & (tt <= hi)
    ax1.plot(tt[sel], rate_head[sel], color=head_c, linewidth=2, label="head pose (PC Service)")
    ax1.plot(tt[sel], rate_img[sel], color=img_c, linewidth=2, label="left image (optical flow)")
    if sel.any():
        last = np.flatnonzero(sel)[-1]
        ax1.annotate("head", (tt[last], rate_head[last]), xytext=(6, 0), textcoords="offset points",
                     color=ink, fontsize=9, va="center")
        ax1.annotate("image", (tt[last], rate_img[last]), xytext=(6, -10), textcoords="offset points",
                     color=ink, fontsize=9, va="center")
    ax1.set_ylabel("rotation speed (deg/s)", color=ink2, fontsize=9)
    ax1.set_xlabel("time from first frame (s)", color=ink2, fontsize=9)
    ax1.legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper right")
    ax1.set_title(f"Head vs camera rotation, frames shifted {latency * 1e3:.0f} ms earlier",
                  color=ink, fontsize=11, loc="left")
    ax2.plot(lags * 1e3, corr, color=head_c, linewidth=2)
    ax2.axvline(latency * 1e3, color=ink2, linewidth=1, linestyle="--")
    ax2.annotate(f"latency {latency * 1e3:.0f} ms, r = {peak:.2f}", (latency * 1e3, peak),
                 xytext=(8, -14), textcoords="offset points", color=ink, fontsize=9)
    ax2.set_xlabel("frame arrival minus capture (ms)", color=ink2, fontsize=9)
    ax2.set_ylabel("correlation", color=ink2, fontsize=9)
    ax2.set_title("Match of head and image rotation per candidate latency", color=ink, fontsize=11,
                  loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------------- sync

def sync_session(session, args):
    """The ``sync`` stage: see the module docstring."""
    stereo = session.root / "stereo"
    smeta = json.loads((stereo / "meta.json").read_text())
    meta = session.meta()
    events = session.events()
    frame, t_packet, _ = read_frames(stereo)
    if len(frame) < 20:
        raise SystemExit(f"Only {len(frame)} stereo frames in {stereo}")
    K = np.asarray(smeta["K_left_nominal"], float)
    eye_w, eye_h = smeta["eye_size"]
    arrival = frame_arrival_report(t_packet)
    period = 1.0 / arrival["fps"]
    print(f"stereo: {len(frame)} frames at {arrival['fps']:.1f} fps; arrival jitter p95 "
          f"{arrival['late_vs_regular_clock_ms_p95']:.0f} ms, {arrival['gaps_over_1_5_intervals']} gaps")

    xr, t_xr, xr_clock = xr_streams(session.pc)
    print(f"trackers/head: headset clock -> host {xr_clock.get('offset_s', 0) * 1e3:+.0f} ms, drift "
          f"{xr_clock.get('drift_ppm', 0):.0f} ppm")
    head_ok = np.linalg.norm(xr["headset"][:, 3:], axis=1) > 0.5
    if head_ok.sum() < 20:
        raise SystemExit("No head pose in the XRoboToolkit log (Head tracking off in the app?)")
    head = HeadOrientation(t_xr[head_ok], xr["headset"][head_ok, 3:])

    # Latency from head motion.
    t0, t1 = t_packet[:-1], t_packet[1:]
    print("optical flow between frames ...", flush=True)
    rot_img, tracks = image_rotations(stereo / "left", frame, K,
                                      progress=lambda n, total: print(f"  {n}/{total}", flush=True))
    lags, corr, used = latency_curve(t0, t1, rot_img, head)
    latency, peak = best_latency(lags, corr)
    if args.latency is not None:
        latency_source = "given (--latency)"
        latency = args.latency
    else:
        latency_source = "head motion"
        if latency is None:
            raise SystemExit("No overlap of head motion and image motion; was the head moving?")
    width = None
    if peak is not None:
        above = np.flatnonzero(corr >= peak - 0.05)
        width = float((lags[above[-1]] - lags[above[0]]) * 1e3) if len(above) else None
    print(f"stream latency {latency * 1e3:.0f} ms ({latency_source}; peak correlation "
          f"{peak if peak is not None else float('nan'):.2f}, within 0.05 of the peak over {width} ms)")
    if peak is not None and peak < 0.6 and args.latency is None:
        raise SystemExit("The image and head rotations do not match (correlation "
                         f"{peak:.2f} < 0.6), so the stream latency is unknown. Was the headset worn and "
                         "the head turning (the headshake step)? Pass --latency SECONDS to sync anyway.")
    head_rot = np.full_like(rot_img, np.nan)
    a, b = t0 - latency, t1 - latency
    ok = head.valid(a) & head.valid(b)
    head_rot[ok] = head.relative(a[ok], b[ok])
    R_hc, resid, n_pairs = camera_in_head(rot_img, head_rot)
    angle_img = np.linalg.norm(rot_img, axis=1)
    angle_head = np.linalg.norm(head_rot, axis=1)
    both = np.isfinite(angle_img) & np.isfinite(angle_head) & (np.rad2deg(angle_head) > MIN_PAIR_ROTATION_DEG)
    scale = float(np.sum(angle_img[both] * angle_head[both]) / np.sum(angle_head[both] ** 2)) \
        if both.sum() > 10 else None
    if R_hc is not None:
        off_nominal = float(np.rad2deg(Rotation.from_matrix(XR_CAMERA_TO_CV.T @ R_hc).magnitude()))
        print(f"camera rotation in the head frame: {off_nominal:.1f} deg from the nominal "
              f"(OpenXR head -> OpenCV camera), residual {resid:.2f} deg over {n_pairs} pairs; "
              + (f"image/head rotation ratio {scale:.3f} (1 if the nominal focal length is right)"
                 if scale is not None else "image/head rotation ratio unknown (too little head turning)"))
    use_estimated = R_hc is not None and resid is not None and resid < 0.5 and not args.nominal_camera
    R_head_cam = R_hc if use_estimated else XR_CAMERA_TO_CV.copy()

    # Every stream at each frame's capture time.
    t_cap = t_packet - latency
    out = {"frame": frame, "t_capture": t_cap, "t_packet": t_packet, "latency_s": np.array(latency),
           "K": K, "image_size": np.array([eye_w, eye_h]),
           "baseline_m": np.array(args.baseline_m if args.baseline_m else (smeta.get("baseline_m") or np.nan)),
           "disparity_scale": np.array(smeta.get("disparity_scale", DISPARITY_SCALE))}
    head_raw = xr["headset"]
    T_head = np.array([to_robot_world(xr_matrix(r)) if ok_ else np.full((4, 4), np.nan)
                       for r, ok_ in zip(head_raw, head_ok)])
    T_world_head, head_valid = demo_hands.interp_poses(*strictly_increasing(t_xr, T_head), t_cap, 0.1)
    T_head_cam = np.eye(4)
    T_head_cam[:3, :3] = R_head_cam
    camera_offset = np.asarray(getattr(args, "camera_offset", None) or (0.0, 0.0, 0.0), float)
    T_head_cam[:3, 3] = camera_offset
    T_world_camera = np.full_like(T_world_head, np.nan)
    T_world_camera[head_valid] = T_world_head[head_valid] @ T_head_cam
    out.update({"T_world_head": T_world_head, "head_valid": head_valid, "T_head_camera": T_head_cam,
                "T_world_camera": T_world_camera})
    coverage = {"head": _gap_stats(t_xr[head_ok], t_cap)}

    serials = meta.get("tracker_serials", [])
    trackers = {s: tracker_series(xr, t_xr, i) for i, s in enumerate(serials)}
    at = events.get("still_end", events.get("demo_start", float(np.median(t_cap))))
    imus = {side: demo_hands.imu_series(session.pc, side) for side in SIDES
            if side in meta.get("gloves", {})}
    matched, table = ({}, {}) if (args.left_tracker or args.right_tracker) else \
        match_trackers_to_gloves(trackers, imus)
    if matched:
        sides, how = matched, "turns with the palm IMU of that glove"
    else:
        sides = assign_sides(trackers, t_xr[head_ok], head_raw[head_ok], at, args.left_tracker,
                             args.right_tracker)
        how = "given" if (args.left_tracker or args.right_tracker) else "position relative to the headset's heading"
    print(f"trackers: {sides if sides else 'none assigned'} ({how})"
          + ("; rotation-speed correlation " + ", ".join(f"{s[-6:]}~{side} {c:.2f}" for (s, side), c in table.items())
             if table else ""))
    calibration = {}
    for side in SIDES:
        if side in sides:
            ts, raw = trackers[sides[side]]
            T_tr = demo_hands.tracker_poses(raw)
            T_at, v = demo_hands.interp_poses(ts, T_tr, t_cap, 0.1)
            out[f"tracker_{side}"], out[f"tracker_valid_{side}"] = T_at, v
            coverage[f"tracker_{side}"] = _gap_stats(ts, t_cap)
        glove_meta = meta.get("gloves", {}).get(side)
        if glove_meta is None:
            continue
        t_glove, kp, conf = demo_hands.glove_series(session.pc, side)
        t_imu, imu_rot = demo_hands.imu_series(session.pc, side)
        kp_at, vk = demo_hands.interp_array(t_glove, kp, t_cap, 0.1)
        conf_at, _ = demo_hands.interp_array(t_glove, conf, t_cap, 0.1)
        imu_at, vi = demo_hands.interp_rotation(t_imu, imu_rot, t_cap, 0.1)
        quat = np.full((len(t_cap), 4), np.nan)
        if vi.any():
            quat[vi] = Rotation.from_matrix(imu_at[vi]).as_quat()
        out.update({f"glove_keypoints_{side}": kp_at, f"glove_confidence_{side}": conf_at,
                    f"glove_valid_{side}": vk, f"imu_quat_{side}": quat, f"imu_valid_{side}": vi})
        coverage[f"glove_{side}"] = {"rate_hz": _rate(t_glove), **_gap_stats(t_glove, t_cap)}
        coverage[f"imu_{side}"] = {"rate_hz": _rate(t_imu), **_gap_stats(t_imu, t_cap)}
        if side not in sides:
            continue
        # World-frame hand: tracker position + palm IMU orientation (wrist calibration).
        cal = None
        cal_source = None
        try:
            if {"still_start", "still_end", "rotate_start", "rotate_end"} <= set(events):
                forward = np.nanmedian(kp[:, 9] - kp[:, 0], axis=0)
                forward /= np.linalg.norm(forward)
                cal = demo_hands.calibrate_wrist(
                    ts, T_tr, t_imu, imu_rot, np.asarray(glove_meta["imu_in_wrist_rotation"]), forward,
                    (events["still_start"], events["still_end"]),
                    (events["rotate_start"], events["rotate_end"]),
                    mount_height=args.mount_height, mount_back=args.mount_back)
                cal_source = "this take"
        except ValueError as exc:
            print(f"  {side} wrist calibration failed: {exc}")
        if cal is None and args.calibration_from:
            # Another take's calibration (same trackers on the same straps, gloves not restarted).
            data = json.loads(Path(args.calibration_from).read_text())
            wrists = data.get("wrists") or data.get("wrist_calibration") or {}
            if side in wrists:
                cal = demo_hands.WristCalibration.from_json(wrists[side])
                cal_source = str(args.calibration_from)
                print(f"  {side}: using the wrist calibration from {args.calibration_from}")
        if cal is None:
            print(f"  {side}: no wrist calibration; world-frame hand skipped (local keypoints kept)")
            continue
        hand = demo_hands.world_hand(t_cap, ts, T_tr, t_imu, imu_rot, t_glove, kp, conf, cal,
                                     orientation=args.hand_orientation)
        out.update({f"keypoints_{side}": hand.keypoints, f"wrist_{side}": hand.wrist,
                    f"hand_valid_{side}": hand.valid})
        calibration[side] = {**cal.to_json(), "source": cal_source}
        print(f"  {side} hand in the world in {hand.valid.mean():.0%} of frames")

    disp_dir = stereo / "disparity"
    have_disp = np.array([(disp_dir / f"{int(f):06d}.png").exists() for f in frame])
    out["has_disparity"] = have_disp
    session.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(session.out / "synced.npz", **out)
    report = {
        "frames": int(len(frame)), "frames_with_disparity": int(have_disp.sum()),
        "stream": arrival, "xr_clock": xr_clock,
        "latency": {"seconds": float(latency), "source": latency_source,
                    "peak_correlation": peak, "width_ms_within_0_05": width,
                    "pairs_used": int(used.max()) if len(used) else 0,
                    "pairs_with_image_rotation": int(np.isfinite(rot_img[:, 0]).sum())},
        "camera_in_head": {"R_head_camera": R_head_cam.tolist(), "estimated": bool(use_estimated),
                           "estimate": R_hc.tolist() if R_hc is not None else None,
                           "residual_deg_median": resid, "pairs": n_pairs,
                           "image_over_head_rotation": scale,
                           "translation_m": camera_offset.tolist(),
                           "translation": ("given (--camera-offset; OpenXR head frame: x right, y up, z back)"
                                           if camera_offset.any() else
                                           "zero (camera at the head origin; the left camera actually sits "
                                           "a few cm away, see --camera-offset)")},
        "trackers": sides, "tracker_assignment": how,
        "tracker_glove_rotation_correlation": {f"{s}~{side}": c for (s, side), c in table.items()},
        "coverage": coverage, "wrist_calibration": calibration,
        "layout": {"left": "stereo/left/{frame:06d}.jpg",
                   "disparity": "stereo/disparity/{frame:06d}.png (uint16, 1/64 px, 0 = none)",
                   "depth": "fx * baseline_m / disparity (needs baseline_m)"},
    }
    session.write_json("sync_report.json", report)
    if not args.no_plot and peak is not None:
        plot_check(session.out / "sync_check.png", (t0 + t1) / 2 - t_packet[0], angle_img, angle_head,
                   lags, corr, latency, peak, period)
    for name, c in coverage.items():
        print(f"  {name}: {c.get('valid_frames', 0):.0%} of frames"
              + (f", {c['rate_hz']:.0f} Hz" if c.get("rate_hz") else ""))
    print(f"Wrote {session.out / 'synced.npz'}, sync_report.json"
          + (", sync_check.png" if not args.no_plot and peak is not None else ""))


def load_frame(session_root, index, synced=None):
    """(left BGR, disparity px with NaN, depth m or None) of one frame."""
    import cv2

    root = Path(session_root)
    left = cv2.imread(str(root / "stereo" / "left" / f"{index:06d}.jpg"))
    png = cv2.imread(str(root / "stereo" / "disparity" / f"{index:06d}.png"), cv2.IMREAD_UNCHANGED)
    if png is None:
        return left, None, None
    disparity = png.astype(np.float32) / DISPARITY_SCALE
    disparity[png == 0] = np.nan
    synced = synced if synced is not None else np.load(root / "processed" / "synced.npz")
    baseline = float(synced["baseline_m"])
    if not np.isfinite(baseline):
        return left, disparity, None
    return left, disparity, float(synced["K"][0, 0]) * baseline / disparity
