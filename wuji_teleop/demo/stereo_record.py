"""Record the PICO stereo stream with Fast-FoundationStereo disparity for every frame.

Runs in Fast-FoundationStereo's environment (torch, OpenCV); ``record.py
--stereo`` starts it next to the glove and tracker log. It writes into
``<session>/stereo/``:

- ``stream.h264`` and ``stream.times.csv``: the raw stream as received, and
  each packet's arrival time (host UTC ns).
- ``frames.csv``: per frame, the arrival time of its picture packet and the
  time it was decoded (host UTC ns). Frame ``i`` is the ``i``-th decoded picture.
- ``left/NNNNNN.jpg``: the left eye (JPEG quality 95; the exact decoded pixels
  can be had again from ``stream.h264``).
- ``disparity/NNNNNN.png``: Fast-FoundationStereo disparity of the left eye in
  1/64 px (uint16, 0 = no match). Depth = fx * baseline / disparity.
- ``meta.json`` at the end, and ``status.json`` every second while recording.

Every frame gets a disparity: frames wait in a queue for the model, and on
stop the recorder finishes the queue before it exits. The headset sends no
capture time; ``process_human_demo.sh sync`` finds the stream's latency from
head motion and puts the frames on the gloves' and trackers' clock.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import numpy as np

from . import headset_stream

DISPARITY_SCALE = 64.0  # PNG value per pixel of disparity
# The field of view the XRoboToolkit app asks PICO's capture library for.
NOMINAL_HFOV_DEG = 76.35
NOMINAL_VFOV_DEG = 61.05
DEFAULT_FFS_DIR = Path.home() / "Desktop" / "Fast-FoundationStereo"
STALL_S = 10.0  # ask the headset again after this long without a new frame
VIEW_WINDOW = "PICO ego camera + depth (recording)"
VIEW_PERIOD_S = 1 / 20  # live view refresh
VIEW_NEAR_M, VIEW_FAR_M = 0.2, 1.5  # depth colour range (hands and table)
VIEW_MAX_DISPARITY = 128.0  # disparity colour range without a baseline


def nominal_K(width, height):
    """Pinhole intrinsics of one eye from the app's nominal field of view."""
    fx = (width / 2.0) / math.tan(math.radians(NOMINAL_HFOV_DEG) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(NOMINAL_VFOV_DEG) / 2.0)
    return np.array([[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def encode_disparity(disparity):
    """float px (NaN = none) -> uint16 in 1/64 px, 0 = none."""
    value = np.nan_to_num(np.asarray(disparity, np.float32), nan=0.0) * DISPARITY_SCALE
    return np.clip(np.round(value), 0, 65535).astype(np.uint16)


def decode_disparity(png):
    out = np.asarray(png, np.float32) / DISPARITY_SCALE
    out[np.asarray(png) == 0] = np.nan
    return out


class FastFoundationStereo:
    """Disparity of the left eye from a rectified BGR pair (NaN where unmatched)."""

    def __init__(self, ffs_dir=DEFAULT_FFS_DIR, checkpoint="23-36-37", iters=8, max_disp=192):
        ffs_dir = Path(ffs_dir).expanduser()
        if str(ffs_dir) not in sys.path:
            sys.path.insert(0, str(ffs_dir))
        import torch
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        self.torch, self.InputPadder, self.dtype = torch, InputPadder, AMP_DTYPE
        path = Path(checkpoint)
        if not path.suffix:
            path = ffs_dir / "weights" / checkpoint / "model_best_bp2_serialize.pth"
        torch.backends.cudnn.benchmark = True
        model = torch.load(str(path), map_location="cpu", weights_only=False)
        model.args.valid_iters = iters
        model.args.max_disp = max_disp
        self.model = model.cuda().eval()
        self.iters = iters
        self.info = {"checkpoint": path.parent.name, "path": str(path), "iters": iters,
                     "max_disp": max_disp}

    def __call__(self, left_bgr, right_bgr):
        torch = self.torch
        h, w = left_bgr.shape[:2]
        with torch.no_grad():
            a = torch.from_numpy(np.ascontiguousarray(left_bgr[..., ::-1])).cuda()
            b = torch.from_numpy(np.ascontiguousarray(right_bgr[..., ::-1])).cuda()
            a = a.permute(2, 0, 1)[None].float()
            b = b.permute(2, 0, 1)[None].float()
            padder = self.InputPadder(a.shape, divis_by=32, force_square=False)
            a, b = padder.pad(a, b)
            with torch.amp.autocast("cuda", enabled=True, dtype=self.dtype):
                d = self.model.forward(a, b, iters=self.iters, test_mode=True,
                                       optimize_build_volume="pytorch1")
            d = padder.unpad(d.float())[0, 0].cpu().numpy()
        d = d.clip(0, None)
        d[np.arange(w)[None, :] - d < 0] = np.nan  # would fall outside the right image
        d[d <= 0] = np.nan
        return d


class StereoRecorder:
    def __init__(self, out, headset, *, size=(1280, 480), fps=30, control_port=headset_stream.CONTROL_PORT,
                 model=None, window=False, baseline_m=None, log=print, reset_pause=1.0):
        import cv2

        self.cv2 = cv2
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "left").mkdir()
        (self.out / "disparity").mkdir()
        self.headset, self.size, self.fps = headset, size, fps
        self.model, self.window, self.baseline_m, self.log = model, window, baseline_m, log
        self.started = time.time()
        self.frames = 0
        self.done = 0
        self.model_ms = []
        self.arrivals = []
        self.error = None
        self.queue: queue.Queue = queue.Queue()
        self.writer = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="stereo-write")
        self._csv = open(self.out / "frames.csv", "w", newline="")
        self._rows = csv.writer(self._csv)
        self._rows.writerow(["frame", "packet_utc_ns", "decoded_utc_ns"])
        self._lock = threading.Lock()
        self._state_read, self._state = 0.0, (None, None)
        self._last_disp = self._last_disp_index = self._cursor = self._fx = None
        self._on_top = True
        self._window_open = False
        request = headset_stream.CameraRequest(width=size[0], height=size[1], fps=fps, port=0)
        self.stream = headset_stream.HeadsetStream(
            headset, request, control_port=control_port, record=self.out / "stream",
            on_frame=self._on_frame, log=log, reset_pause=reset_pause)

    # Decoder thread: stamp, save the left eye, queue for the model.
    def _on_frame(self, frame, packet_ns):
        decoded_ns = time.time_ns()
        with self._lock:
            index = self.frames
            self.frames += 1
            self._rows.writerow([index, packet_ns, decoded_ns])
            now = time.monotonic()
            self.arrivals = [t for t in self.arrivals if now - t < 2.0] + [now]
        half = frame.shape[1] // 2
        self.writer.submit(self.cv2.imwrite, str(self.out / "left" / f"{index:06d}.jpg"),
                           frame[:, :half], [self.cv2.IMWRITE_JPEG_QUALITY, 95])
        if self.model is not None:
            self.queue.put((index, frame))

    def _disparity(self, index, frame):
        half = frame.shape[1] // 2
        t0 = time.perf_counter()
        d = self.model(frame[:, :half], frame[:, half:])
        self.model_ms = (self.model_ms + [1000 * (time.perf_counter() - t0)])[-50:]
        self.writer.submit(self.cv2.imwrite, str(self.out / "disparity" / f"{index:06d}.png"),
                           encode_disparity(d))
        self.done += 1
        return d

    def status(self):
        return {"t": time.time(), "connected": self.stream.connected.is_set(), "frames": self.frames,
                "disparity": self.done, "backlog": self.queue.qsize(),
                "stream_hz": len(self.arrivals) / 2.0,
                "model_ms": float(np.median(self.model_ms)) if self.model_ms else None,
                "size": list(self.stream.size) if self.stream.size else None,
                "error": str(self.stream.error or self.error or "") or None}

    def _write_status(self):
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(self.status()))
        os.replace(tmp, self.out / "status.json")

    # ------------------------------------------------------------- live view

    def _session_state(self):
        """The recorder's current step and link status (<session>/pc), re-read twice a second."""
        now = time.time()
        if now - self._state_read < 0.5:
            return self._state
        self._state_read = now
        pc = self.out.parent / "pc"
        step = status = None
        try:
            lines = (pc / "events.jsonl").read_text().splitlines()
            event = json.loads(lines[-1]) if lines else None
        except (OSError, ValueError):
            event = None
        if event:
            label = event["label"]
            if label.endswith("_start") and "seconds" in event:
                left = event["t"] + event["seconds"] - now
                step = (f"{label[:-6].upper()}  {max(left, 0.0):.0f} s", event.get("text", ""))
            elif label.endswith("_end") and label != "demo_end":
                step = ("NEXT STEP ...", "")
            elif label == "demo_start":
                step = (f"DEMONSTRATE  {now - event['t']:.0f} s", "Ctrl-C in the recorder's terminal stops")
            elif label == "connected":
                step = ("WAITING", "for the gloves, trackers, headset and camera")
        try:
            status = (pc / "status.txt").read_text().strip()
        except OSError:
            pass
        self._state = (step, status)
        return self._state

    def _colour_depth(self, d):
        """Depth (with a baseline) or disparity, near = red, no match = black, out of range = grey."""
        cv2 = self.cv2
        valid = np.isfinite(d)
        if self.baseline_m and self._fx:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = self._fx * self.baseline_m / d
            x = (VIEW_FAR_M - z) / (VIEW_FAR_M - VIEW_NEAR_M)
            outside = valid & ((z < VIEW_NEAR_M) | (z > VIEW_FAR_M))
        else:
            x = d / VIEW_MAX_DISPARITY
            outside = valid & (d > VIEW_MAX_DISPARITY)
        vis = cv2.applyColorMap((np.clip(np.nan_to_num(x), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[outside] = (90, 90, 90)
        vis[~valid] = 0
        return vis

    def _compose(self, draining=False, loading=None):
        cv2 = self.cv2
        latest = self.stream.latest()
        if latest is not None:
            _, _, frame = latest
            half = frame.shape[1] // 2
            ego = np.ascontiguousarray(frame[:, :half])
            if self._fx is None:
                self._fx = float(nominal_K(half, frame.shape[0])[0, 0])
        else:
            ego = np.zeros((self.size[1], self.size[0] // 2, 3), np.uint8)
            lines = ["waiting for the headset camera ..."]
            if self.stream.connected.is_set() and not self.frames:
                lines += [f"video connected, {self.stream.bytes} bytes so far.",
                          "If it stays at 0: close and reopen the",
                          "XRoboToolkit app (cameras stuck)."]
            for k, text in enumerate(lines):
                cv2.putText(ego, text, (20, ego.shape[0] // 2 - 20 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 if k == 0 else 0.6, (255, 255, 255) if k == 0 else (80, 200, 255),
                            2 if k == 0 else 1, cv2.LINE_AA)
        h, w = ego.shape[:2]
        if self._last_disp is not None and self._last_disp.shape == (h, w):
            depth = self._colour_depth(self._last_disp)
        else:
            depth = np.zeros_like(ego)
        body = np.hstack([ego, depth])
        header = np.zeros((96, body.shape[1], 3), np.uint8)
        step, status = self._session_state()
        s = self.status()
        if draining:
            step = ("FINISHING DEPTH", f"{s['disparity']}/{s['frames']} frames")
        elif loading:
            step = ("LOADING THE DEPTH MODEL", loading)
        if step:
            cv2.putText(header, step[0], (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(header, step[1], (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if status:
            warn = "MISSING" in status or "EXITED" in status or "waiting" in status
            cv2.putText(header, status[:180], (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (80, 80, 255) if warn else (190, 190, 190), 1, cv2.LINE_AA)
        behind = s["frames"] - 1 - self._last_disp_index if self._last_disp_index is not None else 0
        what = (f"depth {VIEW_NEAR_M:g}-{VIEW_FAR_M:g} m" if self.baseline_m else
                f"disparity 0-{VIEW_MAX_DISPARITY:g} px (no baseline: no metres)")
        lines = [(f"ego camera (left eye)  {s['stream_hz']:.0f} Hz", 8),
                 (f"{what}  red = near  | model {(s['model_ms'] or 0):.0f} ms"
                  + (f", {behind} frames behind" if behind > 2 else ""), w + 8)]
        for text, x in lines:
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            cv2.rectangle(body, (x - 6, h - 26), (min(x + tw + 6, body.shape[1] - 1), h - 4), (0, 0, 0), -1)
            cv2.putText(body, text, (x, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if self._cursor is not None and self._last_disp is not None:
            cx, cy = self._cursor
            u, v = cx % w, cy - header.shape[0]
            if 0 <= v < h and self._last_disp.shape == (h, w):
                dv = self._last_disp[v, u]
                txt = "no match" if not np.isfinite(dv) else (
                    f"{self._fx * self.baseline_m / dv:.3f} m" if self.baseline_m else f"{dv:.1f} px")
                for px in (u, u + w):
                    cv2.drawMarker(body, (px, v), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
                cv2.putText(body, txt, (u + w + 10, max(v - 10, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([header, body])

    def open_window(self):
        if self.window and not self._window_open:
            cv2 = self.cv2
            cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            cv2.setWindowProperty(VIEW_WINDOW, cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback(VIEW_WINDOW, self._on_mouse)
            self._window_open = True

    def _show(self, draining=False, loading=None):
        cv2 = self.cv2
        cv2.imshow(VIEW_WINDOW, self._compose(draining, loading))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("t"):
            self._on_top = not self._on_top
            cv2.setWindowProperty(VIEW_WINDOW, cv2.WND_PROP_TOPMOST, int(self._on_top))

    def _on_mouse(self, event, x, y, flags, param):
        if event == self.cv2.EVENT_MOUSEMOVE:
            self._cursor = (x, y)

    def run(self, stop: threading.Event, duration=0.0):
        cv2 = self.cv2
        if self.window:
            self.open_window()
            self._show()
        last_status = next_view = 0.0
        t_end = time.monotonic() + duration if duration else None
        try:
            try:
                self.stream.start()
            except ConnectionError as exc:
                self.error = exc
                self.log(str(exc))
                stop.set()
            seen, seen_at = 0, time.monotonic()
            while not stop.is_set():
                if t_end and time.monotonic() > t_end:
                    break
                if self.frames != seen:
                    seen, seen_at = self.frames, time.monotonic()
                elif time.monotonic() - seen_at > STALL_S:
                    received = self.stream.bytes
                    self.log(f"no new frame for {STALL_S:.0f} s ({received} bytes received so far); asking "
                             "the headset again. If this repeats: close and reopen the XRoboToolkit app "
                             "(its cameras can stay busy), keep it in front and the headset awake.")
                    try:
                        self.stream.request_again()
                    except OSError as exc:
                        self.log(f"request failed: {exc}")
                    seen_at = time.monotonic()
                if self.stream.error is not None:
                    self.error = self.stream.error
                    self.log(f"stream error: {self.stream.error}")
                    break
                try:
                    index, frame = self.queue.get(timeout=0.02)
                except queue.Empty:
                    index = frame = None
                if frame is not None:
                    self._last_disp = self._disparity(index, frame)
                    self._last_disp_index = index
                # The view refreshes on its own clock: the ego image is always the newest
                # frame, the depth the newest computed, however long the queue.
                if self.window and time.monotonic() >= next_view:
                    next_view = time.monotonic() + VIEW_PERIOD_S
                    self._show()
                if time.monotonic() - last_status > 1.0:
                    last_status = time.monotonic()
                    self._write_status()
        finally:
            self.stream.stop()  # CLOSE_CAMERA; the decoder hands over its last frames
            if self.model is not None and self.queue.qsize():
                self.log(f"finishing disparity for {self.queue.qsize()} queued frames ...")
            while self.model is not None:
                try:
                    index, frame = self.queue.get_nowait()
                except queue.Empty:
                    break
                self._last_disp, self._last_disp_index = self._disparity(index, frame), index
                if self.done % 50 == 0:
                    self._write_status()
                    if self.window:
                        self._show(draining=True)
            self.writer.shutdown(wait=True)
            with self._lock:
                self._csv.close()
            self._write_status()
            if self.window:
                cv2.destroyAllWindows()
            eye = (self.stream.size[0] // 2, self.stream.size[1]) if self.stream.size else None
            meta = {
                "format": "wuji_demo_stereo_v1",
                "headset": self.headset,
                "requested": {"size": list(self.size), "fps": self.fps},
                "eye_size": list(eye) if eye else None,
                "K_left_nominal": nominal_K(*eye).tolist() if eye else None,
                "intrinsics_source": f"nominal: {NOMINAL_HFOV_DEG} x {NOMINAL_VFOV_DEG} deg field of view, "
                                     "no distortion (the pair arrives rectified: same rows in both eyes)",
                "baseline_m": self.baseline_m,
                "disparity_scale": DISPARITY_SCALE,
                "disparity_model": self.model.info if self.model is not None else None,
                "frames": self.frames, "disparity_frames": self.done,
                "started_utc": self.started, "stopped_utc": time.time(),
                "time_base": "host UTC; packet_utc_ns is when the frame's picture packet arrived. "
                             "The capture time is earlier by the stream latency (sync stage).",
                "error": str(self.error) if self.error else None,
            }
            (self.out / "meta.json").write_text(json.dumps(meta, indent=2))
            self.log(f"stereo: {self.frames} frames, {self.done} with disparity -> {self.out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("headset", help="headset Wi-Fi IP (XRoboToolkit app open, camera access on)")
    parser.add_argument("--out", type=Path, required=True, help="new directory, e.g. <session>/stereo")
    parser.add_argument("--size", default="1280x480", help="side-by-side size, both eyes (default 1280x480)")
    parser.add_argument("--fps", type=int, default=30, help="requested; the app may send fewer")
    parser.add_argument("--control-port", type=int, default=headset_stream.CONTROL_PORT)
    parser.add_argument("--ffs-dir", type=Path, default=DEFAULT_FFS_DIR)
    parser.add_argument("--checkpoint", default="20-30-48",
                        help="20-30-48 keeps up with 30 fps (30 ms); 23-36-37 is more accurate (41 ms)")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument("--no-disparity", action="store_true", help="record the stream only")
    parser.add_argument("--baseline-m", type=float, help="camera baseline, stored in meta.json")
    parser.add_argument("--window", action="store_true",
                        help="live view: ego camera | depth, the recorder's step and link status")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds")
    args = parser.parse_args(argv)
    width, height = (int(v) for v in args.size.lower().split("x"))
    sys.setswitchinterval(0.001)  # packet arrival stamps wait less for the GIL while the model runs
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    rec = StereoRecorder(args.out, args.headset, size=(width, height), fps=args.fps,
                         control_port=args.control_port, model=None, window=args.window,
                         baseline_m=args.baseline_m, log=lambda m: print(m, flush=True))
    # The window opens at once; the model (torch import, CUDA, kernel choice: 10-20 s)
    # loads in the background meanwhile.
    rec.open_window()
    if not args.no_disparity:
        print(f"loading Fast-FoundationStereo {args.checkpoint} ...", flush=True)
        loaded = {}

        def load():  # torch import and weights only: CUDA work stays on the main thread
            try:
                loaded["model"] = FastFoundationStereo(args.ffs_dir, args.checkpoint, args.iters,
                                                       args.max_disp)
            except Exception as exc:  # reported below
                loaded["error"] = exc

        loader = threading.Thread(target=load, name="ffs-load", daemon=True)
        loader.start()
        t0 = time.monotonic()
        while loader.is_alive() and not stop.is_set():
            if args.window:
                rec._show(loading=f"Fast-FoundationStereo {args.checkpoint}, {time.monotonic() - t0:.0f} s "
                                  "(the camera starts when it is ready)")
            loader.join(0.05)
        if "error" in loaded:
            print(f"error: the depth model did not load: {loaded['error']}", flush=True)
            rec.error = loaded["error"]
            stop.set()
        model = loaded.get("model")
        if model is not None and not stop.is_set():
            # Warm up on the thread that runs the model: cuDNN picks its kernels per
            # thread, and a first call during the stream (about 4 s) would leave the
            # depth that far behind for the whole recording.
            dummy = np.zeros((height, width // 2, 3), np.uint8)
            for i in range(3):
                if args.window:
                    rec._show(loading=f"Fast-FoundationStereo {args.checkpoint}: warming up {i + 1}/3")
                model(dummy, dummy)
            rec.model = model
            print(f"model ready in {time.monotonic() - t0:.0f} s", flush=True)
    rec.run(stop, args.duration)
    return 1 if rec.error else 0


if __name__ == "__main__":
    sys.exit(main())
