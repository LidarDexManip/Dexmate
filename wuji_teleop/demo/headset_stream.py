"""Live stereo passthrough video from the PICO headset to this laptop.

The stock XRoboToolkit app (v1.1.0 and later) sends its passthrough cameras to
any address that asks; the headset needs only PICO's enterprise camera access,
switched on in the app. This module asks for the stream, receives it, decodes
it with ffmpeg, and shows or saves it. It is a live view: the offline
demonstration pipeline still uses the headset's own recording, which keeps
frame times and the head pose of every frame.

Protocol, read from the XRoboToolkit Unity client source and its Android
library ``robotassistant_lib`` (unchanged from v1.1.1 to SwanV1.0):

- The app listens on TCP 13579 from launch and serves one client at a time.
  Each message is a u32 big-endian body length, then a body: i32 little-endian
  command length, the UTF-8 command, i32 little-endian data length, the data.
- ``OPEN_CAMERA`` data: ``CA FE``, version byte 1, seven i32 little-endian
  (width, height, fps, bitrate, enableMvHevc, renderMode, port), then the camera
  type and the receiver IP, each a u8 length and UTF-8. The camera type must be
  ``VR``; render mode 2 (``PXRCapture_RenderMode_3D``) gives left | right side
  by side.
- The headset then opens its cameras, connects to that IP and port, and sends
  each H.264 encoder output buffer as a u32 big-endian length and Annex-B
  bytes. The first is the SPS/PPS. The library fixes the bitrate at
  5242880 bit/s and ignores the requested one. No capture time is sent, so
  frames here carry their arrival time on this laptop.
- ``CLOSE_CAMERA`` (no data) stops the cameras.

Found on the real headset (2026-11-12):

- The app handles commands in Unity's frame loop, so it ignores them while the
  headset is asleep or the app is not in front (nothing connects back).
- A stream already running (e.g. left by a killed client) does not resend its
  SPS/PPS to a new connection, so nothing decodes. ``start`` therefore sends
  ``CLOSE_CAMERA`` before ``OPEN_CAMERA``, on the same connection.
- The app's "TCP Server error: Failed to read header" is harmless: its Java
  server logs it whenever a client disconnects, then accepts the next one.
- Asked for 1280x480 at 15 fps, it sent 640x480 per eye at 20 fps.
"""

from __future__ import annotations

import argparse
import collections
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTROL_PORT = 13579
STREAM_PORT = 12345
RENDER_MODE_STEREO = 2
MAGIC = b"\xca\xfe\x01"
MAX_PACKET = 16 << 20
PICTURE_NALS = (1, 5)
IDR_NAL = 5
CONFIG_NALS = (7, 8)
AUD_NAL = 9
# An access unit delimiter after each picture lets ffmpeg's parser end that
# picture at once instead of waiting for the next one (one frame of delay).
# Before an SPS it makes an empty access unit, which ffmpeg skips with a
# warning; no picture is lost.
AUD = b"\x00\x00\x00\x01\x09\xf0"


def encode_message(command: str, data: bytes = b"") -> bytes:
    """One control message as the app's TcpServer reads it."""
    cmd = command.encode()
    body = struct.pack("<i", len(cmd)) + cmd + struct.pack("<i", len(data)) + data
    return struct.pack(">I", len(body)) + body


def decode_message(body: bytes) -> tuple[str, bytes]:
    """Inverse of ``encode_message`` without the outer length."""
    (n,) = struct.unpack_from("<i", body, 0)
    command = body[4:4 + n].decode()
    (m,) = struct.unpack_from("<i", body, 4 + n)
    data = body[8 + n:8 + n + m]
    if len(data) != m:
        raise ValueError("truncated message")
    return command, data


def _compact(text: str) -> bytes:
    raw = text.encode()
    if len(raw) > 255:
        raise ValueError(f"string too long: {text!r}")
    return bytes([len(raw)]) + raw


@dataclass
class CameraRequest:
    """The ``OPEN_CAMERA`` request. Width is the whole side-by-side frame."""

    width: int = 2160
    height: int = 810
    fps: int = 30
    port: int = STREAM_PORT
    ip: str = ""
    bitrate: int = 5242880
    enable_mv_hevc: int = 0
    render_mode: int = RENDER_MODE_STEREO
    camera: str = "VR"

    def encode(self) -> bytes:
        return (MAGIC + struct.pack("<7i", self.width, self.height, self.fps, self.bitrate,
                                    self.enable_mv_hevc, self.render_mode, self.port)
                + _compact(self.camera) + _compact(self.ip))

    @classmethod
    def decode(cls, data: bytes) -> "CameraRequest":
        if data[:3] != MAGIC:
            raise ValueError("not an OPEN_CAMERA request")
        width, height, fps, bitrate, hevc, mode, port = struct.unpack_from("<7i", data, 3)
        off = 3 + 28
        n = data[off]
        camera = data[off + 1:off + 1 + n].decode()
        off += 1 + n
        n = data[off]
        ip = data[off + 1:off + 1 + n].decode()
        return cls(width, height, fps, port, ip, bitrate, hevc, mode, camera)


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_packet(sock: socket.socket) -> bytes | None:
    """One length-prefixed packet, or None when the connection ends."""
    head = _read_exact(sock, 4)
    if head is None:
        return None
    (n,) = struct.unpack(">I", head)
    if n > MAX_PACKET:
        raise ValueError(f"packet of {n} bytes; not an XRoboToolkit video stream")
    return _read_exact(sock, n)


def nal_types(payload: bytes) -> list[int]:
    """H.264 NAL unit types in an Annex-B buffer."""
    types = []
    i = payload.find(b"\x00\x00\x01")
    while i >= 0 and i + 3 < len(payload):
        types.append(payload[i + 3] & 0x1F)
        i = payload.find(b"\x00\x00\x01", i + 3)
    return types


def _with(prefix: Path, suffix: str) -> Path:
    return prefix.parent / (prefix.name + suffix)


def local_ip_for(host: str) -> str:
    """The laptop address the headset should connect back to (no packet is sent)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, 9))
        return s.getsockname()[0]


def adb_headset_ip() -> str | None:
    """The headset's Wi-Fi IPv4 address over adb, if one headset is attached."""
    try:
        out = subprocess.run(["adb", "shell", "ip", "-4", "-o", "addr", "show", "wlan0"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for word in out.split():
        if "/" in word and word.count(".") == 3:
            return word.split("/")[0]
    return None


def probe_size(stream: bytes) -> tuple[int, int] | None:
    """Picture size of an H.264 stream that starts with SPS/PPS and a picture."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-f", "h264", "-i", "pipe:0",
                              "-select_streams", "v:0", "-show_entries", "stream=width,height",
                              "-of", "csv=p=0"], input=stream, capture_output=True, timeout=10)
        width, height = (int(v) for v in out.stdout.decode().split()[0].split(",")[:2])
        return width, height
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class H264Decoder:
    """ffmpeg in a child process: Annex-B in, BGR frames out, in order."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.proc = subprocess.Popen(
            # Not -fflags nobuffer: it drops the packets read while probing,
            # which hold the first IDR picture, and with it the first GOP.
            ["ffmpeg", "-v", "error", "-flags", "low_delay", "-threads", "1",
             "-probesize", "32", "-analyzeduration", "0", "-f", "h264", "-i", "pipe:0",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self._chunks: collections.deque[bytes | None] = collections.deque()
        self._ready = threading.Condition()
        self._feeder = threading.Thread(target=self._feed, name="h264-feed", daemon=True)
        self._feeder.start()

    def _feed(self):
        while True:
            with self._ready:
                while not self._chunks:
                    self._ready.wait()
                chunk = self._chunks.popleft()
            if chunk is None:
                break
            try:
                self.proc.stdin.write(chunk)
            except (BrokenPipeError, ValueError, OSError):
                return
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def write(self, payload: bytes | None):
        with self._ready:
            self._chunks.append(payload)
            self._ready.notify()

    def frames(self):
        n = self.width * self.height * 3
        while True:
            buf = bytearray()
            while len(buf) < n:
                chunk = self.proc.stdout.read(n - len(buf))
                if not chunk:
                    return
                buf += chunk
            yield np.frombuffer(bytes(buf), np.uint8).reshape(self.height, self.width, 3)

    def close(self, timeout: float = 3.0):
        self.write(None)
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


class HeadsetStream:
    """Requests the headset's stereo stream and decodes it in background threads.

    ``latest()`` gives the newest (index, arrival host-UTC ns, BGR frame).
    ``on_frame(frame, t_ns)`` runs in the decoder thread for every frame.
    ``record`` (a path prefix) writes the raw stream to ``PREFIX.h264`` and one
    row per packet to ``PREFIX.times.csv``.
    """

    def __init__(self, headset: str, request: CameraRequest | None = None, *,
                 control_port: int = CONTROL_PORT, record: str | Path | None = None,
                 on_frame=None, log=print, reset_pause: float = 1.0):
        self.headset = headset
        self.request = request or CameraRequest()
        self.control_port = control_port
        self.reset_pause = reset_pause
        self.on_frame = on_frame
        self.log = log
        self.record = Path(record).expanduser() if record else None
        self.connected = threading.Event()
        self.size: tuple[int, int] | None = None
        self.frames = 0
        self.packets = 0
        self.bytes = 0
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._times: collections.deque[int] = collections.deque()
        self._pending: list[bytes] = []
        self._stream_aud = False
        self._decoder: H264Decoder | None = None
        self._reader: threading.Thread | None = None
        self._conn: socket.socket | None = None
        self._control: socket.socket | None = None
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._rec_h264 = self._rec_times = None

    def start(self, timeout: float = 5.0) -> "HeadsetStream":
        req = self.request
        self._server = socket.create_server(("0.0.0.0", req.port))
        self._server.settimeout(0.5)
        req.port = self._server.getsockname()[1]
        if not req.ip:
            req.ip = local_ip_for(self.headset)
        try:
            self._control = socket.create_connection((self.headset, self.control_port), timeout=timeout)
        except OSError as e:
            self._server.close()
            raise ConnectionError(
                f"Cannot reach the XRoboToolkit app at {self.headset}:{self.control_port} ({e}). "
                "Is the app open, the headset awake and on the same network?") from e
        if self.record:
            self.record.parent.mkdir(parents=True, exist_ok=True)
            self._rec_h264 = open(_with(self.record, ".h264"), "wb")
            self._rec_times = open(_with(self.record, ".times.csv"), "w")
            self._rec_times.write("packet,host_utc_ns,bytes,picture\n")
        # Stop a stream left running first: it would not resend its SPS/PPS.
        self._control.sendall(encode_message("CLOSE_CAMERA"))
        time.sleep(self.reset_pause)
        self._control.sendall(encode_message("OPEN_CAMERA", req.encode()))
        self.log(f"Asked {self.headset} for {req.width}x{req.height} at {req.fps} fps, "
                 f"sent to {req.ip}:{req.port}")
        self._thread = threading.Thread(target=self._receive, name="headset-stream", daemon=True)
        self._thread.start()
        return self

    def request_again(self, timeout: float = 5.0):
        """CLOSE_CAMERA then OPEN_CAMERA again (for a headset that did not act on the
        first request, or a stream that never decoded). A control connection the app
        dropped (e.g. the app was restarted) is opened anew."""
        for attempt in (0, 1):
            try:
                self._control.sendall(encode_message("CLOSE_CAMERA"))
                time.sleep(self.reset_pause)
                self._control.sendall(encode_message("OPEN_CAMERA", self.request.encode()))
                return
            except OSError:
                if attempt:
                    raise
                try:
                    self._control.close()
                except OSError:
                    pass
                self._control = socket.create_connection((self.headset, self.control_port), timeout=timeout)
                self.log("Reconnected to the app's control port")

    def _receive(self):
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = self._server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                conn.settimeout(None)
                self._conn = conn
                self.connected.set()
                self.log(f"Stream connected from {addr[0]}")
                with conn:
                    while not self._stop.is_set():
                        payload = read_packet(conn)
                        if payload is None:
                            break
                        self._on_packet(time.time_ns(), payload)
                if not self._stop.is_set():
                    self.log("Stream closed; waiting for the headset to reconnect")
        except Exception as e:  # reported to the caller through .error
            if not self._stop.is_set():
                self.error = e
                self.log(f"Stream error: {e}")

    def _on_packet(self, t_ns: int, payload: bytes):
        types = nal_types(payload)
        picture = any(t in PICTURE_NALS for t in types)
        if self._rec_h264:
            self._rec_h264.write(payload)
            self._rec_times.write(f"{self.packets},{t_ns},{len(payload)},{int(picture)}\n")
        self.packets += 1
        self.bytes += len(payload)
        if self._decoder is None:
            # Start decoding at the first IDR picture, after the SPS/PPS.
            if any(t in CONFIG_NALS for t in types) and not picture:
                self._pending = [payload]
                return
            if IDR_NAL not in types or not (self._pending or CONFIG_NALS[0] in types):
                return
            start = b"".join(self._pending) + payload
            self._pending = []
            self.size = probe_size(start) or (self.request.width, self.request.height)
            self.log(f"Decoding {self.size[0]}x{self.size[1]}")
            self._decoder = H264Decoder(*self.size)
            self._reader = threading.Thread(target=self._read_frames, name="h264-read", daemon=True)
            self._reader.start()
            self._times.append(t_ns)
            self._stream_aud = AUD_NAL in types
            self._decoder.write(start if self._stream_aud else start + AUD)
            return
        if AUD_NAL in types:
            self._stream_aud = True
        if picture:
            self._times.append(t_ns)
            if not self._stream_aud:
                payload += AUD
        self._decoder.write(payload)

    def _read_frames(self):
        for frame in self._decoder.frames():
            t_ns = self._times.popleft() if self._times else time.time_ns()
            with self._lock:
                self._latest = (self.frames, t_ns, frame)
            self.frames += 1
            if self.on_frame is not None:
                self.on_frame(frame, t_ns)

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._control is not None:
            try:
                self._control.sendall(encode_message("CLOSE_CAMERA"))
            except OSError:
                pass
            self._control.close()
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self._thread is not None:
            self._thread.join(3)
        if self._decoder is not None:
            self._decoder.close()
        if self._reader is not None:
            self._reader.join(3)
        for f in (self._rec_h264, self._rec_times):
            if f is not None:
                f.close()


def _show(stream: HeadsetStream, args) -> None:
    import cv2
    name = "PICO stereo passthrough (left | right)"
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    on_top = not args.no_on_top
    try:
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1 if on_top else 0)
    except (cv2.error, AttributeError):
        pass
    save_dir = Path(args.save_dir).expanduser()
    shown, last_stats, last_frames, last_bytes = -1, time.monotonic(), 0, 0
    rate = mbps = 0.0
    end = time.monotonic() + args.duration if args.duration else None
    while not (end and time.monotonic() > end):
        now = time.monotonic()
        if now - last_stats >= 1.0:
            rate = (stream.frames - last_frames) / (now - last_stats)
            mbps = (stream.bytes - last_bytes) * 8e-6 / (now - last_stats)
            last_stats, last_frames, last_bytes = now, stream.frames, stream.bytes
        item = stream.latest()
        if item is not None and item[0] != shown:
            shown, t_ns, frame = item
            scale = min(1.0, args.fit_width / frame.shape[1])
            view = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
            h, w = view.shape[:2]
            cv2.line(view, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)
            age = (time.time_ns() - t_ns) * 1e-6
            text = f"{frame.shape[1]}x{frame.shape[0]}  {rate:4.1f} fps  {mbps:4.1f} Mbit/s  {age:3.0f} ms since arrival"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(view, (4, 4), (16 + tw, 16 + th), (0, 0, 0), -1)
            cv2.putText(view, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(name, view)
        key = cv2.waitKey(15) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("t"):
            on_top = not on_top
            try:
                cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1 if on_top else 0)
            except (cv2.error, AttributeError):
                pass
        if key == ord("s") and item is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / time.strftime("pico-stereo-%Y%m%d-%H%M%S.png")
            cv2.imwrite(str(path), item[2])
            print(f"Saved {path}")
        try:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


def _report(stream: HeadsetStream, args) -> None:
    end = time.monotonic() + args.duration if args.duration else None
    last, last_frames, last_bytes = time.monotonic(), 0, 0
    while not (end and time.monotonic() > end):
        time.sleep(0.2)
        now = time.monotonic()
        if now - last >= 2.0:
            print(f"{stream.frames} frames, {(stream.frames - last_frames) / (now - last):.1f} fps, "
                  f"{(stream.bytes - last_bytes) * 8e-6 / (now - last):.1f} Mbit/s")
            last, last_frames, last_bytes = now, stream.frames, stream.bytes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--headset", help="headset Wi-Fi IP (default: ask adb)")
    p.add_argument("--width", type=int, default=2160, help="whole side-by-side width")
    p.add_argument("--height", type=int, default=810)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=STREAM_PORT, help="laptop port the headset connects to")
    p.add_argument("--laptop-ip", help="address the headset connects back to (default: the route to it)")
    p.add_argument("--control-port", type=int, default=CONTROL_PORT)
    p.add_argument("--record", help="save the raw stream to PREFIX.h264 and PREFIX.times.csv")
    p.add_argument("--no-show", action="store_true", help="no window; print rates")
    p.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds")
    p.add_argument("--wait", type=float, default=15.0, help="seconds to wait for the headset to connect")
    p.add_argument("--fit-width", type=int, default=1600)
    p.add_argument("--save-dir", default="~/Pictures/pico-stereo")
    p.add_argument("--no-on-top", action="store_true")
    args = p.parse_args(argv)
    headset = args.headset or adb_headset_ip()
    if not headset:
        p.error("pass --headset IP (PICO Settings > Wi-Fi > the connected network), "
                "or attach the headset over adb")
    request = CameraRequest(args.width, args.height, args.fps, args.port, args.laptop_ip or "")
    stream = HeadsetStream(headset, request, control_port=args.control_port, record=args.record)
    try:
        stream.start()
    except ConnectionError as e:
        print(e)
        return 1
    try:
        if not stream.connected.wait(args.wait):
            print(f"The headset took the request but did not connect back to {request.ip}:{request.port} "
                  f"within {args.wait:.0f} s. Check that PICO approved camera access and it is switched on "
                  "in the app, the app is v1.1.0 or later, the headset can reach that address, and the "
                  "headset is not recording (Data Collection) at the same time.")
            return 1
        (_report if args.no_show else _show)(stream, args)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        print(f"Received {stream.packets} packets, decoded {stream.frames} frames")
        if args.record:
            prefix = Path(args.record).expanduser()
            print(f"Raw stream: {_with(prefix, '.h264')}, arrival times: {_with(prefix, '.times.csv')}\n"
                  f"To mp4: ffmpeg -r {args.fps} -f h264 -i {_with(prefix, '.h264')} -c copy "
                  f"{_with(prefix, '.mp4')}")
    return 0 if stream.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
