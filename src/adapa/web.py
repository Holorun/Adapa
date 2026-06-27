"""Browser-viewable live feed for Adapa.

Meant to be opened as a plain web page/tab (e.g. inside Tagg, or any
browser) rather than the OpenCV preview window - the host app's own
drag/resize/pin system handles the frame; Adapa just needs to be a
normal page.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

from .camera import DEFAULT_CHANNEL, CameraError, UvcCamera
from .detection import frame_difference

# Cadence for re-rendering the placeholder frame while waiting for the first
# frame or after a disconnect - keeps the multipart stream alive without
# spinning the generator in a tight loop.
PLACEHOLDER_REFRESH_S = 0.2

# A live sensor frame always carries some shot noise (see
# detection.GLITCH_DIFF_THRESHOLD - a static scene still reads ~0.3-0.4).
# Some DirectShow drivers don't error on disconnect; cap.read() just keeps
# returning ok=True with a buffer that stops changing. So a frame-to-frame
# diff under this floor, sustained past STALE_AFTER_S, means the feed itself
# has stalled - not that the exception path caught it.
FROZEN_DIFF_THRESHOLD = 0.05
STALE_AFTER_S = 1.5

# How often to retry opening the camera, both when it's missing at startup
# and after it drops mid-stream.
RECONNECT_RETRY_S = 1.0

INDEX_HTML = """<!doctype html>
<html>
<head>
<title>Adapa</title>
<style>
  body { margin: 0; background: #000; }
  img { display: block; width: 100%; height: 100%; object-fit: contain; }
</style>
</head>
<body>
  <img src="/stream" alt="Adapa live feed">
</body>
</html>
"""


def _message_frame(shape: tuple[int, int], message: str) -> np.ndarray:
    frame = np.zeros(shape, dtype=np.uint8)
    cv2.putText(frame, message, (20, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    return frame


def create_app(
    camera_index: int,
    channel: str,
    exposure: float | None = None,
    brightness: float | None = None,
    contrast: float | None = None,
    auto_white_balance: bool | None = None,
) -> Flask:
    app = Flask(__name__)

    lock = threading.Lock()
    latest = {"frame": None, "error": None, "changed_at": None}

    def _capture_loop() -> None:
        while True:
            try:
                cam = UvcCamera(
                    index=camera_index,
                    channel=channel,
                    exposure=exposure,
                    brightness=brightness,
                    contrast=contrast,
                    auto_white_balance=auto_white_balance,
                )
            except CameraError as exc:
                with lock:
                    latest["error"] = str(exc)
                time.sleep(RECONNECT_RETRY_S)
                continue

            with lock:
                latest["error"] = None

            prev_frame = None
            changed_at = None
            try:
                while True:
                    frame = cam.read()
                    if (
                        prev_frame is None
                        or frame.shape != prev_frame.shape
                        or frame_difference(frame, prev_frame) >= FROZEN_DIFF_THRESHOLD
                    ):
                        changed_at = time.time()
                    prev_frame = frame

                    with lock:
                        latest["frame"] = frame
                        latest["changed_at"] = changed_at
            except CameraError as exc:
                with lock:
                    latest["error"] = str(exc)
            finally:
                cam.close()
            time.sleep(RECONNECT_RETRY_S)

    threading.Thread(target=_capture_loop, daemon=True).start()

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/stream")
    def stream():
        def generate():
            while True:
                with lock:
                    frame, error, changed_at = latest["frame"], latest["error"], latest["changed_at"]
                stale = changed_at is not None and (time.time() - changed_at) > STALE_AFTER_S
                if error or stale:
                    message = f"Camera disconnected: {error}" if error else "Camera disconnected: feed has stalled"
                    shape = frame.shape if frame is not None else (480, 640)
                    frame = _message_frame(shape, message)
                    time.sleep(PLACEHOLDER_REFRESH_S)
                elif frame is None:
                    frame = _message_frame((480, 640), "Connecting to camera...")
                    time.sleep(PLACEHOLDER_REFRESH_S)
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run(
    camera_index: int = 0,
    channel: str = DEFAULT_CHANNEL,
    port: int = 7400,
    exposure: float | None = None,
    brightness: float | None = None,
    contrast: float | None = None,
    auto_white_balance: bool | None = None,
) -> None:
    app = create_app(
        camera_index,
        channel,
        exposure=exposure,
        brightness=brightness,
        contrast=contrast,
        auto_white_balance=auto_white_balance,
    )
    app.run(host="127.0.0.1", port=port, threaded=True)
