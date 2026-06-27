"""Command-line entry point: laser spot detection and focal-length
fitting, either from a live See3CAM_CU27 capture or a recorded video
file exported from the camera.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from .camera import DEFAULT_CHANNEL, PIXEL_SIZE_MM, CameraError, UvcCamera, VideoFileSource
from .detection import looks_like_glitch
from .pipeline import FocalLengthEngine

# How often (ms) to retry cam.read() and repaint the window while disconnected.
DISCONNECTED_RETRY_MS = 200


def _message_frame(shape: tuple[int, int], message: str) -> np.ndarray:
    frame = np.zeros(shape, dtype=np.uint8)
    cv2.putText(frame, message, (20, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    return frame


def _capture_measurement(engine: FocalLengthEngine, frame, z: float, prev_frame=None) -> float:
    if looks_like_glitch(frame, prev_frame):
        print(
            "Warning: this frame differs sharply from the previous one - likely a "
            "momentary obstruction/particle scatter (or decode artifact), not the "
            "steady beam."
        )
        if input("Capture anyway? [y/N]: ").strip().lower() != "y":
            print("Skipped.")
            return z

    z = float(input(f"z position for this frame (current default {z}): ") or z)
    try:
        spot = engine.add_frame(frame, z=z)
        print(
            f"Captured at z={z}: centroid=({spot.cx:.1f}, {spot.cy:.1f}) "
            f"sigma=({spot.sigma_x:.2f}, {spot.sigma_y:.2f}) px"
        )
    except ValueError as exc:
        print(f"Could not measure spot: {exc}")
    return z


def _print_fit(engine: FocalLengthEngine) -> None:
    try:
        result = engine.fit()
        print(
            f"Focal length (z0) = {result.z0:.4f}, waist w0 = {result.w0:.4f}, "
            f"Rayleigh range zR = {result.zR:.4f}"
        )
    except ValueError as exc:
        print(f"Could not fit: {exc}")


def run_live(args: argparse.Namespace) -> int:
    engine = FocalLengthEngine(pixel_size=args.pixel_size, wavelength=args.wavelength)
    print("Press 'c' to capture a measurement at the current z, 'f' to fit, 'q' to quit.")
    z = 0.0
    prev_frame = None
    frame_shape = (480, 640)
    cam: UvcCamera | None = None
    try:
        while True:
            if cam is None:
                try:
                    cam = UvcCamera(
                        index=args.camera_index,
                        channel=args.channel,
                        exposure=args.exposure,
                        brightness=args.brightness,
                        contrast=args.contrast,
                        auto_white_balance=args.auto_white_balance,
                    )
                except CameraError as exc:
                    cv2.imshow("Adapa - laser spot", _message_frame(frame_shape, f"Camera disconnected: {exc}"))
                    if (cv2.waitKey(DISCONNECTED_RETRY_MS) & 0xFF) == ord("q"):
                        break
                    continue
            try:
                frame = cam.read()
            except CameraError as exc:
                cam.close()
                cam = None
                cv2.imshow("Adapa - laser spot", _message_frame(frame_shape, f"Camera disconnected: {exc}"))
                if (cv2.waitKey(DISCONNECTED_RETRY_MS) & 0xFF) == ord("q"):
                    break
                continue
            frame_shape = frame.shape
            cv2.imshow("Adapa - laser spot", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                z = _capture_measurement(engine, frame, z, prev_frame)
            if key == ord("f"):
                _print_fit(engine)
            prev_frame = frame
    finally:
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()
    return 0


def run_from_video(args: argparse.Namespace) -> int:
    engine = FocalLengthEngine(pixel_size=args.pixel_size, wavelength=args.wavelength)
    try:
        with VideoFileSource(args.video, channel=args.channel) as src:
            print(f"{src.frame_count} frames loaded.")
            print("space=pause/play, n=step forward, p=step back, c=capture, f=fit, q=quit.")
            z = 0.0
            paused = False
            prev_frame = None
            frame = src.read()
            while True:
                cv2.imshow("Adapa - laser spot (video)", frame)
                key = cv2.waitKey(0 if paused else 30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = not paused
                elif key == ord("n"):
                    try:
                        prev_frame = frame
                        frame = src.read()
                    except CameraError:
                        print("End of video.")
                        paused = True
                elif key == ord("p"):
                    prev_frame = frame
                    frame = src.step_back()
                elif key == ord("c"):
                    z = _capture_measurement(engine, frame, z, prev_frame)
                elif key == ord("f"):
                    _print_fit(engine)
                elif not paused:
                    prev_frame = frame
                    try:
                        frame = src.read()
                    except CameraError:
                        print("End of video.")
                        paused = True
    except CameraError as exc:
        print(f"Video error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adapa", description="Laser focal-length detection engine")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index for the UVC device")
    parser.add_argument(
        "--video", type=str, default=None, help="Path to a recorded video file instead of live capture"
    )
    parser.add_argument(
        "--pixel-size", type=float, default=PIXEL_SIZE_MM, help="Physical size of one sensor pixel (e.g. mm)"
    )
    parser.add_argument(
        "--wavelength", type=float, default=None, help="Laser wavelength, same length unit as pixel-size"
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL,
        choices=["red", "green", "blue", "gray", "max"],
        help="Which intensity channel to read from each color frame (default: red, tuned for the 780nm laser)",
    )
    parser.add_argument(
        "--web", action="store_true", help="Serve the live feed as a browser page instead of an OpenCV window"
    )
    parser.add_argument("--port", type=int, default=7400, help="Port for the web UI (used with --web)")
    parser.add_argument(
        "--exposure", type=float, default=None, help="Manual exposure value (switches off auto-exposure)"
    )
    parser.add_argument(
        "--brightness", type=float, default=None, help="Manual brightness (confirmed real on the See3CAM_CU27)"
    )
    parser.add_argument(
        "--contrast", type=float, default=None, help="Manual contrast (confirmed real on the See3CAM_CU27)"
    )
    parser.add_argument(
        "--auto-white-balance",
        dest="auto_white_balance",
        action="store_true",
        default=None,
        help="Enable auto white balance (off by default to keep the red channel reading stable)",
    )
    parser.add_argument(
        "--no-auto-white-balance",
        dest="auto_white_balance",
        action="store_false",
        help="Explicitly disable auto white balance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.web:
        from .web import run as run_web

        run_web(
            camera_index=args.camera_index,
            channel=args.channel,
            port=args.port,
            exposure=args.exposure,
            brightness=args.brightness,
            contrast=args.contrast,
            auto_white_balance=args.auto_white_balance,
        )
        return 0
    if args.video:
        return run_from_video(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
