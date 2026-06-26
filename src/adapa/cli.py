"""Command-line entry point: laser spot detection and focal-length
fitting, either from a live See3CAM_CU27 capture or a recorded video
file exported from the camera.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from .camera import DEFAULT_CHANNEL, PIXEL_SIZE_MM, CameraError, UvcCamera, VideoFileSource
from .detection import looks_like_glitch
from .pipeline import FocalLengthEngine


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
    try:
        with UvcCamera(index=args.camera_index, channel=args.channel) as cam:
            print("Press 'c' to capture a measurement at the current z, 'f' to fit, 'q' to quit.")
            z = 0.0
            prev_frame = None
            while True:
                frame = cam.read()
                cv2.imshow("Adapa - laser spot", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    z = _capture_measurement(engine, frame, z, prev_frame)
                if key == ord("f"):
                    _print_fit(engine)
                prev_frame = frame
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video:
        return run_from_video(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
