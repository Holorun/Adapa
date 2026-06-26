"""Command-line entry point: laser spot detection and focal-length
fitting, either from a live See3CAM_CU27 capture or a recorded video
file exported from the camera.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from .camera import PIXEL_SIZE_MM, CameraError, UvcCamera, VideoFileSource
from .pipeline import FocalLengthEngine


def _capture_measurement(engine: FocalLengthEngine, frame, z: float) -> float:
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
        with UvcCamera(index=args.camera_index) as cam:
            print("Press 'c' to capture a measurement at the current z, 'f' to fit, 'q' to quit.")
            z = 0.0
            while True:
                frame = cam.read()
                cv2.imshow("Adapa - laser spot", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    z = _capture_measurement(engine, frame, z)
                if key == ord("f"):
                    _print_fit(engine)
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


def run_from_video(args: argparse.Namespace) -> int:
    engine = FocalLengthEngine(pixel_size=args.pixel_size, wavelength=args.wavelength)
    try:
        with VideoFileSource(args.video) as src:
            print(f"{src.frame_count} frames loaded.")
            print("space=pause/play, n=step forward, p=step back, c=capture, f=fit, q=quit.")
            z = 0.0
            paused = False
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
                        frame = src.read()
                    except CameraError:
                        print("End of video.")
                        paused = True
                elif key == ord("p"):
                    frame = src.step_back()
                elif key == ord("c"):
                    z = _capture_measurement(engine, frame, z)
                elif key == ord("f"):
                    _print_fit(engine)
                elif not paused:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video:
        return run_from_video(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
