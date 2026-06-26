"""Command-line entry point: live capture from the See3CAM_CU27, laser
spot detection, and focal-length fitting.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from .camera import PIXEL_SIZE_MM, CameraError, UvcCamera
from .pipeline import FocalLengthEngine


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
                    try:
                        z = float(input(f"z position for this frame (current default {z}): ") or z)
                        spot = engine.add_frame(frame, z=z)
                        print(
                            f"Captured at z={z}: centroid=({spot.cx:.1f}, {spot.cy:.1f}) "
                            f"sigma=({spot.sigma_x:.2f}, {spot.sigma_y:.2f}) px"
                        )
                    except ValueError as exc:
                        print(f"Could not measure spot: {exc}")
                if key == ord("f"):
                    try:
                        result = engine.fit()
                        print(
                            f"Focal length (z0) = {result.z0:.4f}, waist w0 = {result.w0:.4f}, "
                            f"Rayleigh range zR = {result.zR:.4f}"
                        )
                    except ValueError as exc:
                        print(f"Could not fit: {exc}")
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adapa", description="Laser focal-length detection engine")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index for the UVC device")
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
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
