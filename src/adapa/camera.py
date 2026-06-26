"""UVC capture for the e-con Systems See3CAM_CU27.

See3CAM_CU27 is UVC 1.1 compliant and shows up as a standard DirectShow
capture source on Windows, so plain OpenCV is enough - no vendor SDK
required. Reference: See3CAM_CU27/e-con_See3CAM_CU27_datasheet.pdf and
.../e-con_See3CAM_CU27_lens_datasheet.pdf.
"""
from __future__ import annotations

import cv2
import numpy as np

# Sony IMX462LQR sensor pixel pitch (datasheet section 4.2), in millimeters.
PIXEL_SIZE_MM = 0.0029

# Stock M12 lens shipped with the camera (lens datasheet section 4).
STOCK_LENS_FOCAL_LENGTH_MM = 2.8
STOCK_LENS_APERTURE_F_NUMBER = 1.2

RESOLUTIONS = {
    "vga": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}


class CameraError(RuntimeError):
    pass


class UvcCamera:
    """Thin wrapper around cv2.VideoCapture for a UVC device on Windows."""

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] = RESOLUTIONS["fhd"],
        exposure: float | None = None,
    ):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera at index {index}")

        width, height = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if exposure is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def read(self) -> np.ndarray:
        """Grab one frame as grayscale (laser-spot intensity, no color info needed)."""
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError("Failed to grab frame from camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "UvcCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
