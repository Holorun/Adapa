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

# The IMX462LQR is a STARVIS sensor with strong NIR response. For a 780nm
# laser, the red Bayer channel saturates well before green/blue, so it
# tracks the true spot intensity far better than luminance grayscale
# (which actively discounts red, ~30% weight) - confirmed against real
# footage where luma-based sigma read ~30% larger than red-channel sigma.
DEFAULT_CHANNEL = "red"


class CameraError(RuntimeError):
    pass


def _extract_channel(frame_bgr: np.ndarray, channel: str) -> np.ndarray:
    if channel == "gray":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if channel == "max":
        return frame_bgr.max(axis=2)
    index = {"blue": 0, "green": 1, "red": 2}.get(channel)
    if index is None:
        raise ValueError(f"Unknown channel '{channel}'; expected red/green/blue/gray/max")
    return frame_bgr[:, :, index]


class UvcCamera:
    """Thin wrapper around cv2.VideoCapture for a UVC device on Windows."""

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] = RESOLUTIONS["fhd"],
        exposure: float | None = None,
        channel: str = DEFAULT_CHANNEL,
    ):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera at index {index}")
        self._channel = channel

        width, height = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if exposure is not None:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def read(self) -> np.ndarray:
        """Grab one frame as a single intensity channel (see `DEFAULT_CHANNEL`)."""
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError("Failed to grab frame from camera")
        return _extract_channel(frame, self._channel)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "UvcCamera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource:
    """Frame source reading from a previously recorded video file, so a
    z-scan exported from the camera can be analyzed without the camera
    attached. Same `.read()`/context-manager interface as `UvcCamera`.
    """

    def __init__(self, path: str, channel: str = DEFAULT_CHANNEL):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open video file: {path}")
        self._channel = channel

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def position(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek(self, frame_index: int) -> None:
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError("End of video file reached")
        return _extract_channel(frame, self._channel)

    def step_back(self) -> np.ndarray:
        """Re-read the previous frame (one step back from the current position)."""
        self.seek(self.position - 2)
        return self.read()

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoFileSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
