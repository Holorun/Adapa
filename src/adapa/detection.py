"""Per-frame laser spot detection: locate the spot and fit a 2D Gaussian
to it to get sub-pixel centroid and size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class SpotMeasurement:
    cx: float       # centroid x, pixels (frame coordinates)
    cy: float       # centroid y, pixels (frame coordinates)
    sigma_x: float  # Gaussian sigma along x, pixels
    sigma_y: float  # Gaussian sigma along y, pixels
    amplitude: float
    background: float

    @property
    def diameter_1e2_x(self) -> float:
        """1/e^2 beam diameter along x, in pixels."""
        return 4.0 * self.sigma_x

    @property
    def diameter_1e2_y(self) -> float:
        return 4.0 * self.sigma_y


def _gaussian2d(coords, amplitude, cx, cy, sigma_x, sigma_y, background):
    x, y = coords
    return background + amplitude * np.exp(
        -(((x - cx) ** 2) / (2 * sigma_x**2) + ((y - cy) ** 2) / (2 * sigma_y**2))
    )


def find_spot_roi(frame: np.ndarray, threshold_fraction: float = 0.5) -> tuple[slice, slice]:
    """Crop a region of interest around the brightest blob via intensity threshold."""
    values = frame.astype(np.float64)
    threshold = values.min() + threshold_fraction * (values.max() - values.min())
    mask = values >= threshold
    if not mask.any():
        raise ValueError("No pixels above threshold; spot not found")

    ys, xs = np.nonzero(mask)
    pad = 10
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, frame.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, frame.shape[1])
    return slice(y0, y1), slice(x0, x1)


def measure_spot(frame: np.ndarray, threshold_fraction: float = 0.5) -> SpotMeasurement:
    """Locate the laser spot in a grayscale `frame` and fit a 2D Gaussian to it."""
    row_slice, col_slice = find_spot_roi(frame, threshold_fraction)
    roi = frame[row_slice, col_slice].astype(np.float64)

    yy, xx = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]

    total = roi.sum()
    cx0 = float((xx * roi).sum() / total)
    cy0 = float((yy * roi).sum() / total)
    sigma0 = max(roi.shape) / 6.0
    amp0 = float(roi.max() - roi.min())
    bg0 = float(roi.min())

    p0 = (amp0, cx0, cy0, sigma0, sigma0, bg0)
    try:
        popt, _ = optimize.curve_fit(
            _gaussian2d, (xx.ravel(), yy.ravel()), roi.ravel(), p0=p0, maxfev=5000
        )
    except RuntimeError as exc:
        raise ValueError("Gaussian fit did not converge") from exc

    amplitude, cx, cy, sigma_x, sigma_y, background = popt
    return SpotMeasurement(
        cx=cx + col_slice.start,
        cy=cy + row_slice.start,
        sigma_x=abs(sigma_x),
        sigma_y=abs(sigma_y),
        amplitude=amplitude,
        background=background,
    )
