"""FocalLengthEngine: accumulates spot measurements taken at known z
positions and fits a beam-waist model on demand to report the focal
length (z of minimum spot size), waist radius, and Rayleigh range.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import BeamFitResult, fit_beam_waist
from .camera import PIXEL_SIZE_MM
from .detection import SpotMeasurement, measure_spot


@dataclass
class ZMeasurement:
    z: float
    spot: SpotMeasurement


@dataclass
class FocalLengthEngine:
    """pixel_size: physical size of one sensor pixel (e.g. mm), used to
    convert pixel-domain spot sigmas into physical beam radii. Defaults to
    the See3CAM_CU27's 2.9 micron pixel pitch.
    """

    pixel_size: float = PIXEL_SIZE_MM
    wavelength: float | None = None
    _measurements: list[ZMeasurement] = field(default_factory=list)

    def add_frame(self, frame: np.ndarray, z: float) -> SpotMeasurement:
        spot = measure_spot(frame)
        self._measurements.append(ZMeasurement(z=z, spot=spot))
        return spot

    def reset(self) -> None:
        self._measurements.clear()

    def fit(self) -> BeamFitResult:
        if len(self._measurements) < 3:
            raise ValueError("Need at least 3 frames at distinct z positions before fitting")
        z = np.array([m.z for m in self._measurements])
        radius_px = np.array([(m.spot.sigma_x + m.spot.sigma_y) / 2 for m in self._measurements])
        radius_phys = radius_px * self.pixel_size
        return fit_beam_waist(z, radius_phys, wavelength=self.wavelength)
