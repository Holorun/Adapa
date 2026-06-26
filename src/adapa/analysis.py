"""Fit a Gaussian-beam waist model to spot-size measurements taken at
several z positions, to find the focal length (axial location of
minimum spot size) and waist size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class BeamFitResult:
    w0: float                      # beam waist radius (focus spot radius), input units
    z0: float                      # axial position of the waist = focal length, input units
    zR: float                      # Rayleigh range, input units
    wavelength: float | None = None

    @property
    def divergence_half_angle(self) -> float | None:
        """Far-field half-angle divergence (radians); requires wavelength."""
        if self.wavelength is None:
            return None
        return self.wavelength / (np.pi * self.w0)


def _beam_radius(z, w0, z0, zR):
    return w0 * np.sqrt(1.0 + ((z - z0) / zR) ** 2)


def fit_beam_waist(z: np.ndarray, radius: np.ndarray, wavelength: float | None = None) -> BeamFitResult:
    """Fit w(z) = w0 * sqrt(1 + ((z - z0) / zR)^2) to (z, radius) samples.

    z and radius must be in the same length unit (e.g. mm), with z measured
    from a fixed reference point (e.g. the lens mount) so that the fitted
    z0 is directly the focal length from that reference. Needs at least 3
    distinct z positions, ideally spanning the focus.
    """
    z = np.asarray(z, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    if z.size < 3:
        raise ValueError("Need at least 3 (z, radius) samples to fit a beam waist")

    z0_guess = z[np.argmin(radius)]
    w0_guess = radius.min()
    zR_guess = (z.max() - z.min()) / 4 or 1.0

    popt, _ = optimize.curve_fit(
        _beam_radius, z, radius, p0=(w0_guess, z0_guess, zR_guess), maxfev=5000
    )
    w0, z0, zR = popt
    return BeamFitResult(w0=abs(w0), z0=z0, zR=abs(zR), wavelength=wavelength)
