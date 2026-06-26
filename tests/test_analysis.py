import numpy as np

from adapa.analysis import fit_beam_waist


def test_fit_beam_waist_recovers_known_parameters():
    w0_true, z0_true, zR_true = 0.05, 25.0, 8.0
    z = np.linspace(0.0, 50.0, 9)
    radius = w0_true * np.sqrt(1.0 + ((z - z0_true) / zR_true) ** 2)

    result = fit_beam_waist(z, radius)

    assert abs(result.w0 - w0_true) < 1e-3
    assert abs(result.z0 - z0_true) < 1e-2
    assert abs(result.zR - zR_true) < 1e-2
