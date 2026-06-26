import numpy as np

from adapa.detection import measure_spot


def _synthetic_frame(cx, cy, sigma_x, sigma_y, amplitude=200.0, background=10.0, shape=(240, 320)):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    frame = background + amplitude * np.exp(
        -(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2))
    )
    return frame.astype(np.uint8)


def test_measure_spot_recovers_known_gaussian():
    frame = _synthetic_frame(cx=150.0, cy=100.0, sigma_x=12.0, sigma_y=9.0)
    spot = measure_spot(frame, threshold_fraction=0.3)

    assert abs(spot.cx - 150.0) < 1.0
    assert abs(spot.cy - 100.0) < 1.0
    assert abs(spot.sigma_x - 12.0) < 1.0
    assert abs(spot.sigma_y - 9.0) < 1.0
