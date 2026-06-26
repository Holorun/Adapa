import numpy as np

from adapa.detection import measure_spot, measure_spots


def _synthetic_frame(*spots, amplitude=200.0, background=10.0, shape=(240, 320)):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    frame = np.full(shape, background, dtype=np.float64)
    for cx, cy, sigma_x, sigma_y in spots:
        frame += amplitude * np.exp(
            -(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2))
        )
    return frame.astype(np.uint8)


def test_measure_spot_recovers_known_gaussian():
    frame = _synthetic_frame((150.0, 100.0, 12.0, 9.0))
    spot = measure_spot(frame)

    assert abs(spot.cx - 150.0) < 1.0
    assert abs(spot.cy - 100.0) < 1.0
    assert abs(spot.sigma_x - 12.0) < 1.0
    assert abs(spot.sigma_y - 9.0) < 1.0


def test_measure_spots_detects_multiple_distinct_blobs():
    frame = _synthetic_frame((80.0, 60.0, 10.0, 10.0), (240.0, 180.0, 8.0, 8.0))
    spots = measure_spots(frame)

    assert len(spots) == 2
    centroids = sorted((s.cx, s.cy) for s in spots)
    assert abs(centroids[0][0] - 80.0) < 1.0 and abs(centroids[0][1] - 60.0) < 1.0
    assert abs(centroids[1][0] - 240.0) < 1.0 and abs(centroids[1][1] - 180.0) < 1.0


def test_measure_spots_brightest_first():
    frame = _synthetic_frame((80.0, 60.0, 6.0, 6.0), (240.0, 180.0, 14.0, 14.0))
    spots = measure_spots(frame)

    assert spots[0].amplitude >= spots[1].amplitude
