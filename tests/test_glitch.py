import numpy as np

from adapa.detection import frame_difference, looks_like_glitch


def test_frame_difference_zero_for_identical_frames():
    frame = np.full((100, 100), 50, dtype=np.uint8)
    assert frame_difference(frame, frame) == 0.0


def test_looks_like_glitch_flags_sudden_jump():
    prev_frame = np.full((100, 100), 10, dtype=np.uint8)
    glitch_frame = np.full((100, 100), 10, dtype=np.uint8)
    glitch_frame[40:60, 40:60] = 200  # localized sudden bright patch

    assert looks_like_glitch(glitch_frame, prev_frame)


def test_looks_like_glitch_ignores_small_noise():
    prev_frame = np.full((100, 100), 10, dtype=np.uint8)
    noisy_frame = prev_frame.astype(np.int16) + 1
    noisy_frame = noisy_frame.clip(0, 255).astype(np.uint8)

    assert not looks_like_glitch(noisy_frame, prev_frame)


def test_looks_like_glitch_false_when_no_previous_frame():
    frame = np.full((100, 100), 10, dtype=np.uint8)
    assert not looks_like_glitch(frame, None)
