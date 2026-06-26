import cv2
import numpy as np

from adapa.camera import VideoFileSource


def _write_synthetic_video(path: str, frame_count: int, shape=(120, 160)) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (shape[1], shape[0]))
    for i in range(frame_count):
        frame = np.full((*shape, 3), fill_value=i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_file_source_reads_expected_frame_count(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    _write_synthetic_video(video_path, frame_count=5)

    with VideoFileSource(video_path) as src:
        assert src.frame_count == 5
        frame = src.read()
        assert frame.ndim == 2  # grayscale
        assert src.position == 1


def test_video_file_source_step_back(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    _write_synthetic_video(video_path, frame_count=5)

    with VideoFileSource(video_path) as src:
        first = src.read()
        src.read()
        back = src.step_back()
        assert np.array_equal(first, back)
