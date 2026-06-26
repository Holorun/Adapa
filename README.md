# Adapa

Laser focal-spot detection and focal-length analysis engine.

Adapa reads frames from a UVC-class e-con Systems camera (See3CAM_CU27),
locates a laser spot in each frame, and fits a Gaussian-beam waist model
across frames captured at known z positions to report the beam's focal
length (the z position of minimum spot size), waist radius, and Rayleigh
range.

## Layout

- `src/adapa/camera.py` - UVC camera capture (OpenCV, DirectShow backend); also holds the See3CAM_CU27's sensor/lens constants (2.9 µm pixel pitch, stock 2.8 mm/F1.2 M12 lens)
- `src/adapa/detection.py` - per-frame laser spot detection (2D Gaussian fit) -> `measure_spot`
- `src/adapa/analysis.py` - beam-waist fit across z positions -> focal length -> `fit_beam_waist`
- `src/adapa/pipeline.py` - `FocalLengthEngine`, ties capture/detect/analyze together
- `src/adapa/cli.py` - `adapa` command-line entry point
- `See3CAM_CU27/` - e-con Systems datasheets and SDK docs for the camera (reference only)

## Setup

    cd C:\Users\kamha\Holorun-Vision\Adapa
    py -m venv .venv
    .venv\Scripts\activate
    pip install -e .[dev]

## Usage

Live capture from the camera, prompting for the z position at each
capture, then fitting once you have 3+ points spanning the focus:

    adapa --camera-index 0 --wavelength 0.000650

`--pixel-size` defaults to the See3CAM_CU27's 2.9 µm pitch (in mm); pass
`--wavelength` in the same length unit (e.g. mm) if you want the far-field
divergence angle reported too. `--channel` (default `red`) picks which
color channel to read intensity from - the camera is a color sensor, and
for an NIR laser (e.g. 780nm) the red Bayer channel saturates well before
green/blue, so it tracks true spot intensity far better than converting
to perceptual grayscale (confirmed against real footage: luma-based sigma
read ~30% larger than red-channel sigma on the same frames). Use `--channel
gray` for a non-NIR/visible-balanced source, or `max` to take whichever
channel is strongest per pixel. In the capture window:

- `c` - capture a measurement at the current z (prompts for z on stdin)
- `f` - fit the accumulated measurements and print the focal length
- `q` - quit

### From a recorded video

If you exported a z-scan video from the camera instead of running live,
analyze it the same way by pointing at the file instead of a camera index:

    adapa --video "C:\path\to\export.mp4" --wavelength 0.000650

Same `c`/`f`/`q` keys as live mode, plus playback controls since you're
scrubbing to the frame for each known z position rather than capturing
in real time:

- `space` - pause/play
- `n` - step forward one frame (while paused)
- `p` - step back one frame (while paused)

## Tests

No physical camera is required - the test suite validates the detection
and fitting math against synthetic data with known ground truth:

    pytest

## Fiji

Fiji (already extracted at
`C:\Users\kamha\Holorun-Vision\fiji-latest-win64-jdk`) is not required for
the detection/analysis pipeline above, but is useful for visually
validating spot detection or running additional ImageJ analyses on saved
frames. Install the optional `pyimagej` extra (`pip install -e .[fiji]`)
if you want to drive that Fiji install from Python later.
