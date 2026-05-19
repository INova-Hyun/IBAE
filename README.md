# IBAE

Image-Based Apparent opening angle Extractor for contour-map based VLBI jet analysis.

This repository contains the refactored IBAE workflow and the companion MOJAVE FITS opening-angle tools.

## Layout

- `IBAE` package root: interactive ROI selection, contour reconstruction, ridgeline extraction, FWHM/opening-angle analysis.
- `mojave_fits/`: standalone MOJAVE FITS ridgeline and opening-angle tools.
- `data/tests/`: regression-test fixture files.
- `data/images/`: image inputs, including `0238+711.png`.
- `data/mojave_fits/`: bundled MOJAVE FITS data, currently `0238+711.u.stacked.icc.fits`.

## Requirements

Use Python 3.10 or newer.

Runtime modules:

```bash
python -m pip install numpy scipy matplotlib opencv-python scikit-image PyQt5
```

Test module:

```bash
python -m pip install pytest
```

Editable install from the repository root:

```bash
python -m pip install -e .
```

Equivalent one-step setup:

```bash
python -m pip install -e . pytest
```

## Run IBAE

```python
from IBAE.opening_angle_calculator_v9_qt import run_v9_preview

run_v9_preview("data/images/0238+711.png")
```

For the bundled regression fixture:

```python
from IBAE.opening_angle_calculator_v9_qt import run_v9_preview

run_v9_preview("data/tests/ngc1052_mojave.png")
```

## Run Tests

```bash
python -m pytest
```

The regression tests use files under `data/tests/`.

## Run MOJAVE FITS Tool

Matplotlib UI:

```bash
python mojave_fits/mojave_opening_tool.py data/mojave_fits/0238+711.u.stacked.icc.fits
```

Qt UI:

```bash
python mojave_fits/mojave_opening_tool_qt.py data/mojave_fits/0238+711.u.stacked.icc.fits
```

Generated MOJAVE outputs should be written outside git-tracked data folders or under ignored output directories.

## Notes

- IBAE FWHM Gaussian measurements now use a fixed reconstructed-background baseline of `0.0` by default.
- The old bounded L1/noise Gaussian baseline path remains available through `IBAE.legacy`.
- The default ridgeline extraction path uses the MOJAVE-style polar sampler; the previous cost-path extractor remains available as a legacy mode.
