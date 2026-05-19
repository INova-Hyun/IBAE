# MOJAVE FITS Opening Tool

Standalone tools for MOJAVE stacked FITS images.

Bundled data is intentionally limited to:

```text
../data/mojave_fits/0238+711.u.stacked.icc.fits
```

Run the Matplotlib UI from the repository root:

```bash
python mojave_fits/mojave_opening_tool.py data/mojave_fits/0238+711.u.stacked.icc.fits
```

Run the Qt UI:

```bash
python mojave_fits/mojave_opening_tool_qt.py data/mojave_fits/0238+711.u.stacked.icc.fits
```

The old Pushkarev ASCII comparison script is still included, but its external
ASCII ridgeline file is not bundled in this repository.
