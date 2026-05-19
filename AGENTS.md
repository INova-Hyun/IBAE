# AGENTS.md

## Project context

This is a scientific Python project for VLBI/jet image analysis.
The code estimates jet ridgelines, transverse profiles, Gaussian widths,
beam-deconvolved sizes, and apparent opening angles from image or contour data.

## Review priorities

When reviewing this repository, prioritize scientific correctness over style.
Pay special attention to:

- Coordinate conventions: pixel, mas, image axes, PA angles
- Unit conversions: px to mas, beam major/minor axes, angular distance
- Gaussian fitting: initial guesses, bounds, baseline, sigma, failed fits
- Profile extraction: slice width, interpolation, edge handling, masking
- Beam deconvolution: unresolved cases, negative values, upper limits
- Error handling: NaN, low SNR, missing metadata, failed optimization
- Reproducibility: config, parameters, random seeds, saved outputs

## Do not do

- Do not make broad refactors unless explicitly requested.
- Do not change scientific formulas without explaining the assumption.
- Do not silently change units or coordinate conventions.
- Do not replace working algorithms with generic alternatives without evidence.

## Definition of done

For any code change:
- Explain what changed and why.
- Preserve existing behavior unless the bug requires otherwise.
- Add or suggest a test when feasible.
- Run the relevant tests or explain why they could not be run.
