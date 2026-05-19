"""Legacy Gaussian-baseline entry points.

The default FWHM measurement now fixes the reconstructed-background baseline
at zero.  These wrappers keep the older bounded L1/noise baseline model
available for old-session reproduction and one-off comparisons.
"""

from __future__ import annotations

from typing import Any

from ..ridgeline.analysis import fit_transverse_gaussian, measure_ridgeline_fwhm


def fit_transverse_gaussian_legacy_bounded(*args: Any, **kwargs: Any):
    kwargs["baseline_mode"] = "legacy_bounded_l1_noise"
    return fit_transverse_gaussian(*args, **kwargs)


def measure_ridgeline_fwhm_legacy_bounded(*args: Any, **kwargs: Any):
    kwargs["gaussian_baseline_mode"] = "legacy_bounded_l1_noise"
    return measure_ridgeline_fwhm(*args, **kwargs)


__all__ = [
    "fit_transverse_gaussian_legacy_bounded",
    "measure_ridgeline_fwhm_legacy_bounded",
]
