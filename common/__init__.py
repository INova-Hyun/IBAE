"""Shared helpers used across IBAE implementation blocks."""

from importlib import import_module

__all__ = [
    "ensure_bgr",
    "finite_mean",
    "finite_median",
    "finite_robust_sigma",
    "raster_line_points",
    "robust_sigma",
    "safe_float",
    "unit_or_none",
]


def __getattr__(name: str):
    if name in {"ensure_bgr", "raster_line_points"}:
        image = import_module(f"{__name__}.image")
        return getattr(image, name)
    if name in {"finite_mean", "finite_median", "finite_robust_sigma", "robust_sigma", "safe_float", "unit_or_none"}:
        numeric = import_module(f"{__name__}.numeric")
        return getattr(numeric, name)
    raise AttributeError(name)
