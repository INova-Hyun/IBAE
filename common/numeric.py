from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def unit_or_none(vec: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if (not np.isfinite(norm)) or norm <= 1e-9:
        return None
    return arr / norm


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def robust_sigma(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return float("nan")
    med = float(np.median(finite))
    sigma = float(1.4826 * np.median(np.abs(finite - med)))
    if np.isfinite(sigma) and sigma > 0.0:
        return sigma
    return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")


finite_robust_sigma = robust_sigma
