#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from fits_viewer import (
    MAS_PER_DEG,
    apply_stretch,
    auto_limits,
    image_edges_mas,
    positive_contour_levels,
    read_primary_fits,
    robust_corner_rms,
)
from ridge_builder import sample_image_bilinear


FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


@dataclass
class FitResult:
    success: bool
    reason: str
    baseline: float = float("nan")
    amplitude: float = float("nan")
    mu_mas: float = float("nan")
    sigma_mas: float = float("nan")
    fwhm_mas: float = float("nan")
    fwhm_sigma_mas: float = float("nan")
    rmse: float = float("nan")
    n_fit: int = 0
    fit_min_mas: float = float("nan")
    fit_max_mas: float = float("nan")
    baseline_method: str = ""
    baseline_flag: str = ""
    baseline_n: int = 0
    baseline_left_median: float = float("nan")
    baseline_right_median: float = float("nan")
    source_min_mas: float = float("nan")
    source_max_mas: float = float("nan")
    component_center_offset_mas: float = float("nan")


@dataclass
class FitRegions:
    fit_mask: np.ndarray
    source_mask: np.ndarray
    baseline_mask: np.ndarray
    baseline_left_mask: np.ndarray
    baseline_right_mask: np.ndarray
    source_min_mas: float = float("nan")
    source_max_mas: float = float("nan")
    fit_min_mas: float = float("nan")
    fit_max_mas: float = float("nan")
    component_center_offset_mas: float = float("nan")
    component_peak_mas: float = float("nan")
    flags: Tuple[str, ...] = ()


@dataclass
class OpeningRecord:
    index: int
    radial_mas: float
    path_mas: float
    x_mas: float
    y_mas: float
    tangent_pa_deg: float
    scan_min_mas: float
    scan_max_mas: float
    fit: FitResult
    intrinsic_fwhm_mas: float
    opening_angle_raw_deg: float
    opening_angle_deg: float
    pa_sweep_raw_rms_mas: float
    pa_sweep_intrinsic_rms_mas: float
    pa_sweep_opening_angle_raw_rms_deg: float
    pa_sweep_opening_angle_intrinsic_rms_deg: float
    pa_sweep_success_count: int


class AnalysisCancelled(RuntimeError):
    pass


def _is_cancelled(cancel_event: object = None) -> bool:
    return bool(cancel_event is not None and getattr(cancel_event, "is_set")())


def _check_cancelled(cancel_event: object = None) -> None:
    if _is_cancelled(cancel_event):
        raise AnalysisCancelled("analysis cancelled")


def _beam_mas(header: Dict[str, object]) -> float:
    bmaj = float(header.get("BMAJ", float("nan"))) * MAS_PER_DEG
    bmin = float(header.get("BMIN", float("nan"))) * MAS_PER_DEG
    if np.isfinite(bmaj) and np.isfinite(bmin) and bmaj > 0 and bmin > 0:
        return float(math.sqrt(bmaj * bmin))
    return float("nan")


def _pixel_mas(header: Dict[str, object]) -> float:
    vals = []
    for key in ("CDELT1", "CDELT2"):
        if key in header:
            vals.append(abs(float(header[key])) * MAS_PER_DEG)
    finite = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.median(finite)) if finite else 0.05


def _prefixed_path(prefix: Path, extension: str) -> Path:
    return prefix.parent / f"{prefix.name}{extension}"


def default_ridge_for_fits(fits_path: Path, sector: str) -> Path:
    name = fits_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".fits"):
        name = name[:-5]
    return fits_path.with_name(f"{name}_polar_ridge_{sector}.json")


def expand_axes_to_points(ax, points: np.ndarray, padding_mas: float = 0.5) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.size == 0:
        return
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    if not np.any(finite):
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_low = min(float(xlim[0]), float(xlim[1]), float(np.nanmin(pts[finite, 0])) - padding_mas)
    x_high = max(float(xlim[0]), float(xlim[1]), float(np.nanmax(pts[finite, 0])) + padding_mas)
    y_low = min(float(ylim[0]), float(ylim[1]), float(np.nanmin(pts[finite, 1])) - padding_mas)
    y_high = max(float(ylim[0]), float(ylim[1]), float(np.nanmax(pts[finite, 1])) + padding_mas)
    ax.set_xlim((x_high, x_low) if xlim[0] > xlim[1] else (x_low, x_high))
    ax.set_ylim((y_high, y_low) if ylim[0] > ylim[1] else (y_low, y_high))


def gaussian_model(s: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    sigma = np.maximum(np.abs(sigma), 1e-12)
    return baseline + amplitude * np.exp(-0.5 * ((s - mu) / sigma) ** 2)


def gaussian_jacobian(s: np.ndarray, baseline: float, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    sigma = max(abs(float(sigma)), 1e-12)
    delta = np.asarray(s, dtype=float) - float(mu)
    exp_term = np.exp(-0.5 * (delta / sigma) ** 2)
    jac = np.empty((len(np.asarray(s)), 4), dtype=float)
    jac[:, 0] = 1.0
    jac[:, 1] = exp_term
    jac[:, 2] = float(amplitude) * exp_term * delta / (sigma * sigma)
    jac[:, 3] = float(amplitude) * exp_term * delta * delta / (sigma * sigma * sigma)
    return jac


def gaussian_fixed_baseline_jacobian(s: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    sigma = max(abs(float(sigma)), 1e-12)
    delta = np.asarray(s, dtype=float) - float(mu)
    exp_term = np.exp(-0.5 * (delta / sigma) ** 2)
    jac = np.empty((len(np.asarray(s)), 3), dtype=float)
    jac[:, 0] = exp_term
    jac[:, 1] = float(amplitude) * exp_term * delta / (sigma * sigma)
    jac[:, 2] = float(amplitude) * exp_term * delta * delta / (sigma * sigma * sigma)
    return jac


def default_scan_half_width_mas(beam_mas: float) -> float:
    beam_term = 6.0 * float(beam_mas) if np.isfinite(beam_mas) and beam_mas > 0.0 else float("nan")
    return float(max([v for v in (7.5, beam_term) if np.isfinite(v)]))


def default_fit_padding_mas(beam_mas: float) -> float:
    return float(1.5 * beam_mas) if np.isfinite(beam_mas) and beam_mas > 0.0 else 1.0


def default_baseline_guard_mas(beam_mas: float) -> float:
    return float(0.5 * beam_mas) if np.isfinite(beam_mas) and beam_mas > 0.0 else 0.25


def robust_median(values: Sequence[float], clip_sigma: float = 3.0, iterations: int = 3) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    clipped = arr
    for _ in range(max(0, int(iterations))):
        if clipped.size < 3:
            break
        med = float(np.nanmedian(clipped))
        mad = float(np.nanmedian(np.abs(clipped - med)))
        if not np.isfinite(mad) or mad <= 0.0:
            break
        sigma = 1.4826 * mad
        keep = np.abs(clipped - med) <= float(clip_sigma) * sigma
        if np.count_nonzero(keep) == clipped.size or np.count_nonzero(keep) == 0:
            break
        clipped = clipped[keep]
    return float(np.nanmedian(clipped)) if clipped.size else float("nan")


def pa_to_unit(pa_deg: float) -> np.ndarray:
    pa = math.radians(pa_deg % 360.0)
    return np.asarray([math.sin(pa), math.cos(pa)], dtype=float)


def sample_transverse_profile(
    image: np.ndarray,
    header: Dict[str, object],
    x_mas: float,
    y_mas: float,
    tangent_pa_deg: float,
    half_width_mas: float,
    sample_step_mas: float,
    pa_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    normal_pa = (float(tangent_pa_deg) + 90.0 + float(pa_offset_deg)) % 360.0
    normal = pa_to_unit(normal_pa)
    s = np.arange(-half_width_mas, half_width_mas + 0.5 * sample_step_mas, sample_step_mas, dtype=float)
    x = x_mas + s * normal[0]
    y = y_mas + s * normal[1]
    values = sample_image_bilinear(image, header, x, y)
    return s, values


def transverse_image_s_limits(
    header: Dict[str, object],
    shape: Tuple[int, int],
    x_mas: float,
    y_mas: float,
    normal: np.ndarray,
) -> Tuple[float, float]:
    x0, x1, y0, y1 = image_edges_mas(header, shape)
    xmin, xmax = sorted((float(x0), float(x1)))
    ymin, ymax = sorted((float(y0), float(y1)))
    s_low = -float("inf")
    s_high = float("inf")
    for coord, direction, low, high in (
        (float(x_mas), float(normal[0]), xmin, xmax),
        (float(y_mas), float(normal[1]), ymin, ymax),
    ):
        if abs(direction) < 1e-15:
            if coord < low or coord > high:
                return float("nan"), float("nan")
            continue
        a = (low - coord) / direction
        b = (high - coord) / direction
        s_low = max(s_low, min(a, b))
        s_high = min(s_high, max(a, b))
    if not np.isfinite(s_low) or not np.isfinite(s_high) or s_low >= s_high:
        return float("nan"), float("nan")
    return float(s_low), float(s_high)


def sample_transverse_profile_between(
    image: np.ndarray,
    header: Dict[str, object],
    x_mas: float,
    y_mas: float,
    tangent_pa_deg: float,
    s_min_mas: float,
    s_max_mas: float,
    sample_step_mas: float,
    pa_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    normal_pa = (float(tangent_pa_deg) + 90.0 + float(pa_offset_deg)) % 360.0
    normal = pa_to_unit(normal_pa)
    step = max(abs(float(sample_step_mas)), 1e-12)
    neg = np.arange(0.0, float(s_min_mas) - 0.5 * step, -step, dtype=float)[::-1]
    pos = np.arange(step, float(s_max_mas) + 0.5 * step, step, dtype=float)
    s = np.concatenate([neg, pos])
    x = float(x_mas) + s * normal[0]
    y = float(y_mas) + s * normal[1]
    values = sample_image_bilinear(image, header, x, y)
    return s, values


def sample_wide_transverse_profile(
    image: np.ndarray,
    header: Dict[str, object],
    x_mas: float,
    y_mas: float,
    tangent_pa_deg: float,
    half_width_mas: float,
    sample_step_mas: float,
    pa_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    normal_pa = (float(tangent_pa_deg) + 90.0 + float(pa_offset_deg)) % 360.0
    normal = pa_to_unit(normal_pa)
    s_min, s_max = transverse_image_s_limits(header, image.shape, x_mas, y_mas, normal)
    if np.isfinite(s_min) and np.isfinite(s_max):
        s_min = max(float(s_min), -abs(float(half_width_mas)))
        s_max = min(float(s_max), abs(float(half_width_mas)))
        if s_min < s_max:
            return sample_transverse_profile_between(
                image,
                header,
                x_mas,
                y_mas,
                tangent_pa_deg,
                s_min,
                s_max,
                sample_step_mas,
                pa_offset_deg,
            )
    return sample_transverse_profile(
        image,
        header,
        x_mas,
        y_mas,
        tangent_pa_deg,
        abs(float(half_width_mas)),
        sample_step_mas,
        pa_offset_deg,
    )


def sample_threshold_limited_transverse_profile(
    image: np.ndarray,
    header: Dict[str, object],
    x_mas: float,
    y_mas: float,
    tangent_pa_deg: float,
    threshold: float,
    padding_mas: float,
    min_points: int,
    sample_step_mas: float,
    pa_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    normal_pa = (float(tangent_pa_deg) + 90.0 + float(pa_offset_deg)) % 360.0
    normal = pa_to_unit(normal_pa)
    s_min, s_max = transverse_image_s_limits(header, image.shape, x_mas, y_mas, normal)
    if not np.isfinite(s_min) or not np.isfinite(s_max):
        fallback = max(1.5, abs(float(sample_step_mas)) * max(int(min_points), 8))
        return sample_transverse_profile(image, header, x_mas, y_mas, tangent_pa_deg, fallback, sample_step_mas, pa_offset_deg)
    s, values = sample_transverse_profile_between(
        image,
        header,
        x_mas,
        y_mas,
        tangent_pa_deg,
        s_min,
        s_max,
        sample_step_mas,
        pa_offset_deg,
    )
    mask = select_fit_window(s, values, threshold, padding_mas, min_points)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        finite = np.isfinite(s) & np.isfinite(values)
        if not np.any(finite):
            return s, values
        peak_index = int(np.nanargmax(np.where(finite, values, -np.inf)))
        half = max(int(min_points) // 2, 4)
        start = max(0, peak_index - half)
        stop = min(len(s), peak_index + half + 1)
    else:
        start = int(indices[0])
        stop = int(indices[-1]) + 1
    return s[start:stop], values[start:stop]


def sample_profile_for_opening_fit(
    image: np.ndarray,
    header: Dict[str, object],
    x_mas: float,
    y_mas: float,
    tangent_pa_deg: float,
    fixed_half_width_mas: Optional[float],
    threshold: float,
    padding_mas: float,
    min_points: int,
    sample_step_mas: float,
    pa_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    half_width = (
        float(fixed_half_width_mas)
        if fixed_half_width_mas is not None
        else default_scan_half_width_mas(_beam_mas(header))
    )
    return sample_wide_transverse_profile(
        image,
        header,
        x_mas,
        y_mas,
        tangent_pa_deg,
        half_width,
        sample_step_mas,
        pa_offset_deg,
    )


def profile_sigma_upper(s: np.ndarray, explicit_sigma_upper_mas: Optional[float]) -> float:
    if explicit_sigma_upper_mas is not None:
        return float(explicit_sigma_upper_mas)
    finite = np.asarray(s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        return max(float(np.nanmax(np.abs(finite))), 1e-4)
    return 1e-4


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if bool(flag) and start is None:
            start = idx
        elif not bool(flag) and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _run_distance_to_center(s: np.ndarray, start: int, stop: int) -> float:
    vals = np.asarray(s[start:stop], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("inf")
    low = float(np.nanmin(vals))
    high = float(np.nanmax(vals))
    if low <= 0.0 <= high:
        return 0.0
    return float(min(abs(low), abs(high)))


def select_fit_regions(
    s: np.ndarray,
    y: np.ndarray,
    threshold: float,
    padding_mas: float,
    min_points: int,
    baseline_guard_mas: float = 0.0,
    beam_mas: float = float("nan"),
    rms: float = float("nan"),
) -> FitRegions:
    finite = np.isfinite(s) & np.isfinite(y)
    support = finite & (y > threshold)
    empty = np.zeros_like(support, dtype=bool)
    runs = contiguous_true_runs(support)
    flags: List[str] = []

    chosen: Optional[Tuple[int, int]] = None
    if runs:
        chosen = min(
            runs,
            key=lambda run: (
                _run_distance_to_center(s, run[0], run[1]),
                -float(np.nanmax(np.where(finite[run[0] : run[1]], y[run[0] : run[1]], -np.inf))),
            ),
        )
    else:
        flags.append("no_detection_component")
        return FitRegions(empty, empty, empty, empty, empty, flags=tuple(flags))

    start, stop = chosen
    source_mask = np.zeros_like(support, dtype=bool)
    source_mask[start:stop] = True
    source_vals = np.asarray(s[source_mask & finite], dtype=float)
    source_min = float(np.nanmin(source_vals)) if source_vals.size else float("nan")
    source_max = float(np.nanmax(source_vals)) if source_vals.size else float("nan")
    center_offset = _run_distance_to_center(s, start, stop)
    peak_idx = int(np.nanargmax(np.where(source_mask & finite, y, -np.inf)))
    peak_mas = float(s[peak_idx]) if np.isfinite(s[peak_idx]) else float("nan")
    if np.isfinite(beam_mas) and beam_mas > 0.0 and center_offset > beam_mas:
        flags.append("component_far_from_center")

    pad = max(0.0, float(padding_mas))
    guard = max(0.0, float(baseline_guard_mas))
    fit_low = source_min - pad
    fit_high = source_max + pad
    fit_mask = finite & (s >= fit_low) & (s <= fit_high)

    left_mask = finite & (s < fit_low - guard)
    right_mask = finite & (s > fit_high + guard)
    baseline_mask = left_mask | right_mask

    finite_s = np.asarray(s[finite], dtype=float)
    if finite_s.size:
        scan_min = float(np.nanmin(finite_s))
        scan_max = float(np.nanmax(finite_s))
        ds = float(np.nanmedian(np.diff(np.sort(finite_s)))) if finite_s.size > 1 else 0.0
        edge_tol = 0.5 * abs(ds) if np.isfinite(ds) and ds > 0.0 else 0.0
        if fit_low <= scan_min + edge_tol or fit_high >= scan_max - edge_tol:
            flags.append("fit_window_touches_scan_edge")

    if np.count_nonzero(fit_mask) < min_points:
        flags.append("fit_window_too_few_points")
    if np.count_nonzero(baseline_mask) < min_points:
        flags.append("baseline_too_few_points")
    if np.count_nonzero(baseline_mask & support) > 0:
        flags.append("baseline_region_has_detection")

    left_vals = y[left_mask]
    right_vals = y[right_mask]
    left_med = robust_median(left_vals)
    right_med = robust_median(right_vals)
    if (
        np.isfinite(rms)
        and rms > 0.0
        and np.isfinite(left_med)
        and np.isfinite(right_med)
        and abs(left_med - right_med) > rms
    ):
        flags.append("baseline_lr_diff_gt_rms")

    return FitRegions(
        fit_mask=fit_mask,
        source_mask=source_mask & finite,
        baseline_mask=baseline_mask,
        baseline_left_mask=left_mask,
        baseline_right_mask=right_mask,
        source_min_mas=source_min,
        source_max_mas=source_max,
        fit_min_mas=float(np.nanmin(s[fit_mask])) if np.any(fit_mask) else float("nan"),
        fit_max_mas=float(np.nanmax(s[fit_mask])) if np.any(fit_mask) else float("nan"),
        component_center_offset_mas=center_offset,
        component_peak_mas=peak_mas,
        flags=tuple(flags),
    )


def select_fit_window(
    s: np.ndarray,
    y: np.ndarray,
    threshold: float,
    padding_mas: float,
    min_points: int,
) -> np.ndarray:
    return select_fit_regions(s, y, threshold, padding_mas, min_points).fit_mask


def fit_transverse_gaussian(
    s: np.ndarray,
    y: np.ndarray,
    threshold: float,
    padding_mas: float,
    min_points: int,
    sigma_upper_mas: float,
    rms: float = float("nan"),
    beam_mas: float = float("nan"),
    baseline_guard_mas: float = 0.0,
) -> FitResult:
    regions = select_fit_regions(s, y, threshold, padding_mas, min_points, baseline_guard_mas, beam_mas, rms)
    baseline_values = np.asarray(y[regions.baseline_mask], dtype=float)
    baseline = robust_median(baseline_values)
    left_median = robust_median(y[regions.baseline_left_mask])
    right_median = robust_median(y[regions.baseline_right_mask])
    flags = list(regions.flags)
    if np.isfinite(rms) and rms > 0.0 and np.isfinite(baseline) and abs(baseline) > rms:
        flags.append("baseline_abs_gt_rms")
    flag_text = ";".join(dict.fromkeys(flags))
    common = {
        "baseline": baseline,
        "baseline_method": "fixed_robust_median_outside_fit_window",
        "baseline_flag": flag_text,
        "baseline_n": int(np.count_nonzero(regions.baseline_mask)),
        "baseline_left_median": left_median,
        "baseline_right_median": right_median,
        "source_min_mas": regions.source_min_mas,
        "source_max_mas": regions.source_max_mas,
        "component_center_offset_mas": regions.component_center_offset_mas,
    }

    if "no_detection_component" in flags:
        return FitResult(False, "no_detection_component", **common)
    if not np.isfinite(baseline):
        return FitResult(False, "no_baseline_region", **common)

    mask = regions.fit_mask
    if np.count_nonzero(mask) < min_points:
        return FitResult(False, "too_few_fit_points", n_fit=int(np.count_nonzero(mask)), **common)

    xfit = np.asarray(s[mask], dtype=float)
    yfit = np.asarray(y[mask], dtype=float)
    finite = np.isfinite(xfit) & np.isfinite(yfit)
    xfit = xfit[finite]
    yfit = yfit[finite]
    if xfit.size < min_points:
        return FitResult(False, "too_few_finite_points", n_fit=int(xfit.size), **common)

    ymin = float(np.nanmin(yfit))
    ymax = float(np.nanmax(yfit))
    yrange = max(ymax - ymin, 1e-12)
    amp0 = max(float(ymax - baseline), 1e-12)
    if ymax <= baseline:
        return FitResult(
            False,
            "peak_below_baseline",
            amplitude=amp0,
            n_fit=int(xfit.size),
            fit_min_mas=float(np.nanmin(xfit)),
            fit_max_mas=float(np.nanmax(xfit)),
            **common,
        )
    mu0 = float(xfit[int(np.nanargmax(yfit))])
    sigma0 = max(float((np.nanmax(xfit) - np.nanmin(xfit)) / 4.0), 0.05)
    sigma_lower = max(float(np.nanmedian(np.diff(np.sort(xfit)))) if xfit.size > 1 else 0.01, 1e-4)
    sigma_upper = max(float(sigma_upper_mas), sigma_lower * 2.0)

    lower = [0.0, float(np.nanmin(xfit)), sigma_lower]
    upper = [max(3.0 * yrange, amp0 * 5.0), float(np.nanmax(xfit)), sigma_upper]
    p0 = [
        float(np.clip(amp0, lower[0] + 1e-15, upper[0])),
        float(np.clip(mu0, lower[1], upper[1])),
        float(np.clip(sigma0, lower[2], upper[2])),
    ]

    try:
        def model_fixed(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
            return gaussian_model(x, baseline, amplitude, mu, sigma)

        popt, pcov = curve_fit(
            model_fixed,
            xfit,
            yfit,
            p0=p0,
            jac=gaussian_fixed_baseline_jacobian,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception as exc:
        return FitResult(
            False,
            f"curve_fit_failed:{exc}",
            n_fit=int(xfit.size),
            fit_min_mas=float(np.nanmin(xfit)),
            fit_max_mas=float(np.nanmax(xfit)),
            **common,
        )

    amplitude, mu, sigma = [float(v) for v in popt]
    sigma = abs(sigma)
    ymodel = gaussian_model(xfit, baseline, amplitude, mu, sigma)
    rmse = float(np.sqrt(np.nanmean((yfit - ymodel) ** 2)))
    fwhm = FWHM_FACTOR * sigma
    fwhm_sigma = float("nan")
    if pcov is not None and np.shape(pcov) == (3, 3) and np.isfinite(pcov[2, 2]) and pcov[2, 2] >= 0:
        fwhm_sigma = FWHM_FACTOR * math.sqrt(float(pcov[2, 2]))

    if not np.isfinite(fwhm) or fwhm <= 0:
        return FitResult(False, "non_positive_fwhm", amplitude=amplitude, mu_mas=mu, sigma_mas=sigma, fwhm_mas=fwhm, fwhm_sigma_mas=fwhm_sigma, rmse=rmse, n_fit=int(xfit.size), fit_min_mas=float(np.nanmin(xfit)), fit_max_mas=float(np.nanmax(xfit)), **common)
    if np.isfinite(sigma_upper) and sigma_upper > 0 and sigma >= 0.995 * float(sigma_upper):
        return FitResult(False, "sigma_hit_upper_bound", amplitude=amplitude, mu_mas=mu, sigma_mas=sigma, fwhm_mas=fwhm, fwhm_sigma_mas=fwhm_sigma, rmse=rmse, n_fit=int(xfit.size), fit_min_mas=float(np.nanmin(xfit)), fit_max_mas=float(np.nanmax(xfit)), **common)
    return FitResult(True, "ok", amplitude=amplitude, mu_mas=mu, sigma_mas=sigma, fwhm_mas=fwhm, fwhm_sigma_mas=fwhm_sigma, rmse=rmse, n_fit=int(xfit.size), fit_min_mas=float(np.nanmin(xfit)), fit_max_mas=float(np.nanmax(xfit)), **common)


def load_ridge(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "ridge_points" not in data:
        raise ValueError(f"{path} does not contain ridge_points")
    return data


def intrinsic_width(raw_fwhm_mas: float, beam_mas: float) -> float:
    if not np.isfinite(raw_fwhm_mas) or not np.isfinite(beam_mas):
        return float("nan")
    val = raw_fwhm_mas * raw_fwhm_mas - beam_mas * beam_mas
    if val <= 0:
        return float("nan")
    return float(math.sqrt(val))


def full_opening_angle(width_mas: float, distance_mas: float) -> float:
    if not np.isfinite(width_mas) or not np.isfinite(distance_mas) or distance_mas <= 0:
        return float("nan")
    return float(math.degrees(2.0 * math.atan(0.5 * width_mas / distance_mas)))


def opening_angle_sigma(width_mas: float, width_sigma_mas: float, distance_mas: float) -> float:
    if (
        not np.isfinite(width_mas)
        or not np.isfinite(width_sigma_mas)
        or not np.isfinite(distance_mas)
        or width_sigma_mas < 0.0
        or distance_mas <= 0.0
    ):
        return float("nan")
    ratio = 0.5 * float(width_mas) / float(distance_mas)
    derivative_rad_per_mas = (1.0 / float(distance_mas)) / (1.0 + ratio * ratio)
    return float(math.degrees(derivative_rad_per_mas * float(width_sigma_mas)))


def finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def median_path_spacing(records: Sequence[OpeningRecord]) -> float:
    path = np.asarray([r.path_mas for r in records], dtype=float)
    path = np.sort(path[np.isfinite(path)])
    if path.size < 2:
        return float("nan")
    diff = np.diff(path)
    diff = diff[np.isfinite(diff) & (diff > 0.0)]
    return float(np.nanmedian(diff)) if diff.size else float("nan")


def bootstrap_median_sigma(
    values: Sequence[float],
    n_bootstrap: int,
    seed: int,
    block_size: int = 1,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2 or n_bootstrap <= 0:
        return {
            "sigma": float("nan"),
            "median_of_bootstrap": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "n": n,
            "block_size": int(max(1, block_size)),
            "n_bootstrap": int(n_bootstrap),
        }

    rng = np.random.default_rng(int(seed))
    block_size = int(max(1, min(block_size, n)))
    medians = np.empty(int(n_bootstrap), dtype=float)
    if block_size <= 1:
        for i in range(int(n_bootstrap)):
            medians[i] = float(np.nanmedian(arr[rng.integers(0, n, size=n)]))
    else:
        n_blocks = int(math.ceil(n / block_size))
        block_offsets = np.arange(block_size, dtype=int)
        for i in range(int(n_bootstrap)):
            starts = rng.integers(0, n, size=n_blocks)
            indices = ((starts[:, None] + block_offsets[None, :]) % n).ravel()[:n]
            medians[i] = float(np.nanmedian(arr[indices]))

    p16, p84 = np.nanpercentile(medians, [15.865, 84.135])
    return {
        "sigma": float(0.5 * (p84 - p16)),
        "median_of_bootstrap": float(np.nanmedian(medians)),
        "p16": float(p16),
        "p84": float(p84),
        "n": n,
        "block_size": block_size,
        "n_bootstrap": int(n_bootstrap),
    }


def bootstrap_median_sigma_with_measurement(
    values: Sequence[float],
    sigmas: Sequence[float],
    n_bootstrap: int,
    seed: int,
    block_size: int = 1,
    resample: bool = True,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    err = np.asarray(sigmas, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(err) & (err >= 0.0)
    arr = arr[valid]
    err = err[valid]
    n = int(arr.size)
    if n < 2 or n_bootstrap <= 0:
        return {
            "sigma": float("nan"),
            "median_of_bootstrap": float("nan"),
            "p16": float("nan"),
            "p84": float("nan"),
            "n": n,
            "block_size": int(max(1, block_size)),
            "n_bootstrap": int(n_bootstrap),
            "resample": bool(resample),
        }

    rng = np.random.default_rng(int(seed))
    block_size = int(max(1, min(block_size, n)))
    medians = np.empty(int(n_bootstrap), dtype=float)
    if not resample:
        for i in range(int(n_bootstrap)):
            medians[i] = float(np.nanmedian(arr + rng.normal(0.0, err, size=n)))
    elif block_size <= 1:
        for i in range(int(n_bootstrap)):
            indices = rng.integers(0, n, size=n)
            medians[i] = float(np.nanmedian(arr[indices] + rng.normal(0.0, err[indices], size=n)))
    else:
        n_blocks = int(math.ceil(n / block_size))
        block_offsets = np.arange(block_size, dtype=int)
        for i in range(int(n_bootstrap)):
            starts = rng.integers(0, n, size=n_blocks)
            indices = ((starts[:, None] + block_offsets[None, :]) % n).ravel()[:n]
            medians[i] = float(np.nanmedian(arr[indices] + rng.normal(0.0, err[indices], size=n)))

    p16, p84 = np.nanpercentile(medians, [15.865, 84.135])
    return {
        "sigma": float(0.5 * (p84 - p16)),
        "median_of_bootstrap": float(np.nanmedian(medians)),
        "p16": float(p16),
        "p84": float(p84),
        "n": n,
        "block_size": block_size,
        "n_bootstrap": int(n_bootstrap),
        "resample": bool(resample),
    }


def five_point_smoothed_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    smoothed = np.full_like(arr, np.nan, dtype=float)
    for idx in range(len(arr)):
        start = max(0, idx - 2)
        stop = min(len(arr), idx + 3)
        window = arr[start:stop]
        window = window[np.isfinite(window)]
        if window.size:
            smoothed[idx] = float(np.nanmean(window))
    return smoothed


def separation_mask(distances: np.ndarray, min_mas: float, max_mas: Optional[float]) -> np.ndarray:
    valid = np.isfinite(distances) & (distances > float(min_mas))
    if max_mas is not None and np.isfinite(max_mas):
        valid &= distances <= float(max_mas)
    return valid


def analysis_separation_bounds(args: argparse.Namespace) -> Tuple[float, Optional[float]]:
    min_value = getattr(args, "analysis_sep_min", None)
    if min_value is None:
        min_value = getattr(args, "angle_rmin", 0.5)
    max_value = getattr(args, "analysis_sep_max", None)
    return float(min_value), None if max_value is None else float(max_value)


def five_point_median_angle(records: Sequence[OpeningRecord], sep_min_mas: float, sep_max_mas: Optional[float]) -> float:
    widths = np.asarray([r.intrinsic_fwhm_mas for r in records], dtype=float)
    distances = np.asarray([r.path_mas for r in records], dtype=float)
    valid = np.isfinite(widths) & np.isfinite(distances)
    if np.count_nonzero(valid) < 5:
        return float("nan")
    smoothed = five_point_smoothed_values(widths)
    angles = np.asarray([full_opening_angle(w, d) for w, d in zip(smoothed, distances)], dtype=float)
    return finite_median(angles[valid & separation_mask(distances, sep_min_mas, sep_max_mas)])


def opening_angle_values(
    records: Sequence[OpeningRecord],
    sep_min_mas: float,
    sep_max_mas: Optional[float],
    intrinsic: bool = True,
    smooth5: bool = False,
) -> np.ndarray:
    distances = np.asarray([r.path_mas for r in records], dtype=float)
    if intrinsic:
        widths = np.asarray([r.intrinsic_fwhm_mas for r in records], dtype=float)
    else:
        widths = np.asarray([r.fit.fwhm_mas if r.fit.success else float("nan") for r in records], dtype=float)
    if smooth5:
        widths = five_point_smoothed_values(widths)
    angles = np.asarray([full_opening_angle(w, d) for w, d in zip(widths, distances)], dtype=float)
    valid = np.isfinite(angles) & separation_mask(distances, sep_min_mas, sep_max_mas)
    return angles[valid]


def opening_angle_values_and_errors(
    records: Sequence[OpeningRecord],
    sep_min_mas: float,
    sep_max_mas: Optional[float],
    intrinsic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.asarray([r.path_mas for r in records], dtype=float)
    if intrinsic:
        values = np.asarray([r.opening_angle_deg for r in records], dtype=float)
        errors = np.asarray([r.pa_sweep_opening_angle_intrinsic_rms_deg for r in records], dtype=float)
    else:
        values = np.asarray([r.opening_angle_raw_deg for r in records], dtype=float)
        errors = np.asarray([r.pa_sweep_opening_angle_raw_rms_deg for r in records], dtype=float)
    valid = np.isfinite(values) & separation_mask(distances, sep_min_mas, sep_max_mas)
    return values[valid], errors[valid]


def median_error_summary(
    records: Sequence[OpeningRecord],
    beam_mas: float,
    sep_min_mas: float,
    sep_max_mas: Optional[float],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    spacing = median_path_spacing(records)
    if np.isfinite(spacing) and spacing > 0 and np.isfinite(beam_mas) and beam_mas > 0:
        block_target_mas = 0.5 * float(beam_mas)
        block_size = int(max(1, round(block_target_mas / spacing)))
    else:
        block_target_mas = float("nan")
        block_size = 1
    block_length_mas = float(block_size * spacing) if np.isfinite(spacing) else float("nan")
    raw_values = opening_angle_values(records, sep_min_mas, sep_max_mas, intrinsic=False, smooth5=False)
    intrinsic_values = opening_angle_values(records, sep_min_mas, sep_max_mas, intrinsic=True, smooth5=False)
    smooth_values = opening_angle_values(records, sep_min_mas, sep_max_mas, intrinsic=True, smooth5=True)
    raw_pa_values, raw_pa_errors = opening_angle_values_and_errors(records, sep_min_mas, sep_max_mas, intrinsic=False)
    intrinsic_pa_values, intrinsic_pa_errors = opening_angle_values_and_errors(records, sep_min_mas, sep_max_mas, intrinsic=True)
    return {
        "method": "bootstrap_median",
        "note": "naive resamples individual slices; block resamples contiguous slices over about half the beam; *_with_pa_sweep also perturbs each angle by its PA-sweep RMS.",
        "path_spacing_mas": spacing,
        "beam_mas": beam_mas,
        "block_reference": "half_beam",
        "block_reference_fraction_of_beam": 0.5,
        "block_target_mas": block_target_mas,
        "block_size_slices": block_size,
        "block_length_mas": block_length_mas,
        "raw_naive": bootstrap_median_sigma(raw_values, n_bootstrap, seed + 11, block_size=1),
        "intrinsic_naive": bootstrap_median_sigma(intrinsic_values, n_bootstrap, seed + 23, block_size=1),
        "intrinsic_smooth5_naive": bootstrap_median_sigma(smooth_values, n_bootstrap, seed + 31, block_size=1),
        "raw_block": bootstrap_median_sigma(raw_values, n_bootstrap, seed + 41, block_size=block_size),
        "intrinsic_block": bootstrap_median_sigma(intrinsic_values, n_bootstrap, seed + 53, block_size=block_size),
        "intrinsic_smooth5_block": bootstrap_median_sigma(smooth_values, n_bootstrap, seed + 61, block_size=block_size),
        "raw_pa_sweep_measurement": bootstrap_median_sigma_with_measurement(raw_pa_values, raw_pa_errors, n_bootstrap, seed + 71, block_size=1, resample=False),
        "intrinsic_pa_sweep_measurement": bootstrap_median_sigma_with_measurement(intrinsic_pa_values, intrinsic_pa_errors, n_bootstrap, seed + 83, block_size=1, resample=False),
        "raw_block_with_pa_sweep": bootstrap_median_sigma_with_measurement(raw_pa_values, raw_pa_errors, n_bootstrap, seed + 97, block_size=block_size, resample=True),
        "intrinsic_block_with_pa_sweep": bootstrap_median_sigma_with_measurement(intrinsic_pa_values, intrinsic_pa_errors, n_bootstrap, seed + 109, block_size=block_size, resample=True),
    }


def power_law_fit(
    records: Sequence[OpeningRecord],
    sep_min_mas: float,
    sep_max_mas: Optional[float],
    smooth5: bool = False,
) -> Dict[str, float]:
    distances = np.asarray([r.path_mas for r in records], dtype=float)
    widths = np.asarray([r.intrinsic_fwhm_mas for r in records], dtype=float)
    if smooth5:
        widths = five_point_smoothed_values(widths)
    valid = separation_mask(distances, sep_min_mas, sep_max_mas) & np.isfinite(widths) & (widths > 0.0)
    n = int(np.count_nonzero(valid))
    empty = {
        "k": float("nan"),
        "k_sigma": float("nan"),
        "intercept_log10": float("nan"),
        "amplitude_mas_at_1mas": float("nan"),
        "n": n,
        "r_min_mas": float("nan"),
        "r_max_mas": float("nan"),
        "scatter_log10": float("nan"),
        "smooth5": bool(smooth5),
    }
    if n < 2:
        return empty
    x = np.log10(distances[valid])
    y = np.log10(widths[valid])
    k, intercept = np.polyfit(x, y, 1)
    residual = y - (k * x + intercept)
    scatter = float(np.sqrt(np.nanmean(residual * residual))) if residual.size else float("nan")
    k_sigma = float("nan")
    if n > 2 and np.nanvar(x) > 0:
        dof = n - 2
        residual_var = float(np.nansum(residual * residual) / dof)
        k_sigma = math.sqrt(residual_var / float(np.nansum((x - np.nanmean(x)) ** 2)))
    return {
        "k": float(k),
        "k_sigma": float(k_sigma),
        "intercept_log10": float(intercept),
        "amplitude_mas_at_1mas": float(10.0 ** intercept),
        "n": n,
        "r_min_mas": float(np.nanmin(distances[valid])),
        "r_max_mas": float(np.nanmax(distances[valid])),
        "scatter_log10": scatter,
        "smooth5": bool(smooth5),
    }


def pa_sweep_width_rms(
    image: np.ndarray,
    header: Dict[str, object],
    point: Dict[str, object],
    threshold: float,
    padding_mas: float,
    min_points: int,
    sigma_upper_mas: Optional[float],
    fixed_half_width_mas: Optional[float],
    sample_step_mas: float,
    beam_mas: float,
    rms: float,
    baseline_guard_mas: float,
    sweep_range_deg: float,
    sweep_step_deg: float,
    cancel_event: object = None,
) -> Tuple[float, float, int]:
    raw_widths: List[float] = []
    intrinsic_widths: List[float] = []
    offsets = np.arange(-sweep_range_deg, sweep_range_deg + 0.5 * sweep_step_deg, sweep_step_deg)
    for offset in offsets:
        _check_cancelled(cancel_event)
        s, y = sample_profile_for_opening_fit(
            image,
            header,
            float(point["x_mas"]),
            float(point["y_mas"]),
            float(point["tangent_pa_deg"]),
            fixed_half_width_mas,
            threshold,
            padding_mas,
            min_points,
            sample_step_mas,
            pa_offset_deg=float(offset),
        )
        fit = fit_transverse_gaussian(
            s,
            y,
            threshold,
            padding_mas,
            min_points,
            profile_sigma_upper(s, sigma_upper_mas),
            rms,
            beam_mas,
            baseline_guard_mas,
        )
        if not fit.success:
            continue
        raw_widths.append(fit.fwhm_mas)
        intrinsic_widths.append(intrinsic_width(fit.fwhm_mas, beam_mas))
    raw_arr = np.asarray(raw_widths, dtype=float)
    int_arr = np.asarray(intrinsic_widths, dtype=float)
    raw_arr = raw_arr[np.isfinite(raw_arr)]
    int_arr = int_arr[np.isfinite(int_arr)]
    raw_rms = float(np.sqrt(np.nanmean((raw_arr - np.nanmedian(raw_arr)) ** 2))) if raw_arr.size >= 2 else float("nan")
    int_rms = float(np.sqrt(np.nanmean((int_arr - np.nanmedian(int_arr)) ** 2))) if int_arr.size >= 2 else float("nan")
    return raw_rms, int_rms, int(raw_arr.size)


def _pa_sweep_worker_count(args: argparse.Namespace) -> int:
    requested = int(getattr(args, "pa_sweep_workers", 1) or 1)
    if requested > 0:
        return requested
    return 1


def _reset_pa_sweep(record: OpeningRecord) -> None:
    record.pa_sweep_raw_rms_mas = float("nan")
    record.pa_sweep_intrinsic_rms_mas = float("nan")
    record.pa_sweep_opening_angle_raw_rms_deg = float("nan")
    record.pa_sweep_opening_angle_intrinsic_rms_deg = float("nan")
    record.pa_sweep_success_count = 0


def apply_pa_sweep_to_records(
    image: np.ndarray,
    header: Dict[str, object],
    records: Sequence[OpeningRecord],
    args: argparse.Namespace,
    beam_mas: float,
    threshold: float,
    padding_mas: float,
    min_points: int,
    sigma_upper_mas: Optional[float],
    fixed_half_width_mas: Optional[float],
    sample_step_mas: float,
    rms: float,
    baseline_guard_mas: float,
    sep_min_mas: float,
    sep_max_mas: Optional[float],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: object = None,
) -> None:
    for record in records:
        _reset_pa_sweep(record)
    if not bool(getattr(args, "pa_sweep", False)):
        return

    distances = np.asarray([r.path_mas for r in records], dtype=float)
    eligible_mask = np.asarray([r.fit.success for r in records], dtype=bool)
    if bool(getattr(args, "pa_sweep_analysis_only", False)):
        eligible_mask &= separation_mask(distances, sep_min_mas, sep_max_mas)
    eligible_indices = [idx for idx, flag in enumerate(eligible_mask) if flag]
    total = len(eligible_indices)
    if total == 0:
        return

    def run_one(idx: int) -> Tuple[int, float, float, int]:
        _check_cancelled(cancel_event)
        record = records[idx]
        point = {
            "x_mas": record.x_mas,
            "y_mas": record.y_mas,
            "tangent_pa_deg": record.tangent_pa_deg,
        }
        raw_rms, int_rms, sweep_count = pa_sweep_width_rms(
            image,
            header,
            point,
            threshold,
            padding_mas,
            min_points,
            sigma_upper_mas,
            fixed_half_width_mas,
            sample_step_mas,
            beam_mas,
            rms,
            baseline_guard_mas,
            float(getattr(args, "pa_sweep_range", 15.0)),
            float(getattr(args, "pa_sweep_step", 1.0)),
            cancel_event=cancel_event,
        )
        return idx, raw_rms, int_rms, sweep_count

    completed = 0
    workers = _pa_sweep_worker_count(args)
    if workers <= 1 or total <= 1:
        for idx in eligible_indices:
            result = run_one(idx)
            _assign_pa_sweep_result(records[result[0]], *result[1:])
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
        future_to_index = {executor.submit(run_one, idx): idx for idx in eligible_indices}
        for future in concurrent.futures.as_completed(future_to_index):
            _check_cancelled(cancel_event)
            idx, raw_rms, int_rms, sweep_count = future.result()
            _assign_pa_sweep_result(records[idx], raw_rms, int_rms, sweep_count)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)


def _assign_pa_sweep_result(record: OpeningRecord, raw_rms: float, int_rms: float, sweep_count: int) -> None:
    record.pa_sweep_raw_rms_mas = raw_rms
    record.pa_sweep_intrinsic_rms_mas = int_rms
    record.pa_sweep_opening_angle_raw_rms_deg = opening_angle_sigma(record.fit.fwhm_mas, raw_rms, record.path_mas)
    record.pa_sweep_opening_angle_intrinsic_rms_deg = opening_angle_sigma(record.intrinsic_fwhm_mas, int_rms, record.path_mas)
    record.pa_sweep_success_count = int(sweep_count)


def summarize_opening_records(
    records: Sequence[OpeningRecord],
    beam_mas: float,
    pix_mas: float,
    rms: float,
    threshold: float,
    half_width: Optional[float],
    sample_step: float,
    padding: float,
    baseline_guard: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    sep_min, sep_max = analysis_separation_bounds(args)
    valid_raw = [
        r.opening_angle_raw_deg
        for r in records
        if r.fit.success and r.path_mas > sep_min and (sep_max is None or r.path_mas <= sep_max)
    ]
    valid_intrinsic = [
        r.opening_angle_deg
        for r in records
        if r.fit.success
        and np.isfinite(r.intrinsic_fwhm_mas)
        and r.path_mas > sep_min
        and (sep_max is None or r.path_mas <= sep_max)
    ]
    unresolved = [r for r in records if r.fit.success and not np.isfinite(r.intrinsic_fwhm_mas)]
    fit_fail = [r for r in records if not r.fit.success]
    scan_min_values = [r.scan_min_mas for r in records if np.isfinite(r.scan_min_mas)]
    scan_max_values = [r.scan_max_mas for r in records if np.isfinite(r.scan_max_mas)]
    max_scan_abs = float("nan")
    if scan_min_values or scan_max_values:
        max_scan_abs = float(max([abs(v) for v in [*scan_min_values, *scan_max_values]]))
    baseline_flag_counts: Dict[str, int] = {}
    for record in records:
        for flag in str(record.fit.baseline_flag or "").split(";"):
            if not flag:
                continue
            baseline_flag_counts[flag] = baseline_flag_counts.get(flag, 0) + 1
    sweep_raw_width_errors = [
        r.pa_sweep_raw_rms_mas
        for r in records
        if r.fit.success
        and r.path_mas > sep_min
        and (sep_max is None or r.path_mas <= sep_max)
        and np.isfinite(r.pa_sweep_raw_rms_mas)
    ]
    sweep_intrinsic_width_errors = [
        r.pa_sweep_intrinsic_rms_mas
        for r in records
        if r.fit.success
        and r.path_mas > sep_min
        and (sep_max is None or r.path_mas <= sep_max)
        and np.isfinite(r.pa_sweep_intrinsic_rms_mas)
    ]
    sweep_raw_angle_errors = [
        r.pa_sweep_opening_angle_raw_rms_deg
        for r in records
        if r.fit.success
        and r.path_mas > sep_min
        and (sep_max is None or r.path_mas <= sep_max)
        and np.isfinite(r.pa_sweep_opening_angle_raw_rms_deg)
    ]
    sweep_intrinsic_angle_errors = [
        r.pa_sweep_opening_angle_intrinsic_rms_deg
        for r in records
        if r.fit.success
        and r.path_mas > sep_min
        and (sep_max is None or r.path_mas <= sep_max)
        and np.isfinite(r.pa_sweep_opening_angle_intrinsic_rms_deg)
    ]
    return {
        "beam_mas": beam_mas,
        "pixel_mas": pix_mas,
        "rms_jy_per_beam": rms,
        "fit_threshold_jy_per_beam": threshold,
        "slice_half_width_mas": float("nan") if half_width is None else half_width,
        "slice_scan_mode": "fixed_full_scan" if getattr(args, "slice_half_width", None) is None else "fixed_half_width",
        "slice_scan_max_abs_mas": max_scan_abs,
        "slice_scan_min_mas": float(np.nanmin(scan_min_values)) if scan_min_values else float("nan"),
        "slice_scan_max_mas": float(np.nanmax(scan_max_values)) if scan_max_values else float("nan"),
        "sample_step_mas": sample_step,
        "fit_padding_mas": padding,
        "baseline_guard_mas": baseline_guard,
        "baseline_method": "fixed_robust_median_outside_fit_window",
        "baseline_flag_count": int(sum(1 for r in records if r.fit.baseline_flag)),
        "baseline_flag_counts": baseline_flag_counts,
        "angle_rmin_mas": sep_min,
        "analysis_separation_min_mas": sep_min,
        "analysis_separation_max_mas": sep_max,
        "record_count": len(records),
        "fit_success_count": int(sum(1 for r in records if r.fit.success)),
        "unresolved_count": len(unresolved),
        "fit_failure_count": len(fit_fail),
        "angle_count_raw": int(np.count_nonzero(np.isfinite(valid_raw))),
        "angle_count_intrinsic": int(np.count_nonzero(np.isfinite(valid_intrinsic))),
        "median_opening_angle_raw_deg": finite_median(valid_raw),
        "median_opening_angle_deg": finite_median(valid_intrinsic),
        "mean_opening_angle_raw_deg": float(np.nanmean(valid_raw)) if valid_raw else float("nan"),
        "mean_opening_angle_deg": float(np.nanmean(valid_intrinsic)) if valid_intrinsic else float("nan"),
        "five_point_smoothed_median_opening_angle_deg": five_point_median_angle(records, sep_min, sep_max),
        "power_law_fit": power_law_fit(records, sep_min, sep_max, smooth5=False),
        "power_law_fit_smoothed5": power_law_fit(records, sep_min, sep_max, smooth5=True),
        "median_error": median_error_summary(records, beam_mas, sep_min, sep_max, args.bootstrap_count, args.bootstrap_seed),
        "pa_sweep": {
            "enabled": bool(args.pa_sweep),
            "range_deg": args.pa_sweep_range,
            "step_deg": args.pa_sweep_step,
            "analysis_only": bool(getattr(args, "pa_sweep_analysis_only", False)),
            "workers": _pa_sweep_worker_count(args) if bool(args.pa_sweep) else 0,
            "width_error_count_raw": len(sweep_raw_width_errors),
            "width_error_count_intrinsic": len(sweep_intrinsic_width_errors),
            "median_raw_width_rms_mas": finite_median(sweep_raw_width_errors),
            "median_intrinsic_width_rms_mas": finite_median(sweep_intrinsic_width_errors),
            "median_raw_opening_angle_rms_deg": finite_median(sweep_raw_angle_errors),
            "median_intrinsic_opening_angle_rms_deg": finite_median(sweep_intrinsic_angle_errors),
        },
    }


def measure_opening(
    image: np.ndarray,
    header: Dict[str, object],
    ridge: Dict[str, object],
    args: argparse.Namespace,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    cancel_event: object = None,
) -> Tuple[List[OpeningRecord], Dict[str, object]]:
    beam_mas = _beam_mas(header)
    pix_mas = _pixel_mas(header)
    rms = robust_corner_rms(image)
    threshold = args.fit_threshold if args.fit_threshold is not None else args.fit_threshold_snr * rms
    fixed_half_width = args.slice_half_width if args.slice_half_width is not None else default_scan_half_width_mas(beam_mas)
    sample_step = args.sample_step if args.sample_step is not None else max(pix_mas / 2.0, 0.01)
    padding = args.fit_padding if args.fit_padding is not None else default_fit_padding_mas(beam_mas)
    baseline_guard = (
        args.baseline_guard
        if getattr(args, "baseline_guard", None) is not None
        else default_baseline_guard_mas(beam_mas)
    )
    sigma_upper = args.sigma_upper if args.sigma_upper is not None else None

    records: List[OpeningRecord] = []
    points = list(ridge["ridge_points"])
    if args.max_records is not None:
        points = points[: max(0, int(args.max_records))]
    for idx, point in enumerate(points):
        _check_cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback("Fitting slices", idx, len(points))
        path_mas = float(point.get("path_mas", point.get("radial_mas", float("nan"))))
        tangent_pa = float(point.get("tangent_pa_deg", float("nan")))
        if not np.isfinite(path_mas) or not np.isfinite(tangent_pa) or path_mas <= 0:
            continue
        s, y = sample_profile_for_opening_fit(
            image,
            header,
            float(point["x_mas"]),
            float(point["y_mas"]),
            tangent_pa,
            fixed_half_width,
            threshold,
            padding,
            args.min_fit_points,
            sample_step,
        )
        scan_min = float(np.nanmin(s)) if len(s) else float("nan")
        scan_max = float(np.nanmax(s)) if len(s) else float("nan")
        fit = fit_transverse_gaussian(
            s,
            y,
            threshold,
            padding,
            args.min_fit_points,
            profile_sigma_upper(s, sigma_upper),
            rms,
            beam_mas,
            baseline_guard,
        )
        intrinsic = intrinsic_width(fit.fwhm_mas, beam_mas) if fit.success else float("nan")
        raw_angle = full_opening_angle(fit.fwhm_mas, path_mas) if fit.success else float("nan")
        angle = full_opening_angle(intrinsic, path_mas)
        records.append(
            OpeningRecord(
                index=idx,
                radial_mas=float(point.get("radial_mas", float("nan"))),
                path_mas=path_mas,
                x_mas=float(point["x_mas"]),
                y_mas=float(point["y_mas"]),
                tangent_pa_deg=tangent_pa,
                scan_min_mas=scan_min,
                scan_max_mas=scan_max,
                fit=fit,
                intrinsic_fwhm_mas=intrinsic,
                opening_angle_raw_deg=raw_angle,
                opening_angle_deg=angle,
                pa_sweep_raw_rms_mas=float("nan"),
                pa_sweep_intrinsic_rms_mas=float("nan"),
                pa_sweep_opening_angle_raw_rms_deg=float("nan"),
                pa_sweep_opening_angle_intrinsic_rms_deg=float("nan"),
                pa_sweep_success_count=0,
            )
        )

    sep_min, sep_max = analysis_separation_bounds(args)
    apply_pa_sweep_to_records(
        image,
        header,
        records,
        args,
        beam_mas,
        threshold,
        padding,
        args.min_fit_points,
        sigma_upper,
        fixed_half_width,
        sample_step,
        rms,
        baseline_guard,
        sep_min,
        sep_max,
        progress_callback=(lambda done, total: progress_callback("PA sweep", done, total)) if progress_callback is not None else None,
        cancel_event=cancel_event,
    )
    summary = summarize_opening_records(records, beam_mas, pix_mas, rms, threshold, fixed_half_width, sample_step, padding, baseline_guard, args)
    return records, summary


def records_to_json(records: Sequence[OpeningRecord]) -> List[Dict[str, object]]:
    rows = []
    for r in records:
        rows.append(
            {
                "index": r.index,
                "radial_mas": r.radial_mas,
                "path_mas": r.path_mas,
                "x_mas": r.x_mas,
                "y_mas": r.y_mas,
                "tangent_pa_deg": r.tangent_pa_deg,
                "scan_min_mas": r.scan_min_mas,
                "scan_max_mas": r.scan_max_mas,
                "fit_success": r.fit.success,
                "fit_reason": r.fit.reason,
                "baseline": r.fit.baseline,
                "baseline_method": r.fit.baseline_method,
                "baseline_flag": r.fit.baseline_flag,
                "baseline_n": r.fit.baseline_n,
                "baseline_left_median": r.fit.baseline_left_median,
                "baseline_right_median": r.fit.baseline_right_median,
                "amplitude": r.fit.amplitude,
                "mu_mas": r.fit.mu_mas,
                "sigma_mas": r.fit.sigma_mas,
                "fwhm_mas": r.fit.fwhm_mas,
                "fwhm_sigma_mas": r.fit.fwhm_sigma_mas,
                "intrinsic_fwhm_mas": r.intrinsic_fwhm_mas,
                "opening_angle_raw_deg": r.opening_angle_raw_deg,
                "opening_angle_deg": r.opening_angle_deg,
                "rmse": r.fit.rmse,
                "n_fit": r.fit.n_fit,
                "fit_min_mas": r.fit.fit_min_mas,
                "fit_max_mas": r.fit.fit_max_mas,
                "source_min_mas": r.fit.source_min_mas,
                "source_max_mas": r.fit.source_max_mas,
                "component_center_offset_mas": r.fit.component_center_offset_mas,
                "pa_sweep_raw_rms_mas": r.pa_sweep_raw_rms_mas,
                "pa_sweep_intrinsic_rms_mas": r.pa_sweep_intrinsic_rms_mas,
                "pa_sweep_opening_angle_raw_rms_deg": r.pa_sweep_opening_angle_raw_rms_deg,
                "pa_sweep_opening_angle_intrinsic_rms_deg": r.pa_sweep_opening_angle_intrinsic_rms_deg,
                "pa_sweep_success_count": r.pa_sweep_success_count,
            }
        )
    return rows


def save_outputs(prefix: Path, payload: Dict[str, object], records: Sequence[OpeningRecord]) -> Tuple[Path, Path]:
    json_path = _prefixed_path(prefix, ".json")
    csv_path = _prefixed_path(prefix, ".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = records_to_json(records)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def plot_summary(
    image: np.ndarray,
    header: Dict[str, object],
    ridge: Dict[str, object],
    records: Sequence[OpeningRecord],
    summary: Dict[str, object],
    out_path: Path,
    width_xlim: Optional[Tuple[float, float]] = None,
    width_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax = axes[0]
    rms = float(summary["rms_jy_per_beam"])
    finite = image[np.isfinite(image)]
    display = apply_stretch(image, "asinh", max(float(np.nanpercentile(finite, 1)), -3 * rms), float(np.nanpercentile(finite, 99.9)))
    extent = image_edges_mas(header, image.shape)
    ax.imshow(display, origin="lower", extent=extent, cmap="inferno", interpolation="nearest")
    levels = positive_contour_levels(rms, float(np.nanmax(image)), 3.0, 2.0)
    if levels.size:
        ax.contour(image, levels=levels, colors="white", linewidths=0.45, alpha=0.65, origin="lower", extent=extent)
    xy = np.asarray([(p["x_mas"], p["y_mas"]) for p in ridge["ridge_points"]], dtype=float)
    if xy.size:
        ax.plot(xy[:, 0], xy[:, 1], color="#22ff66", lw=1.8)
    limits = auto_limits(image, header, rms, 0.005, 8.0, 2.0)
    if limits is not None:
        xmin, xmax, ymin, ymax = limits
        ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
        ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
    if xy.size:
        expand_axes_to_points(ax, xy, padding_mas=max(0.5, float(summary["beam_mas"])))
    ax.set_title("FITS image + polar ridge")
    ax.set_xlabel("RA offset (mas)")
    ax.set_ylabel("Dec offset (mas)")

    path = np.asarray([r.path_mas for r in records], dtype=float)
    raw = np.asarray([r.fit.fwhm_mas for r in records], dtype=float)
    intrinsic = np.asarray([r.intrinsic_fwhm_mas for r in records], dtype=float)
    raw_err = np.asarray([r.pa_sweep_raw_rms_mas for r in records], dtype=float)
    intrinsic_err = np.asarray([r.pa_sweep_intrinsic_rms_mas for r in records], dtype=float)
    success = np.asarray([r.fit.success for r in records], dtype=bool)
    intrinsic_smooth5 = five_point_smoothed_values(intrinsic)
    axes[1].plot(path[success], raw[success], ".", ms=3, label="raw D")
    axes[1].plot(path[success], intrinsic[success], ".", ms=3, label="deconvolved d")
    raw_err_mask = success & np.isfinite(path) & np.isfinite(raw) & (raw > 0.0) & np.isfinite(raw_err) & (raw_err > 0.0)
    if np.any(raw_err_mask):
        axes[1].errorbar(
            path[raw_err_mask],
            raw[raw_err_mask],
            yerr=raw_err[raw_err_mask],
            fmt="none",
            ecolor="#1f77b4",
            elinewidth=0.55,
            alpha=0.35,
            capsize=0,
            label="raw PA sweep rms",
        )
    intrinsic_err_mask = success & np.isfinite(path) & np.isfinite(intrinsic) & (intrinsic > 0.0) & np.isfinite(intrinsic_err) & (intrinsic_err > 0.0)
    if np.any(intrinsic_err_mask):
        axes[1].errorbar(
            path[intrinsic_err_mask],
            intrinsic[intrinsic_err_mask],
            yerr=intrinsic_err[intrinsic_err_mask],
            fmt="none",
            ecolor="#ff7f0e",
            elinewidth=0.55,
            alpha=0.35,
            capsize=0,
            label="deconv PA sweep rms",
        )
    failed = (~success) & np.isfinite(raw) & (raw > 0)
    if np.any(failed):
        axes[1].plot(path[failed], raw[failed], "x", ms=4, color="0.55", label="rejected fit")
    axes[1].plot(path, intrinsic_smooth5, "-", lw=0.9, alpha=0.65, label="d smooth5")
    axes[1].axhline(float(summary["beam_mas"]), color="0.5", lw=1.0, ls="--", label="beam")
    fit = dict(summary.get("power_law_fit", {}))
    fit5 = dict(summary.get("power_law_fit_smoothed5", {}))
    if np.isfinite(fit.get("k", float("nan"))) and fit.get("n", 0) >= 2:
        r_line = np.logspace(math.log10(float(fit["r_min_mas"])), math.log10(float(fit["r_max_mas"])), 160)
        d_line = float(fit["amplitude_mas_at_1mas"]) * np.power(r_line, float(fit["k"]))
        axes[1].plot(r_line, d_line, "--", lw=1.0, color="#e76f51", label=f"d fit k={fit['k']:.3g}")
    if np.isfinite(fit5.get("k", float("nan"))) and fit5.get("n", 0) >= 2:
        r_line = np.logspace(math.log10(float(fit5["r_min_mas"])), math.log10(float(fit5["r_max_mas"])), 160)
        d_line = float(fit5["amplitude_mas_at_1mas"]) * np.power(r_line, float(fit5["k"]))
        axes[1].plot(r_line, d_line, "-", lw=1.5, color="#d62828", label=f"smooth5 fit k={fit5['k']:.3g}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    if width_xlim is not None:
        axes[1].set_xlim(width_xlim)
    if width_ylim is not None:
        axes[1].set_ylim(width_ylim)
    axes[1].set_xlabel("Separation along ridgeline (mas)")
    axes[1].set_ylabel("FWHM / width (mas)")
    axes[1].legend(fontsize=8)
    if np.isfinite(fit5.get("k", float("nan"))):
        axes[1].set_title(f"Gaussian widths: d ~ r^k, k={fit5['k']:.3g}")
    else:
        axes[1].set_title("Gaussian widths")

    angle = np.asarray([r.opening_angle_deg for r in records], dtype=float)
    raw_angle = np.asarray([r.opening_angle_raw_deg for r in records], dtype=float)
    raw_angle_err = np.asarray([r.pa_sweep_opening_angle_raw_rms_deg for r in records], dtype=float)
    angle_err = np.asarray([r.pa_sweep_opening_angle_intrinsic_rms_deg for r in records], dtype=float)
    axes[2].plot(path, raw_angle, ".", ms=3, alpha=0.5, label="raw")
    axes[2].plot(path, angle, ".", ms=3, label="deconvolved")
    raw_angle_err_mask = np.isfinite(path) & np.isfinite(raw_angle) & np.isfinite(raw_angle_err) & (raw_angle_err > 0.0)
    if np.any(raw_angle_err_mask):
        axes[2].errorbar(
            path[raw_angle_err_mask],
            raw_angle[raw_angle_err_mask],
            yerr=raw_angle_err[raw_angle_err_mask],
            fmt="none",
            ecolor="#1f77b4",
            elinewidth=0.55,
            alpha=0.35,
            capsize=0,
            label="raw PA sweep rms",
        )
    angle_err_mask = np.isfinite(path) & np.isfinite(angle) & np.isfinite(angle_err) & (angle_err > 0.0)
    if np.any(angle_err_mask):
        axes[2].errorbar(
            path[angle_err_mask],
            angle[angle_err_mask],
            yerr=angle_err[angle_err_mask],
            fmt="none",
            ecolor="#ff7f0e",
            elinewidth=0.55,
            alpha=0.35,
            capsize=0,
            label="deconv PA sweep rms",
        )
    axes[2].axvline(float(summary["analysis_separation_min_mas"]), color="0.5", lw=1.0, ls="--")
    if summary.get("analysis_separation_max_mas") is not None:
        axes[2].axvline(float(summary["analysis_separation_max_mas"]), color="0.5", lw=1.0, ls=":")
    axes[2].axhline(float(summary["median_opening_angle_deg"]), color="#e63946", lw=1.2, label="median d")
    axes[2].set_xlabel("Separation along ridgeline (mas)")
    axes[2].set_ylabel("Full apparent opening angle (deg)")
    median_error = dict(summary.get("median_error", {}))
    err = dict(median_error.get("intrinsic_block", {}))
    err_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
    title_err = err_with_pa if np.isfinite(err_with_pa.get("sigma", float("nan"))) else err
    axes[2].set_title(
        f"median={summary['median_opening_angle_deg']:.3g} +/- {title_err.get('sigma', float('nan')):.2g} deg; "
        f"raw={summary['median_opening_angle_raw_deg']:.3g}"
    )
    axes[2].legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Gaussian transverse FWHM and Pushkarev-style full apparent opening angle "
            "from a MOJAVE FITS image and polar ridgeline JSON."
        )
    )
    parser.add_argument("fits", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "mojave_fits" / "0238+711.u.stacked.icc.fits")
    parser.add_argument("ridge_json", nargs="?", type=Path, default=None)
    parser.add_argument("--sector", default="east", help="Default ridge sector name when ridge_json is omitted.")
    parser.add_argument("--output", type=Path, default=None, help="Output prefix. Default: ridge stem + _opening")
    parser.add_argument("--analysis-sep-min", type=float, default=0.5, help="Minimum separation along ridgeline for opening-angle median and k fit.")
    parser.add_argument("--analysis-sep-max", type=float, default=None, help="Maximum separation along ridgeline for opening-angle median and k fit.")
    parser.add_argument("--angle-rmin", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--slice-half-width", type=float, default=None)
    parser.add_argument("--sample-step", type=float, default=None)
    parser.add_argument("--fit-threshold-snr", type=float, default=3.0)
    parser.add_argument("--fit-threshold", type=float, default=None)
    parser.add_argument("--fit-padding", type=float, default=None)
    parser.add_argument("--baseline-guard", type=float, default=None)
    parser.add_argument("--min-fit-points", type=int, default=8)
    parser.add_argument("--sigma-upper", type=float, default=None)
    parser.add_argument("--pa-sweep", action="store_true", help="Estimate width error by transverse-cut PA +/-15 deg, step 1 deg.")
    parser.add_argument("--pa-sweep-range", type=float, default=15.0)
    parser.add_argument("--pa-sweep-step", type=float, default=1.0)
    parser.add_argument("--pa-sweep-workers", type=int, default=1, help="PA sweep worker threads. Values above 1 are optional and may be slower with threaded BLAS.")
    parser.add_argument("--pa-sweep-analysis-only", action="store_true", help="Run PA sweep only inside the analysis separation range.")
    parser.add_argument("--max-records", type=int, default=None, help="Debug/quick-test limit on ridge records.")
    parser.add_argument("--bootstrap-count", type=int, default=5000, help="Bootstrap trials for median opening-angle error.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260430)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-figure", type=Path, default=None)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.angle_rmin is not None:
        args.analysis_sep_min = float(args.angle_rmin)
    fits_path = args.fits
    ridge_path = args.ridge_json or default_ridge_for_fits(fits_path, args.sector)
    output_prefix = args.output or ridge_path.with_name(f"{ridge_path.stem}_opening")

    image, header = read_primary_fits(fits_path)
    ridge = load_ridge(ridge_path)
    records, summary = measure_opening(image, header, ridge, args)
    payload = {
        "format": "mojave_fits_polar_opening_angle",
        "version": 1,
        "fits_file": str(fits_path),
        "ridge_file": str(ridge_path),
        "ridge_sector": ridge.get("sector", {}),
        "summary": summary,
        "records": records_to_json(records),
    }

    print(f"fits:  {fits_path}")
    print(f"ridge: {ridge_path}")
    print(f"records={summary['record_count']}, fit_success={summary['fit_success_count']}, unresolved={summary['unresolved_count']}")
    print(f"median raw full opening angle:        {summary['median_opening_angle_raw_deg']:.6g} deg")
    print(f"median deconvolved full opening angle:{summary['median_opening_angle_deg']:.6g} deg")
    print(f"5-point smoothed deconv median:       {summary['five_point_smoothed_median_opening_angle_deg']:.6g} deg")
    median_error = dict(summary.get("median_error", {}))
    intrinsic_block = dict(median_error.get("intrinsic_block", {}))
    intrinsic_block_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
    raw_block = dict(median_error.get("raw_block", {}))
    smooth_block = dict(median_error.get("intrinsic_smooth5_block", {}))
    intrinsic_naive = dict(median_error.get("intrinsic_naive", {}))
    block_slices = median_error.get("block_size_slices", "n/a")
    block_length = median_error.get("block_length_mas", float("nan"))
    block_target = median_error.get("block_target_mas", float("nan"))
    print(
        "median error, deconv bootstrap:      "
        f"block={intrinsic_block.get('sigma', float('nan')):.6g} deg "
        f"(block {block_slices} slices = {block_length:.6g} mas; target 0.5 beam = {block_target:.6g} mas), "
        f"naive={intrinsic_naive.get('sigma', float('nan')):.6g} deg"
    )
    print(
        "median error, raw/smooth5 block:     "
        f"raw={raw_block.get('sigma', float('nan')):.6g} deg, "
        f"smooth5={smooth_block.get('sigma', float('nan')):.6g} deg"
    )
    print(
        "median error, block+PA sweep:        "
        f"deconv={intrinsic_block_with_pa.get('sigma', float('nan')):.6g} deg"
    )
    sweep = dict(summary.get("pa_sweep", {}))
    print(
        "PA sweep median rms:                 "
        f"width={sweep.get('median_intrinsic_width_rms_mas', float('nan')):.6g} mas, "
        f"angle={sweep.get('median_intrinsic_opening_angle_rms_deg', float('nan')):.6g} deg"
    )
    fit = dict(summary.get("power_law_fit", {}))
    fit5 = dict(summary.get("power_law_fit_smoothed5", {}))
    print(
        "power-law d=A*r^k:                  "
        f"k={fit.get('k', float('nan')):.6g}"
        f" +/- {fit.get('k_sigma', float('nan')):.3g}, "
        f"A={fit.get('amplitude_mas_at_1mas', float('nan')):.6g} mas, "
        f"n={fit.get('n', 0)}"
    )
    print(
        "power-law d=A*r^k, smooth5:          "
        f"k={fit5.get('k', float('nan')):.6g}"
        f" +/- {fit5.get('k_sigma', float('nan')):.3g}, "
        f"A={fit5.get('amplitude_mas_at_1mas', float('nan')):.6g} mas, "
        f"n={fit5.get('n', 0)}"
    )

    if args.save:
        json_path, csv_path = save_outputs(output_prefix, payload, records)
        print(f"saved result: {json_path}")
        print(f"saved csv:    {csv_path}")
    if args.save_figure is not None:
        plot_summary(image, header, ridge, records, summary, args.save_figure)
        print(f"saved figure: {args.save_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
