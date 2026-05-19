from __future__ import annotations

import hashlib
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from skimage import graph

from ..common.numeric import robust_sigma as _robust_sigma_common
from ..common.numeric import safe_float as _safe_float_common
from ..common.numeric import unit_or_none as _unit_or_none_common

Point = Tuple[int, int]


def _safe_float(value: object, default: float = float("nan")) -> float:
    return _safe_float_common(value, default=default)


def _hash_int_points(points_xy: np.ndarray) -> str:
    pts = np.asarray(points_xy, dtype=np.int32)
    digest = hashlib.sha1()
    digest.update(str(tuple(pts.shape)).encode("ascii"))
    digest.update(pts.tobytes())
    return digest.hexdigest()


def _cache_float(value: object, digits: int = 6) -> str:
    val = _safe_float(value)
    return "nan" if not np.isfinite(val) else f"{val:.{digits}g}"


def _pushkarev_scan_half_width_px(
    scale_mas_per_px: Optional[float],
    beam_px: float,
    *,
    min_half_width_mas: float = 7.5,
    beam_factor: float = 6.0,
) -> Tuple[float, float, str]:
    candidates: List[Tuple[float, str]] = []
    scale = _safe_float(scale_mas_per_px)
    if np.isfinite(scale) and scale > 0.0 and np.isfinite(min_half_width_mas) and min_half_width_mas > 0.0:
        candidates.append((float(min_half_width_mas) / float(scale), "7.5mas"))
    beam = _safe_float(beam_px)
    if np.isfinite(beam) and beam > 0.0 and np.isfinite(beam_factor) and beam_factor > 0.0:
        candidates.append((float(beam_factor) * float(beam), "6beam"))
    if not candidates:
        return float("nan"), float("nan"), "image_bounds"
    half_width_px, source = max(candidates, key=lambda item: item[0])
    half_width_mas = float(half_width_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
    return float(half_width_px), float(half_width_mas), f"pushkarev_max_7.5mas_6beam:{source}"


def _pushkarev_profile_step_px(
    scale_mas_per_px: Optional[float],
    beam_px: float,
    fallback_px: float,
    *,
    max_step_mas: float = 0.05,
    beam_divisor: float = 12.0,
) -> Tuple[float, float, str]:
    candidates: List[Tuple[float, str]] = []
    scale = _safe_float(scale_mas_per_px)
    if np.isfinite(scale) and scale > 0.0 and np.isfinite(max_step_mas) and max_step_mas > 0.0:
        candidates.append((float(max_step_mas) / float(scale), "0.05mas"))
    beam = _safe_float(beam_px)
    if np.isfinite(beam) and beam > 0.0 and np.isfinite(beam_divisor) and beam_divisor > 0.0:
        candidates.append((float(beam) / float(beam_divisor), "beam/12"))
    if candidates:
        step_px, source = min(candidates, key=lambda item: item[0])
        step_px = float(max(1e-3, step_px))
        step_mas = float(step_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
        return step_px, step_mas, f"pushkarev_min_0.05mas_beam12:{source}"
    fallback = _safe_float(fallback_px)
    if not (np.isfinite(fallback) and fallback > 0.0):
        fallback = 0.5
    step_px = float(max(1e-3, fallback))
    step_mas = float(step_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
    return step_px, step_mas, "fallback_px"


def _pushkarev_slice_spacing_px(
    scale_mas_per_px: Optional[float],
    beam_px: float,
    *,
    reference_spacing_mas: float = 0.05,
    beam_divisor: float = 12.0,
) -> Tuple[float, float, str]:
    candidates: List[Tuple[float, str]] = []
    scale = _safe_float(scale_mas_per_px)
    if np.isfinite(scale) and scale > 0.0 and np.isfinite(reference_spacing_mas) and reference_spacing_mas > 0.0:
        candidates.append((float(reference_spacing_mas) / float(scale), "0.05mas"))
    beam = _safe_float(beam_px)
    if np.isfinite(beam) and beam > 0.0 and np.isfinite(beam_divisor) and beam_divisor > 0.0:
        candidates.append((float(beam) / float(beam_divisor), "beam/12"))
    if not candidates:
        return float("nan"), float("nan"), "fallback_count"
    spacing_px, source = max(candidates, key=lambda item: item[0])
    spacing_px = float(max(1e-6, spacing_px))
    spacing_mas = float(spacing_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
    return spacing_px, spacing_mas, f"pushkarev_max_0.05mas_beam12:{source}"


def _compact_gaussian_fit_result(fit: Dict[str, object]) -> Dict[str, object]:
    src = dict(fit or {})
    compact: Dict[str, object] = {
        "success": bool(src.get("success", False)),
    }
    for key in ("reason", "error", "fit_window", "params", "param_errors", "fwhm_px", "fwhm_sigma_px", "rmse"):
        if key in src:
            compact[key] = src[key]
    return compact


def _unit_or_none(vec: Sequence[float]) -> Optional[np.ndarray]:
    return _unit_or_none_common(vec)


def _dedupe_consecutive_points(points_xy: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.size <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    keep = [pts[0]]
    for pt in pts[1:]:
        if float(np.hypot(float(pt[0] - keep[-1][0]), float(pt[1] - keep[-1][1]))) > 1e-6:
            keep.append(pt)
    return np.asarray(keep, dtype=np.float32)


def _moving_average_polyline(points_xy: np.ndarray, window: int = 7) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 3:
        return pts.copy()
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return pts.copy()
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = pts.copy()
    for dim in range(2):
        arr = pts[:, dim]
        padded = np.pad(arr, (pad, pad), mode="edge")
        out[:, dim] = np.convolve(padded, kernel, mode="valid")
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _smooth_path_bspline(points_xy: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    pts = _dedupe_consecutive_points(points_xy)
    if len(pts) < 4:
        return pts.copy()
    try:
        diffs = np.diff(pts.astype(np.float64), axis=0)
        step = float(np.median(np.hypot(diffs[:, 0], diffs[:, 1])))
        if (not np.isfinite(step)) or step <= 1e-6:
            return pts.copy()
        total_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        n_out = int(max(len(pts), round(total_len / max(0.5, step * 0.75))))
        tck, _ = splprep(
            [pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)],
            s=float(max(0.0, smoothing)),
            k=min(3, len(pts) - 1),
        )
        u_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
        x_new, y_new = splev(u_new, tck)
        out = np.stack([x_new, y_new], axis=1).astype(np.float32)
        out[0] = pts[0]
        out[-1] = pts[-1]
        return _dedupe_consecutive_points(out)
    except Exception:
        return pts.copy()


def _distance_transform_score(mask: np.ndarray) -> np.ndarray:
    support = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(support):
        return np.zeros_like(support, dtype=np.float32)
    dist = cv2.distanceTransform(support, distanceType=cv2.DIST_L2, maskSize=5)
    dmax = float(np.max(dist))
    if dmax <= 1e-6:
        return np.zeros_like(dist, dtype=np.float32)
    return (dist / dmax).astype(np.float32)


def _masked_gaussian_for_ridge(flux_map: np.ndarray, support_mask: np.ndarray, sigma_px: float) -> np.ndarray:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    if not np.any(support) or float(sigma_px) <= 1e-6:
        return flux.copy()
    fill_value = float(np.nanmean(flux[support & np.isfinite(flux)])) if np.any(support & np.isfinite(flux)) else 0.0
    dense = np.where(support, np.nan_to_num(flux, nan=fill_value), 0.0).astype(np.float32)
    weight = support.astype(np.float32)
    sigma = float(max(0.1, sigma_px))
    ksize = int(max(3, 2 * math.ceil(3.0 * sigma) + 1))
    blurred_num = cv2.GaussianBlur(dense * weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_den = cv2.GaussianBlur(weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    out = flux.copy()
    ok = support & (blurred_den > 1e-6)
    out[ok] = blurred_num[ok] / blurred_den[ok]
    return out


def _axis_deviation_score(
    shape_hw: Tuple[int, int],
    core_xy: Point,
    tail_xy: Point,
    support_mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, y0 = float(core_xy[0]), float(core_xy[1])
    x1, y1 = float(tail_xy[0]), float(tail_xy[1])
    vx = x1 - x0
    vy = y1 - y0
    denom = float(math.hypot(vx, vy))
    score = np.zeros((h, w), dtype=np.float32)
    if denom <= 1e-6:
        score[(np.asarray(support_mask, dtype=np.uint8) > 0)] = 1.0
        return score
    perp = np.abs(((xx - x0) * vy) - ((yy - y0) * vx)) / denom
    sigma = float(max(1.0, sigma_px))
    score = np.exp(-0.5 * np.square(perp / sigma)).astype(np.float32)
    score[(np.asarray(support_mask, dtype=np.uint8) == 0)] = 0.0
    return score


def _normalize_field_inside_support(field_map: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    field = np.asarray(field_map, dtype=np.float32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    out = np.zeros_like(field, dtype=np.float32)
    if not np.any(support):
        return out
    vals = field[support]
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out
    fmin = float(np.min(vals[finite]))
    fmax = float(np.max(vals[finite]))
    if fmax <= fmin + 1e-9:
        out[support] = 1.0
        return out
    out[support] = ((field[support] - fmin) / max(1e-9, fmax - fmin)).astype(np.float32)
    return out


def _component_center_score_by_depth(depth_map: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.int32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    out = np.zeros(depth.shape, dtype=np.float32)
    positive_levels = sorted(int(v) for v in np.unique(depth[support]) if int(v) > 0)
    for level in positive_levels:
        level_mask = support & (depth == int(level))
        if not np.any(level_mask):
            continue
        n_comp, labels = cv2.connectedComponents(level_mask.astype(np.uint8), connectivity=8)
        for comp_id in range(1, int(n_comp)):
            comp = labels == comp_id
            if not np.any(comp):
                continue
            dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
            mx = float(np.max(dist[comp])) if np.any(comp) else 0.0
            if mx > 1e-6:
                out[comp] = np.maximum(out[comp], (dist[comp] / mx).astype(np.float32))
            else:
                out[comp] = np.maximum(out[comp], 1.0)
    return out


def _build_ridge_field_from_depth(
    region_depth_map: np.ndarray,
    support_mask: np.ndarray,
    flux_map: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    depth = np.asarray(region_depth_map, dtype=np.int32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    depth_norm = np.zeros(depth.shape, dtype=np.float32)
    if np.any(support):
        vals = depth[support]
        dmin = int(np.min(vals))
        dmax = int(np.max(vals))
        if dmax > dmin:
            depth_norm[support] = ((depth[support].astype(np.float32) - float(dmin)) / float(dmax - dmin)).astype(np.float32)
        else:
            depth_norm[support] = 1.0

    depth_float = np.maximum(depth.astype(np.float32), 0.0)
    band_center = _component_center_score_by_depth(depth, support.astype(np.uint8))
    global_center = _distance_transform_score(support.astype(np.uint8))
    flux_norm = np.zeros(depth.shape, dtype=np.float32)
    if flux_map is not None:
        flux_norm = _normalize_field_inside_support(np.asarray(flux_map, dtype=np.float32), support.astype(np.uint8))

    # Follow the v5 idea more closely: use contour depth as the primary field,
    # then add a modest ridge-bias boost inside each level component.
    ridge_bias = np.maximum(band_center, (0.38 * global_center)).astype(np.float32)
    ridge_field = np.zeros(depth.shape, dtype=np.float32)
    ridge_field[support] = (
        depth_float[support] +
        (0.95 * ridge_bias[support]) +
        (0.10 * flux_norm[support])
    ).astype(np.float32)
    ridge_field = _masked_gaussian_for_ridge(ridge_field, support.astype(np.uint8), sigma_px=1.0)
    ridge_field[~support] = 0.0
    return {
        "ridge_field": ridge_field.astype(np.float32),
        "ridge_bias": ridge_bias.astype(np.float32),
        "depth_float": depth_float.astype(np.float32),
        "depth_norm": depth_norm.astype(np.float32),
        "band_center_score": band_center.astype(np.float32),
        "global_center_score": global_center.astype(np.float32),
    }


def _build_v5_style_cost_from_depth(
    depth_map: np.ndarray,
    support_mask: np.ndarray,
    ridge_bias_map: Optional[np.ndarray] = None,
    axis_score_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    depth = np.asarray(depth_map, dtype=np.float32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    cost = np.full(depth.shape, 1e6, dtype=np.float64)
    valid = support & (depth > 0.0) & np.isfinite(depth)
    if not np.any(valid):
        return cost.astype(np.float32)

    ratio = float(math.sqrt(2.0))
    gain_eff = 2.0
    cost[valid] = np.power(ratio, -gain_eff * depth[valid]).astype(np.float64)

    pct = 60.0
    l_floor = float(np.percentile(depth[valid], pct))
    low = valid & (depth < l_floor)
    if np.any(low):
        cost[low] = cost[low] * 3.0

    if ridge_bias_map is not None:
        bias = np.asarray(ridge_bias_map, dtype=np.float32)
        if bias.shape == depth.shape and np.any(bias[valid] > 0.0):
            bmax = float(np.max(bias[valid]))
            if bmax > 1e-6:
                bn = bias / bmax
                cost[valid] = cost[valid] * np.exp(-0.90 * bn[valid])

    if axis_score_map is not None:
        axis = np.asarray(axis_score_map, dtype=np.float32)
        if axis.shape == depth.shape:
            cost[valid] = cost[valid] * np.exp(-0.28 * np.clip(axis[valid], 0.0, 1.0))

    cost[valid] = np.maximum(cost[valid], 1e-6)
    return cost.astype(np.float32)


def _line_k_bounds_to_image(px: int, py: int, nx: float, ny: float, w: int, h: int) -> Tuple[Optional[float], Optional[float]]:
    eps = 1e-9
    if abs(nx) < eps:
        if not (0 <= px <= (w - 1)):
            return None, None
        kx_min, kx_max = -np.inf, np.inf
    else:
        kx0 = (0.0 - float(px)) / float(nx)
        kx1 = ((w - 1.0) - float(px)) / float(nx)
        kx_min, kx_max = min(kx0, kx1), max(kx0, kx1)

    if abs(ny) < eps:
        if not (0 <= py <= (h - 1)):
            return None, None
        ky_min, ky_max = -np.inf, np.inf
    else:
        ky0 = (0.0 - float(py)) / float(ny)
        ky1 = ((h - 1.0) - float(py)) / float(ny)
        ky_min, ky_max = min(ky0, ky1), max(ky0, ky1)

    k_min = max(kx_min, ky_min)
    k_max = min(kx_max, ky_max)
    if not np.isfinite(k_min) or not np.isfinite(k_max) or k_max < k_min:
        return None, None
    return float(k_min), float(k_max)


def estimate_polyline_tangent(polyline_xy: np.ndarray, idx: int, half_window: int = 4) -> Optional[Tuple[float, float]]:
    pts = np.asarray(polyline_xy, dtype=np.float64)
    n_pts = int(len(pts))
    if n_pts < 2:
        return None
    i0 = max(0, int(idx) - int(half_window))
    i1 = min(n_pts, int(idx) + int(half_window) + 1)
    fit_pts = pts[i0:i1]
    if fit_pts.shape[0] < 2:
        return None
    ctr = np.mean(fit_pts, axis=0)
    arr = fit_pts - ctr
    if np.allclose(arr, 0.0):
        return None
    try:
        _, _, vh = np.linalg.svd(arr, full_matrices=False)
    except Exception:
        return None
    tx, ty = float(vh[0, 0]), float(vh[0, 1])
    p_back = pts[max(0, int(idx) - 1)]
    p_fwd = pts[min(n_pts - 1, int(idx) + 1)]
    fdx, fdy = float(p_fwd[0] - p_back[0]), float(p_fwd[1] - p_back[1])
    if (tx * fdx + ty * fdy) < 0:
        tx, ty = -tx, -ty
    unit = _unit_or_none((tx, ty))
    if unit is None:
        return None
    return float(unit[0]), float(unit[1])


def beam_size_mas(beam_major_mas: Optional[float], beam_minor_mas: Optional[float]) -> float:
    if beam_major_mas is None or beam_minor_mas is None:
        return float("nan")
    major = float(beam_major_mas)
    minor = float(beam_minor_mas)
    if (not np.isfinite(major)) or (not np.isfinite(minor)) or major <= 0.0 or minor <= 0.0:
        return float("nan")
    return float(math.sqrt(major * minor))


def beam_size_px(
    scale_mas_per_px: Optional[float],
    beam_major_mas: Optional[float],
    beam_minor_mas: Optional[float],
) -> float:
    scale = float(scale_mas_per_px) if scale_mas_per_px is not None else float("nan")
    beam_mas = beam_size_mas(beam_major_mas, beam_minor_mas)
    if (not np.isfinite(scale)) or scale <= 0.0 or (not np.isfinite(beam_mas)) or beam_mas <= 0.0:
        return float("nan")
    return float(beam_mas / scale)


def opening_angle_deg(width_px: float, distance_px: float) -> float:
    width_px = float(width_px)
    distance_px = float(distance_px)
    if (not np.isfinite(width_px)) or (not np.isfinite(distance_px)):
        return float("nan")
    if width_px <= 0.0 or distance_px <= 0.0:
        return float("nan")
    return float(2.0 * np.degrees(np.arctan(width_px / (2.0 * distance_px))))


def opening_angle_error_deg(width_px: float, distance_px: float, sigma_width_px: float) -> float:
    width_px = float(width_px)
    distance_px = float(distance_px)
    sigma_width_px = float(sigma_width_px)
    if (not np.isfinite(width_px)) or (not np.isfinite(distance_px)) or (not np.isfinite(sigma_width_px)):
        return float("nan")
    if width_px <= 0.0 or distance_px <= 0.0 or sigma_width_px < 0.0:
        return float("nan")
    u = width_px / (2.0 * distance_px)
    dtheta_dw = 1.0 / (distance_px * (1.0 + (u * u)))  # radians per px
    return float(abs(dtheta_dw * sigma_width_px) * (180.0 / math.pi))


def _robust_sigma(values: Sequence[float]) -> float:
    return _robust_sigma_common(values)


def _combine_independent_sigmas(*values: float) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v)) and float(v) >= 0.0]
    if not finite:
        return float("nan")
    return float(math.sqrt(sum(v * v for v in finite)))


def _mean_propagated_sigma(sigmas: Sequence[float]) -> float:
    arr = np.asarray(sigmas, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr >= 0.0)]
    if finite.size <= 0:
        return float("nan")
    return float(math.sqrt(float(np.sum(np.square(finite)))) / float(finite.size))


def _median_propagated_sigma(sigmas: Sequence[float]) -> float:
    base = _mean_propagated_sigma(sigmas)
    if not np.isfinite(base):
        return float("nan")
    return float(math.sqrt(math.pi / 2.0) * base)


def _summary_sigma_with_measurement(values: Sequence[float], sigmas: Sequence[float]) -> Tuple[float, float, float]:
    scatter = _robust_sigma(values)
    propagated = _median_propagated_sigma(sigmas)
    return _combine_independent_sigmas(scatter, propagated), float(scatter), float(propagated)


def _finite_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _median_positive_spacing(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return float("nan")
    diffs = np.diff(finite)
    positive = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    return float(np.median(positive)) if positive.size else float("nan")


def _blocked_median_summary(
    values: Sequence[float],
    sigmas: Sequence[float],
    distances_px: Sequence[float],
    beam_px: float,
    scale_mas_per_px: Optional[float],
) -> Dict[str, object]:
    value_arr = np.asarray(values, dtype=np.float64)
    sigma_arr = np.asarray(sigmas, dtype=np.float64)
    dist_arr = np.asarray(distances_px, dtype=np.float64)
    n = int(min(value_arr.size, dist_arr.size))
    if sigma_arr.size < n:
        padded = np.full((n,), np.nan, dtype=np.float64)
        padded[: sigma_arr.size] = sigma_arr
        sigma_arr = padded
    else:
        sigma_arr = sigma_arr[:n]
    value_arr = value_arr[:n]
    dist_arr = dist_arr[:n]
    valid = np.isfinite(value_arr) & np.isfinite(dist_arr)
    if not np.any(valid):
        return {
            "median": float("nan"),
            "sigma": float("nan"),
            "scatter": float("nan"),
            "measurement_sigma": float("nan"),
            "block_count": 0,
            "block_size_slices": 0,
            "block_target_px": float("nan"),
            "block_target_mas": float("nan"),
            "block_spacing_px": float("nan"),
            "block_spacing_mas": float("nan"),
            "method": "half_beam_block_median_no_valid_values",
        }

    order = np.argsort(dist_arr[valid], kind="stable")
    values_valid = value_arr[valid][order]
    sigmas_valid = sigma_arr[valid][order]
    dist_valid = dist_arr[valid][order]
    scale = _safe_float(scale_mas_per_px)
    spacing_px = _median_positive_spacing(dist_valid)
    spacing_mas = float(spacing_px * scale) if np.isfinite(spacing_px) and np.isfinite(scale) and scale > 0.0 else float("nan")
    beam = _safe_float(beam_px)
    block_target_px = float(0.5 * beam) if np.isfinite(beam) and beam > 0.0 else float("nan")
    block_target_mas = (
        float(block_target_px * scale)
        if np.isfinite(block_target_px) and np.isfinite(scale) and scale > 0.0
        else float("nan")
    )
    if np.isfinite(block_target_px) and block_target_px > 0.0 and np.isfinite(spacing_px) and spacing_px > 0.0:
        block_size_slices = int(max(1, round(block_target_px / spacing_px)))
    else:
        block_size_slices = 1

    blocks: List[np.ndarray] = []
    current: List[int] = []
    block_start = float(dist_valid[0])
    use_distance_blocks = np.isfinite(block_target_px) and block_target_px > 0.0
    for pos, distance_px in enumerate(dist_valid.tolist()):
        if current and use_distance_blocks and (float(distance_px) - block_start) >= block_target_px:
            blocks.append(np.asarray(current, dtype=np.int32))
            current = []
            block_start = float(distance_px)
        current.append(pos)
    if current:
        blocks.append(np.asarray(current, dtype=np.int32))

    block_values: List[float] = []
    block_sigmas: List[float] = []
    for block in blocks:
        vals = values_valid[block]
        block_values.append(_finite_median(vals))
        block_sigmas.append(_median_propagated_sigma(sigmas_valid[block]))
    block_values_arr = np.asarray(block_values, dtype=np.float64)
    block_sigmas_arr = np.asarray(block_sigmas, dtype=np.float64)
    scatter = _robust_sigma(block_values_arr)
    propagated = _median_propagated_sigma(block_sigmas_arr)
    return {
        "median": _finite_median(block_values_arr),
        "sigma": _combine_independent_sigmas(scatter, propagated),
        "scatter": float(scatter),
        "measurement_sigma": float(propagated),
        "block_count": int(len(block_values)),
        "block_size_slices": int(block_size_slices),
        "block_target_px": float(block_target_px),
        "block_target_mas": float(block_target_mas),
        "block_spacing_px": float(spacing_px),
        "block_spacing_mas": float(spacing_mas),
        "method": "half_beam_block_median" if use_distance_blocks else "individual_slice_median_no_beam",
    }


def _linear_fit_with_cov(
    x: Sequence[float],
    y: Sequence[float],
    sigma_y: Optional[Sequence[float]] = None,
    *,
    min_points: int = 2,
) -> Optional[Dict[str, object]]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    n = int(min(xv.size, yv.size))
    if n <= 0:
        return None
    xv = xv[:n]
    yv = yv[:n]
    valid = np.isfinite(xv) & np.isfinite(yv)
    use_weighted = False
    if sigma_y is not None:
        sv = np.asarray(sigma_y, dtype=np.float64)[:n]
        weighted = valid & np.isfinite(sv) & (sv > 0.0)
        if int(np.count_nonzero(weighted)) >= int(min_points):
            valid = weighted
            use_weighted = True
    if int(np.count_nonzero(valid)) < int(min_points):
        return None
    xf = xv[valid]
    yf = yv[valid]
    design = np.column_stack([xf, np.ones_like(xf)])
    if use_weighted:
        sf = np.asarray(sigma_y, dtype=np.float64)[:n][valid]
        weights = 1.0 / np.square(sf)
        lhs = design.T @ (design * weights[:, None])
        rhs = design.T @ (yf * weights)
    else:
        sf = np.full(yf.shape, np.nan, dtype=np.float64)
        lhs = design.T @ design
        rhs = design.T @ yf
    try:
        beta = np.linalg.solve(lhs, rhs)
        inv_lhs = np.linalg.inv(lhs)
    except Exception:
        return None
    slope = float(beta[0])
    intercept = float(beta[1])
    if (not np.isfinite(slope)) or (not np.isfinite(intercept)):
        return None
    residual = yf - (slope * xf + intercept)
    dof = max(0, int(yf.size) - 2)
    if use_weighted:
        chi2 = float(np.sum(np.square(residual / sf)))
        cov = inv_lhs
    else:
        chi2 = float(np.sum(np.square(residual)))
        scale = chi2 / float(dof) if dof > 0 else float("nan")
        cov = inv_lhs * scale if np.isfinite(scale) else np.full((2, 2), np.nan, dtype=np.float64)
    return {
        "slope": slope,
        "intercept": intercept,
        "cov": np.asarray(cov, dtype=np.float64),
        "used_weighted": bool(use_weighted),
        "point_count": int(yf.size),
        "chi2": float(chi2),
        "reduced_chi2": float(chi2 / float(dof)) if dof > 0 else float("nan"),
        "x_fit": xf,
        "y_fit": yf,
        "residual": residual,
    }


def _linear_model_sigma(x: Sequence[float], cov: object) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    out = np.full(xv.shape, np.nan, dtype=np.float64)
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.shape != (2, 2) or not np.all(np.isfinite(cov_arr)):
        return out
    for idx, val in enumerate(xv.tolist()):
        if not np.isfinite(val):
            continue
        vec = np.asarray([float(val), 1.0], dtype=np.float64)
        var = float(vec @ cov_arr @ vec.T)
        if np.isfinite(var) and var >= 0.0:
            out[idx] = float(math.sqrt(var))
    return out


def _opening_model_sigma_from_log_width(
    width: Sequence[float],
    distance: Sequence[float],
    sigma_log_width: Sequence[float],
) -> np.ndarray:
    width_arr = np.asarray(width, dtype=np.float64)
    dist_arr = np.asarray(distance, dtype=np.float64)
    sigma_log = np.asarray(sigma_log_width, dtype=np.float64)
    n = int(min(width_arr.size, dist_arr.size, sigma_log.size))
    out = np.full(n, np.nan, dtype=np.float64)
    for idx in range(n):
        w = float(width_arr[idx])
        d = float(dist_arr[idx])
        slog = float(sigma_log[idx])
        if np.isfinite(w) and w > 0.0 and np.isfinite(d) and d > 0.0 and np.isfinite(slog) and slog >= 0.0:
            sigma_w = float(math.log(10.0) * w * slog)
            out[idx] = opening_angle_error_deg(w, d, sigma_w)
    return out


def snap_point_to_support_peak(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    point_xy: Point,
    radius: int = 8,
) -> Point:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = np.asarray(support_mask, dtype=np.uint8) > 0
    h, w = flux.shape[:2]
    px = int(np.clip(int(point_xy[0]), 0, max(0, w - 1)))
    py = int(np.clip(int(point_xy[1]), 0, max(0, h - 1)))
    if support[py, px] and np.isfinite(flux[py, px]):
        best = (px, py)
        best_val = float(flux[py, px])
    else:
        best = None
        best_val = -float("inf")
    radius = int(max(1, radius))
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    patch_flux = flux[y0:y1, x0:x1]
    patch_support = support[y0:y1, x0:x1]
    if np.any(patch_support):
        ys, xs = np.nonzero(patch_support)
        vals = patch_flux[ys, xs]
        d2 = (xs + x0 - px) ** 2 + (ys + y0 - py) ** 2
        order = np.lexsort((d2.astype(np.float64), -vals.astype(np.float64)))
        idx = int(order[0])
        best = (int(xs[idx] + x0), int(ys[idx] + y0))
        best_val = float(vals[idx])
    if best is not None and np.isfinite(best_val):
        return (int(best[0]), int(best[1]))
    return (px, py)


def _resnap_polyline_to_flux(
    ridge_xy: np.ndarray,
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    pts = np.asarray(ridge_xy, dtype=np.float32)
    if len(pts) <= 0:
        return pts.copy()
    out = pts.copy()
    for idx, pt in enumerate(pts):
        snapped = snap_point_to_support_peak(
            flux_map=flux_map,
            support_mask=support_mask,
            point_xy=(int(round(float(pt[0]))), int(round(float(pt[1])))),
            radius=radius,
        )
        out[idx, 0] = float(snapped[0])
        out[idx, 1] = float(snapped[1])
    out[0] = pts[0]
    out[-1] = pts[-1]
    return _dedupe_consecutive_points(out)


def extract_ridgeline_legacy_cost_path(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    core_xy: Point,
    tail_xy: Point,
    region_depth_map: Optional[np.ndarray] = None,
    snap_radius: int = 8,
    smooth_window: int = 7,
    bspline_smoothing: float = 0.0,
    resnap_radius: int = 2,
    include_debug_maps: bool = True,
) -> Dict[str, object]:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = (np.asarray(support_mask, dtype=np.uint8) > 0) & np.isfinite(flux)
    depth_arr = None
    ridge_field = None
    depth_norm = None
    band_center_score = None
    ridge_bias_map = None
    if region_depth_map is not None:
        depth_arr = np.asarray(region_depth_map, dtype=np.int32)
        if depth_arr.shape[:2] != flux.shape[:2]:
            depth_arr = None
    if not np.any(support):
        raise ValueError("No valid support region for ridgeline extraction.")
    if depth_arr is not None:
        ridge_parts = _build_ridge_field_from_depth(depth_arr, support.astype(np.uint8), flux_map=flux)
        ridge_field = np.asarray(ridge_parts["ridge_field"], dtype=np.float32)
        depth_norm = np.asarray(ridge_parts["depth_norm"], dtype=np.float32)
        band_center_score = np.asarray(ridge_parts["band_center_score"], dtype=np.float32)
        ridge_bias_map = np.asarray(ridge_parts["ridge_bias"], dtype=np.float32)
    else:
        ridge_field = np.asarray(flux, dtype=np.float32)
    core_input = (int(core_xy[0]), int(core_xy[1]))
    tail_input = (int(tail_xy[0]), int(tail_xy[1]))
    core = snap_point_to_support_peak(ridge_field, support, core_input, radius=snap_radius)
    tail = snap_point_to_support_peak(ridge_field, support, tail_input, radius=snap_radius)

    ref_len_px = float(np.hypot(float(tail[0] - core[0]), float(tail[1] - core[1])))
    guide_sigma_px = float(np.clip(0.015 * max(ref_len_px, 1.0), 1.0, 3.0))
    guide_flux = _masked_gaussian_for_ridge(ridge_field, support.astype(np.uint8), sigma_px=guide_sigma_px)
    field_norm = _normalize_field_inside_support(guide_flux, support.astype(np.uint8))

    support_dist = cv2.distanceTransform(support.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    support_width_ref = float(np.percentile(support_dist[support], 75)) if np.any(support) else 1.0
    axis_sigma_px = float(max(6.0, 4.0 * max(1.0, support_width_ref)))
    dist_max = float(np.max(support_dist)) if np.any(support) else 0.0
    dist_score = np.zeros_like(support_dist, dtype=np.float32)
    if dist_max > 1e-6:
        dist_score[support] = (support_dist[support] / dist_max).astype(np.float32)
    axis_score = _axis_deviation_score(flux.shape[:2], core, tail, support.astype(np.uint8), sigma_px=axis_sigma_px)
    if depth_arr is None:
        score = np.zeros_like(field_norm, dtype=np.float32)
        score[support] = (
            (0.58 * field_norm[support]) +
            (0.30 * dist_score[support]) +
            (0.12 * axis_score[support])
        )
        cost = np.full_like(score, 1e6, dtype=np.float64)
        cost[support] = (1.0 + (8.0 * (1.0 - score[support]))).astype(np.float64)
    else:
        cost = _build_v5_style_cost_from_depth(
            depth_map=np.asarray(depth_arr, dtype=np.float32),
            support_mask=support.astype(np.uint8),
            ridge_bias_map=ridge_bias_map,
            axis_score_map=axis_score,
        ).astype(np.float64)

    start_rc = (int(core[1]), int(core[0]))
    end_rc = (int(tail[1]), int(tail[0]))
    mcp = graph.MCP_Geometric(cost)
    try:
        mcp.find_costs([start_rc], [end_rc])
        raw_rc = mcp.traceback(end_rc)
    except Exception as exc:
        raise ValueError(f"Ridgeline traceback failed: {exc}") from exc
    if not raw_rc:
        raise ValueError("Ridgeline traceback returned an empty path.")

    raw_xy = np.asarray([(int(col), int(row)) for row, col in raw_rc], dtype=np.float32)
    raw_xy = _dedupe_consecutive_points(raw_xy)
    seed_xy = _moving_average_polyline(raw_xy, window=smooth_window)
    seed_xy = _smooth_path_bspline(seed_xy, smoothing=bspline_smoothing)
    refine_intensity_map = guide_flux if depth_arr is None else np.asarray(ridge_field, dtype=np.float32)
    smooth_xy = _resnap_polyline_to_flux(
        ridge_xy=seed_xy,
        flux_map=refine_intensity_map,
        support_mask=support,
        radius=resnap_radius,
    )
    smooth_xy = _moving_average_polyline(smooth_xy, window=smooth_window)
    smooth_xy = _smooth_path_bspline(smooth_xy, smoothing=bspline_smoothing)
    if len(smooth_xy) >= 2:
        smooth_xy[0] = np.asarray(core_input, dtype=np.float32)
        smooth_xy[-1] = np.asarray(tail_input, dtype=np.float32)
    result = {
        "core_xy": (int(core_input[0]), int(core_input[1])),
        "tail_xy": (int(tail_input[0]), int(tail_input[1])),
        "snapped_core_xy": (int(core[0]), int(core[1])),
        "snapped_tail_xy": (int(tail[0]), int(tail[1])),
        "raw_ridge_xy": np.rint(raw_xy).astype(np.int32),
        "ridge_xy": np.rint(smooth_xy).astype(np.int32),
    }
    if include_debug_maps:
        result.update(
            {
                "extraction_mode": "legacy_cost_path",
                "cost_map": cost.astype(np.float32),
                "guide_flux_map": guide_flux.astype(np.float32),
                "axis_score_map": axis_score.astype(np.float32),
                "ridge_field_map": ridge_field.astype(np.float32),
            }
        )
    else:
        result["extraction_mode"] = "legacy_cost_path"
    return result


extract_ridgeline_legacy = extract_ridgeline_legacy_cost_path


def _polar_angle_deg_from_xy(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(float(dx), float(dy))))


def _polar_xy(core_xy: Point, radius_px: float, pa_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pa = np.deg2rad(np.asarray(pa_deg, dtype=np.float64))
    x = float(core_xy[0]) + float(radius_px) * np.sin(pa)
    y = float(core_xy[1]) + float(radius_px) * np.cos(pa)
    return x, y


def _sample_float_map(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return map_coordinates(
        arr,
        [np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)],
        order=1,
        mode="constant",
        cval=np.nan,
    ).astype(np.float64)


def _sample_support_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    support = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
    sampled = map_coordinates(
        support,
        [np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)],
        order=0,
        mode="constant",
        cval=0.0,
    )
    return np.asarray(sampled > 0.5, dtype=bool)


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(np.asarray(mask, dtype=bool)):
        if bool(flag) and start is None:
            start = int(idx)
        elif not bool(flag) and start is not None:
            runs.append((int(start), int(idx)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask))))
    return runs


def _select_polar_arc_component(valid: np.ndarray, values: np.ndarray, mode: str) -> np.ndarray:
    valid = np.asarray(valid, dtype=bool)
    selected = np.zeros_like(valid, dtype=bool)
    runs = _contiguous_true_runs(valid)
    if not runs:
        return selected
    mode_norm = str(mode or "peak").strip().lower()
    if mode_norm == "all":
        return valid.copy()
    if mode_norm == "largest":
        start, stop = max(runs, key=lambda item: item[1] - item[0])
        selected[start:stop] = True
        return selected
    if mode_norm != "peak":
        mode_norm = "peak"
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    peak_index = int(np.nanargmax(finite_values))
    for start, stop in runs:
        if start <= peak_index < stop:
            selected[start:stop] = True
            return selected
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    selected[start:stop] = True
    return selected


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return float("nan")
    vals = vals[valid]
    w = w[valid]
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0.0:
        return float("nan")
    idx = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(vals[int(np.clip(idx, 0, len(vals) - 1))])


def _default_polar_threshold(field: np.ndarray, support: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float32)[np.asarray(support, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    eps = max(1e-6, 1e-6 * abs(vmax - vmin))
    return float(vmin - eps)


def _filter_polar_samples(
    samples: List[Dict[str, float]],
    radial_step_px: float,
    max_sample_step_factor: float,
    max_sample_pa_jump_deg: float,
) -> List[Dict[str, float]]:
    if not samples:
        return []
    accepted: List[Dict[str, float]] = []
    step = float(max(float(radial_step_px), 1e-6))
    for sample in samples:
        if not accepted:
            accepted.append(sample)
            continue
        prev = accepted[-1]
        dr = max(abs(float(sample["radial_px"]) - float(prev["radial_px"])), step)
        ds = math.hypot(float(sample["x"]) - float(prev["x"]), float(sample["y"]) - float(prev["y"]))
        dpa = abs((float(sample["pa_deg"]) - float(prev["pa_deg"]) + 180.0) % 360.0 - 180.0)
        if max_sample_step_factor > 0.0 and ds > float(max_sample_step_factor) * dr:
            continue
        if max_sample_pa_jump_deg > 0.0 and min(float(sample["radial_px"]), float(prev["radial_px"])) > (5.0 * step) and dpa > float(max_sample_pa_jump_deg):
            continue
        accepted.append(sample)
    return accepted


def _smooth_resample_polar_samples(
    samples: List[Dict[str, float]],
    core_xy: Point,
    radial_step_px: float,
    smoothing_px: float,
) -> np.ndarray:
    if len(samples) < 1:
        return np.asarray([[int(core_xy[0]), int(core_xy[1])]], dtype=np.float32)
    radial = np.asarray([0.0, *[float(item["radial_px"]) for item in samples]], dtype=np.float64)
    xy = np.asarray(
        [(float(core_xy[0]), float(core_xy[1])), *[(float(item["x"]), float(item["y"])) for item in samples]],
        dtype=np.float64,
    )
    keep = np.concatenate([[True], np.diff(radial) > 1e-6])
    radial = radial[keep]
    xy = xy[keep]
    if len(radial) < 2:
        return xy.astype(np.float32)
    step = float(max(0.25, radial_step_px))
    out_r = np.arange(0.0, float(np.nanmax(radial)) + (0.5 * step), step, dtype=np.float64)
    if len(radial) >= 4:
        try:
            smooth = max(0.0, float(smoothing_px)) ** 2 * float(len(radial))
            sx = UnivariateSpline(radial, xy[:, 0], k=min(3, len(radial) - 1), s=smooth)
            sy = UnivariateSpline(radial, xy[:, 1], k=min(3, len(radial) - 1), s=smooth)
            out = np.column_stack([sx(out_r), sy(out_r)])
        except Exception:
            out = np.column_stack([np.interp(out_r, radial, xy[:, 0]), np.interp(out_r, radial, xy[:, 1])])
    else:
        out = np.column_stack([np.interp(out_r, radial, xy[:, 0]), np.interp(out_r, radial, xy[:, 1])])
    if len(out):
        out[0] = np.asarray(core_xy, dtype=np.float64)
    return out.astype(np.float32)


def extract_ridgeline_mojave_polar(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    core_xy: Point,
    tail_xy: Point,
    *,
    snap_radius: int = 8,
    polar_step_px: float = 1.0,
    polar_pa_step_deg: float = 0.5,
    polar_sector_width_deg: float = 80.0,
    polar_component_mode: str = "peak",
    polar_threshold: Optional[float] = None,
    polar_smoothing_px: float = 0.0,
    polar_max_empty: int = 12,
    polar_min_arc_points: int = 8,
    polar_min_arc_span_deg: float = 2.0,
    polar_min_peak_over_threshold: float = 1.05,
    polar_max_sample_step_factor: float = 5.0,
    polar_max_sample_pa_jump_deg: float = 45.0,
    include_debug_maps: bool = True,
) -> Dict[str, object]:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = (np.asarray(support_mask, dtype=np.uint8) > 0) & np.isfinite(flux)
    if flux.ndim != 2:
        raise ValueError("Flux map must be a 2D array.")
    if support.shape[:2] != flux.shape[:2]:
        raise ValueError("Support mask shape must match flux map shape.")
    if not np.any(support):
        raise ValueError("No valid support region for ridgeline extraction.")

    core_input = (int(core_xy[0]), int(core_xy[1]))
    tail_input = (int(tail_xy[0]), int(tail_xy[1]))
    snapped_core = snap_point_to_support_peak(flux, support, core_input, radius=snap_radius)
    snapped_tail = snap_point_to_support_peak(flux, support, tail_input, radius=snap_radius)
    dx = float(tail_input[0] - core_input[0])
    dy = float(tail_input[1] - core_input[1])
    rmax = float(math.hypot(dx, dy))
    if rmax <= 1e-6:
        raise ValueError("Core and tail points must be separated for polar ridgeline extraction.")

    center_pa = _polar_angle_deg_from_xy(dx, dy)
    half_width = 0.5 * float(max(1.0, polar_sector_width_deg))
    pa_step = float(max(0.05, polar_pa_step_deg))
    pa_grid = np.arange(center_pa - half_width, center_pa + half_width + (0.5 * pa_step), pa_step, dtype=np.float64)
    threshold = _default_polar_threshold(flux, support) if polar_threshold is None else float(polar_threshold)
    if not np.isfinite(threshold):
        threshold = -float("inf")
    radial_step = float(max(0.25, polar_step_px))
    radial_grid = np.arange(radial_step, rmax + (0.5 * radial_step), radial_step, dtype=np.float64)

    raw_samples: List[Dict[str, float]] = []
    consecutive_empty = 0
    for radius in radial_grid:
        x_arc, y_arc = _polar_xy(core_input, float(radius), pa_grid)
        arc_support = _sample_support_mask(support, x_arc, y_arc)
        values = _sample_float_map(flux, x_arc, y_arc)
        valid = arc_support & np.isfinite(values) & (values > threshold)
        selected = _select_polar_arc_component(valid, values, polar_component_mode)
        n_selected = int(np.count_nonzero(selected))
        if n_selected < max(2, int(polar_min_arc_points)):
            consecutive_empty += 1
            if raw_samples and polar_max_empty > 0 and consecutive_empty >= int(polar_max_empty):
                break
            continue
        pa_selected = pa_grid[selected]
        values_selected = values[selected]
        weights = np.clip(values_selected, 0.0, None)
        if not np.any(weights > 0.0):
            weights = np.ones_like(values_selected, dtype=np.float64)
        pa_span = float(np.nanmax(pa_selected) - np.nanmin(pa_selected)) if pa_selected.size else 0.0
        peak = float(np.nanmax(values_selected)) if values_selected.size else float("nan")
        if pa_span < float(max(0.0, polar_min_arc_span_deg)):
            consecutive_empty += 1
            continue
        if np.isfinite(threshold) and threshold > 0.0 and float(polar_min_peak_over_threshold) > 1.0:
            if not (np.isfinite(peak) and peak >= threshold * float(polar_min_peak_over_threshold)):
                consecutive_empty += 1
                continue
        pa_med = _weighted_median(pa_selected, weights)
        if not np.isfinite(pa_med):
            consecutive_empty += 1
            continue
        x_ridge, y_ridge = _polar_xy(core_input, float(radius), np.asarray([pa_med], dtype=np.float64))
        raw_samples.append(
            {
                "radial_px": float(radius),
                "x": float(x_ridge[0]),
                "y": float(y_ridge[0]),
                "pa_deg": float(pa_med),
                "peak": peak,
                "weight_sum": float(np.sum(weights)),
                "n_used": float(n_selected),
                "pa_span_used_deg": pa_span,
            }
        )
        consecutive_empty = 0

    filtered_samples = _filter_polar_samples(
        raw_samples,
        radial_step_px=radial_step,
        max_sample_step_factor=float(polar_max_sample_step_factor),
        max_sample_pa_jump_deg=float(polar_max_sample_pa_jump_deg),
    )
    if len(filtered_samples) < 2:
        raise ValueError("Polar ridgeline extraction found fewer than 2 usable radial samples.")
    smooth_xy = _smooth_resample_polar_samples(
        filtered_samples,
        core_xy=core_input,
        radial_step_px=radial_step,
        smoothing_px=float(max(0.0, polar_smoothing_px)),
    )
    smooth_xy = _dedupe_consecutive_points(smooth_xy)
    result: Dict[str, object] = {
        "extraction_mode": "mojave_polar",
        "core_xy": (int(core_input[0]), int(core_input[1])),
        "tail_xy": (int(tail_input[0]), int(tail_input[1])),
        "snapped_core_xy": (int(snapped_core[0]), int(snapped_core[1])),
        "snapped_tail_xy": (int(snapped_tail[0]), int(snapped_tail[1])),
        "raw_ridge_xy": np.rint(np.asarray([(item["x"], item["y"]) for item in filtered_samples], dtype=np.float32)).astype(np.int32),
        "ridge_xy": np.rint(smooth_xy).astype(np.int32),
        "raw_sample_count": int(len(raw_samples)),
        "filtered_sample_count": int(len(filtered_samples)),
        "outlier_filtered_count": int(max(0, len(raw_samples) - len(filtered_samples))),
        "polar_parameters": {
            "method": "mojave_style_core_centered_polar_arc_flux_median",
            "radial_step_px": float(radial_step),
            "pa_step_deg": float(pa_step),
            "sector_center_pa_deg_image": float(center_pa),
            "sector_width_deg": float(2.0 * half_width),
            "component_mode": str(polar_component_mode),
            "threshold": float(threshold),
            "smoothing_px": float(max(0.0, polar_smoothing_px)),
            "max_empty": int(polar_max_empty),
            "min_arc_points": int(polar_min_arc_points),
            "min_arc_span_deg": float(polar_min_arc_span_deg),
            "min_peak_over_threshold": float(polar_min_peak_over_threshold),
            "max_sample_step_factor": float(polar_max_sample_step_factor),
            "max_sample_pa_jump_deg": float(polar_max_sample_pa_jump_deg),
            "rmax_px": float(rmax),
        },
    }
    if include_debug_maps:
        debug_map = np.zeros_like(flux, dtype=np.float32)
        for item in filtered_samples:
            x = int(round(float(item["x"])))
            y = int(round(float(item["y"])))
            if 0 <= y < debug_map.shape[0] and 0 <= x < debug_map.shape[1]:
                debug_map[y, x] = float(item.get("peak", 1.0))
        result["polar_sample_map"] = debug_map
    return result


def extract_ridgeline(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    core_xy: Point,
    tail_xy: Point,
    region_depth_map: Optional[np.ndarray] = None,
    snap_radius: int = 8,
    smooth_window: int = 7,
    bspline_smoothing: float = 0.0,
    resnap_radius: int = 2,
    include_debug_maps: bool = True,
    mode: str = "mojave_polar",
    polar_step_px: float = 1.0,
    polar_pa_step_deg: float = 0.5,
    polar_sector_width_deg: float = 80.0,
    polar_component_mode: str = "peak",
    polar_threshold: Optional[float] = None,
    polar_smoothing_px: float = 0.0,
    polar_max_empty: int = 12,
    polar_min_arc_points: int = 8,
    polar_min_arc_span_deg: float = 2.0,
    polar_min_peak_over_threshold: float = 1.05,
    polar_max_sample_step_factor: float = 5.0,
    polar_max_sample_pa_jump_deg: float = 45.0,
) -> Dict[str, object]:
    mode_norm = str(mode or "mojave_polar").strip().lower()
    if mode_norm in {"legacy", "legacy_cost", "legacy_cost_path", "cost_path"}:
        return extract_ridgeline_legacy_cost_path(
            flux_map=flux_map,
            support_mask=support_mask,
            core_xy=core_xy,
            tail_xy=tail_xy,
            region_depth_map=region_depth_map,
            snap_radius=snap_radius,
            smooth_window=smooth_window,
            bspline_smoothing=bspline_smoothing,
            resnap_radius=resnap_radius,
            include_debug_maps=include_debug_maps,
        )
    if mode_norm not in {"mojave", "mojave_polar", "polar"}:
        raise ValueError(f"Unknown ridgeline extraction mode: {mode}")
    return extract_ridgeline_mojave_polar(
        flux_map=flux_map,
        support_mask=support_mask,
        core_xy=core_xy,
        tail_xy=tail_xy,
        snap_radius=snap_radius,
        polar_step_px=polar_step_px,
        polar_pa_step_deg=polar_pa_step_deg,
        polar_sector_width_deg=polar_sector_width_deg,
        polar_component_mode=polar_component_mode,
        polar_threshold=polar_threshold,
        polar_smoothing_px=polar_smoothing_px,
        polar_max_empty=polar_max_empty,
        polar_min_arc_points=polar_min_arc_points,
        polar_min_arc_span_deg=polar_min_arc_span_deg,
        polar_min_peak_over_threshold=polar_min_peak_over_threshold,
        polar_max_sample_step_factor=polar_max_sample_step_factor,
        polar_max_sample_pa_jump_deg=polar_max_sample_pa_jump_deg,
        include_debug_maps=include_debug_maps,
    )


def sample_transverse_profile(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    ridge_xy: np.ndarray,
    ridge_idx: int,
    tangent_half_window: int = 4,
    profile_step_px: float = 0.5,
    normal_override_xy: Optional[Sequence[float]] = None,
    scan_half_width_px: Optional[float] = None,
) -> Dict[str, object]:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = (np.asarray(support_mask, dtype=np.uint8) > 0).astype(np.uint8)
    ridge = np.asarray(ridge_xy, dtype=np.int32)
    if len(ridge) < 5:
        raise ValueError("Ridgeline is too short for transverse profile sampling.")
    ridge_idx = int(np.clip(int(ridge_idx), 0, len(ridge) - 1))
    if normal_override_xy is None:
        tangent = estimate_polyline_tangent(ridge, ridge_idx, half_window=tangent_half_window)
        if tangent is None:
            raise ValueError("Failed to estimate ridgeline tangent.")
        tx, ty = tangent
        nx, ny = -ty, tx
    else:
        normal = _unit_or_none(normal_override_xy)
        if normal is None:
            raise ValueError("Invalid normal override.")
        nx, ny = float(normal[0]), float(normal[1])
        tx, ty = ny, -nx
    px, py = int(ridge[ridge_idx, 0]), int(ridge[ridge_idx, 1])
    h, w = flux.shape[:2]
    k_min, k_max = _line_k_bounds_to_image(px, py, nx, ny, w, h)
    if k_min is None or k_max is None:
        raise ValueError("Normal line does not intersect image.")
    image_k_min = float(k_min)
    image_k_max = float(k_max)
    scan_half_width = _safe_float(scan_half_width_px)
    scan_mode = "image_bounds"
    if np.isfinite(scan_half_width) and scan_half_width > 0.0:
        k_min = max(float(k_min), -abs(float(scan_half_width)))
        k_max = min(float(k_max), abs(float(scan_half_width)))
        scan_mode = "fixed_half_width"
    if not (np.isfinite(k_min) and np.isfinite(k_max) and float(k_max) > float(k_min)):
        raise ValueError("Transverse scan range is empty.")
    profile_step_px = float(max(0.25, profile_step_px))
    sample_k = np.arange(k_min, k_max + profile_step_px, profile_step_px, dtype=np.float32)
    if sample_k.size < 5:
        raise ValueError("Profile sampling range is too small.")
    center_idx = int(np.argmin(np.abs(sample_k)))
    samp_y = py + (ny * sample_k)
    samp_x = px + (nx * sample_k)
    sample_pts = np.vstack([samp_y, samp_x])
    profile = map_coordinates(flux, sample_pts, order=1, mode="constant", cval=np.nan).astype(np.float32)
    support_profile = map_coordinates(
        support.astype(np.float32),
        sample_pts,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5
    if center_idx <= 0 or center_idx >= (len(sample_k) - 1) or (not support_profile[center_idx]):
        raise ValueError("Ridgeline center is outside support.")
    left_lim = int(center_idx)
    while left_lim - 1 >= 0 and support_profile[left_lim - 1]:
        left_lim -= 1
    right_lim = int(center_idx)
    while right_lim + 1 < len(support_profile) and support_profile[right_lim + 1]:
        right_lim += 1
    valid_x = sample_k[left_lim:right_lim + 1]
    valid_y = profile[left_lim:right_lim + 1]
    if valid_x.size < 5:
        raise ValueError("Profile support interval is too small.")
    finite = np.isfinite(valid_y)
    if np.count_nonzero(finite) < 5:
        raise ValueError("Profile is not finite inside support.")
    peak_idx_local = int(np.nanargmax(valid_y))
    return {
        "ridge_idx": int(ridge_idx),
        "ridge_xy": (int(px), int(py)),
        "tangent_xy": (float(tx), float(ty)),
        "normal_xy": (float(nx), float(ny)),
        "sample_k": sample_k,
        "profile": profile,
        "support_profile": support_profile.astype(np.uint8),
        "scan_mode": str(scan_mode),
        "scan_half_width_px": float(scan_half_width) if np.isfinite(scan_half_width) and scan_half_width > 0.0 else float("nan"),
        "scan_k_min_px": float(k_min),
        "scan_k_max_px": float(k_max),
        "image_k_min_px": float(image_k_min),
        "image_k_max_px": float(image_k_max),
        "profile_step_px": float(profile_step_px),
        "center_idx": int(center_idx),
        "left_lim": int(left_lim),
        "right_lim": int(right_lim),
        "valid_x": valid_x.astype(np.float32),
        "valid_y": valid_y.astype(np.float32),
        "peak_idx_local": int(peak_idx_local),
        "profile_xy": np.stack([samp_x, samp_y], axis=1).astype(np.float32),
    }


def _smooth_1d_for_lobe(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 2:
        return arr.copy()
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    if window <= 1 or arr.size < 3:
        return arr.copy()
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _significant_local_minima(y_smooth: np.ndarray, center_idx: int) -> Tuple[List[int], List[int]]:
    ys = np.asarray(y_smooth, dtype=np.float64)
    n = int(ys.size)
    if n < 3:
        return [], []
    finite = np.isfinite(ys)
    if not np.any(finite):
        return [], []
    yrange = float(np.nanmax(ys[finite]) - np.nanmin(ys[finite]))
    if not np.isfinite(yrange) or yrange <= 1e-9:
        return [], []
    valley_tol = max(1e-9, 0.04 * yrange)
    center_idx = int(np.clip(center_idx, 0, n - 1))

    minima: List[int] = []
    for idx in range(1, n - 1):
        if not (np.isfinite(ys[idx - 1]) and np.isfinite(ys[idx]) and np.isfinite(ys[idx + 1])):
            continue
        is_min = ys[idx] <= ys[idx - 1] and ys[idx] <= ys[idx + 1]
        has_slope = ys[idx] < ys[idx - 1] or ys[idx] < ys[idx + 1]
        if is_min and has_slope:
            minima.append(int(idx))

    left: List[int] = []
    right: List[int] = []
    for idx in minima:
        if idx < center_idx:
            left_peak = float(np.nanmax(ys[: idx + 1]))
            center_side_peak = float(np.nanmax(ys[idx : center_idx + 1]))
            if (left_peak - ys[idx]) >= valley_tol and (center_side_peak - ys[idx]) >= valley_tol:
                left.append(int(idx))
        elif idx > center_idx:
            center_side_peak = float(np.nanmax(ys[center_idx : idx + 1]))
            right_peak = float(np.nanmax(ys[idx:]))
            if (center_side_peak - ys[idx]) >= valley_tol and (right_peak - ys[idx]) >= valley_tol:
                right.append(int(idx))
    return left, right


def _select_center_lobe_profile_window(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size <= 0 or y_arr.size <= 0 or x_arr.size != y_arr.size:
        return x_arr, y_arr, {"mode": "center_lobe", "fallback": "empty"}

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    center_idx = int(np.argmin(np.abs(x_arr)))

    diffs = np.diff(x_arr)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    step = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
    smooth_window = int(max(3, round(3.0 / max(step, 1e-6))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = int(min(max(3, smooth_window), 21))
    y_smooth = _smooth_1d_for_lobe(y_arr, smooth_window)
    left_minima, right_minima = _significant_local_minima(y_smooth, center_idx)

    left_idx = int(max(left_minima)) if left_minima else 0
    right_idx = int(min(right_minima)) if right_minima else int(x_arr.size - 1)
    if right_idx < left_idx:
        left_idx = 0
        right_idx = int(x_arr.size - 1)

    min_points = int(max(5, min_points))
    if (right_idx - left_idx + 1) < min_points:
        half = max(2, min_points // 2)
        left_idx = int(max(0, center_idx - half))
        right_idx = int(min(x_arr.size - 1, left_idx + min_points - 1))
        left_idx = int(max(0, min(left_idx, right_idx - min_points + 1)))

    selected_x = x_arr[left_idx : right_idx + 1]
    selected_y = y_arr[left_idx : right_idx + 1]
    meta = {
        "mode": "center_lobe",
        "left_x": float(x_arr[left_idx]) if x_arr.size else float("nan"),
        "right_x": float(x_arr[right_idx]) if x_arr.size else float("nan"),
        "left_index": int(left_idx),
        "right_index": int(right_idx),
        "center_index": int(center_idx),
        "center_x": float(x_arr[center_idx]) if x_arr.size else float("nan"),
        "n_points": int(selected_x.size),
        "full_n_points": int(x_arr.size),
        "smooth_window_points": int(smooth_window),
        "left_valley_index": int(left_idx) if left_minima else None,
        "right_valley_index": int(right_idx) if right_minima else None,
    }
    return selected_x, selected_y, meta


def _legacy_free_baseline_bounds(y: np.ndarray) -> Tuple[float, float]:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    dynamic = abs(y_max - y_min)
    return float(y_min - dynamic), float(y_max + dynamic)


_FIXED_ZERO_BASELINE_MODES = {
    "fixed_zero",
    "zero",
    "zero_fixed",
    "fixed_zero_reconstructed_background",
}

_LEGACY_BOUNDED_BASELINE_MODES = {
    "bounded_l1_noise",
    "bounded",
    "legacy_bounded_l1_noise",
    "legacy_bounded",
    "boundary",
    "legacy_boundary",
}


def _is_fixed_zero_baseline_mode(mode: str) -> bool:
    return str(mode or "fixed_zero").strip().lower() in _FIXED_ZERO_BASELINE_MODES


def _gaussian_baseline_bounds(
    y: np.ndarray,
    *,
    mode: str,
    l1_flux: Optional[float],
    noise_sigma_flux: Optional[float],
) -> Tuple[float, float, str, float, float]:
    mode_norm = str(mode or "fixed_zero").strip().lower()
    if mode_norm in _FIXED_ZERO_BASELINE_MODES:
        return 0.0, 0.0, "fixed_zero", float(_safe_float(l1_flux)), float(_safe_float(noise_sigma_flux))
    if mode_norm in ("legacy", "legacy_free", "free"):
        lo, hi = _legacy_free_baseline_bounds(y)
        return lo, hi, "legacy_free", float("nan"), float("nan")

    l1 = _safe_float(l1_flux)
    noise = _safe_float(noise_sigma_flux)
    if not (np.isfinite(l1) and l1 > 0.0):
        lo, hi = _legacy_free_baseline_bounds(y)
        return lo, hi, "legacy_free_fallback_no_l1", float("nan"), float("nan")
    if not (np.isfinite(noise) and noise > 0.0):
        noise = abs(float(l1)) / 3.0
    lo = -abs(float(noise))
    hi = float(l1)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        lo, hi = _legacy_free_baseline_bounds(y)
        return lo, hi, "legacy_free_fallback_invalid_bounds", float(l1), float(noise)
    bound_mode = "legacy_bounded_l1_noise" if mode_norm in _LEGACY_BOUNDED_BASELINE_MODES - {"bounded_l1_noise"} else "bounded_l1_noise"
    return float(lo), float(hi), bound_mode, float(l1), float(noise)


def fit_transverse_gaussian(
    profile_data: Dict[str, object],
    mu_bound_px: Optional[float] = None,
    *,
    baseline_mode: str = "fixed_zero",
    baseline_l1_flux: Optional[float] = None,
    baseline_noise_sigma_flux: Optional[float] = None,
) -> Dict[str, object]:
    x = np.asarray(profile_data.get("valid_x", []), dtype=np.float64)
    y = np.asarray(profile_data.get("valid_y", []), dtype=np.float64)
    if x.size < 5 or y.size < 5:
        return {"success": False, "reason": "too_few_points"}
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 5:
        return {"success": False, "reason": "too_few_finite_points"}
    full_x = x.astype(np.float32)
    full_y = y.astype(np.float32)
    x, y, fit_window = _select_center_lobe_profile_window(x, y, min_points=5)
    if x.size < 5:
        return {
            "success": False,
            "reason": "too_few_center_lobe_points",
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "full_x": full_x,
            "full_y": full_y,
            "fit_window": fit_window,
        }
    edge_floor = float(np.nanmin([y[0], y[-1]]))
    peak_idx = int(np.nanargmax(y))
    peak_val = float(y[peak_idx])
    sigma0 = float(max(np.ptp(x) / 6.0, 0.5))
    baseline_is_fixed_zero = _is_fixed_zero_baseline_mode(baseline_mode)
    if baseline_is_fixed_zero:
        c_lower = 0.0
        c_upper = 0.0
        c0 = 0.0
        baseline_bound_mode = "fixed_zero"
        baseline_l1 = float(_safe_float(baseline_l1_flux))
        baseline_noise = float(_safe_float(baseline_noise_sigma_flux))
    else:
        c_lower, c_upper, baseline_bound_mode, baseline_l1, baseline_noise = _gaussian_baseline_bounds(
            y,
            mode=baseline_mode,
            l1_flux=baseline_l1_flux,
            noise_sigma_flux=baseline_noise_sigma_flux,
        )
        c0 = float(np.clip(edge_floor, c_lower, c_upper))
    amp0 = float(max(peak_val - c0, 1e-6))
    if mu_bound_px is None or (not np.isfinite(float(mu_bound_px))) or float(mu_bound_px) <= 0.0:
        mu_bound = float(max(0.5, 0.25 * np.ptp(x)))
    else:
        mu_bound = float(abs(float(mu_bound_px)))
    mu0 = float(np.clip(float(x[peak_idx]), -mu_bound, mu_bound))

    def gauss1d(s, c, a, mu, sigma):
        sigma = np.maximum(sigma, 1e-6)
        return c + a * np.exp(-0.5 * ((s - mu) / sigma) ** 2)

    def gauss1d_zero(s, a, mu, sigma):
        sigma = np.maximum(sigma, 1e-6)
        return a * np.exp(-0.5 * ((s - mu) / sigma) ** 2)

    lower = np.array(
        [
            float(c_lower),
            0.0,
            -mu_bound,
            0.1,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            float(c_upper),
            float(max(1.0, np.nanmax(y) * 4.0)),
            mu_bound,
            float(max(1.0, np.ptp(x))),
        ],
        dtype=np.float64,
    )
    try:
        if baseline_is_fixed_zero:
            popt, pcov = curve_fit(
                gauss1d_zero,
                x,
                y,
                p0=np.array([amp0, mu0, sigma0], dtype=np.float64),
                bounds=(
                    np.array([0.0, -mu_bound, 0.1], dtype=np.float64),
                    np.array(
                        [
                            float(max(1.0, np.nanmax(y) * 4.0)),
                            mu_bound,
                            float(max(1.0, np.ptp(x))),
                        ],
                        dtype=np.float64,
                    ),
                ),
                maxfev=20000,
            )
            fit_y = gauss1d_zero(x, *popt)
            baseline = 0.0
            amplitude, mu, sigma = [float(v) for v in popt]
            try:
                sigma_mu = float(math.sqrt(max(0.0, float(pcov[1, 1]))))
            except Exception:
                sigma_mu = float("nan")
            try:
                sigma_sigma = float(math.sqrt(max(0.0, float(pcov[2, 2]))))
            except Exception:
                sigma_sigma = float("nan")
        else:
            popt, pcov = curve_fit(
                gauss1d,
                x,
                y,
                p0=np.array([c0, amp0, mu0, sigma0], dtype=np.float64),
                bounds=(lower, upper),
                maxfev=20000,
            )
            fit_y = gauss1d(x, *popt)
            baseline, amplitude, mu, sigma = [float(v) for v in popt]
            try:
                sigma_mu = float(math.sqrt(max(0.0, float(pcov[2, 2]))))
            except Exception:
                sigma_mu = float("nan")
            try:
                sigma_sigma = float(math.sqrt(max(0.0, float(pcov[3, 3]))))
            except Exception:
                sigma_sigma = float("nan")
        residual = y - fit_y
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        fwhm = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
        fwhm_sigma = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_sigma) if np.isfinite(sigma_sigma) else float("nan")
        return {
            "success": True,
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "fit_y": fit_y.astype(np.float32),
            "full_x": full_x,
            "full_y": full_y,
            "fit_window": fit_window,
            "params": {
                "baseline": baseline,
                "amplitude": amplitude,
                "mu": mu,
                "sigma": sigma,
                "mu_fixed": False,
                "mu_bound_px": float(mu_bound),
                "baseline_mode": str(baseline_bound_mode),
                "baseline_fixed": bool(baseline_is_fixed_zero),
                "baseline_bound_lower": float(c_lower),
                "baseline_bound_upper": float(c_upper),
                "baseline_l1_flux": float(baseline_l1),
                "baseline_noise_sigma_flux": float(baseline_noise),
            },
            "param_errors": {
                "mu": float(sigma_mu),
                "sigma": float(sigma_sigma),
            },
            "fwhm_px": float(fwhm),
            "fwhm_sigma_px": float(fwhm_sigma),
            "rmse": float(rmse),
            "covariance": pcov,
        }
    except Exception as exc:
        return {
            "success": False,
            "reason": "fit_failed",
            "error": str(exc),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "full_x": full_x,
            "full_y": full_y,
            "fit_window": fit_window,
        }


def _rotate_unit_vector(vec_xy: Sequence[float], angle_deg: float) -> Tuple[float, float]:
    vec = _unit_or_none(vec_xy)
    if vec is None:
        raise ValueError("Cannot rotate a zero-length vector.")
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    x = float(vec[0])
    y = float(vec[1])
    out = _unit_or_none((c * x - s * y, s * x + c * y))
    if out is None:
        raise ValueError("Rotated vector is invalid.")
    return float(out[0]), float(out[1])


def _pa_sweep_offsets(range_deg: float, step_deg: float) -> np.ndarray:
    limit = float(abs(range_deg))
    step = float(abs(step_deg))
    if (not np.isfinite(limit)) or limit <= 0.0 or (not np.isfinite(step)) or step <= 0.0:
        return np.zeros((0,), dtype=np.float32)
    n_steps = int(max(1, round((2.0 * limit) / step)))
    offsets = np.linspace(-limit, limit, n_steps + 1, dtype=np.float64)
    if not np.any(np.isclose(offsets, 0.0, atol=1e-9)):
        offsets = np.sort(np.concatenate([offsets, np.zeros((1,), dtype=np.float64)]))
    return offsets.astype(np.float32)


def _pa_sweep_measurement_cache_key(
    ridge_xy: np.ndarray,
    slice_indices: np.ndarray,
    *,
    tangent_half_window: int,
    profile_step_px: float,
    beam_px: float,
    mu_bound_px: float,
    scan_half_width_px: float,
    baseline_mode: str,
    baseline_l1_flux: Optional[float],
    baseline_noise_sigma_flux: Optional[float],
    range_deg: float,
    step_deg: float,
) -> str:
    slices = np.asarray(slice_indices, dtype=np.int32)
    digest = hashlib.sha1()
    digest.update(_hash_int_points(np.asarray(ridge_xy, dtype=np.int32)).encode("ascii"))
    digest.update(str(tuple(slices.tolist())).encode("ascii"))
    parts = [
        "pa_sweep_v4",
        f"tangent={int(tangent_half_window)}",
        f"profile_step={_cache_float(profile_step_px)}",
        f"beam_px={_cache_float(beam_px)}",
        f"mu_bound_px={_cache_float(mu_bound_px)}",
        f"scan_half_width_px={_cache_float(scan_half_width_px)}",
        f"baseline_mode={str(baseline_mode)}",
        f"baseline_l1={_cache_float(baseline_l1_flux)}",
        f"baseline_noise={_cache_float(baseline_noise_sigma_flux)}",
        f"range={_cache_float(range_deg)}",
        f"step={_cache_float(step_deg)}",
        digest.hexdigest(),
    ]
    return "|".join(parts)


def _pa_sweep_record_cache_key(measurement_cache_key: str, ridge_idx: int, ridge_xy: Sequence[float]) -> str:
    x = int(round(float(ridge_xy[0])))
    y = int(round(float(ridge_xy[1])))
    return f"{measurement_cache_key}|idx={int(ridge_idx)}|xy={x},{y}"


def _pa_sweep_cache_records(pa_sweep_cache: Optional[Dict[str, object]], measurement_cache_key: str) -> Dict[str, Dict[str, object]]:
    if not isinstance(pa_sweep_cache, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for item in list(pa_sweep_cache.get("fit_records", []) or []):
        if not isinstance(item, dict):
            continue
        sweep = item.get("pa_sweep", {})
        if not isinstance(sweep, dict):
            continue
        if str(sweep.get("measurement_cache_key", "")) != str(measurement_cache_key):
            continue
        record_key = str(sweep.get("record_cache_key", ""))
        if record_key:
            out[record_key] = dict(sweep)
    return out


def _valid_cached_pa_sweep(
    cached: Optional[Dict[str, object]],
    *,
    record_cache_key: str,
    measurement_cache_key: str,
    offsets_deg: np.ndarray,
) -> Optional[Dict[str, object]]:
    if not isinstance(cached, dict):
        return None
    if str(cached.get("record_cache_key", "")) != str(record_cache_key):
        return None
    if str(cached.get("measurement_cache_key", "")) != str(measurement_cache_key):
        return None
    cached_offsets = np.asarray(cached.get("offsets_deg", []), dtype=np.float64)
    expected = np.asarray(offsets_deg, dtype=np.float64)
    if cached_offsets.shape != expected.shape or not np.allclose(cached_offsets, expected, atol=1e-6, rtol=0.0):
        return None
    out = dict(cached)
    out["from_cache"] = True
    return out


def _rms_about_median(values: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float, int]:
    vals = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(vals)
    finite = vals[valid]
    if finite.size <= 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(finite))
    rms = float(np.sqrt(np.mean(np.square(finite - med)))) if finite.size > 1 else 0.0
    return rms, med, int(finite.size)


def _compute_pa_sweep_for_slice(
    *,
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    ridge_xy: np.ndarray,
    ridge_idx: int,
    ridge_point_xy: Sequence[float],
    normal_xy: Sequence[float],
    offsets_deg: np.ndarray,
    tangent_half_window: int,
    profile_step_px: float,
    scan_half_width_px: float,
    mu_bound_px: float,
    baseline_mode: str,
    baseline_l1_flux: Optional[float],
    baseline_noise_sigma_flux: Optional[float],
    beam_px: float,
    nominal_fwhm_px: float,
    nominal_intrinsic_fwhm_px: float,
    measurement_cache_key: str,
    record_cache_key: str,
    min_success_count: int,
) -> Dict[str, object]:
    offsets = np.asarray(offsets_deg, dtype=np.float32)
    widths = np.full(offsets.shape, np.nan, dtype=np.float32)
    intrinsic_widths = np.full(offsets.shape, np.nan, dtype=np.float32)
    success = np.zeros(offsets.shape, dtype=np.uint8)
    reasons: List[str] = []
    for pos, offset in enumerate(offsets.tolist()):
        try:
            if abs(float(offset)) <= 1e-9:
                fwhm_px = float(nominal_fwhm_px)
                intrinsic_px = float(nominal_intrinsic_fwhm_px)
            else:
                rotated_normal = _rotate_unit_vector(normal_xy, float(offset))
                profile = sample_transverse_profile(
                    flux_map=flux_map,
                    support_mask=support_mask,
                    ridge_xy=ridge_xy,
                    ridge_idx=int(ridge_idx),
                    tangent_half_window=tangent_half_window,
                    profile_step_px=profile_step_px,
                    normal_override_xy=rotated_normal,
                    scan_half_width_px=scan_half_width_px,
                )
                fit = fit_transverse_gaussian(
                    profile,
                    mu_bound_px=mu_bound_px,
                    baseline_mode=baseline_mode,
                    baseline_l1_flux=baseline_l1_flux,
                    baseline_noise_sigma_flux=baseline_noise_sigma_flux,
                )
                if not fit.get("success"):
                    reasons.append(f"{float(offset):.6g}: {fit.get('reason', 'fit_failed')}")
                    continue
                fwhm_px = _safe_float(fit.get("fwhm_px", float("nan")))
                if not np.isfinite(fwhm_px) or fwhm_px <= 0.0:
                    reasons.append(f"{float(offset):.6g}: invalid_fwhm")
                    continue
                if np.isfinite(beam_px) and beam_px > 0.0:
                    intrinsic_px = float(math.sqrt(max(0.0, (fwhm_px * fwhm_px) - (beam_px * beam_px))))
                else:
                    intrinsic_px = float("nan")
            widths[pos] = float(fwhm_px)
            intrinsic_widths[pos] = float(intrinsic_px)
            success[pos] = 1
        except Exception as exc:
            reasons.append(f"{float(offset):.6g}: {exc}")
            continue

    fwhm_rms, fwhm_med, success_count = _rms_about_median(widths, success > 0)
    intrinsic_rms, intrinsic_med, intrinsic_success_count = _rms_about_median(intrinsic_widths, success > 0)
    reliable = bool(success_count >= int(max(1, min_success_count)))
    if not reliable:
        fwhm_rms = float("nan")
        intrinsic_rms = float("nan")
    return {
        "enabled": True,
        "from_cache": False,
        "measurement_cache_key": str(measurement_cache_key),
        "record_cache_key": str(record_cache_key),
        "ridge_idx": int(ridge_idx),
        "ridge_xy": [int(round(float(ridge_point_xy[0]))), int(round(float(ridge_point_xy[1])))],
        "range_deg": float(np.max(np.abs(offsets))) if offsets.size else 0.0,
        "step_deg": float(np.median(np.diff(offsets.astype(np.float64)))) if offsets.size > 1 else float("nan"),
        "offsets_deg": offsets,
        "fwhm_px": widths,
        "intrinsic_fwhm_px": intrinsic_widths,
        "success": success,
        "success_count": int(success_count),
        "intrinsic_success_count": int(intrinsic_success_count),
        "min_success_count": int(min_success_count),
        "reliable": bool(reliable),
        "fwhm_rms_px": float(fwhm_rms),
        "fwhm_median_px": float(fwhm_med),
        "intrinsic_fwhm_rms_px": float(intrinsic_rms),
        "intrinsic_fwhm_median_px": float(intrinsic_med),
        "failure_reasons": reasons[:16],
    }


def _cumulative_distance(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(pts) == 1:
        return np.zeros((1,), dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    step = np.hypot(diffs[:, 0], diffs[:, 1]).astype(np.float32)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(step)]).astype(np.float32)


def _median_polyline_spacing_px(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 2:
        return float("nan")
    diffs = np.diff(pts, axis=0)
    step = np.hypot(diffs[:, 0], diffs[:, 1]).astype(np.float64)
    finite = step[np.isfinite(step) & (step > 1e-6)]
    return float(np.median(finite)) if finite.size else float("nan")


def _auto_tangent_half_window(
    points_xy: np.ndarray,
    beam_px: float,
    *,
    default_window: int = 4,
    min_window: int = 2,
    max_window: int = 10,
) -> Tuple[int, float, float]:
    spacing_px = _median_polyline_spacing_px(points_xy)
    target_span_px = float("nan")
    half_window = int(default_window)
    if np.isfinite(beam_px) and beam_px > 0.0 and np.isfinite(spacing_px) and spacing_px > 0.0:
        target_span_px = 0.5 * float(beam_px)
        half_window = int(round(target_span_px / spacing_px))
    half_window = int(np.clip(half_window, int(min_window), int(max_window)))
    return half_window, float(spacing_px), float(target_span_px)


def measure_ridgeline_fwhm(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    ridge_xy: np.ndarray,
    n_slices: int = 25,
    trim_frac: Optional[float] = None,
    trim_start_frac: Optional[float] = None,
    trim_end_frac: Optional[float] = None,
    tangent_half_window: Optional[int] = None,
    profile_step_px: float = 0.5,
    scale_mas_per_px: Optional[float] = None,
    beam_major_mas: Optional[float] = None,
    beam_minor_mas: Optional[float] = None,
    core_separation_px: float = 0.0,
    pa_sweep_enabled: bool = False,
    pa_sweep_range_deg: float = 15.0,
    pa_sweep_step_deg: float = 1.0,
    pa_sweep_cache: Optional[Dict[str, object]] = None,
    gaussian_baseline_mode: str = "fixed_zero",
    gaussian_baseline_l1_flux: Optional[float] = None,
    gaussian_baseline_noise_sigma_flux: Optional[float] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, object]:
    ridge = np.asarray(ridge_xy, dtype=np.int32)
    if len(ridge) < 10:
        raise ValueError("Ridgeline is too short for width measurement.")
    n_slices = int(max(2, n_slices))
    beam_px = beam_size_px(scale_mas_per_px, beam_major_mas, beam_minor_mas)
    mu_bound_px = float(0.5 * beam_px) if np.isfinite(beam_px) and beam_px > 0.0 else float("nan")
    scale = float(scale_mas_per_px) if scale_mas_per_px is not None else float("nan")
    if trim_start_frac is None and trim_end_frac is None:
        base_trim = 0.10 if trim_frac is None else float(trim_frac)
        trim_start_frac = base_trim
        trim_end_frac = base_trim
    trim_start_frac = float(np.clip(0.0 if trim_start_frac is None else trim_start_frac, 0.0, 0.45))
    trim_end_frac = float(np.clip(0.0 if trim_end_frac is None else trim_end_frac, 0.0, 0.45))
    total_len = int(len(ridge))
    start_idx = int(max(3, round(total_len * trim_start_frac)))
    end_idx = int(min(total_len - 4, total_len - 1 - round(total_len * trim_end_frac)))
    if end_idx <= start_idx:
        start_idx = 1
        end_idx = total_len - 2
    ridge_dist = _cumulative_distance(ridge.astype(np.float32))
    slice_spacing_px, slice_spacing_mas, slice_sampling_mode = _pushkarev_slice_spacing_px(scale, beam_px)
    slice_target_count = 0
    if np.isfinite(slice_spacing_px) and slice_spacing_px > 0.0:
        start_dist = float(ridge_dist[start_idx])
        end_dist = float(ridge_dist[end_idx])
        if np.isfinite(start_dist) and np.isfinite(end_dist) and end_dist > start_dist:
            target_dist = np.arange(start_dist, end_dist + (0.5 * slice_spacing_px), slice_spacing_px, dtype=np.float64)
            slice_target_count = int(target_dist.size)
            slice_indices = np.searchsorted(ridge_dist, target_dist, side="left").astype(np.int32)
            slice_indices = np.clip(slice_indices, start_idx, end_idx)
            prev_idx = np.maximum(start_idx, slice_indices - 1)
            next_dist = ridge_dist[slice_indices]
            prev_dist = ridge_dist[prev_idx]
            use_prev = np.abs(prev_dist - target_dist) < np.abs(next_dist - target_dist)
            slice_indices = np.where(use_prev, prev_idx, slice_indices).astype(np.int32)
            slice_indices = np.unique(slice_indices)
        else:
            slice_indices = np.zeros((0,), dtype=np.int32)
    else:
        slice_indices = np.zeros((0,), dtype=np.int32)
    if slice_indices.size < 2:
        slice_indices = np.linspace(start_idx, end_idx, num=n_slices, dtype=int)
        slice_indices = np.unique(slice_indices).astype(np.int32)
        slice_sampling_mode = "fallback_count"
        slice_spacing_px = float("nan")
        slice_spacing_mas = float("nan")
    slice_distances_px = ridge_dist[slice_indices] if slice_indices.size else np.zeros((0,), dtype=np.float32)
    slice_actual_spacing_px = _median_positive_spacing(slice_distances_px)
    slice_actual_spacing_mas = (
        float(slice_actual_spacing_px * scale)
        if np.isfinite(slice_actual_spacing_px) and np.isfinite(scale) and scale > 0.0
        else float("nan")
    )
    requested_profile_step_px = _safe_float(profile_step_px)
    if np.isfinite(slice_spacing_px) and slice_spacing_px > 0.0:
        profile_step_px = float(max(0.25, slice_spacing_px / 5.0))
        profile_step_mode = "slice_spacing_div5"
    else:
        profile_step_px = float(max(0.25, requested_profile_step_px if np.isfinite(requested_profile_step_px) else 0.5))
        profile_step_mode = "fallback_request_px"
    profile_step_mas = (
        float(profile_step_px * scale)
        if np.isfinite(profile_step_px) and np.isfinite(scale) and scale > 0.0
        else float("nan")
    )
    fwhm_px_list: List[float] = []
    fwhm_mas_list: List[float] = []
    fwhm_sigma_px_list: List[float] = []
    fwhm_sigma_mas_list: List[float] = []
    fwhm_fit_sigma_px_list: List[float] = []
    fwhm_pa_rms_px_list: List[float] = []
    intrinsic_px_list: List[float] = []
    intrinsic_mas_list: List[float] = []
    intrinsic_sigma_px_list: List[float] = []
    intrinsic_sigma_mas_list: List[float] = []
    intrinsic_fit_sigma_px_list: List[float] = []
    intrinsic_pa_rms_px_list: List[float] = []
    distance_px_list: List[float] = []
    opening_sigma_deg_list: List[float] = []
    intrinsic_opening_sigma_deg_list: List[float] = []
    width_lines: List[Tuple[Point, Point]] = []
    fit_records: List[Dict[str, object]] = []
    scan_half_width_px, scan_half_width_mas, scan_mode = _pushkarev_scan_half_width_px(scale, beam_px)
    tangent_half_window_auto = tangent_half_window is None
    if tangent_half_window_auto:
        tangent_half_window, ridge_median_spacing_px, tangent_target_span_px = _auto_tangent_half_window(ridge, beam_px)
    else:
        tangent_half_window = int(max(1, int(tangent_half_window)))
        ridge_median_spacing_px = _median_polyline_spacing_px(ridge)
        tangent_target_span_px = float("nan")
    core_separation_px = float(max(0.0, core_separation_px))
    pa_offsets = _pa_sweep_offsets(pa_sweep_range_deg, pa_sweep_step_deg) if bool(pa_sweep_enabled) else np.zeros((0,), dtype=np.float32)
    pa_min_success_count = int(max(3, math.ceil(0.5 * float(pa_offsets.size)))) if pa_offsets.size else 0
    pa_measurement_cache_key = (
        _pa_sweep_measurement_cache_key(
            ridge,
            slice_indices,
            tangent_half_window=tangent_half_window,
            profile_step_px=profile_step_px,
            beam_px=beam_px,
            mu_bound_px=mu_bound_px,
            scan_half_width_px=scan_half_width_px,
            baseline_mode=gaussian_baseline_mode,
            baseline_l1_flux=gaussian_baseline_l1_flux,
            baseline_noise_sigma_flux=gaussian_baseline_noise_sigma_flux,
            range_deg=pa_sweep_range_deg,
            step_deg=pa_sweep_step_deg,
        )
        if pa_offsets.size
        else ""
    )
    pa_cache_map = _pa_sweep_cache_records(pa_sweep_cache, pa_measurement_cache_key) if pa_offsets.size else {}
    pa_cache_hits = 0
    pa_cache_misses = 0
    progress_stage = "Fitting slices + PA sweep" if pa_offsets.size else "Fitting slices"
    progress_total = int(slice_indices.size)

    def _emit_progress(done: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(str(progress_stage), int(done), int(progress_total))
        except Exception:
            pass

    _emit_progress(0)
    for pos, idx in enumerate(slice_indices.tolist(), start=1):
        idx_i = int(idx)
        try:
            profile = sample_transverse_profile(
                flux_map=flux_map,
                support_mask=support_mask,
                ridge_xy=ridge,
                ridge_idx=idx_i,
                tangent_half_window=tangent_half_window,
                profile_step_px=profile_step_px,
                scan_half_width_px=scan_half_width_px,
            )
            fit = fit_transverse_gaussian(
                profile,
                mu_bound_px=mu_bound_px,
                baseline_mode=gaussian_baseline_mode,
                baseline_l1_flux=gaussian_baseline_l1_flux,
                baseline_noise_sigma_flux=gaussian_baseline_noise_sigma_flux,
            )
            record = {
                "ridge_idx": int(idx_i),
                "ridge_xy": tuple(profile.get("ridge_xy", (int(ridge[idx_i, 0]), int(ridge[idx_i, 1])))),
                "tangent_xy": tuple(profile.get("tangent_xy", (float("nan"), float("nan")))),
                "normal_xy": tuple(profile.get("normal_xy", (float("nan"), float("nan")))),
                "profile_step_px": float(profile_step_px),
                "scan_half_width_px": float(scan_half_width_px),
                "profile_scan_k_min_px": _safe_float(profile.get("scan_k_min_px", float("nan"))),
                "profile_scan_k_max_px": _safe_float(profile.get("scan_k_max_px", float("nan"))),
                "profile_full_span_px": (
                    float(_safe_float(profile.get("scan_k_max_px")) - _safe_float(profile.get("scan_k_min_px")))
                    if np.isfinite(_safe_float(profile.get("scan_k_min_px"))) and np.isfinite(_safe_float(profile.get("scan_k_max_px")))
                    else float("nan")
                ),
                "fit": _compact_gaussian_fit_result(fit),
            }
            if not fit.get("success"):
                fit_records.append(record)
                continue
            params = fit.get("params", {})
            mu = float(params.get("mu", float("nan")))
            fwhm_px = float(fit.get("fwhm_px", float("nan")))
            fwhm_fit_sigma_px = float(fit.get("fwhm_sigma_px", float("nan")))
            if (not np.isfinite(mu)) or (not np.isfinite(fwhm_px)) or fwhm_px <= 0.0:
                fit_records.append(record)
                continue
            px0, py0 = profile["ridge_xy"]
            nx, ny = profile["normal_xy"]
            left_k = float(mu - 0.5 * fwhm_px)
            right_k = float(mu + 0.5 * fwhm_px)
            p1 = (int(round(px0 + (nx * left_k))), int(round(py0 + (ny * left_k))))
            p2 = (int(round(px0 + (nx * right_k))), int(round(py0 + (ny * right_k))))
            width_lines.append((p1, p2))
            distance_px = float(ridge_dist[idx_i])
            distance_px_list.append(distance_px)
            fwhm_px_list.append(float(fwhm_px))
            fwhm_fit_sigma_px_list.append(float(fwhm_fit_sigma_px))
            if np.isfinite(scale) and scale > 0.0:
                fwhm_mas = float(fwhm_px * scale)
            else:
                fwhm_mas = float("nan")
            fwhm_mas_list.append(fwhm_mas)
            if np.isfinite(beam_px) and beam_px > 0.0:
                intrinsic_px = float(math.sqrt(max(0.0, (fwhm_px * fwhm_px) - (beam_px * beam_px))))
            else:
                intrinsic_px = float("nan")
            if np.isfinite(intrinsic_px) and intrinsic_px > 0.0 and np.isfinite(fwhm_fit_sigma_px):
                intrinsic_fit_sigma_px = float(abs(fwhm_px / intrinsic_px) * fwhm_fit_sigma_px)
            else:
                intrinsic_fit_sigma_px = float("nan")

            pa_sweep_summary = None
            fwhm_pa_rms_px = float("nan")
            intrinsic_pa_rms_px = float("nan")
            if pa_offsets.size:
                record_cache_key = _pa_sweep_record_cache_key(pa_measurement_cache_key, idx_i, (px0, py0))
                cached = _valid_cached_pa_sweep(
                    pa_cache_map.get(record_cache_key),
                    record_cache_key=record_cache_key,
                    measurement_cache_key=pa_measurement_cache_key,
                    offsets_deg=pa_offsets,
                )
                if cached is not None:
                    pa_sweep_summary = cached
                    pa_cache_hits += 1
                else:
                    pa_sweep_summary = _compute_pa_sweep_for_slice(
                        flux_map=flux_map,
                        support_mask=support_mask,
                        ridge_xy=ridge,
                        ridge_idx=idx_i,
                        ridge_point_xy=(px0, py0),
                        normal_xy=(nx, ny),
                        offsets_deg=pa_offsets,
                        tangent_half_window=tangent_half_window,
                        profile_step_px=profile_step_px,
                        scan_half_width_px=scan_half_width_px,
                        mu_bound_px=mu_bound_px,
                        baseline_mode=gaussian_baseline_mode,
                        baseline_l1_flux=gaussian_baseline_l1_flux,
                        baseline_noise_sigma_flux=gaussian_baseline_noise_sigma_flux,
                        beam_px=beam_px,
                        nominal_fwhm_px=fwhm_px,
                        nominal_intrinsic_fwhm_px=intrinsic_px,
                        measurement_cache_key=pa_measurement_cache_key,
                        record_cache_key=record_cache_key,
                        min_success_count=pa_min_success_count,
                    )
                    pa_cache_misses += 1
                fwhm_pa_rms_px = _safe_float(pa_sweep_summary.get("fwhm_rms_px", float("nan")))
                intrinsic_pa_rms_px = _safe_float(pa_sweep_summary.get("intrinsic_fwhm_rms_px", float("nan")))

            if np.isfinite(fwhm_fit_sigma_px) and np.isfinite(fwhm_pa_rms_px):
                fwhm_sigma_px = float(math.hypot(fwhm_fit_sigma_px, fwhm_pa_rms_px))
            elif np.isfinite(fwhm_pa_rms_px):
                fwhm_sigma_px = float(fwhm_pa_rms_px)
            else:
                fwhm_sigma_px = float(fwhm_fit_sigma_px)
            if np.isfinite(intrinsic_fit_sigma_px) and np.isfinite(intrinsic_pa_rms_px):
                intrinsic_sigma_px = float(math.hypot(intrinsic_fit_sigma_px, intrinsic_pa_rms_px))
            elif np.isfinite(intrinsic_pa_rms_px):
                intrinsic_sigma_px = float(intrinsic_pa_rms_px)
            else:
                intrinsic_sigma_px = float(intrinsic_fit_sigma_px)
            fwhm_sigma_px_list.append(float(fwhm_sigma_px))
            fwhm_pa_rms_px_list.append(float(fwhm_pa_rms_px))
            intrinsic_px_list.append(intrinsic_px)
            intrinsic_fit_sigma_px_list.append(intrinsic_fit_sigma_px)
            intrinsic_sigma_px_list.append(intrinsic_sigma_px)
            if np.isfinite(scale) and scale > 0.0 and np.isfinite(intrinsic_px):
                intrinsic_mas = float(intrinsic_px * scale)
                intrinsic_sigma_mas = float(intrinsic_sigma_px * scale) if np.isfinite(intrinsic_sigma_px) else float("nan")
                intrinsic_fit_sigma_mas = float(intrinsic_fit_sigma_px * scale) if np.isfinite(intrinsic_fit_sigma_px) else float("nan")
                intrinsic_pa_rms_mas = float(intrinsic_pa_rms_px * scale) if np.isfinite(intrinsic_pa_rms_px) else float("nan")
            else:
                intrinsic_mas = float("nan")
                intrinsic_sigma_mas = float("nan")
                intrinsic_fit_sigma_mas = float("nan")
                intrinsic_pa_rms_mas = float("nan")
            intrinsic_mas_list.append(intrinsic_mas)
            intrinsic_sigma_mas_list.append(intrinsic_sigma_mas)
            intrinsic_pa_rms_px_list.append(intrinsic_pa_rms_px)
            if np.isfinite(scale) and scale > 0.0:
                fwhm_sigma_mas = float(fwhm_sigma_px * scale) if np.isfinite(fwhm_sigma_px) else float("nan")
                fwhm_fit_sigma_mas = float(fwhm_fit_sigma_px * scale) if np.isfinite(fwhm_fit_sigma_px) else float("nan")
                fwhm_pa_rms_mas = float(fwhm_pa_rms_px * scale) if np.isfinite(fwhm_pa_rms_px) else float("nan")
            else:
                fwhm_sigma_mas = float("nan")
                fwhm_fit_sigma_mas = float("nan")
                fwhm_pa_rms_mas = float("nan")
            fwhm_sigma_mas_list.append(fwhm_sigma_mas)
            distance_from_core_px = float(distance_px + core_separation_px)
            opening_angle_raw = opening_angle_deg(fwhm_px, distance_from_core_px)
            opening_angle_intrinsic = opening_angle_deg(intrinsic_px, distance_from_core_px)
            opening_sigma_deg = opening_angle_error_deg(fwhm_px, distance_from_core_px, fwhm_sigma_px)
            intrinsic_opening_sigma_deg = opening_angle_error_deg(intrinsic_px, distance_from_core_px, intrinsic_sigma_px)
            opening_sigma_deg_list.append(opening_sigma_deg)
            intrinsic_opening_sigma_deg_list.append(intrinsic_opening_sigma_deg)
            record["width_line_xy"] = (p1, p2)
            record["distance_along_ridge_px"] = float(distance_px)
            record["distance_along_ridge_mas"] = float(distance_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
            record["distance_from_core_px"] = float(distance_from_core_px)
            record["distance_from_core_mas"] = float(distance_from_core_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan")
            record["fwhm_px"] = float(fwhm_px)
            record["fwhm_mas"] = float(fwhm_mas)
            record["fwhm_sigma_px"] = float(fwhm_sigma_px)
            record["fwhm_sigma_mas"] = float(fwhm_sigma_mas)
            record["fwhm_fit_sigma_px"] = float(fwhm_fit_sigma_px)
            record["fwhm_fit_sigma_mas"] = float(fwhm_fit_sigma_mas)
            record["fwhm_pa_rms_px"] = float(fwhm_pa_rms_px)
            record["fwhm_pa_rms_mas"] = float(fwhm_pa_rms_mas)
            record["intrinsic_fwhm_px"] = float(intrinsic_px)
            record["intrinsic_fwhm_mas"] = float(intrinsic_mas)
            record["intrinsic_fwhm_sigma_px"] = float(intrinsic_sigma_px)
            record["intrinsic_fwhm_sigma_mas"] = float(intrinsic_sigma_mas)
            record["intrinsic_fwhm_fit_sigma_px"] = float(intrinsic_fit_sigma_px)
            record["intrinsic_fwhm_fit_sigma_mas"] = float(intrinsic_fit_sigma_mas)
            record["intrinsic_fwhm_pa_rms_px"] = float(intrinsic_pa_rms_px)
            record["intrinsic_fwhm_pa_rms_mas"] = float(intrinsic_pa_rms_mas)
            record["opening_angle_deg"] = float(opening_angle_raw)
            record["intrinsic_opening_angle_deg"] = float(opening_angle_intrinsic)
            record["opening_angle_sigma_deg"] = float(opening_sigma_deg)
            record["intrinsic_opening_angle_sigma_deg"] = float(intrinsic_opening_sigma_deg)
            record["profile_peak_value"] = float(np.nanmax(np.asarray(profile.get("valid_y", []), dtype=np.float64))) if np.asarray(profile.get("valid_y", []), dtype=np.float64).size else float("nan")
            record["beam_deconvolved"] = bool(np.isfinite(beam_px) and beam_px > 0.0)
            if pa_sweep_summary is not None:
                record["pa_sweep"] = pa_sweep_summary
            fit_records.append(record)
        except Exception as exc:
            fit_records.append(
                {
                    "ridge_idx": int(idx_i),
                    "error": str(exc),
                }
            )
        finally:
            _emit_progress(pos)

    fwhm_px_arr = np.asarray(fwhm_px_list, dtype=np.float32)
    fwhm_mas_arr = np.asarray(fwhm_mas_list, dtype=np.float32)
    fwhm_sigma_px_arr = np.asarray(fwhm_sigma_px_list, dtype=np.float32)
    fwhm_sigma_mas_arr = np.asarray(fwhm_sigma_mas_list, dtype=np.float32)
    fwhm_fit_sigma_px_arr = np.asarray(fwhm_fit_sigma_px_list, dtype=np.float32)
    fwhm_pa_rms_px_arr = np.asarray(fwhm_pa_rms_px_list, dtype=np.float32)
    intrinsic_px_arr = np.asarray(intrinsic_px_list, dtype=np.float32)
    intrinsic_mas_arr = np.asarray(intrinsic_mas_list, dtype=np.float32)
    intrinsic_sigma_px_arr = np.asarray(intrinsic_sigma_px_list, dtype=np.float32)
    intrinsic_sigma_mas_arr = np.asarray(intrinsic_sigma_mas_list, dtype=np.float32)
    intrinsic_fit_sigma_px_arr = np.asarray(intrinsic_fit_sigma_px_list, dtype=np.float32)
    intrinsic_pa_rms_px_arr = np.asarray(intrinsic_pa_rms_px_list, dtype=np.float32)
    dist_px_arr = np.asarray(distance_px_list, dtype=np.float32)
    dist_from_core_px_arr = dist_px_arr + float(core_separation_px)
    opening_angle_deg_arr = np.asarray(
        [opening_angle_deg(width_px, dist_px) for width_px, dist_px in zip(fwhm_px_arr.tolist(), dist_from_core_px_arr.tolist())],
        dtype=np.float32,
    )
    intrinsic_opening_angle_deg_arr = np.asarray(
        [opening_angle_deg(width_px, dist_px) for width_px, dist_px in zip(intrinsic_px_arr.tolist(), dist_from_core_px_arr.tolist())],
        dtype=np.float32,
    )
    opening_angle_sigma_deg_arr = np.asarray(opening_sigma_deg_list, dtype=np.float32)
    intrinsic_opening_angle_sigma_deg_arr = np.asarray(intrinsic_opening_sigma_deg_list, dtype=np.float32)

    def _safe_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def _safe_median(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.median(finite)) if finite.size else float("nan")

    def _safe_robust_sigma(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size <= 1:
            return float("nan")
        med = float(np.median(finite))
        sigma = float(1.4826 * np.median(np.abs(finite - med)))
        if np.isfinite(sigma) and sigma > 0.0:
            return sigma
        return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")

    half_opening_angle_deg_arr = (0.5 * opening_angle_deg_arr).astype(np.float32)
    intrinsic_half_opening_angle_deg_arr = (0.5 * intrinsic_opening_angle_deg_arr).astype(np.float32)
    half_opening_angle_point_sigma_deg_arr = (0.5 * opening_angle_sigma_deg_arr).astype(np.float32)
    intrinsic_half_opening_angle_point_sigma_deg_arr = (0.5 * intrinsic_opening_angle_sigma_deg_arr).astype(np.float32)
    half_sigma_total_unblocked, half_sigma_scatter_unblocked, half_sigma_propagated_unblocked = _summary_sigma_with_measurement(
        half_opening_angle_deg_arr,
        half_opening_angle_point_sigma_deg_arr,
    )
    intrinsic_half_sigma_total_unblocked, intrinsic_half_sigma_scatter_unblocked, intrinsic_half_sigma_propagated_unblocked = _summary_sigma_with_measurement(
        intrinsic_half_opening_angle_deg_arr,
        intrinsic_half_opening_angle_point_sigma_deg_arr,
    )
    opening_summary = _blocked_median_summary(opening_angle_deg_arr, opening_angle_sigma_deg_arr, dist_px_arr, beam_px, scale)
    intrinsic_opening_summary = _blocked_median_summary(
        intrinsic_opening_angle_deg_arr,
        intrinsic_opening_angle_sigma_deg_arr,
        dist_px_arr,
        beam_px,
        scale,
    )
    half_summary = _blocked_median_summary(
        half_opening_angle_deg_arr,
        half_opening_angle_point_sigma_deg_arr,
        dist_px_arr,
        beam_px,
        scale,
    )
    intrinsic_half_summary = _blocked_median_summary(
        intrinsic_half_opening_angle_deg_arr,
        intrinsic_half_opening_angle_point_sigma_deg_arr,
        dist_px_arr,
        beam_px,
        scale,
    )

    return {
        "slice_indices": np.asarray(slice_indices, dtype=np.int32),
        "slice_sampling_mode": str(slice_sampling_mode),
        "requested_n_slices": int(n_slices),
        "slice_count": int(slice_indices.size),
        "slice_sampling_step_px": float(slice_spacing_px),
        "slice_sampling_step_mas": float(slice_spacing_mas),
        "slice_sampling_spacing_rule": "max_0.05mas_beam12",
        "slice_sampling_reference_step_mas": 0.05,
        "slice_sampling_max_step_mas": 0.05,
        "slice_sampling_beam_divisor": 12.0,
        "slice_sampling_target_count": int(slice_target_count),
        "slice_sampling_actual_median_spacing_px": float(slice_actual_spacing_px),
        "slice_sampling_actual_median_spacing_mas": float(slice_actual_spacing_mas),
        "requested_profile_step_px": float(requested_profile_step_px),
        "profile_step_px": float(profile_step_px),
        "profile_step_mas": float(profile_step_mas),
        "profile_step_mode": str(profile_step_mode),
        "profile_step_from_slice_spacing_divisor": 5.0,
        "profile_step_min_px": 0.25,
        "distance_along_ridge_px": dist_px_arr,
        "distance_from_core_px": dist_from_core_px_arr.astype(np.float32),
        "distance_from_core_mas": (dist_from_core_px_arr * scale).astype(np.float32) if np.isfinite(scale) and scale > 0.0 else np.full(dist_from_core_px_arr.shape, np.nan, dtype=np.float32),
        "fwhm_px": fwhm_px_arr,
        "fwhm_mas": fwhm_mas_arr,
        "fwhm_sigma_px": fwhm_sigma_px_arr,
        "fwhm_sigma_mas": fwhm_sigma_mas_arr,
        "fwhm_fit_sigma_px": fwhm_fit_sigma_px_arr,
        "fwhm_pa_rms_px": fwhm_pa_rms_px_arr,
        "intrinsic_fwhm_px": intrinsic_px_arr,
        "intrinsic_fwhm_mas": intrinsic_mas_arr,
        "intrinsic_fwhm_sigma_px": intrinsic_sigma_px_arr,
        "intrinsic_fwhm_sigma_mas": intrinsic_sigma_mas_arr,
        "intrinsic_fwhm_fit_sigma_px": intrinsic_fit_sigma_px_arr,
        "intrinsic_fwhm_pa_rms_px": intrinsic_pa_rms_px_arr,
        "opening_angle_deg": opening_angle_deg_arr,
        "intrinsic_opening_angle_deg": intrinsic_opening_angle_deg_arr,
        "half_opening_angle_deg": half_opening_angle_deg_arr,
        "intrinsic_half_opening_angle_deg": intrinsic_half_opening_angle_deg_arr,
        "opening_angle_sigma_deg": opening_angle_sigma_deg_arr,
        "intrinsic_opening_angle_sigma_deg": intrinsic_opening_angle_sigma_deg_arr,
        "half_opening_angle_point_sigma_deg": half_opening_angle_point_sigma_deg_arr,
        "intrinsic_half_opening_angle_point_sigma_deg": intrinsic_half_opening_angle_point_sigma_deg_arr,
        "width_lines_xy": width_lines,
        "beam_size_px": float(beam_px),
        "beam_size_mas": float(beam_size_mas(beam_major_mas, beam_minor_mas)),
        "gaussian_mu_bound_px": float(mu_bound_px),
        "gaussian_mu_bound_mas": float(mu_bound_px * scale) if np.isfinite(mu_bound_px) and np.isfinite(scale) and scale > 0.0 else float("nan"),
        "transverse_scan_mode": str(scan_mode),
        "transverse_scan_half_width_px": float(scan_half_width_px),
        "transverse_scan_half_width_mas": float(scan_half_width_mas),
        "transverse_scan_min_half_width_mas": 7.5,
        "transverse_scan_beam_factor": 6.0,
        "gaussian_baseline_mode": str(gaussian_baseline_mode),
        "gaussian_baseline_l1_flux": float(_safe_float(gaussian_baseline_l1_flux)),
        "gaussian_baseline_noise_sigma_flux": float(_safe_float(gaussian_baseline_noise_sigma_flux)),
        "core_separation_px": float(core_separation_px),
        "core_separation_mas": float(core_separation_px * scale) if np.isfinite(scale) and scale > 0.0 else float("nan"),
        "trim_start_frac": float(trim_start_frac),
        "trim_end_frac": float(trim_end_frac),
        "tangent_half_window": int(tangent_half_window),
        "tangent_half_window_auto": bool(tangent_half_window_auto),
        "ridge_median_spacing_px": float(ridge_median_spacing_px),
        "tangent_target_span_px": float(tangent_target_span_px),
        "fit_records": fit_records,
        "pa_sweep": {
            "enabled": bool(pa_offsets.size),
            "range_deg": float(pa_sweep_range_deg) if pa_offsets.size else float("nan"),
            "step_deg": float(pa_sweep_step_deg) if pa_offsets.size else float("nan"),
            "offsets_deg": pa_offsets,
            "measurement_cache_key": str(pa_measurement_cache_key),
            "min_success_count": int(pa_min_success_count),
            "cache_hits": int(pa_cache_hits),
            "cache_misses": int(pa_cache_misses),
        },
        "valid_count": int(len(fwhm_px_arr)),
        "mean_fwhm_px": _safe_mean(fwhm_px_arr),
        "median_fwhm_px": _safe_median(fwhm_px_arr),
        "mean_fwhm_mas": _safe_mean(fwhm_mas_arr),
        "median_fwhm_mas": _safe_median(fwhm_mas_arr),
        "mean_intrinsic_fwhm_px": _safe_mean(intrinsic_px_arr),
        "mean_intrinsic_fwhm_mas": _safe_mean(intrinsic_mas_arr),
        "mean_opening_angle_deg": _safe_mean(opening_angle_deg_arr),
        "median_opening_angle_deg": float(opening_summary["median"]),
        "median_opening_angle_deg_unblocked": _safe_median(opening_angle_deg_arr),
        "mean_intrinsic_opening_angle_deg": _safe_mean(intrinsic_opening_angle_deg_arr),
        "median_intrinsic_opening_angle_deg": float(intrinsic_opening_summary["median"]),
        "median_intrinsic_opening_angle_deg_unblocked": _safe_median(intrinsic_opening_angle_deg_arr),
        "median_half_opening_angle_deg": float(half_summary["median"]),
        "median_half_opening_angle_deg_unblocked": _safe_median(half_opening_angle_deg_arr),
        "half_opening_angle_sigma_deg": float(half_summary["sigma"]),
        "half_opening_angle_scatter_deg": float(half_summary["scatter"]),
        "half_opening_angle_measurement_sigma_deg": float(half_summary["measurement_sigma"]),
        "half_opening_angle_sigma_deg_unblocked": float(half_sigma_total_unblocked),
        "half_opening_angle_scatter_deg_unblocked": float(half_sigma_scatter_unblocked),
        "half_opening_angle_measurement_sigma_deg_unblocked": float(half_sigma_propagated_unblocked),
        "median_intrinsic_half_opening_angle_deg": float(intrinsic_half_summary["median"]),
        "median_intrinsic_half_opening_angle_deg_unblocked": _safe_median(intrinsic_half_opening_angle_deg_arr),
        "intrinsic_half_opening_angle_sigma_deg": float(intrinsic_half_summary["sigma"]),
        "intrinsic_half_opening_angle_scatter_deg": float(intrinsic_half_summary["scatter"]),
        "intrinsic_half_opening_angle_measurement_sigma_deg": float(intrinsic_half_summary["measurement_sigma"]),
        "intrinsic_half_opening_angle_sigma_deg_unblocked": float(intrinsic_half_sigma_total_unblocked),
        "intrinsic_half_opening_angle_scatter_deg_unblocked": float(intrinsic_half_sigma_scatter_unblocked),
        "intrinsic_half_opening_angle_measurement_sigma_deg_unblocked": float(intrinsic_half_sigma_propagated_unblocked),
        "median_summary_method": str(half_summary["method"]),
        "median_summary_block_reference": "half_beam",
        "median_summary_block_size_slices": int(half_summary["block_size_slices"]),
        "median_summary_block_count": int(half_summary["block_count"]),
        "median_summary_block_target_px": float(half_summary["block_target_px"]),
        "median_summary_block_target_mas": float(half_summary["block_target_mas"]),
        "median_summary_spacing_px": float(half_summary["block_spacing_px"]),
        "median_summary_spacing_mas": float(half_summary["block_spacing_mas"]),
    }


def apply_core_separation_to_measure_result(
    measure_result: Dict[str, object],
    *,
    core_separation_px: float = 0.0,
    scale_mas_per_px: Optional[float] = None,
) -> Dict[str, object]:
    result = dict(measure_result or {})
    core_separation_px = float(max(0.0, core_separation_px))
    scale = float(scale_mas_per_px) if scale_mas_per_px is not None else float("nan")

    def _safe_mean(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def _safe_median(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.median(finite)) if finite.size else float("nan")

    def _safe_robust_sigma(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        if finite.size <= 1:
            return float("nan")
        med = float(np.median(finite))
        sigma = float(1.4826 * np.median(np.abs(finite - med)))
        if np.isfinite(sigma) and sigma > 0.0:
            return sigma
        return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")

    dist_px_arr = np.asarray(result.get("distance_along_ridge_px", np.array([], dtype=np.float32)), dtype=np.float32)
    fwhm_px_arr = np.asarray(result.get("fwhm_px", np.array([], dtype=np.float32)), dtype=np.float32)
    fwhm_sigma_px_arr = np.asarray(result.get("fwhm_sigma_px", np.array([], dtype=np.float32)), dtype=np.float32)
    intrinsic_px_arr = np.asarray(result.get("intrinsic_fwhm_px", np.array([], dtype=np.float32)), dtype=np.float32)
    intrinsic_sigma_px_arr = np.asarray(result.get("intrinsic_fwhm_sigma_px", np.array([], dtype=np.float32)), dtype=np.float32)
    beam_px = _safe_float(result.get("beam_size_px", float("nan")))
    dist_from_core_px_arr = dist_px_arr + float(core_separation_px)
    result["distance_from_core_px"] = dist_from_core_px_arr.astype(np.float32)
    if np.isfinite(scale) and scale > 0.0:
        result["distance_from_core_mas"] = (dist_from_core_px_arr * scale).astype(np.float32)
        result["core_separation_mas"] = float(core_separation_px * scale)
    else:
        result["distance_from_core_mas"] = np.full(dist_from_core_px_arr.shape, np.nan, dtype=np.float32)
        result["core_separation_mas"] = float("nan")
    result["core_separation_px"] = float(core_separation_px)
    result["opening_angle_deg"] = np.asarray(
        [opening_angle_deg(width_px, dist_px) for width_px, dist_px in zip(fwhm_px_arr.tolist(), dist_from_core_px_arr.tolist())],
        dtype=np.float32,
    )
    result["intrinsic_opening_angle_deg"] = np.asarray(
        [opening_angle_deg(width_px, dist_px) for width_px, dist_px in zip(intrinsic_px_arr.tolist(), dist_from_core_px_arr.tolist())],
        dtype=np.float32,
    )
    result["opening_angle_sigma_deg"] = np.asarray(
        [opening_angle_error_deg(width_px, dist_px, sigma_px) for width_px, dist_px, sigma_px in zip(fwhm_px_arr.tolist(), dist_from_core_px_arr.tolist(), fwhm_sigma_px_arr.tolist())],
        dtype=np.float32,
    )
    result["intrinsic_opening_angle_sigma_deg"] = np.asarray(
        [opening_angle_error_deg(width_px, dist_px, sigma_px) for width_px, dist_px, sigma_px in zip(intrinsic_px_arr.tolist(), dist_from_core_px_arr.tolist(), intrinsic_sigma_px_arr.tolist())],
        dtype=np.float32,
    )
    result["half_opening_angle_deg"] = (0.5 * np.asarray(result["opening_angle_deg"], dtype=np.float32)).astype(np.float32)
    result["intrinsic_half_opening_angle_deg"] = (0.5 * np.asarray(result["intrinsic_opening_angle_deg"], dtype=np.float32)).astype(np.float32)
    result["half_opening_angle_point_sigma_deg"] = (0.5 * np.asarray(result["opening_angle_sigma_deg"], dtype=np.float32)).astype(np.float32)
    result["intrinsic_half_opening_angle_point_sigma_deg"] = (0.5 * np.asarray(result["intrinsic_opening_angle_sigma_deg"], dtype=np.float32)).astype(np.float32)
    half_sigma_total_unblocked, half_sigma_scatter_unblocked, half_sigma_propagated_unblocked = _summary_sigma_with_measurement(
        np.asarray(result["half_opening_angle_deg"], dtype=np.float32),
        np.asarray(result["half_opening_angle_point_sigma_deg"], dtype=np.float32),
    )
    intrinsic_half_sigma_total_unblocked, intrinsic_half_sigma_scatter_unblocked, intrinsic_half_sigma_propagated_unblocked = _summary_sigma_with_measurement(
        np.asarray(result["intrinsic_half_opening_angle_deg"], dtype=np.float32),
        np.asarray(result["intrinsic_half_opening_angle_point_sigma_deg"], dtype=np.float32),
    )
    opening_summary = _blocked_median_summary(
        np.asarray(result["opening_angle_deg"], dtype=np.float32),
        np.asarray(result["opening_angle_sigma_deg"], dtype=np.float32),
        dist_px_arr,
        beam_px,
        scale,
    )
    intrinsic_opening_summary = _blocked_median_summary(
        np.asarray(result["intrinsic_opening_angle_deg"], dtype=np.float32),
        np.asarray(result["intrinsic_opening_angle_sigma_deg"], dtype=np.float32),
        dist_px_arr,
        beam_px,
        scale,
    )
    half_summary = _blocked_median_summary(
        np.asarray(result["half_opening_angle_deg"], dtype=np.float32),
        np.asarray(result["half_opening_angle_point_sigma_deg"], dtype=np.float32),
        dist_px_arr,
        beam_px,
        scale,
    )
    intrinsic_half_summary = _blocked_median_summary(
        np.asarray(result["intrinsic_half_opening_angle_deg"], dtype=np.float32),
        np.asarray(result["intrinsic_half_opening_angle_point_sigma_deg"], dtype=np.float32),
        dist_px_arr,
        beam_px,
        scale,
    )
    result["mean_opening_angle_deg"] = _safe_mean(np.asarray(result["opening_angle_deg"], dtype=np.float32))
    result["median_opening_angle_deg"] = float(opening_summary["median"])
    result["median_opening_angle_deg_unblocked"] = _safe_median(np.asarray(result["opening_angle_deg"], dtype=np.float32))
    result["mean_intrinsic_opening_angle_deg"] = _safe_mean(np.asarray(result["intrinsic_opening_angle_deg"], dtype=np.float32))
    result["median_intrinsic_opening_angle_deg"] = float(intrinsic_opening_summary["median"])
    result["median_intrinsic_opening_angle_deg_unblocked"] = _safe_median(np.asarray(result["intrinsic_opening_angle_deg"], dtype=np.float32))
    result["median_half_opening_angle_deg"] = float(half_summary["median"])
    result["median_half_opening_angle_deg_unblocked"] = _safe_median(np.asarray(result["half_opening_angle_deg"], dtype=np.float32))
    result["half_opening_angle_sigma_deg"] = float(half_summary["sigma"])
    result["half_opening_angle_scatter_deg"] = float(half_summary["scatter"])
    result["half_opening_angle_measurement_sigma_deg"] = float(half_summary["measurement_sigma"])
    result["half_opening_angle_sigma_deg_unblocked"] = float(half_sigma_total_unblocked)
    result["half_opening_angle_scatter_deg_unblocked"] = float(half_sigma_scatter_unblocked)
    result["half_opening_angle_measurement_sigma_deg_unblocked"] = float(half_sigma_propagated_unblocked)
    result["median_intrinsic_half_opening_angle_deg"] = float(intrinsic_half_summary["median"])
    result["median_intrinsic_half_opening_angle_deg_unblocked"] = _safe_median(np.asarray(result["intrinsic_half_opening_angle_deg"], dtype=np.float32))
    result["intrinsic_half_opening_angle_sigma_deg"] = float(intrinsic_half_summary["sigma"])
    result["intrinsic_half_opening_angle_scatter_deg"] = float(intrinsic_half_summary["scatter"])
    result["intrinsic_half_opening_angle_measurement_sigma_deg"] = float(intrinsic_half_summary["measurement_sigma"])
    result["intrinsic_half_opening_angle_sigma_deg_unblocked"] = float(intrinsic_half_sigma_total_unblocked)
    result["intrinsic_half_opening_angle_scatter_deg_unblocked"] = float(intrinsic_half_sigma_scatter_unblocked)
    result["intrinsic_half_opening_angle_measurement_sigma_deg_unblocked"] = float(intrinsic_half_sigma_propagated_unblocked)
    result["median_summary_method"] = str(half_summary["method"])
    result["median_summary_block_reference"] = "half_beam"
    result["median_summary_block_size_slices"] = int(half_summary["block_size_slices"])
    result["median_summary_block_count"] = int(half_summary["block_count"])
    result["median_summary_block_target_px"] = float(half_summary["block_target_px"])
    result["median_summary_block_target_mas"] = float(half_summary["block_target_mas"])
    result["median_summary_spacing_px"] = float(half_summary["block_spacing_px"])
    result["median_summary_spacing_mas"] = float(half_summary["block_spacing_mas"])

    fit_records = list(result.get("fit_records", []) or [])
    updated_records: List[Dict[str, object]] = []
    for item in fit_records:
        if not isinstance(item, dict):
            updated_records.append(item)
            continue
        rec = dict(item)
        distance_px = _safe_float(rec.get("distance_along_ridge_px", float("nan")))
        distance_from_core_px = float(distance_px + core_separation_px) if np.isfinite(distance_px) else float("nan")
        rec["distance_from_core_px"] = float(distance_from_core_px)
        rec["distance_from_core_mas"] = float(distance_from_core_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(distance_from_core_px) else float("nan")
        fwhm_px = _safe_float(rec.get("fwhm_px", float("nan")))
        intrinsic_px = _safe_float(rec.get("intrinsic_fwhm_px", float("nan")))
        rec["opening_angle_deg"] = float(opening_angle_deg(fwhm_px, distance_from_core_px))
        rec["intrinsic_opening_angle_deg"] = float(opening_angle_deg(intrinsic_px, distance_from_core_px))
        rec["opening_angle_sigma_deg"] = float(
            opening_angle_error_deg(
                fwhm_px,
                distance_from_core_px,
                _safe_float(rec.get("fwhm_sigma_px", float("nan"))),
            )
        )
        rec["intrinsic_opening_angle_sigma_deg"] = float(
            opening_angle_error_deg(
                intrinsic_px,
                distance_from_core_px,
                _safe_float(rec.get("intrinsic_fwhm_sigma_px", float("nan"))),
            )
        )
        updated_records.append(rec)
    result["fit_records"] = updated_records
    return result
