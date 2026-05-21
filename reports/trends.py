from __future__ import annotations

"""Report and trend calculations for ridgeline width measurements."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ridgeline.analysis import (
    _combine_independent_sigmas,
    _linear_fit_with_cov,
    _linear_model_sigma,
    _median_propagated_sigma,
    _opening_model_sigma_from_log_width,
    _robust_sigma,
    _safe_float,
    evaluate_gaussian_fit_stability,
    opening_angle_deg,
    opening_angle_error_deg,
)


def paper_fig7_eastern_broken_power_law(z_mas: Sequence[float]) -> np.ndarray:
    z_mas = np.asarray(z_mas, dtype=np.float64)
    w0 = 0.21
    ku = 0.17
    kd = 1.01
    zb = 1.51
    sharpness = 10.0
    x = z_mas / zb
    return w0 * (2.0 ** ((ku - kd) / sharpness)) * (x ** ku) * ((1.0 + x ** sharpness) ** ((kd - ku) / sharpness))


def build_gaussian_report_rows(
    measure_result: Dict[str, object],
    scale_mas_per_px: Optional[float] = None,
    *,
    use_raw_width: bool = False,
    use_raw_fallback: Optional[bool] = None,
) -> List[Dict[str, object]]:
    if use_raw_fallback is not None:
        use_raw_width = bool(use_raw_fallback)
    result = dict(measure_result or {})
    fit_records = list(result.get("fit_records", []) or [])
    scale = float(scale_mas_per_px) if scale_mas_per_px is not None else float("nan")
    beam_px = float(result.get("beam_size_px", float("nan")))
    beam_mas = float(result.get("beam_size_mas", float("nan")))
    rows: List[Dict[str, object]] = []
    slice_order = 0
    for item in fit_records:
        if not isinstance(item, dict):
            continue
        fwhm_px = _safe_float(item.get("fwhm_px", float("nan")))
        if not np.isfinite(fwhm_px) or fwhm_px <= 0.0:
            continue
        fwhm_sigma_px = _safe_float(item.get("fwhm_sigma_px", float("nan")))
        fwhm_fit_sigma_px = _safe_float(item.get("fwhm_fit_sigma_px", fwhm_sigma_px))
        fwhm_pa_rms_px = _safe_float(item.get("fwhm_pa_rms_px", float("nan")))
        intrinsic_px = _safe_float(item.get("intrinsic_fwhm_px", float("nan")))
        intrinsic_sigma_px = _safe_float(item.get("intrinsic_fwhm_sigma_px", float("nan")))
        intrinsic_fit_sigma_px = _safe_float(item.get("intrinsic_fwhm_fit_sigma_px", intrinsic_sigma_px))
        intrinsic_pa_rms_px = _safe_float(item.get("intrinsic_fwhm_pa_rms_px", float("nan")))
        has_intrinsic = bool(np.isfinite(intrinsic_px) and intrinsic_px > 0.0)
        used_width_px = intrinsic_px if has_intrinsic else (fwhm_px if use_raw_width else float("nan"))
        used_width_sigma_px = intrinsic_sigma_px if has_intrinsic else (fwhm_sigma_px if use_raw_width else float("nan"))
        used_width_source = "intrinsic" if has_intrinsic else ("raw_width" if np.isfinite(used_width_px) else "excluded")
        if used_width_source == "excluded":
            continue
        distance_px = _safe_float(item.get("distance_from_core_px", item.get("distance_along_ridge_px", float("nan"))))
        distance_mas = _safe_float(item.get("distance_from_core_mas", item.get("distance_along_ridge_mas", float("nan"))))
        raw_angle = _safe_float(item.get("opening_angle_deg", opening_angle_deg(fwhm_px, distance_px)))
        intrinsic_angle = _safe_float(item.get("intrinsic_opening_angle_deg", opening_angle_deg(intrinsic_px, distance_px)))
        used_angle = intrinsic_angle if has_intrinsic else (raw_angle if use_raw_width else float("nan"))
        raw_angle_sigma = _safe_float(item.get("opening_angle_sigma_deg", float("nan")))
        intrinsic_angle_sigma = _safe_float(item.get("intrinsic_opening_angle_sigma_deg", float("nan")))
        used_angle_sigma = intrinsic_angle_sigma if has_intrinsic else (raw_angle_sigma if use_raw_width else float("nan"))
        used_width_mas = float(used_width_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(used_width_px) else float("nan")
        used_width_sigma_mas = float(used_width_sigma_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(used_width_sigma_px) else float("nan")
        raw_width_mas = float(fwhm_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(fwhm_px) else float("nan")
        raw_width_sigma_mas = float(fwhm_sigma_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(fwhm_sigma_px) else float("nan")
        raw_width_fit_sigma_mas = float(fwhm_fit_sigma_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(fwhm_fit_sigma_px) else float("nan")
        raw_width_pa_rms_mas = float(fwhm_pa_rms_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(fwhm_pa_rms_px) else float("nan")
        intrinsic_width_mas = float(intrinsic_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(intrinsic_px) else float("nan")
        intrinsic_width_sigma_mas = float(intrinsic_sigma_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(intrinsic_sigma_px) else float("nan")
        intrinsic_width_fit_sigma_mas = float(intrinsic_fit_sigma_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(intrinsic_fit_sigma_px) else float("nan")
        intrinsic_width_pa_rms_mas = float(intrinsic_pa_rms_px * scale) if np.isfinite(scale) and scale > 0.0 and np.isfinite(intrinsic_pa_rms_px) else float("nan")
        fit = item.get("fit", {})
        fit = fit if isinstance(fit, dict) else {}
        params = fit.get("params", {})
        params = params if isinstance(params, dict) else {}
        fit_window = fit.get("fit_window", {})
        fit_window = fit_window if isinstance(fit_window, dict) else {}
        gaussian_rmse = _safe_float(fit.get("rmse", float("nan")))
        gaussian_baseline = _safe_float(params.get("baseline", float("nan")))
        gaussian_baseline_lower = _safe_float(params.get("baseline_bound_lower", float("nan")))
        gaussian_baseline_upper = _safe_float(params.get("baseline_bound_upper", float("nan")))
        gaussian_amplitude = _safe_float(params.get("amplitude", float("nan")))
        gaussian_sigma_px = _safe_float(params.get("sigma", float("nan")))
        fit_window_left = _safe_float(fit_window.get("left_x", float("nan")))
        fit_window_right = _safe_float(fit_window.get("right_x", float("nan")))
        fit_window_span = (
            float(fit_window_right - fit_window_left)
            if np.isfinite(fit_window_left) and np.isfinite(fit_window_right) and fit_window_right > fit_window_left
            else float("nan")
        )
        try:
            full_fit_x = np.asarray(fit.get("full_x", fit.get("x", [])), dtype=np.float64)
            full_fit_x = full_fit_x[np.isfinite(full_fit_x)]
            full_profile_span = float(np.ptp(full_fit_x)) if full_fit_x.size >= 2 else float("nan")
        except Exception:
            full_profile_span = float("nan")
        if not np.isfinite(full_profile_span):
            full_profile_span = _safe_float(item.get("profile_full_span_px", float("nan")))
        stability = evaluate_gaussian_fit_stability(item, fit=fit)
        if "gaussian_unstable" in item:
            for key in list(stability.keys()):
                if key in item:
                    stability[key] = item[key]
        instability_reasons = list(stability.get("gaussian_unstable_reasons", []) or [])
        slice_order += 1
        rows.append(
            {
                "slice_order": int(slice_order),
                "ridge_idx": int(item.get("ridge_idx", -1)),
                "distance_from_core_px": float(distance_px),
                "distance_from_core_mas": float(distance_mas),
                "beam_deconvolved": bool(np.isfinite(beam_px) and beam_px > 0.0),
                "beam_size_px": float(beam_px),
                "beam_size_mas": float(beam_mas),
                "gaussian_raw_width_px": float(fwhm_px),
                "gaussian_raw_width_sigma_px": float(fwhm_sigma_px),
                "gaussian_raw_width_fit_sigma_px": float(fwhm_fit_sigma_px),
                "gaussian_raw_width_pa_rms_px": float(fwhm_pa_rms_px),
                "gaussian_raw_width_mas": float(raw_width_mas),
                "gaussian_raw_width_sigma_mas": float(raw_width_sigma_mas),
                "gaussian_raw_width_fit_sigma_mas": float(raw_width_fit_sigma_mas),
                "gaussian_raw_width_pa_rms_mas": float(raw_width_pa_rms_mas),
                "gaussian_intrinsic_width_px": float(intrinsic_px),
                "gaussian_intrinsic_width_sigma_px": float(intrinsic_sigma_px),
                "gaussian_intrinsic_width_fit_sigma_px": float(intrinsic_fit_sigma_px),
                "gaussian_intrinsic_width_pa_rms_px": float(intrinsic_pa_rms_px),
                "gaussian_intrinsic_width_mas": float(intrinsic_width_mas),
                "gaussian_intrinsic_width_sigma_mas": float(intrinsic_width_sigma_mas),
                "gaussian_intrinsic_width_fit_sigma_mas": float(intrinsic_width_fit_sigma_mas),
                "gaussian_intrinsic_width_pa_rms_mas": float(intrinsic_width_pa_rms_mas),
                "gaussian_width_px": float(used_width_px),
                "gaussian_width_sigma_px": float(used_width_sigma_px),
                "gaussian_width_mas": float(used_width_mas),
                "gaussian_width_sigma_mas": float(used_width_sigma_mas),
                "gaussian_valid_raw": True,
                "gaussian_valid_intrinsic": bool(has_intrinsic),
                "gaussian_raw_angle_deg": float(raw_angle),
                "gaussian_raw_angle_sigma_deg": float(raw_angle_sigma),
                "gaussian_intrinsic_angle_deg": float(intrinsic_angle),
                "gaussian_intrinsic_angle_sigma_deg": float(intrinsic_angle_sigma),
                "gaussian_angle_deg": float(used_angle),
                "gaussian_angle_sigma_deg": float(used_angle_sigma),
                "profile_peak_value": _safe_float(item.get("profile_peak_value", float("nan"))),
                "gaussian_rmse": float(gaussian_rmse),
                "gaussian_baseline": float(gaussian_baseline),
                "gaussian_baseline_mode": str(params.get("baseline_mode", "")),
                "gaussian_baseline_bound_lower": float(gaussian_baseline_lower),
                "gaussian_baseline_bound_upper": float(gaussian_baseline_upper),
                "gaussian_baseline_l1_flux": _safe_float(params.get("baseline_l1_flux", float("nan"))),
                "gaussian_baseline_noise_sigma_flux": _safe_float(params.get("baseline_noise_sigma_flux", float("nan"))),
                "gaussian_amplitude": float(gaussian_amplitude),
                "gaussian_sigma_px": float(gaussian_sigma_px),
                "gaussian_fit_window_left_px": float(fit_window_left),
                "gaussian_fit_window_right_px": float(fit_window_right),
                "gaussian_fit_window_span_px": float(fit_window_span),
                "gaussian_full_profile_span_px": float(full_profile_span),
                "gaussian_half_left_px": _safe_float(stability.get("gaussian_half_left_px", float("nan"))),
                "gaussian_half_right_px": _safe_float(stability.get("gaussian_half_right_px", float("nan"))),
                "gaussian_mu_bound_fraction": _safe_float(stability.get("gaussian_mu_bound_fraction", float("nan"))),
                "gaussian_pa_rms_fraction": _safe_float(stability.get("gaussian_pa_rms_fraction", float("nan"))),
                "ridge_center_half_flux_level": _safe_float(stability.get("ridge_center_half_flux_level", float("nan"))),
                "ridge_center_half_flux_left_px": _safe_float(stability.get("ridge_center_half_flux_left_px", float("nan"))),
                "ridge_center_half_flux_right_px": _safe_float(stability.get("ridge_center_half_flux_right_px", float("nan"))),
                "ridge_center_half_flux_width_px": _safe_float(stability.get("ridge_center_half_flux_width_px", float("nan"))),
                "gaussian_fwhm_to_half_flux_width_ratio": _safe_float(
                    stability.get("gaussian_fwhm_to_half_flux_width_ratio", float("nan"))
                ),
                "gaussian_unstable": bool(stability.get("gaussian_unstable", bool(instability_reasons))),
                "gaussian_unstable_reasons": list(instability_reasons),
                "used_width_source": str(used_width_source),
            }
        )
    return rows


def compute_local_k(
    distances: Sequence[float],
    widths: Sequence[float],
    *,
    window_points: int = 7,
    min_points: int = 5,
) -> np.ndarray:
    x = np.asarray(distances, dtype=np.float64)
    y = np.asarray(widths, dtype=np.float64)
    n = int(min(x.size, y.size))
    out = np.full(n, np.nan, dtype=np.float32)
    if n <= 0:
        return out
    window_points = int(max(3, window_points))
    if window_points % 2 == 0:
        window_points += 1
    half = window_points // 2
    min_points = int(max(3, min_points))
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        xw = x[i0:i1]
        yw = y[i0:i1]
        valid = np.isfinite(xw) & np.isfinite(yw) & (xw > 0.0) & (yw > 0.0)
        if int(np.count_nonzero(valid)) < min_points:
            continue
        lx = np.log10(xw[valid])
        ly = np.log10(yw[valid])
        if lx.size < min_points or np.allclose(lx, lx[0]):
            continue
        try:
            slope, _intercept = np.polyfit(lx, ly, 1)
        except Exception:
            continue
        out[i] = float(slope)
    return out


def fit_power_law_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    start_slice_order: int,
    end_slice_order: int,
    distance_unit: str = "auto",
) -> Dict[str, object]:
    items = list(rows)
    if not items:
        raise ValueError("No report rows available for fit.")
    distance_unit = str(distance_unit).strip().lower()
    if distance_unit == "auto":
        any_mas = any(np.isfinite(_safe_float(row.get("distance_from_core_mas", float("nan")))) for row in items)
        distance_unit = "mas" if any_mas else "px"
    x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
    subset = [
        row for row in items
        if int(row.get("slice_order", -1)) >= int(start_slice_order)
        and int(row.get("slice_order", -1)) <= int(end_slice_order)
    ]
    if len(subset) < 3:
        raise ValueError("Not enough slices in selected fit range.")
    x = np.asarray([_safe_float(row.get(x_key, float("nan"))) for row in subset], dtype=np.float64)
    y = np.asarray([_safe_float(row.get("gaussian_width_px" if distance_unit == "px" else "gaussian_width_mas", float("nan"))) for row in subset], dtype=np.float64)
    y_sigma = np.asarray([_safe_float(row.get("gaussian_width_sigma_px" if distance_unit == "px" else "gaussian_width_sigma_mas", float("nan"))) for row in subset], dtype=np.float64)
    theta_sigma = np.asarray([_safe_float(row.get("gaussian_angle_sigma_deg", float("nan"))) for row in subset], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("Not enough valid slices in selected fit range.")
    lx = np.log10(x[valid])
    ly = np.log10(y[valid])
    sigma_log_y = np.full(ly.shape, np.nan, dtype=np.float64)
    yv = y[valid]
    ysig = y_sigma[valid]
    finite_sigma = np.isfinite(ysig) & (ysig > 0.0) & np.isfinite(yv) & (yv > 0.0)
    sigma_log_y[finite_sigma] = ysig[finite_sigma] / (yv[finite_sigma] * np.log(10.0))
    fit_info = _linear_fit_with_cov(lx, ly, sigma_log_y, min_points=3)
    if fit_info is None:
        raise ValueError("Failed to fit weighted power-law trend.")
    slope = float(fit_info["slope"])
    intercept = float(fit_info["intercept"])
    cov = np.asarray(fit_info.get("cov", np.full((2, 2), np.nan)), dtype=np.float64)
    k_sigma = float(math.sqrt(max(0.0, float(cov[0, 0])))) if cov.shape == (2, 2) and np.isfinite(cov[0, 0]) else float("nan")
    intercept_sigma = float(math.sqrt(max(0.0, float(cov[1, 1])))) if cov.shape == (2, 2) and np.isfinite(cov[1, 1]) else float("nan")
    all_x = np.asarray([_safe_float(row.get(x_key, float("nan"))) for row in items], dtype=np.float64)
    all_y = np.asarray(
        [_safe_float(row.get("gaussian_width_px" if distance_unit == "px" else "gaussian_width_mas", float("nan"))) for row in items],
        dtype=np.float64,
    )
    model_width = np.full(all_x.shape, np.nan, dtype=np.float64)
    all_valid = np.isfinite(all_x) & (all_x > 0.0)
    model_width[all_valid] = (10.0 ** float(intercept)) * (all_x[all_valid] ** float(slope))
    model_log_width_sigma = _linear_model_sigma(np.log10(np.where(all_valid, all_x, np.nan)), cov)
    model_width_sigma = np.full(all_x.shape, np.nan, dtype=np.float64)
    model_width_sigma[all_valid & np.isfinite(model_log_width_sigma)] = (
        math.log(10.0) * model_width[all_valid & np.isfinite(model_log_width_sigma)] * model_log_width_sigma[all_valid & np.isfinite(model_log_width_sigma)]
    )
    all_log_residual = np.full(all_x.shape, np.nan, dtype=np.float64)
    resid_valid = np.isfinite(all_x) & np.isfinite(all_y) & np.isfinite(model_width) & (all_x > 0.0) & (all_y > 0.0) & (model_width > 0.0)
    if np.any(resid_valid):
        all_log_residual[resid_valid] = np.log10(all_y[resid_valid]) - np.log10(model_width[resid_valid])
    obs_open = np.asarray([_safe_float(row.get("gaussian_angle_deg", float("nan"))) for row in items], dtype=np.float64)
    model_open = np.full(all_x.shape, np.nan, dtype=np.float64)
    for idx, xv in enumerate(all_x.tolist()):
        model_open[idx] = opening_angle_deg(model_width[idx], xv)
    model_open_sigma = _opening_model_sigma_from_log_width(model_width, all_x, model_log_width_sigma)
    fit_obs_open = np.asarray([_safe_float(row.get("gaussian_angle_deg", float("nan"))) for row in subset], dtype=np.float64)[valid]
    fit_model_open = np.asarray([opening_angle_deg((10.0 ** float(intercept)) * (xv ** float(slope)), xv) for xv in x[valid]], dtype=np.float64)
    fit_model_log_sigma = _linear_model_sigma(lx, cov)
    fit_model_width = (10.0 ** float(intercept)) * np.power(x[valid], float(slope))
    fit_model_open_sigma = _opening_model_sigma_from_log_width(fit_model_width, x[valid], fit_model_log_sigma)
    resid = fit_obs_open - fit_model_open
    fit_log = intercept + (slope * lx)
    fit_log_residual = ly - fit_log
    finite_resid = resid[np.isfinite(resid)]
    finite_log_resid = fit_log_residual[np.isfinite(fit_log_residual)]
    finite_fit_model_open_mask = np.isfinite(fit_model_open)
    finite_fit_model_open = fit_model_open[finite_fit_model_open_mask]
    opening_residual_all = np.full(all_x.shape, np.nan, dtype=np.float64)
    opening_resid_valid = np.isfinite(obs_open) & np.isfinite(model_open) & (obs_open > 0.0) & (model_open > 0.0)
    if np.any(opening_resid_valid):
        opening_residual_all[opening_resid_valid] = obs_open[opening_resid_valid] - model_open[opening_resid_valid]
    if finite_fit_model_open.size > 0:
        opening_fit_median = float(np.median(finite_fit_model_open))
        opening_fit_mean = float(np.mean(finite_fit_model_open))
        finite_idx = np.flatnonzero(finite_fit_model_open_mask)
        median_local_idx = int(finite_idx[int(np.argmin(np.abs(fit_model_open[finite_idx] - opening_fit_median)))])
        opening_fit_median_sigma = float(fit_model_open_sigma[median_local_idx]) if np.isfinite(fit_model_open_sigma[median_local_idx]) else float("nan")
        log_width_fit_median_sigma = float(fit_model_log_sigma[median_local_idx]) if np.isfinite(fit_model_log_sigma[median_local_idx]) else float("nan")
    else:
        opening_fit_median = float("nan")
        opening_fit_mean = float("nan")
        opening_fit_median_sigma = float("nan")
        log_width_fit_median_sigma = float("nan")
    finite_x = x[valid]
    if finite_x.size > 0:
        x_center = float(np.exp(np.mean(np.log(finite_x))))
        center_width = (10.0 ** float(intercept)) * (x_center ** float(slope))
        opening_fit_center = opening_angle_deg(center_width, x_center)
        center_log_sigma_arr = _linear_model_sigma(np.asarray([math.log10(x_center)], dtype=np.float64), cov)
        center_log_sigma = float(center_log_sigma_arr[0]) if center_log_sigma_arr.size else float("nan")
        center_width_sigma = float(math.log(10.0) * center_width * center_log_sigma) if np.isfinite(center_log_sigma) else float("nan")
        opening_fit_center_sigma = opening_angle_error_deg(center_width, x_center, center_width_sigma)
    else:
        x_center = float("nan")
        opening_fit_center = float("nan")
        opening_fit_center_sigma = float("nan")
    if finite_resid.size > 0:
        sigma_theta_scatter = _robust_sigma(finite_resid)
    else:
        sigma_theta_scatter = float("nan")
    if finite_log_resid.size > 0:
        sigma_log_width_scatter = _robust_sigma(finite_log_resid)
    else:
        sigma_log_width_scatter = float("nan")
    theta_obs_sigma = theta_sigma[valid]
    opening_data_median_sigma = _median_propagated_sigma(theta_obs_sigma)
    sigma_theta = _combine_independent_sigmas(sigma_theta_scatter, opening_fit_median_sigma, opening_data_median_sigma)
    sigma_log_width = _combine_independent_sigmas(sigma_log_width_scatter, log_width_fit_median_sigma)
    finite_all_log_resid = all_log_residual[np.isfinite(all_log_residual)]
    if finite_all_log_resid.size > 0:
        all_med_log = float(np.median(finite_all_log_resid))
        all_sigma_log_width = float(1.4826 * np.median(np.abs(finite_all_log_resid - all_med_log)))
        all_log_abs_mean = float(np.mean(np.abs(finite_all_log_resid)))
        all_log_mean = float(np.mean(finite_all_log_resid))
        all_log_rms = float(np.sqrt(np.mean(np.square(finite_all_log_resid))))
    else:
        all_sigma_log_width = float("nan")
        all_log_abs_mean = float("nan")
        all_log_mean = float("nan")
        all_log_rms = float("nan")
    fit_log_abs_mean = float(np.mean(np.abs(finite_log_resid))) if finite_log_resid.size > 0 else float("nan")
    fit_log_mean = float(np.mean(finite_log_resid)) if finite_log_resid.size > 0 else float("nan")
    fit_log_rms = float(np.sqrt(np.mean(np.square(finite_log_resid)))) if finite_log_resid.size > 0 else float("nan")
    fit_open_abs_mean = float(np.mean(np.abs(finite_resid))) if finite_resid.size > 0 else float("nan")
    fit_open_mean = float(np.mean(finite_resid)) if finite_resid.size > 0 else float("nan")
    fit_open_rms = float(np.sqrt(np.mean(np.square(finite_resid)))) if finite_resid.size > 0 else float("nan")
    return {
        "distance_unit": str(distance_unit),
        "fit_mode": "k_near_one",
        "k_fit": float(slope),
        "k_sigma": float(k_sigma),
        "intercept_log10": float(intercept),
        "intercept_log10_sigma": float(intercept_sigma),
        "fit_start_slice_order": int(start_slice_order),
        "fit_end_slice_order": int(end_slice_order),
        "fit_point_count": int(fit_info.get("point_count", np.count_nonzero(valid))),
        "fit_valid_point_count": int(np.count_nonzero(valid)),
        "fit_reduced_chi2": float(fit_info.get("reduced_chi2", float("nan"))),
        "x_all": all_x.astype(np.float32),
        "width_model": model_width.astype(np.float32),
        "width_model_sigma": model_width_sigma.astype(np.float32),
        "log_width_model_sigma_dex": model_log_width_sigma.astype(np.float32),
        "log_width_residual_all": all_log_residual.astype(np.float32),
        "log_width_sigma_dex": float(sigma_log_width),
        "log_width_residual_scatter_dex": float(sigma_log_width_scatter),
        "log_width_fit_median_sigma_dex": float(log_width_fit_median_sigma),
        "fit_log_width_residual_mean_dex": float(fit_log_mean),
        "fit_log_width_residual_mean_abs_dex": float(fit_log_abs_mean),
        "fit_log_width_residual_rms_dex": float(fit_log_rms),
        "all_log_width_sigma_dex": float(all_sigma_log_width),
        "all_log_width_residual_mean_dex": float(all_log_mean),
        "all_log_width_residual_mean_abs_dex": float(all_log_abs_mean),
        "all_log_width_residual_rms_dex": float(all_log_rms),
        "used_weighted_fit": bool(fit_info.get("used_weighted", False)),
        "opening_model_deg": model_open.astype(np.float32),
        "opening_model_sigma_deg": model_open_sigma.astype(np.float32),
        "opening_residual_all": opening_residual_all.astype(np.float32),
        "opening_sigma_deg": float(sigma_theta),
        "opening_residual_scatter_deg": float(sigma_theta_scatter),
        "opening_data_median_sigma_deg": float(opening_data_median_sigma),
        "fit_opening_residual_mean_deg": float(fit_open_mean),
        "fit_opening_residual_mean_abs_deg": float(fit_open_abs_mean),
        "fit_opening_residual_rms_deg": float(fit_open_rms),
        "opening_fit_median_deg": float(opening_fit_median),
        "opening_fit_median_sigma_deg": float(opening_fit_median_sigma),
        "half_opening_fit_median_deg": float(0.5 * opening_fit_median) if np.isfinite(opening_fit_median) else float("nan"),
        "half_opening_fit_median_sigma_deg": float(0.5 * opening_fit_median_sigma) if np.isfinite(opening_fit_median_sigma) else float("nan"),
        "opening_fit_mean_deg": float(opening_fit_mean),
        "opening_fit_center_deg": float(opening_fit_center),
        "opening_fit_center_sigma_deg": float(opening_fit_center_sigma),
        "opening_fit_center_distance": float(x_center),
    }


def find_opening_angle_plateau(
    rows: Sequence[Dict[str, object]],
    *,
    distance_unit: str = "auto",
    min_points: int = 8,
    min_log_span: float = 0.25,
    max_abs_theta_slope_deg_per_dex: float = 1.0,
    k_tolerance: float = 0.30,
) -> Dict[str, object]:
    items = list(rows)
    if not items:
        raise ValueError("No report rows available for plateau search.")
    distance_unit = str(distance_unit).strip().lower()
    if distance_unit == "auto":
        any_mas = any(np.isfinite(_safe_float(row.get("distance_from_core_mas", float("nan")))) for row in items)
        distance_unit = "mas" if any_mas else "px"
    x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
    y_key = "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"
    y_sigma_key = "gaussian_width_sigma_mas" if distance_unit == "mas" else "gaussian_width_sigma_px"
    x_all = np.asarray([_safe_float(row.get(x_key, float("nan"))) for row in items], dtype=np.float64)
    width_all = np.asarray([_safe_float(row.get(y_key, float("nan"))) for row in items], dtype=np.float64)
    width_sigma_all = np.asarray([_safe_float(row.get(y_sigma_key, float("nan"))) for row in items], dtype=np.float64)
    theta_all = np.asarray([_safe_float(row.get("gaussian_angle_deg", float("nan"))) for row in items], dtype=np.float64)
    theta_sigma_all = np.asarray([_safe_float(row.get("gaussian_angle_sigma_deg", float("nan"))) for row in items], dtype=np.float64)
    n = int(min(len(items), x_all.size, width_all.size, width_sigma_all.size, theta_all.size, theta_sigma_all.size))
    min_points = int(max(3, min_points))
    min_log_span = float(max(0.0, min_log_span))
    max_abs_theta_slope_deg_per_dex = float(max(1e-9, max_abs_theta_slope_deg_per_dex))
    k_tolerance = float(max(1e-9, k_tolerance))
    best = None
    for i in range(n):
        idx = np.arange(i, n, dtype=np.int64)
        xs = x_all[idx]
        widths = width_all[idx]
        theta = theta_all[idx]
        valid = np.isfinite(xs) & np.isfinite(widths) & np.isfinite(theta) & (xs > 0.0) & (widths > 0.0) & (theta > 0.0)
        if int(np.count_nonzero(valid)) < min_points:
            continue
        valid_idx = idx[valid]
        xv = xs[valid]
        wv = widths[valid]
        tv = theta[valid]
        width_sig_v = width_sigma_all[idx][valid]
        theta_sig_v = theta_sigma_all[idx][valid]
        lx = np.log10(xv)
        log_span = float(np.max(lx) - np.min(lx)) if lx.size >= 2 else 0.0
        if log_span < min_log_span or np.allclose(lx, lx[0]):
            continue
        theta_level = float(np.median(tv))
        theta_mean = float(np.mean(tv))
        theta_resid = tv - theta_level
        scatter = _robust_sigma(theta_resid)
        if not np.isfinite(scatter):
            scatter = 0.0 if theta_resid.size == 1 else float("nan")
        theta_level_sigma = _median_propagated_sigma(theta_sig_v)
        theta_total_sigma = _combine_independent_sigmas(scatter, theta_level_sigma)
        theta_fit = _linear_fit_with_cov(lx, tv, theta_sig_v, min_points=3)
        if theta_fit is None:
            continue
        theta_slope = float(theta_fit["slope"])
        theta_intercept = float(theta_fit["intercept"])
        theta_cov = np.asarray(theta_fit.get("cov", np.full((2, 2), np.nan)), dtype=np.float64)
        theta_slope_sigma = float(math.sqrt(max(0.0, float(theta_cov[0, 0])))) if theta_cov.shape == (2, 2) and np.isfinite(theta_cov[0, 0]) else float("nan")
        sigma_log_w = np.full(wv.shape, np.nan, dtype=np.float64)
        finite_wsig = np.isfinite(width_sig_v) & (width_sig_v > 0.0) & np.isfinite(wv) & (wv > 0.0)
        sigma_log_w[finite_wsig] = width_sig_v[finite_wsig] / (wv[finite_wsig] * np.log(10.0))
        width_fit = _linear_fit_with_cov(lx, np.log10(wv), sigma_log_w, min_points=3)
        if width_fit is not None:
            k_tail = float(width_fit["slope"])
            log_width_intercept = float(width_fit["intercept"])
            width_cov = np.asarray(width_fit.get("cov", np.full((2, 2), np.nan)), dtype=np.float64)
            k_tail_sigma = float(math.sqrt(max(0.0, float(width_cov[0, 0])))) if width_cov.shape == (2, 2) and np.isfinite(width_cov[0, 0]) else float("nan")
            log_width_intercept_sigma = float(math.sqrt(max(0.0, float(width_cov[1, 1])))) if width_cov.shape == (2, 2) and np.isfinite(width_cov[1, 1]) else float("nan")
        else:
            k_tail = float("nan")
            log_width_intercept = float("nan")
            k_tail_sigma = float("nan")
            log_width_intercept_sigma = float("nan")
        used_weighted_fit = bool(theta_fit.get("used_weighted", False)) or bool(width_fit.get("used_weighted", False) if width_fit is not None else False)
        theta_slope = float(theta_slope)
        theta_intercept = float(theta_intercept)
        k_tail = float(k_tail)
        log_width_intercept = float(log_width_intercept)
        slope_penalty = abs(theta_slope) / max_abs_theta_slope_deg_per_dex
        k_penalty = abs(k_tail - 1.0) / k_tolerance if np.isfinite(k_tail) else 1e6
        scatter_penalty = theta_total_sigma / max(1.0, abs(theta_level))
        pass_count = int(abs(theta_slope) > max_abs_theta_slope_deg_per_dex) + int(
            np.isfinite(k_tail) and abs(k_tail - 1.0) > k_tolerance
        )
        score = float(slope_penalty + (0.50 * k_penalty) + (0.25 * scatter_penalty) - (0.05 * log_span))
        candidate_meta = {
            "valid_idx": valid_idx,
            "theta_level": theta_level,
            "theta_mean": theta_mean,
            "theta_slope": theta_slope,
            "theta_intercept": theta_intercept,
            "theta_scatter": scatter,
            "theta_level_sigma": theta_level_sigma,
            "theta_total_sigma": theta_total_sigma,
            "theta_slope_sigma": theta_slope_sigma,
            "k_tail": k_tail,
            "k_tail_sigma": k_tail_sigma,
            "log_width_intercept": log_width_intercept,
            "log_width_intercept_sigma": log_width_intercept_sigma,
            "log_span": log_span,
            "score": score,
            "used_weighted_fit": used_weighted_fit,
        }
        candidate_key = (
            int(pass_count),
            float(score),
            -int(valid_idx.size),
            int(i),
        )
        if best is None or candidate_key < best[0]:
            best = (candidate_key, candidate_meta)
    if best is None:
        raise ValueError("Failed to find a valid opening-angle plateau.")
    (_pass_count, _score, _neg_n, _i_best), meta = best
    valid_idx = np.asarray(meta["valid_idx"], dtype=np.int64)
    theta_level = float(meta["theta_level"])
    theta_level_sigma = float(meta.get("theta_level_sigma", float("nan")))
    theta_total_sigma = float(meta.get("theta_total_sigma", float("nan")))
    theta_rad = np.radians(theta_level)
    model_width = np.full(x_all.shape, np.nan, dtype=np.float64)
    positive_x = np.isfinite(x_all) & (x_all > 0.0) & np.isfinite(theta_rad) & (theta_rad > 0.0)
    model_width[positive_x] = 2.0 * x_all[positive_x] * np.tan(0.5 * theta_rad)
    width_model_sigma = np.full(x_all.shape, np.nan, dtype=np.float64)
    log_width_model_sigma = np.full(x_all.shape, np.nan, dtype=np.float64)
    if np.isfinite(theta_level_sigma) and theta_level_sigma >= 0.0 and np.isfinite(theta_rad) and theta_rad > 0.0:
        sec2 = 1.0 / (math.cos(0.5 * theta_rad) ** 2)
        width_model_sigma[positive_x] = np.abs(x_all[positive_x] * sec2 * (math.pi / 180.0) * theta_level_sigma)
        finite_model_sigma = positive_x & np.isfinite(width_model_sigma) & np.isfinite(model_width) & (model_width > 0.0)
        log_width_model_sigma[finite_model_sigma] = width_model_sigma[finite_model_sigma] / (model_width[finite_model_sigma] * math.log(10.0))
    opening_model = np.full(x_all.shape, np.nan, dtype=np.float64)
    opening_model[positive_x] = theta_level
    opening_model_sigma = np.full(x_all.shape, np.nan, dtype=np.float64)
    opening_model_sigma[positive_x] = theta_level_sigma
    opening_residual = np.full(x_all.shape, np.nan, dtype=np.float64)
    resid_valid = np.isfinite(theta_all) & np.isfinite(opening_model) & (theta_all > 0.0)
    opening_residual[resid_valid] = theta_all[resid_valid] - opening_model[resid_valid]
    log_width_residual = np.full(x_all.shape, np.nan, dtype=np.float64)
    width_resid_valid = np.isfinite(width_all) & np.isfinite(model_width) & (width_all > 0.0) & (model_width > 0.0)
    log_width_residual[width_resid_valid] = np.log10(width_all[width_resid_valid]) - np.log10(model_width[width_resid_valid])
    log_width_residual_scatter = (
        _robust_sigma(log_width_residual[valid_idx])
        if np.any(np.isfinite(log_width_residual[valid_idx]))
        else float("nan")
    )
    log_width_fit_median_sigma = _median_propagated_sigma(log_width_model_sigma[valid_idx])
    log_width_sigma = _combine_independent_sigmas(log_width_residual_scatter, log_width_fit_median_sigma)
    fit_x = x_all[valid_idx]
    start_idx = int(valid_idx[0])
    end_idx = int(valid_idx[-1])
    x_center = float(np.exp(np.mean(np.log(fit_x)))) if fit_x.size > 0 else float("nan")
    return {
        "fit_mode": "opening_plateau",
        "distance_unit": str(distance_unit),
        "k_fit": float(meta["k_tail"]),
        "k_sigma": float(meta.get("k_tail_sigma", float("nan"))),
        "intercept_log10": float(meta["log_width_intercept"]),
        "intercept_log10_sigma": float(meta.get("log_width_intercept_sigma", float("nan"))),
        "x_all": x_all.astype(np.float32),
        "width_model": model_width.astype(np.float32),
        "width_model_sigma": width_model_sigma.astype(np.float32),
        "log_width_model_sigma_dex": log_width_model_sigma.astype(np.float32),
        "log_width_residual_all": log_width_residual.astype(np.float32),
        "log_width_sigma_dex": float(log_width_sigma),
        "log_width_residual_scatter_dex": float(log_width_residual_scatter),
        "log_width_fit_median_sigma_dex": float(log_width_fit_median_sigma),
        "used_weighted_fit": bool(meta.get("used_weighted_fit", False)),
        "opening_model_deg": opening_model.astype(np.float32),
        "opening_model_sigma_deg": opening_model_sigma.astype(np.float32),
        "opening_residual_all": opening_residual.astype(np.float32),
        "opening_sigma_deg": float(theta_total_sigma),
        "opening_residual_scatter_deg": float(meta["theta_scatter"]),
        "opening_data_median_sigma_deg": float(theta_level_sigma),
        "opening_fit_median_deg": float(theta_level),
        "opening_fit_median_sigma_deg": float(theta_level_sigma),
        "half_opening_fit_median_deg": float(0.5 * theta_level),
        "half_opening_fit_median_sigma_deg": float(0.5 * theta_level_sigma) if np.isfinite(theta_level_sigma) else float("nan"),
        "opening_fit_mean_deg": float(meta["theta_mean"]),
        "opening_fit_center_deg": float(theta_level),
        "opening_fit_center_sigma_deg": float(theta_level_sigma),
        "opening_fit_center_distance": float(x_center),
        "theta_slope_deg_per_dex": float(meta["theta_slope"]),
        "theta_slope_sigma_deg_per_dex": float(meta.get("theta_slope_sigma", float("nan"))),
        "theta_intercept_deg": float(meta["theta_intercept"]),
        "plateau_score": float(meta["score"]),
        "plateau_passes_thresholds": bool(_pass_count == 0),
        "plateau_point_count": int(valid_idx.size),
        "plateau_log_span": float(meta["log_span"]),
        "plateau_start_slice_order": int(items[start_idx].get("slice_order", start_idx + 1)),
        "plateau_end_slice_order": int(items[end_idx].get("slice_order", end_idx + 1)),
        "plateau_start_distance": float(x_all[start_idx]),
        "plateau_end_distance": float(x_all[end_idx]),
    }


def auto_select_k_near_one_range(
    rows: Sequence[Dict[str, object]],
    local_k: Sequence[float],
    *,
    distance_unit: str = "auto",
    min_points: int = 5,
) -> Dict[str, object]:
    items = list(rows)
    if not items:
        raise ValueError("No report rows available.")
    distance_unit = str(distance_unit).strip().lower()
    if distance_unit == "auto":
        any_mas = any(np.isfinite(_safe_float(row.get("distance_from_core_mas", float("nan")))) for row in items)
        distance_unit = "mas" if any_mas else "px"
    x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
    y_key = "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"
    y_sigma_key = "gaussian_width_sigma_mas" if distance_unit == "mas" else "gaussian_width_sigma_px"
    x = np.asarray([_safe_float(row.get(x_key, float("nan"))) for row in items], dtype=np.float64)
    y = np.asarray([_safe_float(row.get(y_key, float("nan"))) for row in items], dtype=np.float64)
    y_sigma = np.asarray([_safe_float(row.get(y_sigma_key, float("nan"))) for row in items], dtype=np.float64)
    k_arr = np.asarray(local_k, dtype=np.float64)
    n = int(min(len(items), x.size, y.size, y_sigma.size, k_arr.size))
    min_points = int(max(3, min_points))
    if n <= 0:
        raise ValueError("Failed to find a valid k-fit range.")
    x = x[:n]
    y = y[:n]
    y_sigma = y_sigma[:n]
    k_arr = k_arr[:n]
    valid_all = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(valid_all)) < min_points:
        raise ValueError("Failed to find a valid k-fit range.")
    lx_all = np.zeros(n, dtype=np.float64)
    ly_all = np.zeros(n, dtype=np.float64)
    lx_all[valid_all] = np.log10(x[valid_all])
    ly_all[valid_all] = np.log10(y[valid_all])
    sigma_log_y = np.full(n, np.nan, dtype=np.float64)
    finite_sigma = valid_all & np.isfinite(y_sigma) & (y_sigma > 0.0)
    sigma_log_y[finite_sigma] = y_sigma[finite_sigma] / (y[finite_sigma] * np.log(10.0))
    finite_weight = finite_sigma & np.isfinite(sigma_log_y) & (sigma_log_y > 0.0)
    poly_weight = np.zeros(n, dtype=np.float64)
    poly_weight[finite_weight] = 1.0 / sigma_log_y[finite_weight]
    poly_weight_sq = np.square(poly_weight)
    w_valid = valid_all.astype(np.float64)

    def _prefix(values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(values.astype(np.float64))])

    p_count = _prefix(w_valid)
    p_x = _prefix(np.where(valid_all, lx_all, 0.0))
    p_y = _prefix(np.where(valid_all, ly_all, 0.0))
    p_xx = _prefix(np.where(valid_all, lx_all * lx_all, 0.0))
    p_xy = _prefix(np.where(valid_all, lx_all * ly_all, 0.0))
    p_w_count = _prefix(finite_weight.astype(np.float64))
    p_w = _prefix(poly_weight_sq)
    p_wx = _prefix(poly_weight_sq * lx_all)
    p_wy = _prefix(poly_weight_sq * ly_all)
    p_wxx = _prefix(poly_weight_sq * lx_all * lx_all)
    p_wxy = _prefix(poly_weight_sq * lx_all * ly_all)
    finite_k = np.isfinite(k_arr)

    def _range_sum(prefix: np.ndarray, i0: int, i1: int) -> float:
        return float(prefix[int(i1) + 1] - prefix[int(i0)])

    def _linear_fit_from_sums(sw: float, sx: float, sy: float, sxx: float, sxy: float) -> Optional[Tuple[float, float]]:
        denom = (float(sw) * float(sxx)) - (float(sx) * float(sx))
        if (not np.isfinite(denom)) or abs(denom) <= 1e-12:
            return None
        slope = ((float(sw) * float(sxy)) - (float(sx) * float(sy))) / denom
        intercept = (float(sy) - (float(slope) * float(sx))) / float(sw)
        if (not np.isfinite(slope)) or (not np.isfinite(intercept)):
            return None
        return float(slope), float(intercept)

    best = None
    for i in range(n):
        for j in range(i + min_points - 1, n):
            count = int(round(_range_sum(p_count, i, j)))
            if count < min_points:
                continue
            weight_count = int(round(_range_sum(p_w_count, i, j)))
            if weight_count >= 3:
                fit_parts = _linear_fit_from_sums(
                    _range_sum(p_w, i, j),
                    _range_sum(p_wx, i, j),
                    _range_sum(p_wy, i, j),
                    _range_sum(p_wxx, i, j),
                    _range_sum(p_wxy, i, j),
                )
            else:
                fit_parts = _linear_fit_from_sums(
                    float(count),
                    _range_sum(p_x, i, j),
                    _range_sum(p_y, i, j),
                    _range_sum(p_xx, i, j),
                    _range_sum(p_xy, i, j),
                )
            if fit_parts is None:
                continue
            slope, intercept = fit_parts
            fit_k_score = abs(float(slope) - 1.0)
            finite_ks = k_arr[i:j + 1][finite_k[i:j + 1]]
            local_k_score = float(np.median(np.abs(finite_ks - 1.0))) if finite_ks.size > 0 else fit_k_score
            valid_range = valid_all[i:j + 1]
            lx = lx_all[i:j + 1][valid_range]
            ly = ly_all[i:j + 1][valid_range]
            range_resid = ly - (float(intercept) + (float(slope) * lx))
            range_resid_mad = float(np.median(np.abs(range_resid - np.median(range_resid)))) if range_resid.size > 0 else float("inf")
            all_resid = ly_all[valid_all] - (float(intercept) + (float(slope) * lx_all[valid_all]))
            all_resid_mad = float(np.median(np.abs(all_resid - np.median(all_resid)))) if all_resid.size > 0 else float("inf")
            # Prioritize ranges whose actual fitted power-law slope is close to 1.
            # The residual term is evaluated over all usable points so short
            # candidate ranges are not rewarded just because they contain fewer points.
            fit_k_score_norm = fit_k_score / 0.10
            local_k_score_norm = local_k_score / 0.15
            all_resid_mad_score_norm = all_resid_mad / 0.03
            score = fit_k_score_norm + local_k_score_norm + all_resid_mad_score_norm
            candidate = (
                float(score),
                float(fit_k_score_norm),
                float(local_k_score_norm),
                float(all_resid_mad_score_norm),
                float(range_resid_mad),
                -int(count),
                int(i),
                int(j),
                float(fit_k_score),
                float(local_k_score),
                float(all_resid_mad),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("Failed to find a valid k-fit range.")
    (
        _score,
        _fit_k_score_norm,
        _local_k_score_norm,
        _all_resid_mad_score_norm,
        _range_resid_mad,
        _neg_len,
        i_best,
        j_best,
        _fit_k_score,
        _local_k_score,
        _all_resid_mad,
    ) = best
    fit_best = fit_power_law_from_rows(
        items,
        start_slice_order=int(items[i_best].get("slice_order", i_best + 1)),
        end_slice_order=int(items[j_best].get("slice_order", j_best + 1)),
        distance_unit=distance_unit,
    )
    return {
        "start_slice_order": int(items[i_best].get("slice_order", i_best + 1)),
        "end_slice_order": int(items[j_best].get("slice_order", j_best + 1)),
        "selection_score": float(_score),
        "fit_k_score": float(_fit_k_score),
        "local_k_score": float(_local_k_score),
        "all_residual_mad_dex": float(_all_resid_mad),
        "range_residual_mad_dex": float(_range_resid_mad),
        "range_residual_rms_proxy_dex": float(_range_resid_mad),
        "fit_k_score_norm": float(_fit_k_score_norm),
        "local_k_score_norm": float(_local_k_score_norm),
        "all_residual_mad_score_norm": float(_all_resid_mad_score_norm),
        "range_residual_mad_score_norm": float(_all_resid_mad_score_norm),
        "fit": fit_best,
    }


__all__ = [
    "auto_select_k_near_one_range",
    "build_gaussian_report_rows",
    "compute_local_k",
    "find_opening_angle_plateau",
    "fit_power_law_from_rows",
    "paper_fig7_eastern_broken_power_law",
]
