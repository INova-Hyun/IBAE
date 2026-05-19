#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit

try:
    from .common.numeric import robust_sigma as _robust_sigma
    from .common.numeric import safe_float as _safe_float
    from .ridgeline import fit_transverse_gaussian, opening_angle_deg
    from .session_analysis import load_analysis_session
except Exception:
    import sys

    _HERE = Path(__file__).resolve().parent
    _PARENT = _HERE.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))
    from IBAE.common.numeric import robust_sigma as _robust_sigma
    from IBAE.common.numeric import safe_float as _safe_float
    from IBAE.ridgeline import fit_transverse_gaussian, opening_angle_deg
    from IBAE.session_analysis import load_analysis_session


def _beam_deconvolved_width(width_px: float, beam_px: float) -> float:
    width_px = float(width_px)
    beam_px = float(beam_px)
    if not np.isfinite(width_px) or width_px <= 0.0:
        return float("nan")
    if not np.isfinite(beam_px) or beam_px <= 0.0:
        return float("nan")
    return float(math.sqrt(max(0.0, (width_px * width_px) - (beam_px * beam_px))))


def _select_fit_window(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    bdrop: int,
    edrop: int,
    current_fit_window: Dict[str, Any],
    beam_px: float,
    center_half_beam: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    order = np.argsort(x)
    x = np.asarray(x[order], dtype=np.float64)
    y = np.asarray(y[order], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    meta: Dict[str, Any] = {
        "mode": str(mode),
        "input_n": int(x.size),
        "bdrop": int(max(0, bdrop)),
        "edrop": int(max(0, edrop)),
    }
    if x.size <= 0:
        return x, y, meta

    mode = str(mode).strip().lower()
    if mode == "same-as-current":
        left = _safe_float(current_fit_window.get("left_x", float("nan")))
        right = _safe_float(current_fit_window.get("right_x", float("nan")))
        if np.isfinite(left) and np.isfinite(right) and right > left:
            keep = (x >= left) & (x <= right)
            meta.update({"left_x": float(left), "right_x": float(right)})
            return x[keep], y[keep], meta
        mode = "full-support"

    if mode == "center-beam":
        if np.isfinite(beam_px) and beam_px > 0.0:
            half_width = float(max(0.5, center_half_beam * beam_px))
        else:
            half_width = float(max(0.5, 0.5 * np.ptp(x)))
        keep = np.abs(x) <= half_width
        meta.update({"left_x": -half_width, "right_x": half_width, "center_half_beam": float(center_half_beam)})
        return x[keep], y[keep], meta

    start = int(max(0, bdrop))
    end = int(max(start, x.size - int(max(0, edrop))))
    x_sel = x[start:end]
    y_sel = y[start:end]
    if x_sel.size:
        meta.update({"left_x": float(x_sel[0]), "right_x": float(x_sel[-1])})
    return x_sel, y_sel, meta


def _aips_like_fit(
    x: Sequence[float],
    y: Sequence[float],
    *,
    baseline_order: int,
    mu_bound_px: float,
) -> Dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    if x_arr.size < 5:
        return {"success": False, "reason": "too_few_points", "x": x_arr, "y": y_arr}
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    baseline_order = int(np.clip(int(baseline_order), -1, 2))
    edge_floor = float(np.nanmin([y_arr[0], y_arr[-1]]))
    peak_idx = int(np.nanargmax(y_arr))
    peak_val = float(y_arr[peak_idx])
    amp0 = float(max(peak_val - edge_floor, 1e-6))
    sigma0 = float(max(np.ptp(x_arr) / 6.0, 0.5))
    if np.isfinite(mu_bound_px) and mu_bound_px > 0.0:
        mu_bound = float(mu_bound_px)
    else:
        mu_bound = float(max(0.5, 0.5 * np.ptp(x_arr)))
    mu0 = float(np.clip(float(x_arr[peak_idx]), -mu_bound, mu_bound))
    x_ref = 0.0

    if baseline_order < 0:
        p0 = np.asarray([amp0, mu0, sigma0], dtype=np.float64)
        lower = np.asarray([0.0, -mu_bound, 0.1], dtype=np.float64)
        upper = np.asarray([max(1.0, float(np.nanmax(y_arr) * 4.0)), mu_bound, max(1.0, float(np.ptp(x_arr)))], dtype=np.float64)

        def model(s, amp, mu, sigma):
            sigma = np.maximum(sigma, 1e-6)
            return amp * np.exp(-0.5 * ((s - mu) / sigma) ** 2)

        sigma_param_idx = 2
        mu_param_idx = 1
        baseline = float("nan")
    else:
        n_poly = int(baseline_order) + 1
        p0 = np.asarray([edge_floor] + ([0.0] * (n_poly - 1)) + [amp0, mu0, sigma0], dtype=np.float64)
        lower = np.asarray(([-np.inf] * n_poly) + [0.0, -mu_bound, 0.1], dtype=np.float64)
        upper = np.asarray(( [np.inf] * n_poly) + [max(1.0, float(np.nanmax(y_arr) * 4.0)), mu_bound, max(1.0, float(np.ptp(x_arr)))], dtype=np.float64)

        def model(s, *params):
            coeff = np.asarray(params[:n_poly], dtype=np.float64)
            amp = float(params[n_poly])
            mu = float(params[n_poly + 1])
            sigma = np.maximum(float(params[n_poly + 2]), 1e-6)
            sx = np.asarray(s, dtype=np.float64) - x_ref
            base = np.zeros_like(sx, dtype=np.float64)
            for power, val in enumerate(coeff.tolist()):
                base += float(val) * np.power(sx, power)
            return base + amp * np.exp(-0.5 * ((sx + x_ref - mu) / sigma) ** 2)

        sigma_param_idx = n_poly + 2
        mu_param_idx = n_poly + 1
        baseline = float(p0[0])

    try:
        popt, pcov = curve_fit(model, x_arr, y_arr, p0=p0, bounds=(lower, upper), maxfev=20000)
        fit_y = model(x_arr, *popt)
        residual = y_arr - fit_y
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        sigma = float(popt[sigma_param_idx])
        mu = float(popt[mu_param_idx])
        amplitude = float(popt[mu_param_idx - 1])
        if baseline_order >= 0:
            baseline = float(popt[0])
        sigma_sigma = float(math.sqrt(max(0.0, float(pcov[sigma_param_idx, sigma_param_idx]))))
        mu_sigma = float(math.sqrt(max(0.0, float(pcov[mu_param_idx, mu_param_idx]))))
        fwhm = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
        fwhm_sigma = float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_sigma)
        return {
            "success": True,
            "x": x_arr.astype(np.float32),
            "y": y_arr.astype(np.float32),
            "fit_y": np.asarray(fit_y, dtype=np.float32),
            "params": {
                "baseline": baseline,
                "amplitude": amplitude,
                "mu": mu,
                "sigma": sigma,
                "mu_bound_px": float(mu_bound),
                "baseline_order": int(baseline_order),
            },
            "param_errors": {"mu": float(mu_sigma), "sigma": float(sigma_sigma)},
            "fwhm_px": float(fwhm),
            "fwhm_sigma_px": float(fwhm_sigma),
            "rmse": float(rmse),
        }
    except Exception as exc:
        return {
            "success": False,
            "reason": "fit_failed",
            "error": str(exc),
            "x": x_arr.astype(np.float32),
            "y": y_arr.astype(np.float32),
        }


def _row_width_angle(width_px: float, beam_px: float, distance_px: float, width_mode: str) -> Tuple[float, str, float]:
    width_mode = str(width_mode).strip().lower()
    intrinsic = _beam_deconvolved_width(width_px, beam_px)
    if width_mode == "raw":
        used = float(width_px)
        source = "raw"
    elif width_mode == "intrinsic-or-raw":
        if np.isfinite(intrinsic) and intrinsic > 0.0:
            used = float(intrinsic)
            source = "intrinsic"
        else:
            used = float(width_px)
            source = "raw"
    else:
        used = float(intrinsic) if np.isfinite(intrinsic) and intrinsic > 0.0 else float("nan")
        source = "intrinsic" if np.isfinite(used) else "excluded"
    return used, source, opening_angle_deg(used, distance_px)


def _load_records(path: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    payload = load_analysis_session(path)
    measurement = payload.get("measurement_result", {})
    if not isinstance(measurement, dict):
        raise ValueError("Analysis JSON does not contain measurement_result.")
    records = [dict(item) for item in list(measurement.get("fit_records", []) or []) if isinstance(item, dict)]
    if not records:
        raise ValueError("Analysis JSON does not contain fit_records.")
    return payload, measurement, records


def compare_modes(
    analysis_json: str,
    *,
    window_mode: str,
    bdrop: int,
    edrop: int,
    baseline_order: int,
    width_mode: str,
    center_half_beam: float,
) -> List[Dict[str, Any]]:
    _payload, measurement, records = _load_records(analysis_json)
    beam_px = _safe_float(measurement.get("beam_size_px", float("nan")))
    mu_bound_px = _safe_float(measurement.get("gaussian_mu_bound_px", float("nan")))
    if not np.isfinite(mu_bound_px) and np.isfinite(beam_px) and beam_px > 0.0:
        mu_bound_px = 0.5 * beam_px

    rows: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        profile = record.get("profile", {})
        if not isinstance(profile, dict):
            continue
        x = np.asarray(profile.get("valid_x", []), dtype=np.float64)
        y = np.asarray(profile.get("valid_y", []), dtype=np.float64)
        if x.size < 5 or y.size < 5:
            continue
        current = fit_transverse_gaussian(profile, mu_bound_px=mu_bound_px)
        fit_window = dict(dict(record.get("fit", {}) or {}).get("fit_window", {}) or {})
        x_sel, y_sel, window_meta = _select_fit_window(
            x,
            y,
            mode=window_mode,
            bdrop=int(bdrop),
            edrop=int(edrop),
            current_fit_window=fit_window,
            beam_px=beam_px,
            center_half_beam=float(center_half_beam),
        )
        aips = _aips_like_fit(x_sel, y_sel, baseline_order=int(baseline_order), mu_bound_px=mu_bound_px)
        distance_px = _safe_float(record.get("distance_from_core_px", record.get("distance_along_ridge_px", float("nan"))))
        current_width = _safe_float(current.get("fwhm_px", float("nan"))) if current.get("success") else float("nan")
        aips_width = _safe_float(aips.get("fwhm_px", float("nan"))) if aips.get("success") else float("nan")
        current_used_width, current_source, current_angle = _row_width_angle(current_width, beam_px, distance_px, width_mode)
        aips_used_width, aips_source, aips_angle = _row_width_angle(aips_width, beam_px, distance_px, width_mode)
        cur_params = dict(current.get("params", {}) or {})
        aips_params = dict(aips.get("params", {}) or {})
        rows.append(
            {
                "record_index": int(record_index),
                "ridge_idx": int(record.get("ridge_idx", profile.get("ridge_idx", -1))),
                "distance_px": float(distance_px),
                "window_mode": str(window_meta.get("mode", window_mode)),
                "window_n": int(x_sel.size),
                "window_left_px": _safe_float(window_meta.get("left_x", float("nan"))),
                "window_right_px": _safe_float(window_meta.get("right_x", float("nan"))),
                "baseline_order": int(baseline_order),
                "current_success": bool(current.get("success", False)),
                "aips_success": bool(aips.get("success", False)),
                "current_fwhm_px": float(current_width),
                "aips_fwhm_px": float(aips_width),
                "delta_fwhm_px": float(aips_width - current_width) if np.isfinite(current_width) and np.isfinite(aips_width) else float("nan"),
                "delta_fwhm_frac": float((aips_width - current_width) / current_width) if np.isfinite(current_width) and current_width > 0.0 and np.isfinite(aips_width) else float("nan"),
                "current_mu_px": _safe_float(cur_params.get("mu", float("nan"))),
                "aips_mu_px": _safe_float(aips_params.get("mu", float("nan"))),
                "current_rmse": _safe_float(current.get("rmse", float("nan"))),
                "aips_rmse": _safe_float(aips.get("rmse", float("nan"))),
                "width_mode": str(width_mode),
                "current_used_width_px": float(current_used_width),
                "aips_used_width_px": float(aips_used_width),
                "current_width_source": str(current_source),
                "aips_width_source": str(aips_source),
                "current_opening_angle_deg": float(current_angle),
                "aips_opening_angle_deg": float(aips_angle),
                "delta_opening_angle_deg": float(aips_angle - current_angle) if np.isfinite(current_angle) and np.isfinite(aips_angle) else float("nan"),
                "aips_failure": "" if aips.get("success") else str(aips.get("reason", "fit_failed")),
                "current_failure": "" if current.get("success") else str(current.get("reason", "fit_failed")),
            }
        )
    return rows


def _write_csv(rows: Sequence[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    items = list(rows)
    both = [row for row in items if bool(row.get("current_success")) and bool(row.get("aips_success"))]
    delta_w = np.asarray([_safe_float(row.get("delta_fwhm_px")) for row in both], dtype=np.float64)
    delta_frac = np.asarray([_safe_float(row.get("delta_fwhm_frac")) for row in both], dtype=np.float64)
    delta_theta = np.asarray([_safe_float(row.get("delta_opening_angle_deg")) for row in both], dtype=np.float64)
    current_theta = np.asarray([_safe_float(row.get("current_opening_angle_deg")) for row in both], dtype=np.float64)
    aips_theta = np.asarray([_safe_float(row.get("aips_opening_angle_deg")) for row in both], dtype=np.float64)
    finite_w = delta_w[np.isfinite(delta_w)]
    finite_frac = delta_frac[np.isfinite(delta_frac)]
    theta_pair_mask = np.isfinite(delta_theta)
    finite_theta = delta_theta[theta_pair_mask]
    current_theta_pair = current_theta[theta_pair_mask]
    aips_theta_pair = aips_theta[theta_pair_mask]
    print(f"records evaluated: {len(items)}")
    print(f"both fits successful: {len(both)}")
    print(f"AIPS-like failures: {sum(1 for row in items if not bool(row.get('aips_success')))}")
    if finite_w.size:
        print(f"median delta FWHM (AIPS-current): {float(np.median(finite_w)):.6g} px")
        print(f"robust sigma delta FWHM:          {_robust_sigma(finite_w):.6g} px")
    if finite_frac.size:
        print(f"median fractional delta FWHM:     {100.0 * float(np.median(finite_frac)):.4g} %")
    if finite_theta.size:
        print(f"paired finite opening angles:     {int(finite_theta.size)}")
        print(f"median current opening angle:     {float(np.nanmedian(current_theta_pair)):.6g} deg")
        print(f"median AIPS-like opening angle:   {float(np.nanmedian(aips_theta_pair)):.6g} deg")
        print(f"median delta opening angle:       {float(np.median(finite_theta)):.6g} deg")
        print(f"robust sigma delta opening:       {_robust_sigma(finite_theta):.6g} deg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current center-lobe Gaussian fits with an AIPS-like subsection fit using saved IBAE JSON profiles."
    )
    parser.add_argument("analysis_json", help="Saved IBAE ridgeline analysis JSON.")
    parser.add_argument("--out", help="Output CSV path. Defaults to '<analysis>_profile_fit_mode_compare.csv'.")
    parser.add_argument(
        "--window-mode",
        choices=["full-support", "same-as-current", "center-beam"],
        default="full-support",
        help="AIPS-like fit subsection selection. full-support is BDROP=EDROP=0 over support-masked profile.",
    )
    parser.add_argument("--bdrop", type=int, default=0, help="Drop this many samples from the start for full-support mode.")
    parser.add_argument("--edrop", type=int, default=0, help="Drop this many samples from the end for full-support mode.")
    parser.add_argument("--baseline-order", type=int, choices=[-1, 0, 1, 2], default=0, help="AIPS-like baseline ORDER.")
    parser.add_argument(
        "--width-mode",
        choices=["intrinsic", "raw", "intrinsic-or-raw"],
        default="intrinsic",
        help="Width basis for opening-angle comparison.",
    )
    parser.add_argument(
        "--center-half-beam",
        type=float,
        default=2.0,
        help="Half-width in beam units for --window-mode center-beam.",
    )
    args = parser.parse_args()

    rows = compare_modes(
        args.analysis_json,
        window_mode=args.window_mode,
        bdrop=int(args.bdrop),
        edrop=int(args.edrop),
        baseline_order=int(args.baseline_order),
        width_mode=str(args.width_mode),
        center_half_beam=float(args.center_half_beam),
    )
    out = args.out
    if not out:
        src = Path(args.analysis_json)
        out = str(src.with_name(f"{src.stem}_profile_fit_mode_compare.csv"))
    _write_csv(rows, out)
    _print_summary(rows)
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
