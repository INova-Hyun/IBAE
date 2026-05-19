#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .common.numeric import safe_float as _safe_float
    from .session_analysis import load_analysis_session
except Exception:
    import sys

    _HERE = Path(__file__).resolve().parent
    _PARENT = _HERE.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))
    from IBAE.common.numeric import safe_float as _safe_float
    from IBAE.session_analysis import load_analysis_session


def _load_fit_records(path: str) -> List[Dict[str, Any]]:
    payload = load_analysis_session(path)
    measurement = payload.get("measurement_result", {})
    if not isinstance(measurement, dict):
        raise ValueError("Analysis JSON does not contain a valid measurement_result.")
    fit_records = measurement.get("fit_records", [])
    if not isinstance(fit_records, list) or not fit_records:
        raise ValueError("Analysis JSON does not contain any fit_records.")
    return [dict(rec) for rec in fit_records if isinstance(rec, dict)]


def _successful_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok: List[Dict[str, Any]] = []
    for rec in records:
        fit = rec.get("fit", {})
        if isinstance(fit, dict) and bool(fit.get("success", False)):
            ok.append(rec)
    return ok


def _select_record(
    records: List[Dict[str, Any]],
    record_index: Optional[int],
    success_index: Optional[int],
    ridge_idx: Optional[int],
) -> Dict[str, Any]:
    if ridge_idx is not None:
        matches = [rec for rec in records if int(rec.get("ridge_idx", -10**9)) == int(ridge_idx)]
        if not matches:
            raise ValueError(f"No fit record found with ridge_idx={ridge_idx}.")
        match_ok = [rec for rec in matches if bool(dict(rec.get("fit", {})).get("success", False))]
        return dict(match_ok[0] if match_ok else matches[0])
    if record_index is not None:
        if record_index < 0 or record_index >= len(records):
            raise ValueError(f"record-index {record_index} is out of range (0..{len(records)-1}).")
        return dict(records[record_index])
    ok = _successful_records(records)
    if not ok:
        raise ValueError("No successful Gaussian fit records found in this analysis JSON.")
    idx = 0 if success_index is None else int(success_index)
    if idx < 0 or idx >= len(ok):
        raise ValueError(f"success-index {idx} is out of range (0..{len(ok)-1}).")
    return dict(ok[idx])


def _default_output_path(analysis_path: str, record: Dict[str, Any]) -> Path:
    src = Path(analysis_path)
    ridge_idx = int(record.get("ridge_idx", 0))
    return src.with_name(f"{src.stem}_transverse_fit_r{ridge_idx:04d}.png")


def _plot_record(record: Dict[str, Any], analysis_path: str, out_path: str, dpi: int) -> None:
    profile = dict(record.get("profile", {}))
    fit = dict(record.get("fit", {}))
    if not bool(fit.get("success", False)):
        raise ValueError("Selected record does not contain a successful Gaussian fit.")

    sample_k = np.asarray(profile.get("sample_k", []), dtype=np.float64)
    raw_profile = np.asarray(profile.get("profile", []), dtype=np.float64)
    support_profile = np.asarray(profile.get("support_profile", []), dtype=np.float64)
    x_fit = np.asarray(fit.get("x", []), dtype=np.float64)
    y_fit_input = np.asarray(fit.get("y", []), dtype=np.float64)
    y_fit_model = np.asarray(fit.get("fit_y", []), dtype=np.float64)

    params = dict(fit.get("params", {}))
    mu = _safe_float(params.get("mu"))
    sigma = _safe_float(params.get("sigma"))
    baseline = _safe_float(params.get("baseline"))
    amplitude = _safe_float(params.get("amplitude"))
    fwhm_px = _safe_float(record.get("fwhm_px", fit.get("fwhm_px")))
    fwhm_mas = _safe_float(record.get("fwhm_mas"))
    distance_mas = _safe_float(record.get("distance_from_core_mas"))
    rmse = _safe_float(fit.get("rmse"))
    left_lim = _safe_float(profile.get("left_lim"))
    right_lim = _safe_float(profile.get("right_lim"))

    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)

    if sample_k.size and raw_profile.size == sample_k.size:
        ax.plot(sample_k, raw_profile, color="#b0b0b0", lw=1.2, alpha=0.85, label="Full transverse profile")
    if sample_k.size and support_profile.size == sample_k.size:
        ax.plot(sample_k, support_profile, color="#4d4d4d", lw=1.4, alpha=0.95, label="Support-masked profile")
    if np.isfinite(left_lim) and np.isfinite(right_lim):
        ax.axvspan(left_lim, right_lim, color="#d9ecff", alpha=0.35, label="Fit support window")

    ax.scatter(x_fit, y_fit_input, s=26, color="#16bac5", edgecolor="none", zorder=3, label="Gaussian fit samples")
    ax.plot(x_fit, y_fit_model, color="#e63946", lw=2.2, zorder=4, label="Gaussian model")

    if np.isfinite(mu):
        ax.axvline(mu, color="#e63946", ls="--", lw=1.5, alpha=0.95, label=r"$\mu$")
    if np.isfinite(mu) and np.isfinite(fwhm_px):
        fwhm_left = float(mu - 0.5 * fwhm_px)
        fwhm_right = float(mu + 0.5 * fwhm_px)
        ax.axvline(fwhm_left, color="#ff7f11", ls=":", lw=1.6, alpha=0.95)
        ax.axvline(fwhm_right, color="#ff7f11", ls=":", lw=1.6, alpha=0.95)
        if np.isfinite(baseline) and np.isfinite(amplitude):
            half_level = float(baseline + 0.5 * amplitude)
            ax.hlines(
                half_level,
                fwhm_left,
                fwhm_right,
                colors="#ff7f11",
                lw=2.0,
                alpha=0.95,
                label="FWHM span",
            )

    ridge_idx = int(record.get("ridge_idx", -1))
    title_left = f"{Path(analysis_path).name} | ridge_idx={ridge_idx}"
    title_right = (
        f"distance={distance_mas:.3f} mas | FWHM={fwhm_px:.3f} px"
        + (f" ({fwhm_mas:.3f} mas)" if np.isfinite(fwhm_mas) else "")
        + (f" | RMSE={rmse:.3f}" if np.isfinite(rmse) else "")
    )
    ax.set_title(title_left + "\n" + title_right, fontsize=12)
    ax.set_xlabel("Offset along transverse slice (px)")
    ax.set_ylabel("Intensity / reconstructed flux")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=9)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=int(max(72, dpi)), facecolor="white")
    plt.close(fig)


def _print_available_records(records: List[Dict[str, Any]], limit: int) -> None:
    ok = _successful_records(records)
    print(f"Total fit records: {len(records)}")
    print(f"Successful fits:  {len(ok)}")
    print("success_idx | ridge_idx | dist_core_mas | fwhm_px | fwhm_mas")
    for idx, rec in enumerate(ok[: max(1, limit)]):
        print(
            f"{idx:10d} | "
            f"{int(rec.get('ridge_idx', -1)):9d} | "
            f"{_safe_float(rec.get('distance_from_core_mas')):13.4f} | "
            f"{_safe_float(rec.get('fwhm_px')):7.3f} | "
            f"{_safe_float(rec.get('fwhm_mas')):8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one 1D transverse Gaussian FWHM fit plot from an IBAE analysis JSON."
    )
    parser.add_argument("analysis_json", help="Path to saved ridgeline analysis JSON.")
    parser.add_argument("--out", help="Output PNG path. Defaults to '<analysis>_transverse_fit_rXXXX.png'.")
    parser.add_argument("--record-index", type=int, default=None, help="Absolute fit_records index to plot.")
    parser.add_argument(
        "--success-index",
        type=int,
        default=0,
        help="Index among successful Gaussian fits to plot. Ignored if --record-index or --ridge-idx is given.",
    )
    parser.add_argument("--ridge-idx", type=int, default=None, help="Pick the fit record with this ridge_idx.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved PNG DPI.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available successful fit slices and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=20,
        help="How many successful records to print with --list.",
    )
    args = parser.parse_args()

    records = _load_fit_records(args.analysis_json)
    if args.list:
        _print_available_records(records, limit=int(max(1, args.list_limit)))
        return

    record = _select_record(
        records=records,
        record_index=args.record_index,
        success_index=args.success_index,
        ridge_idx=args.ridge_idx,
    )
    out_path = str(args.out) if args.out else str(_default_output_path(args.analysis_json, record))
    _plot_record(record=record, analysis_path=args.analysis_json, out_path=out_path, dpi=int(args.dpi))
    print(f"Saved transverse Gaussian fit plot: {out_path}")


if __name__ == "__main__":
    main()
