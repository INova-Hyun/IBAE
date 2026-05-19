#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from fits_viewer import MAS_PER_DEG, read_primary_fits, robust_corner_rms
from polar_opening_angle import (
    measure_opening,
    plot_summary,
    power_law_fit,
    records_to_json,
    save_outputs,
)


def _beam_mas(header: Dict[str, object]) -> float:
    bmaj = float(header.get("BMAJ", float("nan"))) * MAS_PER_DEG
    bmin = float(header.get("BMIN", float("nan"))) * MAS_PER_DEG
    if np.isfinite(bmaj) and np.isfinite(bmin) and bmaj > 0.0 and bmin > 0.0:
        return float(math.sqrt(bmaj * bmin))
    return float("nan")


def _pixel_mas(header: Dict[str, object]) -> float:
    vals = []
    for key in ("CDELT1", "CDELT2"):
        if key in header:
            vals.append(abs(float(header[key])) * MAS_PER_DEG)
    finite = [v for v in vals if np.isfinite(v) and v > 0.0]
    return float(np.median(finite)) if finite else float("nan")


def _default_prefix(fits_path: Path, ridge_ascii_path: Path) -> Path:
    fits_name = fits_path.name
    if fits_name.endswith(".gz"):
        fits_name = fits_name[:-3]
    if fits_name.endswith(".fits"):
        fits_name = fits_name[:-5]
    return Path("output") / f"{fits_name}_{ridge_ascii_path.stem}"


def _load_ascii_ridge(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least RA and Dec columns")
    xy = np.asarray(data[:, :2], dtype=float)
    intensity = np.asarray(data[:, 2], dtype=float) if data.shape[1] >= 3 else None
    finite = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if intensity is not None:
        finite &= np.isfinite(intensity)
    xy = xy[finite]
    if intensity is not None:
        intensity = intensity[finite]
    if xy.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two finite ridge points")
    return xy, intensity


def _path_lengths(xy: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment)])


def _tangent_pa_deg(xy: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(xy, dtype=float)
    tangent[0] = xy[1] - xy[0]
    tangent[-1] = xy[-1] - xy[-2]
    if xy.shape[0] > 2:
        tangent[1:-1] = xy[2:] - xy[:-2]
    return (np.degrees(np.arctan2(tangent[:, 0], tangent[:, 1])) + 360.0) % 360.0


def ridge_payload_from_ascii(
    fits_path: Path,
    header: Dict[str, object],
    image: np.ndarray,
    ridge_ascii_path: Path,
) -> Dict[str, object]:
    xy, intensity = _load_ascii_ridge(ridge_ascii_path)
    path = _path_lengths(xy)
    radial = np.linalg.norm(xy, axis=1)
    point_pa = (np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) + 360.0) % 360.0
    tangent_pa = _tangent_pa_deg(xy)
    rms = robust_corner_rms(image)

    ridge_points = []
    raw_samples = []
    for idx, ((x_mas, y_mas), path_mas, radial_mas, pa_deg, tpa_deg) in enumerate(
        zip(xy, path, radial, point_pa, tangent_pa)
    ):
        row = {
            "index": int(idx),
            "radial_mas": float(radial_mas),
            "path_mas": float(path_mas),
            "x_mas": float(x_mas),
            "y_mas": float(y_mas),
            "pa_deg": float(pa_deg),
            "tangent_pa_deg": float(tpa_deg),
        }
        ridge_points.append(row)
        raw_row = dict(row)
        if intensity is not None:
            raw_row["intensity_jy"] = float(intensity[idx])
        raw_samples.append(raw_row)

    return {
        "format": "mojave_fits_external_pushkarev_ascii_ridgeline",
        "version": 1,
        "fits_file": str(fits_path),
        "ridge_ascii_file": str(ridge_ascii_path),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "bunit": str(header.get("BUNIT", "") or ""),
        "beam_mas": _beam_mas(header),
        "pixel_mas": _pixel_mas(header),
        "rms_jy_per_beam": float(rms),
        "core_mas": [0.0, 0.0],
        "sector": {
            "name": "pushkarev_ascii",
            "pa_min_deg": float(np.nanmin(point_pa)),
            "pa_max_deg": float(np.nanmax(point_pa)),
            "pa_width_deg": float(np.nanmax(point_pa) - np.nanmin(point_pa)),
        },
        "parameters": {
            "method": "external Pushkarev ASCII ridgeline; no spline or interpolation applied",
            "coordinate_columns": "RA Dec [I]",
            "path_method": "cumulative distance through input rows",
            "tangent_method": "finite difference on input rows",
            "input_point_count": int(xy.shape[0]),
        },
        "raw_samples": raw_samples,
        "ridge_points": ridge_points,
    }


def save_ridge_payload(prefix: Path, payload: Dict[str, object]) -> Tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_name(f"{prefix.name}_ridge.json")
    csv_path = prefix.with_name(f"{prefix.name}_ridge.csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "radial_mas", "path_mas", "x_mas", "y_mas", "pa_deg", "tangent_pa_deg", "intensity_jy"])
        raw_by_index = {int(row["index"]): row for row in payload.get("raw_samples", [])}
        for point in payload["ridge_points"]:
            raw = raw_by_index.get(int(point["index"]), {})
            writer.writerow(
                [
                    point["index"],
                    point["radial_mas"],
                    point["path_mas"],
                    point["x_mas"],
                    point["y_mas"],
                    point["pa_deg"],
                    point["tangent_pa_deg"],
                    raw.get("intensity_jy", ""),
                ]
            )
    return json_path, csv_path


def load_pushkarev_angle(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 11:
        raise ValueError(f"{path} must contain 11 columns")
    return data


def print_angle_comparison(angle_dat: Path, records: Sequence[object]) -> None:
    data = load_pushkarev_angle(angle_dat)
    push_r = data[:, 8]
    push_fwhm = data[:, 5]
    push_deconv = data[:, 6]
    push_angle = data[:, 9]
    rows = [record for record in records if record.fit.success and np.isfinite(record.path_mas)]
    if not rows:
        print("comparison: no successful local records")
        return

    local_r = np.asarray([r.path_mas for r in rows], dtype=float)
    local_fwhm = np.asarray([r.fit.fwhm_mas for r in rows], dtype=float)
    local_deconv = np.asarray([r.intrinsic_fwhm_mas for r in rows], dtype=float)
    local_angle = np.asarray([r.opening_angle_deg for r in rows], dtype=float)
    nearest = np.asarray([int(np.argmin(np.abs(local_r - value))) for value in push_r], dtype=int)
    close = np.abs(local_r[nearest] - push_r) <= 0.001
    if not np.any(close):
        print("comparison: no records matched to Pushkarev r within 0.001 mas")
        return

    def med_abs(delta: np.ndarray) -> float:
        return float(np.nanmedian(np.abs(delta[close])))

    print(f"comparison angle dat: {angle_dat}")
    print(f"  matched rows: {int(np.count_nonzero(close))} / {len(push_r)}")
    print(f"  median |FWHM_local - FWHM_pushkarev|:   {med_abs(local_fwhm[nearest] - push_fwhm):.6g} mas")
    print(f"  median |deconv_local - deconv_pushkarev|:{med_abs(local_deconv[nearest] - push_deconv):.6g} mas")
    print(f"  median |angle_local - angle_pushkarev|:  {med_abs(local_angle[nearest] - push_angle):.6g} deg")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use a Pushkarev ASCII ridgeline exactly as given, then run the existing "
            "Gaussian transverse-width and opening-angle measurement."
        )
    )
    parser.add_argument("fits", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "mojave_fits" / "0238+711.u.stacked.icc.fits")
    parser.add_argument("ridge_ascii", nargs="?", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Output prefix. Default: output/<fits>_<ridge_ascii>")
    parser.add_argument("--angle-dat", type=Path, default=None, help="Optional Pushkarev angle.dat file for comparison.")
    parser.add_argument("--analysis-sep-min", type=float, default=0.5)
    parser.add_argument("--analysis-sep-max", type=float, default=None)
    parser.add_argument("--k-fit-sep-max", type=float, default=11.0, help="Maximum separation used only for d=A*r^k fitting.")
    parser.add_argument("--slice-half-width", type=float, default=None)
    parser.add_argument("--sample-step", type=float, default=None)
    parser.add_argument("--fit-threshold-snr", type=float, default=3.0)
    parser.add_argument("--fit-threshold", type=float, default=None)
    parser.add_argument("--fit-padding", type=float, default=None)
    parser.add_argument("--baseline-guard", type=float, default=None)
    parser.add_argument("--min-fit-points", type=int, default=8)
    parser.add_argument("--sigma-upper", type=float, default=None)
    parser.add_argument("--pa-sweep", action="store_true")
    parser.add_argument("--pa-sweep-range", type=float, default=15.0)
    parser.add_argument("--pa-sweep-step", type=float, default=1.0)
    parser.add_argument("--pa-sweep-workers", type=int, default=1)
    parser.add_argument("--pa-sweep-analysis-only", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260430)
    parser.add_argument("--save-figure", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true", help="Run only; do not write JSON/CSV outputs.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.ridge_ascii is None:
        parser.error("ridge_ascii is required for Pushkarev ASCII comparison; only the 0238+711 FITS file is bundled.")
    output_prefix = args.output or _default_prefix(args.fits, args.ridge_ascii)

    image, header = read_primary_fits(args.fits)
    ridge_payload = ridge_payload_from_ascii(args.fits, header, image, args.ridge_ascii)
    records, summary = measure_opening(image, header, ridge_payload, args)
    if args.k_fit_sep_max is not None:
        k_sep_min = float(args.analysis_sep_min)
        k_sep_max = float(args.k_fit_sep_max)
        summary["k_fit_separation_min_mas"] = k_sep_min
        summary["k_fit_separation_max_mas"] = k_sep_max
        summary["power_law_fit"] = power_law_fit(records, k_sep_min, k_sep_max, smooth5=False)
        summary["power_law_fit_smoothed5"] = power_law_fit(records, k_sep_min, k_sep_max, smooth5=True)
    opening_payload = {
        "format": "mojave_fits_pushkarev_ascii_opening_angle",
        "version": 1,
        "fits_file": str(args.fits),
        "ridge_file": str(args.ridge_ascii),
        "ridge_sector": ridge_payload.get("sector", {}),
        "ridge_parameters": ridge_payload.get("parameters", {}),
        "summary": summary,
        "records": records_to_json(records),
    }

    print(f"fits:        {args.fits}")
    print(f"ridge ascii: {args.ridge_ascii}")
    print(f"ridge points={len(ridge_payload['ridge_points'])}, path_max={ridge_payload['ridge_points'][-1]['path_mas']:.6g} mas")
    print(f"records={summary['record_count']}, fit_success={summary['fit_success_count']}, unresolved={summary['unresolved_count']}")
    print(f"median raw full opening angle:         {summary['median_opening_angle_raw_deg']:.6g} deg")
    print(f"median deconvolved full opening angle: {summary['median_opening_angle_deg']:.6g} deg")
    print(f"5-point smoothed deconv median:        {summary['five_point_smoothed_median_opening_angle_deg']:.6g} deg")
    median_error = dict(summary.get("median_error", {}))
    intrinsic_block = dict(median_error.get("intrinsic_block", {}))
    intrinsic_block_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
    display_err = intrinsic_block_with_pa if math.isfinite(intrinsic_block_with_pa.get("sigma", float("nan"))) else intrinsic_block
    print(
        "median deconv error:                  "
        f"{display_err.get('sigma', float('nan')):.6g} deg "
        f"(block {median_error.get('block_size_slices', 'n/a')} slices = "
        f"{median_error.get('block_length_mas', float('nan')):.6g} mas; "
        f"target 0.5 beam = {median_error.get('block_target_mas', float('nan')):.6g} mas)"
    )
    fit5 = dict(summary.get("power_law_fit_smoothed5", {}))
    print(
        "power-law d=A*r^k, smooth5:           "
        f"k={fit5.get('k', float('nan')):.6g}, "
        f"A={fit5.get('amplitude_mas_at_1mas', float('nan')):.6g} mas, "
        f"n={fit5.get('n', 0)}"
    )

    if args.angle_dat is not None:
        print_angle_comparison(args.angle_dat, records)

    if not args.no_save:
        ridge_json, ridge_csv = save_ridge_payload(output_prefix, ridge_payload)
        opening_json, opening_csv = save_outputs(output_prefix.with_name(f"{output_prefix.name}_opening"), opening_payload, records)
        print(f"saved ridge json:   {ridge_json}")
        print(f"saved ridge csv:    {ridge_csv}")
        print(f"saved opening json: {opening_json}")
        print(f"saved opening csv:  {opening_csv}")

    if args.save_figure is not None:
        args.save_figure.parent.mkdir(parents=True, exist_ok=True)
        plot_summary(
            image,
            header,
            ridge_payload,
            records,
            summary,
            args.save_figure,
            width_xlim=(0.4, 13.0),
            width_ylim=(0.07, 13.0),
        )
        print(f"saved figure:       {args.save_figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
