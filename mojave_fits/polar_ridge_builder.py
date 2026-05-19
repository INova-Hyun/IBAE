#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fits_viewer import (
    MAS_PER_DEG,
    apply_stretch,
    auto_limits,
    image_edges_mas,
    positive_contour_levels,
    read_primary_fits,
    robust_corner_rms,
    zoom_axes,
)
from ridge_builder import sample_image_bilinear


DEFAULT_EAST = ("east", 35.0, 115.0)
DEFAULT_WEST = ("west", 215.0, 295.0)


@dataclass
class Sector:
    name: str
    pa_min_deg: float
    pa_max_deg: float


@dataclass
class PolarSample:
    radial_mas: float
    x_mas: float
    y_mas: float
    pa_deg: float
    pa_unwrapped_deg: float
    peak: float
    flux_sum: float
    n_used: int
    pa_span_used_deg: float


@dataclass
class RidgePoint:
    radial_mas: float
    path_mas: float
    x_mas: float
    y_mas: float
    pa_deg: float
    tangent_pa_deg: float


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


def _fits_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".fits"):
        name = name[:-5]
    return name


def _default_prefix(path: Path) -> Path:
    return path.with_name(f"{_fits_stem(path)}_polar_ridge")


def _prefixed_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def parse_core(text: str) -> Tuple[float, float]:
    try:
        x_str, y_str = text.replace(" ", "").split(",", 1)
        return float(x_str), float(y_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--core expects x,y in mas") from exc


def parse_sector(items: Sequence[str]) -> Sector:
    if len(items) != 3:
        raise argparse.ArgumentTypeError("--sector expects NAME PA_MIN PA_MAX")
    name, pa_min, pa_max = items
    return Sector(str(name), float(pa_min), float(pa_max))


def sector_width_deg(pa_min: float, pa_max: float) -> float:
    width = (float(pa_max) - float(pa_min)) % 360.0
    if width <= 0:
        width = 360.0
    return width


def pa_grid_for_sector(sector: Sector, pa_step_deg: float) -> np.ndarray:
    width = sector_width_deg(sector.pa_min_deg, sector.pa_max_deg)
    n = int(math.floor(width / pa_step_deg)) + 1
    grid = sector.pa_min_deg + np.arange(n, dtype=float) * pa_step_deg
    if grid[-1] < sector.pa_min_deg + width:
        grid = np.append(grid, sector.pa_min_deg + width)
    return grid


def pa_to_xy(core_mas: Tuple[float, float], radius_mas: float, pa_unwrapped_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pa_rad = np.deg2rad(np.mod(pa_unwrapped_deg, 360.0))
    x = core_mas[0] + radius_mas * np.sin(pa_rad)
    y = core_mas[1] + radius_mas * np.cos(pa_rad)
    return x, y


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


def select_arc_component(valid: np.ndarray, values: np.ndarray, mode: str) -> np.ndarray:
    runs = contiguous_true_runs(valid)
    selected = np.zeros_like(valid, dtype=bool)
    if not runs:
        return selected
    if mode == "all":
        selected[:] = valid
        return selected
    if mode == "largest":
        start, stop = max(runs, key=lambda run: run[1] - run[0])
        selected[start:stop] = True
        return selected
    if mode == "peak":
        peak_index = int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))
        for start, stop in runs:
            if start <= peak_index < stop:
                selected[start:stop] = True
                return selected
        start, stop = max(runs, key=lambda run: run[1] - run[0])
        selected[start:stop] = True
        return selected
    raise ValueError(f"Unknown arc component mode: {mode}")


def weighted_flux_median(pa_unwrapped: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    cumulative = np.cumsum(weights)
    target = 0.5 * total
    idx = int(np.searchsorted(cumulative, target, side="left"))
    if idx <= 0:
        return float(pa_unwrapped[0])
    if idx >= len(pa_unwrapped):
        return float(pa_unwrapped[-1])
    before = cumulative[idx - 1]
    after = cumulative[idx]
    if after <= before:
        return float(pa_unwrapped[idx])
    frac = (target - before) / (after - before)
    return float(pa_unwrapped[idx - 1] * (1.0 - frac) + pa_unwrapped[idx] * frac)


def angular_delta_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def filter_polar_sample_outliers(
    samples: Sequence[PolarSample],
    radial_step_mas: float,
    max_sample_step_factor: float = 5.0,
    max_sample_pa_jump_deg: float = 45.0,
) -> List[PolarSample]:
    if not samples:
        return []
    accepted: List[PolarSample] = []
    step = max(float(radial_step_mas), 1e-12)
    for sample in samples:
        if not accepted:
            accepted.append(sample)
            continue
        prev = accepted[-1]
        dr = max(abs(float(sample.radial_mas) - float(prev.radial_mas)), step)
        ds = math.hypot(float(sample.x_mas) - float(prev.x_mas), float(sample.y_mas) - float(prev.y_mas))
        dpa = angular_delta_deg(sample.pa_deg, prev.pa_deg)
        if max_sample_step_factor > 0 and ds > float(max_sample_step_factor) * dr:
            continue
        if max_sample_pa_jump_deg > 0 and min(sample.radial_mas, prev.radial_mas) > 0.25 and dpa > float(max_sample_pa_jump_deg):
            continue
        accepted.append(sample)
    return accepted


def build_polar_samples(
    image: np.ndarray,
    header: Dict[str, object],
    sector: Sector,
    core_mas: Tuple[float, float],
    rmin_mas: float,
    rmax_mas: float,
    r_step_mas: float,
    pa_step_deg: float,
    threshold: float,
    component_mode: str,
    max_empty: int,
    min_arc_points: int = 8,
    min_arc_span_deg: float = 2.0,
    min_peak_over_threshold: float = 1.05,
) -> List[PolarSample]:
    pa_grid = pa_grid_for_sector(sector, pa_step_deg)
    radial_grid = np.arange(max(0.0, rmin_mas), rmax_mas + 0.5 * r_step_mas, r_step_mas)
    samples: List[PolarSample] = []
    consecutive_empty = 0

    for radius in radial_grid:
        x_arc, y_arc = pa_to_xy(core_mas, float(radius), pa_grid)
        values = sample_image_bilinear(image, header, x_arc, y_arc)
        valid = np.isfinite(values) & (values > threshold)
        selected = select_arc_component(valid, values, component_mode)
        n_selected = int(np.count_nonzero(selected))
        if n_selected < max(2, int(min_arc_points)):
            consecutive_empty += 1
            if samples and max_empty > 0 and consecutive_empty >= max_empty:
                break
            continue
        weights = np.clip(values[selected], 0.0, None)
        flux_sum = float(np.sum(weights))
        if not np.isfinite(flux_sum) or flux_sum <= 0:
            consecutive_empty += 1
            continue

        pa_selected = pa_grid[selected]
        pa_span = float(np.nanmax(pa_selected) - np.nanmin(pa_selected))
        peak = float(np.nanmax(values[selected]))
        peak_ratio = peak / float(threshold) if threshold > 0 else float("inf")
        if pa_span < float(min_arc_span_deg) or peak_ratio < float(min_peak_over_threshold):
            consecutive_empty += 1
            if samples and max_empty > 0 and consecutive_empty >= max_empty:
                break
            continue
        pa_med = weighted_flux_median(pa_selected, weights)
        if not np.isfinite(pa_med):
            consecutive_empty += 1
            continue
        x_ridge, y_ridge = pa_to_xy(core_mas, float(radius), np.asarray([pa_med], dtype=float))
        consecutive_empty = 0
        samples.append(
            PolarSample(
                radial_mas=float(radius),
                x_mas=float(x_ridge[0]),
                y_mas=float(y_ridge[0]),
                pa_deg=float(pa_med % 360.0),
                pa_unwrapped_deg=float(pa_med),
                peak=peak,
                flux_sum=flux_sum,
                n_used=n_selected,
                pa_span_used_deg=pa_span,
            )
        )
    return samples


def smooth_resample_polar_samples(
    samples: Sequence[PolarSample],
    core_mas: Tuple[float, float],
    radial_step_mas: float,
    smoothing_mas: float,
) -> List[RidgePoint]:
    if len(samples) < 2:
        return []
    radial = np.asarray([0.0, *[s.radial_mas for s in samples]], dtype=float)
    xy = np.asarray([core_mas, *[(s.x_mas, s.y_mas) for s in samples]], dtype=float)
    keep = np.concatenate([[True], np.diff(radial) > 1e-8])
    radial = radial[keep]
    xy = xy[keep]
    if len(radial) < 2:
        return []

    out_r = np.arange(0.0, float(np.nanmax(radial)) + 0.5 * radial_step_mas, radial_step_mas)
    if len(radial) >= 4:
        try:
            from scipy.interpolate import UnivariateSpline

            smooth = max(0.0, float(smoothing_mas)) ** 2 * len(radial)
            sx = UnivariateSpline(radial, xy[:, 0], k=min(3, len(radial) - 1), s=smooth)
            sy = UnivariateSpline(radial, xy[:, 1], k=min(3, len(radial) - 1), s=smooth)
            out_xy = np.column_stack([sx(out_r), sy(out_r)])
        except Exception:
            out_xy = np.column_stack([np.interp(out_r, radial, xy[:, 0]), np.interp(out_r, radial, xy[:, 1])])
    else:
        out_xy = np.column_stack([np.interp(out_r, radial, xy[:, 0]), np.interp(out_r, radial, xy[:, 1])])
    out_xy[0] = np.asarray(core_mas, dtype=float)

    segment = np.linalg.norm(np.diff(out_xy, axis=0), axis=1)
    path = np.concatenate([[0.0], np.cumsum(segment)])
    pa = (np.degrees(np.arctan2(out_xy[:, 0] - core_mas[0], out_xy[:, 1] - core_mas[1])) + 360.0) % 360.0
    if len(out_xy) > 1:
        dx = np.gradient(out_xy[:, 0], out_r, edge_order=1)
        dy = np.gradient(out_xy[:, 1], out_r, edge_order=1)
        tangent_pa = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    else:
        tangent_pa = np.asarray([float("nan")])

    return [
        RidgePoint(
            radial_mas=float(r),
            path_mas=float(s),
            x_mas=float(x),
            y_mas=float(y),
            pa_deg=float(point_pa),
            tangent_pa_deg=float(tpa),
        )
        for r, s, (x, y), point_pa, tpa in zip(out_r, path, out_xy, pa, tangent_pa)
    ]


def result_payload(
    fits_path: Path,
    header: Dict[str, object],
    image: np.ndarray,
    rms: float,
    threshold: float,
    core_mas: Tuple[float, float],
    sector: Sector,
    samples: Sequence[PolarSample],
    ridge: Sequence[RidgePoint],
    params: Dict[str, object],
) -> Dict[str, object]:
    return {
        "format": "mojave_fits_polar_ridgeline",
        "version": 1,
        "fits_file": str(fits_path),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "bunit": str(header.get("BUNIT", "") or ""),
        "beam_mas": _beam_mas(header),
        "pixel_mas": _pixel_mas(header),
        "rms_jy_per_beam": float(rms),
        "threshold_jy_per_beam": float(threshold),
        "core_mas": [float(core_mas[0]), float(core_mas[1])],
        "sector": {
            "name": sector.name,
            "pa_min_deg": sector.pa_min_deg,
            "pa_max_deg": sector.pa_max_deg,
            "pa_width_deg": sector_width_deg(sector.pa_min_deg, sector.pa_max_deg),
        },
        "parameters": params,
        "raw_samples": [
            {
                "radial_mas": s.radial_mas,
                "x_mas": s.x_mas,
                "y_mas": s.y_mas,
                "pa_deg": s.pa_deg,
                "pa_unwrapped_deg": s.pa_unwrapped_deg,
                "peak": s.peak,
                "flux_sum": s.flux_sum,
                "n_used": s.n_used,
                "pa_span_used_deg": s.pa_span_used_deg,
            }
            for s in samples
        ],
        "ridge_points": [
            {
                "radial_mas": p.radial_mas,
                "path_mas": p.path_mas,
                "x_mas": p.x_mas,
                "y_mas": p.y_mas,
                "pa_deg": p.pa_deg,
                "tangent_pa_deg": p.tangent_pa_deg,
            }
            for p in ridge
        ],
    }


def save_result(prefix: Path, payload: Dict[str, object]) -> Tuple[Path, Path]:
    name = str(payload["sector"]["name"])
    target_prefix = prefix.parent / f"{prefix.name}_{name}"
    json_path = _prefixed_path(target_prefix, ".json")
    csv_path = _prefixed_path(target_prefix, ".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["radial_mas", "path_mas", "x_mas", "y_mas", "pa_deg", "tangent_pa_deg"])
        for point in payload["ridge_points"]:
            writer.writerow(
                [
                    point["radial_mas"],
                    point["path_mas"],
                    point["x_mas"],
                    point["y_mas"],
                    point["pa_deg"],
                    point["tangent_pa_deg"],
                ]
            )
    return json_path, csv_path


def draw_base(ax, image: np.ndarray, header: Dict[str, object], rms: float, stretch: str, cmap: str, contour_sigma: float, contour_factor: float):
    finite = image[np.isfinite(image)]
    vmin = max(float(np.nanpercentile(finite, 1.0)), -3.0 * rms) if np.isfinite(rms) else float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.9))
    display = apply_stretch(image, stretch, vmin, vmax)
    extent = image_edges_mas(header, image.shape)
    im = ax.imshow(display, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    levels = positive_contour_levels(rms, float(np.nanmax(image)), contour_sigma, contour_factor)
    if levels.size:
        ax.contour(image, levels=levels, colors="white", linewidths=0.55, alpha=0.75, origin="lower", extent=extent)
    return im


def draw_sector_boundaries(ax, core_mas: Tuple[float, float], sector: Sector, radius_mas: float) -> None:
    width = sector_width_deg(sector.pa_min_deg, sector.pa_max_deg)
    pa_arc = sector.pa_min_deg + np.linspace(0.0, width, 128)
    x_arc, y_arc = pa_to_xy(core_mas, radius_mas, pa_arc)
    ax.plot(x_arc, y_arc, lw=0.8, alpha=0.6, color="#00e5ff")
    for pa in (sector.pa_min_deg, sector.pa_min_deg + width):
        x_edge, y_edge = pa_to_xy(core_mas, radius_mas, np.asarray([pa], dtype=float))
        ax.plot([core_mas[0], x_edge[0]], [core_mas[1], y_edge[0]], lw=0.8, alpha=0.6, color="#00e5ff")
    xm, ym = pa_to_xy(core_mas, radius_mas * 0.92, np.asarray([sector.pa_min_deg + 0.5 * width], dtype=float))
    ax.text(xm[0], ym[0], sector.name, color="#00e5ff", fontsize=9, ha="center", va="center")


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


class PolarRidgeApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.fits_path = args.fits
        self.image, self.header = read_primary_fits(self.fits_path)
        self.rms = robust_corner_rms(self.image)
        self.threshold = args.threshold if args.threshold is not None else args.threshold_snr * self.rms
        self.core_mas = tuple(args.core)
        self.sectors = sectors_from_args(args)
        self.output_prefix = args.output or _default_prefix(self.fits_path)
        self.results: List[Dict[str, object]] = []

    def build(self) -> List[Dict[str, object]]:
        params = {
            "rmin_mas": self.args.rmin,
            "rmax_mas": self.args.rmax,
            "r_step_mas": self.args.step,
            "pa_step_deg": self.args.pa_step,
            "threshold_snr": self.args.threshold_snr,
            "threshold": self.threshold,
            "component_mode": self.args.component,
            "smooth_mas": self.args.smooth,
            "max_empty": self.args.max_empty,
            "min_arc_points": self.args.min_arc_points,
            "min_arc_span_deg": self.args.min_arc_span,
            "min_peak_over_threshold": self.args.min_peak_over_threshold,
            "max_sample_step_factor": self.args.max_sample_step_factor,
            "max_sample_pa_jump_deg": self.args.max_sample_pa_jump,
            "method": "core-centered polar arc flux-median ridgeline",
        }
        self.results = []
        for sector in self.sectors:
            samples = build_polar_samples(
                self.image,
                self.header,
                sector,
                self.core_mas,
                self.args.rmin,
                self.args.rmax,
                self.args.step,
                self.args.pa_step,
                self.threshold,
                self.args.component,
                self.args.max_empty,
                self.args.min_arc_points,
                self.args.min_arc_span,
                self.args.min_peak_over_threshold,
            )
            filtered_samples = filter_polar_sample_outliers(
                samples,
                self.args.step,
                self.args.max_sample_step_factor,
                self.args.max_sample_pa_jump,
            )
            sector_params = dict(params)
            sector_params["raw_sample_count_before_outlier_filter"] = len(samples)
            sector_params["outlier_filtered_count"] = max(0, len(samples) - len(filtered_samples))
            ridge = smooth_resample_polar_samples(filtered_samples, self.core_mas, self.args.step, self.args.smooth)
            payload = result_payload(
                self.fits_path,
                self.header,
                self.image,
                self.rms,
                self.threshold,
                self.core_mas,
                sector,
                filtered_samples,
                ridge,
                sector_params,
            )
            self.results.append(payload)
            print(f"{sector.name}: raw_samples={len(filtered_samples)} filtered={len(samples) - len(filtered_samples)}, ridge_points={len(ridge)}")
        return self.results

    def save(self) -> None:
        if not self.results:
            self.build()
        for payload in self.results:
            json_path, csv_path = save_result(self.output_prefix, payload)
            print(f"saved ridge: {json_path}")
            print(f"saved csv:   {csv_path}")

    def plot(self, save_path: Optional[Path] = None, show: bool = True) -> None:
        if not self.results:
            self.build()
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        draw_base(
            ax,
            self.image,
            self.header,
            self.rms,
            self.args.stretch,
            self.args.cmap,
            self.args.contour_sigma,
            self.args.contour_factor,
        )
        ax.scatter([self.core_mas[0]], [self.core_mas[1]], marker="+", s=120, c="#00e5ff", linewidths=1.8, zorder=7)
        for payload in self.results:
            sector_data = payload["sector"]
            sector = Sector(str(sector_data["name"]), float(sector_data["pa_min_deg"]), float(sector_data["pa_max_deg"]))
            draw_sector_boundaries(ax, self.core_mas, sector, self.args.rmax)
            raw = np.asarray([(p["x_mas"], p["y_mas"]) for p in payload["raw_samples"]], dtype=float)
            if raw.size:
                ax.scatter(raw[:, 0], raw[:, 1], s=9, c="#ffdf5d", edgecolors="none", alpha=0.75, zorder=8)
            ridge = np.asarray([(p["x_mas"], p["y_mas"]) for p in payload["ridge_points"]], dtype=float)
            if ridge.size:
                ax.plot(ridge[:, 0], ridge[:, 1], lw=2.0, color="#22ff66", alpha=0.95, zorder=9)
        beam = _beam_mas(self.header)
        pix = _pixel_mas(self.header)
        bunit = str(self.header.get("BUNIT", "") or "")
        ax.set_title(
            f"{self.fits_path.name}\n"
            f"polar ridgeline, beam={beam:.3f} mas, pixel={pix:.4f} mas, rms~{self.rms:.3g} {bunit}"
        )
        ax.set_xlabel("Relative RA offset (mas; positive left if CDELT1<0)")
        ax.set_ylabel("Relative Dec offset (mas)")
        limits = None if self.args.full_view else auto_limits(self.image, self.header, self.rms, 0.005, 8.0, 2.0)
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
            ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
            ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
        if self.args.fov is not None:
            half = float(self.args.fov)
            ax.set_xlim(half, -half)
            ax.set_ylim(-half, half)
        all_xy = []
        for payload in self.results:
            for key in ("raw_samples", "ridge_points"):
                all_xy.extend((p["x_mas"], p["y_mas"]) for p in payload.get(key, []))
        if all_xy:
            expand_axes_to_points(ax, np.asarray(all_xy, dtype=float), padding_mas=max(0.5, beam))

        full_xlim = image_edges_mas(self.header, self.image.shape)[:2]
        full_ylim = image_edges_mas(self.header, self.image.shape)[2:]
        reset_xlim = ax.get_xlim()
        reset_ylim = ax.get_ylim()

        def on_scroll(event) -> None:
            if event.inaxes is not ax:
                return
            center = None
            if event.xdata is not None and event.ydata is not None:
                center = (float(event.xdata), float(event.ydata))
            zoom_axes(ax, 1.0 / 1.25 if event.button == "up" else 1.25, center)
            fig.canvas.draw_idle()

        def on_key(event) -> None:
            if event.key == "q":
                plt.close(fig)
            elif event.key == "r":
                ax.set_xlim(reset_xlim)
                ax.set_ylim(reset_ylim)
                fig.canvas.draw_idle()
            elif event.key == "f":
                ax.set_xlim(full_xlim)
                ax.set_ylim(full_ylim)
                fig.canvas.draw_idle()
            elif event.key in ("+", "="):
                zoom_axes(ax, 1.0 / 1.25)
                fig.canvas.draw_idle()
            elif event.key in ("-", "_"):
                zoom_axes(ax, 1.25)
                fig.canvas.draw_idle()
            elif event.key == "s":
                self.save()

        fig.canvas.mpl_connect("scroll_event", on_scroll)
        fig.canvas.mpl_connect("key_press_event", on_key)
        print("Plot keys: mouse wheel/+/-=zoom, f=full view, r=reset, s=save, q=quit")
        if save_path is not None:
            fig.savefig(save_path, dpi=180)
            print(f"saved preview: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


def sectors_from_args(args: argparse.Namespace) -> List[Sector]:
    sectors: List[Sector] = []
    if args.sector:
        sectors.extend(parse_sector(item) for item in args.sector)
    if args.both:
        names = {s.name for s in sectors}
        if "east" not in names:
            sectors.append(Sector(*DEFAULT_EAST))
        if "west" not in names:
            sectors.append(Sector(*DEFAULT_WEST))
    if not sectors:
        sectors.append(Sector(*DEFAULT_EAST))
    return sectors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pushkarev-like polar ridgeline builder for MOJAVE stacked image FITS. "
            "Default is NGC 1052 eastern/approaching sector: east 35 115, rmax=14.1 mas."
        )
    )
    parser.add_argument("fits", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "mojave_fits" / "0238+711.u.stacked.icc.fits")
    parser.add_argument("--sector", nargs=3, action="append", metavar=("NAME", "PA_MIN", "PA_MAX"))
    parser.add_argument("--both", action="store_true", help="Add default east and west NGC 1052 sectors.")
    parser.add_argument("--core", type=parse_core, default=(0.0, 0.0), help="Core as x,y in mas. Default: 0,0.")
    parser.add_argument("--rmin", type=float, default=0.05)
    parser.add_argument("--rmax", type=float, default=14.1)
    parser.add_argument("--step", type=float, default=0.05, help="Radial/ridge resampling step in mas.")
    parser.add_argument("--pa-step", type=float, default=0.25, help="PA arc sampling step in degrees.")
    parser.add_argument("--threshold-snr", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=None, help="Absolute threshold in image units.")
    parser.add_argument("--component", choices=("peak", "largest", "all"), default="all")
    parser.add_argument("--smooth", type=float, default=0.03, help="Spline smoothing scale in mas.")
    parser.add_argument("--max-empty", type=int, default=12, help="Stop a sector after this many empty radial steps.")
    parser.add_argument("--min-arc-points", type=int, default=8, help="Reject radial arc samples with fewer selected points.")
    parser.add_argument("--min-arc-span", type=float, default=2.0, help="Reject radial arc samples whose selected PA span is smaller than this.")
    parser.add_argument("--min-peak-over-threshold", type=float, default=1.05, help="Reject radial arc samples whose peak is too close to the threshold.")
    parser.add_argument("--max-sample-step-factor", type=float, default=5.0, help="Reject raw ridge jumps longer than this times the radial step/gap.")
    parser.add_argument("--max-sample-pa-jump", type=float, default=45.0, help="Reject raw ridge PA jumps larger than this after the inner core.")
    parser.add_argument("--output", type=Path, default=None, help="Output prefix. Sector name is appended.")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-figure", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--stretch", choices=("linear", "sqrt", "log", "asinh"), default="asinh")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--contour-sigma", type=float, default=3.0)
    parser.add_argument("--contour-factor", type=float, default=2.0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    app = PolarRidgeApp(args)
    app.build()
    if args.save:
        app.save()
    if args.save_figure is not None or not args.no_show:
        app.plot(save_path=args.save_figure, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
