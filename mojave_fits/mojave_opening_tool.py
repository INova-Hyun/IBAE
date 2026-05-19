#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from fits_viewer import (
    apply_stretch,
    auto_limits,
    image_edges_mas,
    pixel_to_mas,
    positive_contour_levels,
    read_primary_fits,
    robust_corner_rms,
    zoom_axes,
)
from polar_opening_angle import (
    measure_opening,
    records_to_json,
    save_outputs,
)
from polar_ridge_builder import (
    Sector,
    build_polar_samples,
    filter_polar_sample_outliers,
    pa_to_xy,
    result_payload,
    save_result,
    sector_width_deg,
    smooth_resample_polar_samples,
)


MAS_PER_DEG = 3600.0 * 1000.0


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


def _prefixed_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def parse_core(text: str) -> Tuple[float, float]:
    try:
        x, y = text.replace(" ", "").split(",", 1)
        return float(x), float(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--core expects x,y in mas") from exc


def parse_prefer(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    aliases = {
        "north": 0.0,
        "n": 0.0,
        "east": 90.0,
        "e": 90.0,
        "south": 180.0,
        "s": 180.0,
        "west": 270.0,
        "w": 270.0,
        "ne": 45.0,
        "nw": 315.0,
        "se": 135.0,
        "sw": 225.0,
    }
    if text in aliases:
        return aliases[text]
    return float(text) % 360.0


def circular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - float(b) + 180.0) % 360.0 - 180.0)


def image_radius_limit_mas(header: Dict[str, object], shape: Tuple[int, int], core_mas: Tuple[float, float]) -> float:
    x0, x1, y0, y1 = image_edges_mas(header, shape)
    corners = np.asarray([[x0, y0], [x0, y1], [x1, y0], [x1, y1]], dtype=float)
    center = np.asarray(core_mas, dtype=float)
    return float(np.nanmax(np.linalg.norm(corners - center, axis=1)))


def smooth_circular_hist(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return hist.astype(float)
    radius = int(max(2, math.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    padded = np.concatenate([hist[-radius:], hist, hist[:radius]])
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def estimate_pa_center(
    image: np.ndarray,
    header: Dict[str, object],
    core_mas: Tuple[float, float],
    threshold: float,
    rmin_mas: float,
    rmax_mas: float,
    bin_deg: float,
    smooth_deg: float,
    prefer_pa: Optional[float],
    prefer_search_deg: float,
) -> Tuple[float, Dict[str, object]]:
    yy, xx = np.indices(image.shape)
    x_mas, y_mas = pixel_to_mas(header, xx.astype(float), yy.astype(float))
    dx = x_mas - float(core_mas[0])
    dy = y_mas - float(core_mas[1])
    radius = np.sqrt(dx * dx + dy * dy)
    pa = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    finite = np.isfinite(image)
    mask = finite & (image > threshold) & (radius >= rmin_mas) & (radius <= rmax_mas)
    if np.count_nonzero(mask) < 10:
        return float(prefer_pa if prefer_pa is not None else 90.0), {
            "status": "fallback_too_few_pixels",
            "n_pixels": int(np.count_nonzero(mask)),
        }

    bins = np.arange(0.0, 360.0 + bin_deg, bin_deg)
    centers = 0.5 * (bins[:-1] + bins[1:])
    weights = np.clip(image[mask] - threshold, 0.0, None) * np.sqrt(np.maximum(radius[mask], 0.0))
    hist, _ = np.histogram(pa[mask], bins=bins, weights=weights)
    smooth = smooth_circular_hist(hist.astype(float), max(0.0, smooth_deg / max(bin_deg, 1e-12)))
    if not np.any(np.isfinite(smooth)) or np.nanmax(smooth) <= 0:
        return float(prefer_pa if prefer_pa is not None else 90.0), {
            "status": "fallback_empty_hist",
            "n_pixels": int(np.count_nonzero(mask)),
        }

    if prefer_pa is not None:
        dist = circular_distance_deg(centers, prefer_pa)
        candidate = dist <= prefer_search_deg
        if np.any(candidate):
            local_indices = np.flatnonzero(candidate)
            peak_index = int(local_indices[np.nanargmax(smooth[local_indices])])
        else:
            peak_index = int(np.nanargmin(dist))
    else:
        peak_index = int(np.nanargmax(smooth))
    center = float(centers[peak_index] % 360.0)
    return center, {
        "status": "ok",
        "n_pixels": int(np.count_nonzero(mask)),
        "bin_deg": float(bin_deg),
        "smooth_deg": float(smooth_deg),
        "prefer_pa_deg": None if prefer_pa is None else float(prefer_pa),
        "center_pa_deg": center,
        "peak_hist": float(smooth[peak_index]),
    }


def sector_from_center(name: str, center_pa_deg: float, width_deg: float) -> Sector:
    half = 0.5 * float(width_deg)
    return Sector(name, float((center_pa_deg - half) % 360.0), float((center_pa_deg + half) % 360.0))


def sector_midpoint(sector: Sector) -> float:
    return float((sector.pa_min_deg + 0.5 * sector_width_deg(sector.pa_min_deg, sector.pa_max_deg)) % 360.0)


def finite_positive_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def apply_width_axis_range(ax, args: argparse.Namespace) -> None:
    x_low = finite_positive_or_none(getattr(args, "width_sep_min", None))
    x_high = finite_positive_or_none(getattr(args, "width_sep_max", None))
    if x_low is not None or x_high is not None:
        current_low, current_high = ax.get_xlim()
        low = x_low if x_low is not None else current_low
        high = x_high if x_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and low > 0.0 and high > low:
            ax.set_xlim(low, high)

    y_low = finite_positive_or_none(getattr(args, "width_value_min", None))
    y_high = finite_positive_or_none(getattr(args, "width_value_max", None))
    if y_low is not None or y_high is not None:
        current_low, current_high = ax.get_ylim()
        low = y_low if y_low is not None else current_low
        high = y_high if y_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and low > 0.0 and high > low:
            ax.set_ylim(low, high)


def apply_angle_axis_range(ax, args: argparse.Namespace) -> None:
    x_low = finite_or_none(getattr(args, "angle_plot_sep_min", None))
    x_high = finite_or_none(getattr(args, "angle_plot_sep_max", None))
    if x_low is not None or x_high is not None:
        current_low, current_high = ax.get_xlim()
        low = x_low if x_low is not None else current_low
        high = x_high if x_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and high > low:
            ax.set_xlim(low, high)

    y_low = finite_or_none(getattr(args, "angle_plot_value_min", None))
    y_high = finite_or_none(getattr(args, "angle_plot_value_max", None))
    if y_low is not None or y_high is not None:
        current_low, current_high = ax.get_ylim()
        low = y_low if y_low is not None else current_low
        high = y_high if y_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and high > low:
            ax.set_ylim(low, high)


def opening_args_from_app(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        angle_rmin=args.analysis_sep_min,
        analysis_sep_min=args.analysis_sep_min,
        analysis_sep_max=args.analysis_sep_max,
        slice_half_width=args.slice_half_width,
        sample_step=args.sample_step,
        fit_threshold_snr=args.fit_threshold_snr,
        fit_threshold=args.fit_threshold,
        fit_padding=args.fit_padding,
        baseline_guard=getattr(args, "baseline_guard", None),
        min_fit_points=args.min_fit_points,
        sigma_upper=args.sigma_upper,
        pa_sweep=args.pa_sweep,
        pa_sweep_range=args.pa_sweep_range,
        pa_sweep_step=args.pa_sweep_step,
        pa_sweep_workers=args.pa_sweep_workers,
        pa_sweep_analysis_only=args.pa_sweep_analysis_only,
        max_records=None,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
    )


def save_payload_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def choose_fits_file(initial_dir: Optional[Path]) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"FITS file dialog is unavailable: {exc}")
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        selected = filedialog.askopenfilename(
            title="Open FITS image",
            initialdir=str(initial_dir or Path.cwd()),
            filetypes=[
                ("FITS files", "*.fits *.fit *.fts *.fits.gz *.fit.gz *.fts.gz"),
                ("All files", "*.*"),
            ],
        )
    except Exception as exc:
        print(f"FITS file dialog failed: {exc}")
        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
    return Path(selected) if selected else None


def default_fits_dir() -> Path:
    candidates = [
        Path.cwd() / "data" / "mojave_fits",
        Path.cwd() / "data_files",
        Path(__file__).resolve().parents[1] / "data" / "mojave_fits",
    ]
    for data_dir in candidates:
        if data_dir.exists():
            return data_dir
    return Path.cwd()


def resolve_fits_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path
    for data_dir in (
        Path.cwd() / "data" / "mojave_fits",
        Path.cwd() / "data_files",
        Path(__file__).resolve().parents[1] / "data" / "mojave_fits",
    ):
        data_candidate = data_dir / path
        if data_candidate.exists():
            return data_candidate
    return path


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


class MojaveOpeningTool:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.fits_path: Optional[Path] = None
        self.image: Optional[np.ndarray] = None
        self.header: Dict[str, object] = {}
        self.rms = float("nan")
        self.beam_mas = float("nan")
        self.pixel_mas = float("nan")
        self.core_mas = args.core
        self.threshold = float("nan")
        self.output_prefix: Optional[Path] = args.output
        self.prefer_pa = parse_prefer(args.prefer)
        self.auto_info: Dict[str, object] = {}
        self.sector: Optional[Sector] = None
        self.ridge_payload: Optional[Dict[str, object]] = None
        self.records = []
        self.summary: Optional[Dict[str, object]] = None
        self.opening_payload: Optional[Dict[str, object]] = None

        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5.8))
        self.fig.subplots_adjust(left=0.055, right=0.985, bottom=0.11, top=0.84, wspace=0.28)
        try:
            self.fig.canvas.manager.set_window_title("MOJAVE Opening Tool")
        except AttributeError:
            pass
        self.buttons = []
        self.create_buttons()
        self.reset_xlim = None
        self.reset_ylim = None
        self.full_xlim = None
        self.full_ylim = None
        self.connect()
        if args.fits is not None:
            self.load_fits(args.fits)
            self.redraw()
        else:
            self.redraw()
        self.print_help()

    def create_buttons(self) -> None:
        specs = [
            ("Open FITS", 0.055, 0.090, self.open_fits_dialog),
            ("Analyze", 0.152, 0.075, lambda _event: self.recompute()),
            ("Save", 0.234, 0.055, lambda _event: self.save_all()),
            ("Auto PA", 0.296, 0.070, lambda _event: self.use_global_auto_pa()),
            ("Opposite", 0.373, 0.075, lambda _event: self.use_opposite_sector()),
        ]
        for label, x, width, callback in specs:
            ax = self.fig.add_axes([x, 0.905, width, 0.052])
            button = Button(ax, label, hovercolor="0.85")
            button.on_clicked(callback)
            self.buttons.append(button)

    def load_fits(self, path: Path) -> bool:
        path = resolve_fits_path(path)
        if not path.exists():
            print(f"FITS file does not exist: {path}")
            return False
        self.fits_path = path
        self.image, self.header = read_primary_fits(path)
        self.rms = robust_corner_rms(self.image)
        self.beam_mas = _beam_mas(self.header)
        self.pixel_mas = _pixel_mas(self.header)
        self.threshold = self.args.threshold if self.args.threshold is not None else self.args.threshold_snr * self.rms
        self.output_prefix = self.args.output or path.with_name(f"{_fits_stem(path)}_opening_tool")
        self.auto_info = {}
        self.sector = self.auto_sector(self.prefer_pa)
        self.ridge_payload = None
        self.records = []
        self.summary = None
        self.opening_payload = None
        self.reset_xlim = None
        self.reset_ylim = None
        self.full_xlim = None
        self.full_ylim = None
        print(f"loaded FITS: {path}")
        return True

    def open_fits_dialog(self, _event=None) -> None:
        initial_dir = self.fits_path.parent if self.fits_path is not None else default_fits_dir()
        selected = choose_fits_file(initial_dir)
        if selected is None:
            return
        if self.load_fits(selected):
            self.redraw()

    def auto_rmax(self) -> float:
        if self.image is None:
            return float("nan")
        if self.args.rmax is not None:
            return float(self.args.rmax)
        return image_radius_limit_mas(self.header, self.image.shape, self.core_mas)

    def auto_sector(self, prefer_pa: Optional[float]) -> Sector:
        if self.image is None:
            return sector_from_center(self.args.sector_name, prefer_pa if prefer_pa is not None else 90.0, self.args.sector_width)
        center, info = estimate_pa_center(
            self.image,
            self.header,
            self.core_mas,
            self.threshold,
            self.args.auto_pa_rmin,
            self.args.auto_pa_rmax if self.args.auto_pa_rmax is not None else self.auto_rmax(),
            self.args.auto_pa_bin,
            self.args.auto_pa_smooth,
            prefer_pa,
            self.args.prefer_search,
        )
        self.auto_info = info
        return sector_from_center(self.args.sector_name, center, self.args.sector_width)

    def recompute(self) -> None:
        if self.image is None or self.fits_path is None:
            print("No FITS loaded. Use the Open FITS button or press l.")
            self.redraw()
            return
        if self.sector is None:
            self.sector = self.auto_sector(self.prefer_pa)
        rmax = self.auto_rmax()
        samples = build_polar_samples(
            self.image,
            self.header,
            self.sector,
            self.core_mas,
            self.args.rmin,
            rmax,
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
        ridge = smooth_resample_polar_samples(filtered_samples, self.core_mas, self.args.step, self.args.smooth)
        params = {
            "rmin_mas": self.args.rmin,
            "rmax_mas": rmax,
            "r_step_mas": self.args.step,
            "pa_step_deg": self.args.pa_step,
            "ridge_search_sep_min_mas": self.args.rmin,
            "ridge_search_sep_max_mas": rmax,
            "ridge_search_sep_step_mas": self.args.step,
            "arc_pa_sample_step_deg": self.args.pa_step,
            "threshold_snr": self.args.threshold_snr,
            "ridge_threshold_snr": self.args.threshold_snr,
            "threshold": self.threshold,
            "component_mode": self.args.component,
            "smooth_mas": self.args.smooth,
            "max_empty": self.args.max_empty,
            "min_arc_points": self.args.min_arc_points,
            "min_arc_span_deg": self.args.min_arc_span,
            "min_peak_over_threshold": self.args.min_peak_over_threshold,
            "max_sample_step_factor": self.args.max_sample_step_factor,
            "max_sample_pa_jump_deg": self.args.max_sample_pa_jump,
            "raw_sample_count_before_outlier_filter": len(samples),
            "outlier_filtered_count": max(0, len(samples) - len(filtered_samples)),
            "method": "integrated auto-sector core-centered polar arc flux-median ridgeline",
            "auto_pa": self.auto_info,
        }
        self.ridge_payload = result_payload(
            self.fits_path,
            self.header,
            self.image,
            self.rms,
            self.threshold,
            self.core_mas,
            self.sector,
            filtered_samples,
            ridge,
            params,
        )
        self.records, self.summary = measure_opening(self.image, self.header, self.ridge_payload, opening_args_from_app(self.args))
        self.opening_payload = {
            "format": "mojave_integrated_opening_tool",
            "version": 1,
            "fits_file": str(self.fits_path),
            "ridge_sector": self.ridge_payload.get("sector", {}),
            "summary": self.summary,
            "records": records_to_json(self.records),
        }
        self.redraw()
        self.print_summary()

    def print_help(self) -> None:
        print(
            "Keys: l/Open FITS=load image, click image=set preferred PA+recompute, a=auto PA global, o=opposite sector, "
            "left/right=rotate sector, [/]=shrink/expand width, m=recompute, s=save, "
            "mouse wheel/+/-=zoom, f=full view, r=reset, q=quit"
        )

    def print_summary(self) -> None:
        if not self.summary or self.sector is None:
            return
        fit5 = dict(self.summary.get("power_law_fit_smoothed5", {}))
        median_error = dict(self.summary.get("median_error", {}))
        err = dict(median_error.get("intrinsic_block", {}))
        err_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
        display_err = err_with_pa if np.isfinite(err_with_pa.get("sigma", float("nan"))) else err
        block_slices = median_error.get("block_size_slices", "n/a")
        block_length = median_error.get("block_length_mas", float("nan"))
        sweep = dict(self.summary.get("pa_sweep", {}))
        print(
            f"sector {self.sector.name}: PA {self.sector.pa_min_deg:.1f}..{self.sector.pa_max_deg:.1f} "
            f"(center {sector_midpoint(self.sector):.1f}), "
            f"ridge={len(self.ridge_payload.get('ridge_points', [])) if self.ridge_payload else 0}, "
            f"fits={self.summary['fit_success_count']}/{self.summary['record_count']}, "
            f"median={self.summary['median_opening_angle_deg']:.4g} +/- {display_err.get('sigma', float('nan')):.3g} deg "
            f"(block {block_slices} slices={block_length:.3g} mas), "
            f"raw={self.summary['median_opening_angle_raw_deg']:.4g} deg, "
            f"k={fit5.get('k', float('nan')):.4g}, "
            f"PA-sweep angle rms={sweep.get('median_intrinsic_opening_angle_rms_deg', float('nan')):.3g} deg"
        )

    def connect(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def draw_image_panel(self, ax) -> None:
        if self.image is None or self.fits_path is None or self.sector is None:
            ax.axis("off")
            ax.text(0.5, 0.55, "Open a FITS image", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.text(0.5, 0.45, "Use the Open FITS button or press l", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return
        finite = self.image[np.isfinite(self.image)]
        vmin = max(float(np.nanpercentile(finite, 1.0)), -3.0 * self.rms)
        vmax = float(np.nanpercentile(finite, 99.9))
        display = apply_stretch(self.image, "asinh", vmin, vmax)
        extent = image_edges_mas(self.header, self.image.shape)
        ax.imshow(display, origin="lower", extent=extent, cmap=self.args.cmap, interpolation="nearest")
        levels = positive_contour_levels(self.rms, float(np.nanmax(self.image)), self.args.contour_sigma, self.args.contour_factor)
        if levels.size:
            ax.contour(self.image, levels=levels, colors="white", linewidths=0.45, alpha=0.65, origin="lower", extent=extent)
        ax.scatter([self.core_mas[0]], [self.core_mas[1]], marker="+", s=110, c="#00e5ff", linewidths=1.8, zorder=7)

        radius = self.auto_rmax()
        width = sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg)
        pa_arc = self.sector.pa_min_deg + np.linspace(0.0, width, 160)
        x_arc, y_arc = pa_to_xy(self.core_mas, radius, pa_arc)
        ax.plot(x_arc, y_arc, lw=0.8, color="#00e5ff", alpha=0.65)
        for pa in (self.sector.pa_min_deg, self.sector.pa_min_deg + width):
            xe, ye = pa_to_xy(self.core_mas, radius, np.asarray([pa], dtype=float))
            ax.plot([self.core_mas[0], xe[0]], [self.core_mas[1], ye[0]], lw=0.8, color="#00e5ff", alpha=0.65)

        if self.ridge_payload:
            raw = np.asarray([(p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get("raw_samples", [])], dtype=float)
            if raw.size:
                ax.scatter(raw[:, 0], raw[:, 1], s=8, c="#ffdf5d", edgecolors="none", alpha=0.75, zorder=8)
            ridge = np.asarray([(p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get("ridge_points", [])], dtype=float)
            if ridge.size:
                ax.plot(ridge[:, 0], ridge[:, 1], color="#22ff66", lw=1.8, zorder=9)

        ax.set_xlabel("RA offset (mas)")
        ax.set_ylabel("Dec offset (mas)")
        ax.set_title(
            f"{self.fits_path.name}\n"
            f"PA {self.sector.pa_min_deg:.1f}..{self.sector.pa_max_deg:.1f}, "
            f"beam={self.beam_mas:.3f} mas"
        )
        if self.reset_xlim is None or self.reset_ylim is None:
            limits = None if self.args.full_view else auto_limits(self.image, self.header, self.rms, 0.005, 8.0, 2.0)
            if limits is not None:
                xmin, xmax, ymin, ymax = limits
                ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
                ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
            if self.args.fov is not None:
                half = float(self.args.fov)
                ax.set_xlim(half, -half)
                ax.set_ylim(-half, half)
            if self.ridge_payload:
                all_xy = []
                for key in ("raw_samples", "ridge_points"):
                    all_xy.extend((p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get(key, []))
                if all_xy:
                    expand_axes_to_points(ax, np.asarray(all_xy, dtype=float), padding_mas=max(0.5, self.beam_mas))
            self.reset_xlim = ax.get_xlim()
            self.reset_ylim = ax.get_ylim()
            self.full_xlim = image_edges_mas(self.header, self.image.shape)[:2]
            self.full_ylim = image_edges_mas(self.header, self.image.shape)[2:]

    def draw_width_panel(self, ax) -> None:
        if not self.records:
            ax.axis("off")
            ax.set_title("No fit records")
            return
        path = np.asarray([r.path_mas for r in self.records], dtype=float)
        raw = np.asarray([r.fit.fwhm_mas for r in self.records], dtype=float)
        deconv = np.asarray([r.intrinsic_fwhm_mas for r in self.records], dtype=float)
        raw_err = np.asarray([r.pa_sweep_raw_rms_mas for r in self.records], dtype=float)
        deconv_err = np.asarray([r.pa_sweep_intrinsic_rms_mas for r in self.records], dtype=float)
        success = np.asarray([r.fit.success for r in self.records], dtype=bool)
        ax.plot(path[success], raw[success], ".", ms=3, label="raw D")
        ax.plot(path[success], deconv[success], ".", ms=3, label="deconvolved d")
        raw_err_mask = success & np.isfinite(path) & np.isfinite(raw) & (raw > 0.0) & np.isfinite(raw_err) & (raw_err > 0.0)
        if np.any(raw_err_mask):
            ax.errorbar(
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
        deconv_err_mask = success & np.isfinite(path) & np.isfinite(deconv) & (deconv > 0.0) & np.isfinite(deconv_err) & (deconv_err > 0.0)
        if np.any(deconv_err_mask):
            ax.errorbar(
                path[deconv_err_mask],
                deconv[deconv_err_mask],
                yerr=deconv_err[deconv_err_mask],
                fmt="none",
                ecolor="#ff7f0e",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="deconv PA sweep rms",
            )
        failed = (~success) & np.isfinite(raw) & (raw > 0)
        if np.any(failed):
            ax.plot(path[failed], raw[failed], "x", ms=4, color="0.55", label="rejected fit")
        ax.axhline(self.beam_mas, color="0.5", lw=1.0, ls="--", label="beam")
        fit5 = dict((self.summary or {}).get("power_law_fit_smoothed5", {}))
        if np.isfinite(fit5.get("k", float("nan"))) and fit5.get("n", 0) >= 2:
            rline = np.logspace(math.log10(float(fit5["r_min_mas"])), math.log10(float(fit5["r_max_mas"])), 160)
            dline = float(fit5["amplitude_mas_at_1mas"]) * np.power(rline, float(fit5["k"]))
            ax.plot(rline, dline, color="#d62828", lw=1.5, label=f"d=A r^k, k={fit5['k']:.3g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_width_axis_range(ax, self.args)
        ax.set_xlabel("Separation along ridgeline (mas)")
        ax.set_ylabel("Gaussian width (mas)")
        ax.legend(fontsize=8)
        ax.set_title("Gaussian widths")

    def draw_angle_panel(self, ax) -> None:
        if not self.records or not self.summary:
            ax.axis("off")
            ax.set_title("No opening-angle result")
            return
        path = np.asarray([r.path_mas for r in self.records], dtype=float)
        raw = np.asarray([r.opening_angle_raw_deg for r in self.records], dtype=float)
        deconv = np.asarray([r.opening_angle_deg for r in self.records], dtype=float)
        raw_err = np.asarray([r.pa_sweep_opening_angle_raw_rms_deg for r in self.records], dtype=float)
        deconv_err = np.asarray([r.pa_sweep_opening_angle_intrinsic_rms_deg for r in self.records], dtype=float)
        ax.plot(path, raw, ".", ms=3, alpha=0.5, label="raw")
        ax.plot(path, deconv, ".", ms=3, label="deconvolved")
        raw_err_mask = np.isfinite(path) & np.isfinite(raw) & np.isfinite(raw_err) & (raw_err > 0.0)
        if np.any(raw_err_mask):
            ax.errorbar(
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
        deconv_err_mask = np.isfinite(path) & np.isfinite(deconv) & np.isfinite(deconv_err) & (deconv_err > 0.0)
        if np.any(deconv_err_mask):
            ax.errorbar(
                path[deconv_err_mask],
                deconv[deconv_err_mask],
                yerr=deconv_err[deconv_err_mask],
                fmt="none",
                ecolor="#ff7f0e",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="deconv PA sweep rms",
            )
        ax.axvline(float(self.summary["analysis_separation_min_mas"]), color="0.5", lw=1.0, ls="--")
        if self.summary.get("analysis_separation_max_mas") is not None:
            ax.axvline(float(self.summary["analysis_separation_max_mas"]), color="0.5", lw=1.0, ls=":")
        median = float(self.summary["median_opening_angle_deg"])
        ax.axhline(median, color="#e63946", lw=1.2, label="median d")
        median_error = dict(self.summary.get("median_error", {}))
        err = dict(median_error.get("intrinsic_block", {}))
        err_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
        title_err = err_with_pa if np.isfinite(err_with_pa.get("sigma", float("nan"))) else err
        fit5 = dict(self.summary.get("power_law_fit_smoothed5", {}))
        ax.set_xlabel("Separation along ridgeline (mas)")
        ax.set_ylabel("Full apparent opening angle (deg)")
        apply_angle_axis_range(ax, self.args)
        ax.legend(fontsize=8)
        ax.set_title(f"median={median:.3g} +/- {title_err.get('sigma', float('nan')):.2g} deg; k={fit5.get('k', float('nan')):.3g}")

    def redraw(self) -> None:
        current_xlim = self.axes[0].get_xlim() if self.reset_xlim is not None else None
        current_ylim = self.axes[0].get_ylim() if self.reset_ylim is not None else None
        for ax in self.axes:
            ax.clear()
        self.draw_image_panel(self.axes[0])
        if current_xlim is not None and current_ylim is not None:
            self.axes[0].set_xlim(current_xlim)
            self.axes[0].set_ylim(current_ylim)
        self.draw_width_panel(self.axes[1])
        self.draw_angle_panel(self.axes[2])
        self.fig.canvas.draw_idle()

    def set_preferred_from_click(self, x: float, y: float) -> None:
        if self.image is None:
            return
        dx = float(x) - float(self.core_mas[0])
        dy = float(y) - float(self.core_mas[1])
        if dx == 0 and dy == 0:
            return
        self.prefer_pa = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        self.sector = self.auto_sector(self.prefer_pa)
        self.recompute()

    def rotate_sector(self, delta_deg: float) -> None:
        if self.sector is None:
            return
        width = sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg)
        center = (sector_midpoint(self.sector) + delta_deg) % 360.0
        self.sector = sector_from_center(self.sector.name, center, width)
        self.recompute()

    def resize_sector(self, delta_width_deg: float) -> None:
        if self.sector is None:
            return
        center = sector_midpoint(self.sector)
        width = float(np.clip(sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg) + delta_width_deg, 10.0, 180.0))
        self.sector = sector_from_center(self.sector.name, center, width)
        self.recompute()

    def use_global_auto_pa(self) -> None:
        self.prefer_pa = None
        self.sector = self.auto_sector(None)
        self.recompute()

    def use_opposite_sector(self) -> None:
        if self.sector is None:
            return
        self.sector = sector_from_center(
            self.sector.name,
            sector_midpoint(self.sector) + 180.0,
            sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg),
        )
        self.recompute()

    def save_all(self) -> None:
        if self.image is None or self.fits_path is None or self.output_prefix is None:
            print("No FITS loaded. Nothing to save.")
            return
        if not self.ridge_payload or not self.summary or self.opening_payload is None:
            self.recompute()
        if not self.ridge_payload or not self.summary or self.opening_payload is None:
            return
        ridge_prefix = self.output_prefix.parent / f"{self.output_prefix.name}_ridge"
        opening_prefix = self.output_prefix.parent / f"{self.output_prefix.name}_opening_{self.sector.name}"
        ridge_json, ridge_csv = save_result(ridge_prefix, self.ridge_payload)
        opening_json, opening_csv = save_outputs(opening_prefix, self.opening_payload, self.records)
        fig_path = _prefixed_path(self.output_prefix, f"_{self.sector.name}.png")
        button_states = [(button.ax, button.ax.get_visible()) for button in self.buttons]
        for ax, _visible in button_states:
            ax.set_visible(False)
        try:
            self.fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        finally:
            for ax, visible in button_states:
                ax.set_visible(visible)
        print(f"saved ridge:   {ridge_json}")
        print(f"saved ridge csv: {ridge_csv}")
        print(f"saved opening: {opening_json}")
        print(f"saved opening csv: {opening_csv}")
        print(f"saved figure:  {fig_path}")

    def on_click(self, event) -> None:
        if event.inaxes is self.axes[0] and event.xdata is not None and event.ydata is not None and event.button == 1:
            self.set_preferred_from_click(float(event.xdata), float(event.ydata))

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.axes[0]:
            return
        center = None
        if event.xdata is not None and event.ydata is not None:
            center = (float(event.xdata), float(event.ydata))
        zoom_axes(self.axes[0], 1.0 / 1.25 if event.button == "up" else 1.25, center)
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "q":
            plt.close(self.fig)
        elif key == "h":
            self.print_help()
        elif key in ("l", "ctrl+o"):
            self.open_fits_dialog()
        elif key == "m":
            self.recompute()
        elif key == "s":
            self.save_all()
        elif key == "a":
            self.use_global_auto_pa()
        elif key == "o":
            self.use_opposite_sector()
        elif key == "left":
            self.rotate_sector(-5.0)
        elif key == "right":
            self.rotate_sector(5.0)
        elif key == "[":
            self.resize_sector(-5.0)
        elif key == "]":
            self.resize_sector(5.0)
        elif key in ("+", "="):
            zoom_axes(self.axes[0], 1.0 / 1.25)
            self.fig.canvas.draw_idle()
        elif key in ("-", "_"):
            zoom_axes(self.axes[0], 1.25)
            self.fig.canvas.draw_idle()
        elif key == "f" and self.full_xlim is not None and self.full_ylim is not None:
            self.axes[0].set_xlim(self.full_xlim)
            self.axes[0].set_ylim(self.full_ylim)
            self.fig.canvas.draw_idle()
        elif key == "r" and self.reset_xlim is not None and self.reset_ylim is not None:
            self.axes[0].set_xlim(self.reset_xlim)
            self.axes[0].set_ylim(self.reset_ylim)
            self.fig.canvas.draw_idle()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Integrated MOJAVE FITS polar-ridge and opening-angle tool.")
    parser.add_argument("fits", nargs="?", type=Path, default=None, help="Optional FITS image path. If omitted, open one from the GUI.")
    parser.add_argument("--prefer", default=None, help="Recommended direction: east/west/north/south/ne/... or PA degrees. Default: global auto PA.")
    parser.add_argument("--sector-name", default="side1")
    parser.add_argument("--sector-width", type=float, default=80.0)
    parser.add_argument("--prefer-search", type=float, default=90.0)
    parser.add_argument("--core", type=parse_core, default=(0.0, 0.0))
    parser.add_argument("--threshold-snr", "--ridge-threshold-snr", dest="threshold_snr", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--auto-pa-rmin", type=float, default=0.5)
    parser.add_argument("--auto-pa-rmax", type=float, default=None)
    parser.add_argument("--auto-pa-bin", type=float, default=1.0)
    parser.add_argument("--auto-pa-smooth", type=float, default=5.0)
    parser.add_argument("--rmin", "--ridge-search-sep-min", dest="rmin", type=float, default=0.05)
    parser.add_argument("--rmax", "--ridge-search-sep-max", dest="rmax", type=float, default=None)
    parser.add_argument("--step", "--ridge-search-sep-step", dest="step", type=float, default=0.05)
    parser.add_argument("--pa-step", "--arc-pa-sample-step", dest="pa_step", type=float, default=0.25)
    parser.add_argument("--component", choices=("peak", "largest", "all"), default="all")
    parser.add_argument("--smooth", type=float, default=0.03)
    parser.add_argument("--max-empty", type=int, default=12)
    parser.add_argument("--min-arc-points", type=int, default=8)
    parser.add_argument("--min-arc-span", type=float, default=2.0)
    parser.add_argument("--min-peak-over-threshold", type=float, default=1.05)
    parser.add_argument("--max-sample-step-factor", type=float, default=5.0)
    parser.add_argument("--max-sample-pa-jump", type=float, default=45.0)
    parser.add_argument("--analysis-sep-min", type=float, default=0.5)
    parser.add_argument("--analysis-sep-max", type=float, default=None)
    parser.add_argument("--angle-rmin", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--width-sep-min", "--width-x-min", dest="width_sep_min", type=float, default=None)
    parser.add_argument("--width-sep-max", "--width-x-max", dest="width_sep_max", type=float, default=None)
    parser.add_argument("--width-value-min", "--width-y-min", dest="width_value_min", type=float, default=None)
    parser.add_argument("--width-value-max", "--width-y-max", dest="width_value_max", type=float, default=None)
    parser.add_argument("--angle-plot-sep-min", "--angle-x-min", dest="angle_plot_sep_min", type=float, default=None)
    parser.add_argument("--angle-plot-sep-max", "--angle-x-max", dest="angle_plot_sep_max", type=float, default=None)
    parser.add_argument("--angle-plot-value-min", "--angle-y-min", dest="angle_plot_value_min", type=float, default=None)
    parser.add_argument("--angle-plot-value-max", "--angle-y-max", dest="angle_plot_value_max", type=float, default=None)
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
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260430)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save", action="store_true", help="Save after initial computation.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--legacy-mpl", action="store_true", help="Use the older Matplotlib-only window instead of the PyQt5 UI.")
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--contour-sigma", type=float, default=3.0)
    parser.add_argument("--contour-factor", type=float, default=2.0)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.angle_rmin is not None:
        args.analysis_sep_min = float(args.angle_rmin)
    if args.no_show and args.fits is None:
        parser.error("a FITS path is required when --no-show is used")
    if not args.no_show and not args.legacy_mpl:
        try:
            from mojave_opening_tool_qt import run_qt_tool

            return run_qt_tool(args)
        except Exception as exc:
            print(f"PyQt5 UI could not start; falling back to Matplotlib UI: {exc}")
    tool = MojaveOpeningTool(args)
    if args.save:
        tool.save_all()
    if not args.no_show:
        plt.show()
    else:
        plt.close(tool.fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
