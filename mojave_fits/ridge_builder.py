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
    mas_to_pixel,
    positive_contour_levels,
    read_primary_fits,
    robust_corner_rms,
    zoom_axes,
)


@dataclass
class RidgeSample:
    guide_path_mas: float
    x_mas: float
    y_mas: float
    offset_mas: float
    peak: float
    weight_sum: float
    n_used: int


@dataclass
class RidgePoint:
    path_mas: float
    x_mas: float
    y_mas: float
    pa_deg: float


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
    if not finite:
        return 0.05
    return float(np.median(finite))


def _default_prefix(fits_path: Path) -> Path:
    name = fits_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".fits"):
        name = name[:-5]
    return fits_path.with_name(f"{name}_ridgeline")


def _prefixed_path(prefix: Path, extension: str) -> Path:
    return prefix.parent / f"{prefix.name}{extension}"


def parse_points(text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not text.strip():
        return points
    for chunk in text.replace(" ", "").split(";"):
        if not chunk:
            continue
        try:
            x_str, y_str = chunk.split(",", 1)
            points.append((float(x_str), float(y_str)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid point '{chunk}', expected x,y;x,y") from exc
    return points


def sample_image_bilinear(image: np.ndarray, header: Dict[str, object], x_mas: np.ndarray, y_mas: np.ndarray) -> np.ndarray:
    x_pix, y_pix = mas_to_pixel(header, x_mas, y_mas)
    x0 = np.floor(x_pix).astype(int)
    y0 = np.floor(y_pix).astype(int)
    dx = x_pix - x0
    dy = y_pix - y0

    out = np.full(np.shape(x_mas), np.nan, dtype=float)
    valid = (x0 >= 0) & (x0 + 1 < image.shape[1]) & (y0 >= 0) & (y0 + 1 < image.shape[0])
    if not np.any(valid):
        return out

    x0v = x0[valid]
    y0v = y0[valid]
    dxv = dx[valid]
    dyv = dy[valid]
    v00 = image[y0v, x0v]
    v10 = image[y0v, x0v + 1]
    v01 = image[y0v + 1, x0v]
    v11 = image[y0v + 1, x0v + 1]
    out[valid] = (
        v00 * (1.0 - dxv) * (1.0 - dyv)
        + v10 * dxv * (1.0 - dyv)
        + v01 * (1.0 - dxv) * dyv
        + v11 * dxv * dyv
    )
    return out


def polyline_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def polyline_position(points: np.ndarray, cumulative: np.ndarray, distance: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError("At least two guide points are required.")
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))
    seg_len = cumulative[idx + 1] - cumulative[idx]
    if seg_len <= 0:
        tangent = np.array([1.0, 0.0], dtype=float)
        return points[idx].copy(), tangent
    frac = (distance - cumulative[idx]) / seg_len
    pos = points[idx] * (1.0 - frac) + points[idx + 1] * frac
    tangent = points[idx + 1] - points[idx]
    tangent = tangent / np.linalg.norm(tangent)
    return pos, tangent


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def choose_profile_segment(mask: np.ndarray, values: np.ndarray, center_index: int) -> np.ndarray:
    runs = contiguous_true_runs(mask)
    if not runs:
        return np.zeros_like(mask, dtype=bool)
    for start, stop in runs:
        if start <= center_index < stop:
            selected = np.zeros_like(mask, dtype=bool)
            selected[start:stop] = True
            return selected
    finite_values = np.where(np.isfinite(values), values, -np.inf)
    peak_index = int(np.argmax(finite_values))
    for start, stop in runs:
        if start <= peak_index < stop:
            selected = np.zeros_like(mask, dtype=bool)
            selected[start:stop] = True
            return selected
    start, stop = max(runs, key=lambda r: r[1] - r[0])
    selected = np.zeros_like(mask, dtype=bool)
    selected[start:stop] = True
    return selected


def build_weighted_ridge_samples(
    image: np.ndarray,
    header: Dict[str, object],
    guide_points_mas: Sequence[Tuple[float, float]],
    core_mas: Tuple[float, float],
    step_mas: float,
    search_half_width_mas: float,
    profile_step_mas: float,
    threshold: float,
    start_mas: float,
) -> List[RidgeSample]:
    guide = np.asarray([core_mas, *guide_points_mas], dtype=float)
    if guide.shape[0] < 2:
        raise ValueError("Click at least one guide point downstream from the core.")
    cumulative = polyline_lengths(guide)
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("Guide path length is zero.")

    distances = np.arange(max(0.0, start_mas), total + 0.5 * step_mas, step_mas)
    offsets = np.arange(-search_half_width_mas, search_half_width_mas + 0.5 * profile_step_mas, profile_step_mas)
    center_index = int(np.argmin(np.abs(offsets)))
    samples: List[RidgeSample] = []

    for dist in distances:
        center, tangent = polyline_position(guide, cumulative, float(dist))
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        x_line = center[0] + offsets * normal[0]
        y_line = center[1] + offsets * normal[1]
        values = sample_image_bilinear(image, header, x_line, y_line)
        valid = np.isfinite(values) & (values > threshold)
        selected = choose_profile_segment(valid, values, center_index)
        if np.count_nonzero(selected) < 3:
            continue
        weights = np.clip(values[selected], 0.0, None)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        offset = float(np.sum(offsets[selected] * weights) / weight_sum)
        ridge_xy = center + offset * normal
        samples.append(
            RidgeSample(
                guide_path_mas=float(dist),
                x_mas=float(ridge_xy[0]),
                y_mas=float(ridge_xy[1]),
                offset_mas=offset,
                peak=float(np.nanmax(values[selected])),
                weight_sum=weight_sum,
                n_used=int(np.count_nonzero(selected)),
            )
        )
    return samples


def smooth_resample_ridge(
    samples: Sequence[RidgeSample],
    core_mas: Tuple[float, float],
    step_mas: float,
    smoothing_mas: float,
) -> List[RidgePoint]:
    points = [(float(core_mas[0]), float(core_mas[1]))]
    points.extend((s.x_mas, s.y_mas) for s in samples)
    raw = np.asarray(points, dtype=float)
    if raw.shape[0] < 2:
        return []

    cumulative = polyline_lengths(raw)
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-6])
    raw = raw[keep]
    cumulative = polyline_lengths(raw)
    if raw.shape[0] < 2 or cumulative[-1] <= 0:
        return []

    out_s = np.arange(0.0, cumulative[-1] + 0.5 * step_mas, step_mas)
    if raw.shape[0] >= 4:
        try:
            from scipy.interpolate import splprep, splev

            smooth = max(0.0, float(smoothing_mas)) ** 2 * raw.shape[0]
            tck, _ = splprep([raw[:, 0], raw[:, 1]], u=cumulative, s=smooth, k=min(3, raw.shape[0] - 1))
            x_out, y_out = splev(out_s, tck)
            resampled = np.column_stack([x_out, y_out])
        except Exception:
            resampled = np.column_stack(
                [np.interp(out_s, cumulative, raw[:, 0]), np.interp(out_s, cumulative, raw[:, 1])]
            )
    else:
        resampled = np.column_stack([np.interp(out_s, cumulative, raw[:, 0]), np.interp(out_s, cumulative, raw[:, 1])])

    if len(resampled):
        resampled[0] = np.asarray(core_mas, dtype=float)

    if len(resampled) == 1:
        pa = np.array([float("nan")])
    else:
        dx = np.gradient(resampled[:, 0], out_s, edge_order=1)
        dy = np.gradient(resampled[:, 1], out_s, edge_order=1)
        pa = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    return [
        RidgePoint(path_mas=float(s), x_mas=float(x), y_mas=float(y), pa_deg=float(p))
        for s, (x, y), p in zip(out_s, resampled, pa)
    ]


def ridge_payload(
    fits_path: Path,
    header: Dict[str, object],
    image: np.ndarray,
    rms: float,
    core_mas: Tuple[float, float],
    guide_points_mas: Sequence[Tuple[float, float]],
    samples: Sequence[RidgeSample],
    ridge: Sequence[RidgePoint],
    params: Dict[str, object],
) -> Dict[str, object]:
    return {
        "format": "mojave_fits_ridgeline",
        "version": 1,
        "fits_file": str(fits_path),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "bunit": str(header.get("BUNIT", "") or ""),
        "beam_mas": _beam_mas(header),
        "pixel_mas": _pixel_mas(header),
        "rms_jy_per_beam": float(rms),
        "core_mas": [float(core_mas[0]), float(core_mas[1])],
        "guide_points_mas": [[float(x), float(y)] for x, y in guide_points_mas],
        "parameters": params,
        "raw_samples": [
            {
                "guide_path_mas": s.guide_path_mas,
                "x_mas": s.x_mas,
                "y_mas": s.y_mas,
                "offset_mas": s.offset_mas,
                "peak": s.peak,
                "weight_sum": s.weight_sum,
                "n_used": s.n_used,
            }
            for s in samples
        ],
        "ridge_points": [
            {"path_mas": p.path_mas, "x_mas": p.x_mas, "y_mas": p.y_mas, "pa_deg": p.pa_deg} for p in ridge
        ],
    }


def save_ridge(prefix: Path, payload: Dict[str, object]) -> Tuple[Path, Path]:
    json_path = _prefixed_path(prefix, ".json")
    csv_path = _prefixed_path(prefix, ".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_mas", "x_mas", "y_mas", "pa_deg"])
        for point in payload["ridge_points"]:
            writer.writerow([point["path_mas"], point["x_mas"], point["y_mas"], point["pa_deg"]])
    return json_path, csv_path


def draw_base_image(
    ax,
    image: np.ndarray,
    header: Dict[str, object],
    rms: float,
    stretch: str,
    cmap: str,
    contour_sigma: float,
    contour_factor: float,
    show_contours: bool,
):
    finite = image[np.isfinite(image)]
    vmin = max(float(np.nanpercentile(finite, 1.0)), -3.0 * rms) if np.isfinite(rms) else float(np.nanpercentile(finite, 1.0))
    vmax = float(np.nanpercentile(finite, 99.9))
    display = apply_stretch(image, stretch, vmin, vmax)
    extent = image_edges_mas(header, image.shape)
    im = ax.imshow(display, origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
    contour_set = None
    if show_contours:
        levels = positive_contour_levels(rms, float(np.nanmax(image)), contour_sigma, contour_factor)
        if levels.size:
            contour_set = ax.contour(
                image,
                levels=levels,
                colors="white",
                linewidths=0.55,
                alpha=0.75,
                origin="lower",
                extent=extent,
            )
    return im, contour_set


class RidgeBuilderApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.fits_path = args.fits
        self.image, self.header = read_primary_fits(self.fits_path)
        self.rms = robust_corner_rms(self.image)
        self.core_mas = tuple(args.core)
        self.guide_points: List[Tuple[float, float]] = list(args.guide or [])
        self.samples: List[RidgeSample] = []
        self.ridge: List[RidgePoint] = []
        self.output_prefix = args.output or _default_prefix(self.fits_path)
        self.show_contours = not args.no_contours

        beam = _beam_mas(self.header)
        pix = _pixel_mas(self.header)
        self.search_half_width_mas = args.search_half_width
        if self.search_half_width_mas is None:
            self.search_half_width_mas = 2.0 * beam if np.isfinite(beam) and beam > 0 else 1.5
        self.profile_step_mas = args.profile_step or max(pix / 2.0, 0.01)
        self.threshold = args.threshold if args.threshold is not None else args.threshold_snr * self.rms

        self.fig, self.ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        self.im, self.contour_set = draw_base_image(
            self.ax,
            self.image,
            self.header,
            self.rms,
            args.stretch,
            args.cmap,
            args.contour_sigma,
            args.contour_factor,
            self.show_contours,
        )
        self.full_xlim = self.ax.get_xlim()
        self.full_ylim = self.ax.get_ylim()
        self.guide_line = None
        self.raw_artist = None
        self.ridge_artist = None
        self.core_artist = None
        self.status_text = self.ax.text(
            0.01,
            0.01,
            "",
            transform=self.ax.transAxes,
            color="white",
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 4},
        )

        self._configure_axes()
        self._connect()
        self.redraw_overlays()
        self.print_help()

    def _configure_axes(self) -> None:
        beam = _beam_mas(self.header)
        pix = _pixel_mas(self.header)
        peak = float(np.nanmax(self.image))
        bunit = str(self.header.get("BUNIT", "") or "")
        self.ax.set_title(
            f"{self.fits_path.name}\n"
            f"beam={beam:.3f} mas, pixel={pix:.4f} mas, peak={peak:.4g} {bunit}, rms~{self.rms:.3g}"
        )
        self.ax.set_xlabel("Relative RA offset (mas; positive left if CDELT1<0)")
        self.ax.set_ylabel("Relative Dec offset (mas)")
        limits = None
        if not self.args.no_auto_fov:
            limits = auto_limits(self.image, self.header, self.rms, 0.005, 8.0, 2.0)
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
            self.ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
            self.ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
        if self.args.fov is not None:
            half = float(self.args.fov)
            self.ax.set_xlim(half, -half)
            self.ax.set_ylim(-half, half)
        self.reset_xlim = self.ax.get_xlim()
        self.reset_ylim = self.ax.get_ylim()

    def _connect(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def print_help(self) -> None:
        print(
            "Ridge builder keys: left-click=add guide point, right-click/backspace=undo, "
            "mouse wheel/+/-=zoom, f=full view, a=auto ridge, s=save JSON/CSV, "
            "p=save PNG, c=contours, r=reset view, x=clear ridge, q=quit"
        )

    def build(self) -> None:
        self.samples = build_weighted_ridge_samples(
            self.image,
            self.header,
            self.guide_points,
            self.core_mas,
            self.args.step,
            self.search_half_width_mas,
            self.profile_step_mas,
            self.threshold,
            self.args.start,
        )
        self.ridge = smooth_resample_ridge(self.samples, self.core_mas, self.args.step, self.args.smooth)
        print(f"built ridge: guide_points={len(self.guide_points)}, raw_samples={len(self.samples)}, ridge_points={len(self.ridge)}")
        self.redraw_overlays()

    def payload(self) -> Dict[str, object]:
        params = {
            "step_mas": self.args.step,
            "start_mas": self.args.start,
            "search_half_width_mas": self.search_half_width_mas,
            "profile_step_mas": self.profile_step_mas,
            "threshold": self.threshold,
            "threshold_snr": self.args.threshold_snr,
            "smooth_mas": self.args.smooth,
            "method": "guide-polyline transverse weighted centroid",
        }
        return ridge_payload(
            self.fits_path,
            self.header,
            self.image,
            self.rms,
            self.core_mas,
            self.guide_points,
            self.samples,
            self.ridge,
            params,
        )

    def save(self) -> None:
        if not self.ridge:
            self.build()
        if not self.ridge:
            print("No ridge points to save.")
            return
        json_path, csv_path = save_ridge(self.output_prefix, self.payload())
        print(f"saved ridge: {json_path}")
        print(f"saved csv:   {csv_path}")

    def save_figure(self, path: Optional[Path] = None) -> Path:
        target = path or _prefixed_path(self.output_prefix, ".png")
        self.fig.savefig(target, dpi=180)
        print(f"saved preview: {target}")
        return target

    def redraw_overlays(self) -> None:
        for artist_name in ("guide_line", "raw_artist", "ridge_artist", "core_artist"):
            artist = getattr(self, artist_name)
            if artist is not None:
                artist.remove()
                setattr(self, artist_name, None)

        self.core_artist = self.ax.scatter(
            [self.core_mas[0]],
            [self.core_mas[1]],
            marker="+",
            s=120,
            c="#00e5ff",
            linewidths=1.8,
            zorder=5,
            label="core",
        )

        if self.guide_points:
            pts = np.asarray([self.core_mas, *self.guide_points], dtype=float)
            (self.guide_line,) = self.ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="#00e5ff",
                marker="o",
                ms=4,
                lw=1.4,
                alpha=0.95,
                zorder=6,
                label="guide",
            )

        if self.samples:
            raw = np.asarray([(s.x_mas, s.y_mas) for s in self.samples], dtype=float)
            self.raw_artist = self.ax.scatter(
                raw[:, 0],
                raw[:, 1],
                s=10,
                c="#ffdf5d",
                edgecolors="none",
                alpha=0.8,
                zorder=7,
                label="raw ridge samples",
            )

        if self.ridge:
            ridge = np.asarray([(p.x_mas, p.y_mas) for p in self.ridge], dtype=float)
            (self.ridge_artist,) = self.ax.plot(
                ridge[:, 0],
                ridge[:, 1],
                color="#22ff66",
                lw=2.0,
                alpha=0.95,
                zorder=8,
                label="smoothed ridge",
            )

        self.status_text.set_text(
            f"guide={len(self.guide_points)} raw={len(self.samples)} ridge={len(self.ridge)}\n"
            f"threshold={self.threshold:.3g}, search=+/-{self.search_half_width_mas:.2f} mas"
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            point = (float(event.xdata), float(event.ydata))
            self.guide_points.append(point)
            self.samples = []
            self.ridge = []
            print(f"added guide point: x={point[0]:.4f} mas, y={point[1]:.4f} mas")
            self.redraw_overlays()
        elif event.button == 3:
            self.undo()

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        factor = 1.0 / 1.25 if event.button == "up" else 1.25
        center = None
        if event.xdata is not None and event.ydata is not None:
            center = (float(event.xdata), float(event.ydata))
        zoom_axes(self.ax, factor, center)
        self.fig.canvas.draw_idle()

    def undo(self) -> None:
        if self.guide_points:
            removed = self.guide_points.pop()
            self.samples = []
            self.ridge = []
            print(f"removed guide point: x={removed[0]:.4f} mas, y={removed[1]:.4f} mas")
            self.redraw_overlays()

    def clear_ridge(self) -> None:
        self.samples = []
        self.ridge = []
        self.redraw_overlays()

    def toggle_contours(self) -> None:
        self.show_contours = not self.show_contours
        if self.contour_set is not None:
            self.contour_set.remove()
            self.contour_set = None
        if self.show_contours:
            levels = positive_contour_levels(self.rms, float(np.nanmax(self.image)), self.args.contour_sigma, self.args.contour_factor)
            if levels.size:
                self.contour_set = self.ax.contour(
                    self.image,
                    levels=levels,
                    colors="white",
                    linewidths=0.55,
                    alpha=0.75,
                    origin="lower",
                    extent=image_edges_mas(self.header, self.image.shape),
                )
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        if event.key in ("backspace", "delete", "u"):
            self.undo()
        elif event.key == "a":
            self.build()
        elif event.key == "s":
            self.save()
        elif event.key == "p":
            self.save_figure()
        elif event.key == "c":
            self.toggle_contours()
        elif event.key == "r":
            self.ax.set_xlim(self.reset_xlim)
            self.ax.set_ylim(self.reset_ylim)
            self.fig.canvas.draw_idle()
        elif event.key == "f":
            self.ax.set_xlim(self.full_xlim)
            self.ax.set_ylim(self.full_ylim)
            self.fig.canvas.draw_idle()
        elif event.key in ("+", "="):
            zoom_axes(self.ax, 1.0 / 1.25)
            self.fig.canvas.draw_idle()
        elif event.key in ("-", "_"):
            zoom_axes(self.ax, 1.25)
            self.fig.canvas.draw_idle()
        elif event.key == "x":
            self.guide_points = []
            self.clear_ridge()
            print("cleared guide/ridge")
        elif event.key == "h":
            self.print_help()
        elif event.key == "q":
            plt.close(self.fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Semi-automatic MOJAVE FITS ridgeline builder. Click guide points along one jet side, "
            "press 'a' to build, press 's' to save JSON/CSV."
        )
    )
    parser.add_argument("fits", nargs="?", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "mojave_fits" / "0238+711.u.stacked.icc.fits")
    parser.add_argument("--core", type=parse_points, default=[(0.0, 0.0)], help="Core as 'x,y' in mas. Default: 0,0")
    parser.add_argument("--guide", type=parse_points, default=[], help="Initial guide points, e.g. '2,1;5,2;8,4'")
    parser.add_argument("--output", type=Path, default=None, help="Output prefix. Default: FITS stem + _ridgeline")
    parser.add_argument("--step", type=float, default=0.05, help="Ridgeline sample spacing in mas.")
    parser.add_argument("--start", type=float, default=0.05, help="Start distance from core along guide path in mas.")
    parser.add_argument("--search-half-width", type=float, default=None, help="Transverse search half-width in mas.")
    parser.add_argument("--profile-step", type=float, default=None, help="Transverse profile sampling step in mas.")
    parser.add_argument("--threshold-snr", type=float, default=8.0)
    parser.add_argument("--threshold", type=float, default=None, help="Absolute threshold in FITS units; overrides --threshold-snr.")
    parser.add_argument("--smooth", type=float, default=0.03, help="Spline smoothing scale in mas. Use 0 for interpolation.")
    parser.add_argument("--stretch", choices=("linear", "sqrt", "log", "asinh"), default="asinh")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--contour-sigma", type=float, default=3.0)
    parser.add_argument("--contour-factor", type=float, default=2.0)
    parser.add_argument("--no-contours", action="store_true")
    parser.add_argument("--fov", type=float, default=None, help="Initial half-width in mas around core/reference.")
    parser.add_argument("--no-auto-fov", action="store_true", help="Start with the full FITS image instead of auto-cropping.")
    parser.add_argument("--build", action="store_true", help="Build immediately from --guide.")
    parser.add_argument("--save", action="store_true", help="Save after building.")
    parser.add_argument("--save-figure", type=Path, default=None, help="Save an overlay PNG to this path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive window.")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if isinstance(args.core, list):
        if len(args.core) != 1:
            raise SystemExit("--core expects exactly one x,y pair")
        args.core = args.core[0]
    if args.fits is None:
        args.fits = Path(__file__).resolve().parents[1] / "data" / "mojave_fits" / "0238+711.u.stacked.icc.fits"
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_arg_parser()
    args = normalize_args(parser.parse_args(list(argv) if argv is not None else None))
    app = RidgeBuilderApp(args)
    if args.build:
        app.build()
    if args.save:
        app.save()
    if args.save_figure is not None:
        app.save_figure(args.save_figure)
    if not args.no_show:
        plt.show()
    else:
        plt.close(app.fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
