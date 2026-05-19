#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


MAS_PER_DEG = 3600.0 * 1000.0


def _open_maybe_gzip(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _parse_value(raw: str):
    stripped = raw.strip()
    if stripped.startswith("'"):
        end = stripped.find("'", 1)
        if end >= 0:
            return stripped[1:end].strip()
    value = raw.split("/", 1)[0].strip()
    if not value:
        return None
    if value in ("T", "F"):
        return value == "T"
    try:
        if any(ch in value.upper() for ch in (".", "E", "D")):
            return float(value.replace("D", "E"))
        return int(value)
    except ValueError:
        return value


def read_primary_fits(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read a simple primary-image FITS file without astropy."""
    with _open_maybe_gzip(path) as f:
        header_bytes = bytearray()
        found_end = False
        while not found_end:
            block = f.read(2880)
            if not block:
                raise ValueError(f"FITS header END card not found in {path}")
            header_bytes.extend(block)
            for start in range(0, len(block), 80):
                card = block[start : start + 80]
                key = card[:8].decode("ascii", errors="replace").strip()
                if key == "END":
                    found_end = True
                    break

        header: Dict[str, object] = {}
        cards = bytes(header_bytes).decode("ascii", errors="replace")
        for start in range(0, len(cards), 80):
            card = cards[start : start + 80]
            key = card[:8].strip()
            if not key:
                continue
            if key == "END":
                break
            if card[8:10] == "= ":
                header[key] = _parse_value(card[10:])

        bitpix = int(header.get("BITPIX", 0))
        naxis = int(header.get("NAXIS", 0))
        if naxis < 2:
            raise ValueError(f"{path} is not an image FITS file (NAXIS={naxis})")
        dims = [int(header[f"NAXIS{i}"]) for i in range(1, naxis + 1)]

        dtype_by_bitpix = {
            8: np.dtype("u1"),
            16: np.dtype(">i2"),
            32: np.dtype(">i4"),
            64: np.dtype(">i8"),
            -32: np.dtype(">f4"),
            -64: np.dtype(">f8"),
        }
        if bitpix not in dtype_by_bitpix:
            raise ValueError(f"Unsupported BITPIX={bitpix}")

        n_values = int(np.prod(dims))
        data = np.frombuffer(f.read(n_values * dtype_by_bitpix[bitpix].itemsize), dtype=dtype_by_bitpix[bitpix])
        if data.size != n_values:
            raise ValueError(f"Expected {n_values} values, read {data.size}")

    image = data.reshape(tuple(reversed(dims))).squeeze().astype(np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing FITS axes, got shape={image.shape}")

    bscale = float(header.get("BSCALE", 1.0) or 1.0)
    bzero = float(header.get("BZERO", 0.0) or 0.0)
    if bscale != 1.0 or bzero != 0.0:
        image = image * bscale + bzero
    return image, header


def robust_corner_rms(image: np.ndarray, corner_size: Optional[int] = None) -> float:
    ny, nx = image.shape
    n = int(corner_size or max(16, min(nx, ny) // 8))
    n = min(n, nx // 2, ny // 2)
    corners = np.concatenate(
        [
            image[:n, :n].ravel(),
            image[:n, -n:].ravel(),
            image[-n:, :n].ravel(),
            image[-n:, -n:].ravel(),
        ]
    )
    finite = corners[np.isfinite(corners)]
    if finite.size < 8:
        return float("nan")
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(finite))


def image_edges_mas(header: Dict[str, object], shape: Tuple[int, int]) -> Tuple[float, float, float, float]:
    ny, nx = shape
    cdelt1 = float(header.get("CDELT1", -1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    crpix1 = float(header.get("CRPIX1", (nx + 1) / 2.0))
    crpix2 = float(header.get("CRPIX2", (ny + 1) / 2.0))
    x0 = (0.5 - crpix1) * cdelt1 * MAS_PER_DEG
    x1 = (nx + 0.5 - crpix1) * cdelt1 * MAS_PER_DEG
    y0 = (0.5 - crpix2) * cdelt2 * MAS_PER_DEG
    y1 = (ny + 0.5 - crpix2) * cdelt2 * MAS_PER_DEG
    return x0, x1, y0, y1


def pixel_to_mas(header: Dict[str, object], x_pix0: np.ndarray, y_pix0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ny = 1
    nx = 1
    cdelt1 = float(header.get("CDELT1", -1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    crpix1 = float(header.get("CRPIX1", (nx + 1) / 2.0))
    crpix2 = float(header.get("CRPIX2", (ny + 1) / 2.0))
    x_mas = (x_pix0 + 1.0 - crpix1) * cdelt1 * MAS_PER_DEG
    y_mas = (y_pix0 + 1.0 - crpix2) * cdelt2 * MAS_PER_DEG
    return x_mas, y_mas


def mas_to_pixel(header: Dict[str, object], x_mas: float, y_mas: float) -> Tuple[float, float]:
    cdelt1 = float(header.get("CDELT1", -1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    crpix1 = float(header.get("CRPIX1", 1.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    x_pix0 = x_mas / (cdelt1 * MAS_PER_DEG) + crpix1 - 1.0
    y_pix0 = y_mas / (cdelt2 * MAS_PER_DEG) + crpix2 - 1.0
    return x_pix0, y_pix0


def apply_stretch(image: np.ndarray, stretch: str, vmin: float, vmax: float) -> np.ndarray:
    scaled = np.clip((image - vmin) / max(vmax - vmin, 1e-30), 0.0, None)
    if stretch == "linear":
        out = np.clip(scaled, 0.0, 1.0)
    elif stretch == "sqrt":
        out = np.sqrt(np.clip(scaled, 0.0, 1.0))
    elif stretch == "log":
        out = np.log10(1.0 + 999.0 * scaled) / 3.0
        out = np.clip(out, 0.0, 1.0)
    elif stretch == "asinh":
        out = np.arcsinh(20.0 * scaled)
        max_val = np.nanmax(out)
        if max_val > 0:
            out = out / max_val
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown stretch: {stretch}")
    return out


def auto_limits(
    image: np.ndarray,
    header: Dict[str, object],
    rms: float,
    threshold_fraction: float,
    threshold_snr: float,
    padding_mas: float,
) -> Optional[Tuple[float, float, float, float]]:
    peak = float(np.nanmax(image))
    if not np.isfinite(peak) or peak <= 0:
        return None
    threshold = max(threshold_fraction * peak, threshold_snr * rms if np.isfinite(rms) else -np.inf)
    yy, xx = np.where(np.isfinite(image) & (image >= threshold))
    if xx.size < 2:
        return None
    x_mas, y_mas = pixel_to_mas(header, xx.astype(float), yy.astype(float))
    xmin, xmax = float(np.nanmin(x_mas) - padding_mas), float(np.nanmax(x_mas) + padding_mas)
    ymin, ymax = float(np.nanmin(y_mas) - padding_mas), float(np.nanmax(y_mas) + padding_mas)
    return xmin, xmax, ymin, ymax


def positive_contour_levels(rms: float, peak: float, sigma: float, factor: float) -> np.ndarray:
    if not np.isfinite(rms) or rms <= 0 or not np.isfinite(peak) or peak <= 0:
        return np.array([], dtype=float)
    levels = []
    value = sigma * rms
    while value < peak:
        levels.append(value)
        value *= factor
    return np.asarray(levels, dtype=float)


def zoom_axes(ax, factor: float, center: Optional[Tuple[float, float]] = None) -> None:
    """Zoom axes by a multiplicative factor, preserving reversed axes."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if center is None:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
    else:
        cx, cy = center
    width = (x1 - x0) * factor
    height = (y1 - y0) * factor
    ax.set_xlim(cx - 0.5 * width, cx + 0.5 * width)
    ax.set_ylim(cy - 0.5 * height, cy + 0.5 * height)


def first_fits_file() -> Path:
    candidates = sorted(Path(".").glob("*.fits")) + sorted(Path(".").glob("*.fits.gz"))
    if not candidates:
        raise SystemExit("No .fits or .fits.gz file found in the current directory.")
    return candidates[0]


def print_click_help() -> None:
    print(
        "Interactive keys: h=help, mouse wheel/+/-=zoom, f=full view, c=toggle contours, "
        "1/2/3/4=linear/sqrt/log/asinh, r=reset view, q=quit. "
        "Click prints x/y mas, pixel, and Jy/beam."
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Small matplotlib FITS image viewer for MOJAVE stacked maps.")
    parser.add_argument("fits", nargs="?", type=Path, default=None, help="FITS image file. Defaults to first *.fits in cwd.")
    parser.add_argument("--stretch", choices=("linear", "sqrt", "log", "asinh"), default="asinh")
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--vmin", type=float, default=None, help="Display minimum in image units.")
    parser.add_argument("--vmax", type=float, default=None, help="Display maximum in image units.")
    parser.add_argument("--pmin", type=float, default=1.0, help="Percentile display minimum when --vmin is absent.")
    parser.add_argument("--pmax", type=float, default=99.9, help="Percentile display maximum when --vmax is absent.")
    parser.add_argument("--no-contours", action="store_true")
    parser.add_argument("--contour-sigma", type=float, default=3.0)
    parser.add_argument("--contour-factor", type=float, default=2.0)
    parser.add_argument("--auto-fov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-view", action="store_true", help="Start with the full FITS image view.")
    parser.add_argument("--fov", type=float, default=None, help="Initial half-width in mas around the FITS reference pixel.")
    parser.add_argument("--save", type=Path, default=None, help="Save PNG instead of or before showing the viewer.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.full_view:
        args.auto_fov = False

    fits_path = args.fits or first_fits_file()
    image, header = read_primary_fits(fits_path)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise SystemExit("Image contains no finite pixels.")

    rms = robust_corner_rms(image)
    peak = float(np.nanmax(image))
    vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(finite, args.pmin))
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(finite, args.pmax))
    if args.vmin is None and np.isfinite(rms):
        vmin = max(vmin, -3.0 * rms)
    display = apply_stretch(image, args.stretch, vmin, vmax)

    extent = image_edges_mas(header, image.shape)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(display, origin="lower", extent=extent, cmap=args.cmap, interpolation="nearest")
    contour_sets = []

    def draw_contours() -> None:
        nonlocal contour_sets
        for contour_set in contour_sets:
            contour_set.remove()
        contour_sets = []
        if args.no_contours:
            fig.canvas.draw_idle()
            return
        levels = positive_contour_levels(rms, peak, args.contour_sigma, args.contour_factor)
        if levels.size:
            cs = ax.contour(
                image,
                levels=levels,
                colors="white",
                linewidths=0.6,
                alpha=0.75,
                origin="lower",
                extent=extent,
            )
            contour_sets = [cs]
        fig.canvas.draw_idle()

    draw_contours()

    bmaj = float(header.get("BMAJ", float("nan"))) * MAS_PER_DEG if "BMAJ" in header else float("nan")
    bmin = float(header.get("BMIN", float("nan"))) * MAS_PER_DEG if "BMIN" in header else float("nan")
    cdelt1 = abs(float(header.get("CDELT1", float("nan")))) * MAS_PER_DEG if "CDELT1" in header else float("nan")
    bunit = str(header.get("BUNIT", "") or "").strip()
    title = fits_path.name
    subtitle = f"beam={bmaj:.3f}x{bmin:.3f} mas, pixel={cdelt1:.4f} mas, peak={peak:.4g} {bunit}, rms~{rms:.3g}"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Relative RA offset (mas; positive left if CDELT1<0)")
    ax.set_ylabel("Relative Dec offset (mas)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"{args.stretch} stretch")

    full_xlim = ax.get_xlim()
    full_ylim = ax.get_ylim()
    if args.fov is not None:
        half = float(args.fov)
        ax.set_xlim(half, -half)
        ax.set_ylim(-half, half)
    elif args.auto_fov:
        limits = auto_limits(image, header, rms, threshold_fraction=0.005, threshold_snr=8.0, padding_mas=2.0)
        if limits is not None:
            xmin, xmax, ymin, ymax = limits
            ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
            ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
    reset_xlim = ax.get_xlim()
    reset_ylim = ax.get_ylim()

    def format_coord(x: float, y: float) -> str:
        xp, yp = mas_to_pixel(header, x, y)
        xi, yi = int(round(xp)), int(round(yp))
        if 0 <= yi < image.shape[0] and 0 <= xi < image.shape[1]:
            return f"x={x:.3f} mas y={y:.3f} mas  pix=({xi},{yi})  I={image[yi, xi]:.6g} {bunit}"
        return f"x={x:.3f} mas y={y:.3f} mas"

    ax.format_coord = format_coord

    stretch_for_key = {"1": "linear", "2": "sqrt", "3": "log", "4": "asinh"}

    def on_click(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        print(format_coord(float(event.xdata), float(event.ydata)))

    def on_scroll(event) -> None:
        if event.inaxes is not ax:
            return
        factor = 1.0 / 1.25 if event.button == "up" else 1.25
        center = None
        if event.xdata is not None and event.ydata is not None:
            center = (float(event.xdata), float(event.ydata))
        zoom_axes(ax, factor, center)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal display
        if event.key == "h":
            print_click_help()
        elif event.key == "c":
            args.no_contours = not args.no_contours
            draw_contours()
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
        elif event.key in stretch_for_key:
            new_stretch = stretch_for_key[event.key]
            display = apply_stretch(image, new_stretch, vmin, vmax)
            im.set_data(display)
            cbar.set_label(f"{new_stretch} stretch")
            fig.canvas.draw_idle()
            print(f"stretch={new_stretch}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"Loaded: {fits_path}")
    print(f"shape={image.shape}, BUNIT={bunit}, peak={peak:.6g}, corner_rms~{rms:.6g}")
    print_click_help()

    if args.save is not None:
        fig.savefig(args.save, dpi=180)
        print(f"Saved: {args.save}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
