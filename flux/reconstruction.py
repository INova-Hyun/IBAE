from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def reconstruct_flux_from_levels(
    region_depth_map: np.ndarray,
    roi_mask: np.ndarray,
    background_flux: float,
    l1_flux: float,
    level_ratio: float,
    smooth_sigma_px: float,
    contour_values: Optional[Sequence[float]] = None,
    include_target_flux_map: bool = True,
    l0_l1_transition_mode: str = "gaussian",
    l0_l1_transition_width_px: Optional[float] = None,
    l0_l1_transition_alpha: float = 3.0,
) -> Dict[str, object]:
    depth = np.asarray(region_depth_map, dtype=np.int32)
    valid = np.asarray(roi_mask, dtype=np.uint8) > 0
    flux = np.full(depth.shape, np.nan, dtype=np.float32)

    if not np.any(valid):
        result = {
            "smoothed_flux_map": flux.copy(),
            "valid_mask": valid.astype(np.uint8),
            "min_flux": float(background_flux),
            "max_flux": float(background_flux),
            "value_range": (float(background_flux), float(background_flux)),
            "l0_l1_transition": {
                "mode": _normalize_l0_l1_transition_mode(l0_l1_transition_mode),
                "width_px": float("nan"),
                "width_source": "empty_valid_mask",
                "alpha": float(l0_l1_transition_alpha),
                "applied_pixel_count": 0,
            },
        }
        if bool(include_target_flux_map):
            result["target_flux_map"] = flux.copy()
        return result

    ratio = float(max(1e-9, level_ratio))
    contour_values_arr = None
    if contour_values is not None:
        try:
            contour_values_arr = np.asarray([float(v) for v in contour_values], dtype=np.float32)
        except Exception:
            contour_values_arr = None
        if contour_values_arr is not None and contour_values_arr.size <= 0:
            contour_values_arr = None
    flux[(depth == 0) & valid] = float(background_flux)

    positive_depths = sorted(int(v) for v in np.unique(depth[valid]) if int(v) > 0)
    for level in positive_depths:
        level_mask = (depth == int(level)) & valid
        if not np.any(level_mask):
            continue
        flux_outer = _level_flux_value(
            int(level),
            float(l1_flux),
            ratio,
            contour_values_arr,
        )
        flux_inner = _level_flux_value(
            int(level) + 1,
            float(l1_flux),
            ratio,
            contour_values_arr,
        )

        comp_count, comp_labels = cv2.connectedComponents(level_mask.astype(np.uint8), connectivity=8)
        outer_region_mask = valid & (depth == int(level) - 1)
        inner_region_mask = valid & (depth == int(level) + 1)
        for comp_id in range(1, int(comp_count)):
            comp = comp_labels == comp_id
            if not np.any(comp):
                continue

            outer_boundary = _boundary_pixels_inside_component(comp, outer_region_mask)
            inner_boundary = _boundary_pixels_inside_component(comp, inner_region_mask)
            has_outer = bool(np.any(outer_boundary))
            has_inner = bool(np.any(inner_boundary))

            if has_outer and has_inner:
                outer_dist = _distance_to_boundary(outer_boundary, comp)
                inner_dist = _distance_to_boundary(inner_boundary, comp)
                denom = outer_dist + inner_dist
                weight_inner = np.zeros_like(outer_dist, dtype=np.float32)
                ok = comp & np.isfinite(denom) & (denom > 1e-6)
                weight_inner[ok] = outer_dist[ok] / denom[ok]
                comp_values = _interpolate_flux_values(
                    flux_outer,
                    flux_inner,
                    weight_inner[comp],
                )
                flux[comp] = comp_values.astype(np.float32)
            elif has_outer:
                # Deepest band: interpolate from the outer contour value to an implied
                # brighter center value one ratio step inward, using normalized
                # distance-to-boundary inside the component.
                dist_to_outer = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
                max_dist = float(np.max(dist_to_outer[comp])) if np.any(comp) else 0.0
                if max_dist > 1e-6:
                    weight_center = np.zeros_like(dist_to_outer, dtype=np.float32)
                    weight_center[comp] = np.clip(dist_to_outer[comp] / max_dist, 0.0, 1.0)
                    comp_values = _interpolate_flux_values(
                        flux_outer,
                        flux_inner,
                        weight_center[comp],
                    )
                    flux[comp] = comp_values.astype(np.float32)
                else:
                    flux[comp] = np.float32(flux_outer)
            else:
                flux[comp] = np.float32(flux_outer)

    l0_l1_transition = _apply_l0_l1_gaussian_transition(
        flux,
        depth,
        valid,
        background_flux=float(background_flux),
        l1_flux=_level_flux_value(1, float(l1_flux), ratio, contour_values_arr),
        smooth_sigma_px=float(smooth_sigma_px),
        mode=str(l0_l1_transition_mode),
        width_px=l0_l1_transition_width_px,
        alpha=float(l0_l1_transition_alpha),
    )
    smoothed = _smooth_inside_level_components(
        flux,
        depth,
        valid,
        sigma_px=float(smooth_sigma_px),
    )
    valid_values = smoothed[valid & np.isfinite(smoothed)]
    if valid_values.size <= 0:
        min_flux = float(background_flux)
        max_flux = float(background_flux)
    else:
        min_flux = float(np.min(valid_values))
        max_flux = float(np.max(valid_values))

    result = {
        "smoothed_flux_map": np.asarray(smoothed, dtype=np.float32),
        "valid_mask": valid.astype(np.uint8),
        "min_flux": float(min_flux),
        "max_flux": float(max_flux),
        "value_range": (float(min_flux), float(max_flux)),
        "contour_values": None if contour_values_arr is None else contour_values_arr.astype(np.float32),
        "l0_l1_transition": dict(l0_l1_transition),
    }
    if bool(include_target_flux_map):
        result["target_flux_map"] = np.asarray(flux, dtype=np.float32)
    return result


def _normalize_l0_l1_transition_mode(mode: str) -> str:
    mode_norm = str(mode or "gaussian").strip().lower()
    if mode_norm in ("gaussian", "gaussian_fade", "truncated_gaussian", "fade"):
        return "gaussian"
    if mode_norm in ("flat", "none", "off", "disabled", "legacy"):
        return "flat"
    return "gaussian"


def _positive_float_or_nan(value: Optional[float]) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return out if np.isfinite(out) and out > 0.0 else float("nan")


def _resolve_l0_l1_transition_width_px(
    depth: np.ndarray,
    valid: np.ndarray,
    requested_width_px: Optional[float],
    smooth_sigma_px: float,
) -> Tuple[float, str]:
    requested = _positive_float_or_nan(requested_width_px)
    if np.isfinite(requested):
        return float(requested), "requested"

    spacing = _median_l1_l2_spacing_px(depth, valid)
    if np.isfinite(spacing) and spacing > 0.0:
        return float(max(1.0, spacing)), "median_l1_l2_spacing"

    sigma = _positive_float_or_nan(smooth_sigma_px)
    if np.isfinite(sigma):
        return float(max(1.0, 2.0 * sigma)), "smooth_sigma_x2_fallback"
    return 1.0, "fallback_1px"


def _median_l1_l2_spacing_px(depth_map: np.ndarray, valid_mask: np.ndarray) -> float:
    depth = np.asarray(depth_map, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    level_mask = (depth == 1) & valid
    if not np.any(level_mask):
        return float("nan")
    outer_region_mask = valid & (depth == 0)
    inner_region_mask = valid & (depth == 2)
    if not np.any(outer_region_mask) or not np.any(inner_region_mask):
        return float("nan")

    values = []
    comp_count, comp_labels = cv2.connectedComponents(level_mask.astype(np.uint8), connectivity=8)
    for comp_id in range(1, int(comp_count)):
        comp = comp_labels == comp_id
        if not np.any(comp):
            continue
        outer_boundary = _boundary_pixels_inside_component(comp, outer_region_mask)
        inner_boundary = _boundary_pixels_inside_component(comp, inner_region_mask)
        if not np.any(outer_boundary) or not np.any(inner_boundary):
            continue
        outer_dist = _distance_to_boundary(outer_boundary, comp)
        inner_dist = _distance_to_boundary(inner_boundary, comp)
        spacing = outer_dist + inner_dist
        finite = spacing[comp & np.isfinite(spacing) & (spacing > 0.0)]
        if finite.size > 0:
            values.append(finite.astype(np.float32, copy=False))
    if not values:
        return float("nan")
    all_values = np.concatenate(values)
    if all_values.size <= 0:
        return float("nan")
    return float(np.nanmedian(all_values))


def _truncated_gaussian_fade_values(
    background_flux: float,
    l1_flux: float,
    distance_px: np.ndarray,
    width_px: float,
    alpha: float,
) -> np.ndarray:
    width = float(max(1e-6, width_px))
    alpha = float(max(1e-6, alpha))
    t = np.clip(np.asarray(distance_px, dtype=np.float32) / width, 0.0, 1.0)
    eps = float(np.exp(-0.5 * alpha * alpha))
    denom = max(1e-12, 1.0 - eps)
    fade = (np.exp(-0.5 * np.square(alpha * t)).astype(np.float32) - eps) / denom
    fade = np.clip(fade, 0.0, 1.0)
    return (float(background_flux) + ((float(l1_flux) - float(background_flux)) * fade)).astype(np.float32)


def _apply_l0_l1_gaussian_transition(
    flux_map: np.ndarray,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    background_flux: float,
    l1_flux: float,
    smooth_sigma_px: float,
    mode: str,
    width_px: Optional[float],
    alpha: float,
) -> Dict[str, object]:
    mode_norm = _normalize_l0_l1_transition_mode(mode)
    info: Dict[str, object] = {
        "mode": mode_norm,
        "width_px": float("nan"),
        "width_source": "disabled" if mode_norm == "flat" else "not_applied",
        "alpha": float(alpha),
        "applied_pixel_count": 0,
    }
    if mode_norm == "flat":
        return info

    flux = np.asarray(flux_map, dtype=np.float32)
    depth = np.asarray(depth_map, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    outer_mask = valid & (depth == 0)
    l1_mask = valid & (depth == 1)
    if not np.any(outer_mask) or not np.any(l1_mask):
        info["width_source"] = "missing_l0_or_l1"
        return info
    if not (np.isfinite(float(background_flux)) and np.isfinite(float(l1_flux))):
        info["width_source"] = "invalid_flux"
        return info

    resolved_width, width_source = _resolve_l0_l1_transition_width_px(
        depth,
        valid,
        width_px,
        smooth_sigma_px,
    )
    if not (np.isfinite(resolved_width) and resolved_width > 0.0):
        info["width_source"] = "invalid_width"
        return info

    l0_boundary = _boundary_pixels_inside_component(outer_mask, l1_mask)
    if not np.any(l0_boundary):
        info["width_px"] = float(resolved_width)
        info["width_source"] = "missing_l0_l1_boundary"
        return info

    src = np.ones(depth.shape[:2], dtype=np.uint8)
    src[l0_boundary] = 0
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)
    transition = outer_mask & np.isfinite(dist) & (dist <= float(resolved_width))
    if not np.any(transition):
        info["width_px"] = float(resolved_width)
        info["width_source"] = width_source
        return info

    flux[transition] = _truncated_gaussian_fade_values(
        float(background_flux),
        float(l1_flux),
        dist[transition],
        float(resolved_width),
        float(alpha),
    )
    info.update(
        {
            "width_px": float(resolved_width),
            "width_source": str(width_source),
            "alpha": float(alpha),
            "applied_pixel_count": int(np.count_nonzero(transition)),
        }
    )
    return info


def _level_flux_value(
    level: int,
    l1_flux: float,
    ratio: float,
    contour_values: Optional[np.ndarray] = None,
) -> float:
    if int(level) <= 0:
        return float(l1_flux)
    if contour_values is not None and contour_values.size > 0:
        idx = int(level) - 1
        if 0 <= idx < int(contour_values.size):
            return float(contour_values[idx])
        last = float(contour_values[-1])
        if contour_values.size >= 2:
            prev = float(contour_values[-2])
            if prev > 0.0 and last > 0.0:
                tail_ratio = max(1e-9, last / prev)
                return float(last * (tail_ratio ** float(idx - (contour_values.size - 1))))
        return float(last)
    return float(l1_flux) * float(ratio ** float(int(level) - 1))


def _interpolate_flux_values(
    flux_outer: float,
    flux_inner: float,
    weight_inner: np.ndarray,
) -> np.ndarray:
    w = np.asarray(weight_inner, dtype=np.float32)
    w = np.clip(w, 0.0, 1.0)
    if float(flux_outer) > 0.0 and float(flux_inner) > 0.0:
        lo = np.log(np.float32(flux_outer))
        li = np.log(np.float32(flux_inner))
        return np.exp(((1.0 - w) * lo) + (w * li)).astype(np.float32)
    return (((1.0 - w) * float(flux_outer)) + (w * float(flux_inner))).astype(np.float32)


def _boundary_pixels_inside_component(
    component_mask: np.ndarray,
    neighbor_region_mask: np.ndarray,
) -> np.ndarray:
    comp = np.asarray(component_mask, dtype=bool)
    neigh = np.asarray(neighbor_region_mask, dtype=bool)
    if not np.any(comp) or not np.any(neigh):
        return np.zeros(comp.shape, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh_dil = cv2.dilate(neigh.astype(np.uint8), kernel, iterations=1) > 0
    return comp & neigh_dil


def _distance_to_boundary(boundary_mask: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
    boundary = np.asarray(boundary_mask, dtype=bool)
    comp = np.asarray(component_mask, dtype=bool)
    if not np.any(boundary):
        return np.full(comp.shape, np.nan, dtype=np.float32)
    src = np.ones(comp.shape, dtype=np.uint8)
    src[boundary] = 0
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)
    dist[~comp] = np.nan
    return dist


def _masked_gaussian(flux_map: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    flux = np.asarray(flux_map, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return flux.copy()
    if float(sigma_px) <= 1e-6:
        return flux.copy()

    fill_value = float(np.nanmean(flux[valid])) if np.any(valid & np.isfinite(flux)) else 0.0
    dense = np.where(valid, np.nan_to_num(flux, nan=fill_value), 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    sigma = max(0.1, float(sigma_px))
    ksize = max(3, int(round(sigma * 6.0)))
    if ksize % 2 == 0:
        ksize += 1

    blurred_num = cv2.GaussianBlur(dense * weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_den = cv2.GaussianBlur(weight, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    out = flux.copy()
    ok = valid & (blurred_den > 1e-6)
    out[ok] = blurred_num[ok] / blurred_den[ok]
    return out


def _gaussian_kernel_radius(sigma_px: float) -> int:
    sigma = max(0.1, float(sigma_px))
    ksize = max(3, int(round(sigma * 6.0)))
    if ksize % 2 == 0:
        ksize += 1
    return int(ksize // 2)


def _smooth_inside_level_components(
    flux_map: np.ndarray,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    flux = np.asarray(flux_map, dtype=np.float32)
    depth = np.asarray(depth_map, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    out = flux.copy()
    out[~valid] = np.nan
    if not np.any(valid) or float(sigma_px) <= 1e-6:
        return out

    height, width = out.shape[:2]
    pad = _gaussian_kernel_radius(float(sigma_px))
    for level in sorted(int(v) for v in np.unique(depth[valid])):
        level_mask = valid & (depth == int(level))
        if not np.any(level_mask):
            continue
        comp_count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(
            level_mask.astype(np.uint8),
            connectivity=8,
        )
        for comp_id in range(1, int(comp_count)):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[comp_id, cv2.CC_STAT_LEFT])
            y = int(stats[comp_id, cv2.CC_STAT_TOP])
            w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
            h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(width, x + w + pad)
            y1 = min(height, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            comp_crop = comp_labels[y0:y1, x0:x1] == comp_id
            if not np.any(comp_crop):
                continue
            out_crop = out[y0:y1, x0:x1]
            sm_crop = _masked_gaussian(out_crop, comp_crop, sigma_px=float(sigma_px))
            out_crop[comp_crop] = sm_crop[comp_crop]
    out[~valid] = np.nan
    return out


def crop_valid_field(
    flux_map: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    flux = np.asarray(flux_map, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid):
        return flux.copy(), valid.astype(np.uint8)
    ys, xs = np.nonzero(valid)
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    return flux[y0:y1, x0:x1].copy(), valid[y0:y1, x0:x1].astype(np.uint8)


def downsample_flux_for_surface(
    flux_map: np.ndarray,
    valid_mask: np.ndarray,
    max_dim: int = 160,
) -> Dict[str, object]:
    flux_crop, valid_crop = crop_valid_field(flux_map, valid_mask)
    h, w = flux_crop.shape[:2]
    if h <= 0 or w <= 0:
        return {
            "x": np.zeros((0, 0), dtype=np.float32),
            "y": np.zeros((0, 0), dtype=np.float32),
            "z": np.zeros((0, 0), dtype=np.float32),
            "valid_mask": np.zeros((0, 0), dtype=np.uint8),
            "stride": 1,
        }

    max_dim = int(max(16, max_dim))
    stride = int(max(1, np.ceil(max(float(h), float(w)) / float(max_dim))))
    z = flux_crop[::stride, ::stride].astype(np.float32)
    valid = valid_crop[::stride, ::stride].astype(np.uint8)
    yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    z_masked = np.ma.masked_where(valid == 0, z)
    return {
        "x": xx.astype(np.float32),
        "y": yy.astype(np.float32),
        "z": z_masked,
        "valid_mask": valid,
        "stride": int(stride),
    }
