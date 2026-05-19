from __future__ import annotations

"""Legacy transverse-peak ridgeline refinement.

This module is intentionally not wired into the current GUI or default
ridgeline extraction path. It keeps the previous iterative transverse peak
refinement code, formerly controlled by ``Transverse Refine Iter`` /
``transverse_refine_iterations``, available for inspection or controlled
experiments.

이 기능은 과거 ridgeline을 transverse peak 후보로 반복 보정하는 상황을
해결하기 위해 ``ridgeline_refine_legacy``로 만들어졌으나, 현재 기본
ridgeline extraction 및 FWHM 측정 workflow와 맞지 않아 더 이상 기본
경로에서 쓰이지 않음으로 ``ridgeline_refine_legacy``는 legacy로
이동되었다.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ridgeline_analysis import (
    Point,
    _dedupe_consecutive_points,
    _moving_average_polyline,
    _smooth_path_bspline,
    sample_transverse_profile,
)


def _select_slice_peak_candidates(
    local_k: np.ndarray,
    local_score: np.ndarray,
    profile_step_px: float,
    max_candidates: int,
    min_rel_score: float,
) -> List[int]:
    local_k = np.asarray(local_k, dtype=np.float32)
    local_score = np.asarray(local_score, dtype=np.float32)
    if local_k.size <= 0 or local_score.size <= 0 or local_k.size != local_score.size:
        return []
    finite = np.isfinite(local_k) & np.isfinite(local_score)
    if not np.any(finite):
        return []
    center_idx = int(np.argmin(np.abs(local_k)))
    finite_score = local_score[finite]
    max_score = float(np.max(finite_score))
    if not np.isfinite(max_score) or max_score <= 1e-8:
        return [center_idx]

    sep_idx = int(max(1, round(1.0 / max(float(profile_step_px), 1e-6))))
    order = np.argsort(-np.where(finite, local_score, -1e9))
    keep: List[int] = []
    for idx in order.tolist():
        idx = int(idx)
        if not finite[idx]:
            continue
        if keep and float(local_score[idx]) < (float(min_rel_score) * max_score):
            break
        if all(abs(idx - prev) >= sep_idx for prev in keep):
            keep.append(idx)
        if len(keep) >= int(max(1, max_candidates)):
            break
    if all(abs(center_idx - prev) >= sep_idx for prev in keep):
        keep.append(center_idx)
    keep = sorted(set(int(v) for v in keep), key=lambda ii: float(local_k[ii]))
    return keep


def refine_ridgeline_by_transverse_peaks_legacy(
    ridge_xy: np.ndarray,
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    region_depth_map: Optional[np.ndarray] = None,
    *,
    iterations: int = 1,
    tangent_half_window: int = 4,
    profile_step_px: float = 0.5,
    smooth_window: int = 3,
    bspline_smoothing: float = 0.0,
    endpoint_xy: Optional[Tuple[Point, Point]] = None,
) -> np.ndarray:
    reference = np.asarray(ridge_xy, dtype=np.float32)
    work = reference.copy()
    iterations = int(max(0, iterations))
    if iterations <= 0 or len(work) < 5:
        return work.copy()
    support = (np.asarray(support_mask, dtype=np.uint8) > 0) & np.isfinite(np.asarray(flux_map, dtype=np.float32))
    depth_arr = None if region_depth_map is None else np.asarray(region_depth_map, dtype=np.int32)
    flux_vals = np.asarray(flux_map, dtype=np.float32)[support]
    if flux_vals.size > 0:
        global_floor = float(np.percentile(flux_vals, 5))
        global_ref = float(np.percentile(flux_vals, 99))
    else:
        global_floor = 0.0
        global_ref = 1.0
    if (not np.isfinite(global_ref)) or global_ref <= global_floor + 1e-9:
        global_floor = float(np.nanmin(flux_vals)) if flux_vals.size > 0 else 0.0
        global_ref = float(np.nanmax(flux_vals)) if flux_vals.size > 0 else 1.0
    global_scale = float(max(1e-6, global_ref - global_floor))
    endpoint0 = None if endpoint_xy is None else np.asarray(endpoint_xy[0], dtype=np.float32)
    endpoint1 = None if endpoint_xy is None else np.asarray(endpoint_xy[1], dtype=np.float32)
    center_penalty = 0.16
    max_candidates = 5
    min_rel_score = 0.92
    continuity_penalty = 2.4
    spatial_penalty = 1.8
    anchor_penalty = 1.2
    blend_alpha = 0.45
    for _ in range(iterations):
        ridge_i32 = np.rint(reference).astype(np.int32)
        slice_candidates: List[List[Dict[str, float]]] = []
        slice_spans: List[float] = []
        for idx in range(len(ridge_i32)):
            try:
                profile = sample_transverse_profile(
                    flux_map=flux_map,
                    support_mask=support_mask,
                    ridge_xy=ridge_i32,
                    ridge_idx=idx,
                    tangent_half_window=tangent_half_window,
                    profile_step_px=profile_step_px,
                )
                valid_x = np.asarray(profile.get("valid_x", []), dtype=np.float32)
                valid_y = np.asarray(profile.get("valid_y", []), dtype=np.float32)
                full_xy = np.asarray(profile.get("profile_xy", []), dtype=np.float32)
                left_lim = int(profile.get("left_lim", 0))
                right_lim = int(profile.get("right_lim", -1))
                if valid_x.size <= 0 or full_xy.ndim != 2 or full_xy.shape[1] < 2 or right_lim < left_lim:
                    raise ValueError("invalid transverse profile")
                valid_xy = full_xy[left_lim:right_lim + 1]
                if len(valid_xy) != len(valid_x):
                    raise ValueError("profile/support mismatch")
                px, py = profile.get("ridge_xy", (int(ridge_i32[idx, 0]), int(ridge_i32[idx, 1])))
                finite = np.isfinite(valid_x) & np.isfinite(valid_y)
                local_k = valid_x[finite]
                local_prof = valid_y[finite]
                valid_xy = valid_xy[finite]
                if local_k.size < 3 or len(valid_xy) != len(local_k):
                    raise ValueError("too few valid profile samples")

                peak_max = float(np.max(local_prof))
                scale = float(max(1.0, np.max(np.abs(local_k))))
                local_score = local_prof.copy()
                if np.isfinite(peak_max) and peak_max > 1e-8:
                    local_score = local_prof - (float(center_penalty) * peak_max * (np.abs(local_k) / scale))
                absolute_reward = np.clip((local_score - global_floor) / global_scale, 0.0, None).astype(np.float32)
                if depth_arr is not None:
                    depth_samples = []
                    h, w = depth_arr.shape[:2]
                    for xy in valid_xy:
                        xi = int(np.clip(round(float(xy[0])), 0, max(0, w - 1)))
                        yi = int(np.clip(round(float(xy[1])), 0, max(0, h - 1)))
                        depth_samples.append(int(depth_arr[yi, xi]))
                    depth_samples = np.asarray(depth_samples, dtype=np.float32)
                    finite_depth = np.isfinite(depth_samples)
                    if np.any(finite_depth):
                        dmin = float(np.min(depth_samples[finite_depth]))
                        dmax = float(np.max(depth_samples[finite_depth]))
                        if dmax > dmin + 1e-6:
                            depth_reward = (depth_samples - dmin) / (dmax - dmin)
                        else:
                            depth_reward = np.ones_like(depth_samples, dtype=np.float32)
                        absolute_reward = ((0.72 * absolute_reward) + (0.28 * depth_reward)).astype(np.float32)
                candidate_indices = _select_slice_peak_candidates(
                    local_k=local_k,
                    local_score=local_score,
                    profile_step_px=profile_step_px,
                    max_candidates=max_candidates,
                    min_rel_score=min_rel_score,
                )
                cur_candidates: List[Dict[str, float]] = []
                for idx_local in candidate_indices:
                    idx_local = int(idx_local)
                    cur_candidates.append(
                        {
                            "k": float(local_k[idx_local]),
                            "x": float(valid_xy[idx_local, 0]),
                            "y": float(valid_xy[idx_local, 1]),
                            "reward": float(absolute_reward[idx_local]),
                        }
                    )
                if not cur_candidates:
                    cur_candidates = [
                        {
                            "k": 0.0,
                            "x": float(px),
                            "y": float(py),
                            "reward": 0.0,
                        }
                    ]
                slice_candidates.append(cur_candidates)
                slice_spans.append(float(max(profile_step_px, np.max(np.abs(local_k)))))
            except Exception:
                slice_candidates.append(
                    [
                        {
                            "k": 0.0,
                            "x": float(ridge_i32[idx, 0]),
                            "y": float(ridge_i32[idx, 1]),
                            "reward": 0.0,
                        }
                    ]
                )
                slice_spans.append(float(max(profile_step_px, 1.0)))

        if not slice_candidates:
            return work.copy()

        dp_scores: List[np.ndarray] = []
        back_ptrs: List[np.ndarray] = []
        first = slice_candidates[0]
        dp_scores.append(np.asarray([c["reward"] for c in first], dtype=np.float32))
        back_ptrs.append(np.full(len(first), -1, dtype=np.int32))

        for i in range(1, len(slice_candidates)):
            prev = slice_candidates[i - 1]
            cur = slice_candidates[i]
            prev_scores = dp_scores[-1]
            cur_scores = np.full(len(cur), -1e9, dtype=np.float32)
            cur_back = np.full(len(cur), -1, dtype=np.int32)
            norm_span = float(max(profile_step_px, 0.5 * (slice_spans[i - 1] + slice_spans[i])))
            expected_step = float(
                max(
                    profile_step_px,
                    np.hypot(
                        float(reference[i, 0] - reference[i - 1, 0]),
                        float(reference[i, 1] - reference[i - 1, 1]),
                    ),
                )
            )

            for j, c in enumerate(cur):
                best_score = -1e9
                best_prev = -1
                for k_prev, p in enumerate(prev):
                    continuity_cost = float(continuity_penalty) * abs(float(c["k"]) - float(p["k"])) / max(norm_span, 1e-6)
                    spatial_step = float(
                        np.hypot(
                            float(c["x"]) - float(p["x"]),
                            float(c["y"]) - float(p["y"]),
                        )
                    )
                    spatial_cost = float(spatial_penalty) * abs(spatial_step - expected_step) / max(expected_step, 1e-6)
                    anchor_cost = float(anchor_penalty) * abs(float(c["k"])) / max(norm_span, 1e-6)
                    score = float(prev_scores[k_prev]) + float(c["reward"]) - continuity_cost - spatial_cost - anchor_cost
                    if score > best_score:
                        best_score = score
                        best_prev = int(k_prev)
                cur_scores[j] = float(best_score)
                cur_back[j] = int(best_prev)
            dp_scores.append(cur_scores)
            back_ptrs.append(cur_back)

        path_choice = [0] * len(slice_candidates)
        path_choice[-1] = int(np.argmax(dp_scores[-1]))
        for i in range(len(slice_candidates) - 1, 0, -1):
            prev_idx = int(back_ptrs[i][path_choice[i]])
            path_choice[i - 1] = max(0, prev_idx)

        candidate_xy = reference.copy()
        for i, cand_idx in enumerate(path_choice):
            chosen = slice_candidates[i][int(cand_idx)]
            candidate_xy[i, 0] = float(chosen["x"])
            candidate_xy[i, 1] = float(chosen["y"])

        updated = ((1.0 - blend_alpha) * work) + (blend_alpha * candidate_xy)
        updated = _moving_average_polyline(updated, window=max(3, int(smooth_window)))
        if endpoint0 is not None and len(updated) >= 1:
            updated[0] = endpoint0
        if endpoint1 is not None and len(updated) >= 2:
            updated[-1] = endpoint1
        work = np.asarray(updated, dtype=np.float32)
    work = _smooth_path_bspline(work, smoothing=float(max(0.0, bspline_smoothing)))
    if endpoint0 is not None and len(work) >= 1:
        work[0] = endpoint0
    if endpoint1 is not None and len(work) >= 2:
        work[-1] = endpoint1
    return np.asarray(_dedupe_consecutive_points(work), dtype=np.float32)


_refine_ridgeline_by_transverse_peaks = refine_ridgeline_by_transverse_peaks_legacy
