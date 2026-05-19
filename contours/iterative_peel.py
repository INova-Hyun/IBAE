from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


RepairCallback = Callable[[np.ndarray], Dict[str, object]]


def _full_mask_border(mask: np.ndarray) -> np.ndarray:
    border = np.zeros_like(mask, dtype=bool)
    if border.size == 0:
        return border
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _background_region_labels(
    thin_mask: np.ndarray,
    roi_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thin = np.asarray(thin_mask, dtype=np.uint8) > 0
    valid = np.asarray(roi_mask, dtype=np.uint8) > 0
    free = valid & (~thin)
    if not np.any(free):
        zero = np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
        return zero, np.zeros_like(zero, dtype=np.int32), np.zeros(0, dtype=np.int32)

    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    if int(n_labels) <= 1:
        zero = np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
        return zero, labels.astype(np.int32), np.zeros(0, dtype=np.int32)

    seed_mask = _full_mask_border(free)
    invalid = ~valid
    if np.any(invalid):
        seam = cv2.dilate(invalid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        seed_mask |= seam
    seed_mask &= free

    seed_labels = labels[seed_mask]
    outside_ids = np.unique(seed_labels[seed_labels > 0]).astype(np.int32)
    outside_mask = np.isin(labels, outside_ids).astype(np.uint8) * 255
    return outside_mask, labels.astype(np.int32), outside_ids


def _extract_outer_touch(
    thin_mask: np.ndarray,
    free_labels: np.ndarray,
    outside_ids: Sequence[int],
) -> np.ndarray:
    thin = np.asarray(thin_mask, dtype=np.uint8) > 0
    if not np.any(thin):
        return np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
    outside_bool = np.isin(free_labels, np.asarray(outside_ids, dtype=np.int32))
    outside_neighbor = cv2.dilate(outside_bool.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    return (thin & outside_neighbor).astype(np.uint8) * 255


def iterative_peel_levels_from_thin_mask(
    thin_mask: np.ndarray,
    roi_mask: np.ndarray,
    max_levels: int = 64,
    repair_callback: Optional[RepairCallback] = None,
) -> Dict[str, object]:
    thin = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    valid = np.asarray(roi_mask, dtype=np.uint8)
    thin[valid == 0] = 0
    zero = np.zeros_like(thin, dtype=np.uint8)

    if not np.any(thin > 0):
        return {
            "status": "empty_mask",
            "message": "No contour pixels remain after thinning.",
            "thin_mask": zero.copy(),
            "classified_thin_mask": zero.copy(),
            "outside_mask": zero.copy(),
            "outer_touch_mask": zero.copy(),
            "l1_map": zero.copy(),
            "l2_map": zero.copy(),
            "level_maps": {1: zero.copy(), 2: zero.copy()},
            "level_label_map": np.zeros_like(thin, dtype=np.int32),
            "region_depth_map": np.zeros_like(thin, dtype=np.int32),
            "residual_map": zero.copy(),
            "max_detected_level": 0,
            "detected_levels": [],
            "unassigned_level_px": 0,
            "recovered_level_px": 0,
            "repair_bridge_count": 0,
            "repair_added_px": 0,
        }

    initial_outside_mask, _labels0, _outside_ids0 = _background_region_labels(thin, valid)
    if not np.any(initial_outside_mask > 0):
        return {
            "status": "no_outside_background",
            "message": "Could not find ROI-connected outside background for iterative peel.",
            "thin_mask": thin.copy(),
            "classified_thin_mask": zero.copy(),
            "outside_mask": zero.copy(),
            "outer_touch_mask": zero.copy(),
            "l1_map": zero.copy(),
            "l2_map": zero.copy(),
            "level_maps": {1: zero.copy(), 2: zero.copy()},
            "level_label_map": np.zeros_like(thin, dtype=np.int32),
            "region_depth_map": np.zeros_like(thin, dtype=np.int32),
            "residual_map": thin.copy(),
            "max_detected_level": 0,
            "detected_levels": [],
            "unassigned_level_px": int(np.count_nonzero(thin)),
            "recovered_level_px": 0,
            "repair_bridge_count": 0,
            "repair_added_px": 0,
        }

    work = thin.copy()
    filled_outside = (initial_outside_mask > 0)
    region_depth_map = np.zeros_like(thin, dtype=np.int32)
    region_depth_map[filled_outside] = 0
    level_label_map = np.zeros_like(thin, dtype=np.int32)
    level_maps: Dict[int, np.ndarray] = {}
    repair_bridge_count = 0
    repair_added_px = 0
    residual_after_l2 = work.copy()
    last_outer_touch = zero.copy()

    for level in range(1, int(max_levels) + 1):
        if not np.any(work > 0):
            break

        outside_mask, free_labels, outside_ids = _background_region_labels(work, valid)
        if not np.any(outside_mask > 0):
            break

        outer_touch = _extract_outer_touch(work, free_labels, outside_ids)
        if not np.any(outer_touch > 0):
            break

        last_outer_touch = outer_touch.copy()
        level_maps[int(level)] = outer_touch.copy()
        level_label_map[outer_touch > 0] = int(level)
        work[outer_touch > 0] = 0

        if repair_callback is not None and np.any(work > 0):
            before = work.copy()
            repaired = dict(repair_callback(work.copy()))
            work = (np.asarray(repaired.get("thin_mask", work), dtype=np.uint8) > 0).astype(np.uint8) * 255
            work[valid == 0] = 0
            repair_bridge_count += int(repaired.get("gap_reconnected_count", 0))
            repair_added_px += max(0, int(np.count_nonzero(work > 0) - np.count_nonzero(before > 0)))

        outside_after, _labels_after, _outside_ids_after = _background_region_labels(work, valid)
        outside_after_bool = outside_after > 0
        newly_open = outside_after_bool & (~filled_outside) & (valid > 0)
        region_depth_map[newly_open] = int(level)
        filled_outside |= outside_after_bool

        if int(level) == 2:
            residual_after_l2 = work.copy()

    detected_levels = sorted(int(v) for v in level_maps.keys())
    max_detected_level = max(detected_levels) if detected_levels else 0
    classified_thin_mask = (level_label_map > 0).astype(np.uint8) * 255
    unassigned_level_px = int(np.count_nonzero((thin > 0) & (level_label_map == 0)))

    if not detected_levels:
        return {
            "status": "no_outer_levels",
            "message": "Iterative peel could not find an outside-touch contour level.",
            "thin_mask": thin.copy(),
            "classified_thin_mask": zero.copy(),
            "outside_mask": initial_outside_mask.astype(np.uint8),
            "outer_touch_mask": zero.copy(),
            "l1_map": zero.copy(),
            "l2_map": zero.copy(),
            "level_maps": {1: zero.copy(), 2: zero.copy()},
            "level_label_map": level_label_map.astype(np.int32),
            "region_depth_map": region_depth_map.astype(np.int32),
            "residual_map": thin.copy(),
            "max_detected_level": 0,
            "detected_levels": [],
            "unassigned_level_px": int(unassigned_level_px),
            "recovered_level_px": int(repair_added_px),
            "repair_bridge_count": int(repair_bridge_count),
            "repair_added_px": int(repair_added_px),
        }

    return {
        "status": "ok",
        "message": (
            f"Iterative peel detected levels up to L{int(max_detected_level)} "
            f"(repair bridges={int(repair_bridge_count)}, repair added px={int(repair_added_px)})."
        ),
        "thin_mask": thin.copy(),
        "classified_thin_mask": classified_thin_mask.astype(np.uint8),
        "outside_mask": initial_outside_mask.astype(np.uint8),
        "outer_touch_mask": last_outer_touch.astype(np.uint8),
        "l1_map": level_maps.get(1, zero.copy()).astype(np.uint8),
        "l2_map": level_maps.get(2, zero.copy()).astype(np.uint8),
        "level_maps": {
            int(level): np.asarray(level_map, dtype=np.uint8)
            for level, level_map in level_maps.items()
        },
        "level_label_map": level_label_map.astype(np.int32),
        "region_depth_map": region_depth_map.astype(np.int32),
        "residual_map": residual_after_l2.astype(np.uint8),
        "max_detected_level": int(max_detected_level),
        "detected_levels": list(detected_levels),
        "unassigned_level_px": int(unassigned_level_px),
        "recovered_level_px": int(repair_added_px),
        "repair_bridge_count": int(repair_bridge_count),
        "repair_added_px": int(repair_added_px),
    }
