from __future__ import annotations

"""Legacy debug map attribute support for the v8 analyzer.

The current Qt workflow does not consume these full-ROI arrays. They are kept
behind an opt-in helper for old notebooks or scripts that accessed the
``line_*`` / debug ``simple_*`` attributes directly.

이 기능은 과거 notebook/script가 ``line_*`` 및 ``simple_*`` debug map 속성에
직접 접근하는 상황을 해결하기 위해 ``legacy_line_maps``로 만들어졌으나,
현재 Qt 분석 workflow가 payload/cache 중심으로 동작하는 상황과 맞지 않아
더 이상 기본 경로에서 쓰이지 않음으로 ``legacy_line_maps``는 legacy로
이동되었다.
"""

from typing import Dict, Tuple

import numpy as np


LEGACY_LINE_MAP_ATTRS = (
    "line_binary_map",
    "line_morph_map",
    "line_centerline_map",
    "line_direct_mask_raw",
    "line_direct_mask_split",
    "line_direct_centerline_map",
    "line_direct_level_map",
    "line_direct_label_map",
    "line_reconstructed_centerline_map",
)

LEGACY_SIMPLE_DEBUG_ATTRS = (
    "simple_clean_mask",
    "simple_thin_mask",
    "simple_l1_map",
    "simple_l2_map",
    "simple_residual_map",
    "simple_level_overlay_map",
)


def clear_legacy_line_map_attributes(target: object) -> None:
    for name in LEGACY_LINE_MAP_ATTRS + LEGACY_SIMPLE_DEBUG_ATTRS:
        setattr(target, name, None)


def populate_legacy_line_map_attributes(
    target: object,
    roi_xywh: Tuple[int, int, int, int],
    payload: Dict[str, object],
) -> None:
    x, y, w, h = [int(v) for v in roi_xywh]
    initial_mask = np.asarray(payload["initial_mask"], dtype=np.uint8)
    cleaned_mask = np.asarray(payload["cleaned_mask"], dtype=np.uint8)
    raw_thin_mask = np.asarray(payload["raw_thin_mask"], dtype=np.uint8)
    final_thin_mask = np.asarray(payload["final_thin_mask"], dtype=np.uint8)
    l1_map = np.asarray(payload["l1_map"], dtype=np.uint8)
    l2_map = np.asarray(payload["l2_map"], dtype=np.uint8)
    level_label_map = np.asarray(payload["level_label_map"], dtype=np.int32)
    region_depth_map = np.asarray(payload["region_depth_map"], dtype=np.int32)
    residual_map = np.asarray(payload["residual_map"], dtype=np.uint8)

    target.simple_clean_mask = cleaned_mask.copy()
    target.simple_thin_mask = raw_thin_mask.copy()
    target.simple_l1_map = l1_map.copy()
    target.simple_l2_map = l2_map.copy()
    target.simple_residual_map = residual_map.copy()
    target.simple_level_overlay_map = np.asarray(
        payload.get("level_overlay_map", np.zeros((h, w, 3), dtype=np.uint8)),
        dtype=np.uint8,
    ).copy()

    target.line_maps_roi_bbox_xywh = (x, y, w, h)
    target.line_binary_map = initial_mask.copy()
    target.line_morph_map = cleaned_mask.copy()
    target.line_centerline_map = final_thin_mask.copy()
    target.line_direct_mask_raw = cleaned_mask.copy()
    target.line_direct_mask_split = l1_map.copy()
    target.line_direct_level_map = level_label_map.copy()
    target.line_direct_label_map = region_depth_map.copy()
    target.line_direct_centerline_map = final_thin_mask.copy()
    target.line_reconstructed_centerline_map = residual_map.copy()
