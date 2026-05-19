from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def build_roi_from_polygon(
    image: np.ndarray,
    points: Sequence[Point],
) -> Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("ROI polygon must contain at least 3 points.")
    img_h, img_w = np.asarray(image).shape[:2]
    x, y, w, h = cv2.boundingRect(pts)
    x0 = int(np.clip(x, 0, img_w))
    y0 = int(np.clip(y, 0, img_h))
    x1 = int(np.clip(x + w, 0, img_w))
    y1 = int(np.clip(y + h, 0, img_h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI polygon is outside image bounds.")
    roi_img = np.asarray(image)[y0:y1, x0:x1].copy()
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_pts = pts - np.asarray([x0, y0], dtype=np.int32)
    cv2.fillPoly(roi_mask, [local_pts], 255)
    return (x0, y0, x1 - x0, y1 - y0), roi_img, roi_mask
