from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr.copy()


def raster_line_points(p0: Point, p1: Point) -> List[Point]:
    x0, y0 = [int(v) for v in p0]
    x1, y1 = [int(v) for v in p1]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    points: List[Point] = []
    while True:
        points.append((int(x), int(y)))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points
