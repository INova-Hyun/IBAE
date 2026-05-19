from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .common.image import ensure_bgr as _ensure_bgr
from .common.numeric import unit_or_none as _unit_or_none_common
from .contours import build_roi_from_polygon as _build_roi_from_polygon
from .contours import iterative_peel_levels_from_thin_mask
from .legacy.line_maps import clear_legacy_line_map_attributes, populate_legacy_line_map_attributes
from .session_replay import build_replay_session_payload, load_replay_session, save_replay_session

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:
    _sk_skeletonize = None

Point = Tuple[int, int]


def _put_panel_title(img: np.ndarray, title: str) -> np.ndarray:
    out = _ensure_bgr(img)
    bar_h = 34
    pad_x = 10
    pad_y = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    target_w = max(20, out.shape[1] - (2 * pad_x))
    while scale > 0.32:
        (text_w, text_h), baseline = cv2.getTextSize(title, font, scale, thickness)
        if text_w <= target_w and (text_h + baseline) <= (bar_h - 6):
            break
        scale -= 0.04
        if scale < 0.48:
            thickness = 1
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (240, 240, 240), -1)
    (_, text_h), _ = cv2.getTextSize(title, font, scale, thickness)
    text_y = min(bar_h - pad_y, max(text_h + 4, 18))
    cv2.putText(out, title, (pad_x, text_y), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)
    return out


def _stack_rows(panels: List[np.ndarray], per_row: int = 3) -> np.ndarray:
    if not panels:
        raise ValueError("panels must not be empty")
    base = _ensure_bgr(panels[0])
    blank = np.full_like(base, 255)
    rows: List[np.ndarray] = []
    for idx in range(0, len(panels), per_row):
        row = [_ensure_bgr(panel) for panel in panels[idx:idx + per_row]]
        while len(row) < per_row:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    return np.vstack(rows)


def select_polygon_roi(*args, **kwargs):
    raise RuntimeError(
        "The legacy OpenCV ROI selector has been removed. "
        "Use the Qt workflow via `run_v9_preview(...)` or provide replay/full-image ROI inputs."
    )


def edit_binary_mask_splits(*args, **kwargs):
    raise RuntimeError(
        "The legacy OpenCV binary split editor has been removed. "
        "Use the Qt workflow via `run_v9_preview(...)`."
    )


def edit_thin_mask_junctions(*args, **kwargs):
    raise RuntimeError(
        "The legacy OpenCV junction editor has been removed. "
        "Use the Qt workflow via `run_v9_preview(...)`."
    )


class JetAnalyzerV8Simple:
    """
    v8 simple contour peel preview.

    Workflow:
    1. Select ROI polygon.
    2. Build and clean a binary contour mask inside the ROI.
    3. Thin the cleaned mask to 1 px.
    4. Resolve local junctions / short open gaps on the thin mask.
    5. Assign contour levels and render a level map preview.
    """

    DEFAULT_PARAMS: Dict[str, object] = {
        "DEBUG_MODE": False,
        "SIMPLE_USE_FULL_IMAGE_ROI": False,
        "SIMPLE_BINARY_PREP_MODE": "manual_threshold",
        "SIMPLE_JUNCTION_RESOLVE_ENABLE": True,
        "SIMPLE_PRETHIN_SPLIT_EDIT_ENABLE": True,
        "SIMPLE_PRETHIN_SPLIT_CUT_WIDTH": 5,
        "SIMPLE_MANUAL_JUNCTION_EDIT_ENABLE": True,
        "SIMPLE_MANUAL_JUNCTION_EDIT_ALWAYS": True,
        "SIMPLE_MANUAL_JUNCTION_CUT_WIDTH": 3,
        "SIMPLE_REPLAY_JSON_LOAD_PATH": "",
        "SIMPLE_REPLAY_JSON_SAVE_PATH": ".ibae_last_replay.json",
        "SIMPLE_REPLAY_STRICT_IMAGE_MATCH": True,
        "SIMPLE_SPUR_PRUNE_ENABLE": True,
        "SIMPLE_SPUR_PRUNE_MAX_PX": 0,
        "SIMPLE_THIN_OPEN_NOISE_MAX_PX": 0,
        "SIMPLE_NODE_POINT_LIMIT": 256,
        "SIMPLE_WINDOW_MAX_WIDTH": 1600,
        "SIMPLE_WINDOW_MAX_HEIGHT": 1000,
        "SIMPLE_MANUAL_GRAY_THRESH": 180,
        "SIMPLE_MANUAL_THRESHOLD_INVERT": False,
        "SIMPLE_JUNCTION_PAIR_MIN_ANGLE_DEG": 110.0,
        "SIMPLE_ENDPOINT_RECONNECT_MAX_ANGLE_DEG": 45.0,
        "SIMPLE_ENDPOINT_SNAP_MAX_ANGLE_DEG": 60.0,
        "SIMPLE_ENDPOINT_RECONNECT_MAX_OVERLAP_FRAC": 0.20,
        "AUTO_CONTOUR_TANGENT_STEPS": 7,
        "AUTO_CONTOUR_MIN_COMPONENT_AREA_FRAC": 0.00015,
        "AUTO_CONTOUR_MIN_COMPONENT_AREA_PX": 10,
        "AUTO_CONTOUR_MIN_COMPONENT_PERIM_PX": 14.0,
        "AUTO_CONTOUR_JUNCTION_ZONE_R": 1,
        "KEEP_LEGACY_LINE_MAPS": False,
    }

    def __init__(self, image_path: str, config: Optional[dict] = None):
        self.image_path = str(image_path)
        self.img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")
        self.clone_base = self.img.copy()
        self.params = dict(self.DEFAULT_PARAMS)
        if config:
            self.params.update(dict(config))
        self.window_name = "JetAnalyzerV8"
        self.results: Dict[str, object] = {}
        self.current_step = 0
        self.points: List[Point] = []
        self.roi_polygon_points: List[Point] = []
        self.final_result_img = None
        self.line_maps_roi_bbox_xywh = None
        self.line_binary_map = None
        self.line_morph_map = None
        self.line_centerline_map = None
        self.line_direct_mask_raw = None
        self.line_direct_mask_split = None
        self.line_direct_centerline_map = None
        self.line_direct_level_map = None
        self.line_direct_label_map = None
        self.line_reconstructed_centerline_map = None
        self.simple_clean_mask = None
        self.simple_thin_mask = None
        self.simple_final_thin_mask = None
        self.simple_l1_map = None
        self.simple_l2_map = None
        self.simple_residual_map = None
        self.simple_roi_view = None
        self.simple_roi_mask = None
        self.simple_region_depth_map = None
        self.simple_level_overlay_map = None
        self.simple_binary_prep_mode = "unknown"
        self.simple_manual_junction_cuts = []
        self._last_cancel_reason = ""
        self._skip_next_manual_threshold_tuner = False
        self._force_skip_binary_split_once = False
        self._force_reopen_binary_split_once = False
        self._force_reopen_junction_editor_once = False
        self._manual_edit_session_override: Optional[Dict[str, object]] = None
        self._loaded_replay_json_path: str = ""

    def _reset_all(self):
        self.results = {}
        self.current_step = 0
        self.points = []
        self.roi_polygon_points = []
        self.final_result_img = None
        self.line_maps_roi_bbox_xywh = None
        self.line_binary_map = None
        self.line_morph_map = None
        self.line_centerline_map = None
        self.line_direct_mask_raw = None
        self.line_direct_mask_split = None
        self.line_direct_centerline_map = None
        self.line_direct_level_map = None
        self.line_direct_label_map = None
        self.line_reconstructed_centerline_map = None
        self.simple_clean_mask = None
        self.simple_thin_mask = None
        self.simple_final_thin_mask = None
        self.simple_l1_map = None
        self.simple_l2_map = None
        self.simple_residual_map = None
        self.simple_roi_view = None
        self.simple_roi_mask = None
        self.simple_region_depth_map = None
        self.simple_level_overlay_map = None
        self.simple_binary_prep_mode = "unknown"
        self.simple_manual_junction_cuts = []
        self._last_cancel_reason = ""
        self._skip_next_manual_threshold_tuner = False
        self._force_skip_binary_split_once = False
        self._force_reopen_binary_split_once = False
        self._force_reopen_junction_editor_once = False
        self._manual_edit_session_override = None
        self._loaded_replay_json_path = ""

    def _reset_step(self):
        self._reset_all()

    def _draw_current_step(self, img):
        frame = _ensure_bgr(img)
        cv2.putText(
            frame,
            "[Step 1] Select ROI (Polygon -> Enter)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "After ROI: threshold binary -> normalize -> manual binary split -> thin -> spur prune -> manual junction split -> level map",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        if self.points:
            cv2.polylines(frame, [np.asarray(self.points, dtype=np.int32)], False, (0, 255, 0), 2)
            for pt in self.points:
                cv2.circle(frame, tuple(map(int, pt)), 3, (0, 0, 255), -1)
        return

    @staticmethod
    def _full_mask_border(mask: np.ndarray) -> np.ndarray:
        border = np.zeros_like(mask, dtype=bool)
        if border.size == 0:
            return border
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        return border

    @staticmethod
    def _unit_or_none(vec: Sequence[float]) -> Optional[np.ndarray]:
        return _unit_or_none_common(vec)

    def _simple_use_full_image_roi(self) -> bool:
        return bool(self.params.get("SIMPLE_USE_FULL_IMAGE_ROI", False))

    def _simple_binary_prep_setting(self) -> str:
        mode = str(self.params.get("SIMPLE_BINARY_PREP_MODE", "auto")).strip().lower()
        if mode == "manual":
            return "manual_threshold"
        if mode == "manual_threshold":
            return "manual_threshold"
        return "auto_threshold"

    def _simple_junction_resolve_enabled(self) -> bool:
        return bool(self.params.get("SIMPLE_JUNCTION_RESOLVE_ENABLE", True))

    def _simple_prethin_split_edit_enabled(self) -> bool:
        return bool(self.params.get("SIMPLE_PRETHIN_SPLIT_EDIT_ENABLE", True))

    def _simple_prethin_split_cut_width(self) -> int:
        return int(max(1, round(self.params.get("SIMPLE_PRETHIN_SPLIT_CUT_WIDTH", 5))))

    def _simple_manual_junction_edit_enabled(self) -> bool:
        return bool(self.params.get("SIMPLE_MANUAL_JUNCTION_EDIT_ENABLE", True))

    def _simple_manual_junction_edit_always(self) -> bool:
        return bool(self.params.get("SIMPLE_MANUAL_JUNCTION_EDIT_ALWAYS", False))

    def _simple_manual_junction_cut_width(self) -> int:
        return int(max(1, round(self.params.get("SIMPLE_MANUAL_JUNCTION_CUT_WIDTH", 3))))

    def _simple_replay_json_load_path(self) -> str:
        return str(self.params.get("SIMPLE_REPLAY_JSON_LOAD_PATH", "") or "").strip()

    def _simple_replay_json_save_path(self) -> str:
        return str(self.params.get("SIMPLE_REPLAY_JSON_SAVE_PATH", "") or "").strip()

    def _simple_replay_strict_image_match(self) -> bool:
        return bool(self.params.get("SIMPLE_REPLAY_STRICT_IMAGE_MATCH", True))

    def _simple_spur_prune_enabled(self) -> bool:
        return bool(self.params.get("SIMPLE_SPUR_PRUNE_ENABLE", True))

    def _simple_spur_prune_max_px(self, mean_width_px: float) -> int:
        configured = int(max(0, round(self.params.get("SIMPLE_SPUR_PRUNE_MAX_PX", 0))))
        if configured > 0:
            return configured
        auto = int(max(2, round(float(mean_width_px) * 1.5)))
        return int(min(auto, 16))

    def _simple_tangent_steps(self) -> int:
        return int(max(3, round(self.params.get("AUTO_CONTOUR_TANGENT_STEPS", 7))))

    def _simple_pair_min_angle_deg(self) -> float:
        return float(max(0.0, self.params.get("SIMPLE_JUNCTION_PAIR_MIN_ANGLE_DEG", 110.0)))

    def _simple_endpoint_reconnect_angle_deg(self) -> float:
        return float(max(0.0, self.params.get("SIMPLE_ENDPOINT_RECONNECT_MAX_ANGLE_DEG", 45.0)))

    def _simple_endpoint_snap_angle_deg(self) -> float:
        # A small relaxation helps with aliased 1 px line-art endpoints.
        return float(max(0.0, self.params.get("SIMPLE_ENDPOINT_SNAP_MAX_ANGLE_DEG", 60.0)))

    def _simple_bridge_overlap_max(self) -> float:
        return float(np.clip(self.params.get("SIMPLE_ENDPOINT_RECONNECT_MAX_OVERLAP_FRAC", 0.20), 0.0, 1.0))

    @staticmethod
    def _apply_cut_records(mask: np.ndarray, cuts: Sequence[Dict[str, object]]) -> np.ndarray:
        out = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        h, w = out.shape[:2]
        for cut in cuts:
            if not isinstance(cut, dict):
                continue
            start = cut.get("start")
            end = cut.get("end", start)
            if not isinstance(start, (list, tuple)) or len(start) < 2:
                continue
            if not isinstance(end, (list, tuple)) or len(end) < 2:
                end = start
            sx, sy = int(start[0]), int(start[1])
            ex, ey = int(end[0]), int(end[1])
            width = int(max(1, cut.get("width", 1)))
            if str(cut.get("kind", "line")) == "point" or (sx == ex and sy == ey):
                size = int(max(1, width))
                half_lo = int((size - 1) // 2)
                half_hi = int(size // 2)
                x0 = int(np.clip(sx - half_lo, 0, max(0, w - 1)))
                y0 = int(np.clip(sy - half_lo, 0, max(0, h - 1)))
                x1 = int(np.clip(sx + half_hi, 0, max(0, w - 1)))
                y1 = int(np.clip(sy + half_hi, 0, max(0, h - 1)))
                out[y0:y1 + 1, x0:x1 + 1] = 0
            else:
                cv2.line(out, (sx, sy), (ex, ey), 0, int(width), cv2.LINE_8)
        return out

    def _manual_edit_session_roi_points(self) -> List[List[int]]:
        if self.roi_polygon_points:
            return [[int(x), int(y)] for x, y in self.roi_polygon_points]
        h, w = self.img.shape[:2]
        return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]

    def _load_replay_session_from_config(self) -> Optional[Dict[str, object]]:
        path = self._simple_replay_json_load_path()
        if not path:
            return None
        session = load_replay_session(path, expected_image_path=self.image_path)
        image_path = str(session.get("image_path", "") or "")
        if self._simple_replay_strict_image_match():
            expected = str(Path(self.image_path).resolve())
            if image_path and image_path != expected:
                raise ValueError(
                    f"Replay JSON image mismatch: {image_path} != {expected}"
                )
        self._loaded_replay_json_path = str(path)
        return dict(session)

    def _current_replay_payload(
        self,
        binary_prep_mode: str,
        binary_split_cuts: Sequence[Dict[str, object]],
        junction_cuts: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        thresh = None
        invert = None
        if str(binary_prep_mode) == "manual_threshold":
            thresh = int(self.params.get("SIMPLE_MANUAL_GRAY_THRESH", 180))
            invert = bool(self.params.get("SIMPLE_MANUAL_THRESHOLD_INVERT", False))
        return build_replay_session_payload(
            image_path=self.image_path,
            roi_points_xy=self._manual_edit_session_roi_points(),
            binary_prep_mode=str(binary_prep_mode),
            manual_gray_thresh=thresh,
            manual_invert=invert,
            binary_split_cuts=binary_split_cuts,
            junction_cuts=junction_cuts,
        )

    @staticmethod
    def _fit_image_to_box(
        image: np.ndarray,
        max_w: int,
        max_h: int,
        allow_upscale: bool = False,
    ) -> np.ndarray:
        img = _ensure_bgr(image)
        h, w = img.shape[:2]
        max_w = int(max(64, max_w))
        max_h = int(max(64, max_h))
        scale = min(float(max_w) / max(float(w), 1.0), float(max_h) / max(float(h), 1.0))
        if not allow_upscale:
            scale = min(1.0, scale)
        if 0.999 <= scale <= 1.001:
            return img.copy()
        out_w = max(1, int(round(float(w) * scale)))
        out_h = max(1, int(round(float(h) * scale)))
        interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        return cv2.resize(img, (out_w, out_h), interpolation=interp)

    @staticmethod
    def _pad_image_to_size(image: np.ndarray, target_w: int, target_h: int, bg: int = 255) -> np.ndarray:
        img = _ensure_bgr(image)
        h, w = img.shape[:2]
        target_w = int(max(target_w, w))
        target_h = int(max(target_h, h))
        canvas = np.full((target_h, target_w, 3), int(bg), dtype=np.uint8)
        y0 = int(max(0, (target_h - h) // 2))
        x0 = int(max(0, (target_w - w) // 2))
        canvas[y0:y0 + h, x0:x0 + w] = img
        return canvas

    @staticmethod
    def _disk_kernel(radius: int) -> np.ndarray:
        radius = int(max(1, radius))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    @staticmethod
    def _normalize_binary_mask(mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        return (((np.asarray(mask, dtype=np.uint8) > 0) & (np.asarray(roi_mask, dtype=np.uint8) > 0)).astype(np.uint8) * 255)

    @staticmethod
    def _dedupe_consecutive_points(points: Sequence[Point]) -> List[Point]:
        out: List[Point] = []
        prev: Optional[Point] = None
        for x, y in points:
            pt = (int(x), int(y))
            if pt != prev:
                out.append(pt)
                prev = pt
        return out

    @staticmethod
    def _unit_vector(vec: Sequence[float]) -> Optional[np.ndarray]:
        return _unit_or_none_common(vec)

    def _skeletonize_fallback(self, binary_roi: np.ndarray) -> np.ndarray:
        mask = (np.asarray(binary_roi, dtype=np.uint8) > 0).astype(np.uint8) * 255
        skel = np.zeros_like(mask, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        work = mask.copy()
        while np.any(work > 0):
            opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(work, opened)
            eroded = cv2.erode(work, kernel)
            skel = cv2.bitwise_or(skel, temp)
            if np.array_equal(eroded, work):
                break
            work = eroded
        return skel

    def _thin_binary(self, binary_roi: np.ndarray) -> np.ndarray:
        mask = (np.asarray(binary_roi, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.uint8)
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            try:
                return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            except Exception:
                pass
        if _sk_skeletonize is not None:
            try:
                return (_sk_skeletonize(mask > 0).astype(np.uint8) * 255)
            except Exception:
                pass
        return self._skeletonize_fallback(mask)

    @staticmethod
    def _skeleton_neighbors(mask: np.ndarray, x: int, y: int) -> List[Point]:
        h, w = mask.shape[:2]
        out: List[Point] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                xx = x + dx
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h and bool(mask[yy, xx]):
                    out.append((xx, yy))
        return out

    @staticmethod
    def _edge_key(a: Point, b: Point) -> Tuple[Point, Point]:
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _neighbor_stats_maps(centerline_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        skel = (np.asarray(centerline_roi, dtype=np.uint8) > 0).astype(np.uint8)
        neighbor_count = np.zeros_like(skel, dtype=np.uint8)
        branch_group_count = np.zeros_like(skel, dtype=np.uint8)
        ys, xs = np.where(skel > 0)
        for x, y in zip(xs.tolist(), ys.tolist()):
            occupied: List[Tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = int(x + dx)
                    yy = int(y + dy)
                    if 0 <= xx < skel.shape[1] and 0 <= yy < skel.shape[0] and skel[yy, xx] > 0:
                        occupied.append((dx, dy))
            neighbor_count[y, x] = np.uint8(len(occupied))
            if not occupied:
                continue
            occupied_set = set(occupied)
            visited: Set[Tuple[int, int]] = set()
            groups = 0
            for start in occupied:
                if start in visited:
                    continue
                groups += 1
                stack = [start]
                visited.add(start)
                while stack:
                    cx, cy = stack.pop()
                    for ndy in (-1, 0, 1):
                        for ndx in (-1, 0, 1):
                            if ndx == 0 and ndy == 0:
                                continue
                            nx = cx + ndx
                            ny = cy + ndy
                            if abs(nx) > 1 or abs(ny) > 1 or (nx == 0 and ny == 0):
                                continue
                            nxt = (nx, ny)
                            if nxt in occupied_set and nxt not in visited:
                                visited.add(nxt)
                                stack.append(nxt)
            branch_group_count[y, x] = np.uint8(groups)
        return neighbor_count, branch_group_count

    def _extract_centerline_paths(self, centerline_roi: np.ndarray) -> Dict[str, object]:
        skel = (np.asarray(centerline_roi, dtype=np.uint8) > 0).astype(np.uint8)
        neigh_count, branch_group_count = self._neighbor_stats_maps(centerline_roi)
        endpoint_mask = (skel > 0) & (neigh_count == 1)
        junction_mask = (skel > 0) & (branch_group_count >= 3)
        node_mask = endpoint_mask | junction_mask

        visited_edges: Set[Tuple[Point, Point]] = set()
        open_paths: List[List[Point]] = []
        loop_paths: List[List[Point]] = []

        ys, xs = np.where(node_mask)
        nodes = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        for node in nodes:
            for nb in self._skeleton_neighbors(skel, node[0], node[1]):
                edge = self._edge_key(node, nb)
                if edge in visited_edges:
                    continue
                path = [node, nb]
                visited_edges.add(edge)
                prev = node
                cur = nb
                while True:
                    if node_mask[cur[1], cur[0]] and cur != node:
                        break
                    next_pts = [p for p in self._skeleton_neighbors(skel, cur[0], cur[1]) if p != prev]
                    next_unvisited = [p for p in next_pts if self._edge_key(cur, p) not in visited_edges]
                    if next_unvisited:
                        nxt = next_unvisited[0]
                    elif next_pts:
                        nxt = next_pts[0]
                    else:
                        break
                    edge2 = self._edge_key(cur, nxt)
                    if edge2 in visited_edges:
                        break
                    path.append(nxt)
                    visited_edges.add(edge2)
                    prev, cur = cur, nxt
                if len(path) > 1:
                    open_paths.append(self._dedupe_consecutive_points(path))

        ys_all, xs_all = np.where(skel > 0)
        for x0, y0 in zip(xs_all.tolist(), ys_all.tolist()):
            start = (int(x0), int(y0))
            if node_mask[start[1], start[0]]:
                continue
            nbs = self._skeleton_neighbors(skel, start[0], start[1])
            if len(nbs) != 2:
                continue
            if all(self._edge_key(start, nb) in visited_edges for nb in nbs):
                continue
            path = [start]
            prev = None
            cur = start
            while True:
                next_pts = [p for p in self._skeleton_neighbors(skel, cur[0], cur[1]) if p != prev]
                next_unvisited = [p for p in next_pts if self._edge_key(cur, p) not in visited_edges]
                if next_unvisited:
                    nxt = next_unvisited[0]
                elif next_pts:
                    nxt = next_pts[0]
                else:
                    break
                edge = self._edge_key(cur, nxt)
                if edge in visited_edges and nxt == start:
                    path.append(nxt)
                    break
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == start:
                    break
            if len(path) > 3 and path[-1] == start:
                loop_paths.append(self._dedupe_consecutive_points(path))

        return {
            "skel": skel,
            "neighbor_count": neigh_count,
            "branch_group_count": branch_group_count,
            "endpoint_mask": endpoint_mask,
            "junction_mask": junction_mask,
            "node_mask": node_mask,
            "open_paths": open_paths,
            "loop_paths": loop_paths,
            "merge_zone_stitch_count": 0,
            "stitched_open_count": 0,
            "rescued_loop_count": 0,
        }

    @staticmethod
    def _raster_line_points(p0: Point, p1: Point) -> List[Point]:
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        n = int(max(abs(dx), abs(dy)))
        if n <= 0:
            return [p0]
        xs = np.linspace(float(p0[0]), float(p1[0]), n + 1)
        ys = np.linspace(float(p0[1]), float(p1[1]), n + 1)
        out: List[Point] = []
        prev: Optional[Point] = None
        for x, y in zip(xs, ys):
            pt = (int(round(x)), int(round(y)))
            if pt != prev:
                out.append(pt)
                prev = pt
        return out

    def _open_path_endpoint_record(self, path: List[Point], side: str, tangent_steps: int) -> Optional[Dict[str, object]]:
        if len(path) < 2:
            return None
        tangent_steps = int(max(1, tangent_steps))
        if side == "start":
            pt = path[0]
            far = path[min(len(path) - 1, tangent_steps)]
        else:
            pt = path[-1]
            far = path[max(0, len(path) - 1 - tangent_steps)]
        tangent = np.array([pt[0] - far[0], pt[1] - far[1]], dtype=np.float64)
        unit = self._unit_vector(tangent)
        if unit is None:
            return None
        return {"point": (int(pt[0]), int(pt[1])), "dir": unit, "side": side}

    def _masked_roi_gray_view(self, roi_img: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        img = np.asarray(roi_img, dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        white_bg = np.full_like(gray_bgr, 255)
        mask = np.asarray(roi_mask, dtype=np.uint8)
        return np.where(mask[:, :, None] == 255, gray_bgr, white_bg)

    def _component_filter_by_area_and_perimeter(self, binary_roi: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        mask = (
            (np.asarray(binary_roi, dtype=np.uint8) > 0)
            & (np.asarray(roi_mask, dtype=np.uint8) > 0)
        ).astype(np.uint8)
        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.uint8)

        roi_area = int(np.count_nonzero(roi_mask))
        min_area = int(
            max(
                self.params.get("AUTO_CONTOUR_MIN_COMPONENT_AREA_PX", 10),
                round(float(self.params.get("AUTO_CONTOUR_MIN_COMPONENT_AREA_FRAC", 0.00015)) * max(roi_area, 1)),
            )
        )
        min_perim = float(max(6.0, self.params.get("AUTO_CONTOUR_MIN_COMPONENT_PERIM_PX", 14.0)))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask, dtype=np.uint8)
        for idx in range(1, int(n_labels)):
            comp = (labels == idx).astype(np.uint8)
            area = int(stats[idx, cv2.CC_STAT_AREA])
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            perim = float(sum(cv2.arcLength(cnt, False) for cnt in cnts))
            if area >= min_area or perim >= min_perim:
                out[comp > 0] = 255
        out[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
        return out

    def _estimate_branch_side_direction(self, points: Sequence[Point], side: str, tangent_steps: int) -> Optional[np.ndarray]:
        pts = self._dedupe_consecutive_points(points)
        if len(pts) < 2:
            return None
        tangent_steps = int(max(1, tangent_steps))
        if side == "start":
            far = pts[min(len(pts) - 1, tangent_steps)]
            near = pts[0]
        else:
            far = pts[max(0, len(pts) - 1 - tangent_steps)]
            near = pts[-1]
        vec = np.array([far[0] - near[0], far[1] - near[1]], dtype=np.float64)
        return self._unit_or_none(vec)

    @staticmethod
    def _junction_anchor(mask: np.ndarray) -> Point:
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            return (0, 0)
        return (int(round(float(np.mean(xx)))), int(round(float(np.mean(yy)))))

    @staticmethod
    def _adjacent_label_id(point: Point, label_map: np.ndarray) -> Optional[int]:
        x, y = int(point[0]), int(point[1])
        h, w = label_map.shape[:2]
        for yy in range(max(0, y - 1), min(h, y + 2)):
            for xx in range(max(0, x - 1), min(w, x + 2)):
                val = int(label_map[yy, xx])
                if val > 0:
                    return val
        return None

    def _extract_branch_records(
        self,
        centerline_roi: np.ndarray,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[int, Dict[str, object]]]:
        skel = (np.asarray(centerline_roi, dtype=np.uint8) > 0).astype(np.uint8)
        if not np.any(skel):
            return [], [], {}
        neighbor_count, branch_group_count = self._neighbor_stats_maps(centerline_roi)
        junction_mask = (skel > 0) & (branch_group_count >= 3)

        n_junc, junc_labels, _, _ = cv2.connectedComponentsWithStats(junction_mask.astype(np.uint8), connectivity=8)
        zone_r = int(max(1, round(self.params.get("AUTO_CONTOUR_JUNCTION_ZONE_R", 1))))
        zone_kernel = self._disk_kernel(zone_r)
        junction_zone_map = np.zeros_like(junc_labels, dtype=np.int32)
        junction_meta: Dict[int, Dict[str, object]] = {}
        for j in range(1, int(n_junc)):
            comp = (junc_labels == j).astype(np.uint8)
            zone = cv2.dilate(comp, zone_kernel, iterations=1) > 0
            zone &= skel > 0
            junction_zone_map[(zone) & (junction_zone_map == 0)] = int(j)
            zone_mask = zone.astype(bool)
            junction_meta[int(j)] = {
                "id": int(j),
                "mask": comp.astype(bool),
                "zone_mask": zone_mask,
                "anchor": self._junction_anchor(zone_mask),
            }

        residual = ((skel > 0) & (junction_zone_map == 0)).astype(np.uint8)
        n_comp, comp_labels, _, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)

        branches: List[Dict[str, object]] = []
        loops: List[Dict[str, object]] = []
        tangent_steps = int(max(2, round(self.params.get("AUTO_CONTOUR_TANGENT_STEPS", 7))))
        for comp_id in range(1, int(n_comp)):
            comp_mask = (comp_labels == comp_id).astype(np.uint8)
            if not np.any(comp_mask):
                continue
            path_info = self._extract_centerline_paths(comp_mask * 255)
            open_paths = list(path_info.get("open_paths", []))
            loop_paths = list(path_info.get("loop_paths", []))

            if loop_paths and not open_paths:
                path = max(loop_paths, key=len)
                loops.append({"branch_id": int(comp_id), "points": self._dedupe_consecutive_points(path), "closed": True})
                continue

            if open_paths:
                path = max(open_paths, key=len)
            else:
                yy, xx = np.nonzero(comp_mask)
                path = [(int(xx[0]), int(yy[0]))] if len(xx) else []
            path = self._dedupe_consecutive_points(path)
            if not path:
                continue
            start_pt = path[0]
            end_pt = path[-1]
            start_junc = self._adjacent_label_id(start_pt, junction_zone_map)
            end_junc = self._adjacent_label_id(end_pt, junction_zone_map)
            branches.append(
                {
                    "branch_id": int(comp_id),
                    "points": path,
                    "closed": False,
                    "start_point": start_pt,
                    "end_point": end_pt,
                    "start_junction": start_junc,
                    "end_junction": end_junc,
                    "start_is_terminal": bool(start_junc is None),
                    "end_is_terminal": bool(end_junc is None),
                    "start_dir": self._estimate_branch_side_direction(path, "start", tangent_steps),
                    "end_dir": self._estimate_branch_side_direction(path, "end", tangent_steps),
                }
            )
        return branches, loops, junction_meta

    def _cleanup_polyline_geometry(self, points: Sequence[Point], closed: bool) -> List[Point]:
        pts = self._dedupe_consecutive_points(points)
        if closed and pts and pts[0] != pts[-1]:
            pts = list(pts) + [pts[0]]
        return pts

    def _rasterize_labeled_centerlines(
        self,
        shape: Tuple[int, int],
        polylines: Sequence[Dict[str, object]],
    ) -> Tuple[np.ndarray, np.ndarray, Set[int]]:
        h, w = [int(v) for v in shape]
        label_map = np.zeros((h, w), dtype=np.int32)
        binary = np.zeros((h, w), dtype=np.uint8)
        overlap_labels: Set[int] = set()
        for item in polylines:
            label = int(item["label"])
            pts = self._dedupe_consecutive_points(item["points_xy"])
            if len(pts) < 2:
                continue
            temp = np.zeros((h, w), dtype=np.uint8)
            for i in range(len(pts) - 1):
                cv2.line(temp, tuple(pts[i]), tuple(pts[i + 1]), 255, 1, cv2.LINE_8)
            overlap = (temp > 0) & (label_map > 0) & (label_map != label)
            if np.any(overlap):
                overlap_labels.add(label)
                overlap_labels.update(int(v) for v in np.unique(label_map[overlap]) if int(v) > 0)
            write_mask = (temp > 0) & (label_map == 0)
            label_map[write_mask] = label
            binary[temp > 0] = 255
        return label_map, binary, overlap_labels

    def _prepare_auto_threshold_mask(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
        img = np.asarray(roi_img, dtype=np.uint8)
        valid = np.asarray(roi_mask, dtype=np.uint8) > 0
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        values = np.asarray(gray, dtype=np.uint8)[valid]
        if values.size <= 0:
            zero = np.zeros_like(roi_mask, dtype=np.uint8)
            return zero, {"mode": "auto_threshold", "threshold": 0, "invert": False}, zero.astype(np.float32)
        try:
            gray_thresh = int(np.clip(round(threshold_otsu(values)), 0, 255))
        except Exception:
            gray_thresh = int(np.clip(round(float(np.median(values))), 0, 255))

        def _candidate(invert: bool) -> Tuple[float, np.ndarray, float, float]:
            if invert:
                mask_bool = (np.asarray(gray, dtype=np.uint8) <= int(gray_thresh)) & valid
            else:
                mask_bool = (np.asarray(gray, dtype=np.uint8) >= int(gray_thresh)) & valid
            mask_u8 = mask_bool.astype(np.uint8) * 255
            fg_frac = float(np.count_nonzero(mask_bool)) / float(max(1, np.count_nonzero(valid)))
            dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 3)
            mean_width = float(np.mean(dist[mask_u8 > 0]) * 2.0) if np.any(mask_u8 > 0) else 0.0
            score = abs(fg_frac - 0.08) + 0.04 * abs(mean_width - 4.0)
            if fg_frac < 0.001:
                score += 2.0
            if fg_frac > 0.45:
                score += 1.5
            return float(score), mask_u8, float(fg_frac), float(mean_width)

        cand_light = _candidate(invert=False)
        cand_dark = _candidate(invert=True)
        score, mask_u8, fg_frac, mean_width = cand_dark if cand_dark[0] < cand_light[0] else cand_light
        invert = bool(cand_dark[0] < cand_light[0])
        mask_u8 = self._normalize_binary_mask(mask_u8, roi_mask)
        evidence = (mask_u8 > 0).astype(np.float32)
        info = {
            "mode": "auto_threshold",
            "threshold": int(gray_thresh),
            "invert": bool(invert),
            "foreground_frac": float(fg_frac),
            "mean_width_px": float(mean_width),
            "score": float(score),
        }
        return mask_u8, info, evidence

    def _prepare_manual_threshold_mask(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
        gray_thresh: Optional[int] = None,
        invert: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
        img = np.asarray(roi_img, dtype=np.uint8)
        valid = np.asarray(roi_mask, dtype=np.uint8) > 0
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray_thresh is None:
            gray_thresh = int(np.clip(round(self.params.get("SIMPLE_MANUAL_GRAY_THRESH", 180)), 0, 255))
        else:
            gray_thresh = int(np.clip(round(gray_thresh), 0, 255))
        if invert is None:
            invert = bool(self.params.get("SIMPLE_MANUAL_THRESHOLD_INVERT", False))
        else:
            invert = bool(invert)
        if invert:
            mask_bool = (np.asarray(gray, dtype=np.uint8) <= int(gray_thresh)) & valid
        else:
            mask_bool = (np.asarray(gray, dtype=np.uint8) >= int(gray_thresh)) & valid

        mask_u8 = self._normalize_binary_mask(mask_bool.astype(np.uint8) * 255, roi_mask)
        evidence = mask_bool.astype(np.float32)
        info = {
            "mode": "manual_threshold",
            "threshold": int(gray_thresh),
            "invert": bool(invert),
        }
        return mask_u8, info, evidence

    def _compose_manual_threshold_tuning_preview(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
        gray_thresh: int,
        invert: bool,
    ) -> np.ndarray:
        mask_u8, _, _ = self._prepare_manual_threshold_mask(
            roi_img=roi_img,
            roi_mask=roi_mask,
            gray_thresh=int(gray_thresh),
            invert=bool(invert),
        )
        roi_view = self._masked_roi_gray_view(roi_img=roi_img, roi_mask=roi_mask)
        panels = [
            _put_panel_title(np.asarray(roi_view, dtype=np.uint8), "1. ROI View"),
            _put_panel_title(_ensure_bgr(mask_u8), "2. Manual Threshold Mask"),
        ]
        target_w = max(panel.shape[1] for panel in panels)
        target_h = max(panel.shape[0] for panel in panels)
        fitted: List[np.ndarray] = []
        for panel in panels:
            img = self._fit_image_to_box(panel, target_w, target_h)
            fitted.append(self._pad_image_to_size(img, target_w, target_h, bg=255))
        return _stack_rows(fitted, per_row=2)

    def _run_manual_threshold_tuner(self, roi_img: np.ndarray, roi_mask: np.ndarray) -> bool:
        window_name = "JetAnalyzerV8 Manual Threshold"
        thresh0 = int(np.clip(round(self.params.get("SIMPLE_MANUAL_GRAY_THRESH", 180)), 0, 255))
        invert0 = 1 if bool(self.params.get("SIMPLE_MANUAL_THRESHOLD_INVERT", False)) else 0
        keep_ratio_flag = getattr(cv2, "WINDOW_KEEPRATIO", cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | keep_ratio_flag)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except Exception:
            pass
        cv2.createTrackbar("Threshold", window_name, int(thresh0), 255, lambda _v: None)
        cv2.createTrackbar("Invert", window_name, int(invert0), 1, lambda _v: None)

        print("Manual threshold window: adjust Threshold/Invert, then Enter/Space to confirm, q/Esc to cancel.")
        last_state: Optional[Tuple[int, int]] = None
        preview = self._compose_manual_threshold_tuning_preview(
            roi_img=roi_img,
            roi_mask=roi_mask,
            gray_thresh=thresh0,
            invert=bool(invert0),
        )
        disp = self._fit_image_to_box(
            preview,
            self._simple_window_max_width(),
            self._simple_window_max_height(),
            allow_upscale=False,
        )
        cv2.resizeWindow(window_name, int(disp.shape[1]), int(disp.shape[0]))
        accepted = False
        try:
            while True:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    pass

                gray_thresh = int(np.clip(cv2.getTrackbarPos("Threshold", window_name), 0, 255))
                invert = int(np.clip(cv2.getTrackbarPos("Invert", window_name), 0, 1))
                state = (gray_thresh, invert)
                if state != last_state or preview is None:
                    preview = self._compose_manual_threshold_tuning_preview(
                        roi_img=roi_img,
                        roi_mask=roi_mask,
                        gray_thresh=gray_thresh,
                        invert=bool(invert),
                    )
                    disp = self._fit_image_to_box(
                        preview,
                        self._simple_window_max_width(),
                        self._simple_window_max_height(),
                        allow_upscale=False,
                    )
                    cv2.imshow(window_name, disp)
                    last_state = state
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 10, 32):
                    self.params["SIMPLE_MANUAL_GRAY_THRESH"] = int(gray_thresh)
                    self.params["SIMPLE_MANUAL_THRESHOLD_INVERT"] = bool(invert)
                    print(
                        f"-> Manual threshold confirmed: threshold={int(gray_thresh)}, invert={bool(invert)}"
                    )
                    accepted = True
                    break
                if key in (27, ord("q")):
                    break
        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
        if not accepted:
            print("-> Manual threshold tuning cancelled.")
            self._last_cancel_reason = "Manual threshold tuning cancelled."
        return accepted

    def _prepare_simple_binary_mask(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object], np.ndarray, str]:
        prep_mode = self._simple_binary_prep_setting()
        if prep_mode == "manual_threshold":
            initial_mask, binary_info, evidence_map = self._prepare_manual_threshold_mask(
                roi_img=roi_img,
                roi_mask=roi_mask,
            )
            return initial_mask, binary_info, evidence_map, "manual_threshold"
        initial_mask, binary_info, evidence_map = self._prepare_auto_threshold_mask(
            roi_img=roi_img,
            roi_mask=roi_mask,
        )
        return initial_mask, binary_info, evidence_map, "auto_threshold"

    def _outside_background_mask(self, thin_mask: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        outside_mask, _, _ = self._background_region_labels(thin_mask=thin_mask, roi_mask=roi_mask)
        return outside_mask

    @staticmethod
    def _local_positive_region_ids(label_map: np.ndarray, x: int, y: int) -> List[int]:
        y0 = max(0, int(y) - 1)
        y1 = min(label_map.shape[0], int(y) + 2)
        x0 = max(0, int(x) - 1)
        x1 = min(label_map.shape[1], int(x) + 2)
        return [int(v) for v in np.unique(label_map[y0:y1, x0:x1]) if int(v) > 0]

    def _background_region_labels(
        self,
        thin_mask: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        thin = np.asarray(thin_mask, dtype=np.uint8) > 0
        valid = np.asarray(roi_mask, dtype=np.uint8) > 0
        free = valid & (~thin)
        if not np.any(free):
            zero = np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
            return zero, np.zeros_like(zero, dtype=np.int32), np.zeros(0, dtype=np.int32)

        # Use 4-connectivity for background so 1 px diagonal contour corners stay sealed.
        n_labels, labels, _, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
        if int(n_labels) <= 1:
            zero = np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
            return zero, labels.astype(np.int32), np.zeros(0, dtype=np.int32)

        seed_mask = self._full_mask_border(free)
        invalid = ~valid
        if np.any(invalid):
            seam = cv2.dilate(invalid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
            seed_mask |= seam
        seed_mask &= free
        outside_ids = [int(v) for v in np.unique(labels[seed_mask]) if int(v) > 0]
        if not outside_ids:
            zero = np.zeros_like(np.asarray(thin_mask, dtype=np.uint8), dtype=np.uint8)
            return zero, labels.astype(np.int32), np.zeros(0, dtype=np.int32)

        outside = np.isin(labels, np.asarray(outside_ids, dtype=np.int32))
        return (
            outside.astype(np.uint8) * 255,
            labels.astype(np.int32),
            np.asarray(outside_ids, dtype=np.int32),
        )

    def _simple_open_noise_max_px(self) -> int:
        return int(max(0, round(self.params.get("SIMPLE_THIN_OPEN_NOISE_MAX_PX", 0))))

    def _simple_node_point_limit(self) -> int:
        return int(max(0, round(self.params.get("SIMPLE_NODE_POINT_LIMIT", 256))))

    def _filter_small_open_components(
        self,
        thin_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        max_px = self._simple_open_noise_max_px()
        if max_px <= 0 or not np.any(thin_u8):
            return thin_u8, {
                "open_noise_max_px": int(max_px),
                "removed_open_noise_component_count": 0,
                "removed_open_noise_px": 0,
            }

        neigh, _branch = self._neighbor_stats_maps(thin_u8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((thin_u8 > 0).astype(np.uint8), connectivity=8)
        filtered = thin_u8.copy()
        removed_components = 0
        removed_px = 0
        for comp_id in range(1, int(n_labels)):
            comp_mask = labels == comp_id
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            endpoint_count = int(np.count_nonzero(comp_mask & (neigh == 1)))
            if area <= max_px and endpoint_count > 0:
                filtered[comp_mask] = 0
                removed_components += 1
                removed_px += area

        return filtered, {
            "open_noise_max_px": int(max_px),
            "removed_open_noise_component_count": int(removed_components),
            "removed_open_noise_px": int(removed_px),
        }

    def _prune_short_spur_branches(
        self,
        thin_mask: np.ndarray,
        mean_width_px: float,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        max_px = self._simple_spur_prune_max_px(mean_width_px)
        if not self._simple_spur_prune_enabled() or max_px <= 0 or not np.any(thin_u8):
            return thin_u8, {
                "spur_prune_enabled": int(self._simple_spur_prune_enabled()),
                "spur_prune_max_px": int(max_px),
                "removed_spur_branch_count": 0,
                "removed_spur_px": 0,
            }

        work = thin_u8.copy()
        removed_branch_count = 0
        removed_px = 0

        for _ in range(8):
            neigh, branch = self._neighbor_stats_maps(work)
            endpoint_mask = (work > 0) & (neigh == 1)
            if not np.any(endpoint_mask):
                break

            points_to_remove: Set[Point] = set()
            pruned_this_round = 0
            ys, xs = np.nonzero(endpoint_mask)
            endpoints = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
            for start in endpoints:
                if work[start[1], start[0]] == 0 or int(neigh[start[1], start[0]]) != 1:
                    continue

                path = [start]
                prev: Optional[Point] = None
                cur = start
                stop_mode = "none"
                while True:
                    next_pts = [p for p in self._skeleton_neighbors(work > 0, cur[0], cur[1]) if p != prev]
                    if len(next_pts) == 0:
                        stop_mode = "dead_end"
                        break
                    if len(next_pts) != 1:
                        stop_mode = "branch"
                        break
                    nxt = next_pts[0]
                    path.append((int(nxt[0]), int(nxt[1])))
                    prev, cur = cur, nxt
                    if len(path) - 1 > int(max_px):
                        stop_mode = "too_long"
                        break

                branch_len_px = int(max(0, len(path) - 1))
                if stop_mode != "branch" or branch_len_px <= 0 or branch_len_px > int(max_px):
                    continue

                removable = path[:-1]
                if not removable:
                    continue
                for pt in removable:
                    points_to_remove.add((int(pt[0]), int(pt[1])))
                pruned_this_round += 1

            if not points_to_remove:
                break

            for x, y in points_to_remove:
                work[int(y), int(x)] = 0
            removed_branch_count += int(pruned_this_round)
            removed_px += int(len(points_to_remove))

        return work, {
            "spur_prune_enabled": int(self._simple_spur_prune_enabled()),
            "spur_prune_max_px": int(max_px),
            "removed_spur_branch_count": int(removed_branch_count),
            "removed_spur_px": int(removed_px),
        }

    def _node_points_xy(self, thin_mask: np.ndarray) -> Dict[str, object]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        neigh, branch = self._neighbor_stats_maps(thin_u8)
        endpoint_mask = (thin_u8 > 0) & (neigh == 1)
        junction_mask = (thin_u8 > 0) & (branch >= 3)
        point_limit = self._simple_node_point_limit()

        def _collect(mask: np.ndarray) -> List[List[int]]:
            ys, xs = np.nonzero(mask)
            pts = [[int(x), int(y)] for x, y in zip(xs.tolist(), ys.tolist())]
            if point_limit > 0:
                pts = pts[:point_limit]
            return pts

        return {
            "endpoint_count": int(np.count_nonzero(endpoint_mask)),
            "junction_count": int(np.count_nonzero(junction_mask)),
            "endpoint_points_xy": _collect(endpoint_mask),
            "junction_points_xy": _collect(junction_mask),
        }

    def _node_masks(self, thin_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(thin_u8):
            zero = np.zeros_like(thin_u8, dtype=bool)
            return zero, zero
        neigh, branch = self._neighbor_stats_maps(thin_u8)
        endpoint_mask = (thin_u8 > 0) & (neigh == 1)
        junction_mask = (thin_u8 > 0) & (branch >= 3)
        return endpoint_mask, junction_mask

    @staticmethod
    def _node_penalty_key(junction_count: int, endpoint_count: int, removed_px: int = 0) -> Tuple[int, int, int, int]:
        j = int(max(0, junction_count))
        e = int(max(0, endpoint_count))
        total = j + e
        return (0 if total == 0 else 1, total, j, int(max(0, removed_px)))

    def _resolve_ambiguous_node_zones(
        self,
        thin_mask: np.ndarray,
    ) -> Dict[str, object]:
        work = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(work):
            return {
                "thin_mask": work,
                "resolved_zone_count": 0,
                "removed_px": 0,
            }

        endpoint_mask, junction_mask = self._node_masks(work)
        node_mask = endpoint_mask | junction_mask
        if not np.any(node_mask):
            return {
                "thin_mask": work,
                "resolved_zone_count": 0,
                "removed_px": 0,
            }

        zone_seed = cv2.dilate(node_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
        n_zone, labels, stats, _ = cv2.connectedComponentsWithStats(zone_seed, connectivity=8)
        if int(n_zone) <= 1:
            return {
                "thin_mask": work,
                "resolved_zone_count": 0,
                "removed_px": 0,
            }

        resolved_zone_count = 0
        removed_px = 0

        for zone_id in range(1, int(n_zone)):
            zone_all = labels == zone_id
            zone_mask = zone_all & (work > 0)
            if not np.any(zone_mask):
                continue

            ys, xs = np.nonzero(zone_mask)
            if len(xs) <= 0:
                continue
            x0 = max(0, int(np.min(xs)) - 2)
            y0 = max(0, int(np.min(ys)) - 2)
            x1 = min(work.shape[1] - 1, int(np.max(xs)) + 2)
            y1 = min(work.shape[0] - 1, int(np.max(ys)) + 2)
            local = work[y0:y1 + 1, x0:x1 + 1].copy()

            zone_pts = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
            if len(zone_pts) > 24:
                continue

            base_j, base_e = self._node_score(work)
            base_key = self._node_penalty_key(base_j, base_e, removed_px=0)
            best_key = base_key
            best_remove: Optional[List[Point]] = None

            local_pts = [(int(x - x0), int(y - y0)) for x, y in zone_pts]
            candidate_sets: List[List[Point]] = [[pt] for pt in local_pts]
            for i in range(len(local_pts)):
                ax, ay = local_pts[i]
                for j in range(i + 1, len(local_pts)):
                    bx, by = local_pts[j]
                    if max(abs(ax - bx), abs(ay - by)) <= 1:
                        candidate_sets.append([(ax, ay), (bx, by)])

            for remove_pts in candidate_sets:
                trial = work.copy()
                for rx, ry in remove_pts:
                    trial[y0 + int(ry), x0 + int(rx)] = 0
                score_j, score_e = self._node_score(trial)
                score_key = self._node_penalty_key(score_j, score_e, removed_px=len(remove_pts))
                if score_key < best_key:
                    best_key = score_key
                    best_remove = list(remove_pts)

            if best_remove is None:
                continue

            for rx, ry in best_remove:
                work[y0 + int(ry), x0 + int(rx)] = 0
            resolved_zone_count += 1
            removed_px += int(len(best_remove))

        return {
            "thin_mask": np.asarray(work, dtype=np.uint8),
            "resolved_zone_count": int(resolved_zone_count),
            "removed_px": int(removed_px),
        }

    def _estimate_simple_mean_stroke_width(self, cleaned_mask: np.ndarray) -> float:
        fg = (np.asarray(cleaned_mask, dtype=np.uint8) > 0).astype(np.uint8)
        if not np.any(fg):
            return 2.0
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        values = dist[fg > 0]
        if values.size <= 0:
            return 2.0
        mean_width = float(np.mean(values) * 2.0)
        return float(np.clip(mean_width, 2.0, 12.0))

    def _simple_endpoint_reconnect_max_px(self, mean_width_px: float) -> int:
        return int(max(2, round(float(mean_width_px) * 2.0)))

    def _node_score(self, thin_mask: np.ndarray) -> Tuple[int, int]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(thin_u8):
            return (0, 0)
        neigh, branch = self._neighbor_stats_maps(thin_u8)
        endpoint_count = int(np.count_nonzero((thin_u8 > 0) & (neigh == 1)))
        junction_count = int(np.count_nonzero((thin_u8 > 0) & (branch >= 3)))
        return (junction_count, endpoint_count)

    def _local_centerline_tangent(
        self,
        mask_bool: np.ndarray,
        start_point: Point,
        blocked_mask: np.ndarray,
        steps: int,
    ) -> Optional[np.ndarray]:
        h, w = mask_bool.shape[:2]
        start = (int(start_point[0]), int(start_point[1]))
        queue: deque[Tuple[Point, int]] = deque([(start, 0)])
        visited = {start}
        best = start
        best_depth = 0

        while queue:
            (x, y), depth = queue.popleft()
            if depth > best_depth:
                best = (x, y)
                best_depth = depth
            if depth >= steps:
                continue

            neighbors: List[Point] = []
            for ny in range(max(0, y - 1), min(h, y + 2)):
                for nx in range(max(0, x - 1), min(w, x + 2)):
                    if nx == x and ny == y:
                        continue
                    if not bool(mask_bool[ny, nx]) or bool(blocked_mask[ny, nx]):
                        continue
                    pt = (int(nx), int(ny))
                    if pt in visited:
                        continue
                    neighbors.append(pt)

            neighbors.sort(
                key=lambda pt: (pt[0] - start[0]) * (pt[0] - start[0]) + (pt[1] - start[1]) * (pt[1] - start[1]),
                reverse=True,
            )
            for pt in neighbors:
                visited.add(pt)
                queue.append((pt, depth + 1))

        return self._unit_or_none((best[0] - start[0], best[1] - start[1]))

    def _endpoint_direction(
        self,
        mask_bool: np.ndarray,
        endpoint: Point,
        steps: int,
    ) -> Optional[np.ndarray]:
        h, w = mask_bool.shape[:2]
        start = (int(endpoint[0]), int(endpoint[1]))
        prev: Optional[Point] = None
        cur = start
        path = [start]

        for _ in range(max(1, steps)):
            neighbors: List[Point] = []
            for ny in range(max(0, cur[1] - 1), min(h, cur[1] + 2)):
                for nx in range(max(0, cur[0] - 1), min(w, cur[0] + 2)):
                    if nx == cur[0] and ny == cur[1]:
                        continue
                    if not bool(mask_bool[ny, nx]):
                        continue
                    nxt = (int(nx), int(ny))
                    if prev is not None and nxt == prev:
                        continue
                    neighbors.append(nxt)

            if not neighbors:
                break

            neighbors.sort(
                key=lambda pt: (pt[0] - start[0]) * (pt[0] - start[0]) + (pt[1] - start[1]) * (pt[1] - start[1]),
                reverse=True,
            )
            nxt = neighbors[0]
            path.append(nxt)
            prev = cur
            cur = nxt

        if len(path) < 2:
            return None
        return self._unit_or_none((path[-1][0] - path[0][0], path[-1][1] - path[0][1]))

    def _bridge_overlap_fraction(
        self,
        mask_u8: np.ndarray,
        bridge_points: Sequence[Point],
    ) -> float:
        if len(bridge_points) <= 2:
            return 0.0
        interior = bridge_points[1:-1]
        overlap = sum(1 for x, y in interior if mask_u8[int(y), int(x)] > 0)
        return float(overlap) / max(float(len(interior)), 1.0)

    def _apply_bridge(
        self,
        mask_u8: np.ndarray,
        bridge_points: Sequence[Point],
    ) -> np.ndarray:
        out = np.asarray(mask_u8, dtype=np.uint8).copy()
        for x, y in bridge_points:
            out[int(y), int(x)] = 255
        return out

    def _collect_junction_zones(
        self,
        thin_mask: np.ndarray,
        zone_r: int,
    ) -> List[Dict[str, object]]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(thin_u8):
            return []

        _neigh, branch = self._neighbor_stats_maps(thin_u8)
        junction_mask = ((thin_u8 > 0) & (branch >= 3)).astype(np.uint8)
        n_junc, labels, _, _ = cv2.connectedComponentsWithStats(junction_mask, connectivity=8)
        if int(n_junc) <= 1:
            return []

        zone_kernel = self._disk_kernel(max(1, int(zone_r)))
        clear_kernel = np.ones((3, 3), dtype=np.uint8)
        thin_bool = thin_u8 > 0

        zones: List[Dict[str, object]] = []
        for junction_id in range(1, int(n_junc)):
            comp = (labels == junction_id).astype(np.uint8)
            if not np.any(comp):
                continue
            zone_mask = (cv2.dilate(comp, zone_kernel, iterations=1) > 0) & thin_bool
            clear_mask = (cv2.dilate(zone_mask.astype(np.uint8), clear_kernel, iterations=1) > 0) & thin_bool
            zones.append(
                {
                    "junction_id": int(junction_id),
                    "zone_mask": zone_mask,
                    "clear_mask": clear_mask,
                    "anchor": self._junction_anchor(zone_mask),
                }
            )
        return zones

    def _collect_zone_ports(
        self,
        base_mask: np.ndarray,
        clear_mask: np.ndarray,
        anchor: Point,
        dist_to_outside: np.ndarray,
        tangent_steps: int,
    ) -> List[Dict[str, object]]:
        base_bool = np.asarray(base_mask, dtype=np.uint8) > 0
        clear_bool = np.asarray(clear_mask, dtype=bool)
        ring = (cv2.dilate(clear_bool.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0) & base_bool
        n_group, labels, _, _ = cv2.connectedComponentsWithStats(ring.astype(np.uint8), connectivity=8)
        ports: List[Dict[str, object]] = []

        for group_id in range(1, int(n_group)):
            ys, xs = np.nonzero(labels == group_id)
            if len(xs) <= 0:
                continue
            pts = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
            point = min(
                pts,
                key=lambda pt: (pt[0] - int(anchor[0])) * (pt[0] - int(anchor[0])) + (pt[1] - int(anchor[1])) * (pt[1] - int(anchor[1])),
            )
            direction = self._local_centerline_tangent(
                mask_bool=base_bool,
                start_point=point,
                blocked_mask=clear_bool,
                steps=tangent_steps,
            )
            if direction is None:
                direction = self._unit_or_none((point[0] - int(anchor[0]), point[1] - int(anchor[1])))
            if direction is None:
                continue

            samples: List[float] = []
            for step in range(0, 6):
                sx = int(round(point[0] + float(direction[0]) * step))
                sy = int(round(point[1] + float(direction[1]) * step))
                if 0 <= sx < dist_to_outside.shape[1] and 0 <= sy < dist_to_outside.shape[0]:
                    samples.append(float(dist_to_outside[sy, sx]))
            outside_score = min(samples) if samples else float("inf")
            ports.append(
                {
                    "group_id": int(group_id),
                    "point": (int(point[0]), int(point[1])),
                    "dir": np.asarray(direction, dtype=np.float64),
                    "outside_score": float(outside_score),
                }
            )

        ports.sort(key=lambda item: (item["point"][0], item["point"][1]))
        return ports

    def _resolve_junctions_on_thin_mask(
        self,
        thin_mask: np.ndarray,
        roi_mask: np.ndarray,
        mean_width_px: float,
        zone_r: int,
    ) -> Dict[str, object]:
        thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        zero = np.zeros_like(thin_u8, dtype=np.uint8)
        if not np.any(thin_u8):
            return {
                "resolved_thin_mask": zero.copy(),
                "junction_resolved_count": 0,
                "junction_unresolved_count": 0,
            }

        if not self._simple_junction_resolve_enabled():
            return {
                "resolved_thin_mask": thin_u8.copy(),
                "junction_resolved_count": 0,
                "junction_unresolved_count": 0,
            }

        zones = self._collect_junction_zones(thin_u8, zone_r=zone_r)
        if not zones:
            return {
                "resolved_thin_mask": thin_u8.copy(),
                "junction_resolved_count": 0,
                "junction_unresolved_count": 0,
            }

        outside_mask = self._outside_background_mask(thin_mask=thin_u8, roi_mask=roi_mask)
        if np.any(outside_mask > 0):
            dist_to_outside = cv2.distanceTransform((outside_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        else:
            dist_to_outside = np.zeros_like(thin_u8, dtype=np.float32)

        base_remove = np.zeros_like(thin_u8, dtype=bool)
        for zone in zones:
            base_remove |= np.asarray(zone["clear_mask"], dtype=bool)
        base_mask = ((thin_u8 > 0) & (~base_remove)).astype(np.uint8) * 255

        resolved_mask = base_mask.copy()
        tangent_steps = self._simple_tangent_steps()
        min_angle = self._simple_pair_min_angle_deg()
        resolved_count = 0
        unresolved_count = 0

        for zone in zones:
            clear_mask = np.asarray(zone["clear_mask"], dtype=bool)
            anchor = tuple(zone["anchor"])
            ports = self._collect_zone_ports(
                base_mask=base_mask,
                clear_mask=clear_mask,
                anchor=anchor,
                dist_to_outside=dist_to_outside,
                tangent_steps=tangent_steps,
            )
            if len(ports) < 2:
                unresolved_count += 1
                continue

            pair_candidates: List[Tuple[float, float, int, int]] = []
            for i in range(len(ports)):
                for j in range(i + 1, len(ports)):
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(ports[i]["dir"], ports[j]["dir"]), -1.0, 1.0))))
                    outside_sum = float(ports[i]["outside_score"] + ports[j]["outside_score"])
                    pair_candidates.append((angle, outside_sum, i, j))

            if not pair_candidates:
                unresolved_count += 1
                continue

            best_angle = max(item[0] for item in pair_candidates)
            shortlist = [item for item in pair_candidates if best_angle - item[0] <= 10.0]
            shortlist.sort(key=lambda item: (item[1], -item[0], item[2], item[3]))
            angle, _outside_sum, i, j = shortlist[0]
            if angle < min_angle:
                unresolved_count += 1
                continue

            resolved_count += 1
            if len(ports) > 2:
                unresolved_count += 1

            point_a = tuple(ports[i]["point"])
            point_b = tuple(ports[j]["point"])
            for start, end in ((point_a, anchor), (anchor, point_b)):
                bridge = self._raster_line_points(tuple(map(int, start)), tuple(map(int, end)))
                for x, y in bridge:
                    resolved_mask[int(y), int(x)] = 255

        return {
            "resolved_thin_mask": np.asarray(resolved_mask, dtype=np.uint8),
            "junction_resolved_count": int(resolved_count),
            "junction_unresolved_count": int(unresolved_count),
        }

    def _find_best_endpoint_bridge(
        self,
        work_mask: np.ndarray,
        max_gap_px: float,
        max_angle_deg: float,
        max_overlap_frac: float,
        allow_endpoint_snap: bool,
    ) -> Optional[Dict[str, object]]:
        work_u8 = (np.asarray(work_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(work_u8):
            return None

        neigh, branch = self._neighbor_stats_maps(work_u8)
        ys, xs = np.nonzero((work_u8 > 0) & (neigh == 1))
        endpoints = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        if len(endpoints) <= 0:
            return None

        dirs = {
            ep: self._endpoint_direction(work_u8 > 0, ep, steps=self._simple_tangent_steps())
            for ep in endpoints
        }
        base_score = self._node_score(work_u8)
        point_set = [(int(x), int(y)) for x, y in zip(*np.nonzero(work_u8 > 0)[::-1])]
        endpoint_set = set(endpoints)
        best: Optional[Dict[str, object]] = None

        def _consider_bridge(start: Point, end: Point, angle_score: float) -> None:
            nonlocal best
            bridge = self._raster_line_points(tuple(map(int, start)), tuple(map(int, end)))
            if len(bridge) <= 1:
                return
            overlap_frac = self._bridge_overlap_fraction(work_u8, bridge)
            if overlap_frac > max_overlap_frac:
                return
            temp = self._apply_bridge(work_u8, bridge)
            score_after = self._node_score(temp)
            if score_after >= base_score:
                return
            dist = float(np.linalg.norm(np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)))
            record = {
                "bridge_points": bridge,
                "score_after": score_after,
                "dist_px": dist,
                "angle_score": float(angle_score),
                "overlap_frac": float(overlap_frac),
                "target_point": tuple(map(int, end)),
                "target_is_endpoint": bool(tuple(map(int, end)) in endpoint_set),
            }
            candidate_key = (
                int(record["score_after"][0]),
                int(record["score_after"][1]),
                float(record["dist_px"]),
                float(record["angle_score"]),
                float(record["overlap_frac"]),
                0 if record["target_is_endpoint"] else 1,
            )
            if best is None or candidate_key < best["candidate_key"]:
                record["candidate_key"] = candidate_key
                best = record

        for idx_a in range(len(endpoints)):
            start = endpoints[idx_a]
            dir_a = dirs.get(start)
            if dir_a is None:
                continue
            for idx_b in range(idx_a + 1, len(endpoints)):
                end = endpoints[idx_b]
                dir_b = dirs.get(end)
                if dir_b is None:
                    continue
                vec = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
                dist = float(np.linalg.norm(vec))
                if not np.isfinite(dist) or dist <= 1e-6 or dist > max_gap_px:
                    continue
                unit = vec / dist
                ang_a = float(np.degrees(np.arccos(np.clip(np.dot(dir_a, unit), -1.0, 1.0))))
                ang_b = float(np.degrees(np.arccos(np.clip(np.dot(dir_b, -unit), -1.0, 1.0))))
                if max(ang_a, ang_b) > max_angle_deg:
                    continue
                _consider_bridge(start, end, angle_score=max(ang_a, ang_b))

        if best is not None or not allow_endpoint_snap:
            return best

        target_neigh = neigh.copy()
        target_branch = branch.copy()
        for start in endpoints:
            dir_a = dirs.get(start)
            if dir_a is None:
                continue
            for end in point_set:
                if end == start:
                    continue
                if abs(end[0] - start[0]) <= 1 and abs(end[1] - start[1]) <= 1:
                    continue
                if tuple(map(int, end)) in endpoint_set:
                    continue
                if int(target_branch[int(end[1]), int(end[0])]) >= 3:
                    continue
                if int(target_neigh[int(end[1]), int(end[0])]) <= 0:
                    continue
                vec = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
                dist = float(np.linalg.norm(vec))
                if not np.isfinite(dist) or dist <= 1e-6 or dist > max_gap_px:
                    continue
                unit = vec / dist
                ang_a = float(np.degrees(np.arccos(np.clip(np.dot(dir_a, unit), -1.0, 1.0))))
                if ang_a > max_angle_deg:
                    continue
                _consider_bridge(start, end, angle_score=ang_a)

        return best

    def _reconnect_short_open_ends(
        self,
        thin_mask: np.ndarray,
        mean_width_px: float,
        zone_r: int,
    ) -> Dict[str, object]:
        work = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(work):
            return {
                "thin_mask": work,
                "gap_reconnected_count": 0,
            }

        base_gap = self._simple_endpoint_reconnect_max_px(mean_width_px)
        reconnect_count = 0

        while True:
            candidate = self._find_best_endpoint_bridge(
                work_mask=work,
                max_gap_px=float(base_gap),
                max_angle_deg=self._simple_endpoint_reconnect_angle_deg(),
                max_overlap_frac=self._simple_bridge_overlap_max(),
                allow_endpoint_snap=False,
            )
            if candidate is None:
                break
            work = self._apply_bridge(work, candidate["bridge_points"])
            reconnect_count += 1

        snap_gap = max(float(base_gap), float(int(np.ceil(float(mean_width_px) * 3.0))), float(zone_r + 3))
        while True:
            candidate = self._find_best_endpoint_bridge(
                work_mask=work,
                max_gap_px=snap_gap,
                max_angle_deg=self._simple_endpoint_snap_angle_deg(),
                max_overlap_frac=self._simple_bridge_overlap_max(),
                allow_endpoint_snap=True,
            )
            if candidate is None:
                break
            work = self._apply_bridge(work, candidate["bridge_points"])
            reconnect_count += 1

        return {
            "thin_mask": np.asarray(work, dtype=np.uint8),
            "gap_reconnected_count": int(reconnect_count),
        }

    def _extract_simple_levels_from_thin_mask(
        self,
        thin_mask: np.ndarray,
        roi_mask: np.ndarray,
        mean_width_px: float,
    ) -> Dict[str, object]:
        thin = np.asarray(thin_mask, dtype=np.uint8)
        valid = np.asarray(roi_mask, dtype=np.uint8)
        zero = np.zeros_like(thin, dtype=np.uint8)
        thin_bool = (thin > 0) & (valid > 0)

        thin_px = int(np.count_nonzero(thin_bool))
        if thin_px <= 0:
            return {
                "status": "empty_mask",
                "message": "No contour pixels remain after cleaning/thinning.",
                "thin_mask": zero.copy(),
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
                "endpoint_count": 0,
                "junction_count": 0,
            }

        thin_u8 = thin_bool.astype(np.uint8) * 255
        node_info = self._node_points_xy(thin_u8)
        endpoint_count = int(node_info["endpoint_count"])
        junction_count = int(node_info["junction_count"])
        if endpoint_count > 0 or junction_count > 0:
            return {
                "status": "open_or_branching_contour",
                "message": (
                    "Thin mask is not a pure closed-loop set "
                    f"(endpoints={endpoint_count}, junctions={junction_count}), so level extraction was skipped."
                ),
                "thin_mask": thin_u8.copy(),
                "outside_mask": zero.copy(),
                "outer_touch_mask": zero.copy(),
                "l1_map": zero.copy(),
                "l2_map": zero.copy(),
                "level_maps": {1: zero.copy(), 2: zero.copy()},
                "level_label_map": np.zeros_like(thin, dtype=np.int32),
                "region_depth_map": np.zeros_like(thin, dtype=np.int32),
                "residual_map": thin_u8.copy(),
                "max_detected_level": 0,
                "detected_levels": [],
                "unassigned_level_px": int(np.count_nonzero(thin_bool)),
                "recovered_level_px": 0,
                "endpoint_count": endpoint_count,
                "junction_count": junction_count,
                "endpoint_points_xy": list(node_info["endpoint_points_xy"]),
                "junction_points_xy": list(node_info["junction_points_xy"]),
            }

        assigned = iterative_peel_levels_from_thin_mask(
            thin_mask=thin_u8,
            roi_mask=valid,
            max_levels=64,
            repair_callback=lambda work: self._repair_iterative_peel_residual(
                thin_mask=np.asarray(work, dtype=np.uint8),
                mean_width_px=float(mean_width_px),
            ),
        )
        assigned["endpoint_count"] = int(endpoint_count)
        assigned["junction_count"] = int(junction_count)
        assigned["endpoint_points_xy"] = list(node_info["endpoint_points_xy"])
        assigned["junction_points_xy"] = list(node_info["junction_points_xy"])
        return assigned

    def _extract_simple_l1_from_thin_mask(
        self,
        thin_mask: np.ndarray,
        roi_mask: np.ndarray,
        mean_width_px: float = 2.0,
    ) -> Dict[str, object]:
        return self._extract_simple_levels_from_thin_mask(
            thin_mask=thin_mask,
            roi_mask=roi_mask,
            mean_width_px=float(mean_width_px),
        )

    def _repair_iterative_peel_residual(
        self,
        thin_mask: np.ndarray,
        mean_width_px: float,
    ) -> Dict[str, object]:
        work = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if not np.any(work):
            return {
                "thin_mask": work,
                "gap_reconnected_count": 0,
            }

        node_info = self._node_points_xy(work)
        if int(node_info.get("endpoint_count", 0)) <= 0:
            return {
                "thin_mask": work,
                "gap_reconnected_count": 0,
            }

        work, _noise_info = self._filter_small_open_components(work)
        max_gap = int(max(2, min(4, round(float(mean_width_px)))))
        max_angle = min(35.0, self._simple_endpoint_reconnect_angle_deg())
        reconnect_count = 0

        while True:
            candidate = self._find_best_endpoint_bridge(
                work_mask=work,
                max_gap_px=float(max_gap),
                max_angle_deg=float(max_angle),
                max_overlap_frac=min(0.10, self._simple_bridge_overlap_max()),
                allow_endpoint_snap=False,
            )
            if candidate is None:
                break
            work = self._apply_bridge(work, candidate["bridge_points"])
            reconnect_count += 1
            if reconnect_count >= 64:
                break

        return {
            "thin_mask": np.asarray(work, dtype=np.uint8),
            "gap_reconnected_count": int(reconnect_count),
        }

    @staticmethod
    def _level_fill_palette_bgr() -> List[Tuple[int, int, int]]:
        return [
            (74, 123, 255),
            (68, 190, 135),
            (74, 201, 232),
            (75, 130, 255),
            (116, 94, 232),
            (82, 205, 247),
            (80, 175, 76),
            (54, 67, 244),
            (57, 220, 205),
            (39, 156, 245),
        ]

    def _render_level_region_overlay(
        self,
        region_depth_map: np.ndarray,
        thin_mask: np.ndarray,
        roi_mask: np.ndarray,
    ) -> np.ndarray:
        depth_map = np.asarray(region_depth_map, dtype=np.int32)
        thin_bool = np.asarray(thin_mask, dtype=np.uint8) > 0
        valid = np.asarray(roi_mask, dtype=np.uint8) > 0

        overlay = np.full((depth_map.shape[0], depth_map.shape[1], 3), 255, dtype=np.uint8)
        overlay[valid] = np.array([16, 16, 16], dtype=np.uint8)

        palette = self._level_fill_palette_bgr()
        positive_levels = sorted(int(v) for v in np.unique(depth_map[valid]) if int(v) > 0)
        for level in positive_levels:
            color = palette[(int(level) - 1) % len(palette)]
            overlay[(depth_map == int(level)) & valid] = np.array(color, dtype=np.uint8)

        overlay[thin_bool & valid] = np.array([245, 245, 245], dtype=np.uint8)
        return overlay

    @staticmethod
    def _fit_overlay_text_style(
        text: str,
        target_w: int,
        max_h: int,
        base_scale: float = 0.62,
        base_thickness: int = 2,
    ) -> Tuple[float, int]:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(base_scale)
        thickness = int(max(1, base_thickness))
        target_w = int(max(24, target_w))
        max_h = int(max(12, max_h))
        while scale > 0.26:
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            if text_w <= target_w and (text_h + baseline) <= max_h:
                break
            scale -= 0.04
            if scale < 0.48:
                thickness = 1
        return float(scale), int(thickness)

    def _draw_overlay_status_text(
        self,
        image: np.ndarray,
        lines: Sequence[str],
    ) -> np.ndarray:
        canvas = _ensure_bgr(image)
        x = 10
        y = 56
        target_w = max(24, canvas.shape[1] - 20)
        for raw_line in lines[:3]:
            line = str(raw_line).strip()
            if not line:
                continue
            scale, thickness = self._fit_overlay_text_style(
                line,
                target_w=target_w,
                max_h=20,
                base_scale=0.62,
                base_thickness=2,
            )
            (_, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv2.putText(
                canvas,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (120, 200, 255),
                thickness,
                cv2.LINE_AA,
            )
            y += max(18, int(text_h + baseline + 6))
        return canvas

    def _draw_skip_node_markers(
        self,
        image: np.ndarray,
        endpoint_points_xy: Sequence[Sequence[int]],
        junction_points_xy: Sequence[Sequence[int]],
    ) -> np.ndarray:
        canvas = _ensure_bgr(image)
        endpoint_boxes = self._node_zone_boxes(endpoint_points_xy, canvas.shape[:2], pad=2)
        junction_boxes = self._node_zone_boxes(junction_points_xy, canvas.shape[:2], pad=3)
        for x0, y0, x1, y1 in endpoint_boxes:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 180, 40), 1, cv2.LINE_AA)
        for x0, y0, x1, y1 in junction_boxes:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 255), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _node_zone_boxes(
        points_xy: Sequence[Sequence[int]],
        shape_hw: Tuple[int, int],
        pad: int = 3,
    ) -> List[Tuple[int, int, int, int]]:
        h, w = [int(v) for v in shape_hw]
        if h <= 0 or w <= 0:
            return []
        seed = np.zeros((h, w), dtype=np.uint8)
        for pt in points_xy:
            if len(pt) < 2:
                continue
            x = int(pt[0])
            y = int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                seed[y, x] = 255
        if not np.any(seed > 0):
            return []
        radius = int(max(1, pad))
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        grown = cv2.dilate(seed, kernel, iterations=1)
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        boxes: List[Tuple[int, int, int, int]] = []
        for label in range(1, int(count)):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, x + bw - 1, y + bh - 1))
        return boxes

    def _compose_simple_preview(
        self,
        roi_view: np.ndarray,
        cleaned_mask: np.ndarray,
        thin_mask_for_levels: np.ndarray,
        level_overlay_map: np.ndarray,
        level_status_text: Optional[Sequence[str]] = None,
        endpoint_points_xy: Optional[Sequence[Sequence[int]]] = None,
        junction_points_xy: Optional[Sequence[Sequence[int]]] = None,
    ) -> np.ndarray:
        endpoint_points_xy = list(endpoint_points_xy or [])
        junction_points_xy = list(junction_points_xy or [])
        thin_panel = self._draw_skip_node_markers(
            _ensure_bgr(thin_mask_for_levels),
            endpoint_points_xy=endpoint_points_xy,
            junction_points_xy=junction_points_xy,
        )
        overlay = _ensure_bgr(level_overlay_map)
        overlay = self._draw_skip_node_markers(
            overlay,
            endpoint_points_xy=endpoint_points_xy,
            junction_points_xy=junction_points_xy,
        )
        status_lines = [str(line) for line in (level_status_text or []) if str(line).strip()]

        base_panels = [
            (np.asarray(roi_view, dtype=np.uint8), "1. ROI View"),
            (_ensure_bgr(cleaned_mask), "2. Cleaned Binary Mask"),
            (thin_panel, "3. Thin Mask Used For Levels"),
            (overlay, "4. Level Map Overlay"),
        ]
        base_body_w = max(panel.shape[1] for panel, _title in base_panels)
        base_body_h = max(panel.shape[0] for panel, _title in base_panels)
        max_cell_w = int(max(320, self._simple_window_max_width() // 2))
        max_cell_h = int(max(240, self._simple_window_max_height() // 2))
        target_cell_w = int(max(1, min(base_body_w, max_cell_w)))
        target_cell_h = int(max(68, min(base_body_h + 34, max_cell_h)))
        body_h = max(64, target_cell_h - 34)

        fitted: List[np.ndarray] = []
        for idx, (panel_img, panel_title) in enumerate(base_panels):
            scaled_body = self._fit_image_to_box(
                panel_img,
                target_cell_w,
                body_h,
                allow_upscale=True,
            )
            body = self._pad_image_to_size(scaled_body, target_cell_w, body_h, bg=255)
            if idx == 3 and status_lines:
                body = self._draw_overlay_status_text(body, status_lines)
            titled = _put_panel_title(body, panel_title)
            fitted.append(self._pad_image_to_size(titled, target_cell_w, target_cell_h, bg=255))
        return _stack_rows(fitted, per_row=2)

    def _build_simple_payload(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Dict[str, object]:
        initial_mask, binary_info, evidence_map, binary_prep_mode = self._prepare_simple_binary_mask(
            roi_img=roi_img,
            roi_mask=roi_mask,
        )
        # Keep the cleaned mask as close to the original binary evidence as possible.
        # Open contour recovery belongs to the thinned centerline stage, not the thick mask stage.
        cleaned_mask_base = self._normalize_binary_mask(initial_mask, roi_mask)
        cleaned_mask_base = self._component_filter_by_area_and_perimeter(cleaned_mask_base, roi_mask)
        cleaned_mask_base[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
        saved_session = None
        if self._manual_edit_session_override is not None:
            saved_session = dict(self._manual_edit_session_override)
            self._manual_edit_session_override = None
        loaded_from_replay = bool(saved_session is not None)
        binary_split_cuts: List[Dict[str, object]] = list((saved_session or {}).get("binary_split_cuts", []))
        junction_cuts: List[Dict[str, object]] = list((saved_session or {}).get("junction_cuts", []))
        manual_binary_split_info = {
            "opened": False,
            "edited": False,
            "loaded": bool(loaded_from_replay),
            "cut_count": int(len(binary_split_cuts)),
            "cuts": list(binary_split_cuts),
        }
        manual_junction_edit_info = {
            "opened": False,
            "edited": False,
            "loaded": bool(loaded_from_replay),
            "cut_count": int(len(junction_cuts)),
            "cuts": list(junction_cuts),
        }
        while True:
            cleaned_mask = self._apply_cut_records(cleaned_mask_base, binary_split_cuts)
            cleaned_mask[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
            skip_binary_editor_once = bool(self._force_skip_binary_split_once)
            self._force_skip_binary_split_once = False
            force_binary_editor_once = bool(self._force_reopen_binary_split_once)
            self._force_reopen_binary_split_once = False
            open_binary_editor = (
                self._simple_prethin_split_edit_enabled()
                and np.any(cleaned_mask_base > 0)
                and (
                    force_binary_editor_once
                    or (
                        (not loaded_from_replay)
                        and (not binary_split_cuts)
                        and (not skip_binary_editor_once)
                    )
                )
            )
            if open_binary_editor:
                print("-> Opening binary split editor before thinning.")
                split_hint_thin = self._thin_binary(cleaned_mask)
                split_hint_thin[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
                split_hint_nodes = self._node_points_xy(split_hint_thin)
                edit_info = edit_binary_mask_splits(
                    binary_mask=cleaned_mask_base,
                    junction_points_xy=list(split_hint_nodes.get("junction_points_xy", [])),
                    endpoint_points_xy=list(split_hint_nodes.get("endpoint_points_xy", [])),
                    window_name="JetAnalyzerV8 Binary Split Editor",
                    max_width=self._simple_window_max_width(),
                    max_height=self._simple_window_max_height(),
                    initial_cut_width=self._simple_prethin_split_cut_width(),
                    initial_cuts=binary_split_cuts,
                )
                if edit_info is None:
                    raise RuntimeError("Manual binary split editing cancelled.")
                binary_split_cuts = list(edit_info.get("cuts", []))
                cleaned_mask = np.asarray(edit_info["edited_mask"], dtype=np.uint8)
                cleaned_mask[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
                manual_binary_split_info = {
                    "opened": True,
                    "edited": bool(int(edit_info.get("cut_count", 0)) > 0),
                    "loaded": False,
                    "cut_count": int(edit_info.get("cut_count", 0)),
                    "cuts": list(binary_split_cuts),
                }

            mean_width_px = self._estimate_simple_mean_stroke_width(cleaned_mask)

            raw_thin_mask = self._thin_binary(cleaned_mask)
            raw_thin_mask[np.asarray(roi_mask, dtype=np.uint8) == 0] = 0
            raw_thin_mask, noise_filter_info = self._filter_small_open_components(raw_thin_mask)
            raw_thin_mask, spur_prune_info = self._prune_short_spur_branches(
                raw_thin_mask,
                mean_width_px=mean_width_px,
            )
            raw_node_info = self._node_points_xy(raw_thin_mask)
            if junction_cuts:
                raw_thin_mask = self._apply_cut_records(raw_thin_mask, junction_cuts)
                raw_node_info = self._node_points_xy(raw_thin_mask)
                manual_junction_edit_info = {
                    "opened": False,
                    "edited": bool(len(junction_cuts) > 0),
                    "loaded": True,
                    "cut_count": int(len(junction_cuts)),
                    "cuts": list(junction_cuts),
                }
            force_junction_editor_once = bool(self._force_reopen_junction_editor_once)
            self._force_reopen_junction_editor_once = False
            should_consider_junction_editor = self._simple_manual_junction_edit_enabled() and np.any(raw_thin_mask > 0)
            if should_consider_junction_editor and (force_junction_editor_once or not manual_junction_edit_info["loaded"]):
                show_editor = force_junction_editor_once or self._simple_manual_junction_edit_always()
                show_editor = show_editor or int(raw_node_info.get("junction_count", 0)) > 0
                show_editor = show_editor or int(raw_node_info.get("endpoint_count", 0)) > 0
                if show_editor:
                    print(
                        "-> Opening junction editor: "
                        f"endpoints={int(raw_node_info.get('endpoint_count', 0))}, "
                        f"junctions={int(raw_node_info.get('junction_count', 0))}"
                    )
                    edit_info = edit_thin_mask_junctions(
                        thin_mask=raw_thin_mask,
                        junction_points_xy=list(raw_node_info.get("junction_points_xy", [])),
                        endpoint_points_xy=list(raw_node_info.get("endpoint_points_xy", [])),
                        window_name="JetAnalyzerV8 Junction Editor",
                        max_width=self._simple_window_max_width(),
                        max_height=self._simple_window_max_height(),
                        initial_cut_width=self._simple_manual_junction_cut_width(),
                        initial_cuts=junction_cuts,
                    )
                    if edit_info is None:
                        raise RuntimeError("Manual junction editing cancelled.")
                    if str(edit_info.get("action", "accept")) == "back":
                        junction_cuts = list(edit_info.get("cuts", junction_cuts))
                        manual_junction_edit_info = {
                            "opened": True,
                            "edited": bool(len(junction_cuts) > 0),
                            "loaded": False,
                            "cut_count": int(len(junction_cuts)),
                            "cuts": list(junction_cuts),
                        }
                        self._skip_next_manual_threshold_tuner = True
                        self._force_reopen_binary_split_once = True
                        self._force_reopen_junction_editor_once = True
                        continue
                    junction_cuts = list(edit_info.get("cuts", []))
                    raw_thin_mask = np.asarray(edit_info["edited_mask"], dtype=np.uint8)
                    raw_node_info = self._node_points_xy(raw_thin_mask)
                    manual_junction_edit_info = {
                        "opened": True,
                        "edited": bool(int(edit_info.get("cut_count", 0)) > 0),
                        "loaded": False,
                        "cut_count": int(edit_info.get("cut_count", 0)),
                        "cuts": list(junction_cuts),
                    }
            break

        zone_r = int(max(1, round(mean_width_px / 2.0)))

        if int(manual_junction_edit_info.get("cut_count", 0)) > 0:
            junction_info = {
                "resolved_thin_mask": np.asarray(raw_thin_mask, dtype=np.uint8),
                "junction_resolved_count": 0,
                "junction_unresolved_count": int(raw_node_info.get("junction_count", 0)),
            }
            reconnect_info = {
                "thin_mask": np.asarray(raw_thin_mask, dtype=np.uint8),
                "gap_reconnected_count": 0,
            }
            final_thin_mask = np.asarray(raw_thin_mask, dtype=np.uint8)
        elif self._simple_junction_resolve_enabled():
            junction_info = self._resolve_junctions_on_thin_mask(
                thin_mask=raw_thin_mask,
                roi_mask=roi_mask,
                mean_width_px=mean_width_px,
                zone_r=zone_r,
            )
            reconnect_info = self._reconnect_short_open_ends(
                thin_mask=np.asarray(junction_info["resolved_thin_mask"], dtype=np.uint8),
                mean_width_px=mean_width_px,
                zone_r=zone_r,
            )
            final_thin_mask = np.asarray(reconnect_info["thin_mask"], dtype=np.uint8)
        else:
            junction_info = {
                "resolved_thin_mask": np.asarray(raw_thin_mask, dtype=np.uint8),
                "junction_resolved_count": 0,
                "junction_unresolved_count": 0,
            }
            reconnect_info = {
                "thin_mask": np.asarray(raw_thin_mask, dtype=np.uint8),
                "gap_reconnected_count": 0,
            }
            final_thin_mask = np.asarray(raw_thin_mask, dtype=np.uint8)

        ambiguous_node_info = self._resolve_ambiguous_node_zones(final_thin_mask)
        final_thin_mask = np.asarray(ambiguous_node_info["thin_mask"], dtype=np.uint8)
        if int(ambiguous_node_info.get("resolved_zone_count", 0)) > 0:
            post_reconnect_info = self._reconnect_short_open_ends(
                thin_mask=final_thin_mask,
                mean_width_px=mean_width_px,
                zone_r=zone_r,
            )
            final_thin_mask = np.asarray(post_reconnect_info["thin_mask"], dtype=np.uint8)
            reconnect_info["gap_reconnected_count"] = int(reconnect_info.get("gap_reconnected_count", 0)) + int(
                post_reconnect_info.get("gap_reconnected_count", 0)
            )
        final_node_info = self._node_points_xy(final_thin_mask)

        replay_save_path = self._simple_replay_json_save_path()
        replay_payload = self._current_replay_payload(
            binary_prep_mode=str(binary_prep_mode),
            binary_split_cuts=binary_split_cuts,
            junction_cuts=junction_cuts,
        )
        if replay_save_path:
            save_replay_session(replay_save_path, replay_payload)

        peel = self._extract_simple_levels_from_thin_mask(
            thin_mask=final_thin_mask,
            roi_mask=roi_mask,
            mean_width_px=mean_width_px,
        )
        roi_view = self._masked_roi_gray_view(roi_img=roi_img, roi_mask=roi_mask)
        level_overlay_map = self._render_level_region_overlay(
            region_depth_map=np.asarray(peel["region_depth_map"], dtype=np.int32),
            thin_mask=np.asarray(peel.get("classified_thin_mask", final_thin_mask), dtype=np.uint8),
            roi_mask=roi_mask,
        )
        level_status_text: List[str] = []
        if str(peel.get("status", "")) != "ok":
            status_label = str(peel.get("status", "unknown")).replace("_", " ")
            level_status_text.append(f"Status: {status_label}")
            if int(peel.get("endpoint_count", 0)) > 0 or int(peel.get("junction_count", 0)) > 0:
                level_status_text.append(
                    f"Endpoints: {int(peel.get('endpoint_count', 0))}  Junctions: {int(peel.get('junction_count', 0))}"
                )
        elif int(peel.get("max_detected_level", 0)) <= 0:
            level_status_text.append("No assigned levels")
        preview = self._compose_simple_preview(
            roi_view=roi_view,
            cleaned_mask=cleaned_mask,
            thin_mask_for_levels=np.asarray(final_thin_mask, dtype=np.uint8),
            level_overlay_map=np.asarray(level_overlay_map, dtype=np.uint8),
            level_status_text=level_status_text,
            endpoint_points_xy=list(peel.get("endpoint_points_xy", [])),
            junction_points_xy=list(peel.get("junction_points_xy", [])),
        )

        return {
            "roi_mask": np.asarray(roi_mask, dtype=np.uint8),
            "binary_info": dict(binary_info),
            "evidence_map": np.asarray(evidence_map, dtype=np.float32),
            "initial_mask": np.asarray(initial_mask, dtype=np.uint8),
            "cleaned_mask": np.asarray(cleaned_mask, dtype=np.uint8),
            "raw_thin_mask": np.asarray(raw_thin_mask, dtype=np.uint8),
            "final_thin_mask": np.asarray(final_thin_mask, dtype=np.uint8),
            "outside_mask": np.asarray(peel["outside_mask"], dtype=np.uint8),
            "outer_touch_mask": np.asarray(peel["outer_touch_mask"], dtype=np.uint8),
            "classified_thin_mask": np.asarray(peel.get("classified_thin_mask", peel["thin_mask"]), dtype=np.uint8),
            "l1_map": np.asarray(peel["l1_map"], dtype=np.uint8),
            "l2_map": np.asarray(peel["l2_map"], dtype=np.uint8),
            "level_maps": {
                int(level): np.asarray(level_map, dtype=np.uint8)
                for level, level_map in dict(peel.get("level_maps", {})).items()
            },
            "level_label_map": np.asarray(peel["level_label_map"], dtype=np.int32),
            "region_depth_map": np.asarray(peel["region_depth_map"], dtype=np.int32),
            "level_overlay_map": np.asarray(level_overlay_map, dtype=np.uint8),
            "residual_map": np.asarray(peel["residual_map"], dtype=np.uint8),
            "status": str(peel["status"]),
            "message": str(peel["message"]),
            "detected_levels": list(peel.get("detected_levels", [])),
            "max_detected_level": int(peel.get("max_detected_level", 0)),
            "unassigned_level_px": int(peel.get("unassigned_level_px", 0)),
            "recovered_level_px": int(peel.get("recovered_level_px", 0)),
            "repair_bridge_count": int(peel.get("repair_bridge_count", 0)),
            "repair_added_px": int(peel.get("repair_added_px", 0)),
            "raw_node_info": dict(raw_node_info),
            "final_node_info": dict(final_node_info),
            "noise_filter_info": dict(noise_filter_info),
            "spur_prune_info": dict(spur_prune_info),
            "manual_junction_edit_info": dict(manual_junction_edit_info),
            "ambiguous_node_info": dict(ambiguous_node_info),
            "manual_binary_split_info": dict(manual_binary_split_info),
            "replay_json_path": str(replay_save_path),
            "replay_json_loaded": bool(saved_session is not None),
            "replay_json_loaded_path": str(self._loaded_replay_json_path),
            "replay_payload": dict(replay_payload),
            "mean_width_px": float(mean_width_px),
            "junction_zone_radius_px": int(zone_r),
            "junction_resolved_count": int(junction_info["junction_resolved_count"]),
            "junction_unresolved_count": int(junction_info["junction_unresolved_count"]),
            "gap_reconnected_count": int(reconnect_info["gap_reconnected_count"]),
            "preview": np.asarray(preview, dtype=np.uint8),
            "roi_view": np.asarray(roi_view, dtype=np.uint8),
            "binary_prep_mode": str(binary_prep_mode),
        }

    def _build_simple_payload_with_optional_manual_tuning(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Optional[Dict[str, object]]:
        skip_manual_tuner = bool(self._skip_next_manual_threshold_tuner)
        self._skip_next_manual_threshold_tuner = False
        if self._simple_binary_prep_setting() == "manual_threshold" and not skip_manual_tuner:
            if not self._run_manual_threshold_tuner(roi_img=roi_img, roi_mask=roi_mask):
                return None
        try:
            return self._build_simple_payload(roi_img=roi_img, roi_mask=roi_mask)
        except RuntimeError as exc:
            if "Manual junction editing cancelled" in str(exc):
                self._last_cancel_reason = "Manual junction editing cancelled."
                return None
            if "Manual binary split editing cancelled" in str(exc):
                self._last_cancel_reason = "Manual binary split editing cancelled."
                return None
            raise

    def _build_payload_for_current_mode(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Optional[Dict[str, object]]:
        return self._build_simple_payload_with_optional_manual_tuning(roi_img=roi_img, roi_mask=roi_mask)

    def _apply_simple_payload(
        self,
        roi_xywh: Tuple[int, int, int, int],
        payload: Dict[str, object],
    ) -> bool:
        x, y, w, h = [int(v) for v in roi_xywh]
        cleaned_mask = np.asarray(payload["cleaned_mask"], dtype=np.uint8)
        raw_thin_mask = np.asarray(payload["raw_thin_mask"], dtype=np.uint8)
        final_thin_mask = np.asarray(payload["final_thin_mask"], dtype=np.uint8)
        l1_map = np.asarray(payload["l1_map"], dtype=np.uint8)
        l2_map = np.asarray(payload["l2_map"], dtype=np.uint8)
        region_depth_map = np.asarray(payload["region_depth_map"], dtype=np.int32)
        residual_map = np.asarray(payload["residual_map"], dtype=np.uint8)

        self.final_result_img = np.asarray(payload["preview"], dtype=np.uint8).copy()
        self.simple_roi_view = np.asarray(payload["roi_view"], dtype=np.uint8).copy()
        self.simple_roi_mask = np.asarray(payload.get("roi_mask", np.zeros((h, w), dtype=np.uint8)), dtype=np.uint8).copy()
        self.simple_final_thin_mask = final_thin_mask.copy()
        self.simple_region_depth_map = region_depth_map.copy()
        self.simple_binary_prep_mode = str(payload.get("binary_prep_mode", "unknown"))

        clear_legacy_line_map_attributes(self)
        self.line_maps_roi_bbox_xywh = (x, y, w, h)
        keep_legacy_line_maps = bool(self.params.get("KEEP_LEGACY_LINE_MAPS", False))
        if keep_legacy_line_maps:
            populate_legacy_line_map_attributes(self, roi_xywh=(x, y, w, h), payload=payload)

        status = str(payload.get("status", ""))
        message = str(payload.get("message", ""))
        binary_info = dict(payload.get("binary_info", {}))
        detected_levels = [int(v) for v in payload.get("detected_levels", [])]
        max_detected_level = int(payload.get("max_detected_level", 0))
        unassigned_level_px = int(payload.get("unassigned_level_px", 0))
        recovered_level_px = int(payload.get("recovered_level_px", 0))
        repair_bridge_count = int(payload.get("repair_bridge_count", 0))
        repair_added_px = int(payload.get("repair_added_px", 0))
        raw_node_info = dict(payload.get("raw_node_info", {}))
        final_node_info = dict(payload.get("final_node_info", {}))
        noise_filter_info = dict(payload.get("noise_filter_info", {}))
        spur_prune_info = dict(payload.get("spur_prune_info", {}))
        manual_junction_edit_info = dict(payload.get("manual_junction_edit_info", {}))
        ambiguous_node_info = dict(payload.get("ambiguous_node_info", {}))
        manual_binary_split_info = dict(payload.get("manual_binary_split_info", {}))
        mean_width_px = float(payload.get("mean_width_px", 2.0))
        zone_r = int(payload.get("junction_zone_radius_px", 1))
        junction_resolved_count = int(payload.get("junction_resolved_count", 0))
        junction_unresolved_count = int(payload.get("junction_unresolved_count", 0))
        gap_reconnected_count = int(payload.get("gap_reconnected_count", 0))

        self.results["roi_bbox_xywh"] = [x, y, w, h]
        self.results["line_maps_coordinate_frame"] = "roi"
        self.results["line_maps_roi_bbox_xywh"] = [x, y, w, h]
        self.results["line_maps_original_image_shape_hw"] = [int(self.img.shape[0]), int(self.img.shape[1])]
        self.results["legacy_line_maps_kept"] = bool(keep_legacy_line_maps)
        self.results["direct_contour_method"] = "roi_cleanmask_thin_iterative_outer_peel"
        self.results["simple_status"] = status
        self.results["simple_message"] = message
        self.results["simple_clean_px"] = int(np.count_nonzero(cleaned_mask))
        self.results["simple_thin_px"] = int(np.count_nonzero(raw_thin_mask))
        self.results["simple_final_thin_px"] = int(np.count_nonzero(final_thin_mask))
        self.results["simple_l1_px"] = int(np.count_nonzero(l1_map))
        self.results["simple_l2_px"] = int(np.count_nonzero(l2_map))
        self.results["simple_residual_px"] = int(np.count_nonzero(residual_map))
        self.results["simple_detected_level_count"] = int(len(detected_levels))
        self.results["simple_max_detected_level"] = int(max_detected_level)
        self.results["simple_level_assignment_mode"] = "iterative_outer_peel"
        self.results["simple_unassigned_level_px"] = int(unassigned_level_px)
        self.results["simple_recovered_level_px"] = int(recovered_level_px)
        self.results["simple_iterative_repair_bridge_count"] = int(repair_bridge_count)
        self.results["simple_iterative_repair_added_px"] = int(repair_added_px)
        self.results["simple_raw_endpoint_count"] = int(raw_node_info.get("endpoint_count", 0))
        self.results["simple_raw_junction_count"] = int(raw_node_info.get("junction_count", 0))
        self.results["simple_raw_endpoint_points_xy"] = list(raw_node_info.get("endpoint_points_xy", []))
        self.results["simple_raw_junction_points_xy"] = list(raw_node_info.get("junction_points_xy", []))
        self.results["simple_endpoint_count"] = int(final_node_info.get("endpoint_count", 0))
        self.results["simple_junction_count"] = int(final_node_info.get("junction_count", 0))
        self.results["simple_endpoint_points_xy"] = list(final_node_info.get("endpoint_points_xy", []))
        self.results["simple_junction_points_xy"] = list(final_node_info.get("junction_points_xy", []))
        self.results["simple_mean_stroke_width_px"] = float(mean_width_px)
        self.results["simple_junction_zone_radius_px"] = int(zone_r)
        self.results["simple_junction_resolved_count"] = int(junction_resolved_count)
        self.results["simple_junction_unresolved_count"] = int(junction_unresolved_count)
        self.results["simple_gap_reconnected_count"] = int(gap_reconnected_count)
        self.results["simple_open_noise_max_px"] = int(noise_filter_info.get("open_noise_max_px", 0))
        self.results["simple_removed_open_noise_component_count"] = int(
            noise_filter_info.get("removed_open_noise_component_count", 0)
        )
        self.results["simple_removed_open_noise_px"] = int(noise_filter_info.get("removed_open_noise_px", 0))
        self.results["simple_spur_prune_enabled"] = bool(spur_prune_info.get("spur_prune_enabled", 0))
        self.results["simple_spur_prune_max_px"] = int(spur_prune_info.get("spur_prune_max_px", 0))
        self.results["simple_removed_spur_branch_count"] = int(spur_prune_info.get("removed_spur_branch_count", 0))
        self.results["simple_removed_spur_px"] = int(spur_prune_info.get("removed_spur_px", 0))
        self.results["simple_manual_junction_editor_opened"] = bool(manual_junction_edit_info.get("opened", False))
        self.results["simple_manual_junction_edit_used"] = bool(manual_junction_edit_info.get("edited", False))
        self.results["simple_manual_junction_cut_count"] = int(manual_junction_edit_info.get("cut_count", 0))
        self.results["simple_manual_junction_cuts"] = list(manual_junction_edit_info.get("cuts", []))
        self.results["simple_manual_binary_split_editor_opened"] = bool(manual_binary_split_info.get("opened", False))
        self.results["simple_manual_binary_split_used"] = bool(manual_binary_split_info.get("edited", False))
        self.results["simple_manual_binary_split_loaded"] = bool(manual_binary_split_info.get("loaded", False))
        self.results["simple_manual_binary_split_cut_count"] = int(manual_binary_split_info.get("cut_count", 0))
        self.results["simple_manual_binary_split_cuts"] = list(manual_binary_split_info.get("cuts", []))
        self.results["simple_ambiguous_node_resolved_zone_count"] = int(ambiguous_node_info.get("resolved_zone_count", 0))
        self.results["simple_ambiguous_node_removed_px"] = int(ambiguous_node_info.get("removed_px", 0))
        self.results["simple_replay_json_path"] = str(payload.get("replay_json_path", ""))
        self.results["simple_replay_json_loaded"] = bool(payload.get("replay_json_loaded", False))
        self.results["simple_replay_json_loaded_path"] = str(payload.get("replay_json_loaded_path", ""))
        self.results["simple_replay_payload"] = dict(payload.get("replay_payload", {}))
        self.results["simple_binary_prep_mode"] = self.simple_binary_prep_mode
        self.results["simple_binary_info"] = dict(binary_info)
        self.results["simple_manual_gray_thresh"] = binary_info.get("threshold", None)
        self.results["simple_manual_threshold_invert"] = binary_info.get("invert", None)
        self.results["level1_result_px"] = int(np.count_nonzero(l1_map))
        self.results["level2_result_px"] = int(np.count_nonzero(l2_map))
        self.results["level1_residual_px"] = int(np.count_nonzero(residual_map))
        self.results["v8_simple_mode"] = True
        self.results["direct_contour_pipeline_mode"] = "simple_junction_resolve_iterative_outer_peel"

        print(f"-> {message}")
        return True

    def _generate_line_centerline_preview(self) -> bool:
        print(" -> Generating v8 simple level-map preview ...")

        (x, y, w, h), roi_img, roi_mask = _build_roi_from_polygon(self.img, self.points)
        payload = self._build_payload_for_current_mode(roi_img=roi_img, roi_mask=roi_mask)
        if payload is None:
            self.results["simple_status"] = "cancelled"
            self.results["simple_message"] = self._last_cancel_reason or "Processing cancelled."
            return False
        return self._apply_simple_payload(roi_xywh=(int(x), int(y), int(w), int(h)), payload=payload)

    def _simple_window_max_width(self) -> int:
        return int(max(640, round(self.params.get("SIMPLE_WINDOW_MAX_WIDTH", 1600))))

    def _simple_window_max_height(self) -> int:
        return int(max(480, round(self.params.get("SIMPLE_WINDOW_MAX_HEIGHT", 1000))))

    def _prepare_back_to_junction_from_preview(self) -> bool:
        binary_split_cuts = list(self.results.get("simple_manual_binary_split_cuts", []))
        junction_cuts = list(self.results.get("simple_manual_junction_cuts", []))
        self._manual_edit_session_override = {
            "binary_split_cuts": list(binary_split_cuts),
            "junction_cuts": list(junction_cuts),
        }
        self._skip_next_manual_threshold_tuner = True
        self._force_skip_binary_split_once = True
        self._force_reopen_junction_editor_once = True
        return True

    def _show_final_preview_window(self) -> str:
        if self.final_result_img is None:
            return "close"
        window_name = "JetAnalyzerV8 simple level-map preview"
        disp = self._fit_image_to_box(
            self.final_result_img,
            self._simple_window_max_width(),
            self._simple_window_max_height(),
            allow_upscale=False,
        )
        keep_ratio_flag = getattr(cv2, "WINDOW_KEEPRATIO", cv2.WINDOW_NORMAL)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | keep_ratio_flag)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        except Exception:
            pass
        cv2.resizeWindow(window_name, int(disp.shape[1]), int(disp.shape[0]))
        action = "close"
        print("Preview window: Enter/Space/q/Esc to close, b to go back to junction split.")
        try:
            while True:
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), 13, 10, 32):
                    break
                if key in (ord("b"), ord("B")):
                    action = "back_to_junction"
                    break
        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
        return action

    def _run_roi_processing_loop(
        self,
        roi_xywh: Tuple[int, int, int, int],
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> bool:
        while True:
            payload = self._build_payload_for_current_mode(
                roi_img=np.asarray(roi_img, dtype=np.uint8).copy(),
                roi_mask=np.asarray(roi_mask, dtype=np.uint8).copy(),
            )
            if payload is None:
                self.results["simple_status"] = "cancelled"
                self.results["simple_message"] = self._last_cancel_reason or "Processing cancelled."
                return False
            if not self._apply_simple_payload(roi_xywh=roi_xywh, payload=payload):
                return False
            action = self._show_final_preview_window()
            if action == "back_to_junction":
                self._prepare_back_to_junction_from_preview()
                continue
            return True

    def run(self):
        print(f"=== Jet Analyzer v8 Started: {self.image_path} ===")
        replay_session = self._load_replay_session_from_config()
        if replay_session is not None:
            print(f"-> Loading replay JSON: {self._loaded_replay_json_path}")
            replay_mode = str(replay_session.get("binary_prep_mode", self.params.get("SIMPLE_BINARY_PREP_MODE", "manual_threshold")) or "manual_threshold")
            self.params["SIMPLE_BINARY_PREP_MODE"] = replay_mode
            if replay_session.get("manual_gray_thresh", None) is not None:
                self.params["SIMPLE_MANUAL_GRAY_THRESH"] = int(replay_session.get("manual_gray_thresh"))
            if replay_session.get("manual_threshold_invert", None) is not None:
                self.params["SIMPLE_MANUAL_THRESHOLD_INVERT"] = bool(replay_session.get("manual_threshold_invert"))
            self._skip_next_manual_threshold_tuner = True
            self._manual_edit_session_override = {
                "binary_split_cuts": list(replay_session.get("binary_split_cuts", [])),
                "junction_cuts": list(replay_session.get("junction_cuts", [])),
            }
            points = [
                (int(pt[0]), int(pt[1]))
                for pt in list(replay_session.get("roi_points_xy", []))
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if len(points) < 3:
                raise ValueError("Replay JSON must contain at least 3 ROI points.")
            self.roi_polygon_points = list(points)
            self.points = list(points)
            print("Workflow: replay JSON -> threshold binary -> manual binary split -> thin -> spur prune -> manual junction split -> level map")
            (x, y, w, h), roi_img, roi_mask = _build_roi_from_polygon(self.img, self.points)
            self._run_roi_processing_loop(
                roi_xywh=(int(x), int(y), int(w), int(h)),
                roi_img=roi_img,
                roi_mask=roi_mask,
            )
            return
        if self._simple_use_full_image_roi():
            print("Workflow: full image ROI -> threshold binary -> normalized mask -> manual binary split -> thin -> spur prune -> manual junction split -> level map")
            if self.params.get("DEBUG_MODE", False):
                print(f"[DEBUG] Params: {self.params}")
            roi_mask = np.ones(self.img.shape[:2], dtype=np.uint8) * 255
            self._run_roi_processing_loop(
                roi_xywh=(0, 0, int(self.img.shape[1]), int(self.img.shape[0])),
                roi_img=self.img.copy(),
                roi_mask=roi_mask,
            )
            return
        print("Workflow: polygon ROI -> threshold binary -> normalized mask -> thin -> spur prune -> manual junction split -> level map")
        if self.params.get("DEBUG_MODE", False):
            print(f"[DEBUG] Params: {self.params}")
        points = select_polygon_roi(
            image=self.clone_base,
            window_name="JetAnalyzerV8 ROI",
            max_width=self._simple_window_max_width(),
            max_height=self._simple_window_max_height(),
        )
        if not points or len(points) < 3:
            self.results["simple_status"] = "cancelled"
            self.results["simple_message"] = "ROI selection cancelled."
            print("-> ROI selection cancelled.")
            return
        self.roi_polygon_points = list(points)
        self.points = list(points)
        print(" -> Generating v8 simple level-map preview ...")
        (x, y, w, h), roi_img, roi_mask = _build_roi_from_polygon(self.img, self.points)
        self._run_roi_processing_loop(
            roi_xywh=(int(x), int(y), int(w), int(h)),
            roi_img=roi_img,
            roi_mask=roi_mask,
        )


def run_v8_preview(image_path: str, config: Optional[dict] = None) -> JetAnalyzerV8Simple:
    analyzer = JetAnalyzerV8Simple(image_path=image_path, config=config)
    analyzer.run()
    return analyzer
