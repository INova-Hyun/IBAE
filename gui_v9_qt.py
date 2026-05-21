from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:
    _sk_skeletonize = None
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cbook as _mpl_cbook
import matplotlib.cm as mpl_cm
from matplotlib import colors as mcolors
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .common.image import ensure_bgr as _ensure_bgr
from .common.numeric import safe_float as _safe_float
from .flux import downsample_flux_for_surface, reconstruct_flux_from_levels
from .reports import (
    auto_select_k_near_one_range,
    build_gaussian_report_rows,
    compute_local_k,
    find_opening_angle_plateau,
    fit_power_law_from_rows,
    paper_fig7_eastern_broken_power_law,
)
from .session_analysis import (
    array_sha256,
    default_reconstruction_cache_path,
    load_analysis_session,
    load_reconstruction_cache,
    save_analysis_session,
    save_reconstruction_cache,
)
from .ridgeline import (
    _blocked_median_summary,
    apply_core_separation_to_measure_result,
    beam_size_px as analysis_beam_size_px,
    evaluate_gaussian_fit_stability,
    extract_ridgeline,
    fit_transverse_gaussian,
    measure_ridgeline_fwhm,
    sample_transverse_profile,
)

if not hasattr(_mpl_cbook.Grouper, "clean"):
    _mpl_cbook.Grouper.clean = lambda self: None

Point = Tuple[int, int]


def _prepare_qt_runtime_env() -> None:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "")
    try:
        if (not runtime_dir) or (not os.path.isdir(runtime_dir)) or ((os.stat(runtime_dir).st_mode & 0o777) != 0o700):
            fallback_runtime = os.path.join(tempfile.gettempdir(), f"ibae_qt_runtime_{os.getuid()}")
            os.makedirs(fallback_runtime, mode=0o700, exist_ok=True)
            try:
                os.chmod(fallback_runtime, 0o700)
            except Exception:
                pass
            os.environ["XDG_RUNTIME_DIR"] = fallback_runtime
    except Exception:
        pass
    plugin_root = str(QLibraryInfo.location(QLibraryInfo.PluginsPath))
    platform_dir = os.path.join(plugin_root, "platforms")
    if os.path.isdir(platform_dir):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_dir
    if plugin_root:
        current_plugin_path = os.environ.get("QT_PLUGIN_PATH", "")
        if (not current_plugin_path) or ("/cv2/qt/plugins" in current_plugin_path):
            os.environ["QT_PLUGIN_PATH"] = plugin_root


_QT_APP = None


def ensure_qt_app() -> QApplication:
    global _QT_APP
    _prepare_qt_runtime_env()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
        app.setApplicationName("IBAE Opening Angle Qt")
    _QT_APP = app
    return app


def _analysis_image_path(analysis_context: Dict[str, object]) -> str:
    return str(dict(analysis_context or {}).get("image_path", "") or "")


def _load_calibration_image_from_path(image_path: str) -> Optional[np.ndarray]:
    path = str(image_path or "").strip()
    if not path:
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _safe_eval_ratio_expression(text: str) -> Optional[float]:
    expr = str(text or "").strip()
    if not expr:
        return None
    namespace = {
        "__builtins__": {},
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e,
    }
    try:
        value = eval(expr, namespace, {})
    except Exception:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return float(value)


def _normalize_mpl_color(value: object, fallback: str = "#00bcd4") -> str:
    try:
        return str(mcolors.to_hex(mcolors.to_rgba(value), keep_alpha=False))
    except Exception:
        return str(fallback)


def _apply_color_button_preview(button: QPushButton, color_value: object) -> None:
    color_hex = _normalize_mpl_color(color_value)
    button.setText(color_hex)
    qcolor = QColor(color_hex)
    text_color = "#000000" if qcolor.lightnessF() > 0.6 else "#ffffff"
    button.setStyleSheet(
        f"QPushButton {{ background-color: {color_hex}; color: {text_color}; "
        "border: 1px solid #888; padding: 3px 6px; }}"
    )


class QtTrendPlotStyleDialog(QDialog):
    def __init__(self, current: Dict[str, object], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Trend / Report Plot Style")
        self._defaults = {
            "main_color": "#17becf",
            "main_size": 34.0,
            "raw_color": "#17becf",
            "raw_size": 28.0,
            "peak_color": "#1f77b4",
            "peak_size": 34.0,
            "residual_color": "#9467bd",
            "residual_size": 18.0,
            "show_raw_width": True,
            "show_raw_angle": True,
        }
        self._current = dict(self._defaults)
        self._current.update(dict(current or {}))

        self.main_color_button = QPushButton()
        self.raw_color_button = QPushButton()
        self.peak_color_button = QPushButton()
        self.residual_color_button = QPushButton()
        self.main_size_spin = QDoubleSpinBox()
        self.raw_size_spin = QDoubleSpinBox()
        self.peak_size_spin = QDoubleSpinBox()
        self.residual_size_spin = QDoubleSpinBox()
        for spin in (
            self.main_size_spin,
            self.raw_size_spin,
            self.peak_size_spin,
            self.residual_size_spin,
        ):
            spin.setRange(2.0, 200.0)
            spin.setDecimals(1)
            spin.setSingleStep(2.0)
        self.show_raw_width_check = QCheckBox("Show raw FWHM in width plot")
        self.show_raw_angle_check = QCheckBox("Show raw angle in angle plot")

        self.main_size_spin.setValue(float(self._current.get("main_size", 34.0)))
        self.raw_size_spin.setValue(float(self._current.get("raw_size", 28.0)))
        self.peak_size_spin.setValue(float(self._current.get("peak_size", 34.0)))
        self.residual_size_spin.setValue(float(self._current.get("residual_size", 18.0)))
        self.show_raw_width_check.setChecked(bool(self._current.get("show_raw_width", True)))
        self.show_raw_angle_check.setChecked(bool(self._current.get("show_raw_angle", True)))
        _apply_color_button_preview(self.main_color_button, self._current.get("main_color", "#17becf"))
        _apply_color_button_preview(self.raw_color_button, self._current.get("raw_color", "#17becf"))
        _apply_color_button_preview(self.peak_color_button, self._current.get("peak_color", "#1f77b4"))
        _apply_color_button_preview(self.residual_color_button, self._current.get("residual_color", "#9467bd"))

        self.main_color_button.clicked.connect(lambda: self._choose_color(self.main_color_button))
        self.raw_color_button.clicked.connect(lambda: self._choose_color(self.raw_color_button))
        self.peak_color_button.clicked.connect(lambda: self._choose_color(self.peak_color_button))
        self.residual_color_button.clicked.connect(lambda: self._choose_color(self.residual_color_button))

        form = QFormLayout()
        form.addRow("Deconv point color", self.main_color_button)
        form.addRow("Deconv point size", self.main_size_spin)
        form.addRow("Raw point color", self.raw_color_button)
        form.addRow("Raw point size", self.raw_size_spin)
        form.addRow("Peak point color", self.peak_color_button)
        form.addRow("Peak point size", self.peak_size_spin)
        form.addRow("Residual point color", self.residual_color_button)
        form.addRow("Residual point size", self.residual_size_spin)
        form.addRow("", self.show_raw_width_check)
        form.addRow("", self.show_raw_angle_check)

        reset_button = QPushButton("Reset Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(reset_button)
        layout.addWidget(buttons)
        self.resize(360, 0)

    def _choose_color(self, button: QPushButton) -> None:
        current = QColor(button.text().strip() or "#ffffff")
        picked = QColorDialog.getColor(current, self, "Choose Plot Color")
        if picked.isValid():
            _apply_color_button_preview(button, picked.name())

    def _reset_defaults(self) -> None:
        self.main_size_spin.setValue(float(self._defaults["main_size"]))
        self.raw_size_spin.setValue(float(self._defaults["raw_size"]))
        self.peak_size_spin.setValue(float(self._defaults["peak_size"]))
        self.residual_size_spin.setValue(float(self._defaults["residual_size"]))
        self.show_raw_width_check.setChecked(bool(self._defaults["show_raw_width"]))
        self.show_raw_angle_check.setChecked(bool(self._defaults["show_raw_angle"]))
        _apply_color_button_preview(self.main_color_button, self._defaults["main_color"])
        _apply_color_button_preview(self.raw_color_button, self._defaults["raw_color"])
        _apply_color_button_preview(self.peak_color_button, self._defaults["peak_color"])
        _apply_color_button_preview(self.residual_color_button, self._defaults["residual_color"])

    def get_settings(self) -> Dict[str, object]:
        return {
            "main_color": _normalize_mpl_color(self.main_color_button.text(), "#17becf"),
            "main_size": float(self.main_size_spin.value()),
            "raw_color": _normalize_mpl_color(self.raw_color_button.text(), "#17becf"),
            "raw_size": float(self.raw_size_spin.value()),
            "peak_color": _normalize_mpl_color(self.peak_color_button.text(), "#1f77b4"),
            "peak_size": float(self.peak_size_spin.value()),
            "residual_color": _normalize_mpl_color(self.residual_color_button.text(), "#9467bd"),
            "residual_size": float(self.residual_size_spin.value()),
            "show_raw_width": bool(self.show_raw_width_check.isChecked()),
            "show_raw_angle": bool(self.show_raw_angle_check.isChecked()),
        }


def _format_contour_values_text(values: Sequence[float]) -> str:
    return "\n".join(format(float(v), ".12g") for v in list(values))


def _parse_contour_values_text(text: str) -> List[float]:
    raw = str(text or "").replace(",", " ").replace(";", " ").split()
    out: List[float] = []
    for token in raw:
        out.append(float(token))
    return out


def _build_color_norm_for_values(values: np.ndarray, log_enabled: bool):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return None, "linear"
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if (not log_enabled) or (abs(vmax - vmin) <= 1e-12):
        return None, "linear"
    if np.all(vals > 0.0):
        positive_min = float(np.min(vals[vals > 0.0]))
        return mcolors.LogNorm(vmin=positive_min, vmax=vmax), "log"
    nonzero_abs = np.abs(vals[np.abs(vals) > 1e-12])
    if nonzero_abs.size <= 0:
        return None, "linear"
    linthresh = float(np.min(nonzero_abs))
    linthresh = max(linthresh, float(np.max(nonzero_abs)) * 1e-6, 1e-9)
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10.0), "symlog"


def _normalize_for_colormap(values: np.ndarray, color_norm) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if color_norm is not None:
        return np.asarray(color_norm(vals), dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size <= 0:
        return np.zeros_like(vals, dtype=np.float64)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) <= 1e-12:
        return np.zeros_like(vals, dtype=np.float64)
    return (vals - vmin) / (vmax - vmin)


def _render_flux_color_image(
    flux_map: np.ndarray,
    valid_mask: np.ndarray,
    thin_mask: Optional[np.ndarray],
    cmap_name: str,
    log_enabled: bool,
) -> np.ndarray:
    flux = np.asarray(flux_map, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.uint8) > 0
    out = np.zeros((flux.shape[0], flux.shape[1], 3), dtype=np.uint8)
    values = flux[valid & np.isfinite(flux)]
    color_norm, _ = _build_color_norm_for_values(values, bool(log_enabled))
    rgba = mpl_cm.get_cmap(str(cmap_name))(
        np.clip(_normalize_for_colormap(flux, color_norm), 0.0, 1.0)
    )
    rgb = np.clip(rgba[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
    out[valid] = rgb[valid][:, ::-1]
    if thin_mask is not None:
        thin = np.asarray(thin_mask, dtype=np.uint8) > 0
        out[thin] = (255, 255, 255)
    return out


def _fit_image_to_box(image: np.ndarray, max_w: int, max_h: int, allow_upscale: bool = False) -> np.ndarray:
    img = _ensure_bgr(image)
    h, w = img.shape[:2]
    max_w = int(max(64, max_w))
    max_h = int(max(64, max_h))
    scale = min(float(max_w) / max(float(w), 1.0), float(max_h) / max(float(h), 1.0))
    if not allow_upscale:
        scale = min(scale, 1.0)
    if 0.999 <= scale <= 1.001:
        return img.copy()
    out_w = max(1, int(round(float(w) * scale)))
    out_h = max(1, int(round(float(h) * scale)))
    interp = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def _display_stride_for_1080p(shape_hw: Tuple[int, int]) -> int:
    h, w = [int(v) for v in shape_hw[:2]]
    if h <= 0 or w <= 0:
        return 1
    # Keep the interactive display buffer within a 1920x1080-sized canvas.
    stride = int(np.ceil(max(float(w) / 1920.0, float(h) / 1080.0)))
    return int(max(1, stride))


def _max_pool_mask_for_display(mask: np.ndarray, stride: int) -> np.ndarray:
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    stride_i = int(max(1, stride))
    if stride_i <= 1:
        return arr * 255
    h, w = arr.shape[:2]
    out_h = int(np.ceil(float(h) / float(stride_i)))
    out_w = int(np.ceil(float(w) / float(stride_i)))
    pad_h = max(0, (out_h * stride_i) - h)
    pad_w = max(0, (out_w * stride_i) - w)
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    pooled = arr.reshape(out_h, stride_i, out_w, stride_i).max(axis=(1, 3))
    return pooled.astype(np.uint8) * 255


def _square_bounds(center: Point, width: int, shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    cx, cy = [int(v) for v in center]
    size = int(max(1, width))
    half_lo = int((size - 1) // 2)
    half_hi = int(size // 2)
    h, w = [int(v) for v in shape_hw]
    x0 = int(np.clip(cx - half_lo, 0, max(0, w - 1)))
    y0 = int(np.clip(cy - half_lo, 0, max(0, h - 1)))
    x1 = int(np.clip(cx + half_hi, 0, max(0, w - 1)))
    y1 = int(np.clip(cy + half_hi, 0, max(0, h - 1)))
    return x0, y0, x1, y1


def _raster_line_points(p0: Point, p1: Point) -> List[Point]:
    x0, y0 = [int(v) for v in p0]
    x1, y1 = [int(v) for v in p1]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points: List[Point] = []
    while True:
        points.append((int(x0), int(y0)))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def _cluster_point_boxes(
    points: Sequence[Point],
    shape_hw: Tuple[int, int],
    pad: int = 3,
) -> List[Tuple[int, int, int, int]]:
    h, w = [int(v) for v in shape_hw]
    if h <= 0 or w <= 0 or not points:
        return []
    seed = np.zeros((h, w), dtype=np.uint8)
    for x, y in points:
        if 0 <= int(x) < w and 0 <= int(y) < h:
            seed[int(y), int(x)] = 255
    radius = int(max(1, pad))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    grown = cv2.dilate(seed, kernel, iterations=1)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, int(num)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + bw - 1, y + bh - 1))
    return boxes


def _skeleton_neighbors(mask: np.ndarray, x: int, y: int) -> List[Point]:
    h, w = mask.shape[:2]
    out: List[Point] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = int(x + dx)
            ny = int(y + dy)
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                out.append((nx, ny))
    return out


def _neighbor_stats_maps(centerline_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    skel = (np.asarray(centerline_roi, dtype=np.uint8) > 0)
    h, w = skel.shape[:2]
    neighbor_count = np.zeros((h, w), dtype=np.uint8)
    branch_group_count = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.nonzero(skel)
    for x, y in zip(xs.tolist(), ys.tolist()):
        x_i = int(x)
        y_i = int(y)
        nbs = _skeleton_neighbors(skel, x_i, y_i)
        neighbor_count[y_i, x_i] = int(len(nbs))
        if not nbs:
            continue
        ring = np.zeros((3, 3), dtype=np.uint8)
        for nx, ny in nbs:
            ring[int(ny - y_i + 1), int(nx - x_i + 1)] = 1
        count, _ = cv2.connectedComponents(ring, connectivity=8)
        branch_group_count[y_i, x_i] = int(max(0, int(count) - 1))
    return neighbor_count, branch_group_count


def _node_points_xy_local(thin_mask: np.ndarray) -> Dict[str, object]:
    thin_u8 = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if not np.any(thin_u8):
        return {
            "endpoint_count": 0,
            "junction_count": 0,
            "endpoint_points_xy": [],
            "junction_points_xy": [],
        }
    neigh, branch = _neighbor_stats_maps(thin_u8)
    endpoint_mask = (thin_u8 > 0) & (neigh == 1)
    junction_mask = (thin_u8 > 0) & (branch >= 3)
    ys_e, xs_e = np.nonzero(endpoint_mask)
    ys_j, xs_j = np.nonzero(junction_mask)
    return {
        "endpoint_count": int(len(xs_e)),
        "junction_count": int(len(xs_j)),
        "endpoint_points_xy": [[int(x), int(y)] for x, y in zip(xs_e.tolist(), ys_e.tolist())],
        "junction_points_xy": [[int(x), int(y)] for x, y in zip(xs_j.tolist(), ys_j.tolist())],
    }


def _skeletonize_fallback(binary_roi: np.ndarray) -> np.ndarray:
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


def _thin_binary_local(binary_roi: np.ndarray) -> np.ndarray:
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
    return _skeletonize_fallback(mask)


def _qimage_from_array(image: np.ndarray) -> QImage:
    bgr = _ensure_bgr(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
    return qimg.copy()


def _set_label_image(label: QLabel, image: np.ndarray) -> None:
    qimg = _qimage_from_array(image)
    label.setPixmap(QPixmap.fromImage(qimg))
    label.setFixedSize(qimg.size())


class InteractiveImageLabel(QLabel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.on_left_press: Optional[Callable[[QMouseEvent], None]] = None
        self.on_left_release: Optional[Callable[[QMouseEvent], None]] = None
        self.on_move: Optional[Callable[[QMouseEvent], None]] = None
        self.on_right_press: Optional[Callable[[QMouseEvent], None]] = None
        self.on_middle_press: Optional[Callable[[QMouseEvent], None]] = None
        self.on_middle_release: Optional[Callable[[QMouseEvent], None]] = None
        self.on_wheel: Optional[Callable[[QWheelEvent], None]] = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.on_left_press is not None:
            self.on_left_press(event)
        elif event.button() == Qt.RightButton and self.on_right_press is not None:
            self.on_right_press(event)
        elif event.button() == Qt.MiddleButton and self.on_middle_press is not None:
            self.on_middle_press(event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self.on_left_release is not None:
            self.on_left_release(event)
        elif event.button() == Qt.MiddleButton and self.on_middle_release is not None:
            self.on_middle_release(event)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.on_move is not None:
            self.on_move(event)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.on_wheel is not None:
            self.on_wheel(event)
        super().wheelEvent(event)


class _QtViewportMixin:
    def _init_viewport_state(self, image_shape: Tuple[int, int], max_panel_w: int, max_panel_h: int) -> None:
        self.max_panel_w = int(max(160, max_panel_w))
        self.max_panel_h = int(max(160, max_panel_h))
        self.zoom = 1.0
        self._zoom_step = 10
        h, w = [int(v) for v in image_shape[:2]]
        self.view_center = (float(max(0, w - 1)) / 2.0, float(max(0, h - 1)) / 2.0)
        self._display_shape = (h, w)
        self._view_rect_xyxy = (0, 0, w, h)
        self._trackbars_sync_enabled = True

    def _clamped_view_center(self, base_shape: Tuple[int, int]) -> Tuple[float, float]:
        base_h, base_w = [int(v) for v in base_shape[:2]]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        return (
            float(np.clip(self.view_center[0], min_cx, max_cx)),
            float(np.clip(self.view_center[1], min_cy, max_cy)),
        )

    def _current_view_rect(self, base_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        base_h, base_w = [int(v) for v in base_shape[:2]]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(base_shape)
        x0 = int(round(cx - (view_w / 2.0)))
        y0 = int(round(cy - (view_h / 2.0)))
        x0 = int(np.clip(x0, 0, max(0, base_w - view_w)))
        y0 = int(np.clip(y0, 0, max(0, base_h - view_h)))
        x1 = int(min(base_w, x0 + view_w))
        y1 = int(min(base_h, y0 + view_h))
        return (x0, y0, x1, y1)

    def _render_zoom_view(self, source: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self._current_view_rect(source.shape[:2])
        crop = _ensure_bgr(source)[y0:y1, x0:x1]
        disp = _fit_image_to_box(crop, max_w=self.max_panel_w, max_h=self.max_panel_h, allow_upscale=True)
        self._view_rect_xyxy = (int(x0), int(y0), int(x1), int(y1))
        self._display_shape = (int(disp.shape[0]), int(disp.shape[1]))
        return disp

    def _display_to_image_point(self, pos: QPoint, base_shape: Tuple[int, int]) -> Optional[Point]:
        disp_h, disp_w = [int(v) for v in self._display_shape]
        x = int(pos.x())
        y = int(pos.y())
        if x < 0 or x >= disp_w or y < 0 or y >= disp_h:
            return None
        x0, y0, x1, y1 = self._view_rect_xyxy
        view_w = max(1, int(x1 - x0))
        view_h = max(1, int(y1 - y0))
        px = int(round(float(x0) + (float(x) * float(view_w) / float(max(1, disp_w)))))
        py = int(round(float(y0) + (float(y) * float(view_h) / float(max(1, disp_h)))))
        base_h, base_w = [int(v) for v in base_shape[:2]]
        px = int(np.clip(px, 0, max(0, base_w - 1)))
        py = int(np.clip(py, 0, max(0, base_h - 1)))
        return (px, py)

    def _image_to_display_point(self, pt: Point) -> Optional[Point]:
        px, py = [int(v) for v in pt]
        x0, y0, x1, y1 = self._view_rect_xyxy
        if px < x0 or px >= x1 or py < y0 or py >= y1:
            return None
        disp_h, disp_w = [int(v) for v in self._display_shape]
        view_w = max(1, int(x1 - x0))
        view_h = max(1, int(y1 - y0))
        dx = int(round(float(px - x0) * float(disp_w) / float(view_w)))
        dy = int(round(float(py - y0) * float(disp_h) / float(view_h)))
        dx = int(np.clip(dx, 0, max(0, disp_w - 1)))
        dy = int(np.clip(dy, 0, max(0, disp_h - 1)))
        return (dx, dy)


class QtThresholdTunerDialog(QDialog, _QtViewportMixin):
    def __init__(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
        preview_callback: Callable[[int, bool], np.ndarray],
        initial_threshold: int,
        initial_invert: bool,
        max_width: int,
        max_height: int,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("JetAnalyzerV9 Manual Threshold")
        self.roi_img = np.asarray(roi_img, dtype=np.uint8)
        self.roi_mask = np.asarray(roi_mask, dtype=np.uint8)
        self.preview_callback = preview_callback
        self._init_viewport_state(self.roi_img.shape[:2], max_width // 2, max_height - 180)
        self.accepted_threshold = int(initial_threshold)
        self.accepted_invert = bool(initial_invert)

        self.left_label = QLabel()
        self.right_label = QLabel()
        self.left_title = QLabel("ROI grayscale")
        self.right_title = QLabel("Thresholded binary")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(int(initial_threshold))
        self.invert_check = QCheckBox("Invert")
        self.invert_check.setChecked(bool(initial_invert))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(10)
        self.pan_x_slider = QSlider(Qt.Horizontal)
        self.pan_x_slider.setRange(0, 1000)
        self.pan_y_slider = QSlider(Qt.Horizontal)
        self.pan_y_slider.setRange(0, 1000)

        top = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(self.left_title)
        left_box.addWidget(self.left_label)
        right_box = QVBoxLayout()
        right_box.addWidget(self.right_title)
        right_box.addWidget(self.right_label)
        top.addLayout(left_box)
        top.addLayout(right_box)

        controls = QFormLayout()
        controls.addRow("Threshold", self.threshold_slider)
        controls.addRow("", self.invert_check)
        controls.addRow("Zoom x10", self.zoom_slider)
        controls.addRow("Pan X", self.pan_x_slider)
        controls.addRow("Pan Y", self.pan_y_slider)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(controls)
        layout.addWidget(buttons)

        self.threshold_slider.valueChanged.connect(self.refresh_views)
        self.invert_check.stateChanged.connect(self.refresh_views)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.pan_x_slider.valueChanged.connect(self._on_pan_changed)
        self.pan_y_slider.valueChanged.connect(self._on_pan_changed)
        self.refresh_views()

    def _on_zoom_changed(self, value: int) -> None:
        self.zoom = float(max(10, int(value))) / 10.0
        self.refresh_views(sync_sliders=True)

    def _on_pan_changed(self, _value: int) -> None:
        base_h, base_w = self.roi_img.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        max_pos = 1000.0
        px = float(self.pan_x_slider.value())
        py = float(self.pan_y_slider.value())
        if max_cx <= min_cx + 1e-6:
            cx = min_cx
        else:
            cx = min_cx + (px * float(max_cx - min_cx) / max_pos)
        if max_cy <= min_cy + 1e-6:
            cy = min_cy
        else:
            cy = min_cy + (py * float(max_cy - min_cy) / max_pos)
        self.view_center = (float(cx), float(cy))
        self.refresh_views(sync_sliders=False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.max_panel_w = int(max(160, (self.width() - 80) // 2))
        self.max_panel_h = int(max(160, self.height() - 220))
        self.refresh_views(sync_sliders=False)

    def _sync_sliders(self) -> None:
        if not self._trackbars_sync_enabled:
            return
        base_h, base_w = self.roi_img.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(self.roi_img.shape[:2])
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        self._trackbars_sync_enabled = False
        self.zoom_slider.setValue(int(round(self.zoom * 10.0)))
        if max_cx <= min_cx + 1e-6:
            self.pan_x_slider.setValue(0)
        else:
            self.pan_x_slider.setValue(int(round((cx - min_cx) * 1000.0 / float(max_cx - min_cx))))
        if max_cy <= min_cy + 1e-6:
            self.pan_y_slider.setValue(0)
        else:
            self.pan_y_slider.setValue(int(round((cy - min_cy) * 1000.0 / float(max_cy - min_cy))))
        self._trackbars_sync_enabled = True

    def refresh_views(self, *_args, sync_sliders: bool = True) -> None:
        threshold = int(self.threshold_slider.value())
        invert = bool(self.invert_check.isChecked())
        preview = np.asarray(self.preview_callback(threshold, invert), dtype=np.uint8)
        h, w = preview.shape[:2]
        left = preview[:, : (w // 2)]
        right = preview[:, (w // 2):]
        left_disp = self._render_zoom_view(left)
        right_disp = self._render_zoom_view(right)
        _set_label_image(self.left_label, left_disp)
        _set_label_image(self.right_label, right_disp)
        if sync_sliders:
            self._sync_sliders()

    def get_values(self) -> Tuple[int, bool]:
        return int(self.threshold_slider.value()), bool(self.invert_check.isChecked())


class QtRoiSelectorDialog(QDialog, _QtViewportMixin):
    def __init__(self, image: np.ndarray, max_width: int, max_height: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("JetAnalyzerV9 ROI")
        self.base_image = _ensure_bgr(image)
        self.points: List[Point] = []
        self.hover_point: Optional[Point] = None
        self.loaded_json_path: str = ""
        self._init_viewport_state(self.base_image.shape[:2], max_width, max_height - 170)
        self.label = InteractiveImageLabel()
        self.info = QLabel("Left click: add point | Right click: undo | Enter: confirm | C: clear")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(10)
        self.pan_x_slider = QSlider(Qt.Horizontal)
        self.pan_x_slider.setRange(0, 1000)
        self.pan_y_slider = QSlider(Qt.Horizontal)
        self.pan_y_slider.setRange(0, 1000)
        self.reset_button = QPushButton("Clear")
        self.load_json_button = QPushButton("Load JSON...")
        self.ok_button = QPushButton("Confirm ROI")
        self.ok_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")

        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.label)
        controls = QFormLayout()
        controls.addRow("Zoom x10", self.zoom_slider)
        controls.addRow("Pan X", self.pan_x_slider)
        controls.addRow("Pan Y", self.pan_y_slider)
        layout.addLayout(controls)
        buttons = QHBoxLayout()
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.load_json_button)
        buttons.addStretch(1)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.label.on_left_press = self._on_left_press
        self.label.on_move = self._on_move
        self.label.on_right_press = self._on_right_press
        self.label.on_wheel = self._on_wheel
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.pan_x_slider.valueChanged.connect(self._on_pan_changed)
        self.pan_y_slider.valueChanged.connect(self._on_pan_changed)
        self.reset_button.clicked.connect(self._clear_points)
        self.load_json_button.clicked.connect(self._load_json)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.refresh_view()

    def _load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Replay / Analysis JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self.loaded_json_path = str(path)
        self.accept()

    def _on_zoom_changed(self, value: int) -> None:
        self.zoom = float(max(10, int(value))) / 10.0
        self.refresh_view(sync_sliders=True)

    def _on_pan_changed(self, _value: int) -> None:
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        px = float(self.pan_x_slider.value())
        py = float(self.pan_y_slider.value())
        if max_cx <= min_cx + 1e-6:
            cx = min_cx
        else:
            cx = min_cx + (px * float(max_cx - min_cx) / 1000.0)
        if max_cy <= min_cy + 1e-6:
            cy = min_cy
        else:
            cy = min_cy + (py * float(max_cy - min_cy) / 1000.0)
        self.view_center = (float(cx), float(cy))
        self.refresh_view(sync_sliders=False)

    def _sync_sliders(self) -> None:
        if not self._trackbars_sync_enabled:
            return
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(self.base_image.shape[:2])
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        self._trackbars_sync_enabled = False
        self.zoom_slider.setValue(int(round(self.zoom * 10.0)))
        self.pan_x_slider.setValue(0 if max_cx <= min_cx + 1e-6 else int(round((cx - min_cx) * 1000.0 / float(max_cx - min_cx))))
        self.pan_y_slider.setValue(0 if max_cy <= min_cy + 1e-6 else int(round((cy - min_cy) * 1000.0 / float(max_cy - min_cy))))
        self._trackbars_sync_enabled = True

    def _on_left_press(self, event: QMouseEvent) -> None:
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if pt is None:
            return
        self.points.append((int(pt[0]), int(pt[1])))
        self.hover_point = pt
        self.ok_button.setEnabled(len(self.points) >= 3)
        self.refresh_view(sync_sliders=False)

    def _on_move(self, event: QMouseEvent) -> None:
        self.hover_point = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        self.refresh_view(sync_sliders=False)

    def _on_right_press(self, _event: QMouseEvent) -> None:
        if self.points:
            self.points.pop()
            if not self.points:
                self.hover_point = None
            self.ok_button.setEnabled(len(self.points) >= 3)
            self.refresh_view(sync_sliders=False)

    def _clear_points(self) -> None:
        self.points = []
        self.hover_point = None
        self.ok_button.setEnabled(False)
        self.refresh_view(sync_sliders=False)

    def _on_wheel(self, event: QWheelEvent) -> None:
        delta = int(event.angleDelta().y())
        if delta == 0:
            return
        self.zoom = float(np.clip((self.zoom * 10.0) + (1 if delta > 0 else -1), 10.0, 200.0)) / 10.0
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if pt is not None:
            self.view_center = (float(pt[0]), float(pt[1]))
        self.refresh_view(sync_sliders=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.max_panel_w = int(max(160, self.width() - 80))
        self.max_panel_h = int(max(160, self.height() - 220))
        self.refresh_view(sync_sliders=False)

    def refresh_view(self, sync_sliders: bool = True) -> None:
        canvas = self.base_image.copy()
        if self.points:
            cv2.polylines(canvas, [np.asarray(self.points, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
            if self.hover_point is not None:
                cv2.line(
                    canvas,
                    tuple(map(int, self.points[-1])),
                    tuple(map(int, self.hover_point)),
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
            for pt in self.points:
                cv2.circle(canvas, tuple(map(int, pt)), 3, (0, 0, 255), -1, cv2.LINE_AA)
            if len(self.points) >= 3:
                cv2.line(
                    canvas,
                    tuple(map(int, self.points[-1])),
                    tuple(map(int, self.points[0])),
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )
        disp = self._render_zoom_view(canvas)
        _set_label_image(self.label, disp)
        if sync_sliders:
            self._sync_sliders()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        if key in (Qt.Key_Return, Qt.Key_Enter) and len(self.points) >= 3:
            self.accept()
            return
        if key == Qt.Key_C:
            self._clear_points()
            return
        super().keyPressEvent(event)

    def get_points(self) -> List[Point]:
        return [(int(x), int(y)) for x, y in self.points]

    def get_loaded_json_path(self) -> str:
        return str(self.loaded_json_path or "")


class QtCutEditorDialog(QDialog, _QtViewportMixin):
    def __init__(
        self,
        base_mask: np.ndarray,
        junction_points_xy: Sequence[Sequence[int]],
        endpoint_points_xy: Sequence[Sequence[int]],
        left_title: str,
        right_title: str,
        subtitle_text: str,
        max_width: int,
        max_height: int,
        initial_cut_width: int,
        initial_cuts: Optional[List[Dict[str, object]]],
        live_node_mode: str = "none",
        show_original_panel: bool = True,
        allow_back: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.base_mask = (np.asarray(base_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        self.base_image = _ensure_bgr(self.base_mask)
        self.edited_mask = self.base_mask.copy()
        self.cuts: List[Dict[str, object]] = []
        self.drag_start: Optional[Point] = None
        self.drag_end: Optional[Point] = None
        self.drag_start_widget: Optional[QPoint] = None
        self.hover_point: Optional[Point] = None
        self.pan_anchor_widget: Optional[QPoint] = None
        self.pan_anchor_center: Optional[Tuple[float, float]] = None
        self.back_requested = False
        self.allow_back = bool(allow_back)
        self.live_node_mode = str(live_node_mode or "none")
        self.show_original_panel = bool(show_original_panel)
        self.junction_points = [(int(pt[0]), int(pt[1])) for pt in junction_points_xy if len(pt) >= 2]
        self.endpoint_points = [(int(pt[0]), int(pt[1])) for pt in endpoint_points_xy if len(pt) >= 2]
        self.junction_boxes = _cluster_point_boxes(self.junction_points, self.base_mask.shape[:2], pad=3)
        self.endpoint_boxes = _cluster_point_boxes(self.endpoint_points, self.base_mask.shape[:2], pad=2)
        self.live_junction_boxes = list(self.junction_boxes)
        self.live_endpoint_boxes = list(self.endpoint_boxes)
        self.live_junction_count = int(len(self.junction_points))
        self.live_endpoint_count = int(len(self.endpoint_points))
        self._edited_version = 0
        self._live_node_version = -1
        self._left_cache_key = None
        self._left_cache_img = None
        self._right_cache_key = None
        self._right_cache_img = None
        panel_count = 2 if self.show_original_panel else 1
        self._init_viewport_state(self.base_mask.shape[:2], max_width // panel_count, max_height - 220)

        self.left_title_text = str(left_title)
        self.right_title_text = str(right_title)
        self.left_title_label = QLabel(self.left_title_text if self.show_original_panel else self.right_title_text)
        self.right_title_label = QLabel(self.right_title_text)
        self.subtitle_label = QLabel(subtitle_text)
        self.left_label = InteractiveImageLabel()
        self.right_label = QLabel()
        self.cut_slider = QSlider(Qt.Horizontal)
        self.cut_slider.setRange(1, 25)
        self.cut_slider.setValue(int(max(1, initial_cut_width)))
        self.cut_value_label = QLabel(str(int(self.cut_slider.value())))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(10)
        self.pan_x_slider = QSlider(Qt.Horizontal)
        self.pan_x_slider.setRange(0, 1000)
        self.pan_y_slider = QSlider(Qt.Horizontal)
        self.pan_y_slider.setRange(0, 1000)
        self.reset_button = QPushButton("Clear Cuts")
        self.accept_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        self.back_button = QPushButton("Back to Split")
        self.back_button.setVisible(self.allow_back)

        self.left_label.on_left_press = self._on_left_press
        self.left_label.on_left_release = self._on_left_release
        self.left_label.on_move = self._on_move
        self.left_label.on_right_press = self._on_right_press
        self.left_label.on_middle_press = self._on_middle_press
        self.left_label.on_middle_release = self._on_middle_release
        self.left_label.on_wheel = self._on_wheel

        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.pan_x_slider.valueChanged.connect(self._on_pan_changed)
        self.pan_y_slider.valueChanged.connect(self._on_pan_changed)
        self.cut_slider.valueChanged.connect(self._on_cut_width_changed)
        self.reset_button.clicked.connect(self._reset_cuts)
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.back_button.clicked.connect(self._request_back)

        layout = QVBoxLayout(self)
        layout.addWidget(self.subtitle_label)
        top = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(self.left_title_label)
        left_box.addWidget(self.left_label)
        top.addLayout(left_box)
        if self.show_original_panel:
            right_box = QVBoxLayout()
            right_box.addWidget(self.right_title_label)
            right_box.addWidget(self.right_label)
            top.addLayout(right_box)
        layout.addLayout(top)
        controls = QFormLayout()
        cut_row = QHBoxLayout()
        cut_row.addWidget(self.cut_slider, 1)
        cut_row.addWidget(self.cut_value_label, 0)
        controls.addRow("CutWidth", cut_row)
        controls.addRow("Zoom x10", self.zoom_slider)
        controls.addRow("Pan X", self.pan_x_slider)
        controls.addRow("Pan Y", self.pan_y_slider)
        layout.addLayout(controls)
        buttons = QHBoxLayout()
        if self.allow_back:
            buttons.addWidget(self.back_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        for cut in self._sanitize_cuts(initial_cuts):
            self._append_cut(cut)
        self.refresh_views()

    @staticmethod
    def _sanitize_cuts(cuts: Optional[List[Dict[str, object]]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for cut in cuts or []:
            if not isinstance(cut, dict):
                continue
            start = cut.get("start")
            end = cut.get("end")
            if not isinstance(start, (list, tuple)) or len(start) < 2:
                continue
            if not isinstance(end, (list, tuple)) or len(end) < 2:
                end = start
            out.append(
                {
                    "kind": str(cut.get("kind", "line")),
                    "start": (int(start[0]), int(start[1])),
                    "end": (int(end[0]), int(end[1])),
                    "width": int(max(1, cut.get("width", 1))),
                }
            )
        return out

    @staticmethod
    def _point_to_segment_distance2(point: Point, start: Point, end: Point) -> float:
        px, py = float(point[0]), float(point[1])
        x1, y1 = float(start[0]), float(start[1])
        x2, y2 = float(end[0]), float(end[1])
        vx = x2 - x1
        vy = y2 - y1
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            dx = px - x1
            dy = py - y1
            return float(dx * dx + dy * dy)
        t = ((px - x1) * vx + (py - y1) * vy) / float(vx * vx + vy * vy)
        t = float(np.clip(t, 0.0, 1.0))
        qx = x1 + (t * vx)
        qy = y1 + (t * vy)
        dx = px - qx
        dy = py - qy
        return float(dx * dx + dy * dy)

    def _cut_distance2(self, cut: Dict[str, object], point: Point) -> float:
        start = tuple(map(int, cut["start"]))
        end = tuple(map(int, cut["end"]))
        return self._point_to_segment_distance2(point, start, end)

    def _apply_single_cut_to_mask(self, mask: np.ndarray, cut: Dict[str, object]) -> None:
        start = tuple(map(int, cut["start"]))
        end = tuple(map(int, cut["end"]))
        width = int(max(1, cut["width"]))
        if str(cut.get("kind", "line")) == "point" or start == end:
            x0, y0, x1, y1 = _square_bounds(start, width, mask.shape[:2])
            mask[y0:y1 + 1, x0:x1 + 1] = 0
        else:
            if width <= 1:
                for px, py in _raster_line_points(start, end):
                    if 0 <= int(px) < mask.shape[1] and 0 <= int(py) < mask.shape[0]:
                        mask[int(py), int(px)] = 0
            else:
                cv2.line(mask, start, end, 0, int(width), cv2.LINE_8)

    def _append_cut(self, cut: Dict[str, object]) -> None:
        normalized = self._sanitize_cuts([cut])
        if not normalized:
            return
        cut_norm = normalized[0]
        self.cuts.append(cut_norm)
        self._apply_single_cut_to_mask(self.edited_mask, cut_norm)
        self._edited_version += 1
        self._right_cache_key = None
        self._right_cache_img = None

    def _rebuild_edited_mask(self) -> None:
        self.edited_mask = self.base_mask.copy()
        for cut in self.cuts:
            self._apply_single_cut_to_mask(self.edited_mask, cut)
        self._edited_version += 1
        self._right_cache_key = None
        self._right_cache_img = None

    def _reset_cuts(self) -> None:
        self.cuts = []
        self.edited_mask = self.base_mask.copy()
        self._edited_version += 1
        self._right_cache_key = None
        self._right_cache_img = None
        self.refresh_views(sync_sliders=False)

    def _remove_nearest_cut(self, point: Optional[Point]) -> None:
        if not self.cuts:
            return
        if point is None:
            self.cuts.pop()
        else:
            best_idx = 0
            best_dist = self._cut_distance2(self.cuts[0], point)
            for idx in range(1, len(self.cuts)):
                dist = self._cut_distance2(self.cuts[idx], point)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            self.cuts.pop(int(best_idx))
        self._rebuild_edited_mask()
        self.refresh_views(sync_sliders=False)

    def _cut_width_display_thickness(self) -> int:
        return int(max(1, self.cut_slider.value()))

    def _ensure_live_node_overlay(self) -> None:
        if self.live_node_mode not in ("thin", "binary"):
            return
        if self._live_node_version == self._edited_version:
            return
        probe_mask = self.edited_mask
        if self.live_node_mode == "binary":
            probe_mask = _thin_binary_local(self.edited_mask)
        node_info = _node_points_xy_local(probe_mask)
        live_junction_points = [(int(pt[0]), int(pt[1])) for pt in node_info.get("junction_points_xy", []) if len(pt) >= 2]
        live_endpoint_points = [(int(pt[0]), int(pt[1])) for pt in node_info.get("endpoint_points_xy", []) if len(pt) >= 2]
        self.live_junction_boxes = _cluster_point_boxes(live_junction_points, self.base_mask.shape[:2], pad=3)
        self.live_endpoint_boxes = _cluster_point_boxes(live_endpoint_points, self.base_mask.shape[:2], pad=2)
        self.live_junction_count = int(node_info.get("junction_count", 0))
        self.live_endpoint_count = int(node_info.get("endpoint_count", 0))
        self._live_node_version = self._edited_version

    def _draw_live_boxes(self, canvas: np.ndarray) -> np.ndarray:
        self._ensure_live_node_overlay()
        out = canvas.copy()
        for x0, y0, x1, y1 in self.live_endpoint_boxes:
            p0 = self._image_to_display_point((x0, y0))
            p1 = self._image_to_display_point((x1, y1))
            if p0 is None or p1 is None:
                continue
            cv2.rectangle(out, p0, p1, (255, 180, 40), 1, cv2.LINE_8)
        for x0, y0, x1, y1 in self.live_junction_boxes:
            p0 = self._image_to_display_point((x0, y0))
            p1 = self._image_to_display_point((x1, y1))
            if p0 is None or p1 is None:
                continue
            cv2.rectangle(out, p0, p1, (40, 40, 255), 1, cv2.LINE_8)
        return out

    def _render_view_cached(self, source: np.ndarray, cache_name: str, version: int = 0) -> np.ndarray:
        x0, y0, x1, y1 = self._current_view_rect(source.shape[:2])
        key = (int(x0), int(y0), int(x1), int(y1), int(version))
        cache_key_attr = f"_{cache_name}_cache_key"
        cache_img_attr = f"_{cache_name}_cache_img"
        cached_key = getattr(self, cache_key_attr)
        cached_img = getattr(self, cache_img_attr)
        if cached_key == key and cached_img is not None:
            disp = cached_img.copy()
        else:
            crop = np.asarray(source)[y0:y1, x0:x1]
            disp = _fit_image_to_box(crop, max_w=self.max_panel_w, max_h=self.max_panel_h, allow_upscale=True)
            setattr(self, cache_key_attr, key)
            setattr(self, cache_img_attr, disp.copy())
        self._view_rect_xyxy = (int(x0), int(y0), int(x1), int(y1))
        self._display_shape = (int(disp.shape[0]), int(disp.shape[1]))
        return disp

    def _draw_boxes(self, canvas: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        for x0, y0, x1, y1 in self.endpoint_boxes:
            p0 = self._image_to_display_point((x0, y0))
            p1 = self._image_to_display_point((x1, y1))
            if p0 is None or p1 is None:
                continue
            cv2.rectangle(out, p0, p1, (255, 180, 40), 1, cv2.LINE_8)
        for x0, y0, x1, y1 in self.junction_boxes:
            p0 = self._image_to_display_point((x0, y0))
            p1 = self._image_to_display_point((x1, y1))
            if p0 is None or p1 is None:
                continue
            cv2.rectangle(out, p0, p1, (40, 40, 255), 1, cv2.LINE_8)
        return out

    def _draw_cuts(self, canvas: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        for cut in self.cuts:
            p0 = self._image_to_display_point(tuple(cut["start"]))
            p1 = self._image_to_display_point(tuple(cut["end"]))
            if p0 is None or p1 is None:
                continue
            thickness = int(max(1, round(self._cut_width_display_thickness())))
            if str(cut.get("kind", "line")) == "point" or tuple(cut["start"]) == tuple(cut["end"]):
                x0, y0, x1, y1 = _square_bounds(tuple(cut["start"]), int(max(1, cut["width"])), self.base_mask.shape[:2])
                q0 = self._image_to_display_point((x0, y0))
                q1 = self._image_to_display_point((x1, y1))
                if q0 is not None and q1 is not None:
                    cv2.rectangle(out, q0, q1, (0, 0, 255), 1, cv2.LINE_8)
            else:
                cv2.line(out, p0, p1, (0, 0, 255), thickness, cv2.LINE_8)
        if self.drag_start is not None and self.drag_end is not None:
            p0 = self._image_to_display_point(self.drag_start)
            p1 = self._image_to_display_point(self.drag_end)
            if p0 is not None and p1 is not None:
                thickness = int(max(1, round(self._cut_width_display_thickness())))
                if self.drag_start == self.drag_end:
                    x0, y0, x1, y1 = _square_bounds(self.drag_start, int(max(1, self.cut_slider.value())), self.base_mask.shape[:2])
                    q0 = self._image_to_display_point((x0, y0))
                    q1 = self._image_to_display_point((x1, y1))
                    if q0 is not None and q1 is not None:
                        cv2.rectangle(out, q0, q1, (0, 180, 255), 1, cv2.LINE_8)
                else:
                    cv2.line(out, p0, p1, (0, 180, 255), thickness, cv2.LINE_8)
        return out

    def _draw_hover_footprint(self, canvas: np.ndarray) -> np.ndarray:
        if self.hover_point is None or self.drag_start is not None:
            return canvas
        x0, y0, x1, y1 = _square_bounds(self.hover_point, int(max(1, self.cut_slider.value())), self.base_mask.shape[:2])
        q0 = self._image_to_display_point((x0, y0))
        q1 = self._image_to_display_point((x1, y1))
        if q0 is None or q1 is None:
            return canvas
        out = canvas.copy()
        cv2.rectangle(out, q0, q1, (0, 255, 255), 1, cv2.LINE_8)
        return out

    def _sync_sliders(self) -> None:
        if not self._trackbars_sync_enabled:
            return
        base_h, base_w = self.base_mask.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(self.base_mask.shape[:2])
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        self._trackbars_sync_enabled = False
        self.zoom_slider.setValue(int(round(self.zoom * 10.0)))
        self.pan_x_slider.setValue(0 if max_cx <= min_cx + 1e-6 else int(round((cx - min_cx) * 1000.0 / float(max_cx - min_cx))))
        self.pan_y_slider.setValue(0 if max_cy <= min_cy + 1e-6 else int(round((cy - min_cy) * 1000.0 / float(max_cy - min_cy))))
        self._trackbars_sync_enabled = True

    def _on_zoom_changed(self, value: int) -> None:
        self.zoom = float(max(10, int(value))) / 10.0
        self.refresh_views(sync_sliders=True)

    def _on_cut_width_changed(self, value: int) -> None:
        self.cut_value_label.setText(str(int(max(1, value))))
        if self.drag_start is not None:
            self.refresh_views(sync_sliders=False)

    def _on_pan_changed(self, _value: int) -> None:
        base_h, base_w = self.base_mask.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        px = float(self.pan_x_slider.value())
        py = float(self.pan_y_slider.value())
        if max_cx <= min_cx + 1e-6:
            cx = min_cx
        else:
            cx = min_cx + (px * float(max_cx - min_cx) / 1000.0)
        if max_cy <= min_cy + 1e-6:
            cy = min_cy
        else:
            cy = min_cy + (py * float(max_cy - min_cy) / 1000.0)
        self.view_center = (float(cx), float(cy))
        self.refresh_views(sync_sliders=False)

    def _pan_from_drag(self, start_widget: QPoint, now_widget: QPoint) -> None:
        if self.pan_anchor_center is None:
            return
        x0, y0, x1, y1 = self._view_rect_xyxy
        disp_h, disp_w = [int(v) for v in self._display_shape]
        view_w = max(1.0, float(x1 - x0))
        view_h = max(1.0, float(y1 - y0))
        scale_x = view_w / float(max(1, disp_w))
        scale_y = view_h / float(max(1, disp_h))
        dx = float(now_widget.x() - start_widget.x())
        dy = float(now_widget.y() - start_widget.y())
        self.view_center = (
            float(self.pan_anchor_center[0] - (dx * scale_x)),
            float(self.pan_anchor_center[1] - (dy * scale_y)),
        )
        self.refresh_views(sync_sliders=True)

    def _on_left_press(self, event: QMouseEvent) -> None:
        mapped = self._display_to_image_point(event.pos(), self.base_mask.shape[:2])
        self.drag_start = mapped
        self.drag_end = mapped
        self.drag_start_widget = QPoint(event.pos())
        self.hover_point = mapped

    def _on_left_release(self, event: QMouseEvent) -> None:
        mapped = self._display_to_image_point(event.pos(), self.base_mask.shape[:2])
        if self.drag_start is not None and mapped is not None:
            start = tuple(map(int, self.drag_start))
            end = tuple(map(int, mapped))
            width = int(max(1, self.cut_slider.value()))
            widget_dx = 0 if self.drag_start_widget is None else abs(int(self.drag_start_widget.x()) - int(event.pos().x()))
            widget_dy = 0 if self.drag_start_widget is None else abs(int(self.drag_start_widget.y()) - int(event.pos().y()))
            # Decide click vs drag in display space so tiny jitter still counts as a point erase.
            if max(widget_dx, widget_dy) >= 6:
                self._append_cut({"kind": "line", "start": start, "end": end, "width": width})
            else:
                self._append_cut({"kind": "point", "start": start, "end": start, "width": width})
        self.drag_start = None
        self.drag_end = None
        self.drag_start_widget = None
        self.hover_point = mapped
        self.refresh_views(sync_sliders=False)

    def _on_move(self, event: QMouseEvent) -> None:
        if self.pan_anchor_widget is not None:
            self._pan_from_drag(self.pan_anchor_widget, event.pos())
            return
        mapped = self._display_to_image_point(event.pos(), self.base_mask.shape[:2])
        if self.hover_point != mapped:
            self.hover_point = mapped
        if self.drag_start is not None:
            self.drag_end = mapped
            self.refresh_views(sync_sliders=False)
        else:
            self.refresh_views(sync_sliders=False)

    def _on_right_press(self, event: QMouseEvent) -> None:
        mapped = self._display_to_image_point(event.pos(), self.base_mask.shape[:2])
        self._remove_nearest_cut(mapped)

    def _on_middle_press(self, event: QMouseEvent) -> None:
        self.pan_anchor_widget = QPoint(event.pos())
        self.pan_anchor_center = (float(self.view_center[0]), float(self.view_center[1]))

    def _on_middle_release(self, _event: QMouseEvent) -> None:
        self.pan_anchor_widget = None
        self.pan_anchor_center = None

    def _on_wheel(self, event: QWheelEvent) -> None:
        delta = int(event.angleDelta().y())
        if delta == 0:
            return
        mapped = self._display_to_image_point(event.pos(), self.base_mask.shape[:2])
        if mapped is not None:
            self.view_center = (float(mapped[0]), float(mapped[1]))
        self.zoom = float(np.clip((self.zoom * 10.0) + (1 if delta > 0 else -1), 10.0, 200.0)) / 10.0
        self.refresh_views(sync_sliders=True)

    def _request_back(self) -> None:
        self.back_requested = True
        self.accept()

    def refresh_views(self, sync_sliders: bool = True) -> None:
        if self.show_original_panel:
            left = self._render_view_cached(self.base_image, "left", version=0)
            left = self._draw_boxes(left)
            left = self._draw_cuts(left)
            left = self._draw_hover_footprint(left)
            _set_label_image(self.left_label, left)
            right = self._render_view_cached(self.edited_mask, "right", version=self._edited_version)
            if self.live_node_mode in ("thin", "binary"):
                right = self._draw_live_boxes(right)
                mode_label = "thin" if self.live_node_mode == "thin" else "thin-preview"
                self.right_title_label.setText(
                    f"{self.right_title_text} | live {mode_label} E:{int(self.live_endpoint_count)} J:{int(self.live_junction_count)}"
                )
            else:
                self.right_title_label.setText(self.right_title_text)
            _set_label_image(self.right_label, right)
        else:
            work = self._render_view_cached(self.edited_mask, "right", version=self._edited_version)
            if self.live_node_mode in ("thin", "binary"):
                work = self._draw_live_boxes(work)
                mode_label = "thin" if self.live_node_mode == "thin" else "thin-preview"
                self.left_title_label.setText(
                    f"{self.right_title_text} | live {mode_label} E:{int(self.live_endpoint_count)} J:{int(self.live_junction_count)}"
                )
            else:
                self.left_title_label.setText(self.right_title_text)
            work = self._draw_cuts(work)
            work = self._draw_hover_footprint(work)
            _set_label_image(self.left_label, work)
        if sync_sliders:
            self._sync_sliders()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        panel_count = 2 if self.show_original_panel else 1
        self.max_panel_w = int(max(160, (self.width() - 80) // panel_count))
        self.max_panel_h = int(max(160, self.height() - 220))
        self._left_cache_key = None
        self._left_cache_img = None
        self._right_cache_key = None
        self._right_cache_img = None
        self.refresh_views(sync_sliders=False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
            return
        if key in (Qt.Key_Escape,):
            self.reject()
            return
        if key in (Qt.Key_C, Qt.Key_R):
            self._reset_cuts()
            return
        if self.allow_back and key == Qt.Key_B:
            self._request_back()
            return
        super().keyPressEvent(event)

    def result_payload(self) -> Optional[Dict[str, object]]:
        if self.result() != QDialog.Accepted:
            return None
        action = "back" if self.back_requested else "accept"
        return {
            "action": action,
            "edited_mask": np.asarray(self.edited_mask, dtype=np.uint8),
            "cut_count": int(len(self.cuts)),
            "cuts": [
                {
                    "kind": str(cut.get("kind", "line")),
                    "start": [int(cut["start"][0]), int(cut["start"][1])],
                    "end": [int(cut["end"][0]), int(cut["end"][1])],
                    "width": int(cut["width"]),
                }
                for cut in self.cuts
            ],
        }


def _format_float_or_na(value: Optional[float], fmt: str = ".4g") -> str:
    if value is None:
        return "n/a"
    try:
        val = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(val):
        return "n/a"
    return format(val, fmt)


def _finite_or_zero(value: object) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return float(val)


def _scale_px_values_to_mas(values_px: object, scale_mas_per_px: object) -> np.ndarray:
    values = np.asarray(values_px, dtype=np.float64)
    scale = _safe_float(scale_mas_per_px)
    if np.isfinite(scale) and scale > 0.0:
        return (values * float(scale)).astype(np.float64)
    return np.full(values.shape, np.nan, dtype=np.float64)


def _mas_axis_values_from_measurement(
    values_px: object,
    values_mas: object,
    scale_mas_per_px: object,
) -> np.ndarray:
    px_values = np.asarray(values_px, dtype=np.float64)
    mas_values = np.asarray(values_mas, dtype=np.float64)
    if mas_values.shape == px_values.shape and np.any(np.isfinite(mas_values)):
        return mas_values.astype(np.float64)
    return _scale_px_values_to_mas(px_values, scale_mas_per_px)


def infer_mojave_polar_core_tail(
    flux_map: np.ndarray,
    support_mask: np.ndarray,
    core_xy: Optional[Point] = None,
    tail_xy: Optional[Point] = None,
) -> Dict[str, object]:
    flux = np.asarray(flux_map, dtype=np.float32)
    support = (np.asarray(support_mask, dtype=np.uint8) > 0) & np.isfinite(flux)
    if flux.ndim != 2:
        raise ValueError("Flux map must be a 2D array.")
    if support.shape[:2] != flux.shape[:2]:
        raise ValueError("Support mask shape must match flux map shape.")
    if not np.any(support):
        raise ValueError("No valid support region for automatic MOJAVE polar picks.")

    if core_xy is None:
        ys, xs = np.where(support)
        values = flux[ys, xs]
        idx = int(np.nanargmax(values))
        core = (int(xs[idx]), int(ys[idx]))
        core_source = "flux_peak"
    else:
        core = (int(core_xy[0]), int(core_xy[1]))
        core_source = "provided"

    if tail_xy is None:
        h, w = flux.shape[:2]
        cx = int(np.clip(core[0], 0, w - 1))
        cy = int(np.clip(core[1], 0, h - 1))
        comp_mask = support
        comp_count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
        if int(comp_count) > 1:
            label_at_core = int(labels[cy, cx])
            if label_at_core <= 0:
                ys_all, xs_all = np.where(support)
                dist2_all = np.square(xs_all.astype(np.float64) - float(core[0])) + np.square(ys_all.astype(np.float64) - float(core[1]))
                nearest = int(np.argmin(dist2_all))
                label_at_core = int(labels[int(ys_all[nearest]), int(xs_all[nearest])])
            if label_at_core > 0:
                comp_mask = labels == label_at_core
        ys, xs = np.where(comp_mask)
        if xs.size <= 1:
            raise ValueError("Automatic MOJAVE polar tail pick needs at least two support pixels.")
        dist2 = np.square(xs.astype(np.float64) - float(core[0])) + np.square(ys.astype(np.float64) - float(core[1]))
        idx = int(np.argmax(dist2))
        tail = (int(xs[idx]), int(ys[idx]))
        tail_source = "farthest_support"
        tail_distance_px = float(np.sqrt(float(dist2[idx])))
    else:
        tail = (int(tail_xy[0]), int(tail_xy[1]))
        tail_source = "provided"
        tail_distance_px = float(np.hypot(float(tail[0] - core[0]), float(tail[1] - core[1])))

    if tail_distance_px <= 1e-6:
        raise ValueError("Automatic MOJAVE polar core and tail picks are not separated.")
    return {
        "core_xy": core,
        "tail_xy": tail,
        "core_source": core_source,
        "tail_source": tail_source,
        "tail_distance_px": float(tail_distance_px),
        "core_flux": float(flux[core[1], core[0]]) if 0 <= core[1] < flux.shape[0] and 0 <= core[0] < flux.shape[1] else float("nan"),
    }


def build_flux_zero_support_mask(
    flux_map: np.ndarray,
    roi_mask: np.ndarray,
    region_depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    flux = np.asarray(flux_map, dtype=np.float32)
    roi = np.asarray(roi_mask, dtype=np.uint8) > 0
    valid = roi & np.isfinite(flux)
    finite_values = flux[valid]
    positive_values = finite_values[finite_values > 0.0]
    if positive_values.size > 0:
        eps = float(max(1e-12, 1e-7 * float(np.nanmax(positive_values))))
    else:
        eps = 1e-12

    positive = valid & (flux > eps)
    if region_depth_map is not None:
        depth = np.asarray(region_depth_map, dtype=np.int32)
        if depth.shape[:2] == flux.shape[:2]:
            positive |= valid & (depth > 0)
    if not np.any(positive):
        return np.zeros(flux.shape[:2], dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    adjacent_to_positive = cv2.dilate(positive.astype(np.uint8), kernel, iterations=1) > 0
    zero_boundary = valid & (np.abs(flux) <= eps) & adjacent_to_positive
    return (positive | zero_boundary).astype(np.uint8)


def _calibration_summary_text(calibration_context: Optional[Dict[str, object]]) -> str:
    ctx = dict(calibration_context or {})
    scale = float(ctx.get("scale_mas_per_px", float("nan")))
    beam_major = float(ctx.get("beam_major_mas", float("nan")))
    beam_minor = float(ctx.get("beam_minor_mas", float("nan")))
    if np.isfinite(scale) and scale > 0.0:
        scale_txt = f"{scale:.6g} mas/px"
    else:
        scale_txt = "not set"
    if np.isfinite(beam_major) and beam_major > 0.0 and np.isfinite(beam_minor) and beam_minor > 0.0:
        beam_txt = f"{beam_major:.4g} x {beam_minor:.4g} mas"
    else:
        beam_txt = "not set"
    return f"Scale: {scale_txt} | Beam: {beam_txt}"


class QtScaleBeamDialog(QDialog, _QtViewportMixin):
    def __init__(
        self,
        roi_image: np.ndarray,
        roi_mask: Optional[np.ndarray],
        initial_state: Optional[Dict[str, object]],
        max_width: int,
        max_height: int,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Scale / Beam Calibration")
        base = _ensure_bgr(roi_image)
        mask = None if roi_mask is None else (np.asarray(roi_mask, dtype=np.uint8) > 0)
        if mask is not None and mask.shape[:2] == base.shape[:2]:
            dimmed = base.copy()
            dimmed[~mask] = np.clip(dimmed[~mask].astype(np.float32) * 0.35, 0.0, 255.0).astype(np.uint8)
            self.base_image = dimmed
        else:
            self.base_image = base
        self.scale_points: List[Point] = []
        self.beam_major_points: List[Point] = []
        self.beam_minor_points: List[Point] = []
        self.hover_point: Optional[Point] = None
        self.drag_start: Optional[Point] = None
        self.drag_end: Optional[Point] = None
        self._init_viewport_state(self.base_image.shape[:2], max_width, max_height - 220)
        self.label = InteractiveImageLabel()
        self.info_label = QLabel(
            "Select a measurement mode, then drag a line on the image. "
            "Scale Bar: drag along the scale bar, enter its length, then press 'Use Line -> Scale'. "
            "Beam Major / Beam Minor: drag directly along the beam axes. You can still type Scale / Beam values directly."
        )
        self.status_label = QLabel("")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(10)
        self.pan_x_slider = QSlider(Qt.Horizontal)
        self.pan_x_slider.setRange(0, 1000)
        self.pan_y_slider = QSlider(Qt.Horizontal)
        self.pan_y_slider.setRange(0, 1000)
        self.bar_length_spin = QDoubleSpinBox()
        self.bar_length_spin.setRange(0.0, 1_000_000_000.0)
        self.bar_length_spin.setDecimals(6)
        self.measure_mode_combo = QComboBox()
        self.measure_mode_combo.addItem("Scale Bar", "scale")
        self.measure_mode_combo.addItem("Beam Major", "beam_major")
        self.measure_mode_combo.addItem("Beam Minor", "beam_minor")
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1_000_000_000.0)
        self.scale_spin.setDecimals(9)
        self.beam_major_spin = QDoubleSpinBox()
        self.beam_major_spin.setRange(0.0, 1_000_000_000.0)
        self.beam_major_spin.setDecimals(6)
        self.beam_minor_spin = QDoubleSpinBox()
        self.beam_minor_spin.setRange(0.0, 1_000_000_000.0)
        self.beam_minor_spin.setDecimals(6)
        self.use_line_button = QPushButton("Use Line -> Scale")
        self.clear_line_button = QPushButton("Clear Line")
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")

        init = dict(initial_state or {})
        pts = [
            (int(pt[0]), int(pt[1]))
            for pt in list(init.get("scale_points_xy", []))
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        self.scale_points = pts[:2]
        beam_major_pts = [
            (int(pt[0]), int(pt[1]))
            for pt in list(init.get("beam_major_points_xy", []))
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        beam_minor_pts = [
            (int(pt[0]), int(pt[1]))
            for pt in list(init.get("beam_minor_points_xy", []))
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        self.beam_major_points = beam_major_pts[:2]
        self.beam_minor_points = beam_minor_pts[:2]
        self.bar_length_spin.setValue(_finite_or_zero(init.get("scale_bar_length_mas", 0.0)))
        self.scale_spin.setValue(_finite_or_zero(init.get("scale_mas_per_px", 0.0)))
        self.beam_major_spin.setValue(_finite_or_zero(init.get("beam_major_mas", 0.0)))
        self.beam_minor_spin.setValue(_finite_or_zero(init.get("beam_minor_mas", 0.0)))

        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(False)
        image_scroll.setWidget(self.label)
        image_scroll.setMinimumHeight(320)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(image_scroll, stretch=1)
        controls = QFormLayout()
        controls.addRow("Zoom x10", self.zoom_slider)
        controls.addRow("Pan X", self.pan_x_slider)
        controls.addRow("Pan Y", self.pan_y_slider)
        controls.addRow("Measure Mode", self.measure_mode_combo)
        controls.addRow("Scale Bar Length (mas)", self.bar_length_spin)
        controls.addRow("Scale (mas/px)", self.scale_spin)
        controls.addRow("Beam Major (mas)", self.beam_major_spin)
        controls.addRow("Beam Minor (mas)", self.beam_minor_spin)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        button_row = QHBoxLayout()
        button_row.addWidget(self.use_line_button)
        button_row.addWidget(self.clear_line_button)
        button_row.addStretch(1)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            max_w = max(720, int(geom.width()) - 64)
            max_h = max(640, int(geom.height()) - 64)
            self.setMaximumSize(max_w, max_h)
            target_w = int(min(max_w, max(900, int(max_width) + 220)))
            target_h = int(min(max_h, max(760, int(max_height) + 260)))
            self.resize(target_w, target_h)

        self.label.on_left_press = self._on_left_press
        self.label.on_left_release = self._on_left_release
        self.label.on_move = self._on_move
        self.label.on_right_press = self._on_right_press
        self.label.on_wheel = self._on_wheel
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.pan_x_slider.valueChanged.connect(self._on_pan_changed)
        self.pan_y_slider.valueChanged.connect(self._on_pan_changed)
        self.measure_mode_combo.currentIndexChanged.connect(self.refresh_view)
        self.use_line_button.clicked.connect(self._use_line_scale)
        self.clear_line_button.clicked.connect(self._clear_points)
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.bar_length_spin.valueChanged.connect(self.refresh_view)
        self.scale_spin.valueChanged.connect(self.refresh_view)
        self.beam_major_spin.valueChanged.connect(self.refresh_view)
        self.beam_minor_spin.valueChanged.connect(self.refresh_view)
        self.refresh_view()

    def _on_zoom_changed(self, value: int) -> None:
        self.zoom = float(max(10, int(value))) / 10.0
        self.refresh_view(sync_sliders=True)

    def _on_pan_changed(self, _value: int) -> None:
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        px = float(self.pan_x_slider.value())
        py = float(self.pan_y_slider.value())
        cx = min_cx if max_cx <= min_cx + 1e-6 else min_cx + (px * float(max_cx - min_cx) / 1000.0)
        cy = min_cy if max_cy <= min_cy + 1e-6 else min_cy + (py * float(max_cy - min_cy) / 1000.0)
        self.view_center = (float(cx), float(cy))
        self.refresh_view(sync_sliders=False)

    def _sync_sliders(self) -> None:
        if not self._trackbars_sync_enabled:
            return
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(self.base_image.shape[:2])
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        self._trackbars_sync_enabled = False
        self.zoom_slider.setValue(int(round(self.zoom * 10.0)))
        self.pan_x_slider.setValue(0 if max_cx <= min_cx + 1e-6 else int(round((cx - min_cx) * 1000.0 / float(max_cx - min_cx))))
        self.pan_y_slider.setValue(0 if max_cy <= min_cy + 1e-6 else int(round((cy - min_cy) * 1000.0 / float(max_cy - min_cy))))
        self._trackbars_sync_enabled = True

    def _replace_or_append_point(self, pt: Point) -> None:
        self.scale_points = [(int(pt[0]), int(pt[1]))]

    def _remove_nearest_point(self, pt: Optional[Point]) -> None:
        mode = self._current_measure_mode()
        if mode == "beam_major":
            self.beam_major_points = []
        elif mode == "beam_minor":
            self.beam_minor_points = []
        else:
            self.scale_points = []

    @staticmethod
    def _line_distance_px_from(points: Sequence[Point]) -> float:
        if len(points) < 2:
            return float("nan")
        dx = float(points[1][0] - points[0][0])
        dy = float(points[1][1] - points[0][1])
        return float(np.hypot(dx, dy))

    def _line_distance_px(self) -> float:
        return self._line_distance_px_from(self.scale_points)

    def _beam_major_px(self) -> float:
        return self._line_distance_px_from(self.beam_major_points)

    def _beam_minor_px(self) -> float:
        return self._line_distance_px_from(self.beam_minor_points)

    def _current_measure_mode(self) -> str:
        data = self.measure_mode_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return "scale"

    def _set_active_line(self, p0: Point, p1: Point) -> None:
        mode = self._current_measure_mode()
        line = [(int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))]
        if mode == "beam_major":
            self.beam_major_points = line
        elif mode == "beam_minor":
            self.beam_minor_points = line
        else:
            self.scale_points = line

    def _recompute_beam_values_from_lines(self) -> None:
        scale = float(self.scale_spin.value())
        if not np.isfinite(scale) or scale <= 0.0:
            return
        beam_major_px = self._beam_major_px()
        beam_minor_px = self._beam_minor_px()
        if np.isfinite(beam_major_px) and beam_major_px > 0.0:
            self.beam_major_spin.blockSignals(True)
            self.beam_major_spin.setValue(float(beam_major_px * scale))
            self.beam_major_spin.blockSignals(False)
        if np.isfinite(beam_minor_px) and beam_minor_px > 0.0:
            self.beam_minor_spin.blockSignals(True)
            self.beam_minor_spin.setValue(float(beam_minor_px * scale))
            self.beam_minor_spin.blockSignals(False)

    def _computed_scale_mas_per_px(self) -> float:
        dist_px = self._line_distance_px()
        bar_length = float(self.bar_length_spin.value())
        if (not np.isfinite(dist_px)) or dist_px <= 1e-9 or bar_length <= 0.0:
            return float("nan")
        return float(bar_length / dist_px)

    def _use_line_scale(self) -> None:
        scale = self._computed_scale_mas_per_px()
        if np.isfinite(scale) and scale > 0.0:
            self.scale_spin.setValue(scale)
            self._recompute_beam_values_from_lines()
        self.refresh_view(sync_sliders=False)

    def _clear_points(self) -> None:
        mode = self._current_measure_mode()
        if mode == "beam_major":
            self.beam_major_points = []
        elif mode == "beam_minor":
            self.beam_minor_points = []
        else:
            self.scale_points = []
        self.drag_start = None
        self.drag_end = None
        self.refresh_view(sync_sliders=False)

    def _on_left_press(self, event: QMouseEvent) -> None:
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if pt is None:
            return
        self.drag_start = pt
        self.drag_end = pt
        self.hover_point = pt
        self.refresh_view(sync_sliders=False)

    def _on_left_release(self, event: QMouseEvent) -> None:
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if pt is not None and self.drag_start is not None:
            self._set_active_line(self.drag_start, pt)
            self.drag_end = pt
            self._recompute_beam_values_from_lines()
        self.drag_start = None
        self.drag_end = None
        self.refresh_view(sync_sliders=False)

    def _on_move(self, event: QMouseEvent) -> None:
        self.hover_point = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if self.drag_start is not None and self.hover_point is not None:
            self.drag_end = self.hover_point
        self.refresh_view(sync_sliders=False)

    def _on_right_press(self, event: QMouseEvent) -> None:
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        self._remove_nearest_point(pt)
        self.refresh_view(sync_sliders=False)

    def _on_wheel(self, event: QWheelEvent) -> None:
        delta = int(event.angleDelta().y())
        if delta == 0:
            return
        pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if pt is not None:
            self.view_center = (float(pt[0]), float(pt[1]))
        self.zoom = float(np.clip((self.zoom * 10.0) + (1 if delta > 0 else -1), 10.0, 200.0)) / 10.0
        self.refresh_view(sync_sliders=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.max_panel_w = int(max(160, self.width() - 80))
        self.max_panel_h = int(max(160, self.height() - 240))
        self.refresh_view(sync_sliders=False)

    def refresh_view(self, *_args, sync_sliders: bool = True) -> None:
        self._recompute_beam_values_from_lines()
        canvas = self.base_image.copy()
        if len(self.scale_points) >= 2:
            cv2.line(canvas, self.scale_points[0], self.scale_points[1], (0, 255, 255), 2, cv2.LINE_AA)
        if len(self.beam_major_points) >= 2:
            cv2.line(canvas, self.beam_major_points[0], self.beam_major_points[1], (0, 255, 0), 2, cv2.LINE_AA)
        if len(self.beam_minor_points) >= 2:
            cv2.line(canvas, self.beam_minor_points[0], self.beam_minor_points[1], (255, 200, 0), 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_end is not None:
            mode = self._current_measure_mode()
            color = (0, 255, 255)
            if mode == "beam_major":
                color = (0, 255, 0)
            elif mode == "beam_minor":
                color = (255, 200, 0)
            cv2.line(canvas, self.drag_start, self.drag_end, color, 1, cv2.LINE_AA)
        if len(self.scale_points) >= 2:
            pass
        for idx, pt in enumerate(self.scale_points):
            color = (0, 255, 255) if idx == 0 else (0, 200, 255)
            cv2.circle(canvas, tuple(map(int, pt)), 4, color, -1, cv2.LINE_AA)
        for idx, pt in enumerate(self.beam_major_points):
            color = (0, 255, 0) if idx == 0 else (0, 180, 0)
            cv2.circle(canvas, tuple(map(int, pt)), 4, color, -1, cv2.LINE_AA)
        for idx, pt in enumerate(self.beam_minor_points):
            color = (255, 200, 0) if idx == 0 else (220, 150, 0)
            cv2.circle(canvas, tuple(map(int, pt)), 4, color, -1, cv2.LINE_AA)
        disp = self._render_zoom_view(canvas)
        _set_label_image(self.label, disp)
        dist_px = self._line_distance_px()
        comp_scale = self._computed_scale_mas_per_px()
        beam_major_px = self._beam_major_px()
        beam_minor_px = self._beam_minor_px()
        mode = self._current_measure_mode()
        live_len_px = float("nan")
        if self.drag_start is not None and self.drag_end is not None:
            live_len_px = self._line_distance_px_from([self.drag_start, self.drag_end])
        scale_val = float(self.scale_spin.value())
        beam_major_mas = beam_major_px * scale_val if np.isfinite(beam_major_px) and np.isfinite(scale_val) and scale_val > 0.0 else float("nan")
        beam_minor_mas = beam_minor_px * scale_val if np.isfinite(beam_minor_px) and np.isfinite(scale_val) and scale_val > 0.0 else float("nan")
        self.status_label.setText(
            f"Mode: {mode} | "
            f"Live px: {_format_float_or_na(live_len_px, '.3f')} | "
            f"Scale line px: {_format_float_or_na(dist_px, '.3f')} | "
            f"Computed scale: {_format_float_or_na(comp_scale, '.8g')} mas/px | "
            f"Chosen scale: {_format_float_or_na(self.scale_spin.value(), '.8g')} mas/px | "
            f"Beam major: {_format_float_or_na(beam_major_px, '.3f')} px / {_format_float_or_na(beam_major_mas, '.6g')} mas | "
            f"Beam minor: {_format_float_or_na(beam_minor_px, '.3f')} px / {_format_float_or_na(beam_minor_mas, '.6g')} mas"
        )
        if sync_sliders:
            self._sync_sliders()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
            return
        if key in (Qt.Key_C, Qt.Key_R):
            self._clear_points()
            return
        super().keyPressEvent(event)

    def get_calibration(self) -> Dict[str, object]:
        scale = float(self.scale_spin.value())
        beam_major = float(self.beam_major_spin.value())
        beam_minor = float(self.beam_minor_spin.value())
        out = {
            "scale_mas_per_px": float(scale) if scale > 0.0 else float("nan"),
            "beam_major_mas": float(beam_major) if beam_major > 0.0 else float("nan"),
            "beam_minor_mas": float(beam_minor) if beam_minor > 0.0 else float("nan"),
            "scale_points_xy": [[int(pt[0]), int(pt[1])] for pt in self.scale_points],
            "beam_major_points_xy": [[int(pt[0]), int(pt[1])] for pt in self.beam_major_points],
            "beam_minor_points_xy": [[int(pt[0]), int(pt[1])] for pt in self.beam_minor_points],
            "scale_bar_length_mas": float(self.bar_length_spin.value()),
            "scale_line_px": float(self._line_distance_px()),
            "beam_major_line_px": float(self._beam_major_px()),
            "beam_minor_line_px": float(self._beam_minor_px()),
        }
        out["beam_size_px"] = float(
            analysis_beam_size_px(
                out.get("scale_mas_per_px"),
                out.get("beam_major_mas"),
                out.get("beam_minor_mas"),
            )
        )
        return out


class QtSliceDetailsDialog(QDialog):
    def __init__(
        self,
        record: Dict[str, object],
        calibration_context: Optional[Dict[str, object]],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Slice Details")
        self.record = dict(record or {})
        self.calibration_context = dict(calibration_context or {})

        self.info_label = QLabel(self._summary_text())
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setStyleSheet(
            "QLabel { border: 1px solid #cfcfcf; padding: 6px; background: #fafafa; }"
        )

        self.fig = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.profile_ax = self.fig.add_subplot(211)
        self.residual_ax = self.fig.add_subplot(212, sharex=self.profile_ax)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(buttons)
        self.resize(900, 680)
        self._plot()

    @staticmethod
    def _point_text(value: object) -> str:
        try:
            if isinstance(value, np.ndarray):
                arr = value.tolist()
            else:
                arr = value
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return f"({_format_float_or_na(arr[0], '.4g')}, {_format_float_or_na(arr[1], '.4g')})"
        except Exception:
            pass
        return "n/a"

    def _profile_dict(self) -> Dict[str, object]:
        profile = self.record.get("profile", {})
        return dict(profile) if isinstance(profile, dict) else {}

    def _fit_dict(self) -> Dict[str, object]:
        fit = self.record.get("fit", {})
        return dict(fit) if isinstance(fit, dict) else {}

    def _params_dict(self) -> Dict[str, object]:
        params = self._fit_dict().get("params", {})
        return dict(params) if isinstance(params, dict) else {}

    def _fit_window_dict(self) -> Dict[str, object]:
        fit_window = self._fit_dict().get("fit_window", {})
        return dict(fit_window) if isinstance(fit_window, dict) else {}

    def _scan_xlim_mas(self, profile: Dict[str, object], scale_mas_per_px: object) -> Optional[Tuple[float, float]]:
        scale = _safe_float(scale_mas_per_px)
        if not (np.isfinite(scale) and scale > 0.0):
            return None
        half_width_px = _safe_float(profile.get("scan_half_width_px", self.record.get("scan_half_width_px", float("nan"))))
        if np.isfinite(half_width_px) and half_width_px > 0.0:
            half_width_mas = float(abs(half_width_px) * scale)
            return -half_width_mas, half_width_mas
        left_px = _safe_float(profile.get("scan_k_min_px", self.record.get("profile_scan_k_min_px", float("nan"))))
        right_px = _safe_float(profile.get("scan_k_max_px", self.record.get("profile_scan_k_max_px", float("nan"))))
        if not (np.isfinite(left_px) and np.isfinite(right_px) and right_px > left_px):
            return None
        return float(left_px * scale), float(right_px * scale)

    def _gaussian_plot_x_px(self, profile: Dict[str, object], fit_x: np.ndarray) -> np.ndarray:
        half_width_px = _safe_float(profile.get("scan_half_width_px", self.record.get("scan_half_width_px", float("nan"))))
        if np.isfinite(half_width_px) and half_width_px > 0.0:
            return np.linspace(-abs(float(half_width_px)), abs(float(half_width_px)), 600, dtype=np.float64)
        left_px = _safe_float(profile.get("scan_k_min_px", self.record.get("profile_scan_k_min_px", float("nan"))))
        right_px = _safe_float(profile.get("scan_k_max_px", self.record.get("profile_scan_k_max_px", float("nan"))))
        if np.isfinite(left_px) and np.isfinite(right_px) and right_px > left_px:
            return np.linspace(float(left_px), float(right_px), 600, dtype=np.float64)
        full_fit_x = np.asarray(self._fit_dict().get("full_x", []), dtype=np.float64)
        if full_fit_x.size > 0:
            return full_fit_x
        return np.asarray(fit_x, dtype=np.float64)

    @staticmethod
    def _gaussian_model_y(x: np.ndarray, params: Dict[str, object]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        baseline = _safe_float(params.get("baseline", float("nan")))
        amplitude = _safe_float(params.get("amplitude", float("nan")))
        mu = _safe_float(params.get("mu", float("nan")))
        sigma = _safe_float(params.get("sigma", float("nan")))
        if not (
            np.isfinite(baseline)
            and np.isfinite(amplitude)
            and np.isfinite(mu)
            and np.isfinite(sigma)
            and sigma > 0.0
        ):
            return np.full(x_arr.shape, np.nan, dtype=np.float64)
        return baseline + (amplitude * np.exp(-0.5 * np.square((x_arr - mu) / max(sigma, 1e-12))))

    def _mas_value(self, px_key: str, mas_key: str) -> float:
        mas_val = _safe_float(self.record.get(mas_key, float("nan")))
        if np.isfinite(mas_val):
            return float(mas_val)
        px_val = _safe_float(self.record.get(px_key, float("nan")))
        scale = _safe_float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        if np.isfinite(px_val) and np.isfinite(scale) and scale > 0.0:
            return float(px_val * scale)
        return float("nan")

    def _profile_xy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        profile = self._profile_dict()
        x = np.asarray(profile.get("valid_x", []), dtype=np.float64)
        y = np.asarray(profile.get("valid_y", []), dtype=np.float64)
        if x.size <= 0 or y.size <= 0 or x.size != y.size:
            fit = self._fit_dict()
            x = np.asarray(fit.get("x", []), dtype=np.float64)
            y = np.asarray(fit.get("y", []), dtype=np.float64)
        if x.size <= 0 or y.size <= 0 or x.size != y.size:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        xf = x[finite]
        yf = y[finite]
        order = np.argsort(xf)
        return xf[order], yf[order]

    @staticmethod
    def _interp_y_at_zero(x: np.ndarray, y: np.ndarray) -> float:
        if x.size <= 0 or y.size <= 0 or x.size != y.size:
            return float("nan")
        if float(np.min(x)) > 0.0 or float(np.max(x)) < 0.0:
            return float("nan")
        return float(np.interp(0.0, x, y))

    @staticmethod
    def _profile_level_crossing(
        x: np.ndarray,
        y: np.ndarray,
        *,
        center_y: float,
        level: float,
        side: str,
    ) -> float:
        if (
            x.size <= 0
            or y.size <= 0
            or x.size != y.size
            or (not np.isfinite(center_y))
            or (not np.isfinite(level))
        ):
            return float("nan")
        if side == "left":
            items = [(float(xi), float(yi)) for xi, yi in zip(x.tolist(), y.tolist()) if float(xi) < 0.0]
            items.sort(key=lambda item: item[0], reverse=True)
        else:
            items = [(float(xi), float(yi)) for xi, yi in zip(x.tolist(), y.tolist()) if float(xi) > 0.0]
            items.sort(key=lambda item: item[0])
        prev_x = 0.0
        prev_y = float(center_y)
        prev_delta = prev_y - float(level)
        for cur_x, cur_y in items:
            if not (np.isfinite(cur_x) and np.isfinite(cur_y)):
                continue
            cur_delta = cur_y - float(level)
            if abs(cur_delta) <= 1e-12:
                return float(cur_x)
            if prev_delta * cur_delta < 0.0:
                denom = cur_y - prev_y
                if abs(denom) <= 1e-12:
                    return float(cur_x)
                return float(prev_x + ((float(level) - prev_y) * (cur_x - prev_x) / denom))
            prev_x = cur_x
            prev_y = cur_y
            prev_delta = cur_delta
        return float("nan")

    def _ridge_center_half_flux_metrics(self) -> Dict[str, float]:
        x, y = self._profile_xy_arrays()
        center_flux = self._interp_y_at_zero(x, y)
        half_flux = 0.5 * center_flux if np.isfinite(center_flux) else float("nan")
        left_x = self._profile_level_crossing(x, y, center_y=center_flux, level=half_flux, side="left")
        right_x = self._profile_level_crossing(x, y, center_y=center_flux, level=half_flux, side="right")
        width = float(right_x - left_x) if np.isfinite(left_x) and np.isfinite(right_x) else float("nan")
        return {
            "center_flux": float(center_flux),
            "half_flux": float(half_flux),
            "left_x": float(left_x),
            "right_x": float(right_x),
            "width_px": float(width),
        }

    def _summary_text(self) -> str:
        profile = self._profile_dict()
        fit = self._fit_dict()
        params = self._params_dict()
        ridge_idx = self.record.get("ridge_idx", profile.get("ridge_idx", "n/a"))
        ridge_xy = profile.get("ridge_xy", self.record.get("ridge_xy", None))
        dist_px = _safe_float(self.record.get("distance_from_core_px", float("nan")))
        dist_mas = self._mas_value("distance_from_core_px", "distance_from_core_mas")
        raw_px = _safe_float(self.record.get("fwhm_px", fit.get("fwhm_px", float("nan"))))
        raw_mas = self._mas_value("fwhm_px", "fwhm_mas")
        intr_px = _safe_float(self.record.get("intrinsic_fwhm_px", float("nan")))
        intr_mas = self._mas_value("intrinsic_fwhm_px", "intrinsic_fwhm_mas")
        raw_ang = _safe_float(self.record.get("opening_angle_deg", float("nan")))
        intr_ang = _safe_float(self.record.get("intrinsic_opening_angle_deg", float("nan")))
        sigma = _safe_float(params.get("sigma", float("nan")))
        rmse = _safe_float(fit.get("rmse", float("nan")))
        mu = _safe_float(params.get("mu", float("nan")))
        mu_bound = _safe_float(params.get("mu_bound_px", float("nan")))
        baseline = _safe_float(params.get("baseline", float("nan")))
        baseline_lower = _safe_float(params.get("baseline_bound_lower", float("nan")))
        baseline_upper = _safe_float(params.get("baseline_bound_upper", float("nan")))
        baseline_mode = str(params.get("baseline_mode", ""))
        mu_text = "fixed at 0 px" if bool(params.get("mu_fixed", False)) else (
            f"{_format_float_or_na(mu)} px"
            f" (bound ±{_format_float_or_na(mu_bound)} px)" if np.isfinite(mu_bound) else f"{_format_float_or_na(mu)} px"
        )
        fwhm_pa_rms = _safe_float(self.record.get("fwhm_pa_rms_px", float("nan")))
        intrinsic_pa_rms = _safe_float(self.record.get("intrinsic_fwhm_pa_rms_px", float("nan")))
        pa_sweep = self.record.get("pa_sweep", {})
        pa_success = int(pa_sweep.get("success_count", 0) or 0) if isinstance(pa_sweep, dict) else 0
        half_metrics = self._ridge_center_half_flux_metrics()
        fit_window = self._fit_window_dict()
        if "gaussian_unstable" in self.record:
            gaussian_unstable = bool(self.record.get("gaussian_unstable", False))
            stability_reasons = list(self.record.get("gaussian_unstable_reasons", []) or [])
        else:
            try:
                stability = evaluate_gaussian_fit_stability(self.record, profile=profile, fit=fit)
                gaussian_unstable = bool(stability.get("gaussian_unstable", False))
                stability_reasons = list(stability.get("gaussian_unstable_reasons", []) or [])
            except Exception:
                gaussian_unstable = False
                stability_reasons = []
        stability_text = (
            "unstable: " + ", ".join(str(reason) for reason in stability_reasons)
            if gaussian_unstable
            else "stable"
        )
        lines = [
            f"Slice: ridge_idx={ridge_idx} | ridge center={self._point_text(ridge_xy)} | Gaussian mu={mu_text}",
            (
                "Distance from core: "
                f"{_format_float_or_na(dist_px)} px / {_format_float_or_na(dist_mas)} mas"
            ),
            (
                "FWHM raw: "
                f"{_format_float_or_na(raw_px)} px / {_format_float_or_na(raw_mas)} mas | "
                "intrinsic deconvolved: "
                f"{_format_float_or_na(intr_px)} px / {_format_float_or_na(intr_mas)} mas"
            ),
            (
                "Opening angle raw/intrinsic: "
                f"{_format_float_or_na(raw_ang)} / {_format_float_or_na(intr_ang)} deg | "
                f"sigma={_format_float_or_na(sigma)} px | RMSE={_format_float_or_na(rmse)}"
            ),
            (
                "Gaussian baseline: "
                f"c={_format_float_or_na(baseline)} | "
                f"mode={baseline_mode or 'n/a'} | "
                f"bounds=[{_format_float_or_na(baseline_lower)}, {_format_float_or_na(baseline_upper)}]"
            ),
            (
                "PA sweep error: "
                f"raw={_format_float_or_na(fwhm_pa_rms)} px | "
                f"intrinsic={_format_float_or_na(intrinsic_pa_rms)} px | "
                f"success={pa_success}"
            ),
            (
                "Ridge-center half flux: "
                f"level={_format_float_or_na(half_metrics['half_flux'])} | "
                f"xL/xR={_format_float_or_na(half_metrics['left_x'])}/"
                f"{_format_float_or_na(half_metrics['right_x'])} px | "
                f"width={_format_float_or_na(half_metrics['width_px'])} px"
            ),
            f"Gaussian stability: {stability_text}",
        ]
        if fit_window:
            lines.append(
                "Gaussian fit window: "
                f"{fit_window.get('mode', 'window')} | "
                f"x=[{_format_float_or_na(fit_window.get('left_x'))}, "
                f"{_format_float_or_na(fit_window.get('right_x'))}] px | "
                f"n={fit_window.get('n_points', 'n/a')}/{fit_window.get('full_n_points', 'n/a')}"
            )
        return "\n".join(lines)

    def _plot(self) -> None:
        self.profile_ax.clear()
        self.residual_ax.clear()
        profile = self._profile_dict()
        fit = self._fit_dict()
        params = self._params_dict()
        fit_window = self._fit_window_dict()

        valid_x, valid_y = self._profile_xy_arrays()
        profile_scale = _safe_float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        valid_x_mas = _scale_px_values_to_mas(valid_x, profile_scale)
        if valid_x.size == valid_y.size and valid_x.size > 0:
            finite = np.isfinite(valid_x_mas) & np.isfinite(valid_y)
            if np.any(finite):
                self.profile_ax.plot(
                    valid_x_mas[finite],
                    valid_y[finite],
                    "o-",
                    color="#1f77b4",
                    linewidth=1.2,
                    markersize=3.0,
                    label="profile samples",
                )
        window_left = _safe_float(fit_window.get("left_x", float("nan")))
        window_right = _safe_float(fit_window.get("right_x", float("nan")))
        window_left_mas = float(window_left * profile_scale) if np.isfinite(window_left) and np.isfinite(profile_scale) and profile_scale > 0.0 else float("nan")
        window_right_mas = float(window_right * profile_scale) if np.isfinite(window_right) and np.isfinite(profile_scale) and profile_scale > 0.0 else float("nan")
        if np.isfinite(window_left_mas) and np.isfinite(window_right_mas) and window_right_mas > window_left_mas:
            self.profile_ax.axvspan(
                window_left_mas,
                window_right_mas,
                color="#d62728",
                alpha=0.08,
                label="Gaussian fit window",
            )

        fit_x = np.asarray(fit.get("x", []), dtype=np.float64)
        fit_y = np.asarray(fit.get("fit_y", []), dtype=np.float64)
        fit_data_y = np.asarray(fit.get("y", []), dtype=np.float64)
        full_fit_x = self._gaussian_plot_x_px(profile, fit_x)
        full_fit_y = self._gaussian_model_y(full_fit_x, params)
        full_fit_x_mas = _scale_px_values_to_mas(full_fit_x, profile_scale)
        if full_fit_x.size == full_fit_y.size and full_fit_x.size > 0:
            finite_fit = np.isfinite(full_fit_x_mas) & np.isfinite(full_fit_y)
            if np.any(finite_fit):
                order = np.argsort(full_fit_x_mas[finite_fit])
                self.profile_ax.plot(
                    full_fit_x_mas[finite_fit][order],
                    full_fit_y[finite_fit][order],
                    "-",
                    color="#d62728",
                    linewidth=1.6,
                    label="Gaussian fit (scan window)",
                )

        baseline = _safe_float(params.get("baseline", float("nan")))
        amplitude = _safe_float(params.get("amplitude", float("nan")))
        if np.isfinite(baseline) and np.isfinite(amplitude):
            self.profile_ax.axhline(
                baseline,
                color="#9467bd",
                linestyle="-.",
                linewidth=1.0,
                label="Gaussian baseline",
            )
            half_level = float(baseline + (0.5 * amplitude))
            self.profile_ax.axhline(
                half_level,
                color="#7f7f7f",
                linestyle=":",
                linewidth=1.1,
                label="Gaussian half level",
            )

        self.profile_ax.axvline(0.0, color="#2ca02c", linestyle="-", linewidth=1.2, label="ridge center")
        half_metrics = self._ridge_center_half_flux_metrics()
        half_flux = float(half_metrics.get("half_flux", float("nan")))
        half_left = float(half_metrics.get("left_x", float("nan")))
        half_right = float(half_metrics.get("right_x", float("nan")))
        half_left_mas = float(half_left * profile_scale) if np.isfinite(half_left) and np.isfinite(profile_scale) and profile_scale > 0.0 else float("nan")
        half_right_mas = float(half_right * profile_scale) if np.isfinite(half_right) and np.isfinite(profile_scale) and profile_scale > 0.0 else float("nan")
        if np.isfinite(half_flux):
            self.profile_ax.axhline(
                half_flux,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.1,
                label="ridge-center half flux",
            )
        if np.isfinite(half_flux) and np.isfinite(half_left_mas):
            self.profile_ax.axvline(
                half_left_mas,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.2,
                label="ridge half-flux points",
            )
            self.profile_ax.plot([half_left_mas], [half_flux], "s", color="#2ca02c", markersize=5.0)
        if np.isfinite(half_flux) and np.isfinite(half_right_mas):
            self.profile_ax.axvline(half_right_mas, color="#2ca02c", linestyle="--", linewidth=1.2)
            self.profile_ax.plot([half_right_mas], [half_flux], "s", color="#2ca02c", markersize=5.0)

        raw_fwhm = _safe_float(self.record.get("fwhm_px", fit.get("fwhm_px", float("nan"))))
        if np.isfinite(raw_fwhm) and raw_fwhm > 0.0 and np.isfinite(profile_scale) and profile_scale > 0.0:
            half_raw = 0.5 * float(raw_fwhm) * float(profile_scale)
            self.profile_ax.axvline(-half_raw, color="#ff7f0e", linestyle="--", linewidth=1.2, label="FWHM half points")
            self.profile_ax.axvline(half_raw, color="#ff7f0e", linestyle="--", linewidth=1.2)

        fit_x_mas = _scale_px_values_to_mas(fit_x, profile_scale)
        if fit_x.size == fit_y.size == fit_data_y.size and fit_x.size > 0:
            finite_res = np.isfinite(fit_x_mas) & np.isfinite(fit_y) & np.isfinite(fit_data_y)
            if np.any(finite_res):
                residual = fit_data_y[finite_res] - fit_y[finite_res]
                self.residual_ax.plot(
                    fit_x_mas[finite_res],
                    residual,
                    "o-",
                    color="#4c78a8",
                    linewidth=1.0,
                    markersize=3.0,
                )
        self.residual_ax.axhline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.0)

        self.profile_ax.set_title("Transverse Slice Details")
        self.profile_ax.set_ylabel("Flux / reconstructed level")
        self.profile_ax.grid(alpha=0.25)
        self.profile_ax.legend(loc="best")
        scan_xlim = self._scan_xlim_mas(profile, profile_scale)
        if scan_xlim is not None:
            self.profile_ax.set_xlim(*scan_xlim)
            self.residual_ax.set_xlim(*scan_xlim)
        if not (np.isfinite(profile_scale) and profile_scale > 0.0):
            self.profile_ax.text(
                0.5,
                0.5,
                "Scale calibration required for mas profile axis",
                transform=self.profile_ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
        self.residual_ax.set_xlabel("Transverse offset from ridgeline center (mas)")
        self.residual_ax.set_ylabel("Residual")
        self.residual_ax.grid(alpha=0.25)
        self.canvas.draw_idle()


class QtRidgelineAnalysisDialog(QDialog, _QtViewportMixin):
    def __init__(
        self,
        flux_map: np.ndarray,
        region_depth_map: np.ndarray,
        roi_mask: np.ndarray,
        thin_mask: np.ndarray,
        cmap_name: str,
        log_enabled: bool,
        calibration_context: Optional[Dict[str, object]],
        analysis_context: Optional[Dict[str, object]],
        max_width: int,
        max_height: int,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Ridgeline / FWHM Analysis")
        self.flux_map = np.asarray(flux_map, dtype=np.float32)
        self.region_depth_map = np.asarray(region_depth_map, dtype=np.int32)
        self.roi_mask = np.asarray(roi_mask, dtype=np.uint8)
        self.support_mask = build_flux_zero_support_mask(
            self.flux_map,
            self.roi_mask,
            self.region_depth_map,
        )
        self.thin_mask = np.asarray(thin_mask, dtype=np.uint8)
        self.cmap_name = str(cmap_name)
        self.log_enabled = bool(log_enabled)
        self.calibration_context = dict(calibration_context or {})
        self.analysis_context = dict(analysis_context or {})
        self.display_stride = _display_stride_for_1080p(self.flux_map.shape[:2])
        display_flux_map = self.flux_map[::self.display_stride, ::self.display_stride]
        display_roi_mask = _max_pool_mask_for_display(self.roi_mask, self.display_stride)
        display_thin_mask = _max_pool_mask_for_display(self.thin_mask, self.display_stride)
        self.base_image = _render_flux_color_image(
            flux_map=display_flux_map,
            valid_mask=display_roi_mask,
            thin_mask=display_thin_mask,
            cmap_name=self.cmap_name,
            log_enabled=self.log_enabled,
        )
        self.core_xy: Optional[Point] = None
        self.tail_xy: Optional[Point] = None
        self.hover_point: Optional[Point] = None
        self.ridge_xy: Optional[np.ndarray] = None
        self.ridgeline_metadata: Dict[str, object] = {}
        self.width_lines_xy: List[Tuple[Point, Point]] = []
        self.measure_result: Optional[Dict[str, object]] = None
        self.pending_slice_click_xy: Optional[Point] = None
        self.selected_slice_record_index: Optional[int] = None
        self.trend_rows: List[Dict[str, object]] = []
        self.trend_result: Optional[Dict[str, object]] = None
        self.trend_plot_style: Dict[str, object] = {
            "main_color": "#17becf",
            "main_size": 34.0,
            "raw_color": "#17becf",
            "raw_size": 28.0,
            "peak_color": "#1f77b4",
            "peak_size": 34.0,
            "residual_color": "#9467bd",
            "residual_size": 18.0,
            "show_raw_width": True,
            "show_raw_angle": True,
        }
        self._auto_fit_range: Optional[Tuple[int, int]] = None
        self._pending_loaded_trend_report: Optional[Dict[str, object]] = None
        self._trend_controls_sync_enabled = True
        self._init_viewport_state(self.base_image.shape[:2], max_width, max_height - 280)

        self.info_label = QLabel(
            "MOJAVE polar can auto-pick core/tail | Left click: override core then tail | "
            "Right click: remove nearest point | After FWHM, left click stages a slice for Details."
        )
        self.calibration_label = QLabel(_calibration_summary_text(self.calibration_context))
        self.stats_label = QLabel("Ridgeline: not extracted")
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("QLabel { color: #666666; font-size: 11px; }")
        self.label = InteractiveImageLabel()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(10)
        self.pan_x_slider = QSlider(Qt.Horizontal)
        self.pan_x_slider.setRange(0, 1000)
        self.pan_y_slider = QSlider(Qt.Horizontal)
        self.pan_y_slider.setRange(0, 1000)
        self.snap_radius_spin = QSpinBox()
        self.snap_radius_spin.setRange(1, 30)
        self.snap_radius_spin.setValue(8)
        self.ridge_mode_combo = QComboBox()
        self.ridge_mode_combo.addItem("MOJAVE polar", "mojave_polar")
        self.ridge_mode_combo.addItem("Legacy cost path", "legacy_cost_path")
        self.slice_count_spin = QSpinBox()
        self.slice_count_spin.setRange(2, 200)
        self.slice_count_spin.setValue(25)
        self.core_separation_spin = QDoubleSpinBox()
        self.core_separation_spin.setRange(0.0, 1_000_000.0)
        self.core_separation_spin.setDecimals(4)
        self.core_separation_spin.setSingleStep(0.1)
        self.core_separation_spin.setValue(0.0)
        self.trim_core_percent_spin = QDoubleSpinBox()
        self.trim_core_percent_spin.setRange(0.0, 45.0)
        self.trim_core_percent_spin.setDecimals(1)
        self.trim_core_percent_spin.setSingleStep(1.0)
        self.trim_core_percent_spin.setValue(10.0)
        self.trim_tail_percent_spin = QDoubleSpinBox()
        self.trim_tail_percent_spin.setRange(0.0, 45.0)
        self.trim_tail_percent_spin.setDecimals(1)
        self.trim_tail_percent_spin.setSingleStep(1.0)
        self.trim_tail_percent_spin.setValue(10.0)
        self.ridge_smooth_spin = QSpinBox()
        self.ridge_smooth_spin.setRange(0, 20)
        self.ridge_smooth_spin.setValue(4)
        self.profile_step_spin = QDoubleSpinBox()
        self.profile_step_spin.setRange(0.25, 5.0)
        self.profile_step_spin.setDecimals(2)
        self.profile_step_spin.setSingleStep(0.25)
        self.profile_step_spin.setValue(0.5)
        self.pa_sweep_error_check = QCheckBox("PA sweep error (+/-15 deg, 1 deg step)")
        self.pa_sweep_error_check.setChecked(False)
        self.clear_button = QPushButton("Clear Picks")
        self.extract_button = QPushButton("Extract Ridgeline")
        self.measure_button = QPushButton("Measure FWHM")
        self.details_button = QPushButton("Details")
        self.details_button.setEnabled(False)
        self.save_button = QPushButton("Save Result...")
        self.load_button = QPushButton("Load Result...")
        self.close_button = QPushButton("Close")
        self.details_status_label = QLabel("Details: measure FWHM, click a map slice, then press Details.")
        self.details_status_label.setWordWrap(True)

        self.profile_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.profile_canvas.setMinimumSize(0, 0)
        self.profile_ax = self.profile_fig.add_subplot(211)
        self.angle_ax = self.profile_fig.add_subplot(212, sharex=self.profile_ax)
        self.profile_ax.set_title("Width Result")
        self.profile_ax.set_xlabel("Distance from Core (mas)")
        self.profile_ax.set_ylabel("FWHM (px)")
        self.profile_ax.grid(alpha=0.25)
        self.angle_ax.set_title("Opening Angle")
        self.angle_ax.set_xlabel("Distance from Core (mas)")
        self.angle_ax.set_ylabel("Opening angle (deg)")
        self.angle_ax.grid(alpha=0.25)

        self.report_fig = Figure(figsize=(8, 6), constrained_layout=True)
        self.report_canvas = FigureCanvas(self.report_fig)
        self.report_width_ax = self.report_fig.add_subplot(221)
        self.report_angle_ax = self.report_fig.add_subplot(222)
        self.report_peak_ax = self.report_fig.add_subplot(223)
        self.report_k_ax = self.report_fig.add_subplot(224)
        self._init_report_axes()

        self.tabs = QTabWidget()
        map_tab = QWidget()
        map_layout = QVBoxLayout(map_tab)
        map_scroll = QScrollArea()
        map_scroll.setWidgetResizable(False)
        map_scroll.setWidget(self.label)
        map_scroll.setMinimumHeight(320)
        map_layout.addWidget(map_scroll)
        details_row = QHBoxLayout()
        details_row.addWidget(self.details_status_label, stretch=1)
        details_row.addWidget(self.details_button)
        map_layout.addLayout(details_row)
        self.tabs.addTab(map_tab, "Map")
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.addWidget(self.profile_canvas)
        self.tabs.addTab(plot_tab, "Width Profile")
        report_tab = QWidget()
        report_layout = QHBoxLayout(report_tab)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_layout.setSpacing(6)
        self.report_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.report_canvas.setMinimumSize(0, 0)
        report_side = QVBoxLayout()
        self.trend_stats_label = QLabel("Trend / Report: measure widths first.")
        self.trend_stats_label.setWordWrap(True)
        self.trend_stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.trend_stats_label.setMinimumHeight(44)
        self.trend_stats_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.trend_stats_label.setStyleSheet(
            "QLabel { border: 1px solid #cfcfcf; padding: 4px; background: #fafafa; }"
        )
        self.use_raw_width_check = QCheckBox("Use raw width")
        self.use_raw_width_check.setChecked(False)
        self.filter_unstable_gaussian_check = QCheckBox("Filter unstable Gaussian fits")
        self.filter_unstable_gaussian_check.setChecked(False)
        self.trend_fit_mode_combo = QComboBox()
        self.trend_fit_mode_combo.addItem("k≈1 range", "k_near_one")
        self.trend_fit_mode_combo.addItem("Opening plateau", "opening_plateau")
        self.show_paper_model_check = QCheckBox("Show paper Fig.7 model")
        self.show_paper_model_check.setChecked(True)
        self.auto_x_range_check = QCheckBox("Auto X Range")
        self.auto_x_range_check.setChecked(True)
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(0.0, 1_000_000.0)
        self.x_min_spin.setDecimals(4)
        self.x_min_spin.setSingleStep(0.1)
        self.x_min_spin.setValue(0.1)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(0.0, 1_000_000.0)
        self.x_max_spin.setDecimals(4)
        self.x_max_spin.setSingleStep(0.1)
        self.x_max_spin.setValue(10.0)
        self.auto_width_y_range_check = QCheckBox("Auto Width Y Range")
        self.auto_width_y_range_check.setChecked(True)
        self.width_y_min_spin = QDoubleSpinBox()
        self.width_y_min_spin.setRange(0.0, 1_000_000.0)
        self.width_y_min_spin.setDecimals(4)
        self.width_y_min_spin.setSingleStep(0.1)
        self.width_y_min_spin.setValue(0.1)
        self.width_y_max_spin = QDoubleSpinBox()
        self.width_y_max_spin.setRange(0.0, 1_000_000.0)
        self.width_y_max_spin.setDecimals(4)
        self.width_y_max_spin.setSingleStep(0.1)
        self.width_y_max_spin.setValue(10.0)
        self.trend_cut_min_sep_spin = QDoubleSpinBox()
        self.trend_cut_min_sep_spin.setRange(0.0, 1_000_000.0)
        self.trend_cut_min_sep_spin.setDecimals(4)
        self.trend_cut_min_sep_spin.setSingleStep(0.1)
        self.trend_cut_min_sep_spin.setValue(0.0)
        self.trend_cut_max_sep_spin = QDoubleSpinBox()
        self.trend_cut_max_sep_spin.setRange(0.0, 1_000_000.0)
        self.trend_cut_max_sep_spin.setDecimals(4)
        self.trend_cut_max_sep_spin.setSingleStep(0.1)
        self.trend_cut_max_sep_spin.setValue(0.0)
        self.local_k_window_spin = QSpinBox()
        self.local_k_window_spin.setRange(3, 31)
        self.local_k_window_spin.setSingleStep(2)
        self.local_k_window_spin.setValue(7)
        self.fit_start_spin = QSpinBox()
        self.fit_start_spin.setRange(1, 1)
        self.fit_start_spin.setValue(1)
        self.fit_end_spin = QSpinBox()
        self.fit_end_spin.setRange(1, 1)
        self.fit_end_spin.setValue(1)
        self.trend_plot_style_button = QPushButton("Plot Style...")
        self.apply_trend_button = QPushButton("Apply Trend Fit")
        self.auto_fit_button = QPushButton("Auto Select k≈1 Range")
        self.trend_result_label = QLabel("Trend fit result: not analyzed")
        self.trend_result_label.setWordWrap(True)
        self.trend_window_details = QPlainTextEdit()
        self.trend_window_details.setReadOnly(True)
        self.trend_window_details.setPlainText("Window ensemble: not analyzed")
        self.trend_window_details.setMinimumHeight(120)
        self.trend_window_details.setMaximumHeight(190)
        self.trend_window_details.setStyleSheet(
            "QPlainTextEdit { font-family: monospace; font-size: 10px; background: #fbfbfb; }"
        )
        self.trend_cut_min_sep_label = QLabel("Cut Min Separation")
        self.trend_cut_max_sep_label = QLabel("Cut Max Separation")
        self.core_separation_label = QLabel("Core Separation")
        self.x_min_label = QLabel("X Min")
        self.x_max_label = QLabel("X Max")
        self.width_y_min_label = QLabel("Width Y Min")
        self.width_y_max_label = QLabel("Width Y Max")
        trend_controls = QFormLayout()
        trend_controls.addRow("", self.use_raw_width_check)
        trend_controls.addRow("", self.filter_unstable_gaussian_check)
        trend_controls.addRow("Fit Mode", self.trend_fit_mode_combo)
        trend_controls.addRow("", self.show_paper_model_check)
        trend_controls.addRow("", self.auto_x_range_check)
        trend_controls.addRow(self.x_min_label, self.x_min_spin)
        trend_controls.addRow(self.x_max_label, self.x_max_spin)
        trend_controls.addRow("", self.auto_width_y_range_check)
        trend_controls.addRow(self.width_y_min_label, self.width_y_min_spin)
        trend_controls.addRow(self.width_y_max_label, self.width_y_max_spin)
        trend_controls.addRow(self.trend_cut_min_sep_label, self.trend_cut_min_sep_spin)
        trend_controls.addRow(self.trend_cut_max_sep_label, self.trend_cut_max_sep_spin)
        trend_controls.addRow(self.core_separation_label, self.core_separation_spin)
        trend_controls.addRow("Local k Window", self.local_k_window_spin)
        trend_controls.addRow("Fit Start Slice", self.fit_start_spin)
        trend_controls.addRow("Fit End Slice", self.fit_end_spin)
        trend_controls.addRow("", self.trend_plot_style_button)
        report_side.addWidget(self.trend_stats_label, 0)
        report_side.addLayout(trend_controls)
        report_side.addWidget(self.apply_trend_button)
        report_side.addWidget(self.auto_fit_button)
        report_side.addWidget(self.trend_result_label)
        report_side.addWidget(self.trend_window_details)
        report_side.addStretch(1)
        report_side_widget = QWidget()
        report_side_widget.setLayout(report_side)
        report_side_widget.setMinimumWidth(280)
        report_side_widget.setMaximumWidth(360)
        report_side_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.report_splitter = QSplitter(Qt.Horizontal)
        self.report_splitter.setChildrenCollapsible(False)
        self.report_splitter.addWidget(self.report_canvas)
        self.report_splitter.addWidget(report_side_widget)
        self.report_splitter.setStretchFactor(0, 5)
        self.report_splitter.setStretchFactor(1, 1)
        self.report_splitter.setSizes([1000, 320])
        report_layout.addWidget(self.report_splitter, stretch=1)
        self.tabs.addTab(report_tab, "Trend / Report")
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.calibration_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.tabs, stretch=1)
        controls = QFormLayout()
        controls.addRow("Zoom x10", self.zoom_slider)
        controls.addRow("Pan X", self.pan_x_slider)
        controls.addRow("Pan Y", self.pan_y_slider)
        controls.addRow("Ridgeline Mode", self.ridge_mode_combo)
        controls.addRow("Snap Radius", self.snap_radius_spin)
        controls.addRow("Fallback Slice Count", self.slice_count_spin)
        controls.addRow("Trim Core Side (%)", self.trim_core_percent_spin)
        controls.addRow("Trim Tail Side (%)", self.trim_tail_percent_spin)
        controls.addRow("Ridgeline Smooth", self.ridge_smooth_spin)
        controls.addRow("Fallback Profile Step (px)", self.profile_step_spin)
        controls.addRow("", self.pa_sweep_error_check)
        layout.addLayout(controls)
        button_row = QHBoxLayout()
        button_row.addWidget(self.clear_button)
        button_row.addWidget(self.extract_button)
        button_row.addWidget(self.measure_button)
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.save_button)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            max_w = max(900, int(geom.width()) - 64)
            max_h = max(760, int(geom.height()) - 64)
            self.setMaximumSize(max_w, max_h)
            target_w = int(min(max_w, max(1100, int(max_width) + 280)))
            target_h = int(min(max_h, max(820, int(max_height) + 360)))
            self.resize(target_w, target_h)

        self.label.on_left_press = self._on_left_press
        self.label.on_move = self._on_move
        self.label.on_right_press = self._on_right_press
        self.label.on_wheel = self._on_wheel
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.pan_x_slider.valueChanged.connect(self._on_pan_changed)
        self.pan_y_slider.valueChanged.connect(self._on_pan_changed)
        self.clear_button.clicked.connect(self._clear_points)
        self.extract_button.clicked.connect(self._extract_ridgeline)
        self.measure_button.clicked.connect(self._measure_widths)
        self.details_button.clicked.connect(self._show_slice_details)
        self.load_button.clicked.connect(self._load_result)
        self.save_button.clicked.connect(self._save_result)
        self.close_button.clicked.connect(self.accept)
        self.use_raw_width_check.toggled.connect(self._mark_trend_dirty)
        self.filter_unstable_gaussian_check.toggled.connect(self._on_unstable_gaussian_filter_changed)
        self.trend_fit_mode_combo.currentIndexChanged.connect(self._on_trend_fit_mode_changed)
        self.show_paper_model_check.toggled.connect(self._mark_trend_dirty)
        self.auto_x_range_check.toggled.connect(self._mark_trend_dirty)
        self.auto_x_range_check.toggled.connect(self._update_axis_override_controls)
        self.x_min_spin.valueChanged.connect(self._mark_trend_dirty)
        self.x_max_spin.valueChanged.connect(self._mark_trend_dirty)
        self.auto_width_y_range_check.toggled.connect(self._mark_trend_dirty)
        self.auto_width_y_range_check.toggled.connect(self._update_axis_override_controls)
        self.width_y_min_spin.valueChanged.connect(self._mark_trend_dirty)
        self.width_y_max_spin.valueChanged.connect(self._mark_trend_dirty)
        self.trend_cut_min_sep_spin.valueChanged.connect(self._mark_trend_dirty)
        self.trend_cut_max_sep_spin.valueChanged.connect(self._mark_trend_dirty)
        self.core_separation_spin.valueChanged.connect(self._mark_trend_dirty)
        self.local_k_window_spin.valueChanged.connect(self._mark_trend_dirty)
        self.fit_start_spin.valueChanged.connect(self._mark_trend_dirty)
        self.fit_end_spin.valueChanged.connect(self._mark_trend_dirty)
        self.trend_plot_style_button.clicked.connect(self._open_trend_plot_style_dialog)
        self.apply_trend_button.clicked.connect(self._apply_trend_fit)
        self.auto_fit_button.clicked.connect(self._auto_select_trend_range)
        self._core_separation_unit = self._distance_unit_label()
        self._trend_cut_unit = self._distance_unit_label()
        self._update_distance_control_labels()
        self._update_axis_override_controls()
        self._update_trend_fit_mode_controls()
        self.refresh_view()

    def _analysis_to_display_point(self, pt: Tuple[float, float]) -> Point:
        stride = int(max(1, self.display_stride))
        x = int(round(float(pt[0]) / float(stride)))
        y = int(round(float(pt[1]) / float(stride)))
        h, w = self.base_image.shape[:2]
        return (
            int(np.clip(x, 0, max(0, w - 1))),
            int(np.clip(y, 0, max(0, h - 1))),
        )

    def _display_to_analysis_point(self, pt: Point) -> Point:
        stride = int(max(1, self.display_stride))
        x = int(round(float(pt[0]) * float(stride)))
        y = int(round(float(pt[1]) * float(stride)))
        h, w = self.flux_map.shape[:2]
        return (
            int(np.clip(x, 0, max(0, w - 1))),
            int(np.clip(y, 0, max(0, h - 1))),
        )

    def _analysis_polyline_to_display(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.empty((0, 2), dtype=np.int32)
        stride = float(max(1, self.display_stride))
        out = np.rint(pts[:, :2] / stride).astype(np.int32)
        h, w = self.base_image.shape[:2]
        if out.size:
            out[:, 0] = np.clip(out[:, 0], 0, max(0, w - 1))
            out[:, 1] = np.clip(out[:, 1], 0, max(0, h - 1))
        return out

    def _init_report_axes(self) -> None:
        self.report_width_ax.set_title("Width vs core distance")
        self.report_width_ax.set_xlabel("Distance from core")
        self.report_width_ax.set_ylabel("Width")
        self.report_width_ax.grid(alpha=0.25)
        self.report_angle_ax.set_title("Opening angle vs core distance")
        self.report_angle_ax.set_xlabel("Distance from core")
        self.report_angle_ax.set_ylabel("Opening angle (deg)")
        self.report_angle_ax.grid(alpha=0.25)
        self.report_peak_ax.set_title("Peak level vs core distance")
        self.report_peak_ax.set_xlabel("Distance from core")
        self.report_peak_ax.set_ylabel("Peak level")
        self.report_peak_ax.grid(alpha=0.25)
        self.report_k_ax.set_title("Residual vs core distance")
        self.report_k_ax.set_xlabel("Distance from core")
        self.report_k_ax.set_ylabel("log10 width residual (dex)")
        self.report_k_ax.grid(alpha=0.25)

    def _trend_distance_unit(self) -> str:
        if self.measure_result is None:
            return self._distance_unit_label()
        direct_raw = self.measure_result.get("distance_from_core_mas", self.measure_result.get("distance_along_ridge_mas", []))
        try:
            direct_mas = np.asarray([] if direct_raw is None else direct_raw, dtype=np.float64)
        except Exception:
            direct_mas = np.asarray([], dtype=np.float64)
        if np.any(np.isfinite(direct_mas) & (direct_mas > 0.0)):
            return "mas"
        dist_mas = np.asarray(self.measure_result.get("fit_records", []), dtype=object)
        for item in dist_mas.tolist():
            if isinstance(item, dict):
                val = float(
                    item.get(
                        "distance_from_core_mas",
                        item.get("distance_along_ridge_mas", float("nan")),
                    )
                )
                if np.isfinite(val) and val > 0.0:
                    return "mas"
        return "px"

    def _distance_unit_label(self) -> str:
        scale = float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        return "mas" if np.isfinite(scale) and scale > 0.0 else "px"

    def _convert_distance_value(self, value: float, from_unit: str, to_unit: str) -> float:
        from_u = str(from_unit or "").strip().lower()
        to_u = str(to_unit or "").strip().lower()
        if from_u == to_u:
            return float(value)
        scale = float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        if not np.isfinite(scale) or scale <= 0.0:
            return float(value)
        if from_u == "px" and to_u == "mas":
            return float(value) * scale
        if from_u == "mas" and to_u == "px":
            return float(value) / scale
        return float(value)

    def _update_distance_control_labels(self, *, convert_values: bool = False) -> None:
        new_unit = self._distance_unit_label()
        if convert_values:
            self.core_separation_spin.blockSignals(True)
            self.trend_cut_min_sep_spin.blockSignals(True)
            self.trend_cut_max_sep_spin.blockSignals(True)
            self.x_min_spin.blockSignals(True)
            self.x_max_spin.blockSignals(True)
            self.width_y_min_spin.blockSignals(True)
            self.width_y_max_spin.blockSignals(True)
            self.core_separation_spin.setValue(
                self._convert_distance_value(
                    float(self.core_separation_spin.value()),
                    self._core_separation_unit,
                    new_unit,
                )
            )
            self.trend_cut_min_sep_spin.setValue(
                self._convert_distance_value(
                    float(self.trend_cut_min_sep_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.trend_cut_max_sep_spin.setValue(
                self._convert_distance_value(
                    float(self.trend_cut_max_sep_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.x_min_spin.setValue(
                self._convert_distance_value(
                    float(self.x_min_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.x_max_spin.setValue(
                self._convert_distance_value(
                    float(self.x_max_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.width_y_min_spin.setValue(
                self._convert_distance_value(
                    float(self.width_y_min_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.width_y_max_spin.setValue(
                self._convert_distance_value(
                    float(self.width_y_max_spin.value()),
                    self._trend_cut_unit,
                    new_unit,
                )
            )
            self.core_separation_spin.blockSignals(False)
            self.trend_cut_min_sep_spin.blockSignals(False)
            self.trend_cut_max_sep_spin.blockSignals(False)
            self.x_min_spin.blockSignals(False)
            self.x_max_spin.blockSignals(False)
            self.width_y_min_spin.blockSignals(False)
            self.width_y_max_spin.blockSignals(False)
        self._core_separation_unit = new_unit
        self._trend_cut_unit = new_unit
        self.core_separation_label.setText(f"Core Separation ({new_unit})")
        self.trend_cut_min_sep_label.setText(f"Cut Min Separation ({new_unit})")
        self.trend_cut_max_sep_label.setText(f"Cut Max Separation ({new_unit})")
        self.x_min_label.setText(f"X Min ({new_unit})")
        self.x_max_label.setText(f"X Max ({new_unit})")
        self.width_y_min_label.setText(f"Width Y Min ({new_unit})")
        self.width_y_max_label.setText(f"Width Y Max ({new_unit})")

    def _effective_core_separation_px(self) -> float:
        value = float(self.core_separation_spin.value())
        unit = self._distance_unit_label()
        if unit == "mas":
            scale = float(self.calibration_context.get("scale_mas_per_px", float("nan")))
            if np.isfinite(scale) and scale > 0.0:
                return float(max(0.0, value / scale))
        return float(max(0.0, value))

    def _apply_core_separation_to_current_measure_result(self) -> None:
        if not isinstance(self.measure_result, dict):
            return
        self.measure_result = apply_core_separation_to_measure_result(
            self.measure_result,
            core_separation_px=self._effective_core_separation_px(),
            scale_mas_per_px=self.calibration_context.get("scale_mas_per_px", None),
        )

    @staticmethod
    def _point_from_value(value: object) -> Optional[Tuple[float, float]]:
        try:
            arr = value.tolist() if isinstance(value, np.ndarray) else value
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                x = float(arr[0])
                y = float(arr[1])
                if np.isfinite(x) and np.isfinite(y):
                    return (x, y)
        except Exception:
            pass
        return None

    @classmethod
    def _record_center_xy(cls, record: Dict[str, object]) -> Optional[Tuple[float, float]]:
        center = cls._point_from_value(record.get("ridge_xy", None))
        if center is not None:
            return center
        profile = record.get("profile", {})
        if isinstance(profile, dict):
            center = cls._point_from_value(profile.get("ridge_xy", None))
            if center is not None:
                return center
        return None

    @classmethod
    def _record_width_line_xy(
        cls,
        record: Dict[str, object],
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        line = record.get("width_line_xy", None)
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            return None
        p1 = cls._point_from_value(line[0])
        p2 = cls._point_from_value(line[1])
        if p1 is None or p2 is None:
            return None
        return (p1, p2)

    @staticmethod
    def _point_segment_distance_px(
        pt: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        p = np.asarray(pt, dtype=np.float64)
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-9:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(float(np.dot(p - a, ab) / denom), 0.0, 1.0))
        closest = a + (t * ab)
        return float(np.linalg.norm(p - closest))

    def _valid_slice_detail_records(self) -> List[Tuple[int, Dict[str, object]]]:
        if not isinstance(self.measure_result, dict):
            return []
        records = self.measure_result.get("fit_records", [])
        if not isinstance(records, list):
            return []
        out: List[Tuple[int, Dict[str, object]]] = []
        for idx, item in enumerate(records):
            if not isinstance(item, dict):
                continue
            fit = item.get("fit", {})
            if not isinstance(fit, dict) or not bool(fit.get("success", False)):
                continue
            if self._record_center_xy(item) is None:
                continue
            out.append((int(idx), item))
        return out

    def _update_slice_details_controls(self, message: Optional[str] = None) -> None:
        has_records = bool(self._valid_slice_detail_records())
        self.details_button.setEnabled(has_records)
        if message is not None:
            self.details_status_label.setText(message)
            return
        if not has_records:
            self.details_status_label.setText("Details: measure FWHM, click a map slice, then press Details.")
        elif self.pending_slice_click_xy is None:
            self.details_status_label.setText("Details: click a map slice, then press Details.")

    def _clear_slice_details_selection(self, message: Optional[str] = None) -> None:
        self.pending_slice_click_xy = None
        self.selected_slice_record_index = None
        self._update_slice_details_controls(message)

    def _nearest_slice_detail_record(
        self,
        pt: Point,
    ) -> Optional[Tuple[int, Dict[str, object], float]]:
        click = (float(pt[0]), float(pt[1]))
        best: Optional[Tuple[int, Dict[str, object], float]] = None
        best_score = float("inf")
        for idx, record in self._valid_slice_detail_records():
            center = self._record_center_xy(record)
            if center is None:
                continue
            center_dist = float(np.hypot(click[0] - center[0], click[1] - center[1]))
            line = self._record_width_line_xy(record)
            if line is not None:
                line_dist = self._point_segment_distance_px(click, line[0], line[1])
                display_dist = float(min(center_dist, line_dist))
                score = float(display_dist + (0.03 * center_dist))
            else:
                display_dist = center_dist
                score = center_dist
            if score < best_score:
                best_score = float(score)
                best = (int(idx), record, float(display_dist))
        return best

    def _selected_slice_detail_record(self) -> Optional[Dict[str, object]]:
        if self.selected_slice_record_index is None or not isinstance(self.measure_result, dict):
            return None
        records = self.measure_result.get("fit_records", [])
        if not isinstance(records, list):
            return None
        idx = int(self.selected_slice_record_index)
        if idx < 0 or idx >= len(records):
            return None
        record = records[idx]
        return dict(record) if isinstance(record, dict) else None

    def _detail_record_with_profile(self, record: Dict[str, object]) -> Dict[str, object]:
        out = dict(record or {})
        if self.ridge_xy is None or len(self.ridge_xy) < 5 or not isinstance(self.measure_result, dict):
            return out
        ridge_idx = int(out.get("ridge_idx", -1))
        if ridge_idx < 0:
            return out
        try:
            profile = sample_transverse_profile(
                flux_map=self.flux_map,
                support_mask=self.support_mask,
                ridge_xy=self.ridge_xy,
                ridge_idx=ridge_idx,
                tangent_half_window=int(self.measure_result.get("tangent_half_window", 4)),
                profile_step_px=float(out.get("profile_step_px", self.measure_result.get("profile_step_px", self.profile_step_spin.value()))),
                scan_half_width_px=float(out.get("scan_half_width_px", self.measure_result.get("transverse_scan_half_width_px", float("nan")))),
            )
            fit = fit_transverse_gaussian(
                profile,
                mu_bound_px=float(self.measure_result.get("gaussian_mu_bound_px", float("nan"))),
                baseline_mode=str(self.measure_result.get("gaussian_baseline_mode", "fixed_zero")),
                baseline_l1_flux=float(self.measure_result.get("gaussian_baseline_l1_flux", float("nan"))),
                baseline_noise_sigma_flux=float(
                    self.measure_result.get("gaussian_baseline_noise_sigma_flux", float("nan"))
                ),
            )
            out["profile"] = profile
            out["fit"] = fit
            out["fit_recomputed_for_details"] = True
        except Exception as exc:
            out["profile_error"] = str(exc)
        return out

    def _show_slice_details(self) -> None:
        if not isinstance(self.measure_result, dict):
            self._update_slice_details_controls("Details: measure FWHM first.")
            return
        if self.pending_slice_click_xy is None:
            self._update_slice_details_controls("Details: click a point on the map first, then press Details.")
            return
        nearest = self._nearest_slice_detail_record(self.pending_slice_click_xy)
        if nearest is None:
            self._update_slice_details_controls("Details: no successful Gaussian slice is available.")
            return
        idx, record, distance = nearest
        self.selected_slice_record_index = int(idx)
        profile = record.get("profile", {})
        ridge_idx = record.get("ridge_idx", profile.get("ridge_idx", "n/a") if isinstance(profile, dict) else "n/a")
        self._update_slice_details_controls(
            f"Details: selected slice ridge_idx={ridge_idx} from staged map point "
            f"({int(self.pending_slice_click_xy[0])}, {int(self.pending_slice_click_xy[1])}); "
            f"selection distance={_format_float_or_na(distance)} px."
        )
        self.refresh_view(sync_sliders=False)
        dialog = QtSliceDetailsDialog(self._detail_record_with_profile(record), self.calibration_context, parent=self)
        dialog.exec_()

    def _filtered_trend_rows(
        self,
        rows: Sequence[Dict[str, object]],
        *,
        distance_unit: str,
    ) -> List[Dict[str, object]]:
        items = list(rows)
        min_sep = max(0.0, float(self.trend_cut_min_sep_spin.value()))
        max_sep = max(0.0, float(self.trend_cut_max_sep_spin.value()))
        use_max = bool(max_sep > 0.0)
        if min_sep <= 0.0 and not use_max:
            return items
        key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
        out: List[Dict[str, object]] = []
        for row in items:
            try:
                sep = float(row.get(key, float("nan")))
            except Exception:
                sep = float("nan")
            if np.isfinite(sep) and sep >= min_sep and (not use_max or sep <= max_sep):
                out.append(row)
        return out

    def _trend_cut_bounds(self) -> Tuple[float, float, str]:
        min_sep = max(0.0, float(self.trend_cut_min_sep_spin.value()))
        max_sep = max(0.0, float(self.trend_cut_max_sep_spin.value()))
        return float(min_sep), float(max_sep), self._trend_distance_unit()

    def _trend_cut_description(self) -> str:
        min_sep, max_sep, unit = self._trend_cut_bounds()
        if max_sep > 0.0:
            return f"[{_format_float_or_na(min_sep, '.4g')}, {_format_float_or_na(max_sep, '.4g')}] {unit}"
        return f">= {_format_float_or_na(min_sep, '.4g')} {unit}"

    def _intrinsic_half_opening_cut_summary(self, result: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        result = self.measure_result if result is None else result
        if not isinstance(result, dict):
            return {"median": float("nan"), "sigma": float("nan"), "count": 0}
        def _array(value: object) -> np.ndarray:
            try:
                arr = np.asarray([] if value is None else value, dtype=np.float64)
                return np.ravel(arr)
            except Exception:
                return np.asarray([], dtype=np.float64)

        min_sep, max_sep, unit = self._trend_cut_bounds()
        dist_key = "distance_from_core_mas" if unit == "mas" else "distance_from_core_px"
        fallback_dist_key = "distance_along_ridge_mas" if unit == "mas" else "distance_along_ridge_px"
        dist = _array(result.get(dist_key, result.get(fallback_dist_key, [])))
        dist_px = _array(result.get("distance_from_core_px", result.get("distance_along_ridge_px", [])))
        values = _array(result.get("intrinsic_half_opening_angle_deg", []))
        sigmas = _array(result.get("intrinsic_half_opening_angle_point_sigma_deg", []))
        n = int(min(dist.size, dist_px.size, values.size))
        if n <= 0:
            return {"median": float("nan"), "sigma": float("nan"), "count": 0}
        dist = dist[:n]
        dist_px = dist_px[:n]
        values = values[:n]
        sigmas = sigmas[:n] if sigmas.size >= n else np.pad(sigmas, (0, n - sigmas.size), constant_values=np.nan)
        valid = np.isfinite(dist) & np.isfinite(dist_px) & np.isfinite(values) & (dist >= min_sep)
        if max_sep > 0.0:
            valid &= dist <= max_sep
        scale = self.calibration_context.get("scale_mas_per_px", None)
        summary = _blocked_median_summary(
            values[valid],
            sigmas[valid],
            dist_px[valid],
            _safe_float(result.get("beam_size_px", float("nan"))),
            scale,
        )
        out = dict(summary)
        out["count"] = int(np.count_nonzero(valid))
        out["cut_min"] = float(min_sep)
        out["cut_max"] = float(max_sep)
        out["cut_unit"] = str(unit)
        return out

    def _update_measurement_stats_label(self, result: Optional[Dict[str, object]] = None) -> None:
        result = self.measure_result if result is None else result
        if not isinstance(result, dict):
            return
        mean_px = float(result.get("mean_fwhm_px", float("nan")))
        mean_mas = float(result.get("mean_fwhm_mas", float("nan")))
        mean_intrinsic = float(result.get("mean_intrinsic_fwhm_mas", float("nan")))
        cut_summary = self._intrinsic_half_opening_cut_summary(result)
        median_intrinsic_half_opening = float(cut_summary.get("median", float("nan")))
        intrinsic_half_opening_sigma = float(cut_summary.get("sigma", float("nan")))
        cut_count = int(cut_summary.get("count", 0) or 0)
        core_sep_px = float(result.get("core_separation_px", float("nan")))
        core_sep_mas = float(result.get("core_separation_mas", float("nan")))
        pa_sweep = result.get("pa_sweep", {})
        pa_suffix = ""
        if isinstance(pa_sweep, dict) and bool(pa_sweep.get("enabled", False)):
            hits = int(pa_sweep.get("cache_hits", 0) or 0)
            misses = int(pa_sweep.get("cache_misses", 0) or 0)
            pa_suffix = f" | PA sweep cache hit/miss={hits}/{misses}"
        tangent_suffix = ""
        tangent_half_window = result.get("tangent_half_window", None)
        if tangent_half_window is not None:
            tangent_suffix = f" | tangent window(auto)={int(tangent_half_window)}"
        slice_spacing_mas = float(result.get("slice_sampling_actual_median_spacing_mas", float("nan")))
        target_slice_spacing_mas = float(result.get("slice_sampling_step_mas", float("nan")))
        slice_sampling_suffix = ""
        if np.isfinite(slice_spacing_mas):
            slice_sampling_suffix = f" | slice step={_format_float_or_na(slice_spacing_mas, '.3g')} mas"
            if np.isfinite(target_slice_spacing_mas):
                slice_sampling_suffix += f" target={_format_float_or_na(target_slice_spacing_mas, '.3g')}"
        block_suffix = ""
        block_count = int(cut_summary.get("block_count", 0) or 0)
        block_size_slices = int(cut_summary.get("block_size_slices", 0) or 0)
        block_target_mas = float(cut_summary.get("block_target_mas", float("nan")))
        if block_count > 0:
            block_suffix = (
                f" | median block=0.5 beam"
                f"({_format_float_or_na(block_target_mas, '.3g')} mas, ~{block_size_slices} slices, n={block_count})"
            )
        self.stats_label.setText(
            f"FWHM valid slices: {int(result.get('valid_count', 0))} | "
            f"FWHM={_format_float_or_na(mean_px, '.4g')} px / {_format_float_or_na(mean_mas, '.4g')} mas | "
            f"deconv. intrinsic={_format_float_or_na(mean_intrinsic, '.4g')} mas | "
            f"half-opening median deconv="
            f"{_format_float_or_na(median_intrinsic_half_opening, '.4g')}±"
            f"{_format_float_or_na(intrinsic_half_opening_sigma, '.3g')} deg "
            f"(cut {self._trend_cut_description()}, n={cut_count}) | "
            f"core sep={_format_float_or_na(core_sep_px, '.4g')} px / {_format_float_or_na(core_sep_mas, '.4g')} mas | "
            f"trim(core/tail)={float(self.trim_core_percent_spin.value()):.1f}%/{float(self.trim_tail_percent_spin.value()):.1f}%"
            f"{slice_sampling_suffix}"
            f"{block_suffix}"
            f"{tangent_suffix}"
            f"{pa_suffix}"
        )

    def _split_unstable_gaussian_rows(
        self,
        rows: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        items = list(rows)
        if not bool(self.filter_unstable_gaussian_check.isChecked()):
            return items, []
        kept: List[Dict[str, object]] = []
        excluded: List[Dict[str, object]] = []
        for row in items:
            if bool(row.get("gaussian_unstable", False)):
                excluded.append(row)
            else:
                kept.append(row)
        return kept, excluded

    def _on_unstable_gaussian_filter_changed(self, *_args) -> None:
        if not self._trend_controls_sync_enabled:
            return
        self._auto_fit_range = None
        self._mark_trend_dirty()

    def _trend_fit_mode(self) -> str:
        try:
            mode = str(self.trend_fit_mode_combo.currentData() or "")
        except Exception:
            mode = ""
        return mode if mode in {"k_near_one", "opening_plateau"} else "k_near_one"

    def _update_trend_fit_mode_controls(self) -> None:
        if self._trend_fit_mode() == "opening_plateau":
            self.auto_fit_button.setText("Auto Find Plateau")
        else:
            self.auto_fit_button.setText("Auto Select k≈1 Range")

    def _on_trend_fit_mode_changed(self, *_args) -> None:
        self._update_trend_fit_mode_controls()
        if not self._trend_controls_sync_enabled:
            return
        self._auto_fit_range = None
        self._mark_trend_dirty()

    def _effective_local_k_window_points(self) -> int:
        window_points = int(max(3, int(self.local_k_window_spin.value())))
        if window_points % 2 == 0:
            window_points += 1
        return int(window_points)

    def _local_k_window_ensemble_points(self, row_count: int) -> List[int]:
        row_count = int(max(0, row_count))
        if row_count < 3:
            return []
        max_window = min(21, row_count if row_count % 2 == 1 else row_count - 1)
        candidates = set(range(5, 22, 2))
        return [int(w) for w in sorted(candidates) if 3 <= int(w) <= int(max_window)]

    def _compute_k_window_ensemble(
        self,
        rows: Sequence[Dict[str, object]],
        *,
        distance_unit: str,
    ) -> List[Dict[str, object]]:
        items = list(rows or [])
        if not items:
            return []
        x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
        y_key = "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"
        x = np.asarray([float(row.get(x_key, float("nan"))) for row in items], dtype=np.float64)
        y = np.asarray([float(row.get(y_key, float("nan"))) for row in items], dtype=np.float64)
        ensemble: List[Dict[str, object]] = []
        for window_points in self._local_k_window_ensemble_points(len(items)):
            try:
                local_k = compute_local_k(
                    x,
                    y,
                    window_points=int(window_points),
                    min_points=5,
                )
                auto = auto_select_k_near_one_range(
                    items,
                    local_k,
                    distance_unit=distance_unit,
                    min_points=max(5, int(window_points)),
                )
                fit = dict(auto.get("fit", {}) or {})
                start_order = int(auto.get("start_slice_order", fit.get("fit_start_slice_order", -1)))
                end_order = int(auto.get("end_slice_order", fit.get("fit_end_slice_order", -1)))
                ensemble.append(
                    {
                        "local_k_window_points": int(window_points),
                        "fit_start_slice_order": int(start_order),
                        "fit_end_slice_order": int(end_order),
                        "k_fit": float(fit.get("k_fit", float("nan"))),
                        "k_fit_sigma": float(fit.get("k_sigma", float("nan"))),
                        "opening_fit_median_deg": float(fit.get("opening_fit_median_deg", float("nan"))),
                        "half_opening_fit_median_deg": 0.5 * float(fit.get("opening_fit_median_deg", float("nan"))),
                        "half_opening_fit_median_sigma_deg": float(fit.get("half_opening_fit_median_sigma_deg", 0.5 * float(fit.get("opening_fit_median_sigma_deg", float("nan"))))),
                        "opening_sigma_deg": float(fit.get("opening_sigma_deg", float("nan"))),
                        "half_opening_sigma_deg": 0.5 * float(fit.get("opening_sigma_deg", float("nan"))),
                        "log_width_residual_mean_abs_dex": float(fit.get("fit_log_width_residual_mean_abs_dex", float("nan"))),
                        "all_log_width_residual_mean_abs_dex": float(fit.get("all_log_width_residual_mean_abs_dex", float("nan"))),
                        "all_log_width_sigma_dex": float(fit.get("all_log_width_sigma_dex", float("nan"))),
                        "log_width_residual_mean_dex": float(fit.get("fit_log_width_residual_mean_dex", float("nan"))),
                        "log_width_residual_rms_dex": float(fit.get("fit_log_width_residual_rms_dex", float("nan"))),
                        "opening_residual_mean_abs_deg": float(fit.get("fit_opening_residual_mean_abs_deg", float("nan"))),
                        "opening_residual_rms_deg": float(fit.get("fit_opening_residual_rms_deg", float("nan"))),
                        "fit_point_count": int(fit.get("fit_point_count", 0)),
                        "selection_score": float(auto.get("selection_score", float("nan"))),
                        "fit_k_score": float(auto.get("fit_k_score", float("nan"))),
                        "local_k_score": float(auto.get("local_k_score", float("nan"))),
                        "all_residual_mad_dex": float(auto.get("all_residual_mad_dex", float("nan"))),
                        "range_residual_mad_dex": float(auto.get("range_residual_mad_dex", auto.get("range_residual_rms_proxy_dex", float("nan")))),
                        "fit_k_score_norm": float(auto.get("fit_k_score_norm", float("nan"))),
                        "local_k_score_norm": float(auto.get("local_k_score_norm", float("nan"))),
                        "all_residual_mad_score_norm": float(auto.get("all_residual_mad_score_norm", auto.get("range_residual_mad_score_norm", float("nan")))),
                    }
                )
            except Exception as exc:
                ensemble.append(
                    {
                        "local_k_window_points": int(window_points),
                        "error": str(exc),
                    }
                )
        return ensemble

    def _k_window_ensemble_summary(self, ensemble: Sequence[Dict[str, object]]) -> Dict[str, object]:
        valid = [
            dict(item)
            for item in (ensemble or [])
            if isinstance(item, dict)
            and not item.get("error")
            and np.isfinite(_safe_float(item.get("k_fit", float("nan"))))
            and np.isfinite(_safe_float(item.get("half_opening_fit_median_deg", float("nan"))))
        ]
        if not valid:
            return {"valid_count": 0}
        k_values = np.asarray([_safe_float(item.get("k_fit", float("nan"))) for item in valid], dtype=np.float64)
        k_sigmas = np.asarray([_safe_float(item.get("k_fit_sigma", float("nan"))) for item in valid], dtype=np.float64)
        theta_half = np.asarray([_safe_float(item.get("half_opening_fit_median_deg", float("nan"))) for item in valid], dtype=np.float64)
        theta_half_sigmas = np.asarray([_safe_float(item.get("half_opening_fit_median_sigma_deg", float("nan"))) for item in valid], dtype=np.float64)
        log_abs = np.asarray([_safe_float(item.get("all_log_width_residual_mean_abs_dex", float("nan"))) for item in valid], dtype=np.float64)
        score_values = np.asarray([_safe_float(item.get("selection_score", float("nan"))) for item in valid], dtype=np.float64)

        def robust_sigma(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if finite.size <= 1:
                return float("nan")
            med = float(np.median(finite))
            sigma = float(1.4826 * np.median(np.abs(finite - med)))
            if np.isfinite(sigma) and sigma > 0.0:
                return sigma
            return float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")

        def propagated_median_sigma(values: np.ndarray) -> float:
            finite = values[np.isfinite(values) & (values >= 0.0)]
            if finite.size <= 0:
                return float("nan")
            return float(np.sqrt(np.pi / 2.0) * np.sqrt(np.sum(np.square(finite))) / float(finite.size))

        def combine_sigma(*values: float) -> float:
            finite = [float(v) for v in values if np.isfinite(float(v)) and float(v) >= 0.0]
            return float(np.sqrt(np.sum(np.square(finite)))) if finite else float("nan")

        k_scatter = robust_sigma(k_values)
        k_propagated = propagated_median_sigma(k_sigmas)
        theta_scatter = robust_sigma(theta_half)
        theta_propagated = propagated_median_sigma(theta_half_sigmas)
        return {
            "valid_count": int(len(valid)),
            "k_median": float(np.nanmedian(k_values)),
            "k_sigma": combine_sigma(k_scatter, k_propagated),
            "k_scatter": float(k_scatter),
            "k_measurement_sigma": float(k_propagated),
            "half_opening_median_deg": float(np.nanmedian(theta_half)),
            "half_opening_sigma_deg": combine_sigma(theta_scatter, theta_propagated),
            "half_opening_scatter_deg": float(theta_scatter),
            "half_opening_measurement_sigma_deg": float(theta_propagated),
            "log_width_residual_mean_abs_median_dex": float(np.nanmedian(log_abs[np.isfinite(log_abs)])) if np.any(np.isfinite(log_abs)) else float("nan"),
            "selection_score_median": float(np.nanmedian(score_values[np.isfinite(score_values)])) if np.any(np.isfinite(score_values)) else float("nan"),
        }

    def _best_k_window_ensemble_item(self, ensemble: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
        best_key = None
        best_item = None
        for item in ensemble or []:
            if not isinstance(item, dict) or item.get("error"):
                continue
            score = _safe_float(item.get("selection_score", float("nan")))
            k_fit = _safe_float(item.get("k_fit", float("nan")))
            log_abs = _safe_float(item.get("all_log_width_residual_mean_abs_dex", float("nan")))
            n_fit = int(item.get("fit_point_count", 0) or 0)
            if not np.isfinite(score) or not np.isfinite(k_fit):
                continue
            key = (
                float(score),
                abs(float(k_fit) - 1.0),
                float(log_abs) if np.isfinite(log_abs) else 1e9,
                -int(n_fit),
                int(item.get("local_k_window_points", 0) or 0),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_item = dict(item)
        return best_item

    def _format_k_window_ensemble_report(self, ensemble: Sequence[Dict[str, object]]) -> str:
        summary = self._k_window_ensemble_summary(ensemble)
        if int(summary.get("valid_count", 0)) <= 0:
            return "window ensemble: no valid k≈1 ranges"
        best = self._best_k_window_ensemble_item(ensemble)
        lines = [
            "k≈1 window ensemble: windows 5..21",
            "score: lower is better; |k_fit-1|/0.10 + median|local_k-1|/0.15 + MAD_all(logR)/0.03",
            (
                "median: "
                f"k={_format_float_or_na(float(summary.get('k_median', float('nan'))), '.4g')}±"
                f"{_format_float_or_na(float(summary.get('k_sigma', float('nan'))), '.3g')}, "
                f"θ1/2={_format_float_or_na(float(summary.get('half_opening_median_deg', float('nan'))), '.4g')}±"
                f"{_format_float_or_na(float(summary.get('half_opening_sigma_deg', float('nan'))), '.3g')} deg, "
                f"median |logR|={_format_float_or_na(float(summary.get('log_width_residual_mean_abs_median_dex', float('nan'))), '.3g')} dex"
            ),
            (
                "best score: "
                f"w={int(best.get('local_k_window_points', 0)) if best else 'n/a'}, "
                f"slices={int(best.get('fit_start_slice_order', -1)) if best else -1}-{int(best.get('fit_end_slice_order', -1)) if best else -1}, "
                f"score={_format_float_or_na(float(best.get('selection_score', float('nan'))) if best else float('nan'), '.4g')}"
            ),
            "",
            " win    score       k    theta1/2   slices   N  MADall  |logR|all",
            "------------------------------------------------------------------",
        ]
        for item in ensemble:
            if not isinstance(item, dict):
                continue
            window_points = int(item.get("local_k_window_points", 0))
            if item.get("error"):
                lines.append(f"{window_points:4d}  failed  {item.get('error')}")
                continue
            lines.append(
                f"{window_points:4d}  "
                f"{_format_float_or_na(float(item.get('selection_score', float('nan'))), '.4g'):>8}  "
                f"{_format_float_or_na(float(item.get('k_fit', float('nan'))), '.4g'):>7}  "
                f"{_format_float_or_na(float(item.get('half_opening_fit_median_deg', float('nan'))), '.4g'):>9}  "
                f"{int(item.get('fit_start_slice_order', -1)):>3}-{int(item.get('fit_end_slice_order', -1)):<3}  "
                f"{int(item.get('fit_point_count', 0)):>3}  "
                f"{_format_float_or_na(float(item.get('all_residual_mad_dex', float('nan'))), '.3g'):>6}  "
                f"{_format_float_or_na(float(item.get('all_log_width_residual_mean_abs_dex', float('nan'))), '.3g'):>9}"
            )
        return "\n".join(lines)

    def _update_axis_override_controls(self, *_args) -> None:
        x_manual = not bool(self.auto_x_range_check.isChecked())
        y_manual = not bool(self.auto_width_y_range_check.isChecked())
        self.x_min_spin.setEnabled(x_manual)
        self.x_max_spin.setEnabled(x_manual)
        self.width_y_min_spin.setEnabled(y_manual)
        self.width_y_max_spin.setEnabled(y_manual)

    def _update_trend_spin_ranges(self, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            self._trend_controls_sync_enabled = False
            self.fit_start_spin.setRange(1, 1)
            self.fit_end_spin.setRange(1, 1)
            self.fit_start_spin.setValue(1)
            self.fit_end_spin.setValue(1)
            self._trend_controls_sync_enabled = True
            return
        lo = int(min(int(row.get("slice_order", 1)) for row in rows))
        hi = int(max(int(row.get("slice_order", 1)) for row in rows))
        start_val = int(np.clip(int(self.fit_start_spin.value()), lo, hi))
        end_val = int(np.clip(int(self.fit_end_spin.value()), lo, hi))
        if end_val < start_val:
            end_val = start_val
        self._trend_controls_sync_enabled = False
        self.fit_start_spin.setRange(lo, hi)
        self.fit_end_spin.setRange(lo, hi)
        self.fit_start_spin.setValue(start_val)
        self.fit_end_spin.setValue(end_val)
        self._trend_controls_sync_enabled = True

    def _mark_trend_dirty(self, *_args) -> None:
        if not self._trend_controls_sync_enabled:
            return
        if self.measure_result is None:
            self.trend_stats_label.setText("Trend / Report: measure widths first.")
            self.trend_result_label.setText("Trend fit result: measure widths first.")
            self.trend_window_details.setPlainText("Window ensemble: measure widths first")
            return
        self.trend_stats_label.setText("Trend / Report: parameters changed. Click Apply Trend Fit.")
        self.trend_result_label.setText("Trend fit result: parameters changed. Click Apply Trend Fit.")
        self.trend_window_details.setPlainText("Window ensemble: parameters changed. Click Apply Trend Fit.")

    def _open_trend_plot_style_dialog(self) -> None:
        dialog = QtTrendPlotStyleDialog(self.trend_plot_style, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self.trend_plot_style = dict(dialog.get_settings())
        if self.measure_result is None:
            self.trend_stats_label.setText("Trend / Report: style updated.")
            self.trend_result_label.setText("Trend fit result: style updated.")
        else:
            self.trend_stats_label.setText("Trend / Report: plot style updated. Click Apply Trend Fit.")
            self.trend_result_label.setText("Trend fit result: plot style updated. Click Apply Trend Fit.")
            self.trend_window_details.setPlainText("Window ensemble: plot style changed. Click Apply Trend Fit.")

    def _apply_trend_fit(self, *_args) -> None:
        if self.measure_result is None:
            return
        self._apply_core_separation_to_current_measure_result()
        self._update_measurement_stats_label()
        self._update_profile_plot()
        self._update_trend_report_plot()

    def _auto_select_trend_range(self, *_args) -> None:
        if self.measure_result is None:
            return
        rows = build_gaussian_report_rows(
            self.measure_result,
            self.calibration_context.get("scale_mas_per_px", None),
            use_raw_width=bool(self.use_raw_width_check.isChecked()),
        )
        rows = self._filtered_trend_rows(rows, distance_unit=self._trend_distance_unit())
        rows, excluded_rows = self._split_unstable_gaussian_rows(rows)
        if not rows:
            suffix = f" ({len(excluded_rows)} unstable rows filtered)." if excluded_rows else "."
            self.trend_stats_label.setText(f"Trend / Report: no valid rows for auto-select{suffix}")
            self.trend_result_label.setText(f"Trend fit result: no valid rows for auto-select{suffix}")
            self.trend_window_details.setPlainText("Window ensemble: no valid rows")
            return
        distance_unit = self._trend_distance_unit()
        x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
        y_key = "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"
        x = np.asarray([float(row.get(x_key, float("nan"))) for row in rows], dtype=np.float64)
        y = np.asarray([float(row.get(y_key, float("nan"))) for row in rows], dtype=np.float64)
        effective_window = self._effective_local_k_window_points()
        local_k = compute_local_k(
            x,
            y,
            window_points=effective_window,
            min_points=5,
        )
        try:
            if self._trend_fit_mode() == "opening_plateau":
                auto = find_opening_angle_plateau(
                    rows,
                    distance_unit=distance_unit,
                    min_points=max(8, effective_window),
                )
                start_order = int(auto["plateau_start_slice_order"])
                end_order = int(auto["plateau_end_slice_order"])
            else:
                ensemble = self._compute_k_window_ensemble(rows, distance_unit=distance_unit)
                best = self._best_k_window_ensemble_item(ensemble)
                if best is None:
                    raise ValueError("No valid k≈1 window candidates.")
                start_order = int(best["fit_start_slice_order"])
                end_order = int(best["fit_end_slice_order"])
                self._trend_controls_sync_enabled = False
                self.local_k_window_spin.setValue(int(best["local_k_window_points"]))
                self._trend_controls_sync_enabled = True
                self.trend_window_details.setPlainText(self._format_k_window_ensemble_report(ensemble))
        except Exception as exc:
            self.trend_stats_label.setText(f"Trend / Report: auto-select failed ({exc}).")
            self.trend_result_label.setText(f"Trend fit result: auto-select failed ({exc}).")
            self.trend_window_details.setPlainText(f"Window ensemble: auto-select failed ({exc})")
            return
        self._auto_fit_range = (
            int(start_order),
            int(end_order),
        )
        self._update_trend_spin_ranges(rows)
        self._trend_controls_sync_enabled = False
        self.fit_start_spin.setValue(int(start_order))
        self.fit_end_spin.setValue(int(end_order))
        self._trend_controls_sync_enabled = True
        self.trend_stats_label.setText("Trend / Report: auto range selected. Click Apply Trend Fit.")
        if self._trend_fit_mode() == "opening_plateau":
            self.trend_result_label.setText(
                f"Trend fit result: auto-selected plateau slices {int(start_order)}-{int(end_order)}. "
                "Click Apply Trend Fit."
            )
        else:
            self.trend_result_label.setText(
                f"Trend fit result: auto-selected window {int(self.local_k_window_spin.value())}, "
                f"slices {int(start_order)}-{int(end_order)}. "
                "Click Apply Trend Fit."
            )

    def _clear_trend_report_plot(
        self,
        *,
        stats: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        self.report_width_ax.clear()
        self.report_angle_ax.clear()
        self.report_peak_ax.clear()
        self.report_k_ax.clear()
        self._init_report_axes()
        self.trend_rows = []
        self.trend_result = None
        if stats is not None:
            self.trend_stats_label.setText(stats)
        if result is not None:
            self.trend_result_label.setText(result)
        if details is not None:
            self.trend_window_details.setPlainText(details)
        self.report_canvas.draw_idle()

    def _update_trend_report_plot(self, *_args) -> None:
        if not self._trend_controls_sync_enabled:
            return
        self.report_width_ax.clear()
        self.report_angle_ax.clear()
        self.report_peak_ax.clear()
        self.report_k_ax.clear()
        self._init_report_axes()
        self.trend_rows = []
        self.trend_result = None
        if self.measure_result is None:
            self.trend_stats_label.setText("Trend / Report: measure widths first.")
            self.trend_result_label.setText("Trend fit result: measure widths first.")
            self.trend_window_details.setPlainText("Window ensemble: measure widths first")
            self.report_canvas.draw_idle()
            return

        pending_range = None
        if self._pending_loaded_trend_report is not None:
            trend = dict(self._pending_loaded_trend_report)
            self._pending_loaded_trend_report = None
            pending_range = (
                trend.get("fit_start_slice", None),
                trend.get("fit_end_slice", None),
            )
            self._trend_controls_sync_enabled = False
            use_raw_width = trend.get("use_raw_width", trend.get("use_raw_fallback", None))
            if use_raw_width is not None:
                self.use_raw_width_check.setChecked(bool(use_raw_width))
            if trend.get("filter_unstable_gaussian", None) is not None:
                self.filter_unstable_gaussian_check.setChecked(bool(trend.get("filter_unstable_gaussian")))
            if trend.get("fit_mode", None) is not None:
                fit_mode = str(trend.get("fit_mode"))
                idx = self.trend_fit_mode_combo.findData(fit_mode)
                if idx >= 0:
                    self.trend_fit_mode_combo.setCurrentIndex(idx)
            if trend.get("show_paper_model", None) is not None:
                self.show_paper_model_check.setChecked(bool(trend.get("show_paper_model")))
            if trend.get("local_k_window_points", None) is not None:
                self.local_k_window_spin.setValue(int(trend.get("local_k_window_points")))
            self._update_trend_fit_mode_controls()
            self._trend_controls_sync_enabled = True

        rows = build_gaussian_report_rows(
            self.measure_result,
            self.calibration_context.get("scale_mas_per_px", None),
            use_raw_width=bool(self.use_raw_width_check.isChecked()),
        )
        distance_unit = self._trend_distance_unit()
        rows = self._filtered_trend_rows(rows, distance_unit=distance_unit)
        rows, excluded_rows = self._split_unstable_gaussian_rows(rows)
        self.trend_rows = list(rows)
        if not rows:
            suffix = f" ({len(excluded_rows)} unstable rows filtered)." if excluded_rows else "."
            self.trend_stats_label.setText(f"Trend / Report: no valid Gaussian width rows{suffix}")
            self.trend_result_label.setText(f"Trend fit result: no valid rows available{suffix}")
            self.trend_window_details.setPlainText(f"Window ensemble: no valid rows{suffix}")
            self._update_trend_spin_ranges(rows)
            self.report_canvas.draw_idle()
            return

        beam_key = "beam_size_mas" if distance_unit == "mas" else "beam_size_px"
        raw_width_key = "gaussian_raw_width_mas" if distance_unit == "mas" else "gaussian_raw_width_px"
        usable_rows = len(rows)
        over_beam_rows = 0
        sep_values_for_stats: List[float] = []
        for row in rows:
            sep = _safe_float(row.get("distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px", float("nan")))
            if np.isfinite(sep) and sep > 0.0:
                sep_values_for_stats.append(float(sep))
            beam_val = _safe_float(row.get(beam_key, float("nan")))
            raw_width_val = _safe_float(row.get(raw_width_key, float("nan")))
            if np.isfinite(beam_val) and beam_val > 0.0 and np.isfinite(raw_width_val) and raw_width_val > beam_val:
                over_beam_rows += 1
        sep_min = float(np.min(sep_values_for_stats)) if sep_values_for_stats else float("nan")
        sep_max = float(np.max(sep_values_for_stats)) if sep_values_for_stats else float("nan")

        x_key = "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"
        y_key = "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"
        y_sigma_key = "gaussian_width_sigma_mas" if distance_unit == "mas" else "gaussian_width_sigma_px"
        x = np.asarray([float(row.get(x_key, float("nan"))) for row in rows], dtype=np.float64)
        y = np.asarray([float(row.get(y_key, float("nan"))) for row in rows], dtype=np.float64)
        y_sigma = np.asarray([float(row.get(y_sigma_key, float("nan"))) for row in rows], dtype=np.float64)
        excluded_x = np.asarray([float(row.get(x_key, float("nan"))) for row in excluded_rows], dtype=np.float64)
        excluded_y = np.asarray([float(row.get(y_key, float("nan"))) for row in excluded_rows], dtype=np.float64)
        excluded_open_y = np.asarray([float(row.get("gaussian_angle_deg", float("nan"))) for row in excluded_rows], dtype=np.float64)
        excluded_peak_y = np.asarray([float(row.get("profile_peak_value", float("nan"))) for row in excluded_rows], dtype=np.float64)
        raw_y = np.asarray(
            [
                float(
                    row.get(
                        "gaussian_raw_width_px" if distance_unit == "px" else (
                            "gaussian_width_mas" if row.get("used_width_source") == "raw_width" else "gaussian_raw_width_px"
                        ),
                        float("nan"),
                    )
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        if distance_unit == "mas":
            scale = float(self.calibration_context.get("scale_mas_per_px", float("nan")))
            if np.isfinite(scale) and scale > 0.0:
                raw_y = np.asarray([float(row.get("gaussian_raw_width_px", float("nan"))) * scale for row in rows], dtype=np.float64)
        open_y = np.asarray([float(row.get("gaussian_angle_deg", float("nan"))) for row in rows], dtype=np.float64)
        open_sigma = np.asarray([float(row.get("gaussian_angle_sigma_deg", float("nan"))) for row in rows], dtype=np.float64)
        raw_open_y = np.asarray([float(row.get("gaussian_raw_angle_deg", float("nan"))) for row in rows], dtype=np.float64)
        peak_y = np.asarray([float(row.get("profile_peak_value", float("nan"))) for row in rows], dtype=np.float64)
        effective_window = self._effective_local_k_window_points()
        local_k = compute_local_k(x, y, window_points=effective_window, min_points=5)

        self._update_trend_spin_ranges(rows)
        if pending_range is not None:
            self._trend_controls_sync_enabled = False
            if pending_range[0] is not None:
                self.fit_start_spin.setValue(int(pending_range[0]))
            if pending_range[1] is not None:
                self.fit_end_spin.setValue(int(pending_range[1]))
            self._trend_controls_sync_enabled = True
            local_k = compute_local_k(x, y, window_points=effective_window, min_points=5)

        fit_mode = self._trend_fit_mode()
        if self._auto_fit_range is None:
            try:
                if fit_mode == "opening_plateau":
                    auto = find_opening_angle_plateau(
                        rows,
                        distance_unit=distance_unit,
                        min_points=max(8, effective_window),
                    )
                    self._auto_fit_range = (
                        int(auto["plateau_start_slice_order"]),
                        int(auto["plateau_end_slice_order"]),
                    )
                else:
                    auto = auto_select_k_near_one_range(
                        rows,
                        local_k,
                        distance_unit=distance_unit,
                        min_points=max(5, effective_window),
                    )
                    self._auto_fit_range = (int(auto["start_slice_order"]), int(auto["end_slice_order"]))
                self._trend_controls_sync_enabled = False
                self.fit_start_spin.setValue(self._auto_fit_range[0])
                self.fit_end_spin.setValue(self._auto_fit_range[1])
                self._trend_controls_sync_enabled = True
            except Exception:
                self._auto_fit_range = None

        fit_start = int(self.fit_start_spin.value())
        fit_end = int(self.fit_end_spin.value())
        if fit_end < fit_start:
            fit_end = fit_start
            self._trend_controls_sync_enabled = False
            self.fit_end_spin.setValue(fit_end)
            self._trend_controls_sync_enabled = True

        trend_result = None
        try:
            if fit_mode == "opening_plateau":
                trend_result = find_opening_angle_plateau(
                    rows,
                    distance_unit=distance_unit,
                    min_points=max(8, effective_window),
                )
                self._trend_controls_sync_enabled = False
                self.fit_start_spin.setValue(int(trend_result["plateau_start_slice_order"]))
                self.fit_end_spin.setValue(int(trend_result["plateau_end_slice_order"]))
                self._trend_controls_sync_enabled = True
                fit_start = int(trend_result["plateau_start_slice_order"])
                fit_end = int(trend_result["plateau_end_slice_order"])
            else:
                trend_result = fit_power_law_from_rows(
                    rows,
                    start_slice_order=fit_start,
                    end_slice_order=fit_end,
                    distance_unit=distance_unit,
                )
        except Exception:
            trend_result = None
        if trend_result is not None and fit_mode != "opening_plateau":
            trend_result = dict(trend_result)
            trend_result["primary_local_k_window_points"] = int(effective_window)
            k_window_ensemble = self._compute_k_window_ensemble(rows, distance_unit=distance_unit)
            trend_result["k_window_ensemble"] = k_window_ensemble
            trend_result["k_window_ensemble_summary"] = self._k_window_ensemble_summary(k_window_ensemble)
            for item in k_window_ensemble:
                if (
                    isinstance(item, dict)
                    and not item.get("error")
                    and int(item.get("local_k_window_points", -1)) == int(effective_window)
                    and int(item.get("fit_start_slice_order", -2)) == int(fit_start)
                    and int(item.get("fit_end_slice_order", -2)) == int(fit_end)
                ):
                    trend_result["primary_selection_score"] = float(item.get("selection_score", float("nan")))
                    break
        self.trend_result = trend_result

        mask_main = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        mask_raw = np.isfinite(x) & np.isfinite(raw_y) & (x > 0.0) & (raw_y > 0.0)
        mask_open = np.isfinite(x) & np.isfinite(open_y) & (x > 0.0) & (open_y > 0.0)
        mask_raw_open = np.isfinite(x) & np.isfinite(raw_open_y) & (x > 0.0) & (raw_open_y > 0.0)
        mask_peak = np.isfinite(x) & np.isfinite(peak_y) & (x > 0.0) & (peak_y > 0.0)
        mask_excluded_width = np.isfinite(excluded_x) & np.isfinite(excluded_y) & (excluded_x > 0.0) & (excluded_y > 0.0)
        mask_excluded_open = (
            np.isfinite(excluded_x)
            & np.isfinite(excluded_open_y)
            & (excluded_x > 0.0)
            & (excluded_open_y > 0.0)
        )
        mask_excluded_peak = (
            np.isfinite(excluded_x)
            & np.isfinite(excluded_peak_y)
            & (excluded_x > 0.0)
            & (excluded_peak_y > 0.0)
        )
        style = dict(self.trend_plot_style or {})
        main_color = _normalize_mpl_color(style.get("main_color", "#17becf"), "#17becf")
        raw_color = _normalize_mpl_color(style.get("raw_color", "#17becf"), "#17becf")
        peak_color = _normalize_mpl_color(style.get("peak_color", "#1f77b4"), "#1f77b4")
        residual_color = _normalize_mpl_color(style.get("residual_color", "#9467bd"), "#9467bd")
        main_size = float(max(2.0, _safe_float(style.get("main_size", 34.0), 34.0)))
        raw_size = float(max(2.0, _safe_float(style.get("raw_size", 28.0), 28.0)))
        peak_size = float(max(2.0, _safe_float(style.get("peak_size", 34.0), 34.0)))
        residual_size = float(max(2.0, _safe_float(style.get("residual_size", 18.0), 18.0)))
        show_raw_width = bool(style.get("show_raw_width", True))
        show_raw_angle = bool(style.get("show_raw_angle", True))
        if np.any(mask_main):
            self.report_width_ax.errorbar(
                x[mask_main],
                y[mask_main],
                yerr=y_sigma[mask_main] if y_sigma.shape == y.shape else None,
                fmt="s",
                markersize=float(np.sqrt(main_size)),
                color=main_color,
                ecolor="0.25",
                elinewidth=0.9,
                capsize=2.5,
                alpha=0.9,
                label="Gaussian deconv width",
            )
        if np.any(mask_excluded_width):
            self.report_width_ax.scatter(
                excluded_x[mask_excluded_width],
                excluded_y[mask_excluded_width],
                s=max(28.0, raw_size),
                c="0.45",
                marker="x",
                linewidths=1.2,
                alpha=0.75,
                label="Filtered unstable fit",
            )
        if show_raw_width and np.any(mask_raw) and np.any(mask_main):
            self.report_width_ax.scatter(
                x[mask_raw],
                raw_y[mask_raw],
                s=raw_size,
                marker="s",
                facecolors="none",
                edgecolors=raw_color,
                linewidths=1.0,
                alpha=0.45,
                label="Gaussian raw FWHM",
            )
        if np.any(mask_open):
            self.report_angle_ax.errorbar(
                x[mask_open],
                open_y[mask_open],
                yerr=open_sigma[mask_open] if open_sigma.shape == open_y.shape else None,
                fmt="s",
                markersize=float(np.sqrt(main_size)),
                color=main_color,
                ecolor="0.25",
                elinewidth=0.9,
                capsize=2.5,
                alpha=0.9,
                label="Gaussian deconv angle",
            )
        if np.any(mask_excluded_open):
            self.report_angle_ax.scatter(
                excluded_x[mask_excluded_open],
                excluded_open_y[mask_excluded_open],
                s=max(28.0, raw_size),
                c="0.45",
                marker="x",
                linewidths=1.2,
                alpha=0.75,
                label="Filtered unstable fit",
            )
        if show_raw_angle and np.any(mask_raw_open) and np.any(mask_open):
            self.report_angle_ax.scatter(
                x[mask_raw_open],
                raw_open_y[mask_raw_open],
                s=raw_size,
                marker="s",
                facecolors="none",
                edgecolors=raw_color,
                linewidths=1.0,
                alpha=0.45,
                label="Gaussian raw angle",
            )
        if np.any(mask_peak):
            self.report_peak_ax.scatter(
                x[mask_peak],
                peak_y[mask_peak],
                s=peak_size,
                c=peak_color,
                alpha=0.9,
                edgecolors="none",
                label="Contour-profile peak level",
            )
        if np.any(mask_excluded_peak):
            self.report_peak_ax.scatter(
                excluded_x[mask_excluded_peak],
                excluded_peak_y[mask_excluded_peak],
                s=max(28.0, raw_size),
                c="0.45",
                marker="x",
                linewidths=1.2,
                alpha=0.75,
                label="Filtered unstable fit",
            )
        self.report_k_ax.axhline(0.0, color="tab:red", lw=1.0, ls="--", label="Residual = 0")

        if trend_result is not None:
            result_mode = str(trend_result.get("fit_mode", "k_near_one"))
            x_all = np.asarray(trend_result.get("x_all", np.array([], dtype=np.float32)), dtype=np.float64)
            width_model = np.asarray(trend_result.get("width_model", np.array([], dtype=np.float32)), dtype=np.float64)
            log_width_residual = np.asarray(trend_result.get("log_width_residual_all", np.array([], dtype=np.float32)), dtype=np.float64)
            opening_residual = np.asarray(trend_result.get("opening_residual_all", np.array([], dtype=np.float32)), dtype=np.float64)
            sigma_log_width = float(trend_result.get("log_width_sigma_dex", float("nan")))
            opening_model = np.asarray(trend_result.get("opening_model_deg", np.array([], dtype=np.float32)), dtype=np.float64)
            sigma_theta = float(trend_result.get("opening_sigma_deg", float("nan")))
            mask_model_w = np.isfinite(x_all) & np.isfinite(width_model) & (x_all > 0.0) & (width_model > 0.0)
            if result_mode == "opening_plateau":
                mask_resid = np.isfinite(x_all) & np.isfinite(opening_residual) & (x_all > 0.0)
            else:
                mask_resid = np.isfinite(x_all) & np.isfinite(log_width_residual) & (x_all > 0.0)
            mask_model_o = np.isfinite(x_all) & np.isfinite(opening_model) & (x_all > 0.0) & (opening_model > 0.0)
            if np.any(mask_model_w):
                if result_mode == "opening_plateau":
                    label = f"Plateau width model θ={float(trend_result.get('opening_fit_median_deg', float('nan'))):.3g}°"
                else:
                    label = f"PL fit k={float(trend_result.get('k_fit', float('nan'))):.3f}"
                self.report_width_ax.plot(x_all[mask_model_w], width_model[mask_model_w], color="black", lw=1.8, ls="--", label=label)
            if np.any(mask_model_o):
                angle_label = "Opening-angle plateau" if result_mode == "opening_plateau" else "Opening-angle trend"
                self.report_angle_ax.plot(x_all[mask_model_o], opening_model[mask_model_o], color="black", lw=1.8, ls="--", label=angle_label)
                if np.isfinite(sigma_theta) and sigma_theta > 0.0:
                    self.report_angle_ax.fill_between(
                        x_all[mask_model_o],
                        opening_model[mask_model_o] - sigma_theta,
                        opening_model[mask_model_o] + sigma_theta,
                        color="black",
                        alpha=0.12,
                        label="Trend ±1σ",
                    )
            if np.any(mask_resid):
                residual_values = opening_residual if result_mode == "opening_plateau" else log_width_residual
                residual_label = "opening residual" if result_mode == "opening_plateau" else "log-width residual"
                self.report_k_ax.scatter(
                    x_all[mask_resid],
                    residual_values[mask_resid],
                    s=residual_size,
                    c=residual_color,
                    alpha=0.85,
                    edgecolors="none",
                    label=residual_label,
                )
                residual_sigma = sigma_theta if result_mode == "opening_plateau" else sigma_log_width
                if np.isfinite(residual_sigma) and residual_sigma > 0.0:
                    self.report_k_ax.fill_between(
                        x_all[mask_resid],
                        -residual_sigma,
                        residual_sigma,
                        color=residual_color,
                        alpha=0.10,
                        label="Residual ±1σ",
                    )
            x_start = None
            x_end = None
            for row in rows:
                order = int(row.get("slice_order", -1))
                if order == fit_start:
                    x_start = float(row.get(x_key, float("nan")))
                if order == fit_end:
                    x_end = float(row.get(x_key, float("nan")))
            if np.isfinite(x_start if x_start is not None else np.nan) and np.isfinite(x_end if x_end is not None else np.nan):
                self.report_k_ax.axvspan(float(min(x_start, x_end)), float(max(x_start, x_end)), color="tab:gray", alpha=0.15, label="Fit range")
            if str(trend_result.get("fit_mode", "")) == "opening_plateau":
                plateau_quality = "ok" if bool(trend_result.get("plateau_passes_thresholds", False)) else "weak"
                self.trend_stats_label.setText(
                    f"usable={usable_rows} | filtered={len(excluded_rows)} | raw>beam={over_beam_rows} | "
                    f"sep={_format_float_or_na(sep_min, '.4g')}-{_format_float_or_na(sep_max, '.4g')} {distance_unit} | "
                    f"plateau={_format_float_or_na(float(trend_result.get('opening_fit_median_deg', float('nan'))), '.4g')}±"
                    f"{_format_float_or_na(float(trend_result.get('opening_fit_median_sigma_deg', float('nan'))), '.3g')} deg | "
                    f"sigma_theta={_format_float_or_na(float(trend_result.get('opening_sigma_deg', float('nan'))), '.4g')} deg | "
                    f"slope_theta={_format_float_or_na(float(trend_result.get('theta_slope_deg_per_dex', float('nan'))), '.4g')}±"
                    f"{_format_float_or_na(float(trend_result.get('theta_slope_sigma_deg_per_dex', float('nan'))), '.3g')} deg/dex | "
                    f"k_tail={_format_float_or_na(float(trend_result.get('k_fit', float('nan'))), '.4g')}±"
                    f"{_format_float_or_na(float(trend_result.get('k_sigma', float('nan'))), '.3g')} | "
                    f"quality={plateau_quality} | "
                    f"plateau slices={fit_start}-{fit_end} | "
                    f"basis={'intrinsic+raw width' if self.use_raw_width_check.isChecked() else 'intrinsic only'} | "
                    f"unstable filter={'on' if self.filter_unstable_gaussian_check.isChecked() else 'off'}"
                )
            else:
                self.trend_stats_label.setText(
                    f"usable={usable_rows} | filtered={len(excluded_rows)} | raw>beam={over_beam_rows} | "
                    f"sep={_format_float_or_na(sep_min, '.4g')}-{_format_float_or_na(sep_max, '.4g')} {distance_unit} | "
                    f"k_fit={_format_float_or_na(float(trend_result.get('k_fit', float('nan'))), '.4g')}±"
                    f"{_format_float_or_na(float(trend_result.get('k_sigma', float('nan'))), '.3g')} | "
                    f"log10(C)={_format_float_or_na(float(trend_result.get('intercept_log10', float('nan'))), '.4g')} | "
                    f"opening={_format_float_or_na(float(trend_result.get('opening_fit_median_deg', float('nan'))), '.4g')}±"
                    f"{_format_float_or_na(float(trend_result.get('opening_fit_median_sigma_deg', float('nan'))), '.3g')} deg | "
                    f"sigma_theta={_format_float_or_na(float(trend_result.get('opening_sigma_deg', float('nan'))), '.4g')} deg | "
                    f"sigma_logW={_format_float_or_na(float(trend_result.get('log_width_sigma_dex', float('nan'))), '.4g')} dex | "
                    f"fit slices={fit_start}-{fit_end} | "
                    f"basis={'intrinsic+raw width' if self.use_raw_width_check.isChecked() else 'intrinsic only'} | "
                    f"unstable filter={'on' if self.filter_unstable_gaussian_check.isChecked() else 'off'}"
                )
        else:
            self.trend_stats_label.setText(
                f"usable={usable_rows} | filtered={len(excluded_rows)} | raw>beam={over_beam_rows} | "
                f"sep={_format_float_or_na(sep_min, '.4g')}-{_format_float_or_na(sep_max, '.4g')} {distance_unit} | "
                "Trend / Report: fit range is not valid."
            )
            self.trend_result_label.setText("Trend fit result: fit range is not valid.")

        if bool(self.show_paper_model_check.isChecked()) and distance_unit == "mas":
            positive = x[np.isfinite(x) & (x > 0.0)]
            if positive.size > 0:
                z_min = max(0.02, float(np.min(positive)) * 0.9)
                z_max = max(80.0, float(np.max(positive)) * 1.05)
                z = np.logspace(np.log10(z_min), np.log10(z_max), 400)
                self.report_width_ax.plot(
                    z,
                    paper_fig7_eastern_broken_power_law(z),
                    color="black",
                    lw=2.0,
                    ls=":",
                    label="Paper Fig.7 eastern jet broken PL",
                )

        self.report_width_ax.set_xscale("log")
        self.report_width_ax.set_yscale("log")
        self.report_width_ax.set_xlabel(f"Distance from core ({distance_unit})")
        self.report_width_ax.set_ylabel(f"Width ({distance_unit})")
        self.report_angle_ax.set_xscale("log")
        self.report_angle_ax.set_xlabel(f"Distance from core ({distance_unit})")
        self.report_angle_ax.set_ylabel("Opening angle (deg)")
        self.report_peak_ax.set_xscale("log")
        self.report_peak_ax.set_yscale("log")
        self.report_peak_ax.set_xlabel(f"Distance from core ({distance_unit})")
        self.report_peak_ax.set_ylabel("Peak level")
        self.report_k_ax.set_xscale("log")
        self.report_k_ax.set_xlabel(f"Distance from core ({distance_unit})")
        if trend_result is not None and str(trend_result.get("fit_mode", "")) == "opening_plateau":
            self.report_k_ax.set_ylabel("opening residual (deg)")
        else:
            self.report_k_ax.set_ylabel("log-width residual")
        if not bool(self.auto_x_range_check.isChecked()):
            x_min = float(self.x_min_spin.value())
            x_max = float(self.x_max_spin.value())
            if np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0.0 and x_max > x_min:
                for ax in (self.report_width_ax, self.report_angle_ax, self.report_peak_ax, self.report_k_ax):
                    ax.set_xlim(x_min, x_max)
        if not bool(self.auto_width_y_range_check.isChecked()):
            y_min = float(self.width_y_min_spin.value())
            y_max = float(self.width_y_max_spin.value())
            if np.isfinite(y_min) and np.isfinite(y_max) and y_min > 0.0 and y_max > y_min:
                self.report_width_ax.set_ylim(y_min, y_max)
        for ax in (self.report_width_ax, self.report_angle_ax, self.report_peak_ax, self.report_k_ax):
            ax.grid(alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")
        if trend_result is not None:
            if str(trend_result.get("fit_mode", "")) == "opening_plateau":
                plateau_quality = "ok" if bool(trend_result.get("plateau_passes_thresholds", False)) else "weak"
                self.trend_window_details.setPlainText(
                    "Plateau mode does not use Local k Window.\n"
                    f"quality={plateau_quality}\n"
                    f"theta slope={_format_float_or_na(float(trend_result.get('theta_slope_deg_per_dex', float('nan'))), '.6g')}±"
                    f"{_format_float_or_na(float(trend_result.get('theta_slope_sigma_deg_per_dex', float('nan'))), '.3g')} deg/dex\n"
                    f"k_tail={_format_float_or_na(float(trend_result.get('k_fit', float('nan'))), '.6g')}±"
                    f"{_format_float_or_na(float(trend_result.get('k_sigma', float('nan'))), '.3g')}\n"
                    f"plateau slices={fit_start}-{fit_end}"
                )
                self.trend_result_label.setText(
                    f"plateau opening = {_format_float_or_na(float(trend_result.get('opening_fit_median_deg', float('nan'))), '.6g')}±"
                    f"{_format_float_or_na(float(trend_result.get('opening_fit_median_sigma_deg', float('nan'))), '.3g')} deg\n"
                    f"opening σ = {_format_float_or_na(float(trend_result.get('opening_sigma_deg', float('nan'))), '.6g')} deg\n"
                    f"quality = {plateau_quality} | slices = {fit_start} - {fit_end}"
                )
            else:
                ensemble = list(trend_result.get("k_window_ensemble", []) or [])
                ensemble_report = self._format_k_window_ensemble_report(ensemble)
                ensemble_summary = dict(trend_result.get("k_window_ensemble_summary", {}) or {})
                self.trend_window_details.setPlainText(ensemble_report)
                primary_window = int(trend_result.get("primary_local_k_window_points", self._effective_local_k_window_points()))
                self.trend_result_label.setText(
                    f"plot fit: w={primary_window}, slices={fit_start}-{fit_end}, "
                    f"k={_format_float_or_na(float(trend_result.get('k_fit', float('nan'))), '.5g')}±"
                    f"{_format_float_or_na(float(trend_result.get('k_sigma', float('nan'))), '.3g')}, "
                    f"θ1/2={_format_float_or_na(float(trend_result.get('half_opening_fit_median_deg', 0.5 * float(trend_result.get('opening_fit_median_deg', float('nan'))))), '.5g')}±"
                    f"{_format_float_or_na(float(trend_result.get('half_opening_fit_median_sigma_deg', float('nan'))), '.3g')} deg\n"
                    f"residual: |logR|all={_format_float_or_na(float(trend_result.get('all_log_width_residual_mean_abs_dex', float('nan'))), '.4g')} dex, "
                    f"|logR|fit={_format_float_or_na(float(trend_result.get('fit_log_width_residual_mean_abs_dex', float('nan'))), '.4g')} dex\n"
                    f"ensemble: k={_format_float_or_na(float(ensemble_summary.get('k_median', float('nan'))), '.5g')}±"
                    f"{_format_float_or_na(float(ensemble_summary.get('k_sigma', float('nan'))), '.3g')}, "
                    f"θ1/2={_format_float_or_na(float(ensemble_summary.get('half_opening_median_deg', float('nan'))), '.5g')}±"
                    f"{_format_float_or_na(float(ensemble_summary.get('half_opening_sigma_deg', float('nan'))), '.3g')} deg"
                )
        else:
            self.trend_window_details.setPlainText("Window ensemble: fit range is not valid.")
        self.report_canvas.draw_idle()

    def _on_zoom_changed(self, value: int) -> None:
        self.zoom = float(max(10, int(value))) / 10.0
        self.refresh_view(sync_sliders=True)

    def _on_pan_changed(self, _value: int) -> None:
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        px = float(self.pan_x_slider.value())
        py = float(self.pan_y_slider.value())
        cx = min_cx if max_cx <= min_cx + 1e-6 else min_cx + (px * float(max_cx - min_cx) / 1000.0)
        cy = min_cy if max_cy <= min_cy + 1e-6 else min_cy + (py * float(max_cy - min_cy) / 1000.0)
        self.view_center = (float(cx), float(cy))
        self.refresh_view(sync_sliders=False)

    def _sync_sliders(self) -> None:
        if not self._trackbars_sync_enabled:
            return
        base_h, base_w = self.base_image.shape[:2]
        zoom = float(max(1.0, self.zoom))
        view_w = int(max(16, round(float(base_w) / zoom)))
        view_h = int(max(16, round(float(base_h) / zoom)))
        cx, cy = self._clamped_view_center(self.base_image.shape[:2])
        min_cx = float(view_w) / 2.0
        max_cx = max(min_cx, float(base_w) - (float(view_w) / 2.0))
        min_cy = float(view_h) / 2.0
        max_cy = max(min_cy, float(base_h) - (float(view_h) / 2.0))
        self._trackbars_sync_enabled = False
        self.zoom_slider.setValue(int(round(self.zoom * 10.0)))
        self.pan_x_slider.setValue(0 if max_cx <= min_cx + 1e-6 else int(round((cx - min_cx) * 1000.0 / float(max_cx - min_cx))))
        self.pan_y_slider.setValue(0 if max_cy <= min_cy + 1e-6 else int(round((cy - min_cy) * 1000.0 / float(max_cy - min_cy))))
        self._trackbars_sync_enabled = True

    def _replace_or_append_pick(self, pt: Point) -> None:
        if self.core_xy is None:
            self.core_xy = (int(pt[0]), int(pt[1]))
            return
        if self.tail_xy is None:
            self.tail_xy = (int(pt[0]), int(pt[1]))
            return
        d_core = (self.core_xy[0] - pt[0]) ** 2 + (self.core_xy[1] - pt[1]) ** 2
        d_tail = (self.tail_xy[0] - pt[0]) ** 2 + (self.tail_xy[1] - pt[1]) ** 2
        if d_core <= d_tail:
            self.core_xy = (int(pt[0]), int(pt[1]))
        else:
            self.tail_xy = (int(pt[0]), int(pt[1]))

    def _clear_points(self) -> None:
        self.core_xy = None
        self.tail_xy = None
        self.ridge_xy = None
        self.ridgeline_metadata = {}
        self.width_lines_xy = []
        self.measure_result = None
        self.trend_rows = []
        self.trend_result = None
        self._auto_fit_range = None
        self._clear_slice_details_selection()
        self._update_profile_plot()
        self._clear_trend_report_plot(
            stats="Trend / Report: measure widths first.",
            result="Trend fit result: measure widths first.",
            details="Window ensemble: measure widths first",
        )
        self.refresh_view(sync_sliders=False)

    def _remove_nearest_pick(self, pt: Optional[Point]) -> None:
        candidates: List[Tuple[str, Point]] = []
        if self.core_xy is not None:
            candidates.append(("core", self.core_xy))
        if self.tail_xy is not None:
            candidates.append(("tail", self.tail_xy))
        if not candidates:
            return
        if pt is None:
            name = candidates[-1][0]
        else:
            dists = [
                ((cand[1][0] - pt[0]) ** 2 + (cand[1][1] - pt[1]) ** 2, cand[0])
                for cand in candidates
            ]
            dists.sort(key=lambda item: item[0])
            name = dists[0][1]
        if name == "core":
            self.core_xy = None
        elif name == "tail":
            self.tail_xy = None
        self.ridge_xy = None
        self.ridgeline_metadata = {}
        self.width_lines_xy = []
        self.measure_result = None
        self.trend_rows = []
        self.trend_result = None
        self._auto_fit_range = None
        self._clear_slice_details_selection()
        self._update_profile_plot()
        self._clear_trend_report_plot(
            stats="Trend / Report: measure widths first.",
            result="Trend fit result: measure widths first.",
            details="Window ensemble: measure widths first",
        )

    def _on_left_press(self, event: QMouseEvent) -> None:
        display_pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if display_pt is None:
            return
        pt = self._display_to_analysis_point(display_pt)
        if self.measure_result is not None:
            self.pending_slice_click_xy = (int(pt[0]), int(pt[1]))
            self.selected_slice_record_index = None
            self._update_slice_details_controls(
                f"Details: staged map point ({int(pt[0])}, {int(pt[1])}). Press Details to inspect the nearest slice."
            )
            self.hover_point = pt
            self.refresh_view(sync_sliders=False)
            return
        self._replace_or_append_pick(pt)
        self.hover_point = pt
        self.ridge_xy = None
        self.width_lines_xy = []
        self.measure_result = None
        self.trend_rows = []
        self.trend_result = None
        self._auto_fit_range = None
        self._clear_slice_details_selection()
        self._update_profile_plot()
        self._clear_trend_report_plot(
            stats="Trend / Report: measure widths first.",
            result="Trend fit result: measure widths first.",
            details="Window ensemble: measure widths first",
        )
        self.refresh_view(sync_sliders=False)

    def _on_move(self, event: QMouseEvent) -> None:
        display_pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        self.hover_point = None if display_pt is None else self._display_to_analysis_point(display_pt)
        self.refresh_view(sync_sliders=False)

    def _on_right_press(self, event: QMouseEvent) -> None:
        display_pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        pt = None if display_pt is None else self._display_to_analysis_point(display_pt)
        self._remove_nearest_pick(pt)
        self.refresh_view(sync_sliders=False)

    def _on_wheel(self, event: QWheelEvent) -> None:
        delta = int(event.angleDelta().y())
        if delta == 0:
            return
        display_pt = self._display_to_image_point(event.pos(), self.base_image.shape[:2])
        if display_pt is not None:
            self.view_center = (float(display_pt[0]), float(display_pt[1]))
        self.zoom = float(np.clip((self.zoom * 10.0) + (1 if delta > 0 else -1), 10.0, 200.0)) / 10.0
        self.refresh_view(sync_sliders=True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.max_panel_w = int(max(220, self.width() - 80))
        self.max_panel_h = int(max(220, self.height() - 340))
        self.refresh_view(sync_sliders=False)

    def _extract_ridgeline(self) -> None:
        smooth_level = int(self.ridge_smooth_spin.value())
        smooth_window = max(1, 1 + (4 * max(0, smooth_level)))
        bspline_smoothing = float(max(0, smooth_level) ** 2) * 1.5
        mode = str(self.ridge_mode_combo.currentData() or "mojave_polar")
        auto_pick_info: Dict[str, object] = {}
        if mode.strip().lower() in {"mojave", "mojave_polar", "polar"} and (self.core_xy is None or self.tail_xy is None):
            try:
                auto_pick_info = infer_mojave_polar_core_tail(
                    self.flux_map,
                    self.support_mask,
                    core_xy=self.core_xy,
                    tail_xy=self.tail_xy,
                )
                self.core_xy = tuple(map(int, auto_pick_info["core_xy"]))
                self.tail_xy = tuple(map(int, auto_pick_info["tail_xy"]))
            except Exception as exc:
                self.stats_label.setText(f"Ridgeline auto-pick failed: {exc}")
                return
        if self.core_xy is None or self.tail_xy is None:
            self.stats_label.setText("Ridgeline: select core and tail first.")
            return
        polar_smoothing_px = float(max(0, smooth_level)) * 0.25
        polar_threshold = None
        l0_rms_flux = _safe_float(
            self.analysis_context.get("l0_rms_flux", self.analysis_context.get("background_flux", float("nan")))
        )
        if np.isfinite(l0_rms_flux) and l0_rms_flux > 0.0:
            polar_threshold = float(3.0 * l0_rms_flux)
        polar_step_px = 4.0
        scale_mas_per_px = _safe_float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        if np.isfinite(scale_mas_per_px) and scale_mas_per_px > 0.0:
            polar_step_px = float(0.05 / scale_mas_per_px)
        else:
            beam_px = _safe_float(self.calibration_context.get("beam_size_px", float("nan")))
            if np.isfinite(beam_px) and beam_px > 0.0:
                polar_step_px = float(beam_px / 16.0)
        polar_step_px = float(np.clip(polar_step_px, 1.0, 16.0))
        try:
            result = extract_ridgeline(
                flux_map=self.flux_map,
                support_mask=self.support_mask,
                core_xy=self.core_xy,
                tail_xy=self.tail_xy,
                region_depth_map=self.region_depth_map,
                snap_radius=int(self.snap_radius_spin.value()),
                smooth_window=smooth_window,
                bspline_smoothing=bspline_smoothing,
                resnap_radius=2,
                include_debug_maps=False,
                mode=mode,
                polar_step_px=polar_step_px,
                polar_pa_step_deg=0.5,
                polar_sector_width_deg=80.0,
                polar_component_mode="peak",
                polar_threshold=polar_threshold,
                polar_smoothing_px=polar_smoothing_px,
            )
        except Exception as exc:
            self.stats_label.setText(f"Ridgeline failed: {exc}")
            return
        self.core_xy = tuple(map(int, result["core_xy"]))
        self.tail_xy = tuple(map(int, result["tail_xy"]))
        self.ridge_xy = np.asarray(result["ridge_xy"], dtype=np.int32)
        self.ridgeline_metadata = {
            "extraction_mode": str(result.get("extraction_mode", mode)),
            "snapped_core_xy": list(result.get("snapped_core_xy", [])),
            "snapped_tail_xy": list(result.get("snapped_tail_xy", [])),
            "raw_sample_count": int(result.get("raw_sample_count", len(result.get("raw_ridge_xy", [])))),
            "filtered_sample_count": int(result.get("filtered_sample_count", len(result.get("raw_ridge_xy", [])))),
            "outlier_filtered_count": int(result.get("outlier_filtered_count", 0)),
            "polar_parameters": dict(result.get("polar_parameters", {}) or {}),
        }
        if auto_pick_info:
            self.ridgeline_metadata["auto_picks"] = {
                "core_source": str(auto_pick_info.get("core_source", "")),
                "tail_source": str(auto_pick_info.get("tail_source", "")),
                "tail_distance_px": float(auto_pick_info.get("tail_distance_px", float("nan"))),
                "core_flux": float(auto_pick_info.get("core_flux", float("nan"))),
            }
        self.width_lines_xy = []
        self.measure_result = None
        self.trend_rows = []
        self.trend_result = None
        self._auto_fit_range = None
        self._clear_slice_details_selection()
        self._update_profile_plot()
        self._clear_trend_report_plot(
            stats="Trend / Report: measure widths first.",
            result="Trend fit result: measure widths first.",
            details="Window ensemble: measure widths first",
        )
        auto_suffix = " | auto picks" if auto_pick_info else ""
        self.stats_label.setText(
            f"Ridgeline extracted ({self.ridgeline_metadata['extraction_mode']}): "
            f"{int(len(self.ridge_xy))} points | Core={self.core_xy} Tail={self.tail_xy}{auto_suffix}"
        )
        self.refresh_view(sync_sliders=False)

    def _save_result(self) -> None:
        image_path = str(self.analysis_context.get("image_path", "") or "")
        base_name = Path(image_path).stem if image_path else "ridgeline_analysis"
        default_name = f"{base_name}_ridgeline_analysis.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Ridgeline Analysis",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        flux_reconstruction = {
            "background_flux": 0.0,
            "l0_rms_flux": self.analysis_context.get("l0_rms_flux", self.analysis_context.get("background_flux", None)),
            "l1_flux": self.analysis_context.get("l1_flux", None),
            "level_ratio": self.analysis_context.get("level_ratio", None),
            "custom_contour_values": self.analysis_context.get("custom_contour_values", None),
            "sigma_px": self.analysis_context.get("smooth_sigma_px", None),
            "l0_l1_transition_mode": self.analysis_context.get("l0_l1_transition_mode", "gaussian"),
            "l0_l1_transition_width_px": self.analysis_context.get("l0_l1_transition_width_px", None),
            "l0_l1_transition_width_source": self.analysis_context.get("l0_l1_transition_width_source", None),
            "l0_l1_transition_alpha": self.analysis_context.get("l0_l1_transition_alpha", 3.0),
            "l0_l1_transition_applied_pixel_count": self.analysis_context.get(
                "l0_l1_transition_applied_pixel_count",
                None,
            ),
            "colormap": self.analysis_context.get("cmap_name", self.cmap_name),
            "log_color": self.analysis_context.get("log_enabled", self.log_enabled),
        }
        cache_error = ""
        try:
            cache_info = self._save_reconstruction_cache_sidecar(path)
            if cache_info:
                flux_reconstruction["cache_npz"] = cache_info
        except Exception as exc:
            cache_error = str(exc)
            flux_reconstruction["cache_npz_error"] = cache_error
        tangent_half_window = None
        if isinstance(self.measure_result, dict) and self.measure_result.get("tangent_half_window", None) is not None:
            try:
                tangent_half_window = int(self.measure_result.get("tangent_half_window"))
            except Exception:
                tangent_half_window = None
        baseline_l1_flux = self._gaussian_baseline_l1_flux()
        baseline_noise_sigma_flux = self._gaussian_baseline_noise_sigma_flux(baseline_l1_flux)
        payload = {
            "image_path": image_path,
            "loaded_replay_json_path": str(self.analysis_context.get("loaded_replay_json_path", "") or ""),
            "roi_bbox_xywh": self.analysis_context.get("roi_bbox_xywh", None),
            "roi_points_xy": self.analysis_context.get("roi_points_xy", []),
            "binary_prep_mode": self.analysis_context.get("binary_prep_mode", None),
            "replay_snapshot": self.analysis_context.get("replay_snapshot", None),
            "flux_reconstruction": flux_reconstruction,
            "calibration": dict(self.calibration_context),
            "ridgeline": {
                "core_xy": None if self.core_xy is None else [int(self.core_xy[0]), int(self.core_xy[1])],
                "tail_xy": None if self.tail_xy is None else [int(self.tail_xy[0]), int(self.tail_xy[1])],
                "ridgeline_xy": []
                if self.ridge_xy is None
                else [[int(pt[0]), int(pt[1])] for pt in np.asarray(self.ridge_xy, dtype=np.int32).tolist()],
                "extraction_mode": str(self.ridgeline_metadata.get("extraction_mode", self.ridge_mode_combo.currentData() or "mojave_polar")),
                "snapped_core_xy": self.ridgeline_metadata.get("snapped_core_xy", []),
                "snapped_tail_xy": self.ridgeline_metadata.get("snapped_tail_xy", []),
                "raw_sample_count": int(self.ridgeline_metadata.get("raw_sample_count", 0) or 0),
                "filtered_sample_count": int(self.ridgeline_metadata.get("filtered_sample_count", 0) or 0),
                "outlier_filtered_count": int(self.ridgeline_metadata.get("outlier_filtered_count", 0) or 0),
                "polar_parameters": dict(self.ridgeline_metadata.get("polar_parameters", {}) or {}),
                "snap_radius": int(self.snap_radius_spin.value()),
                "slice_count": int(self.slice_count_spin.value()),
                "trim_core_percent": float(self.trim_core_percent_spin.value()),
                "trim_tail_percent": float(self.trim_tail_percent_spin.value()),
                "ridgeline_smooth": int(self.ridge_smooth_spin.value()),
                "tangent_half_window": tangent_half_window,
                "tangent_half_window_mode": "auto",
                "profile_step_px": float(self.profile_step_spin.value()),
                "effective_profile_step_px": float(
                    _safe_float(self.measure_result.get("profile_step_px", float("nan")))
                    if isinstance(self.measure_result, dict)
                    else float("nan")
                ),
                "profile_step_mode": str(
                    self.measure_result.get("profile_step_mode", "")
                    if isinstance(self.measure_result, dict)
                    else ""
                ),
                "slice_sampling_mode": str(
                    self.measure_result.get("slice_sampling_mode", "")
                    if isinstance(self.measure_result, dict)
                    else ""
                ),
                "slice_sampling_step_mas": float(
                    _safe_float(self.measure_result.get("slice_sampling_step_mas", float("nan")))
                    if isinstance(self.measure_result, dict)
                    else float("nan")
                ),
                "slice_sampling_actual_median_spacing_mas": float(
                    _safe_float(self.measure_result.get("slice_sampling_actual_median_spacing_mas", float("nan")))
                    if isinstance(self.measure_result, dict)
                    else float("nan")
                ),
                "pa_sweep_error": bool(self.pa_sweep_error_check.isChecked()),
                "pa_sweep_range_deg": 15.0,
                "pa_sweep_step_deg": 1.0,
                "gaussian_baseline_mode": str(
                    self.measure_result.get("gaussian_baseline_mode", "fixed_zero")
                    if isinstance(self.measure_result, dict)
                    else "fixed_zero"
                ),
                "gaussian_baseline_l1_flux": float(
                    _safe_float(self.measure_result.get("gaussian_baseline_l1_flux", baseline_l1_flux), baseline_l1_flux)
                    if isinstance(self.measure_result, dict)
                    else baseline_l1_flux
                ),
                "gaussian_baseline_noise_sigma_flux": float(
                    _safe_float(
                        self.measure_result.get("gaussian_baseline_noise_sigma_flux", baseline_noise_sigma_flux),
                        baseline_noise_sigma_flux,
                    )
                    if isinstance(self.measure_result, dict)
                    else baseline_noise_sigma_flux
                ),
                "core_separation_value": float(self.core_separation_spin.value()),
                "core_separation_unit": str(self._core_separation_unit),
                "core_separation_px": float(self._effective_core_separation_px()),
            },
            "measurement_result": self.measure_result,
            "trend_report": {
                "use_raw_width": bool(self.use_raw_width_check.isChecked()),
                "filter_unstable_gaussian": bool(self.filter_unstable_gaussian_check.isChecked()),
                "fit_mode": str(self._trend_fit_mode()),
                "show_paper_model": bool(self.show_paper_model_check.isChecked()),
                "auto_x_range": bool(self.auto_x_range_check.isChecked()),
                "x_min_value": float(self.x_min_spin.value()),
                "x_max_value": float(self.x_max_spin.value()),
                "x_range_unit": str(self._trend_cut_unit),
                "auto_width_y_range": bool(self.auto_width_y_range_check.isChecked()),
                "width_y_min_value": float(self.width_y_min_spin.value()),
                "width_y_max_value": float(self.width_y_max_spin.value()),
                "width_y_range_unit": str(self._trend_cut_unit),
                "local_k_window_points": int(self.local_k_window_spin.value()),
                "cut_min_separation_value": float(self.trend_cut_min_sep_spin.value()),
                "cut_min_separation_unit": str(self._trend_cut_unit),
                "cut_max_separation_value": float(self.trend_cut_max_sep_spin.value()),
                "cut_max_separation_unit": str(self._trend_cut_unit),
                "auto_fit_start_slice": None if self._auto_fit_range is None else int(self._auto_fit_range[0]),
                "auto_fit_end_slice": None if self._auto_fit_range is None else int(self._auto_fit_range[1]),
                "fit_start_slice": int(self.fit_start_spin.value()),
                "fit_end_slice": int(self.fit_end_spin.value()),
                "trend_result": self.trend_result,
            },
        }
        save_analysis_session(path, payload)
        suffix = "" if not cache_error else f" | reconstruction cache failed: {cache_error}"
        self.stats_label.setText(f"Saved analysis: {path}{suffix}")

    def _save_reconstruction_cache_sidecar(self, analysis_json_path: str) -> Dict[str, object]:
        cache_path = default_reconstruction_cache_path(analysis_json_path)
        metadata = {
            "source": "ibae_level_map_reconstruction",
            "image_path": str(self.analysis_context.get("image_path", "") or ""),
            "loaded_replay_json_path": str(self.analysis_context.get("loaded_replay_json_path", "") or ""),
            "roi_bbox_xywh": self.analysis_context.get("roi_bbox_xywh", None),
            "background_flux": 0.0,
            "l0_rms_flux": self.analysis_context.get("l0_rms_flux", self.analysis_context.get("background_flux", None)),
            "l1_flux": self.analysis_context.get("l1_flux", None),
            "level_ratio": self.analysis_context.get("level_ratio", None),
            "custom_contour_values": self.analysis_context.get("custom_contour_values", None),
            "smooth_sigma_px": self.analysis_context.get("smooth_sigma_px", None),
            "l0_l1_transition_mode": self.analysis_context.get("l0_l1_transition_mode", "gaussian"),
            "l0_l1_transition_width_px": self.analysis_context.get("l0_l1_transition_width_px", None),
            "l0_l1_transition_width_source": self.analysis_context.get("l0_l1_transition_width_source", None),
            "l0_l1_transition_alpha": self.analysis_context.get("l0_l1_transition_alpha", 3.0),
            "l0_l1_transition_applied_pixel_count": self.analysis_context.get(
                "l0_l1_transition_applied_pixel_count",
                None,
            ),
            "cmap_name": str(self.cmap_name),
            "log_enabled": bool(self.log_enabled),
        }
        saved_meta = save_reconstruction_cache(
            str(cache_path),
            flux_map=self.flux_map,
            valid_mask=self.roi_mask,
            region_depth_map=self.region_depth_map,
            roi_mask=self.roi_mask,
            thin_mask=self.thin_mask,
            metadata=metadata,
        )
        try:
            rel_path = os.path.relpath(str(cache_path), start=str(Path(analysis_json_path).parent))
        except Exception:
            rel_path = str(cache_path)
        return {
            "path": rel_path,
            "format": saved_meta.get("format", ""),
            "version": int(saved_meta.get("version", 0) or 0),
            "shape": saved_meta.get("shape", None),
            "region_depth_hash": saved_meta.get("region_depth_hash", None),
            "roi_mask_hash": saved_meta.get("roi_mask_hash", None),
            "thin_mask_hash": saved_meta.get("thin_mask_hash", None),
        }

    def apply_loaded_analysis_payload(self, payload: Dict[str, object], source_path: str = "") -> None:
        loaded = dict(payload or {})
        calibration = loaded.get("calibration", {})
        if isinstance(calibration, dict):
            self.calibration_context = dict(calibration)
        self._update_distance_control_labels(convert_values=False)
        ridgeline = loaded.get("ridgeline", {})
        if isinstance(ridgeline, dict):
            core_xy = ridgeline.get("core_xy", None)
            tail_xy = ridgeline.get("tail_xy", None)
            ridge_xy = ridgeline.get("ridgeline_xy", [])
            if isinstance(core_xy, (list, tuple)) and len(core_xy) >= 2:
                self.core_xy = (int(core_xy[0]), int(core_xy[1]))
            if isinstance(tail_xy, (list, tuple)) and len(tail_xy) >= 2:
                self.tail_xy = (int(tail_xy[0]), int(tail_xy[1]))
            if isinstance(ridge_xy, list) and len(ridge_xy) > 0:
                self.ridge_xy = np.asarray([[int(pt[0]), int(pt[1])] for pt in ridge_xy if isinstance(pt, (list, tuple)) and len(pt) >= 2], dtype=np.int32)
            else:
                self.ridge_xy = None
            extraction_mode = str(ridgeline.get("extraction_mode", "") or "")
            if extraction_mode:
                for idx in range(self.ridge_mode_combo.count()):
                    if str(self.ridge_mode_combo.itemData(idx)) == extraction_mode:
                        self.ridge_mode_combo.setCurrentIndex(idx)
                        break
            self.ridgeline_metadata = {
                "extraction_mode": extraction_mode or str(self.ridge_mode_combo.currentData() or "mojave_polar"),
                "snapped_core_xy": ridgeline.get("snapped_core_xy", []),
                "snapped_tail_xy": ridgeline.get("snapped_tail_xy", []),
                "raw_sample_count": int(ridgeline.get("raw_sample_count", 0) or 0),
                "filtered_sample_count": int(ridgeline.get("filtered_sample_count", 0) or 0),
                "outlier_filtered_count": int(ridgeline.get("outlier_filtered_count", 0) or 0),
                "polar_parameters": dict(ridgeline.get("polar_parameters", {}) or {}),
            }
            if ridgeline.get("snap_radius", None) is not None:
                self.snap_radius_spin.setValue(int(ridgeline.get("snap_radius")))
            if ridgeline.get("slice_count", None) is not None:
                self.slice_count_spin.setValue(int(ridgeline.get("slice_count")))
            if ridgeline.get("trim_core_percent", None) is not None:
                self.trim_core_percent_spin.setValue(float(ridgeline.get("trim_core_percent")))
            elif ridgeline.get("trim_percent", None) is not None:
                self.trim_core_percent_spin.setValue(float(ridgeline.get("trim_percent")))
            if ridgeline.get("trim_tail_percent", None) is not None:
                self.trim_tail_percent_spin.setValue(float(ridgeline.get("trim_tail_percent")))
            elif ridgeline.get("trim_percent", None) is not None:
                self.trim_tail_percent_spin.setValue(float(ridgeline.get("trim_percent")))
            if ridgeline.get("ridgeline_smooth", None) is not None:
                self.ridge_smooth_spin.setValue(int(ridgeline.get("ridgeline_smooth")))
            if ridgeline.get("profile_step_px", None) is not None:
                self.profile_step_spin.setValue(float(ridgeline.get("profile_step_px")))
            if ridgeline.get("pa_sweep_error", None) is not None:
                self.pa_sweep_error_check.setChecked(bool(ridgeline.get("pa_sweep_error")))
            core_sep_unit = str(ridgeline.get("core_separation_unit", self._core_separation_unit) or self._core_separation_unit)
            core_sep_value = ridgeline.get("core_separation_value", None)
            if core_sep_value is None and ridgeline.get("core_separation_px", None) is not None:
                core_sep_value = self._convert_distance_value(
                    float(ridgeline.get("core_separation_px", 0.0)),
                    "px",
                    self._core_separation_unit,
                )
            if core_sep_value is not None:
                self.core_separation_spin.blockSignals(True)
                self.core_separation_spin.setValue(
                    float(
                        self._convert_distance_value(
                            float(core_sep_value),
                            core_sep_unit,
                            self._core_separation_unit,
                        )
                    )
                )
                self.core_separation_spin.blockSignals(False)
            if self.ridge_xy is not None and len(self.ridge_xy) > 0:
                self.stats_label.setText(
                    f"Ridgeline loaded from JSON ({self.ridgeline_metadata['extraction_mode']}). "
                    "Click Extract Ridgeline to recompute with the selected algorithm."
                )
        measure_result = loaded.get("measurement_result", None)
        self.measure_result = dict(measure_result) if isinstance(measure_result, dict) else None
        if self.measure_result is not None:
            self._apply_core_separation_to_current_measure_result()
        trend_report = loaded.get("trend_report", {})
        self._pending_loaded_trend_report = dict(trend_report) if isinstance(trend_report, dict) else None
        if isinstance(trend_report, dict):
            if trend_report.get("fit_mode", None) is not None:
                idx = self.trend_fit_mode_combo.findData(str(trend_report.get("fit_mode")))
                if idx >= 0:
                    self.trend_fit_mode_combo.setCurrentIndex(idx)
                    self._update_trend_fit_mode_controls()
            if trend_report.get("auto_x_range", None) is not None:
                self.auto_x_range_check.setChecked(bool(trend_report.get("auto_x_range")))
            if trend_report.get("auto_width_y_range", None) is not None:
                self.auto_width_y_range_check.setChecked(bool(trend_report.get("auto_width_y_range")))
            cut_unit = str(trend_report.get("cut_min_separation_unit", self._trend_cut_unit) or self._trend_cut_unit)
            cut_value = trend_report.get("cut_min_separation_value", None)
            if cut_value is not None:
                self.trend_cut_min_sep_spin.blockSignals(True)
                self.trend_cut_min_sep_spin.setValue(
                    float(
                        self._convert_distance_value(
                            float(cut_value),
                            cut_unit,
                            self._trend_cut_unit,
                        )
                    )
                )
                self.trend_cut_min_sep_spin.blockSignals(False)
            cut_max_unit = str(trend_report.get("cut_max_separation_unit", self._trend_cut_unit) or self._trend_cut_unit)
            cut_max_value = trend_report.get("cut_max_separation_value", None)
            if cut_max_value is not None:
                self.trend_cut_max_sep_spin.blockSignals(True)
                self.trend_cut_max_sep_spin.setValue(
                    float(
                        self._convert_distance_value(
                            float(cut_max_value),
                            cut_max_unit,
                            self._trend_cut_unit,
                        )
                    )
                )
                self.trend_cut_max_sep_spin.blockSignals(False)
            x_unit = str(trend_report.get("x_range_unit", self._trend_cut_unit) or self._trend_cut_unit)
            x_min_value = trend_report.get("x_min_value", None)
            x_max_value = trend_report.get("x_max_value", None)
            if x_min_value is not None:
                self.x_min_spin.blockSignals(True)
                self.x_min_spin.setValue(
                    float(self._convert_distance_value(float(x_min_value), x_unit, self._trend_cut_unit))
                )
                self.x_min_spin.blockSignals(False)
            if x_max_value is not None:
                self.x_max_spin.blockSignals(True)
                self.x_max_spin.setValue(
                    float(self._convert_distance_value(float(x_max_value), x_unit, self._trend_cut_unit))
                )
                self.x_max_spin.blockSignals(False)
            y_unit = str(trend_report.get("width_y_range_unit", self._trend_cut_unit) or self._trend_cut_unit)
            y_min_value = trend_report.get("width_y_min_value", None)
            y_max_value = trend_report.get("width_y_max_value", None)
            if y_min_value is not None:
                self.width_y_min_spin.blockSignals(True)
                self.width_y_min_spin.setValue(
                    float(self._convert_distance_value(float(y_min_value), y_unit, self._trend_cut_unit))
                )
                self.width_y_min_spin.blockSignals(False)
            if y_max_value is not None:
                self.width_y_max_spin.blockSignals(True)
                self.width_y_max_spin.setValue(
                    float(self._convert_distance_value(float(y_max_value), y_unit, self._trend_cut_unit))
                )
                self.width_y_max_spin.blockSignals(False)
        auto_start = None if not isinstance(trend_report, dict) else trend_report.get("auto_fit_start_slice", None)
        auto_end = None if not isinstance(trend_report, dict) else trend_report.get("auto_fit_end_slice", None)
        if auto_start is not None and auto_end is not None:
            try:
                self._auto_fit_range = (int(auto_start), int(auto_end))
            except Exception:
                self._auto_fit_range = None
        else:
            self._auto_fit_range = None
        self.width_lines_xy = []
        if isinstance(self.measure_result, dict):
            for item in list(self.measure_result.get("width_lines_xy", [])):
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[0], (list, tuple))
                    and isinstance(item[1], (list, tuple))
                    and len(item[0]) >= 2
                    and len(item[1]) >= 2
                ):
                    self.width_lines_xy.append(
                        (
                            (int(item[0][0]), int(item[0][1])),
                            (int(item[1][0]), int(item[1][1])),
                        )
                    )
        self.analysis_context.update(
            {
                "loaded_analysis_json_path": str(source_path or self.analysis_context.get("loaded_analysis_json_path", "")),
            }
        )
        label_path = str(source_path or loaded.get("loaded_replay_json_path", "") or "")
        self.stats_label.setText(f"Loaded analysis: {label_path or 'session JSON'}")
        if self.measure_result is not None:
            self._clear_slice_details_selection("Details: loaded measurement. Click a map slice, then press Details.")
        else:
            self._clear_slice_details_selection()
        self._update_profile_plot()
        if self.measure_result is not None:
            self._update_measurement_stats_label()
            self._clear_trend_report_plot(
                stats="Trend / Report: loaded measurement. Click Apply Trend Fit.",
                result="Trend fit result: loaded measurement. Click Apply Trend Fit.",
                details="Window ensemble: loaded measurement. Click Apply Trend Fit.",
            )
        else:
            self._clear_trend_report_plot(
                stats="Trend / Report: measure widths first.",
                result="Trend fit result: measure widths first.",
                details="Window ensemble: measure widths first",
            )
        self.refresh_view(sync_sliders=False)

    def _load_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Ridgeline Analysis",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            payload = load_analysis_session(path)
        except Exception as exc:
            self.stats_label.setText(f"Load failed: {exc}")
            return
        self.apply_loaded_analysis_payload(payload, source_path=path)

    def _gaussian_baseline_l1_flux(self) -> float:
        custom_values = self.analysis_context.get("custom_contour_values", None)
        if isinstance(custom_values, (list, tuple, np.ndarray)) and len(custom_values) > 0:
            first = _safe_float(list(custom_values)[0])
            if np.isfinite(first) and first > 0.0:
                return float(first)
        l1 = _safe_float(self.analysis_context.get("l1_flux", float("nan")))
        if np.isfinite(l1) and l1 > 0.0:
            return float(l1)
        values = self.flux_map[(self.support_mask > 0) & np.isfinite(self.flux_map)]
        positive = values[values > 0.0]
        if positive.size > 0:
            return float(np.min(positive))
        return float("nan")

    def _gaussian_baseline_noise_sigma_flux(self, l1_flux: float) -> float:
        l0_rms = _safe_float(
            self.analysis_context.get(
                "l0_rms_flux",
                self.analysis_context.get("background_flux", float("nan")),
            )
        )
        if np.isfinite(l0_rms) and l0_rms > 0.0:
            return float(l0_rms)
        if np.isfinite(l1_flux) and l1_flux > 0.0:
            return float(l1_flux / 3.0)
        return float("nan")

    def _measure_widths(self) -> None:
        if self.ridge_xy is None or len(self.ridge_xy) < 5:
            self.stats_label.setText("Width measurement: extract ridgeline first.")
            return
        scale = self.calibration_context.get("scale_mas_per_px", float("nan"))
        beam_major = self.calibration_context.get("beam_major_mas", float("nan"))
        beam_minor = self.calibration_context.get("beam_minor_mas", float("nan"))
        pa_sweep_enabled = bool(self.pa_sweep_error_check.isChecked())
        pa_sweep_cache = self.measure_result if pa_sweep_enabled and isinstance(self.measure_result, dict) else None
        baseline_l1_flux = self._gaussian_baseline_l1_flux()
        baseline_noise_sigma_flux = self._gaussian_baseline_noise_sigma_flux(baseline_l1_flux)
        self.measure_button.setEnabled(False)
        self.progress_label.setText("Measure FWHM: preparing...")
        QApplication.processEvents()
        last_progress_update = [0.0]

        def _progress(stage: str, done: int, total: int) -> None:
            total_i = int(max(0, total))
            done_i = int(max(0, min(done, total_i))) if total_i > 0 else int(max(0, done))
            now = time.monotonic()
            if done_i > 0 and done_i < total_i and (now - last_progress_update[0]) < 0.5:
                return
            last_progress_update[0] = now
            if total_i > 0:
                pct = 100.0 * float(done_i) / float(total_i)
                self.progress_label.setText(f"Measure FWHM: {stage}: {done_i}/{total_i} ({pct:.0f}%)")
            else:
                self.progress_label.setText(f"Measure FWHM: {stage}")
            QApplication.processEvents()

        if pa_sweep_enabled:
            self.stats_label.setText("Width measurement: running PA sweep error (+/-15 deg, 1 deg step)...")
            QApplication.processEvents()
        try:
            result = measure_ridgeline_fwhm(
                flux_map=self.flux_map,
                support_mask=self.support_mask,
                ridge_xy=self.ridge_xy,
                n_slices=int(self.slice_count_spin.value()),
                trim_start_frac=float(self.trim_core_percent_spin.value()) / 100.0,
                trim_end_frac=float(self.trim_tail_percent_spin.value()) / 100.0,
                tangent_half_window=None,
                profile_step_px=float(self.profile_step_spin.value()),
                core_separation_px=self._effective_core_separation_px(),
                scale_mas_per_px=scale,
                beam_major_mas=beam_major,
                beam_minor_mas=beam_minor,
                pa_sweep_enabled=pa_sweep_enabled,
                pa_sweep_range_deg=15.0,
                pa_sweep_step_deg=1.0,
                pa_sweep_cache=pa_sweep_cache,
                gaussian_baseline_mode="fixed_zero",
                gaussian_baseline_l1_flux=baseline_l1_flux,
                gaussian_baseline_noise_sigma_flux=baseline_noise_sigma_flux,
                progress_callback=_progress,
            )
        except Exception as exc:
            self.stats_label.setText(f"Width measurement failed: {exc}")
            self.progress_label.setText("Measure FWHM: failed")
            return
        finally:
            self.measure_button.setEnabled(True)
        self.measure_result = result
        self._auto_fit_range = None
        self.width_lines_xy = list(result.get("width_lines_xy", []))
        self._clear_slice_details_selection("Details: FWHM measured. Click a map slice, then press Details.")
        self._update_measurement_stats_label(result)
        self.progress_label.setText(
            f"Measure FWHM: complete ({int(result.get('valid_count', 0))}/{int(result.get('slice_count', result.get('valid_count', 0)))} valid)"
        )
        self._update_profile_plot()
        self._clear_trend_report_plot(
            stats="Trend / Report: FWHM measured. Click Apply Trend Fit.",
            result="Trend fit result: FWHM measured. Click Apply Trend Fit.",
            details="Window ensemble: FWHM measured. Click Apply Trend Fit.",
        )
        self.refresh_view(sync_sliders=False)

    def _update_profile_plot(self) -> None:
        self.profile_ax.clear()
        self.angle_ax.clear()
        if self.measure_result is None:
            self.profile_ax.set_title("Width Result")
            self.profile_ax.set_xlabel("Distance from Core (mas)")
            self.profile_ax.set_ylabel("FWHM (px)")
            self.profile_ax.grid(alpha=0.25)
            self.angle_ax.set_title("Opening Angle")
            self.angle_ax.set_xlabel("Distance from Core (mas)")
            self.angle_ax.set_ylabel("Opening angle (deg)")
            self.angle_ax.grid(alpha=0.25)
            self.profile_canvas.draw_idle()
            return
        dist_px = np.asarray(
            self.measure_result.get(
                "distance_from_core_px",
                self.measure_result.get("distance_along_ridge_px", np.array([], dtype=np.float32)),
            ),
            dtype=np.float32,
        )
        fwhm_px = np.asarray(self.measure_result.get("fwhm_px", np.array([], dtype=np.float32)), dtype=np.float32)
        dist_mas = np.asarray(self.measure_result.get("distance_from_core_mas", np.array([], dtype=np.float32)), dtype=np.float32)
        fwhm_mas = np.asarray(self.measure_result.get("fwhm_mas", np.array([], dtype=np.float32)), dtype=np.float32)
        fwhm_sigma_px = np.asarray(self.measure_result.get("fwhm_sigma_px", np.array([], dtype=np.float32)), dtype=np.float32)
        fwhm_sigma_mas = np.asarray(self.measure_result.get("fwhm_sigma_mas", np.array([], dtype=np.float32)), dtype=np.float32)
        intrinsic_px = np.asarray(self.measure_result.get("intrinsic_fwhm_px", np.array([], dtype=np.float32)), dtype=np.float32)
        intrinsic_mas = np.asarray(self.measure_result.get("intrinsic_fwhm_mas", np.array([], dtype=np.float32)), dtype=np.float32)
        intrinsic_sigma_px = np.asarray(self.measure_result.get("intrinsic_fwhm_sigma_px", np.array([], dtype=np.float32)), dtype=np.float32)
        intrinsic_sigma_mas = np.asarray(self.measure_result.get("intrinsic_fwhm_sigma_mas", np.array([], dtype=np.float32)), dtype=np.float32)
        opening_angle_deg = np.asarray(self.measure_result.get("opening_angle_deg", np.array([], dtype=np.float32)), dtype=np.float32)
        opening_angle_sigma_deg = np.asarray(self.measure_result.get("opening_angle_sigma_deg", np.array([], dtype=np.float32)), dtype=np.float32)
        intrinsic_opening_angle_deg = np.asarray(
            self.measure_result.get("intrinsic_opening_angle_deg", np.array([], dtype=np.float32)),
            dtype=np.float32,
        )
        intrinsic_opening_angle_sigma_deg = np.asarray(
            self.measure_result.get("intrinsic_opening_angle_sigma_deg", np.array([], dtype=np.float32)),
            dtype=np.float32,
        )
        scale = float(self.calibration_context.get("scale_mas_per_px", float("nan")))
        x = _mas_axis_values_from_measurement(dist_px, dist_mas, scale).astype(np.float32)
        x_label = "Distance from Core (mas)"
        use_y_mas = np.any(np.isfinite(fwhm_mas))
        if not use_y_mas and np.isfinite(scale) and scale > 0.0:
            fwhm_mas = (fwhm_px * float(scale)).astype(np.float32)
            fwhm_sigma_mas = (fwhm_sigma_px * float(scale)).astype(np.float32)
            intrinsic_mas = (intrinsic_px * float(scale)).astype(np.float32)
            intrinsic_sigma_mas = (intrinsic_sigma_px * float(scale)).astype(np.float32)
            use_y_mas = True
        y = fwhm_mas if use_y_mas else fwhm_px
        y_sigma = fwhm_sigma_mas if use_y_mas else fwhm_sigma_px
        intrinsic_y = intrinsic_mas if use_y_mas else intrinsic_px
        intrinsic_y_sigma = intrinsic_sigma_mas if use_y_mas else intrinsic_sigma_px
        y_label = "FWHM (mas)" if use_y_mas else "FWHM (px)"

        mask_obs = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
        mask_intr = np.isfinite(x) & np.isfinite(intrinsic_y) & (x > 0.0) & (intrinsic_y > 0.0)
        mask_open = np.isfinite(x) & np.isfinite(opening_angle_deg) & (x > 0.0) & (opening_angle_deg > 0.0)
        mask_intr_open = (
            np.isfinite(x)
            & np.isfinite(intrinsic_opening_angle_deg)
            & (x > 0.0)
            & (intrinsic_opening_angle_deg > 0.0)
        )

        if np.any(mask_obs):
            self.profile_ax.errorbar(
                x[mask_obs],
                y[mask_obs],
                yerr=y_sigma[mask_obs] if y_sigma.shape == y.shape else None,
                color="tab:orange",
                lw=1.8,
                marker="o",
                ms=3,
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                capthick=1.0,
                zorder=3,
                label="FWHM",
            )
        if np.any(mask_intr):
            self.profile_ax.errorbar(
                x[mask_intr],
                intrinsic_y[mask_intr],
                yerr=intrinsic_y_sigma[mask_intr] if intrinsic_y_sigma.shape == intrinsic_y.shape else None,
                color="tab:cyan",
                lw=1.2,
                marker="s",
                ms=2,
                ecolor="dimgray",
                elinewidth=1.0,
                capsize=3,
                capthick=0.9,
                zorder=2,
                label="Intrinsic FWHM (beam-deconvolved)",
            )
        if np.any(mask_open):
            self.angle_ax.errorbar(
                x[mask_open],
                opening_angle_deg[mask_open],
                yerr=opening_angle_sigma_deg[mask_open] if opening_angle_sigma_deg.shape == opening_angle_deg.shape else None,
                color="tab:orange",
                lw=1.4,
                marker="o",
                ms=2,
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
                capthick=1.0,
                zorder=3,
                label="Opening angle",
            )
        if np.any(mask_intr_open):
            self.angle_ax.errorbar(
                x[mask_intr_open],
                intrinsic_opening_angle_deg[mask_intr_open],
                yerr=intrinsic_opening_angle_sigma_deg[mask_intr_open] if intrinsic_opening_angle_sigma_deg.shape == intrinsic_opening_angle_deg.shape else None,
                color="tab:cyan",
                lw=1.2,
                marker="s",
                ms=2,
                ecolor="dimgray",
                elinewidth=1.0,
                capsize=3,
                capthick=0.9,
                zorder=2,
                label="Intrinsic opening angle (beam-deconvolved)",
            )

        self.profile_ax.set_title("Ridgeline Transverse FWHM")
        self.profile_ax.set_xlabel(x_label)
        self.profile_ax.set_ylabel(y_label)
        self.profile_ax.set_xscale("log")
        self.profile_ax.set_yscale("log")
        self.profile_ax.grid(alpha=0.25)
        self.profile_ax.legend(loc="best")
        self.angle_ax.set_title("Ridgeline Opening Angle")
        self.angle_ax.set_xlabel(x_label)
        self.angle_ax.set_ylabel("Opening angle (deg)")
        self.angle_ax.set_xscale("log")
        self.angle_ax.grid(alpha=0.25)
        if np.any(mask_open) or np.any(mask_intr_open):
            self.angle_ax.legend(loc="best")
        self.profile_canvas.draw_idle()

    def refresh_view(self, sync_sliders: bool = True) -> None:
        canvas = self.base_image.copy()
        if self.ridge_xy is not None and len(self.ridge_xy) >= 2:
            ridge_disp = self._analysis_polyline_to_display(np.asarray(self.ridge_xy, dtype=np.float64))
            if len(ridge_disp) >= 2:
                cv2.polylines(canvas, [ridge_disp], False, (0, 0, 255), 2, cv2.LINE_AA)
        for p1, p2 in self.width_lines_xy:
            cv2.line(
                canvas,
                self._analysis_to_display_point(p1),
                self._analysis_to_display_point(p2),
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
        selected = self._selected_slice_detail_record()
        if selected is not None:
            line = self._record_width_line_xy(selected)
            if line is not None:
                p1, p2 = line
                cv2.line(
                    canvas,
                    self._analysis_to_display_point(p1),
                    self._analysis_to_display_point(p2),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            center = self._record_center_xy(selected)
            if center is not None:
                cv2.circle(
                    canvas,
                    self._analysis_to_display_point(center),
                    4,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
        if self.pending_slice_click_xy is not None:
            cv2.drawMarker(
                canvas,
                self._analysis_to_display_point(self.pending_slice_click_xy),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if self.core_xy is not None:
            cv2.circle(canvas, self._analysis_to_display_point(self.core_xy), 3, (0, 255, 0), -1, cv2.LINE_AA)
        if self.tail_xy is not None:
            cv2.circle(canvas, self._analysis_to_display_point(self.tail_xy), 3, (0, 255, 255), -1, cv2.LINE_AA)
        disp = self._render_zoom_view(canvas)
        _set_label_image(self.label, disp)
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))
        if sync_sliders:
            self._sync_sliders()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._extract_ridgeline()
            return
        if key == Qt.Key_M:
            self._measure_widths()
            return
        if key in (Qt.Key_C, Qt.Key_R):
            self._clear_points()
            return
        super().keyPressEvent(event)

    def get_result_payload(self) -> Dict[str, object]:
        return {
            "core_xy": None if self.core_xy is None else [int(self.core_xy[0]), int(self.core_xy[1])],
            "tail_xy": None if self.tail_xy is None else [int(self.tail_xy[0]), int(self.tail_xy[1])],
            "ridgeline_xy": []
            if self.ridge_xy is None
            else [[int(pt[0]), int(pt[1])] for pt in np.asarray(self.ridge_xy, dtype=np.int32).tolist()],
            "extraction_mode": str(self.ridgeline_metadata.get("extraction_mode", self.ridge_mode_combo.currentData() or "mojave_polar")),
            "ridgeline_metadata": dict(self.ridgeline_metadata),
            "measure_result": self.measure_result,
        }


class QtFluxReconstructionDialog(QDialog):
    def __init__(
        self,
        region_depth_map: np.ndarray,
        roi_mask: np.ndarray,
        thin_mask: np.ndarray,
        max_detected_level: int,
        max_width: int,
        max_height: int,
        calibration_context: Optional[Dict[str, object]] = None,
        original_roi_image: Optional[np.ndarray] = None,
        calibration_image: Optional[np.ndarray] = None,
        calibration_roi_mask: Optional[np.ndarray] = None,
        analysis_context: Optional[Dict[str, object]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Flux Reconstruction")
        self.region_depth_map = np.asarray(region_depth_map, dtype=np.int32)
        self.roi_mask = np.asarray(roi_mask, dtype=np.uint8)
        self.thin_mask = np.asarray(thin_mask, dtype=np.uint8)
        self.max_detected_level = int(max_detected_level)
        self.calibration_context = dict(calibration_context or {})
        self.analysis_context = dict(analysis_context or {})
        self.original_roi_image = original_roi_image
        self.calibration_image = None if calibration_image is None else _ensure_bgr(calibration_image)
        self.calibration_roi_mask = None if calibration_roi_mask is None else np.asarray(calibration_roi_mask, dtype=np.uint8)
        self._latest_reconstruction_payload: Optional[Dict[str, object]] = None
        self.custom_contour_values: Optional[List[float]] = None
        self.display_max_dim = int(max(512, min(1600, max(int(max_width), int(max_height)))))

        self.bg_flux_spin = QDoubleSpinBox()
        self.bg_flux_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.bg_flux_spin.setDecimals(6)
        self.bg_flux_spin.setValue(0.0)

        self.l1_flux_spin = QDoubleSpinBox()
        self.l1_flux_spin.setRange(-1_000_000_000.0, 1_000_000_000.0)
        self.l1_flux_spin.setDecimals(6)
        self.l1_flux_spin.setValue(1.0)

        self.ratio_edit = QLineEdit("2.0")
        self.ratio_hint_label = QLabel("Examples: 2, 1.41421356, sqrt(2)")
        self.ratio_hint_label.setStyleSheet("color: #666666;")
        self.contour_values_button = QPushButton("Contour Values...")
        self.contour_values_label = QLabel("Mode: ratio")
        self.contour_values_label.setStyleSheet("color: #666666;")

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.5)

        self.log_color_check = QCheckBox("Log color")
        self.log_color_check.setChecked(False)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItem("Viridis", "viridis")
        self.colormap_combo.addItem("Heatmap", "hot")
        self.colormap_combo.addItem("Rainbow", "jet")

        self.surface_max_dim_spin = QSpinBox()
        self.surface_max_dim_spin.setRange(64, 512)
        self.surface_max_dim_spin.setSingleStep(16)
        self.surface_max_dim_spin.setValue(160)

        self.stats_label = QLabel(f"Detected levels: L1..L{int(max(0, self.max_detected_level))}")
        self.formula_label = QLabel("Formula: L0 = rms floor, Lk = L1 * ratio^(k-1) for k >= 1")
        self.calibration_label = QLabel(_calibration_summary_text(self.calibration_context))

        self.two_d_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.two_d_canvas = FigureCanvas(self.two_d_fig)
        self.two_d_ax = self.two_d_fig.add_subplot(111)
        self._two_d_im = None
        self._two_d_colorbar = None
        self._two_d_contour_ready = False
        self.three_d_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.three_d_canvas = FigureCanvas(self.three_d_fig)
        self.three_d_ax = self.three_d_fig.add_subplot(111, projection="3d")
        self._three_d_surface_artist = None
        self._three_d_empty_text = None

        self._reconstruction_cache: Dict[Tuple[float, float, float, float], Dict[str, object]] = {}
        self._surface_cache: Dict[Tuple[Tuple[float, float, float, float], int], Dict[str, object]] = {}
        self._last_two_d_render_key = None
        self._last_three_d_render_key = None
        self._pending_three_d_payload = None
        self._pending_three_d_request = None
        self._three_d_dirty = True
        self._pending_loaded_ridgeline_payload: Optional[Dict[str, object]] = None

        tabs = QTabWidget()
        self.tabs = tabs
        two_d_tab = QWidget()
        two_d_layout = QVBoxLayout(two_d_tab)
        two_d_layout.addWidget(self.two_d_canvas)
        three_d_tab = QWidget()
        three_d_layout = QVBoxLayout(three_d_tab)
        three_d_layout.addWidget(self.three_d_canvas)
        tabs.addTab(two_d_tab, "2D Flux Map")
        tabs.addTab(three_d_tab, "3D Surface")

        controls = QFormLayout()
        controls.addRow("L0 / RMS", self.bg_flux_spin)
        controls.addRow("L1 Flux", self.l1_flux_spin)
        controls.addRow("Level Ratio", self.ratio_edit)
        controls.addRow("", self.ratio_hint_label)
        controls.addRow("", self.contour_values_button)
        controls.addRow("", self.contour_values_label)
        controls.addRow("Smoothing Sigma", self.sigma_spin)
        controls.addRow("Colormap", self.colormap_combo)
        controls.addRow("", self.log_color_check)
        controls.addRow("3D Max Dim", self.surface_max_dim_spin)

        calibration_button = QPushButton("Scale / Beam...")
        calibration_button.setEnabled(self._calibration_available())
        load_button = QPushButton("Load Result...")
        ridgeline_button = QPushButton("Ridgeline / FWHM")
        apply_button = QPushButton("Apply")
        close_button = QPushButton("Close")
        calibration_button.clicked.connect(self._open_calibration)
        load_button.clicked.connect(self._load_analysis_result)
        ridgeline_button.clicked.connect(self._open_ridgeline_analysis)
        apply_button.clicked.connect(self.refresh_views)
        close_button.clicked.connect(self.accept)
        self.contour_values_button.clicked.connect(self._open_contour_values_dialog)

        layout = QVBoxLayout(self)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.formula_label)
        layout.addWidget(self.calibration_label)
        layout.addLayout(controls)
        layout.addWidget(tabs)
        button_row = QHBoxLayout()
        button_row.addWidget(calibration_button)
        button_row.addWidget(load_button)
        button_row.addWidget(ridgeline_button)
        button_row.addStretch(1)
        button_row.addWidget(apply_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.tabs.currentChanged.connect(self._handle_tab_changed)
        self.resize(int(max(720, max_width)), int(max(720, max_height)))
        startup_loaded = self.analysis_context.get("startup_loaded_analysis_payload", None)
        startup_source = str(self.analysis_context.get("loaded_analysis_json_path", "") or "")
        if isinstance(startup_loaded, dict):
            self._apply_loaded_analysis_payload(startup_loaded, source_path=startup_source)
        else:
            self.refresh_views()

    def _open_contour_values_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Contour Values")
        layout = QVBoxLayout(dialog)
        expected_levels = int(max(1, self.max_detected_level))
        expected_with_peak = expected_levels + 1
        info = QLabel(
            f"Enter L1..L{expected_levels} and Peak as the last value "
            f"(total {expected_with_peak} values), one per line or separated by commas/spaces.\n"
            "Older saved inputs with only L1..Lmax are still accepted. Leave empty to return to ratio mode."
        )
        editor = QPlainTextEdit()
        if self.custom_contour_values:
            editor.setPlainText(_format_contour_values_text(self.custom_contour_values))
        else:
            try:
                ratio = float(self._ratio_value())
            except Exception:
                ratio = 2.0
            generated = [
                float(self.l1_flux_spin.value()) * (ratio ** float(level - 1))
                for level in range(1, expected_with_peak + 1)
            ]
            editor.setPlainText(_format_contour_values_text(generated))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        clear_button = buttons.addButton("Use Ratio Model", QDialogButtonBox.ResetRole)

        def _on_accept() -> None:
            text = editor.toPlainText().strip()
            if not text:
                self.custom_contour_values = None
                dialog.accept()
                return
            try:
                values = _parse_contour_values_text(text)
            except Exception as exc:
                info.setText(f"Invalid contour values: {exc}")
                info.setStyleSheet("color: #b71c1c;")
                return
            if len(values) not in (expected_levels, expected_with_peak):
                info.setText(
                    f"Need {expected_with_peak} values for L1..L{expected_levels} + Peak "
                    f"(or {expected_levels} legacy values)."
                )
                info.setStyleSheet("color: #b71c1c;")
                return
            self.custom_contour_values = [float(v) for v in values]
            dialog.accept()

        def _on_clear() -> None:
            self.custom_contour_values = None
            dialog.accept()

        buttons.accepted.connect(_on_accept)
        buttons.rejected.connect(dialog.reject)
        clear_button.clicked.connect(_on_clear)
        layout.addWidget(info)
        layout.addWidget(editor)
        layout.addWidget(buttons)
        dialog.resize(520, 420)
        if dialog.exec_() == QDialog.Accepted:
            self.refresh_views()

    def _ratio_value(self) -> float:
        parsed = _safe_eval_ratio_expression(self.ratio_edit.text())
        if parsed is None:
            raise ValueError("Invalid ratio expression.")
        return float(parsed)

    def _fallback_ratio_from_custom_values(self) -> float:
        vals = list(self.custom_contour_values or [])
        if len(vals) >= 2:
            prev = float(vals[-2])
            last = float(vals[-1])
            if prev > 0.0 and last > 0.0:
                return float(max(1e-9, last / prev))
        return 2.0

    @staticmethod
    def _color_norm_signature(color_norm, color_mode: str):
        if color_norm is None:
            return (str(color_mode), None)
        if isinstance(color_norm, mcolors.LogNorm):
            return (str(color_mode), float(color_norm.vmin), float(color_norm.vmax))
        if isinstance(color_norm, mcolors.SymLogNorm):
            return (
                str(color_mode),
                float(color_norm.vmin),
                float(color_norm.vmax),
                float(color_norm.linthresh),
            )
        return (str(color_mode), repr(color_norm))

    @staticmethod
    def _normalize_l0_l1_transition_mode(mode: object) -> str:
        mode_norm = str(mode or "gaussian").strip().lower()
        if mode_norm in ("flat", "none", "off", "disabled", "legacy"):
            return "flat"
        return "gaussian"

    def _l0_l1_transition_mode(self) -> str:
        return self._normalize_l0_l1_transition_mode(
            self.analysis_context.get("l0_l1_transition_mode", "gaussian")
        )

    def _l0_l1_transition_width_px(self) -> Optional[float]:
        value = _safe_float(self.analysis_context.get("l0_l1_transition_width_px", None))
        if np.isfinite(value) and value > 0.0:
            return float(value)
        return None

    def _l0_l1_transition_alpha(self) -> float:
        value = _safe_float(self.analysis_context.get("l0_l1_transition_alpha", 3.0), 3.0)
        if np.isfinite(value) and value > 0.0:
            return float(value)
        return 3.0

    def _reconstruction_key(self, ratio_value: float):
        custom_key = None
        if self.custom_contour_values:
            custom_key = tuple(float(v) for v in self.custom_contour_values)
        transition_width = self._l0_l1_transition_width_px()
        return (
            0.0,
            float(self.l1_flux_spin.value()),
            float(ratio_value),
            float(self.sigma_spin.value()),
            custom_key,
            self._l0_l1_transition_mode(),
            None if transition_width is None else float(transition_width),
            float(self._l0_l1_transition_alpha()),
        )

    def _build_color_norm(
        self,
        values: np.ndarray,
        log_enabled: bool,
    ):
        return _build_color_norm_for_values(values, log_enabled)

    def _reconstruction(self, ratio_value: float) -> Dict[str, object]:
        key = self._reconstruction_key(ratio_value)
        cached = self._reconstruction_cache.get(key)
        if cached is not None:
            return cached
        rec = reconstruct_flux_from_levels(
            region_depth_map=self.region_depth_map,
            roi_mask=self.roi_mask,
            background_flux=0.0,
            l1_flux=float(self.l1_flux_spin.value()),
            level_ratio=float(ratio_value),
            smooth_sigma_px=float(self.sigma_spin.value()),
            contour_values=self.custom_contour_values,
            include_target_flux_map=False,
            l0_l1_transition_mode=self._l0_l1_transition_mode(),
            l0_l1_transition_width_px=self._l0_l1_transition_width_px(),
            l0_l1_transition_alpha=self._l0_l1_transition_alpha(),
        )
        self._reconstruction_cache.clear()
        self._reconstruction_cache[key] = rec
        return rec

    def _calibration_available(self) -> bool:
        if self.calibration_image is not None:
            return True
        return bool(_analysis_image_path(self.analysis_context))

    def _calibration_inputs(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.calibration_image is not None:
            return self.calibration_image, self.calibration_roi_mask
        image_path = _analysis_image_path(self.analysis_context)
        image = _load_calibration_image_from_path(image_path)
        if image is None:
            self.stats_label.setText(f"Calibration image load failed: {image_path or 'no image path'}")
            return None, None
        return image, None

    def _open_calibration(self) -> None:
        calibration_image, calibration_roi_mask = self._calibration_inputs()
        if calibration_image is None:
            return
        dialog = QtScaleBeamDialog(
            roi_image=calibration_image,
            roi_mask=calibration_roi_mask,
            initial_state=self.calibration_context,
            max_width=int(self.size().width() * 0.9),
            max_height=int(self.size().height() * 0.9),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self.calibration_context = dialog.get_calibration()
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))

    def _open_ridgeline_analysis(self) -> None:
        if self._latest_reconstruction_payload is None:
            self.refresh_views()
        payload = dict(self._latest_reconstruction_payload or {})
        flux = np.asarray(payload.get("flux_map", np.zeros((1, 1), dtype=np.float32)), dtype=np.float32)
        if flux.size <= 1:
            return
        merged_analysis_context = dict(self.analysis_context)
        merged_analysis_context.update(
            {
                "background_flux": payload.get("background_flux", None),
                "l0_rms_flux": payload.get("l0_rms_flux", payload.get("background_flux", None)),
                "l1_flux": payload.get("l1_flux", None),
                "level_ratio": payload.get("level_ratio", None),
                "custom_contour_values": payload.get("custom_contour_values", None),
                "smooth_sigma_px": payload.get("smooth_sigma_px", None),
                "l0_l1_transition_mode": payload.get("l0_l1_transition_mode", self._l0_l1_transition_mode()),
                "l0_l1_transition_width_px": payload.get("l0_l1_transition_width_px", None),
                "l0_l1_transition_width_source": payload.get("l0_l1_transition_width_source", None),
                "l0_l1_transition_alpha": payload.get(
                    "l0_l1_transition_alpha",
                    self._l0_l1_transition_alpha(),
                ),
                "l0_l1_transition_applied_pixel_count": payload.get(
                    "l0_l1_transition_applied_pixel_count",
                    None,
                ),
                "cmap_name": payload.get("cmap_name", self._current_cmap_name()),
                "log_enabled": payload.get("log_enabled", self.log_color_check.isChecked()),
            }
        )
        dialog = QtRidgelineAnalysisDialog(
            flux_map=flux,
            region_depth_map=self.region_depth_map,
            roi_mask=self.roi_mask,
            thin_mask=self.thin_mask,
            cmap_name=str(payload.get("cmap_name", self._current_cmap_name())),
            log_enabled=bool(payload.get("log_enabled", self.log_color_check.isChecked())),
            calibration_context=self.calibration_context,
            analysis_context=merged_analysis_context,
            max_width=int(self.size().width() * 0.95),
            max_height=int(self.size().height() * 0.95),
            parent=self,
        )
        if self._pending_loaded_ridgeline_payload is not None:
            dialog.apply_loaded_analysis_payload(
                self._pending_loaded_ridgeline_payload,
                source_path=str(self._pending_loaded_ridgeline_payload.get("__source_path__", "")),
            )
            self._pending_loaded_ridgeline_payload = None
        dialog.exec_()
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))

    def _current_cmap_name(self) -> str:
        data = self.colormap_combo.currentData()
        if isinstance(data, str) and data:
            return data
        return "viridis"

    def _set_colormap_by_name(self, cmap_name: str) -> None:
        target = str(cmap_name or "").strip().lower()
        if not target:
            return
        for idx in range(self.colormap_combo.count()):
            data = str(self.colormap_combo.itemData(idx) or "").strip().lower()
            text = str(self.colormap_combo.itemText(idx) or "").strip().lower()
            if target in (data, text):
                self.colormap_combo.setCurrentIndex(idx)
                return

    def _resolve_reconstruction_cache_path(self, cache_info: Dict[str, object], source_path: str) -> Optional[Path]:
        raw_path = str(cache_info.get("path", "") or cache_info.get("npz_path", "") or "").strip()
        if not raw_path:
            return None
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if source_path:
            return Path(source_path).parent / path
        return path

    @staticmethod
    def _metadata_float_matches(value: object, expected: float, *, atol: float = 1e-9) -> bool:
        try:
            val = float(value)
        except Exception:
            return False
        return bool(np.isfinite(val) and np.isfinite(expected) and abs(val - expected) <= atol)

    @staticmethod
    def _metadata_custom_values_match(value: object, expected: Optional[Sequence[float]]) -> bool:
        expected_vals = [] if expected is None else [float(v) for v in expected]
        if value is None:
            saved_vals: List[float] = []
        elif isinstance(value, (list, tuple)):
            try:
                saved_vals = [float(v) for v in value]
            except Exception:
                return False
        else:
            return False
        if len(saved_vals) != len(expected_vals):
            return False
        if not saved_vals:
            return True
        return bool(np.allclose(np.asarray(saved_vals, dtype=float), np.asarray(expected_vals, dtype=float), rtol=0.0, atol=1e-9))

    def _reconstruction_cache_matches_current(self, metadata: Dict[str, object]) -> bool:
        shape = metadata.get("shape", None)
        if not (isinstance(shape, (list, tuple)) and len(shape) >= 2):
            return False
        if [int(shape[0]), int(shape[1])] != [int(self.region_depth_map.shape[0]), int(self.region_depth_map.shape[1])]:
            return False
        if str(metadata.get("region_depth_hash", "")) != array_sha256(self.region_depth_map):
            return False
        roi_u8 = (np.asarray(self.roi_mask, dtype=np.uint8) > 0).astype(np.uint8)
        if str(metadata.get("roi_mask_hash", "")) != array_sha256(roi_u8):
            return False
        try:
            ratio_value = float(self._ratio_value())
        except Exception:
            return False
        if not self._metadata_float_matches(metadata.get("background_flux", None), 0.0):
            return False
        l0_meta = metadata.get("l0_rms_flux", None)
        if l0_meta is not None and not self._metadata_float_matches(l0_meta, float(self.bg_flux_spin.value())):
            return False
        if not self._metadata_float_matches(metadata.get("l1_flux", None), float(self.l1_flux_spin.value())):
            return False
        if not self._metadata_float_matches(metadata.get("level_ratio", None), ratio_value):
            return False
        if not self._metadata_float_matches(metadata.get("smooth_sigma_px", None), float(self.sigma_spin.value())):
            return False
        if not self._metadata_custom_values_match(metadata.get("custom_contour_values", None), self.custom_contour_values):
            return False
        expected_mode = self._l0_l1_transition_mode()
        stored_mode = self._normalize_l0_l1_transition_mode(metadata.get("l0_l1_transition_mode", "flat"))
        if stored_mode != expected_mode:
            return False
        if expected_mode == "gaussian":
            if not self._metadata_float_matches(
                metadata.get("l0_l1_transition_alpha", 3.0),
                self._l0_l1_transition_alpha(),
            ):
                return False
            expected_width = self._l0_l1_transition_width_px()
            if expected_width is not None:
                if not self._metadata_float_matches(
                    metadata.get("l0_l1_transition_width_px", None),
                    float(expected_width),
                ):
                    return False
        return True

    def _seed_reconstruction_cache_from_arrays(
        self,
        flux_array: object,
        valid_array: object,
        cache_path: str = "",
    ) -> bool:
        flux = np.asarray(flux_array, dtype=np.float32)
        valid_u8 = (np.asarray(valid_array, dtype=np.uint8) > 0).astype(np.uint8)
        if flux.shape[:2] != self.region_depth_map.shape[:2] or valid_u8.shape[:2] != self.region_depth_map.shape[:2]:
            return False
        finite_valid = (valid_u8 > 0) & np.isfinite(flux)
        if np.any(finite_valid):
            min_flux = float(np.min(flux, where=finite_valid, initial=np.inf))
            max_flux = float(np.max(flux, where=finite_valid, initial=-np.inf))
        else:
            min_flux = float(self.bg_flux_spin.value())
            max_flux = float(self.bg_flux_spin.value())
        ratio_value = float(self._ratio_value())
        key = self._reconstruction_key(ratio_value)
        self._reconstruction_cache.clear()
        self._reconstruction_cache[key] = {
            "smoothed_flux_map": flux,
            "valid_mask": valid_u8,
            "min_flux": float(min_flux),
            "max_flux": float(max_flux),
            "value_range": (float(min_flux), float(max_flux)),
            "contour_values": None if not self.custom_contour_values else np.asarray(self.custom_contour_values, dtype=np.float32),
            "cache_npz_path": str(cache_path or ""),
        }
        return True

    def _try_use_preloaded_reconstruction_cache(self, flux_reconstruction: Dict[str, object]) -> bool:
        preloaded = self.analysis_context.get("startup_cached_reconstruction_payload", None)
        if not isinstance(preloaded, dict):
            return False
        metadata = dict(preloaded.get("metadata", {}) or {})
        if not self._reconstruction_cache_matches_current(metadata):
            return False
        cache_path = str(preloaded.get("cache_path", "") or "")
        cache_info = flux_reconstruction.get("cache_npz", None)
        if not isinstance(cache_info, dict):
            return False
        resolved = self._resolve_reconstruction_cache_path(
            cache_info,
            str(self.analysis_context.get("loaded_analysis_json_path", "") or ""),
        )
        if resolved is not None and cache_path:
            try:
                if resolved.resolve() != Path(cache_path).resolve():
                    return False
            except Exception:
                return False
        return self._seed_reconstruction_cache_from_arrays(
            preloaded.get("flux_map", []),
            preloaded.get("valid_mask", []),
            cache_path,
        )

    def _try_load_reconstruction_cache(self, flux_reconstruction: Dict[str, object], source_path: str) -> bool:
        cache_info = flux_reconstruction.get("cache_npz", None)
        if not isinstance(cache_info, dict):
            return False
        cache_path = self._resolve_reconstruction_cache_path(cache_info, source_path)
        if cache_path is None or not cache_path.exists():
            return False
        try:
            loaded = load_reconstruction_cache(str(cache_path))
            metadata = dict(loaded.get("metadata", {}) or {})
            if not self._reconstruction_cache_matches_current(metadata):
                return False
            return self._seed_reconstruction_cache_from_arrays(
                loaded.get("flux_map", []),
                loaded.get("valid_mask", []),
                str(cache_path),
            )
        except Exception:
            return False

    def _apply_loaded_analysis_payload(self, payload: Dict[str, object], source_path: str = "") -> None:
        loaded = dict(payload or {})
        flux_reconstruction = loaded.get("flux_reconstruction", {})
        if isinstance(flux_reconstruction, dict):
            l0_value = flux_reconstruction.get("l0_rms_flux", flux_reconstruction.get("background_flux", None))
            if l0_value is not None:
                self.bg_flux_spin.setValue(float(l0_value))
            if flux_reconstruction.get("l1_flux", None) is not None:
                self.l1_flux_spin.setValue(float(flux_reconstruction.get("l1_flux")))
            if flux_reconstruction.get("level_ratio", None) is not None:
                self.ratio_edit.setText(format(float(flux_reconstruction.get("level_ratio")), ".12g"))
            if flux_reconstruction.get("sigma_px", None) is not None:
                self.sigma_spin.setValue(float(flux_reconstruction.get("sigma_px")))
            if flux_reconstruction.get("colormap", None) is not None:
                self._set_colormap_by_name(str(flux_reconstruction.get("colormap")))
            if flux_reconstruction.get("log_color", None) is not None:
                self.log_color_check.setChecked(bool(flux_reconstruction.get("log_color")))
            self.analysis_context["l0_l1_transition_mode"] = self._normalize_l0_l1_transition_mode(
                flux_reconstruction.get("l0_l1_transition_mode", "gaussian")
            )
            transition_width = _safe_float(flux_reconstruction.get("l0_l1_transition_width_px", None))
            self.analysis_context["l0_l1_transition_width_px"] = (
                float(transition_width) if np.isfinite(transition_width) and transition_width > 0.0 else None
            )
            transition_alpha = _safe_float(flux_reconstruction.get("l0_l1_transition_alpha", 3.0), 3.0)
            self.analysis_context["l0_l1_transition_alpha"] = (
                float(transition_alpha) if np.isfinite(transition_alpha) and transition_alpha > 0.0 else 3.0
            )
            if flux_reconstruction.get("l0_l1_transition_width_source", None) is not None:
                self.analysis_context["l0_l1_transition_width_source"] = str(
                    flux_reconstruction.get("l0_l1_transition_width_source")
                )
            if flux_reconstruction.get("l0_l1_transition_applied_pixel_count", None) is not None:
                try:
                    self.analysis_context["l0_l1_transition_applied_pixel_count"] = int(
                        flux_reconstruction.get("l0_l1_transition_applied_pixel_count")
                    )
                except Exception:
                    self.analysis_context["l0_l1_transition_applied_pixel_count"] = None
            custom_values = flux_reconstruction.get("custom_contour_values", None)
            if isinstance(custom_values, (list, tuple)):
                parsed_values: List[float] = []
                for item in custom_values:
                    try:
                        parsed_values.append(float(item))
                    except Exception:
                        pass
                self.custom_contour_values = parsed_values if parsed_values else None
            else:
                self.custom_contour_values = None
        calibration = loaded.get("calibration", {})
        if isinstance(calibration, dict):
            self.calibration_context = dict(calibration)
            self.calibration_label.setText(_calibration_summary_text(self.calibration_context))
        self.analysis_context["loaded_analysis_json_path"] = str(source_path)
        loaded["__source_path__"] = str(source_path)
        self._pending_loaded_ridgeline_payload = loaded
        cache_used = False
        if isinstance(flux_reconstruction, dict):
            cache_used = self._try_use_preloaded_reconstruction_cache(flux_reconstruction)
            if not cache_used:
                cache_used = self._try_load_reconstruction_cache(flux_reconstruction, source_path)
        self.refresh_views()
        if source_path:
            suffix = " (cached reconstruction)" if cache_used else ""
            self.stats_label.setText(f"Loaded analysis: {source_path}{suffix}")

    def _load_analysis_result(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Ridgeline Analysis",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            payload = load_analysis_session(path)
        except Exception as exc:
            self.stats_label.setText(f"Load failed: {exc}")
            return
        self._apply_loaded_analysis_payload(payload, source_path=path)

    def _surface_payload(
        self,
        rec_key: Tuple[float, float, float, float],
        flux: np.ndarray,
    ) -> Dict[str, object]:
        max_dim = int(self.surface_max_dim_spin.value())
        key = (rec_key, max_dim)
        cached = self._surface_cache.get(key)
        if cached is not None:
            return cached
        surface_flux = np.asarray(flux, dtype=np.float32).copy()
        surface_flux[~np.isfinite(surface_flux)] = 0.0
        surf = downsample_flux_for_surface(
            flux_map=surface_flux,
            valid_mask=np.ones_like(self.roi_mask, dtype=np.uint8),
            max_dim=max_dim,
        )
        self._surface_cache.clear()
        self._surface_cache[key] = surf
        return surf

    def _two_d_display_arrays(
        self,
        flux: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        flux_arr = np.asarray(flux, dtype=np.float32)
        valid_arr = np.asarray(valid, dtype=bool)
        if np.any(valid_arr):
            ys, xs = np.nonzero(valid_arr)
            y0 = int(np.min(ys))
            y1 = int(np.max(ys)) + 1
            x0 = int(np.min(xs))
            x1 = int(np.max(xs)) + 1
        else:
            y0, x0 = 0, 0
            y1, x1 = flux_arr.shape[:2]
        h = int(max(1, y1 - y0))
        w = int(max(1, x1 - x0))
        stride = int(max(1, np.ceil(max(float(h), float(w)) / float(max(1, self.display_max_dim)))))
        disp_flux = flux_arr[y0:y1:stride, x0:x1:stride]
        disp_valid = valid_arr[y0:y1:stride, x0:x1:stride]
        thin = (np.asarray(self.thin_mask, dtype=np.uint8) > 0)
        if thin.shape[:2] != flux_arr.shape[:2]:
            thin = np.zeros(flux_arr.shape[:2], dtype=bool)
        disp_thin = thin[y0:y1:stride, x0:x1:stride]
        return disp_flux, disp_valid, disp_thin, stride

    def _update_two_d_view(
        self,
        flux: np.ndarray,
        valid: np.ndarray,
        cmap_name: str,
        color_norm,
        render_key,
    ) -> None:
        if render_key == self._last_two_d_render_key and self._two_d_im is not None:
            return
        disp_flux, disp_valid, disp_thin, stride = self._two_d_display_arrays(flux, valid)
        masked = np.ma.masked_where(~disp_valid, disp_flux)
        if self._two_d_im is None:
            self.two_d_ax.clear()
            self._two_d_im = self.two_d_ax.imshow(masked, cmap=cmap_name, origin="upper", norm=color_norm)
            try:
                self.two_d_ax.contour(disp_thin.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.35)
                self._two_d_contour_ready = True
            except Exception:
                self._two_d_contour_ready = False
            title = "Smoothed Flux Reconstruction"
            if int(stride) > 1:
                title += f" (display stride={int(stride)})"
            self.two_d_ax.set_title(title)
            self.two_d_ax.set_xticks([])
            self.two_d_ax.set_yticks([])
            self._two_d_colorbar = self.two_d_fig.colorbar(self._two_d_im, ax=self.two_d_ax, fraction=0.046, pad=0.04)
        else:
            self._two_d_im.set_data(masked)
            self._two_d_im.set_cmap(cmap_name)
            self._two_d_im.set_norm(color_norm)
            title = "Smoothed Flux Reconstruction"
            if int(stride) > 1:
                title += f" (display stride={int(stride)})"
            self.two_d_ax.set_title(title)
            if self._two_d_colorbar is not None:
                self._two_d_colorbar.update_normal(self._two_d_im)
        self._last_two_d_render_key = render_key
        self.two_d_canvas.draw_idle()

    def _render_three_d_payload(
        self,
        surf: Dict[str, object],
        cmap_name: str,
        color_norm,
        render_key,
    ) -> None:
        if render_key == self._last_three_d_render_key and self._three_d_surface_artist is not None:
            self._three_d_dirty = False
            return
        if self._three_d_surface_artist is not None:
            try:
                self._three_d_surface_artist.remove()
            except Exception:
                pass
            self._three_d_surface_artist = None
        if self._three_d_empty_text is not None:
            try:
                self._three_d_empty_text.remove()
            except Exception:
                pass
            self._three_d_empty_text = None
        self.three_d_ax.cla()
        x = np.asarray(surf["x"], dtype=np.float32)
        y = np.asarray(surf["y"], dtype=np.float32)
        z = surf["z"]
        if z.size > 0:
            z_plot = np.asarray(np.ma.filled(z, np.nan), dtype=np.float32)
            self._three_d_surface_artist = self.three_d_ax.plot_surface(
                x,
                y,
                z_plot,
                cmap=cmap_name,
                norm=color_norm,
                linewidth=0,
                antialiased=True,
                rcount=z_plot.shape[0],
                ccount=z_plot.shape[1],
            )
            self.three_d_ax.set_title(f"3D Flux Surface with L0 floor (downsample stride={int(surf['stride'])})")
            self.three_d_ax.set_xlabel("X")
            self.three_d_ax.set_ylabel("Y")
            self.three_d_ax.set_zlabel("Flux")
            self.three_d_ax.view_init(elev=42.0, azim=-58.0)
        else:
            self._three_d_empty_text = self.three_d_ax.text2D(
                0.1,
                0.5,
                "No valid ROI region for 3D display.",
                transform=self.three_d_ax.transAxes,
            )
        self._last_three_d_render_key = render_key
        self._three_d_dirty = False
        self.three_d_canvas.draw_idle()

    def _handle_tab_changed(self, index: int) -> None:
        if int(index) != 1:
            return
        if not self._three_d_dirty:
            return
        if self._pending_three_d_payload is None and self._pending_three_d_request is not None:
            rec_key, flux, cmap_name, color_norm, render_key = self._pending_three_d_request
            surf = self._surface_payload(rec_key, flux)
            self._pending_three_d_payload = (surf, cmap_name, color_norm, render_key)
        if self._pending_three_d_payload is None:
            return
        surf, cmap_name, color_norm, render_key = self._pending_three_d_payload
        self._render_three_d_payload(surf, cmap_name, color_norm, render_key)

    def refresh_views(self, *_args) -> None:
        try:
            ratio_value = self._ratio_value()
        except ValueError:
            if self.custom_contour_values:
                ratio_value = self._fallback_ratio_from_custom_values()
            else:
                self.ratio_edit.setStyleSheet("background-color: #fff1f0; color: #b71c1c;")
                self.stats_label.setText(
                    f"Detected levels: L1..L{int(max(0, self.max_detected_level))} | Invalid ratio expression"
                )
                self.formula_label.setText("Formula: L0 = rms floor, Lk = L1 * ratio^(k-1) for k >= 1")
                return
        self.ratio_edit.setStyleSheet("" if _safe_eval_ratio_expression(self.ratio_edit.text()) is not None else "background-color: #fff9db; color: #8a6d1d;")

        rec_key = self._reconstruction_key(ratio_value)
        rec = self._reconstruction(ratio_value)
        flux = np.asarray(rec["smoothed_flux_map"], dtype=np.float32)
        valid_u8 = np.asarray(rec["valid_mask"], dtype=np.uint8)
        transition_info = dict(rec.get("l0_l1_transition", {}) or {})
        valid = valid_u8 > 0
        min_flux = float(rec["min_flux"])
        max_flux = float(rec["max_flux"])
        valid_values = flux[valid & np.isfinite(flux)]
        cmap_name = self._current_cmap_name()
        color_norm, color_mode = self._build_color_norm(
            values=valid_values,
            log_enabled=bool(self.log_color_check.isChecked()),
        )

        self.stats_label.setText(
            f"Detected levels: L1..L{int(max(0, self.max_detected_level))} | "
            f"Flux range: {min_flux:.4g} .. {max_flux:.4g}"
        )
        if self.custom_contour_values:
            first = float(self.custom_contour_values[0])
            if len(self.custom_contour_values) >= int(max(1, self.max_detected_level)) + 1:
                peak = float(self.custom_contour_values[-1])
                self.contour_values_label.setText(
                    f"Mode: custom contour values (L1={first:.4g}, Peak={peak:.4g})"
                )
            else:
                last = float(self.custom_contour_values[-1])
                self.contour_values_label.setText(
                    f"Mode: custom contour values (legacy {len(self.custom_contour_values)} values, L1={first:.4g}, last={last:.4g})"
                )
        else:
            self.contour_values_label.setText("Mode: ratio")
        self.formula_label.setText(
            (
                f"Formula: L0/rms={float(self.bg_flux_spin.value()):.4g}, "
                f"L1={float(self.l1_flux_spin.value()):.4g}, "
                f"ratio={float(ratio_value):.6g}, "
                f"sigma={float(self.sigma_spin.value()):.3g}, "
                f"color={color_mode}, cmap={cmap_name}"
            )
            if not self.custom_contour_values
            else (
                f"Formula: L0/rms={float(self.bg_flux_spin.value()):.4g}, "
                f"custom contour values active, "
                f"sigma={float(self.sigma_spin.value()):.3g}, "
                f"color={color_mode}, cmap={cmap_name}"
            )
        )
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))

        norm_sig = self._color_norm_signature(color_norm, color_mode)
        two_d_render_key = (rec_key, cmap_name, norm_sig, int(self.display_max_dim))
        self._update_two_d_view(
            flux=flux,
            valid=valid,
            cmap_name=cmap_name,
            color_norm=color_norm,
            render_key=two_d_render_key,
        )

        three_d_render_key = (rec_key, int(self.surface_max_dim_spin.value()), cmap_name, norm_sig)
        self._pending_three_d_payload = None
        self._pending_three_d_request = (rec_key, flux, cmap_name, color_norm, three_d_render_key)
        self._three_d_dirty = True
        self._latest_reconstruction_payload = {
            "flux_map": flux,
            "valid_mask": valid_u8,
            "background_flux": 0.0,
            "l0_rms_flux": float(self.bg_flux_spin.value()),
            "l1_flux": float(self.l1_flux_spin.value()),
            "level_ratio": float(ratio_value),
            "custom_contour_values": None if not self.custom_contour_values else [float(v) for v in self.custom_contour_values],
            "smooth_sigma_px": float(self.sigma_spin.value()),
            "l0_l1_transition_mode": self._normalize_l0_l1_transition_mode(
                transition_info.get("mode", self._l0_l1_transition_mode())
            ),
            "l0_l1_transition_width_px": float(
                _safe_float(transition_info.get("width_px", float("nan")))
            ),
            "l0_l1_transition_width_source": str(transition_info.get("width_source", "")),
            "l0_l1_transition_alpha": float(
                _safe_float(
                    transition_info.get("alpha", self._l0_l1_transition_alpha()),
                    self._l0_l1_transition_alpha(),
                )
            ),
            "l0_l1_transition_applied_pixel_count": int(
                transition_info.get("applied_pixel_count", 0) or 0
            ),
            "cmap_name": str(cmap_name),
            "log_enabled": bool(self.log_color_check.isChecked()),
            "color_mode": str(color_mode),
        }
        if self.tabs.currentIndex() == 1:
            self._handle_tab_changed(1)


class QtFinalPreviewDialog(QDialog):
    def __init__(
        self,
        image: np.ndarray,
        max_width: int,
        max_height: int,
        reconstruction_context: Optional[Dict[str, object]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("JetAnalyzerV9 Qt preview")
        self.action = "close"
        self.preview_image = np.asarray(image, dtype=np.uint8).copy()
        self.reconstruction_context = dict(reconstruction_context or {})
        self.calibration_context = dict(self.reconstruction_context.get("calibration_context", {}) or {})
        label = QLabel()
        disp = _fit_image_to_box(image, max_w=max_width, max_h=max_height, allow_upscale=False)
        _set_label_image(label, disp)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.addWidget(label)
        scroll.setWidget(wrapper)
        self.calibration_label = QLabel(_calibration_summary_text(self.calibration_context))
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        buttons = QHBoxLayout()
        calibration_button = QPushButton("Scale / Beam")
        reconstruct_button = QPushButton("Flux Reconstruct")
        save_button = QPushButton("Save...")
        back_button = QPushButton("Back to Junction")
        close_button = QPushButton("Close")
        buttons.addStretch(1)
        if self._calibration_available():
            buttons.addWidget(calibration_button)
        if self.reconstruction_context:
            buttons.addWidget(reconstruct_button)
        buttons.addWidget(save_button)
        buttons.addWidget(back_button)
        buttons.addWidget(close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(self.calibration_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)
        calibration_button.clicked.connect(self._open_calibration)
        reconstruct_button.clicked.connect(self._open_reconstruction)
        save_button.clicked.connect(self._save_current)
        back_button.clicked.connect(self._back)
        close_button.clicked.connect(self.accept)

    def _analysis_context(self) -> Dict[str, object]:
        return dict(self.reconstruction_context.get("analysis_context", {}) or {})

    def _calibration_available(self) -> bool:
        if self.reconstruction_context.get("calibration_image", None) is not None:
            return True
        return bool(_analysis_image_path(self._analysis_context()))

    def _calibration_inputs(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        calibration_image = self.reconstruction_context.get("calibration_image", None)
        calibration_roi_mask = self.reconstruction_context.get("calibration_roi_mask", None)
        if calibration_image is not None:
            return np.asarray(calibration_image, dtype=np.uint8), (
                None if calibration_roi_mask is None else np.asarray(calibration_roi_mask, dtype=np.uint8)
            )
        image_path = _analysis_image_path(self._analysis_context())
        image = _load_calibration_image_from_path(image_path)
        if image is None:
            self.status_label.setText(f"Calibration image load failed: {image_path or 'no image path'}")
            return None, None
        return image, None

    def _open_calibration(self) -> None:
        calibration_image, calibration_roi_mask = self._calibration_inputs()
        if calibration_image is None:
            return
        dialog = QtScaleBeamDialog(
            roi_image=calibration_image,
            roi_mask=calibration_roi_mask,
            initial_state=self.calibration_context,
            max_width=int(self.reconstruction_context.get("max_width", 1200)),
            max_height=int(self.reconstruction_context.get("max_height", 900)),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self.calibration_context = dialog.get_calibration()
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))

    def _open_reconstruction(self) -> None:
        if not self.reconstruction_context:
            return
        dialog = QtFluxReconstructionDialog(
            region_depth_map=np.asarray(self.reconstruction_context.get("region_depth_map", np.zeros((1, 1), dtype=np.int32)), dtype=np.int32),
            roi_mask=np.asarray(self.reconstruction_context.get("roi_mask", np.zeros((1, 1), dtype=np.uint8)), dtype=np.uint8),
            thin_mask=np.asarray(self.reconstruction_context.get("thin_mask", np.zeros((1, 1), dtype=np.uint8)), dtype=np.uint8),
            max_detected_level=int(self.reconstruction_context.get("max_detected_level", 0)),
            max_width=int(self.reconstruction_context.get("max_width", 1200)),
            max_height=int(self.reconstruction_context.get("max_height", 900)),
            calibration_context=self.calibration_context,
            original_roi_image=self.reconstruction_context.get("original_roi_image", None),
            calibration_image=self.reconstruction_context.get("calibration_image", None),
            calibration_roi_mask=self.reconstruction_context.get("calibration_roi_mask", None),
            analysis_context=self.reconstruction_context.get("analysis_context", None),
            parent=self,
        )
        dialog.exec_()
        self.calibration_context = dict(dialog.calibration_context)
        self.calibration_label.setText(_calibration_summary_text(self.calibration_context))

    def _default_save_stem(self) -> str:
        analysis_context = dict(self.reconstruction_context.get("analysis_context", {}) or {})
        image_path = str(analysis_context.get("image_path", "") or "")
        if image_path:
            return f"{Path(image_path).stem}_contour_separation"
        return "contour_separation"

    def _contour_session_payload(self) -> Dict[str, object]:
        analysis_context = dict(self.reconstruction_context.get("analysis_context", {}) or {})
        return {
            "image_path": str(analysis_context.get("image_path", "") or ""),
            "loaded_replay_json_path": str(analysis_context.get("loaded_replay_json_path", "") or ""),
            "loaded_analysis_json_path": str(analysis_context.get("loaded_analysis_json_path", "") or ""),
            "roi_bbox_xywh": analysis_context.get("roi_bbox_xywh", None),
            "roi_points_xy": analysis_context.get("roi_points_xy", []),
            "binary_prep_mode": analysis_context.get("binary_prep_mode", None),
            "replay_snapshot": analysis_context.get("replay_snapshot", None),
            "calibration": dict(self.calibration_context),
            "contour_separation": {
                "stage": "final_preview",
                "saved_from": "contour_separation_preview",
            },
        }

    def _save_current(self) -> None:
        default_name = f"{self._default_save_stem()}.json"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Contour Separation",
            default_name,
            "Analysis JSON (*.json);;PNG Preview (*.png);;All Files (*)",
        )
        if not path:
            return
        target = Path(path)
        suffix = target.suffix.lower()
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        save_as_image = selected_filter.startswith("PNG") or suffix in image_suffixes
        if save_as_image:
            if target.suffix.lower() not in image_suffixes:
                target = target.with_suffix(".png")
            if not cv2.imwrite(str(target), self.preview_image):
                self.status_label.setText(f"Save failed: {target}")
                return
            self.status_label.setText(f"Saved preview: {target}")
            return
        if not target.suffix:
            target = target.with_suffix(".json")
        save_analysis_session(str(target), self._contour_session_payload())
        self.status_label.setText(f"Saved session: {target}")

    def _back(self) -> None:
        self.action = "back_to_junction"
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = int(event.key())
        if key == Qt.Key_S and (int(event.modifiers()) & int(Qt.ControlModifier)):
            self._save_current()
            return
        if key == Qt.Key_B:
            self._back()
            return
        if key == Qt.Key_F and self.reconstruction_context:
            self._open_reconstruction()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape):
            self.accept()
            return
        super().keyPressEvent(event)


def select_polygon_roi_qt(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> Optional[List[Point]]:
    ensure_qt_app()
    dialog = QtRoiSelectorDialog(image=image, max_width=max_width, max_height=max_height)
    if dialog.exec_() != QDialog.Accepted:
        return None
    return dialog.get_points()


def select_polygon_roi_or_load_json_qt(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> Dict[str, object]:
    ensure_qt_app()
    dialog = QtRoiSelectorDialog(image=image, max_width=max_width, max_height=max_height)
    if dialog.exec_() != QDialog.Accepted:
        return {"action": "cancel"}
    loaded_path = dialog.get_loaded_json_path()
    if loaded_path:
        return {"action": "load_json", "path": loaded_path}
    return {"action": "roi", "points": dialog.get_points()}


def run_manual_threshold_dialog_qt(
    roi_img: np.ndarray,
    roi_mask: np.ndarray,
    preview_callback: Callable[[int, bool], np.ndarray],
    initial_threshold: int,
    initial_invert: bool,
    max_width: int,
    max_height: int,
) -> Optional[Tuple[int, bool]]:
    ensure_qt_app()
    dialog = QtThresholdTunerDialog(
        roi_img=roi_img,
        roi_mask=roi_mask,
        preview_callback=preview_callback,
        initial_threshold=initial_threshold,
        initial_invert=initial_invert,
        max_width=max_width,
        max_height=max_height,
    )
    if dialog.exec_() != QDialog.Accepted:
        return None
    return dialog.get_values()


def edit_binary_mask_splits_qt(
    binary_mask: np.ndarray,
    junction_points_xy: Optional[List[List[int]]] = None,
    endpoint_points_xy: Optional[List[List[int]]] = None,
    window_name: str = "JetAnalyzerV9 Binary Split Editor",
    max_width: int = 1600,
    max_height: int = 1000,
    initial_cut_width: int = 5,
    initial_cuts: Optional[List[Dict[str, object]]] = None,
) -> Optional[Dict[str, object]]:
    ensure_qt_app()
    dialog = QtCutEditorDialog(
        base_mask=binary_mask,
        junction_points_xy=list(junction_points_xy or []),
        endpoint_points_xy=list(endpoint_points_xy or []),
        left_title="Original cleaned binary + node zones",
        right_title="Edited binary before thinning",
        subtitle_text="Click=point erase(square), drag=line cut | Right click removes nearest cut",
        max_width=max_width,
        max_height=max_height,
        initial_cut_width=initial_cut_width,
        initial_cuts=initial_cuts,
        live_node_mode="binary",
        show_original_panel=True,
        allow_back=False,
    )
    dialog.setWindowTitle(window_name)
    dialog.exec_()
    return dialog.result_payload()


def edit_thin_mask_junctions_qt(
    thin_mask: np.ndarray,
    junction_points_xy: List[List[int]],
    endpoint_points_xy: List[List[int]],
    window_name: str = "JetAnalyzerV9 Junction Editor",
    max_width: int = 1600,
    max_height: int = 1000,
    initial_cut_width: int = 3,
    initial_cuts: Optional[List[Dict[str, object]]] = None,
) -> Optional[Dict[str, object]]:
    ensure_qt_app()
    dialog = QtCutEditorDialog(
        base_mask=thin_mask,
        junction_points_xy=junction_points_xy,
        endpoint_points_xy=endpoint_points_xy,
        left_title="Edited thin after cuts",
        right_title="Edited thin after cuts",
        subtitle_text="Click=point erase(square), drag=line cut | Right click removes nearest cut | B=Back to Split",
        max_width=max_width,
        max_height=max_height,
        initial_cut_width=initial_cut_width,
        initial_cuts=initial_cuts,
        live_node_mode="thin",
        show_original_panel=False,
        allow_back=True,
    )
    dialog.setWindowTitle(window_name)
    dialog.exec_()
    return dialog.result_payload()


def show_final_preview_qt(
    image: np.ndarray,
    max_width: int,
    max_height: int,
    reconstruction_context: Optional[Dict[str, object]] = None,
) -> str:
    ensure_qt_app()
    dialog = QtFinalPreviewDialog(
        image=image,
        max_width=max_width,
        max_height=max_height,
        reconstruction_context=reconstruction_context,
    )
    dialog.exec_()
    return str(dialog.action)
