#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import os
import csv
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import QObject, QLibraryInfo, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fits_viewer import (
    apply_stretch,
    auto_limits,
    image_edges_mas,
    positive_contour_levels,
    read_primary_fits,
    robust_corner_rms,
    zoom_axes,
)
from mojave_opening_tool import (
    _beam_mas,
    _fits_stem,
    _pixel_mas,
    _prefixed_path,
    default_fits_dir,
    estimate_pa_center,
    expand_axes_to_points,
    opening_args_from_app,
    parse_prefer,
    resolve_fits_path,
    sector_from_center,
    sector_midpoint,
)
from polar_opening_angle import (
    AnalysisCancelled,
    apply_pa_sweep_to_records,
    default_baseline_guard_mas,
    default_fit_padding_mas,
    default_scan_half_width_mas,
    gaussian_model,
    load_ridge,
    measure_opening,
    pa_to_unit,
    records_to_json,
    sample_profile_for_opening_fit,
    save_outputs,
    select_fit_regions,
    summarize_opening_records,
)
from pushkarev_ridge import ridge_payload_from_ascii
from polar_ridge_builder import (
    Sector,
    build_polar_samples,
    filter_polar_sample_outliers,
    pa_to_xy,
    result_payload,
    save_result,
    sector_width_deg,
    smooth_resample_polar_samples,
)


def _prepare_qt_runtime_env() -> None:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "")
    try:
        bad_runtime = (
            (not runtime_dir)
            or (not os.path.isdir(runtime_dir))
            or ((os.stat(runtime_dir).st_mode & 0o777) != 0o700)
        )
        if bad_runtime:
            fallback_runtime = os.path.join(tempfile.gettempdir(), f"mojave_qt_runtime_{os.getuid()}")
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


def ensure_qt_app() -> QApplication:
    _prepare_qt_runtime_env()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
        app.setApplicationName("MOJAVE Opening Tool")
    return app


def _spin(value: float, minimum: float, maximum: float, step: float, decimals: int = 3) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def _int_spin(value: int, minimum: int, maximum: int, step: int = 1) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def _finite_positive_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0.0:
        return None
    return result


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _apply_log_axis_range(ax, x_min, x_max, y_min, y_max) -> None:
    x_low = _finite_positive_or_none(x_min)
    x_high = _finite_positive_or_none(x_max)
    if x_low is not None or x_high is not None:
        current_low, current_high = ax.get_xlim()
        low = x_low if x_low is not None else current_low
        high = x_high if x_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and low > 0.0 and high > low:
            ax.set_xlim(low, high)

    y_low = _finite_positive_or_none(y_min)
    y_high = _finite_positive_or_none(y_max)
    if y_low is not None or y_high is not None:
        current_low, current_high = ax.get_ylim()
        low = y_low if y_low is not None else current_low
        high = y_high if y_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and low > 0.0 and high > low:
            ax.set_ylim(low, high)


def _apply_axis_range(ax, x_min, x_max, y_min, y_max) -> None:
    x_low = _finite_or_none(x_min)
    x_high = _finite_or_none(x_max)
    if x_low is not None or x_high is not None:
        current_low, current_high = ax.get_xlim()
        low = x_low if x_low is not None else current_low
        high = x_high if x_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and high > low:
            ax.set_xlim(low, high)

    y_low = _finite_or_none(y_min)
    y_high = _finite_or_none(y_max)
    if y_low is not None or y_high is not None:
        current_low, current_high = ax.get_ylim()
        low = y_low if y_low is not None else current_low
        high = y_high if y_high is not None else current_high
        if np.isfinite(low) and np.isfinite(high) and high > low:
            ax.set_ylim(low, high)


def _sig_float(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(result):
        return None
    return round(result, 12)


class AnalysisWorker(QObject):
    status = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, fn, cancel_event: threading.Event):
        super().__init__()
        self.fn = fn
        self.cancel_event = cancel_event

    def run(self) -> None:
        try:
            result = self.fn(self.cancel_event, self.status.emit, self.progress.emit)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.finished.emit(result)


class MojaveOpeningQtWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fits_path: Optional[Path] = None
        self.image: Optional[np.ndarray] = None
        self.header: Dict[str, object] = {}
        self.rms = float("nan")
        self.beam_mas = float("nan")
        self.pixel_mas = float("nan")
        self.threshold = float("nan")
        self.output_prefix: Optional[Path] = args.output
        self.core_mas = args.core
        self.prefer_pa = parse_prefer(args.prefer)
        self.auto_info: Dict[str, object] = {}
        self.sector: Optional[Sector] = None
        self.ridge_payload: Optional[Dict[str, object]] = None
        self.external_ridge_payload: Optional[Dict[str, object]] = None
        self.external_ridge_path: Optional[Path] = None
        self.records = []
        self.summary: Optional[Dict[str, object]] = None
        self.opening_payload: Optional[Dict[str, object]] = None
        self.selected_slice_index = 0
        self.reset_xlim = None
        self.reset_ylim = None
        self.full_xlim = None
        self.full_ylim = None
        self.analysis_thread: Optional[QThread] = None
        self.analysis_worker: Optional[AnalysisWorker] = None
        self.analysis_cancel_event: Optional[threading.Event] = None
        self.ridge_signature = None
        self.fit_signature = None
        self.sweep_signature = None
        self.summary_signature = None

        self.setWindowTitle("MOJAVE Opening Tool")
        self.resize(1440, 900)
        self._build_ui()
        self._connect_canvas()
        if args.fits is not None:
            if self.load_fits(args.fits):
                self.redraw()
                self.statusBar().showMessage("FITS loaded. Press Apply Settings to analyze.")
        else:
            self.redraw()
            self.statusBar().showMessage("Open a FITS file to start.")

    def _build_ui(self) -> None:
        self.file_label = QLabel("No FITS loaded")
        self.file_label.setWordWrap(True)
        self.status_label = QLabel("Result: not analyzed")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { border: 1px solid #cfcfcf; padding: 6px; background: #fafafa; }")

        self.map_fig = Figure(figsize=(7, 6), tight_layout=True)
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_toolbar = NavigationToolbar(self.map_canvas, self)

        self.width_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.width_canvas = FigureCanvas(self.width_fig)
        self.width_ax = self.width_fig.add_subplot(111)

        self.angle_fig = Figure(figsize=(6, 5), tight_layout=True)
        self.angle_canvas = FigureCanvas(self.angle_fig)
        self.angle_ax = self.angle_fig.add_subplot(111)

        self.profile_fig = Figure(figsize=(7, 5), tight_layout=True)
        self.profile_canvas = FigureCanvas(self.profile_fig)
        self.profile_ax = self.profile_fig.add_subplot(211)
        self.profile_resid_ax = self.profile_fig.add_subplot(212, sharex=self.profile_ax)
        self.profile_info_label = QLabel("Slice profile: analyze first.")
        self.profile_info_label.setWordWrap(True)
        self.profile_index_spin = _int_spin(0, 0, 0)
        self.profile_prev_button = QPushButton("Previous")
        self.profile_next_button = QPushButton("Next")
        self.profile_save_png_button = QPushButton("Save PNG...")
        self.profile_save_csv_button = QPushButton("Save CSV...")

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("QPlainTextEdit { font-family: monospace; font-size: 11px; background: #fbfbfb; }")

        self.tabs = QTabWidget()
        self.map_tab = QWidget()
        map_layout = QVBoxLayout(self.map_tab)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_splitter = QSplitter(Qt.Horizontal)
        map_splitter.setChildrenCollapsible(False)
        map_left = QWidget()
        map_left_layout = QVBoxLayout(map_left)
        map_left_layout.setContentsMargins(0, 0, 0, 0)
        map_left_layout.addWidget(self.map_toolbar)
        map_left_layout.addWidget(self.map_canvas, stretch=1)
        self.profile_panel = QWidget()
        self.profile_panel.setMinimumWidth(360)
        profile_layout = QVBoxLayout(self.profile_panel)
        profile_layout.setContentsMargins(6, 0, 0, 0)
        profile_controls = QHBoxLayout()
        profile_controls.addWidget(QLabel("Slice"))
        profile_controls.addWidget(self.profile_index_spin)
        profile_controls.addWidget(self.profile_prev_button)
        profile_controls.addWidget(self.profile_next_button)
        profile_controls.addWidget(self.profile_save_png_button)
        profile_controls.addWidget(self.profile_save_csv_button)
        profile_controls.addStretch(1)
        profile_layout.addLayout(profile_controls)
        profile_layout.addWidget(self.profile_info_label)
        profile_layout.addWidget(self.profile_canvas, stretch=1)
        map_splitter.addWidget(map_left)
        map_splitter.addWidget(self.profile_panel)
        map_splitter.setStretchFactor(0, 3)
        map_splitter.setStretchFactor(1, 2)
        map_splitter.setSizes([700, 480])
        map_layout.addWidget(map_splitter)
        self.tabs.addTab(self.map_tab, "Map / Slice")
        width_tab = QWidget()
        width_layout = QVBoxLayout(width_tab)
        width_layout.setContentsMargins(0, 0, 0, 0)
        width_layout.addWidget(self.width_canvas)
        self.tabs.addTab(width_tab, "Gaussian Width")
        angle_tab = QWidget()
        angle_layout = QVBoxLayout(angle_tab)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.addWidget(self.angle_canvas)
        self.tabs.addTab(angle_tab, "Opening Angle")
        self.profile_tab = self.map_tab
        self.tabs.addTab(self.summary_text, "Summary")

        self.open_button = QPushButton("Open FITS...")
        self.load_ridge_button = QPushButton("Load Ridge...")
        self.apply_button = QPushButton("Apply Settings")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.save_button = QPushButton("Save Result...")
        self.auto_pa_button = QPushButton("Auto PA")
        self.opposite_button = QPushButton("Opposite")
        self.reset_view_button = QPushButton("Reset View")
        self.full_view_button = QPushButton("Full View")
        self.rotate_left_button = QPushButton("-5 deg")
        self.rotate_right_button = QPushButton("+5 deg")

        self.prefer_edit = QLineEdit("" if self.args.prefer is None else str(self.args.prefer))
        self.prefer_edit.setPlaceholderText("blank=global auto, east/west/north/south or PA deg")
        self.sector_width_spin = _spin(float(self.args.sector_width), 10.0, 180.0, 5.0, 1)
        self.threshold_snr_spin = _spin(float(self.args.threshold_snr), 0.1, 1000.0, 1.0, 2)
        self.rmin_spin = _spin(float(self.args.rmin), 0.0, 1000.0, 0.05, 3)
        self.step_spin = _spin(float(self.args.step), 0.001, 10.0, 0.01, 3)
        self.pa_step_spin = _spin(float(self.args.pa_step), 0.01, 10.0, 0.05, 3)
        self.auto_rmax_check = QCheckBox("Auto")
        self.auto_rmax_check.setChecked(self.args.rmax is None)
        self.rmax_spin = _spin(float(self.args.rmax or 10.0), 0.01, 10000.0, 0.5, 3)
        analysis_sep_min = getattr(self.args, "analysis_sep_min", getattr(self.args, "angle_rmin", 0.5))
        analysis_sep_max = getattr(self.args, "analysis_sep_max", None)
        self.analysis_sep_min_spin = _spin(float(analysis_sep_min), 0.0, 1000.0, 0.1, 3)
        self.no_analysis_sep_max_check = QCheckBox("No max")
        self.no_analysis_sep_max_check.setChecked(analysis_sep_max is None)
        self.analysis_sep_max_spin = _spin(float(analysis_sep_max or 10.0), 0.001, 10000.0, 0.5, 3)
        width_sep_min = getattr(self.args, "width_sep_min", None)
        width_sep_max = getattr(self.args, "width_sep_max", None)
        width_value_min = getattr(self.args, "width_value_min", None)
        width_value_max = getattr(self.args, "width_value_max", None)
        self.width_sep_auto_check = QCheckBox("Auto")
        self.width_sep_auto_check.setChecked(width_sep_min is None and width_sep_max is None)
        self.width_sep_min_spin = _spin(float(width_sep_min or 0.05), 0.000001, 10000.0, 0.1, 4)
        self.width_sep_max_spin = _spin(float(width_sep_max or 100.0), 0.000001, 10000.0, 0.5, 4)
        self.width_value_auto_check = QCheckBox("Auto")
        self.width_value_auto_check.setChecked(width_value_min is None and width_value_max is None)
        self.width_value_min_spin = _spin(float(width_value_min or 0.01), 0.000001, 10000.0, 0.01, 4)
        self.width_value_max_spin = _spin(float(width_value_max or 10.0), 0.000001, 10000.0, 0.1, 4)
        angle_plot_sep_min = getattr(self.args, "angle_plot_sep_min", None)
        angle_plot_sep_max = getattr(self.args, "angle_plot_sep_max", None)
        angle_plot_value_min = getattr(self.args, "angle_plot_value_min", None)
        angle_plot_value_max = getattr(self.args, "angle_plot_value_max", None)
        self.angle_plot_sep_auto_check = QCheckBox("Auto")
        self.angle_plot_sep_auto_check.setChecked(angle_plot_sep_min is None and angle_plot_sep_max is None)
        self.angle_plot_sep_min_spin = _spin(float(angle_plot_sep_min if angle_plot_sep_min is not None else 0.0), 0.0, 10000.0, 0.5, 4)
        self.angle_plot_sep_max_spin = _spin(float(angle_plot_sep_max if angle_plot_sep_max is not None else 100.0), 0.000001, 10000.0, 0.5, 4)
        self.angle_plot_value_auto_check = QCheckBox("Auto")
        self.angle_plot_value_auto_check.setChecked(angle_plot_value_min is None and angle_plot_value_max is None)
        self.angle_plot_value_min_spin = _spin(float(angle_plot_value_min if angle_plot_value_min is not None else 0.0), 0.0, 360.0, 1.0, 3)
        self.angle_plot_value_max_spin = _spin(float(angle_plot_value_max if angle_plot_value_max is not None else 90.0), 0.000001, 360.0, 1.0, 3)
        self.min_arc_points_spin = _int_spin(int(self.args.min_arc_points), 2, 1000)
        self.min_arc_span_spin = _spin(float(self.args.min_arc_span), 0.0, 180.0, 0.5, 2)
        self.min_peak_ratio_spin = _spin(float(self.args.min_peak_over_threshold), 1.0, 100.0, 0.05, 2)
        self.max_sample_step_factor_spin = _spin(float(self.args.max_sample_step_factor), 0.0, 1000.0, 0.5, 2)
        self.max_sample_pa_jump_spin = _spin(float(self.args.max_sample_pa_jump), 0.0, 180.0, 5.0, 1)
        self.ridge_smooth_spin = _spin(float(self.args.smooth), 0.0, 10.0, 0.01, 4)
        self.pa_sweep_check = QCheckBox("PA sweep error (+/-15 deg, 1 deg step)")
        self.pa_sweep_check.setChecked(bool(self.args.pa_sweep))
        self.pa_sweep_workers_spin = _int_spin(int(getattr(self.args, "pa_sweep_workers", 1)), 0, 128)
        self.pa_sweep_analysis_only_check = QCheckBox("PA sweep only analysis sep")
        self.pa_sweep_analysis_only_check.setChecked(bool(getattr(self.args, "pa_sweep_analysis_only", False)))
        self.full_view_check = QCheckBox("Initial full image view")
        self.full_view_check.setChecked(bool(self.args.full_view))
        self.fine_tuning_button = QPushButton("Show Fine Tuning")
        self.fine_tuning_button.setCheckable(True)

        controls_box = QGroupBox("Controls")
        controls = QVBoxLayout(controls_box)
        controls.addWidget(self.file_label)
        controls.addWidget(self.status_label)
        button_row = QHBoxLayout()
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.load_ridge_button)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.cancel_button)
        controls.addLayout(button_row)
        save_row = QHBoxLayout()
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.auto_pa_button)
        save_row.addWidget(self.opposite_button)
        controls.addLayout(save_row)
        rotate_row = QHBoxLayout()
        rotate_row.addWidget(self.rotate_left_button)
        rotate_row.addWidget(self.rotate_right_button)
        rotate_row.addWidget(self.reset_view_button)
        rotate_row.addWidget(self.full_view_button)
        controls.addLayout(rotate_row)

        form = QFormLayout()
        form.addRow("Preferred jet PA", self.prefer_edit)
        form.addRow("PA sector width (deg)", self.sector_width_spin)
        form.addRow("Ridge threshold SNR", self.threshold_snr_spin)
        form.addRow("Ridge search sep min (mas)", self.rmin_spin)
        form.addRow("Ridge search sep max", self.auto_rmax_check)
        form.addRow("Ridge search sep max value (mas)", self.rmax_spin)
        form.addRow("Ridge search sep step (mas)", self.step_spin)
        form.addRow("Analysis sep min (mas)", self.analysis_sep_min_spin)
        form.addRow("Analysis sep max", self.no_analysis_sep_max_check)
        form.addRow("Analysis sep max value (mas)", self.analysis_sep_max_spin)
        form.addRow("Width plot sep range", self.width_sep_auto_check)
        form.addRow("Width plot sep min (mas)", self.width_sep_min_spin)
        form.addRow("Width plot sep max (mas)", self.width_sep_max_spin)
        form.addRow("Width plot width range", self.width_value_auto_check)
        form.addRow("Width plot width min (mas)", self.width_value_min_spin)
        form.addRow("Width plot width max (mas)", self.width_value_max_spin)
        form.addRow("Angle plot sep range", self.angle_plot_sep_auto_check)
        form.addRow("Angle plot sep min (mas)", self.angle_plot_sep_min_spin)
        form.addRow("Angle plot sep max (mas)", self.angle_plot_sep_max_spin)
        form.addRow("Angle plot angle range", self.angle_plot_value_auto_check)
        form.addRow("Angle plot angle min (deg)", self.angle_plot_value_min_spin)
        form.addRow("Angle plot angle max (deg)", self.angle_plot_value_max_spin)
        form.addRow("", self.pa_sweep_check)
        form.addRow("", self.full_view_check)
        controls.addLayout(form)

        controls.addWidget(self.fine_tuning_button)
        self.fine_tuning_box = QGroupBox("Fine Tuning")
        fine_form = QFormLayout(self.fine_tuning_box)
        fine_form.addRow("Ridge smoothing scale (mas)", self.ridge_smooth_spin)
        fine_form.addRow("Arc PA sample step (deg)", self.pa_step_spin)
        fine_form.addRow("PA sweep workers", self.pa_sweep_workers_spin)
        fine_form.addRow("", self.pa_sweep_analysis_only_check)
        fine_form.addRow("Arc min points", self.min_arc_points_spin)
        fine_form.addRow("Arc min PA span (deg)", self.min_arc_span_spin)
        fine_form.addRow("Arc peak / threshold min", self.min_peak_ratio_spin)
        fine_form.addRow("Ridge raw jump max factor", self.max_sample_step_factor_spin)
        fine_form.addRow("Ridge PA jump max (deg)", self.max_sample_pa_jump_spin)
        self.fine_tuning_box.setVisible(False)
        controls.addWidget(self.fine_tuning_box)
        controls.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_box)
        right_layout.addWidget(controls_scroll)
        right_panel.setMinimumWidth(380)
        right_panel.setMaximumWidth(500)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1020, 420])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar(self))

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_action = QAction("Open FITS", self)
        load_ridge_action = QAction("Load Ridge", self)
        apply_action = QAction("Apply Settings", self)
        save_action = QAction("Save", self)
        exit_action = QAction("Quit", self)
        toolbar.addAction(open_action)
        toolbar.addAction(load_ridge_action)
        toolbar.addAction(apply_action)
        toolbar.addAction(save_action)
        toolbar.addSeparator()
        toolbar.addAction(exit_action)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(load_ridge_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        open_action.triggered.connect(lambda _checked=False: self.open_fits_dialog())
        load_ridge_action.triggered.connect(lambda _checked=False: self.load_ridge_dialog())
        apply_action.triggered.connect(lambda _checked=False: self.recompute(apply_controls=True))
        save_action.triggered.connect(lambda _checked=False: self.save_all(prompt=True))
        exit_action.triggered.connect(self.close)
        self.open_button.clicked.connect(lambda _checked=False: self.open_fits_dialog())
        self.load_ridge_button.clicked.connect(lambda _checked=False: self.load_ridge_dialog())
        self.apply_button.clicked.connect(lambda _checked=False: self.recompute(apply_controls=True))
        self.cancel_button.clicked.connect(lambda _checked=False: self.cancel_analysis())
        self.save_button.clicked.connect(lambda _checked=False: self.save_all(prompt=True))
        self.auto_pa_button.clicked.connect(lambda _checked=False: self.use_global_auto_pa())
        self.opposite_button.clicked.connect(lambda _checked=False: self.use_opposite_sector())
        self.reset_view_button.clicked.connect(lambda _checked=False: self.reset_view())
        self.full_view_button.clicked.connect(lambda _checked=False: self.full_view())
        self.rotate_left_button.clicked.connect(lambda _checked=False: self.rotate_sector(-5.0))
        self.rotate_right_button.clicked.connect(lambda _checked=False: self.rotate_sector(5.0))
        self.fine_tuning_button.toggled.connect(self.set_fine_tuning_visible)
        self.profile_index_spin.valueChanged.connect(lambda _value: self.set_profile_index(int(self.profile_index_spin.value()), redraw_map=True))
        self.profile_prev_button.clicked.connect(lambda _checked=False: self.set_profile_index(self.selected_slice_index - 1, redraw_map=True))
        self.profile_next_button.clicked.connect(lambda _checked=False: self.set_profile_index(self.selected_slice_index + 1, redraw_map=True))
        self.profile_save_png_button.clicked.connect(lambda _checked=False: self.save_current_profile_png())
        self.profile_save_csv_button.clicked.connect(lambda _checked=False: self.save_current_profile_csv())

    def _connect_canvas(self) -> None:
        self.map_canvas.mpl_connect("button_press_event", self.on_map_click)
        self.map_canvas.mpl_connect("scroll_event", self.on_map_scroll)
        self.width_canvas.mpl_connect("button_press_event", self.on_result_plot_click)
        self.angle_canvas.mpl_connect("button_press_event", self.on_result_plot_click)

    def set_fine_tuning_visible(self, visible: bool) -> None:
        self.fine_tuning_box.setVisible(bool(visible))
        self.fine_tuning_button.setText("Hide Fine Tuning" if visible else "Show Fine Tuning")

    def update_args_from_controls(self, update_prefer: bool = True) -> bool:
        if update_prefer:
            try:
                self.prefer_pa = parse_prefer(self.prefer_edit.text())
            except Exception as exc:
                QMessageBox.warning(self, "Invalid preferred PA", f"Preferred PA could not be parsed:\n{exc}")
                return False
        self.args.sector_width = float(self.sector_width_spin.value())
        self.args.threshold_snr = float(self.threshold_snr_spin.value())
        self.args.rmin = float(self.rmin_spin.value())
        self.args.rmax = None if self.auto_rmax_check.isChecked() else float(self.rmax_spin.value())
        self.args.step = float(self.step_spin.value())
        self.args.pa_step = float(self.pa_step_spin.value())
        self.args.smooth = float(self.ridge_smooth_spin.value())
        self.args.analysis_sep_min = float(self.analysis_sep_min_spin.value())
        self.args.analysis_sep_max = None if self.no_analysis_sep_max_check.isChecked() else float(self.analysis_sep_max_spin.value())
        self.args.angle_rmin = self.args.analysis_sep_min
        if self.args.analysis_sep_max is not None and self.args.analysis_sep_max <= self.args.analysis_sep_min:
            QMessageBox.warning(
                self,
                "Invalid analysis separation",
                "Analysis sep max must be larger than Analysis sep min.",
            )
            return False
        if self.width_sep_auto_check.isChecked():
            self.args.width_sep_min = None
            self.args.width_sep_max = None
        else:
            self.args.width_sep_min = float(self.width_sep_min_spin.value())
            self.args.width_sep_max = float(self.width_sep_max_spin.value())
            if self.args.width_sep_max <= self.args.width_sep_min:
                QMessageBox.warning(self, "Invalid width plot range", "Width plot sep max must be larger than sep min.")
                return False
        if self.width_value_auto_check.isChecked():
            self.args.width_value_min = None
            self.args.width_value_max = None
        else:
            self.args.width_value_min = float(self.width_value_min_spin.value())
            self.args.width_value_max = float(self.width_value_max_spin.value())
            if self.args.width_value_max <= self.args.width_value_min:
                QMessageBox.warning(self, "Invalid width plot range", "Width plot width max must be larger than width min.")
                return False
        if self.angle_plot_sep_auto_check.isChecked():
            self.args.angle_plot_sep_min = None
            self.args.angle_plot_sep_max = None
        else:
            self.args.angle_plot_sep_min = float(self.angle_plot_sep_min_spin.value())
            self.args.angle_plot_sep_max = float(self.angle_plot_sep_max_spin.value())
            if self.args.angle_plot_sep_max <= self.args.angle_plot_sep_min:
                QMessageBox.warning(self, "Invalid angle plot range", "Angle plot sep max must be larger than sep min.")
                return False
        if self.angle_plot_value_auto_check.isChecked():
            self.args.angle_plot_value_min = None
            self.args.angle_plot_value_max = None
        else:
            self.args.angle_plot_value_min = float(self.angle_plot_value_min_spin.value())
            self.args.angle_plot_value_max = float(self.angle_plot_value_max_spin.value())
            if self.args.angle_plot_value_max <= self.args.angle_plot_value_min:
                QMessageBox.warning(self, "Invalid angle plot range", "Angle plot angle max must be larger than angle min.")
                return False
        self.args.min_arc_points = int(self.min_arc_points_spin.value())
        self.args.min_arc_span = float(self.min_arc_span_spin.value())
        self.args.min_peak_over_threshold = float(self.min_peak_ratio_spin.value())
        self.args.max_sample_step_factor = float(self.max_sample_step_factor_spin.value())
        self.args.max_sample_pa_jump = float(self.max_sample_pa_jump_spin.value())
        self.args.pa_sweep = bool(self.pa_sweep_check.isChecked())
        self.args.pa_sweep_workers = int(self.pa_sweep_workers_spin.value())
        self.args.pa_sweep_analysis_only = bool(self.pa_sweep_analysis_only_check.isChecked())
        self.args.full_view = bool(self.full_view_check.isChecked())
        return True

    def load_fits(self, path: Path) -> bool:
        path = resolve_fits_path(path)
        if not path.exists():
            QMessageBox.warning(self, "FITS not found", f"FITS file does not exist:\n{path}")
            return False
        try:
            image, header = read_primary_fits(path)
        except Exception as exc:
            QMessageBox.critical(self, "FITS read failed", f"Could not read FITS file:\n{path}\n\n{exc}")
            return False
        self.fits_path = path
        self.image = image
        self.header = header
        self.rms = robust_corner_rms(self.image)
        self.beam_mas = _beam_mas(self.header)
        self.pixel_mas = _pixel_mas(self.header)
        self.threshold = self.args.threshold if self.args.threshold is not None else self.args.threshold_snr * self.rms
        self.output_prefix = self.args.output or path.with_name(f"{_fits_stem(path)}_opening_tool")
        self.reset_xlim = None
        self.reset_ylim = None
        self.full_xlim = None
        self.full_ylim = None
        self.auto_info = {}
        self.sector = self.auto_sector(self.prefer_pa)
        self.ridge_payload = None
        self.external_ridge_payload = None
        self.external_ridge_path = None
        self.records = []
        self.summary = None
        self.opening_payload = None
        self.ridge_signature = None
        self.fit_signature = None
        self.sweep_signature = None
        self.summary_signature = None
        self.selected_slice_index = 0
        self.update_profile_index_range()
        self.file_label.setText(str(path))
        self.statusBar().showMessage(f"Loaded FITS: {path}")
        return True

    def open_fits_dialog(self) -> None:
        initial_dir = self.fits_path.parent if self.fits_path is not None else default_fits_dir()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open FITS image",
            str(initial_dir),
            "FITS files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fts.gz);;All files (*)",
        )
        if selected and self.load_fits(Path(selected)):
            self.redraw()
            self.statusBar().showMessage("FITS loaded. Press Apply Settings to analyze.")

    def load_ridge_dialog(self) -> None:
        if self.image is None or self.fits_path is None:
            QMessageBox.information(self, "No FITS loaded", "Open a FITS image before loading an external ridgeline.")
            return
        initial_dir = self.external_ridge_path.parent if self.external_ridge_path is not None else default_fits_dir()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load external ridgeline",
            str(initial_dir),
            "Ridgeline files (*.json *.csv *.dat *.txt);;All files (*)",
        )
        if selected:
            self.load_external_ridge(Path(selected))

    def load_external_ridge(self, path: Path) -> bool:
        if self.image is None or self.fits_path is None:
            return False
        if not path.exists():
            QMessageBox.warning(self, "Ridgeline not found", f"Ridgeline file does not exist:\n{path}")
            return False
        try:
            if path.suffix.lower() == ".json":
                payload = load_ridge(path)
                payload = copy.deepcopy(payload)
                payload.setdefault("sector", {"name": "external", "pa_min_deg": 0.0, "pa_max_deg": 0.0, "pa_width_deg": 0.0})
                payload.setdefault("parameters", {})
                payload["parameters"]["loaded_external_ridge_file"] = str(path)
            else:
                payload = ridge_payload_from_ascii(self.fits_path, self.header, self.image, path)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            QMessageBox.critical(self, "Ridgeline load failed", f"Could not load ridgeline:\n{path}\n\n{exc}")
            return False
        except Exception as exc:
            QMessageBox.critical(self, "Ridgeline load failed", f"Could not load ridgeline:\n{path}\n\n{exc}")
            return False

        points = payload.get("ridge_points", [])
        if not points:
            QMessageBox.warning(self, "Invalid ridgeline", "The selected file does not contain ridge_points.")
            return False
        self.external_ridge_payload = payload
        self.external_ridge_path = path
        self.ridge_payload = copy.deepcopy(payload)
        self.records = []
        self.summary = None
        self.opening_payload = None
        self.ridge_signature = None
        self.fit_signature = None
        self.sweep_signature = None
        self.summary_signature = None
        self.selected_slice_index = 0
        self.update_profile_index_range()
        self.redraw()
        self.statusBar().showMessage(f"Loaded external ridgeline: {path.name}. Press Apply Settings to fit widths.")
        return True

    def auto_rmax(self) -> float:
        if self.image is None:
            return float("nan")
        if self.args.rmax is not None:
            return float(self.args.rmax)
        return float(np.nanmax([0.0, self._image_radius_limit()]))

    def _image_radius_limit(self) -> float:
        from mojave_opening_tool import image_radius_limit_mas

        if self.image is None:
            return float("nan")
        return image_radius_limit_mas(self.header, self.image.shape, self.core_mas)

    def auto_sector(self, prefer_pa: Optional[float]) -> Sector:
        if self.image is None:
            return sector_from_center(self.args.sector_name, prefer_pa if prefer_pa is not None else 90.0, self.args.sector_width)
        center, info = estimate_pa_center(
            self.image,
            self.header,
            self.core_mas,
            self.threshold,
            self.args.auto_pa_rmin,
            self.args.auto_pa_rmax if self.args.auto_pa_rmax is not None else self.auto_rmax(),
            self.args.auto_pa_bin,
            self.args.auto_pa_smooth,
            prefer_pa,
            self.args.prefer_search,
        )
        self.auto_info = info
        return sector_from_center(self.args.sector_name, center, self.args.sector_width)

    def _fit_context(self, args):
        beam_mas = _beam_mas(self.header)
        pix_mas = _pixel_mas(self.header)
        rms = self.rms
        fit_threshold = args.fit_threshold if args.fit_threshold is not None else args.fit_threshold_snr * rms
        half_width = args.slice_half_width if args.slice_half_width is not None else default_scan_half_width_mas(beam_mas)
        sample_step = args.sample_step if args.sample_step is not None else max(pix_mas / 2.0, 0.01)
        padding = args.fit_padding if args.fit_padding is not None else default_fit_padding_mas(beam_mas)
        baseline_guard = args.baseline_guard if getattr(args, "baseline_guard", None) is not None else default_baseline_guard_mas(beam_mas)
        sigma_upper = args.sigma_upper if args.sigma_upper is not None else None
        return beam_mas, pix_mas, rms, fit_threshold, half_width, sample_step, padding, baseline_guard, sigma_upper

    def _ridge_sig(self, args, sector: Sector, rmax: float, threshold: float):
        if self.external_ridge_payload is not None:
            points = list(self.external_ridge_payload.get("ridge_points", []))
            first = points[0] if points else {}
            last = points[-1] if points else {}
            return (
                "external",
                str(self.external_ridge_path),
                len(points),
                _sig_float(first.get("x_mas", float("nan"))),
                _sig_float(first.get("y_mas", float("nan"))),
                _sig_float(last.get("x_mas", float("nan"))),
                _sig_float(last.get("y_mas", float("nan"))),
            )
        return (
            str(self.fits_path),
            tuple(_sig_float(v) for v in self.core_mas),
            sector.name,
            _sig_float(sector.pa_min_deg),
            _sig_float(sector.pa_max_deg),
            _sig_float(args.rmin),
            _sig_float(rmax),
            _sig_float(args.step),
            _sig_float(args.pa_step),
            _sig_float(threshold),
            args.component,
            int(args.max_empty),
            int(args.min_arc_points),
            _sig_float(args.min_arc_span),
            _sig_float(args.min_peak_over_threshold),
            _sig_float(args.max_sample_step_factor),
            _sig_float(args.max_sample_pa_jump),
            _sig_float(args.smooth),
        )

    def _fit_sig(self, args):
        return (
            _sig_float(args.slice_half_width),
            _sig_float(args.sample_step),
            _sig_float(args.fit_threshold_snr),
            _sig_float(args.fit_threshold),
            _sig_float(args.fit_padding),
            _sig_float(getattr(args, "baseline_guard", None)),
            int(args.min_fit_points),
            _sig_float(args.sigma_upper),
        )

    def _sweep_sig(self, args):
        return (
            bool(args.pa_sweep),
            _sig_float(args.pa_sweep_range),
            _sig_float(args.pa_sweep_step),
            bool(getattr(args, "pa_sweep_analysis_only", False)),
            _sig_float(args.analysis_sep_min) if getattr(args, "pa_sweep_analysis_only", False) else None,
            _sig_float(args.analysis_sep_max) if getattr(args, "pa_sweep_analysis_only", False) else None,
        )

    def _summary_sig(self, args):
        return (
            _sig_float(args.analysis_sep_min),
            _sig_float(args.analysis_sep_max),
            int(args.bootstrap_count),
            int(args.bootstrap_seed),
        )

    def _make_opening_payload(self, summary, records):
        return {
            "format": "mojave_integrated_opening_tool_qt",
            "version": 1,
            "fits_file": str(self.fits_path),
            "ridge_sector": (self.ridge_payload or {}).get("sector", {}),
            "summary": summary,
            "records": records_to_json(records),
        }

    def recompute(self, apply_controls: bool = False) -> None:
        if self.analysis_thread is not None:
            self.statusBar().showMessage("Analysis is already running.")
            return
        previous_prefer_pa = self.prefer_pa
        if apply_controls and not self.update_args_from_controls(update_prefer=True):
            return
        if self.image is None or self.fits_path is None:
            self.statusBar().showMessage("No FITS loaded.")
            self.redraw()
            return
        self.threshold = self.args.threshold if self.args.threshold is not None else self.args.threshold_snr * self.rms
        prefer_changed = previous_prefer_pa != self.prefer_pa
        if self.sector is None or prefer_changed:
            self.sector = self.auto_sector(self.prefer_pa)
        else:
            self.sector = sector_from_center(self.sector.name, sector_midpoint(self.sector), self.args.sector_width)

        rmax = self.auto_rmax()
        args_snapshot = copy.deepcopy(self.args)
        sector_snapshot = copy.deepcopy(self.sector)
        ridge_sig = self._ridge_sig(args_snapshot, sector_snapshot, rmax, self.threshold)
        fit_sig = self._fit_sig(args_snapshot)
        sweep_sig = self._sweep_sig(args_snapshot)
        summary_sig = self._summary_sig(args_snapshot)

        if not self.ridge_payload or not self.records or ridge_sig != self.ridge_signature or fit_sig != self.fit_signature:
            mode = "full"
        elif sweep_sig != self.sweep_signature:
            mode = "sweep_summary"
        elif summary_sig != self.summary_signature:
            mode = "summary"
        else:
            self.redraw()
            self.update_summary_text()
            self.statusBar().showMessage("Display updated.")
            return

        self._start_analysis_worker(mode, args_snapshot, sector_snapshot, rmax, self.threshold, ridge_sig, fit_sig, sweep_sig, summary_sig)

    def _start_analysis_worker(self, mode, args_snapshot, sector_snapshot, rmax, threshold, ridge_sig, fit_sig, sweep_sig, summary_sig) -> None:
        image = self.image
        header = dict(self.header)
        fits_path = self.fits_path
        rms = self.rms
        core_mas = tuple(self.core_mas)
        auto_info = copy.deepcopy(self.auto_info)
        records_snapshot = copy.deepcopy(self.records)
        ridge_payload_snapshot = copy.deepcopy(self.ridge_payload)
        external_ridge_payload_snapshot = copy.deepcopy(self.external_ridge_payload)

        def job(cancel_event, status_emit, progress_emit):
            if mode == "full":
                if external_ridge_payload_snapshot is not None:
                    status_emit("Using external ridgeline...")
                    ridge_payload = external_ridge_payload_snapshot
                else:
                    status_emit("Building ridgeline...")
                    samples = build_polar_samples(
                        image,
                        header,
                        sector_snapshot,
                        core_mas,
                        args_snapshot.rmin,
                        rmax,
                        args_snapshot.step,
                        args_snapshot.pa_step,
                        threshold,
                        args_snapshot.component,
                        args_snapshot.max_empty,
                        args_snapshot.min_arc_points,
                        args_snapshot.min_arc_span,
                        args_snapshot.min_peak_over_threshold,
                    )
                    if cancel_event.is_set():
                        raise AnalysisCancelled("analysis cancelled")
                    filtered_samples = filter_polar_sample_outliers(
                        samples,
                        args_snapshot.step,
                        args_snapshot.max_sample_step_factor,
                        args_snapshot.max_sample_pa_jump,
                    )
                    ridge = smooth_resample_polar_samples(filtered_samples, core_mas, args_snapshot.step, args_snapshot.smooth)
                    params = {
                        "rmin_mas": args_snapshot.rmin,
                        "rmax_mas": rmax,
                        "r_step_mas": args_snapshot.step,
                        "pa_step_deg": args_snapshot.pa_step,
                        "ridge_search_sep_min_mas": args_snapshot.rmin,
                        "ridge_search_sep_max_mas": rmax,
                        "ridge_search_sep_step_mas": args_snapshot.step,
                        "arc_pa_sample_step_deg": args_snapshot.pa_step,
                        "threshold_snr": args_snapshot.threshold_snr,
                        "ridge_threshold_snr": args_snapshot.threshold_snr,
                        "threshold": threshold,
                        "component_mode": args_snapshot.component,
                        "smooth_mas": args_snapshot.smooth,
                        "max_empty": args_snapshot.max_empty,
                        "min_arc_points": args_snapshot.min_arc_points,
                        "min_arc_span_deg": args_snapshot.min_arc_span,
                        "min_peak_over_threshold": args_snapshot.min_peak_over_threshold,
                        "max_sample_step_factor": args_snapshot.max_sample_step_factor,
                        "max_sample_pa_jump_deg": args_snapshot.max_sample_pa_jump,
                        "raw_sample_count_before_outlier_filter": len(samples),
                        "outlier_filtered_count": max(0, len(samples) - len(filtered_samples)),
                        "method": "qt integrated auto-sector core-centered polar arc flux-median ridgeline",
                        "auto_pa": auto_info,
                    }
                    ridge_payload = result_payload(
                        fits_path,
                        header,
                        image,
                        rms,
                        threshold,
                        core_mas,
                        sector_snapshot,
                        filtered_samples,
                        ridge,
                        params,
                    )
                status_emit("Fitting Gaussian widths...")
                records, summary = measure_opening(
                    image,
                    header,
                    ridge_payload,
                    opening_args_from_app(args_snapshot),
                    progress_callback=progress_emit,
                    cancel_event=cancel_event,
                )
            else:
                records = records_snapshot
                ridge_payload = ridge_payload_snapshot
                beam_mas, pix_mas, fit_rms, fit_threshold, half_width, sample_step, padding, baseline_guard, sigma_upper = self._fit_context(args_snapshot)
                sep_min = float(args_snapshot.analysis_sep_min)
                sep_max = None if args_snapshot.analysis_sep_max is None else float(args_snapshot.analysis_sep_max)
                if mode == "sweep_summary":
                    status_emit("Updating PA sweep...")
                    apply_pa_sweep_to_records(
                        image,
                        header,
                        records,
                        opening_args_from_app(args_snapshot),
                        beam_mas,
                        fit_threshold,
                        padding,
                        args_snapshot.min_fit_points,
                        sigma_upper,
                        half_width,
                        sample_step,
                        fit_rms,
                        baseline_guard,
                        sep_min,
                        sep_max,
                        progress_callback=lambda done, total: progress_emit("PA sweep", done, total),
                        cancel_event=cancel_event,
                    )
                status_emit("Updating summary...")
                summary = summarize_opening_records(
                    records,
                    beam_mas,
                    pix_mas,
                    fit_rms,
                    fit_threshold,
                    half_width,
                    sample_step,
                    padding,
                    baseline_guard,
                    opening_args_from_app(args_snapshot),
                )
            return {
                "mode": mode,
                "ridge_payload": ridge_payload,
                "records": records,
                "summary": summary,
                "ridge_signature": ridge_sig,
                "fit_signature": fit_sig,
                "sweep_signature": sweep_sig,
                "summary_signature": summary_sig,
            }

        self.analysis_cancel_event = threading.Event()
        self.analysis_thread = QThread(self)
        self.analysis_worker = AnalysisWorker(job, self.analysis_cancel_event)
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.status.connect(lambda message: self.statusBar().showMessage(message))
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.failed.connect(self.on_analysis_failed)
        self.analysis_worker.cancelled.connect(self.on_analysis_cancelled)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.failed.connect(self.analysis_thread.quit)
        self.analysis_worker.cancelled.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.on_analysis_thread_finished)
        self.set_analysis_running(True)
        self.statusBar().showMessage("Analyzing...")
        self.analysis_thread.start()

    def set_analysis_running(self, running: bool) -> None:
        for widget in (
            self.open_button,
            self.apply_button,
            self.save_button,
            self.auto_pa_button,
            self.opposite_button,
            self.rotate_left_button,
            self.rotate_right_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def cancel_analysis(self) -> None:
        if self.analysis_cancel_event is not None:
            self.analysis_cancel_event.set()
            self.statusBar().showMessage("Cancelling analysis...")

    def on_analysis_progress(self, stage: str, done: int, total: int) -> None:
        if total > 0:
            self.statusBar().showMessage(f"{stage}: {done}/{total}")
        else:
            self.statusBar().showMessage(stage)

    def on_analysis_finished(self, result) -> None:
        self.ridge_payload = result["ridge_payload"]
        self.records = result["records"]
        self.summary = result["summary"]
        self.ridge_signature = result["ridge_signature"]
        self.fit_signature = result["fit_signature"]
        self.sweep_signature = result["sweep_signature"]
        self.summary_signature = result["summary_signature"]
        self.update_profile_index_range()
        self.opening_payload = self._make_opening_payload(self.summary, self.records)
        self.redraw()
        self.update_summary_text()
        self.statusBar().showMessage(f"Analysis complete ({result['mode']}).")

    def on_analysis_failed(self, details: str) -> None:
        QMessageBox.critical(self, "Analysis failed", details)
        self.statusBar().showMessage("Analysis failed.")

    def on_analysis_cancelled(self) -> None:
        self.statusBar().showMessage("Analysis cancelled.")

    def on_analysis_thread_finished(self) -> None:
        self.set_analysis_running(False)
        if self.analysis_thread is not None:
            self.analysis_thread.deleteLater()
        self.analysis_thread = None
        self.analysis_worker = None
        self.analysis_cancel_event = None

    def draw_map(self) -> None:
        ax = self.map_ax
        ax.clear()
        if self.image is None or self.fits_path is None or self.sector is None:
            ax.axis("off")
            ax.text(0.5, 0.55, "Open a FITS image", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            ax.text(0.5, 0.45, "File > Open FITS", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            self.map_canvas.draw_idle()
            return
        finite = self.image[np.isfinite(self.image)]
        vmin = max(float(np.nanpercentile(finite, 1.0)), -3.0 * self.rms)
        vmax = float(np.nanpercentile(finite, 99.9))
        display = apply_stretch(self.image, "asinh", vmin, vmax)
        extent = image_edges_mas(self.header, self.image.shape)
        ax.imshow(display, origin="lower", extent=extent, cmap=self.args.cmap, interpolation="nearest")
        levels = positive_contour_levels(self.rms, float(np.nanmax(self.image)), self.args.contour_sigma, self.args.contour_factor)
        if levels.size:
            ax.contour(self.image, levels=levels, colors="white", linewidths=0.45, alpha=0.65, origin="lower", extent=extent)
        ax.scatter([self.core_mas[0]], [self.core_mas[1]], marker="+", s=110, c="#00e5ff", linewidths=1.8, zorder=7)

        radius = self.auto_rmax()
        width = sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg)
        pa_arc = self.sector.pa_min_deg + np.linspace(0.0, width, 160)
        x_arc, y_arc = pa_to_xy(self.core_mas, radius, pa_arc)
        ax.plot(x_arc, y_arc, lw=0.8, color="#00e5ff", alpha=0.65)
        for pa in (self.sector.pa_min_deg, self.sector.pa_min_deg + width):
            xe, ye = pa_to_xy(self.core_mas, radius, np.asarray([pa], dtype=float))
            ax.plot([self.core_mas[0], xe[0]], [self.core_mas[1], ye[0]], lw=0.8, color="#00e5ff", alpha=0.65)

        if self.ridge_payload:
            raw = np.asarray([(p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get("raw_samples", [])], dtype=float)
            if raw.size:
                ax.scatter(raw[:, 0], raw[:, 1], s=8, c="#ffdf5d", edgecolors="none", alpha=0.75, zorder=8)
            ridge = np.asarray([(p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get("ridge_points", [])], dtype=float)
            if ridge.size:
                ax.plot(ridge[:, 0], ridge[:, 1], color="#22ff66", lw=1.8, zorder=9)
        if self.records:
            rec = self.records[int(np.clip(self.selected_slice_index, 0, len(self.records) - 1))]
            normal = pa_to_unit((rec.tangent_pa_deg + 90.0) % 360.0)
            s0 = rec.scan_min_mas if np.isfinite(rec.scan_min_mas) else -max(4.0 * self.beam_mas, 1.5)
            s1 = rec.scan_max_mas if np.isfinite(rec.scan_max_mas) else max(4.0 * self.beam_mas, 1.5)
            x0 = rec.x_mas + s0 * normal[0]
            y0 = rec.y_mas + s0 * normal[1]
            x1 = rec.x_mas + s1 * normal[0]
            y1 = rec.y_mas + s1 * normal[1]
            ax.plot([x0, x1], [y0, y1], color="#00ffff", lw=1.0, alpha=0.9, zorder=10)
            ax.scatter([rec.x_mas], [rec.y_mas], s=45, facecolors="none", edgecolors="#00ffff", linewidths=1.2, zorder=11)

        ax.set_xlabel("RA offset (mas)")
        ax.set_ylabel("Dec offset (mas)")
        ax.set_title(f"{self.fits_path.name}\nPA {self.sector.pa_min_deg:.1f}..{self.sector.pa_max_deg:.1f}; beam={self.beam_mas:.3f} mas")
        if self.reset_xlim is None or self.reset_ylim is None:
            limits = None if self.args.full_view else auto_limits(self.image, self.header, self.rms, 0.005, 8.0, 2.0)
            if limits is not None:
                xmin, xmax, ymin, ymax = limits
                ax.set_xlim(max(xmin, xmax), min(xmin, xmax))
                ax.set_ylim(min(ymin, ymax), max(ymin, ymax))
            if self.args.fov is not None:
                half = float(self.args.fov)
                ax.set_xlim(half, -half)
                ax.set_ylim(-half, half)
            if self.ridge_payload:
                all_xy = []
                for key in ("raw_samples", "ridge_points"):
                    all_xy.extend((p["x_mas"], p["y_mas"]) for p in self.ridge_payload.get(key, []))
                if all_xy:
                    expand_axes_to_points(ax, np.asarray(all_xy, dtype=float), padding_mas=max(0.5, self.beam_mas))
            self.reset_xlim = ax.get_xlim()
            self.reset_ylim = ax.get_ylim()
            edges = image_edges_mas(self.header, self.image.shape)
            self.full_xlim = edges[:2]
            self.full_ylim = edges[2:]
        self.map_canvas.draw_idle()

    def draw_width(self) -> None:
        ax = self.width_ax
        ax.clear()
        if not self.records:
            ax.axis("off")
            ax.set_title("No fit records")
            self.width_canvas.draw_idle()
            return
        path = np.asarray([r.path_mas for r in self.records], dtype=float)
        raw = np.asarray([r.fit.fwhm_mas for r in self.records], dtype=float)
        deconv = np.asarray([r.intrinsic_fwhm_mas for r in self.records], dtype=float)
        raw_err = np.asarray([r.pa_sweep_raw_rms_mas for r in self.records], dtype=float)
        deconv_err = np.asarray([r.pa_sweep_intrinsic_rms_mas for r in self.records], dtype=float)
        success = np.asarray([r.fit.success for r in self.records], dtype=bool)
        ax.plot(path[success], raw[success], ".", ms=3, label="raw D")
        ax.plot(path[success], deconv[success], ".", ms=3, label="deconvolved d")
        raw_err_mask = success & np.isfinite(path) & np.isfinite(raw) & (raw > 0.0) & np.isfinite(raw_err) & (raw_err > 0.0)
        if np.any(raw_err_mask):
            ax.errorbar(
                path[raw_err_mask],
                raw[raw_err_mask],
                yerr=raw_err[raw_err_mask],
                fmt="none",
                ecolor="#1f77b4",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="raw PA sweep rms",
            )
        deconv_err_mask = success & np.isfinite(path) & np.isfinite(deconv) & (deconv > 0.0) & np.isfinite(deconv_err) & (deconv_err > 0.0)
        if np.any(deconv_err_mask):
            ax.errorbar(
                path[deconv_err_mask],
                deconv[deconv_err_mask],
                yerr=deconv_err[deconv_err_mask],
                fmt="none",
                ecolor="#ff7f0e",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="deconv PA sweep rms",
            )
        failed = (~success) & np.isfinite(raw) & (raw > 0)
        if np.any(failed):
            ax.plot(path[failed], raw[failed], "x", ms=4, color="0.55", label="rejected fit")
        if len(path):
            sel = int(np.clip(self.selected_slice_index, 0, len(path) - 1))
            if np.isfinite(path[sel]):
                ax.axvline(path[sel], color="#00a6fb", lw=1.0, alpha=0.75, label="selected slice")
        ax.axhline(self.beam_mas, color="0.5", lw=1.0, ls="--", label="beam")
        fit5 = dict((self.summary or {}).get("power_law_fit_smoothed5", {}))
        if np.isfinite(fit5.get("k", float("nan"))) and fit5.get("n", 0) >= 2:
            rline = np.logspace(math.log10(float(fit5["r_min_mas"])), math.log10(float(fit5["r_max_mas"])), 160)
            dline = float(fit5["amplitude_mas_at_1mas"]) * np.power(rline, float(fit5["k"]))
            ax.plot(rline, dline, color="#d62828", lw=1.5, label=f"d=A r^k, k={fit5['k']:.3g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _apply_log_axis_range(
            ax,
            getattr(self.args, "width_sep_min", None),
            getattr(self.args, "width_sep_max", None),
            getattr(self.args, "width_value_min", None),
            getattr(self.args, "width_value_max", None),
        )
        ax.set_xlabel("Separation along ridgeline (mas)")
        ax.set_ylabel("Gaussian width (mas)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("Gaussian widths")
        self.width_canvas.draw_idle()

    def draw_angle(self) -> None:
        ax = self.angle_ax
        ax.clear()
        if not self.records or not self.summary:
            ax.axis("off")
            ax.set_title("No opening-angle result")
            self.angle_canvas.draw_idle()
            return
        path = np.asarray([r.path_mas for r in self.records], dtype=float)
        raw = np.asarray([r.opening_angle_raw_deg for r in self.records], dtype=float)
        deconv = np.asarray([r.opening_angle_deg for r in self.records], dtype=float)
        raw_err = np.asarray([r.pa_sweep_opening_angle_raw_rms_deg for r in self.records], dtype=float)
        deconv_err = np.asarray([r.pa_sweep_opening_angle_intrinsic_rms_deg for r in self.records], dtype=float)
        ax.plot(path, raw, ".", ms=3, alpha=0.5, label="raw")
        ax.plot(path, deconv, ".", ms=3, label="deconvolved")
        raw_err_mask = np.isfinite(path) & np.isfinite(raw) & np.isfinite(raw_err) & (raw_err > 0.0)
        if np.any(raw_err_mask):
            ax.errorbar(
                path[raw_err_mask],
                raw[raw_err_mask],
                yerr=raw_err[raw_err_mask],
                fmt="none",
                ecolor="#1f77b4",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="raw PA sweep rms",
            )
        deconv_err_mask = np.isfinite(path) & np.isfinite(deconv) & np.isfinite(deconv_err) & (deconv_err > 0.0)
        if np.any(deconv_err_mask):
            ax.errorbar(
                path[deconv_err_mask],
                deconv[deconv_err_mask],
                yerr=deconv_err[deconv_err_mask],
                fmt="none",
                ecolor="#ff7f0e",
                elinewidth=0.55,
                alpha=0.35,
                capsize=0,
                label="deconv PA sweep rms",
            )
        if len(path):
            sel = int(np.clip(self.selected_slice_index, 0, len(path) - 1))
            if np.isfinite(path[sel]):
                ax.axvline(path[sel], color="#00a6fb", lw=1.0, alpha=0.75, label="selected slice")
        ax.axvline(float(self.summary["analysis_separation_min_mas"]), color="0.5", lw=1.0, ls="--")
        if self.summary.get("analysis_separation_max_mas") is not None:
            ax.axvline(float(self.summary["analysis_separation_max_mas"]), color="0.5", lw=1.0, ls=":")
        median = float(self.summary["median_opening_angle_deg"])
        ax.axhline(median, color="#e63946", lw=1.2, label="median d")
        median_error = dict(self.summary.get("median_error", {}))
        err = dict(median_error.get("intrinsic_block", {}))
        err_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
        title_err = err_with_pa if np.isfinite(err_with_pa.get("sigma", float("nan"))) else err
        fit5 = dict(self.summary.get("power_law_fit_smoothed5", {}))
        ax.set_xlabel("Separation along ridgeline (mas)")
        ax.set_ylabel("Full apparent opening angle (deg)")
        _apply_axis_range(
            ax,
            getattr(self.args, "angle_plot_sep_min", None),
            getattr(self.args, "angle_plot_sep_max", None),
            getattr(self.args, "angle_plot_value_min", None),
            getattr(self.args, "angle_plot_value_max", None),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"median={median:.3g} +/- {title_err.get('sigma', float('nan')):.2g} deg; k={fit5.get('k', float('nan')):.3g}")
        self.angle_canvas.draw_idle()

    def update_profile_index_range(self) -> None:
        if not self.records:
            self.selected_slice_index = 0
            self.profile_index_spin.blockSignals(True)
            self.profile_index_spin.setRange(0, 0)
            self.profile_index_spin.setValue(0)
            self.profile_index_spin.blockSignals(False)
            return
        self.selected_slice_index = int(np.clip(self.selected_slice_index, 0, len(self.records) - 1))
        self.profile_index_spin.blockSignals(True)
        self.profile_index_spin.setRange(0, len(self.records) - 1)
        self.profile_index_spin.setValue(self.selected_slice_index)
        self.profile_index_spin.blockSignals(False)

    def set_profile_index(self, index: int, redraw_map: bool = False) -> None:
        if not self.records:
            return
        self.selected_slice_index = int(np.clip(index, 0, len(self.records) - 1))
        self.profile_index_spin.blockSignals(True)
        self.profile_index_spin.setValue(self.selected_slice_index)
        self.profile_index_spin.blockSignals(False)
        self.draw_slice_profile()
        self.draw_width()
        self.draw_angle()
        if redraw_map:
            self.draw_map()
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.profile_tab))

    def draw_slice_profile(self) -> None:
        self.profile_ax.clear()
        self.profile_resid_ax.clear()
        if self.image is None or not self.records or not self.summary:
            self.profile_ax.axis("off")
            self.profile_resid_ax.axis("off")
            self.profile_ax.set_title("No slice profile")
            self.profile_info_label.setText("Slice profile: analyze first.")
            self.profile_canvas.draw_idle()
            return

        rec = self.records[int(np.clip(self.selected_slice_index, 0, len(self.records) - 1))]
        fixed_half_width = float(self.summary["slice_half_width_mas"]) if np.isfinite(float(self.summary["slice_half_width_mas"])) else None
        sample_step = float(self.summary["sample_step_mas"])
        threshold = float(self.summary["fit_threshold_jy_per_beam"])
        padding = float(self.summary["fit_padding_mas"])
        baseline_guard = float(self.summary.get("baseline_guard_mas", 0.0))
        min_points = int(self.args.min_fit_points)
        s, y = sample_profile_for_opening_fit(
            self.image,
            self.header,
            rec.x_mas,
            rec.y_mas,
            rec.tangent_pa_deg,
            fixed_half_width,
            threshold,
            padding,
            min_points,
            sample_step,
        )
        regions = select_fit_regions(s, y, threshold, padding, min_points, baseline_guard, float(self.summary["beam_mas"]), float(self.summary["rms_jy_per_beam"]))
        fit_mask = regions.fit_mask
        self.profile_ax.plot(s, y, color="0.25", lw=1.0, label="slice profile")
        if np.any(regions.baseline_mask):
            self.profile_ax.plot(s[regions.baseline_mask], y[regions.baseline_mask], ".", ms=3, color="#2a9d8f", alpha=0.75, label="baseline region")
        if np.any(regions.source_mask):
            self.profile_ax.plot(s[regions.source_mask], y[regions.source_mask], ".", ms=4, color="#f4a261", label="3rms source")
        if np.any(fit_mask):
            self.profile_ax.plot(s[fit_mask], y[fit_mask], ".", ms=4, color="#1f77b4", label="fit window")
        self.profile_ax.axhline(threshold, color="0.55", ls="--", lw=1.0, label="fit threshold")
        fit = rec.fit
        model = None
        if np.isfinite(fit.fwhm_mas) and np.isfinite(fit.baseline) and np.isfinite(fit.amplitude) and np.isfinite(fit.mu_mas) and np.isfinite(fit.sigma_mas):
            x_model = np.linspace(float(np.nanmin(s)), float(np.nanmax(s)), 800)
            y_model = gaussian_model(x_model, fit.baseline, fit.amplitude, fit.mu_mas, fit.sigma_mas)
            self.profile_ax.plot(x_model, y_model, color="#d62828", lw=1.5, label=f"Gaussian ({fit.reason})")
            half_level = fit.baseline + 0.5 * fit.amplitude
            left = fit.mu_mas - 0.5 * fit.fwhm_mas
            right = fit.mu_mas + 0.5 * fit.fwhm_mas
            self.profile_ax.axhline(half_level, color="#d62828", ls=":", lw=1.0)
            self.profile_ax.axvline(fit.mu_mas, color="#d62828", ls="-", lw=0.9)
            self.profile_ax.axvline(left, color="#d62828", ls="--", lw=0.9)
            self.profile_ax.axvline(right, color="#d62828", ls="--", lw=0.9)
            model = gaussian_model(s, fit.baseline, fit.amplitude, fit.mu_mas, fit.sigma_mas)
        self.profile_ax.set_ylabel(str(self.header.get("BUNIT", "") or "image value"))
        self.profile_ax.set_title(f"Slice {rec.index}: transverse Gaussian FWHM profile")
        self.profile_ax.grid(alpha=0.25)
        self.profile_ax.legend(fontsize=8, loc="best")

        if model is not None:
            self.profile_resid_ax.plot(s, y - model, color="#9467bd", lw=1.0)
            self.profile_resid_ax.axhline(0.0, color="0.4", lw=0.8)
        self.profile_resid_ax.set_xlabel("Transverse offset from ridge point (mas)")
        self.profile_resid_ax.set_ylabel("Residual")
        self.profile_resid_ax.grid(alpha=0.25)

        self.profile_info_label.setText(
            f"index={rec.index}, radial={rec.radial_mas:.4g} mas, separation={rec.path_mas:.4g} mas, "
            f"tangent PA={rec.tangent_pa_deg:.3g} deg, scan=[{rec.scan_min_mas:.4g}, {rec.scan_max_mas:.4g}] mas, "
            f"fit={fit.success} ({fit.reason}), "
            f"baseline={fit.baseline:.4g}, flags={fit.baseline_flag or 'none'}, "
            f"mu={fit.mu_mas:.4g} mas, sigma={fit.sigma_mas:.4g} mas, "
            f"FWHM={fit.fwhm_mas:.4g} mas, deconvolved={rec.intrinsic_fwhm_mas:.4g} mas, "
            f"opening={rec.opening_angle_deg:.4g} deg"
        )
        self.profile_canvas.draw_idle()

    def current_profile_arrays(self):
        if self.image is None or not self.records or not self.summary:
            return None
        rec = self.records[int(np.clip(self.selected_slice_index, 0, len(self.records) - 1))]
        fixed_half_width = float(self.summary["slice_half_width_mas"]) if np.isfinite(float(self.summary["slice_half_width_mas"])) else None
        sample_step = float(self.summary["sample_step_mas"])
        threshold = float(self.summary["fit_threshold_jy_per_beam"])
        padding = float(self.summary["fit_padding_mas"])
        baseline_guard = float(self.summary.get("baseline_guard_mas", 0.0))
        s, y = sample_profile_for_opening_fit(
            self.image,
            self.header,
            rec.x_mas,
            rec.y_mas,
            rec.tangent_pa_deg,
            fixed_half_width,
            threshold,
            padding,
            int(self.args.min_fit_points),
            sample_step,
        )
        regions = select_fit_regions(s, y, threshold, padding, int(self.args.min_fit_points), baseline_guard, float(self.summary["beam_mas"]), float(self.summary["rms_jy_per_beam"]))
        fit_mask = regions.fit_mask
        model = np.full_like(s, np.nan, dtype=float)
        fit = rec.fit
        if np.isfinite(fit.baseline) and np.isfinite(fit.amplitude) and np.isfinite(fit.mu_mas) and np.isfinite(fit.sigma_mas):
            model = gaussian_model(s, fit.baseline, fit.amplitude, fit.mu_mas, fit.sigma_mas)
        return rec, s, y, fit_mask, regions.baseline_mask, regions.source_mask, model

    def default_profile_prefix(self) -> Path:
        base = self.output_prefix or Path("mojave_opening_tool")
        rec_index = self.records[self.selected_slice_index].index if self.records else 0
        return base.parent / f"{base.name}_slice_{int(rec_index):04d}_profile"

    def save_current_profile_png(self) -> None:
        if not self.records:
            QMessageBox.information(self, "No profile", "Analyze a FITS image first.")
            return
        default_path = self.default_profile_prefix().with_suffix(".png")
        selected, _ = QFileDialog.getSaveFileName(self, "Save slice profile PNG", str(default_path), "PNG image (*.png);;All files (*)")
        if not selected:
            return
        self.profile_fig.savefig(selected, dpi=180)
        self.statusBar().showMessage(f"Saved slice profile PNG: {selected}")

    def save_current_profile_csv(self) -> None:
        data = self.current_profile_arrays()
        if data is None:
            QMessageBox.information(self, "No profile", "Analyze a FITS image first.")
            return
        rec, s, y, fit_mask, baseline_mask, source_mask, model = data
        default_path = self.default_profile_prefix().with_suffix(".csv")
        selected, _ = QFileDialog.getSaveFileName(self, "Save slice profile CSV", str(default_path), "CSV file (*.csv);;All files (*)")
        if not selected:
            return
        with Path(selected).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"index={rec.index}",
                f"radial_mas={rec.radial_mas}",
                f"path_mas={rec.path_mas}",
                f"x_mas={rec.x_mas}",
                f"y_mas={rec.y_mas}",
                f"tangent_pa_deg={rec.tangent_pa_deg}",
                f"fit_success={rec.fit.success}",
                f"fit_reason={rec.fit.reason}",
                f"baseline={rec.fit.baseline}",
                f"baseline_method={rec.fit.baseline_method}",
                f"baseline_flag={rec.fit.baseline_flag}",
                f"baseline_n={rec.fit.baseline_n}",
                f"amplitude={rec.fit.amplitude}",
                f"mu_mas={rec.fit.mu_mas}",
                f"sigma_mas={rec.fit.sigma_mas}",
                f"fwhm_mas={rec.fit.fwhm_mas}",
                f"intrinsic_fwhm_mas={rec.intrinsic_fwhm_mas}",
            ])
            writer.writerow(["s_mas", "image_value", "fit_window", "baseline_region", "source_component", "gaussian_model", "residual"])
            for si, yi, mi, bi, ci, gi in zip(s, y, fit_mask, baseline_mask, source_mask, model):
                residual = yi - gi if np.isfinite(yi) and np.isfinite(gi) else float("nan")
                writer.writerow([si, yi, int(bool(mi)), int(bool(bi)), int(bool(ci)), gi, residual])
        self.statusBar().showMessage(f"Saved slice profile CSV: {selected}")

    def redraw(self) -> None:
        current_xlim = self.map_ax.get_xlim() if self.reset_xlim is not None else None
        current_ylim = self.map_ax.get_ylim() if self.reset_ylim is not None else None
        self.draw_map()
        if current_xlim is not None and current_ylim is not None:
            self.map_ax.set_xlim(current_xlim)
            self.map_ax.set_ylim(current_ylim)
            self.map_canvas.draw_idle()
        self.draw_width()
        self.draw_angle()
        self.draw_slice_profile()
        self.update_summary_text()

    def update_summary_text(self) -> None:
        if not self.summary or self.sector is None:
            self.status_label.setText("Result: not analyzed")
            self.summary_text.setPlainText("No analysis result.")
            return
        fit5 = dict(self.summary.get("power_law_fit_smoothed5", {}))
        median_error = dict(self.summary.get("median_error", {}))
        err = dict(median_error.get("intrinsic_block", {}))
        err_with_pa = dict(median_error.get("intrinsic_block_with_pa_sweep", {}))
        display_err = err_with_pa if np.isfinite(err_with_pa.get("sigma", float("nan"))) else err
        block_slices = median_error.get("block_size_slices", "n/a")
        block_length = median_error.get("block_length_mas", float("nan"))
        block_target = median_error.get("block_target_mas", float("nan"))
        sweep = dict(self.summary.get("pa_sweep", {}))
        ridge_count = len(self.ridge_payload.get("ridge_points", [])) if self.ridge_payload else 0
        ridge_params = dict((self.ridge_payload or {}).get("parameters", {}))
        ridge_source = "external" if self.external_ridge_payload is not None else "computed"
        if self.external_ridge_path is not None:
            ridge_source = f"external ({self.external_ridge_path.name})"
        sep_max = self.summary.get("analysis_separation_max_mas")
        sep_max_text = "none" if sep_max is None else f"{float(sep_max):.6g}"
        text = (
            f"FITS: {self.fits_path}\n"
            f"Beam: {self.beam_mas:.6g} mas\n"
            f"Pixel: {self.pixel_mas:.6g} mas\n"
            f"RMS: {self.rms:.6g}\n"
            f"Threshold: {self.threshold:.6g}\n"
            f"Sector: {self.sector.name}, PA {self.sector.pa_min_deg:.3f}..{self.sector.pa_max_deg:.3f} deg "
            f"(center {sector_midpoint(self.sector):.3f})\n"
            f"Ridgeline source: {ridge_source}\n"
            f"Ridgeline points: {ridge_count}\n"
            f"Raw samples before/after outlier filter: "
            f"{ridge_params.get('raw_sample_count_before_outlier_filter', 'n/a')}/"
            f"{len((self.ridge_payload or {}).get('raw_samples', []))}\n"
            f"Outlier-filtered samples: {ridge_params.get('outlier_filtered_count', 0)}\n"
            f"Gaussian fits: {self.summary['fit_success_count']}/{self.summary['record_count']}\n"
            f"Analysis separation: {self.summary['analysis_separation_min_mas']:.6g}"
            f"..{sep_max_text} mas\n"
            f"Median deconvolved full opening angle: {self.summary['median_opening_angle_deg']:.6g} deg\n"
            f"Median error sampling scale: {block_slices} slices = {block_length:.6g} mas "
            f"(target 0.5 beam = {block_target:.6g} mas)\n"
            f"Median block bootstrap sigma: {err.get('sigma', float('nan')):.6g} deg\n"
            f"Median block + PA-sweep sigma: {err_with_pa.get('sigma', float('nan')):.6g} deg\n"
            f"PA sweep median deconv width rms: {sweep.get('median_intrinsic_width_rms_mas', float('nan')):.6g} mas\n"
            f"PA sweep median deconv angle rms: {sweep.get('median_intrinsic_opening_angle_rms_deg', float('nan')):.6g} deg\n"
            f"Median raw full opening angle: {self.summary['median_opening_angle_raw_deg']:.6g} deg\n"
            f"Power-law width fit: d = A r^k, k={fit5.get('k', float('nan')):.6g}, "
            f"A={fit5.get('amplitude_mas_at_1mas', float('nan')):.6g} mas at 1 mas\n"
        )
        self.summary_text.setPlainText(text)
        self.status_label.setText(
            f"Median: {self.summary['median_opening_angle_deg']:.4g} +/- {display_err.get('sigma', float('nan')):.3g} deg "
            f"(block {block_slices}={block_length:.3g} mas)\n"
            f"Fits: {self.summary['fit_success_count']}/{self.summary['record_count']} | k={fit5.get('k', float('nan')):.4g}"
        )

    def save_all(self, prompt: bool = True) -> None:
        if self.analysis_thread is not None:
            QMessageBox.information(self, "Analysis running", "Wait for the current analysis to finish before saving.")
            return
        if self.image is None or self.fits_path is None or self.output_prefix is None:
            QMessageBox.information(self, "No FITS loaded", "Open and analyze a FITS image first.")
            return
        if not self.ridge_payload or not self.summary or self.opening_payload is None:
            self.recompute()
            QMessageBox.information(self, "Analysis started", "Save again after the analysis finishes.")
            return
        if not self.ridge_payload or not self.summary or self.opening_payload is None or self.sector is None:
            return
        if prompt:
            prefix_text, _ = QFileDialog.getSaveFileName(
                self,
                "Save output prefix",
                str(self.output_prefix),
                "Output prefix (*)",
            )
            if not prefix_text:
                return
            self.output_prefix = Path(prefix_text)
        ridge_prefix = self.output_prefix.parent / f"{self.output_prefix.name}_ridge"
        opening_prefix = self.output_prefix.parent / f"{self.output_prefix.name}_opening_{self.sector.name}"
        ridge_json, ridge_csv = save_result(ridge_prefix, self.ridge_payload)
        opening_json, opening_csv = save_outputs(opening_prefix, self.opening_payload, self.records)
        fig_path = _prefixed_path(self.output_prefix, f"_{self.sector.name}.png")
        self.map_fig.savefig(fig_path, dpi=180)
        message = (
            f"Ridge JSON: {ridge_json}\n"
            f"Ridge CSV: {ridge_csv}\n"
            f"Opening JSON: {opening_json}\n"
            f"Opening CSV: {opening_csv}\n"
            f"Map figure: {fig_path}"
        )
        if prompt:
            QMessageBox.information(self, "Saved", message)
        else:
            print(message)
        self.statusBar().showMessage(f"Saved result prefix: {self.output_prefix}")

    def use_global_auto_pa(self) -> None:
        self.prefer_edit.setText("")
        self.prefer_pa = None
        self.sector = self.auto_sector(None)
        self.recompute()

    def use_opposite_sector(self) -> None:
        if self.sector is None:
            return
        self.sector = sector_from_center(
            self.sector.name,
            sector_midpoint(self.sector) + 180.0,
            sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg),
        )
        self.redraw()
        self.recompute()

    def rotate_sector(self, delta_deg: float) -> None:
        if self.sector is None:
            return
        width = sector_width_deg(self.sector.pa_min_deg, self.sector.pa_max_deg)
        self.sector = sector_from_center(self.sector.name, sector_midpoint(self.sector) + delta_deg, width)
        self.redraw()

    def reset_view(self) -> None:
        if self.reset_xlim is not None and self.reset_ylim is not None:
            self.map_ax.set_xlim(self.reset_xlim)
            self.map_ax.set_ylim(self.reset_ylim)
            self.map_canvas.draw_idle()

    def full_view(self) -> None:
        if self.full_xlim is not None and self.full_ylim is not None:
            self.map_ax.set_xlim(self.full_xlim)
            self.map_ax.set_ylim(self.full_ylim)
            self.map_canvas.draw_idle()

    def on_map_click(self, event) -> None:
        if self.image is None or event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None or event.button != 1:
            return
        dx = float(event.xdata) - float(self.core_mas[0])
        dy = float(event.ydata) - float(self.core_mas[1])
        if dx == 0 and dy == 0:
            return
        pa = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        self.prefer_pa = pa
        self.prefer_edit.setText(f"{pa:.3f}")
        self.sector = self.auto_sector(self.prefer_pa)
        self.recompute()

    def on_result_plot_click(self, event) -> None:
        if not self.records or event.xdata is None or event.button != 1:
            return
        paths = np.asarray([r.path_mas for r in self.records], dtype=float)
        finite = np.isfinite(paths)
        if not np.any(finite):
            return
        valid_indices = np.flatnonzero(finite)
        nearest = valid_indices[int(np.nanargmin(np.abs(paths[finite] - float(event.xdata))))]
        self.set_profile_index(int(nearest), redraw_map=True)

    def on_map_scroll(self, event) -> None:
        if event.inaxes is not self.map_ax:
            return
        center = None
        if event.xdata is not None and event.ydata is not None:
            center = (float(event.xdata), float(event.ydata))
        zoom_axes(self.map_ax, 1.0 / 1.25 if event.button == "up" else 1.25, center)
        self.map_canvas.draw_idle()

    def closeEvent(self, event) -> None:
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            self.cancel_analysis()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
        super().closeEvent(event)


def run_qt_tool(args) -> int:
    app = ensure_qt_app()
    window = MojaveOpeningQtWindow(args)
    window.show()
    if args.save and args.fits is not None:
        if window.analysis_thread is None and window.summary is None:
            window.recompute()
        if window.analysis_thread is not None:
            window.analysis_thread.finished.connect(lambda: window.save_all(prompt=False))
        else:
            window.save_all(prompt=False)
    return int(app.exec_())
