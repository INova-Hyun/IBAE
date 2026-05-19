from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .gui_v9_qt import ensure_qt_app
from .common.numeric import safe_float as _safe_float
from .reports import (
    build_gaussian_report_rows,
    paper_fig7_eastern_broken_power_law,
)
from .ridgeline import opening_angle_deg
from .session_analysis import load_analysis_session


@dataclass
class LoadedTrendSession:
    path: str
    label: str
    image_path: str
    calibration: Dict[str, object]
    measurement_result: Dict[str, object]
    trend_report: Dict[str, object]


def _lighten_color(color_value: object, mix: float = 0.55) -> Tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color_value), dtype=np.float64)
    white = np.ones(3, dtype=np.float64)
    return tuple(np.clip(((1.0 - float(mix)) * rgb) + (float(mix) * white), 0.0, 1.0).tolist())


def _normalize_mpl_color(value: object, fallback: str = "#1f77b4") -> str:
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


class QtComparePlotStyleDialog(QDialog):
    def __init__(
        self,
        session_labels: Sequence[str],
        session_colors: Dict[str, str],
        style: Dict[str, object],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compare Plot Style")
        self._labels = list(session_labels)
        self._color_buttons: Dict[str, QPushButton] = {}
        self._defaults = {
            "main_size": 30.0,
            "raw_size": 26.0,
            "residual_size": 18.0,
        }

        self.main_size_spin = QDoubleSpinBox()
        self.raw_size_spin = QDoubleSpinBox()
        self.residual_size_spin = QDoubleSpinBox()
        for spin in (self.main_size_spin, self.raw_size_spin, self.residual_size_spin):
            spin.setRange(2.0, 200.0)
            spin.setDecimals(1)
            spin.setSingleStep(2.0)
        self.main_size_spin.setValue(float(style.get("main_size", self._defaults["main_size"])))
        self.raw_size_spin.setValue(float(style.get("raw_size", self._defaults["raw_size"])))
        self.residual_size_spin.setValue(float(style.get("residual_size", self._defaults["residual_size"])))

        form = QFormLayout()
        form.addRow("Main marker size", self.main_size_spin)
        form.addRow("Raw marker size", self.raw_size_spin)
        form.addRow("Residual marker size", self.residual_size_spin)
        for label in self._labels:
            btn = QPushButton()
            _apply_color_button_preview(btn, session_colors.get(label, "#1f77b4"))
            btn.clicked.connect(lambda _checked=False, b=btn: self._choose_color(b))
            self._color_buttons[label] = btn
            form.addRow(f"{label} color", btn)

        reset_button = QPushButton("Reset Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(reset_button)
        layout.addWidget(buttons)
        self.resize(420, 0)

    def _choose_color(self, button: QPushButton) -> None:
        current = QColor(button.text().strip() or "#ffffff")
        picked = QColorDialog.getColor(current, self, "Choose Session Color")
        if picked.isValid():
            _apply_color_button_preview(button, picked.name())

    def _reset_defaults(self) -> None:
        self.main_size_spin.setValue(float(self._defaults["main_size"]))
        self.raw_size_spin.setValue(float(self._defaults["raw_size"]))
        self.residual_size_spin.setValue(float(self._defaults["residual_size"]))
        palette = list(mcolors.TABLEAU_COLORS.values())
        for idx, label in enumerate(self._labels):
            _apply_color_button_preview(self._color_buttons[label], palette[idx % len(palette)])

    def get_settings(self) -> Dict[str, object]:
        return {
            "main_size": float(self.main_size_spin.value()),
            "raw_size": float(self.raw_size_spin.value()),
            "residual_size": float(self.residual_size_spin.value()),
            "session_colors": {
                label: _normalize_mpl_color(button.text(), "#1f77b4")
                for label, button in self._color_buttons.items()
            },
        }


def _distance_unit_for_sessions(sessions: Sequence[LoadedTrendSession]) -> str:
    if not sessions:
        return "px"
    all_have_scale = True
    for session in sessions:
        scale = _safe_float(session.calibration.get("scale_mas_per_px", float("nan")))
        if not np.isfinite(scale) or scale <= 0.0:
            all_have_scale = False
            break
    return "mas" if all_have_scale else "px"


def _width_key(distance_unit: str) -> str:
    return "gaussian_width_mas" if distance_unit == "mas" else "gaussian_width_px"


def _raw_width_key(distance_unit: str) -> str:
    return "gaussian_raw_width_mas" if distance_unit == "mas" else "gaussian_raw_width_px"


def _distance_key(distance_unit: str) -> str:
    return "distance_from_core_mas" if distance_unit == "mas" else "distance_from_core_px"


def _build_rows_for_session(
    session: LoadedTrendSession,
    *,
    distance_unit: str,
) -> List[Dict[str, object]]:
    scale = _safe_float(session.calibration.get("scale_mas_per_px", float("nan")))
    return build_gaussian_report_rows(
        session.measurement_result,
        scale if np.isfinite(scale) and scale > 0.0 else None,
        use_raw_width=False,
    )


def _filter_rows_by_x_range(
    rows: Sequence[Dict[str, object]],
    *,
    distance_unit: str,
    x_min: Optional[float],
    x_max: Optional[float],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    x_key = _distance_key(distance_unit)
    xmin = None if x_min is None else float(x_min)
    xmax = None if x_max is None else float(x_max)
    for row in list(rows):
        xv = _safe_float(row.get(x_key, float("nan")))
        if not np.isfinite(xv) or xv <= 0.0:
            continue
        if xmin is not None and xv < xmin:
            continue
        if xmax is not None and xv > xmax:
            continue
        out.append(dict(row))
    return out


def _format_float_or_na(value: object, fmt: str = ".4g") -> str:
    val = _safe_float(value, float("nan"))
    return format(val, fmt) if np.isfinite(val) else "n/a"


class QtTrendReportCompareDialog(QDialog):
    def __init__(self, initial_paths: Optional[Sequence[str]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Trend Report Compare")
        self.sessions: List[LoadedTrendSession] = []
        self.distance_unit: str = "px"
        self._sync_enabled = True
        self._color_cycle = list(mcolors.TABLEAU_COLORS.values())
        self.session_colors: Dict[str, str] = {}
        self.plot_style: Dict[str, object] = {
            "main_size": 30.0,
            "raw_size": 26.0,
            "residual_size": 18.0,
        }

        self.info_label = QLabel(
            "Load multiple ridgeline analysis JSON files to compare saved width / opening-angle trends. "
            "This viewer uses stored measurement results and saved trend results only; it does not recompute ridgelines, widths, or trend fits. "
            "The selected X range controls the visible plot range."
        )
        self.info_label.setWordWrap(True)

        self.load_button = QPushButton("Load JSONs...")
        self.clear_button = QPushButton("Clear")
        self.save_figure_button = QPushButton("Save Figure...")
        self.save_settings_button = QPushButton("Save Settings...")
        self.load_settings_button = QPushButton("Load Settings...")
        self.show_raw_width_check = QCheckBox("Show raw FWHM")
        self.show_raw_width_check.setChecked(True)
        self.show_raw_angle_check = QCheckBox("Show raw angle")
        self.show_raw_angle_check.setChecked(True)
        self.show_paper_model_check = QCheckBox("Show paper Fig.7 broken PL")
        self.show_paper_model_check.setChecked(False)
        self.plot_style_button = QPushButton("Plot Style...")
        self.auto_x_range_check = QCheckBox("Auto X Range")
        self.auto_x_range_check.setChecked(True)
        self.auto_width_y_range_check = QCheckBox("Auto Width Y Range")
        self.auto_width_y_range_check.setChecked(True)
        self.x_min_spin = QDoubleSpinBox()
        self.x_max_spin = QDoubleSpinBox()
        self.width_y_min_spin = QDoubleSpinBox()
        self.width_y_max_spin = QDoubleSpinBox()
        for spin in (self.x_min_spin, self.x_max_spin, self.width_y_min_spin, self.width_y_max_spin):
            spin.setRange(0.0, 1_000_000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
        self.x_min_spin.setValue(0.1)
        self.x_max_spin.setValue(10.0)
        self.width_y_min_spin.setValue(0.1)
        self.width_y_max_spin.setValue(10.0)
        self.apply_button = QPushButton("Apply")
        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Loaded session summary will appear here.")

        self.fig = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self.width_ax = self.fig.add_subplot(311)
        self.angle_ax = self.fig.add_subplot(312)
        self.resid_ax = self.fig.add_subplot(313)
        self._init_axes()

        controls = QFormLayout()
        controls.addRow("", self.load_button)
        controls.addRow("", self.clear_button)
        controls.addRow("", self.save_figure_button)
        controls.addRow("", self.save_settings_button)
        controls.addRow("", self.load_settings_button)
        controls.addRow("", self.show_raw_width_check)
        controls.addRow("", self.show_raw_angle_check)
        controls.addRow("", self.show_paper_model_check)
        controls.addRow("", self.plot_style_button)
        controls.addRow("", self.auto_x_range_check)
        self.x_min_label = QLabel("Plot X Min")
        self.x_max_label = QLabel("Plot X Max")
        self.width_y_min_label = QLabel("Width Y Min")
        self.width_y_max_label = QLabel("Width Y Max")
        controls.addRow(self.x_min_label, self.x_min_spin)
        controls.addRow(self.x_max_label, self.x_max_spin)
        controls.addRow("", self.auto_width_y_range_check)
        controls.addRow(self.width_y_min_label, self.width_y_min_spin)
        controls.addRow(self.width_y_max_label, self.width_y_max_spin)
        controls.addRow("", self.apply_button)

        side_widget = QWidget()
        side_layout = QVBoxLayout(side_widget)
        side_layout.addLayout(controls)
        side_layout.addWidget(QLabel("Loaded Sessions / Saved Trend Summary"))
        side_layout.addWidget(self.summary_box, stretch=1)
        side_widget.setMinimumWidth(320)
        side_widget.setMaximumWidth(420)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.canvas)
        splitter.addWidget(side_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1100, 360])

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(splitter, stretch=1)

        screen = ensure_qt_app().primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            max_w = max(1000, int(geom.width()) - 64)
            max_h = max(760, int(geom.height()) - 64)
            self.setMaximumSize(max_w, max_h)
            self.resize(min(max_w, 1500), min(max_h, 980))

        self.load_button.clicked.connect(self._load_jsons)
        self.clear_button.clicked.connect(self._clear_sessions)
        self.save_figure_button.clicked.connect(self._save_figure)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.load_settings_button.clicked.connect(self._load_settings)
        self.apply_button.clicked.connect(self._update_plot)
        self.plot_style_button.clicked.connect(self._open_plot_style_dialog)
        self.auto_x_range_check.toggled.connect(self._update_axis_controls)
        self.auto_width_y_range_check.toggled.connect(self._update_axis_controls)

        self._update_axis_controls()
        if initial_paths:
            self._load_paths(list(initial_paths))
        else:
            self._update_plot()

    def _init_axes(self) -> None:
        self.width_ax.clear()
        self.angle_ax.clear()
        self.resid_ax.clear()
        self.width_ax.set_title("Width vs core distance")
        self.width_ax.set_xlabel("Distance from core")
        self.width_ax.set_ylabel("Width")
        self.angle_ax.set_title("Opening angle vs core distance")
        self.angle_ax.set_xlabel("Distance from core")
        self.angle_ax.set_ylabel("Opening angle (deg)")
        self.resid_ax.set_title("Residual vs core distance")
        self.resid_ax.set_xlabel("Distance from core")
        self.resid_ax.set_ylabel("log-width residual")
        for ax in (self.width_ax, self.angle_ax, self.resid_ax):
            ax.grid(alpha=0.25)

    def _update_axis_controls(self) -> None:
        self.x_min_spin.setEnabled(not bool(self.auto_x_range_check.isChecked()))
        self.x_max_spin.setEnabled(not bool(self.auto_x_range_check.isChecked()))
        self.width_y_min_spin.setEnabled(not bool(self.auto_width_y_range_check.isChecked()))
        self.width_y_max_spin.setEnabled(not bool(self.auto_width_y_range_check.isChecked()))

    def _open_plot_style_dialog(self) -> None:
        if not self.sessions:
            self.summary_box.setPlainText("No sessions loaded.")
            return
        session_labels = [session.label for session in self.sessions]
        for idx, label in enumerate(session_labels):
            self.session_colors.setdefault(
                label,
                _normalize_mpl_color(self._color_cycle[idx % len(self._color_cycle)], "#1f77b4"),
            )
        dialog = QtComparePlotStyleDialog(
            session_labels=session_labels,
            session_colors=self.session_colors,
            style=self.plot_style,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        settings = dialog.get_settings()
        self.plot_style["main_size"] = float(settings.get("main_size", self.plot_style["main_size"]))
        self.plot_style["raw_size"] = float(settings.get("raw_size", self.plot_style["raw_size"]))
        self.plot_style["residual_size"] = float(settings.get("residual_size", self.plot_style["residual_size"]))
        loaded_colors = settings.get("session_colors", {})
        if isinstance(loaded_colors, dict):
            self.session_colors = {
                str(label): _normalize_mpl_color(color, "#1f77b4")
                for label, color in loaded_colors.items()
            }
        self._update_plot()

    def _settings_payload(self) -> Dict[str, object]:
        return {
            "version": 1,
            "session_paths": [session.path for session in self.sessions],
            "auto_x_range": bool(self.auto_x_range_check.isChecked()),
            "plot_x_min": float(self.x_min_spin.value()),
            "plot_x_max": float(self.x_max_spin.value()),
            "auto_width_y_range": bool(self.auto_width_y_range_check.isChecked()),
            "width_y_min": float(self.width_y_min_spin.value()),
            "width_y_max": float(self.width_y_max_spin.value()),
            "show_raw_width": bool(self.show_raw_width_check.isChecked()),
            "show_raw_angle": bool(self.show_raw_angle_check.isChecked()),
            "show_paper_model": bool(self.show_paper_model_check.isChecked()),
            "plot_style": {
                "main_size": float(_safe_float(self.plot_style.get("main_size", 30.0), 30.0)),
                "raw_size": float(_safe_float(self.plot_style.get("raw_size", 26.0), 26.0)),
                "residual_size": float(_safe_float(self.plot_style.get("residual_size", 18.0), 18.0)),
                "session_colors": {
                    str(label): _normalize_mpl_color(color, "#1f77b4")
                    for label, color in self.session_colors.items()
                },
            },
        }

    def _apply_settings_payload(self, payload: Dict[str, object]) -> None:
        if not isinstance(payload, dict):
            return
        self._sync_enabled = False
        self.auto_x_range_check.setChecked(bool(payload.get("auto_x_range", True)))
        self.x_min_spin.setValue(float(_safe_float(payload.get("plot_x_min", self.x_min_spin.value()), self.x_min_spin.value())))
        self.x_max_spin.setValue(float(_safe_float(payload.get("plot_x_max", self.x_max_spin.value()), self.x_max_spin.value())))
        self.auto_width_y_range_check.setChecked(bool(payload.get("auto_width_y_range", True)))
        self.width_y_min_spin.setValue(float(_safe_float(payload.get("width_y_min", self.width_y_min_spin.value()), self.width_y_min_spin.value())))
        self.width_y_max_spin.setValue(float(_safe_float(payload.get("width_y_max", self.width_y_max_spin.value()), self.width_y_max_spin.value())))
        self.show_raw_width_check.setChecked(bool(payload.get("show_raw_width", self.show_raw_width_check.isChecked())))
        self.show_raw_angle_check.setChecked(bool(payload.get("show_raw_angle", self.show_raw_angle_check.isChecked())))
        self.show_paper_model_check.setChecked(bool(payload.get("show_paper_model", self.show_paper_model_check.isChecked())))
        plot_style = payload.get("plot_style", {})
        if isinstance(plot_style, dict):
            self.plot_style["main_size"] = float(_safe_float(plot_style.get("main_size", self.plot_style["main_size"]), self.plot_style["main_size"]))
            self.plot_style["raw_size"] = float(_safe_float(plot_style.get("raw_size", self.plot_style["raw_size"]), self.plot_style["raw_size"]))
            self.plot_style["residual_size"] = float(_safe_float(plot_style.get("residual_size", self.plot_style["residual_size"]), self.plot_style["residual_size"]))
            loaded_colors = plot_style.get("session_colors", {})
            if isinstance(loaded_colors, dict):
                self.session_colors = {
                    str(label): _normalize_mpl_color(color, "#1f77b4")
                    for label, color in loaded_colors.items()
                }
        self._sync_enabled = True
        self._update_axis_controls()

    def _load_jsons(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Ridgeline Analysis JSONs",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not paths:
            return
        self._load_paths(paths)

    def _load_paths(self, paths: Sequence[str]) -> None:
        loaded: List[LoadedTrendSession] = []
        errors: List[str] = []
        for raw_path in list(paths):
            path = str(raw_path or "").strip()
            if not path:
                continue
            try:
                payload = load_analysis_session(path)
                measurement_result = payload.get("measurement_result", None)
                if not isinstance(measurement_result, dict):
                    raise ValueError("missing measurement_result")
                calibration = payload.get("calibration", {})
                if not isinstance(calibration, dict):
                    calibration = {}
                trend_report = payload.get("trend_report", {})
                if not isinstance(trend_report, dict):
                    trend_report = {}
                loaded.append(
                    LoadedTrendSession(
                        path=path,
                        label=Path(path).stem,
                        image_path=str(payload.get("image_path", "") or ""),
                        calibration=dict(calibration),
                        measurement_result=dict(measurement_result),
                        trend_report=dict(trend_report),
                    )
                )
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        self.sessions = loaded
        new_colors: Dict[str, str] = {}
        for idx, session in enumerate(self.sessions):
            new_colors[session.label] = _normalize_mpl_color(
                self.session_colors.get(session.label, self._color_cycle[idx % len(self._color_cycle)]),
                "#1f77b4",
            )
        self.session_colors = new_colors
        self.distance_unit = _distance_unit_for_sessions(self.sessions)
        self.x_min_label.setText(f"Plot X Min ({self.distance_unit})")
        self.x_max_label.setText(f"Plot X Max ({self.distance_unit})")
        self.width_y_min_label.setText(f"Width Y Min ({self.distance_unit})")
        self.width_y_max_label.setText(f"Width Y Max ({self.distance_unit})")
        self.show_paper_model_check.setEnabled(self.distance_unit == "mas")
        if self.distance_unit != "mas":
            self.show_paper_model_check.setChecked(False)
        self._update_plot(extra_messages=errors)

    def _clear_sessions(self) -> None:
        self.sessions = []
        self.session_colors = {}
        self.distance_unit = "px"
        self._update_plot()

    def _save_figure(self) -> None:
        if not self.sessions:
            self.summary_box.setPlainText("No sessions loaded.")
            return
        default_name = "trend_report_compare.png"
        if self.sessions:
            default_name = f"{self.sessions[0].label}_trend_compare.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Figure",
            default_name,
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)",
        )
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            current = self.summary_box.toPlainText().strip()
            suffix = f"\nSaved figure: {path}"
            self.summary_box.setPlainText((current + suffix).strip())
        except Exception as exc:
            current = self.summary_box.toPlainText().strip()
            suffix = f"\nSave figure failed: {exc}"
            self.summary_box.setPlainText((current + suffix).strip())

    def _save_settings(self) -> None:
        default_name = "trend_report_compare_settings.json"
        if self.sessions:
            default_name = f"{self.sessions[0].label}_trend_compare_settings.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Compare Viewer Settings",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._settings_payload(), f, ensure_ascii=True, indent=2)
            current = self.summary_box.toPlainText().strip()
            suffix = f"\nSaved settings: {path}"
            self.summary_box.setPlainText((current + suffix).strip())
        except Exception as exc:
            current = self.summary_box.toPlainText().strip()
            suffix = f"\nSave settings failed: {exc}"
            self.summary_box.setPlainText((current + suffix).strip())

    def _load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Compare Viewer Settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            messages: List[str] = [f"Loaded settings: {path}"]
            session_paths = payload.get("session_paths", [])
            valid_paths = [str(p) for p in list(session_paths) if str(p or "").strip()]
            if valid_paths:
                existing_paths = [p for p in valid_paths if Path(p).exists()]
                missing_paths = [p for p in valid_paths if not Path(p).exists()]
                if existing_paths:
                    self._load_paths(existing_paths)
                if missing_paths:
                    messages.extend([f"Missing session path: {p}" for p in missing_paths])
            self._apply_settings_payload(payload)
            self._update_plot(extra_messages=messages)
        except Exception as exc:
            current = self.summary_box.toPlainText().strip()
            suffix = f"\nLoad settings failed: {exc}"
            self.summary_box.setPlainText((current + suffix).strip())

    def _collect_session_rows(self) -> Dict[str, List[Dict[str, object]]]:
        rows_by_path: Dict[str, List[Dict[str, object]]] = {}
        for session in self.sessions:
            rows_by_path[session.path] = _build_rows_for_session(
                session,
                distance_unit=self.distance_unit,
            )
        return rows_by_path

    def _auto_ranges(self, rows_by_path: Dict[str, List[Dict[str, object]]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        x_vals: List[float] = []
        y_vals: List[float] = []
        x_key = _distance_key(self.distance_unit)
        y_key = _width_key(self.distance_unit)
        for rows in rows_by_path.values():
            for row in rows:
                xv = _safe_float(row.get(x_key, float("nan")))
                yv = _safe_float(row.get(y_key, float("nan")))
                if np.isfinite(xv) and xv > 0.0:
                    x_vals.append(float(xv))
                if np.isfinite(xv) and xv > 0.0 and np.isfinite(yv) and yv > 0.0:
                    y_vals.append(float(yv))
        if not x_vals:
            return None, None, None, None
        x_arr = np.asarray(x_vals, dtype=np.float64)
        y_arr = np.asarray(y_vals, dtype=np.float64) if y_vals else np.asarray([], dtype=np.float64)
        xmin = float(np.min(x_arr))
        xmax = float(np.max(x_arr))
        ymin = float(np.min(y_arr)) if y_arr.size > 0 else None
        ymax = float(np.max(y_arr)) if y_arr.size > 0 else None
        return xmin, xmax, ymin, ymax

    def _update_plot(self, *, extra_messages: Optional[Sequence[str]] = None) -> None:
        self._init_axes()
        messages: List[str] = list(extra_messages or [])
        if not self.sessions:
            self.summary_box.setPlainText("No sessions loaded.")
            self.canvas.draw_idle()
            return

        rows_by_path = self._collect_session_rows()
        auto_xmin, auto_xmax, auto_ymin, auto_ymax = self._auto_ranges(rows_by_path)
        if bool(self.auto_x_range_check.isChecked()) and auto_xmin is not None and auto_xmax is not None:
            self._sync_enabled = False
            self.x_min_spin.setValue(float(auto_xmin))
            self.x_max_spin.setValue(float(auto_xmax))
            self._sync_enabled = True
        if bool(self.auto_width_y_range_check.isChecked()) and auto_ymin is not None and auto_ymax is not None:
            self._sync_enabled = False
            self.width_y_min_spin.setValue(float(auto_ymin))
            self.width_y_max_spin.setValue(float(auto_ymax))
            self._sync_enabled = True

        plot_x_min = float(self.x_min_spin.value()) if np.isfinite(self.x_min_spin.value()) and self.x_min_spin.value() > 0.0 else None
        plot_x_max = float(self.x_max_spin.value()) if np.isfinite(self.x_max_spin.value()) and self.x_max_spin.value() > 0.0 else None
        if plot_x_min is not None and plot_x_max is not None and plot_x_max <= plot_x_min:
            plot_x_max = None

        x_key = _distance_key(self.distance_unit)
        width_key = _width_key(self.distance_unit)
        raw_width_key = _raw_width_key(self.distance_unit)
        show_paper_model = bool(self.show_paper_model_check.isChecked()) and self.distance_unit == "mas"
        main_size = float(max(2.0, _safe_float(self.plot_style.get("main_size", 30.0), 30.0)))
        raw_size = float(max(2.0, _safe_float(self.plot_style.get("raw_size", 26.0), 26.0)))
        residual_size = float(max(2.0, _safe_float(self.plot_style.get("residual_size", 18.0), 18.0)))

        summary_lines: List[str] = []
        for idx, session in enumerate(self.sessions):
            rows_all = list(rows_by_path.get(session.path, []))
            rows = _filter_rows_by_x_range(rows_all, distance_unit=self.distance_unit, x_min=plot_x_min, x_max=plot_x_max)
            color = _normalize_mpl_color(
                self.session_colors.get(session.label, self._color_cycle[idx % len(self._color_cycle)]),
                "#1f77b4",
            )
            raw_color = _lighten_color(color, 0.50)
            label = session.label

            x = np.asarray([_safe_float(row.get(x_key, float("nan"))) for row in rows], dtype=np.float64)
            width = np.asarray([_safe_float(row.get(width_key, float("nan"))) for row in rows], dtype=np.float64)
            raw_width = np.asarray([_safe_float(row.get(raw_width_key, float("nan"))) for row in rows], dtype=np.float64)
            angle = np.asarray([_safe_float(row.get("gaussian_angle_deg", float("nan"))) for row in rows], dtype=np.float64)
            raw_angle = np.asarray([_safe_float(row.get("gaussian_raw_angle_deg", float("nan"))) for row in rows], dtype=np.float64)

            mask_w = np.isfinite(x) & np.isfinite(width) & (x > 0.0) & (width > 0.0)
            mask_raw_w = np.isfinite(x) & np.isfinite(raw_width) & (x > 0.0) & (raw_width > 0.0)
            mask_a = np.isfinite(x) & np.isfinite(angle) & (x > 0.0)
            mask_raw_a = np.isfinite(x) & np.isfinite(raw_angle) & (x > 0.0)

            if np.any(mask_w):
                self.width_ax.scatter(
                    x[mask_w],
                    width[mask_w],
                    s=main_size,
                    c=[color],
                    marker="s",
                    alpha=0.85,
                    edgecolors="none",
                    label=f"{label} width",
                )
            if bool(self.show_raw_width_check.isChecked()) and np.any(mask_raw_w):
                self.width_ax.scatter(
                    x[mask_raw_w],
                    raw_width[mask_raw_w],
                    s=raw_size,
                    marker="s",
                    facecolors="none",
                    edgecolors=[raw_color],
                    linewidths=1.0,
                    alpha=0.55,
                    label=f"{label} raw",
                )
            if np.any(mask_a):
                self.angle_ax.scatter(
                    x[mask_a],
                    angle[mask_a],
                    s=main_size,
                    c=[color],
                    marker="s",
                    alpha=0.85,
                    edgecolors="none",
                    label=f"{label} angle",
                )
            if bool(self.show_raw_angle_check.isChecked()) and np.any(mask_raw_a):
                self.angle_ax.scatter(
                    x[mask_raw_a],
                    raw_angle[mask_raw_a],
                    s=raw_size,
                    marker="s",
                    facecolors="none",
                    edgecolors=[raw_color],
                    linewidths=1.0,
                    alpha=0.55,
                    label=f"{label} raw",
                )

            trend_result = None
            loaded_trend = session.trend_report.get("trend_result", None)
            if isinstance(loaded_trend, dict):
                trend_result = dict(loaded_trend)

            if trend_result is not None:
                x_all = np.asarray(trend_result.get("x_all", np.array([], dtype=np.float32)), dtype=np.float64)
                width_model = np.asarray(trend_result.get("width_model", np.array([], dtype=np.float32)), dtype=np.float64)
                opening_model = np.asarray(trend_result.get("opening_model_deg", np.array([], dtype=np.float32)), dtype=np.float64)
                residual = np.asarray(trend_result.get("log_width_residual_all", np.array([], dtype=np.float32)), dtype=np.float64)
                sigma_theta = _safe_float(trend_result.get("opening_sigma_deg", float("nan")))
                sigma_log_w = _safe_float(trend_result.get("log_width_sigma_dex", float("nan")))
                model_mask = np.isfinite(x_all) & (x_all > 0.0)
                if plot_x_min is not None:
                    model_mask &= x_all >= plot_x_min
                if plot_x_max is not None:
                    model_mask &= x_all <= plot_x_max
                model_w_mask = model_mask & np.isfinite(width_model) & (width_model > 0.0)
                model_o_mask = model_mask & np.isfinite(opening_model)
                resid_mask = model_mask & np.isfinite(residual)
                if np.any(model_w_mask):
                    self.width_ax.plot(x_all[model_w_mask], width_model[model_w_mask], color=color, lw=1.8, ls="--")
                if np.any(model_o_mask):
                    self.angle_ax.plot(x_all[model_o_mask], opening_model[model_o_mask], color=color, lw=1.8, ls="--")
                    if np.isfinite(sigma_theta) and sigma_theta > 0.0:
                        self.angle_ax.fill_between(
                            x_all[model_o_mask],
                            opening_model[model_o_mask] - sigma_theta,
                            opening_model[model_o_mask] + sigma_theta,
                            color=color,
                            alpha=0.10,
                        )
                if np.any(resid_mask):
                    self.resid_ax.scatter(
                        x_all[resid_mask],
                        residual[resid_mask],
                        s=residual_size,
                        c=[color],
                        alpha=0.80,
                        edgecolors="none",
                        label=label,
                    )
                    if np.isfinite(sigma_log_w) and sigma_log_w > 0.0:
                        self.resid_ax.fill_between(
                            x_all[resid_mask],
                            -sigma_log_w,
                            sigma_log_w,
                            color=color,
                            alpha=0.08,
                        )
                summary_lines.append(
                    f"{label}: rows={len(rows)} | k_fit={_format_float_or_na(trend_result.get('k_fit', float('nan')), '.4g')} | "
                    f"opening={_format_float_or_na(trend_result.get('opening_fit_median_deg', float('nan')), '.4g')} deg | "
                    f"sigma_theta={_format_float_or_na(trend_result.get('opening_sigma_deg', float('nan')), '.4g')} deg | "
                    f"sigma_logW={_format_float_or_na(trend_result.get('log_width_sigma_dex', float('nan')), '.4g')} dex"
                )
            else:
                summary_lines.append(f"{label}: rows={len(rows)} | no saved trend result")

        if show_paper_model:
            z_min = plot_x_min
            z_max = plot_x_max
            if z_min is None or z_max is None or z_max <= z_min:
                x_for_model: List[float] = []
                for rows in rows_by_path.values():
                    for row in rows:
                        xv = _safe_float(row.get(x_key, float("nan")))
                        if np.isfinite(xv) and xv > 0.0:
                            x_for_model.append(float(xv))
                if x_for_model:
                    z_min = float(np.min(x_for_model))
                    z_max = float(np.max(x_for_model))
            if z_min is not None and z_max is not None and z_max > z_min:
                paper_break_mas = 1.51
                z = np.logspace(np.log10(z_min), np.log10(z_max), 400)
                paper_width = np.asarray(paper_fig7_eastern_broken_power_law(z), dtype=np.float64)
                paper_open = np.asarray(
                    [opening_angle_deg(width_mas, dist_mas) for width_mas, dist_mas in zip(paper_width.tolist(), z.tolist())],
                    dtype=np.float64,
                )
                paper_width_at_break = float(paper_fig7_eastern_broken_power_law([paper_break_mas])[0])
                paper_open_at_break = float(opening_angle_deg(paper_width_at_break, paper_break_mas))
                width_mask = np.isfinite(paper_width) & (paper_width > 0.0)
                open_mask = np.isfinite(paper_open)
                if np.any(width_mask):
                    self.width_ax.plot(
                        z[width_mask],
                        paper_width[width_mask],
                        color="black",
                        lw=2.0,
                        ls=":",
                        label="Paper Fig.7 eastern jet broken PL",
                    )
                if np.any(open_mask):
                    self.angle_ax.plot(
                        z[open_mask],
                        paper_open[open_mask],
                        color="black",
                        lw=2.0,
                        ls=":",
                        label=f"Paper opening-angle model (theta@z_b={paper_open_at_break:.2f} deg)",
                    )
                if paper_break_mas >= z_min and paper_break_mas <= z_max:
                    self.width_ax.axvline(
                        paper_break_mas,
                        color="black",
                        lw=1.2,
                        ls="-.",
                        alpha=0.9,
                        label=f"Paper break z_b={paper_break_mas:.2f} mas",
                    )
                    self.angle_ax.axvline(
                        paper_break_mas,
                        color="black",
                        lw=1.2,
                        ls="-.",
                        alpha=0.9,
                    )
                    self.resid_ax.axvline(
                        paper_break_mas,
                        color="black",
                        lw=1.2,
                        ls="-.",
                        alpha=0.9,
                    )
                summary_lines.append(
                    f"Paper Fig.7 model: z_b={paper_break_mas:.2f} mas | width(z_b)={paper_width_at_break:.3f} mas | "
                    f"opening(z_b)={paper_open_at_break:.3f} deg"
                )

        self.resid_ax.axhline(0.0, color="tab:red", lw=1.0, ls="--")
        self.width_ax.set_xscale("log")
        self.width_ax.set_yscale("log")
        self.angle_ax.set_xscale("log")
        self.resid_ax.set_xscale("log")
        unit = self.distance_unit
        self.width_ax.set_xlabel(f"Distance from core ({unit})")
        self.width_ax.set_ylabel(f"Width ({unit})")
        self.angle_ax.set_xlabel(f"Distance from core ({unit})")
        self.resid_ax.set_xlabel(f"Distance from core ({unit})")
        if plot_x_min is not None and plot_x_max is not None and plot_x_max > plot_x_min:
            for ax in (self.width_ax, self.angle_ax, self.resid_ax):
                ax.set_xlim(plot_x_min, plot_x_max)
        if not bool(self.auto_width_y_range_check.isChecked()):
            ymin = float(self.width_y_min_spin.value())
            ymax = float(self.width_y_max_spin.value())
            if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0.0 and ymax > ymin:
                self.width_ax.set_ylim(ymin, ymax)
        for ax in (self.width_ax, self.angle_ax, self.resid_ax):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best")

        combined_summary = [
            f"Sessions loaded: {len(self.sessions)} | Unit: {unit} | Mode: Saved Trend | "
            f"plot range={_format_float_or_na(plot_x_min, '.4g')} - {_format_float_or_na(plot_x_max, '.4g')} {unit}"
        ]
        combined_summary.extend(summary_lines)
        if messages:
            combined_summary.append("")
            combined_summary.append("Messages:")
            combined_summary.extend(str(msg) for msg in messages)
        self.summary_box.setPlainText("\n".join(combined_summary))
        self.canvas.draw_idle()


def run_trend_report_compare_qt(paths: Optional[Sequence[str]] = None) -> QtTrendReportCompareDialog:
    ensure_qt_app()
    dialog = QtTrendReportCompareDialog(initial_paths=paths)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog.exec_()
    return dialog


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare multiple IBAE trend-report JSON sessions.")
    parser.add_argument("json_paths", nargs="*", help="Ridgeline analysis JSON files to preload.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_trend_report_compare_qt(args.json_paths or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
