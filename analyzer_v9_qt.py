from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import analyzer_v8 as cv_v8_mod
from . import gui_v9_qt as qt_v9_mod
from .session_analysis import load_analysis_session, load_reconstruction_cache
from .session_replay import load_replay_session, normalize_replay_session_payload
from .analyzer_v8 import JetAnalyzerV8Simple, _build_roi_from_polygon


IBAE_VERSION_NAME = "IBAE_v1"


def _ibae_package_version() -> str:
    try:
        return str(version("IBAE"))
    except PackageNotFoundError:
        return "0.1.0"


def _startup_banner(image_path: str) -> str:
    return f"=== Jet Analyzer v9 (Qt) / {IBAE_VERSION_NAME} v{_ibae_package_version()} Started: {image_path} ==="


def _normalize_l0_l1_transition_mode(mode: object) -> str:
    mode_norm = str(mode or "gaussian").strip().lower()
    if mode_norm in {"flat", "none", "off", "disabled", "legacy"}:
        return "flat"
    return "gaussian"


class JetAnalyzerV9Qt(JetAnalyzerV8Simple):
    """
    v9 Qt GUI version.

    Processing logic stays aligned with v8, while GUI interactions move from
    OpenCV HighGUI to PyQt5/Qt.
    """

    def __init__(self, image_path: str, config: Optional[dict] = None):
        qt_v9_mod.ensure_qt_app()
        super().__init__(image_path=image_path, config=config)
        self._startup_loaded_analysis_payload: Optional[Dict[str, object]] = None
        self._startup_loaded_analysis_path: str = ""
        self._startup_cached_reconstruction_payload: Optional[Dict[str, object]] = None

    @staticmethod
    def _image_path_matches_or_moved_copy(saved_path: object, current_path: object) -> bool:
        saved = str(saved_path or "").strip()
        current = str(current_path or "").strip()
        if not saved or not current:
            return True
        saved_p = Path(saved)
        current_p = Path(current)
        try:
            if saved_p.resolve() == current_p.resolve():
                return True
        except Exception:
            pass
        if saved_p.name != current_p.name:
            return False
        if not saved_p.is_absolute():
            return True
        try:
            if saved_p.exists() and current_p.exists():
                return int(saved_p.stat().st_size) == int(current_p.stat().st_size)
        except Exception:
            return False
        return False

    @staticmethod
    def _is_missing_session_value(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value == ""
        if isinstance(value, (list, tuple, dict)):
            return len(value) == 0
        return False

    @staticmethod
    def _merge_analysis_replay_defaults(
        replay_payload: Dict[str, object],
        analysis_payload: Dict[str, object],
    ) -> Dict[str, object]:
        merged = dict(replay_payload or {})
        for key in (
            "image_path",
            "roi_points_xy",
            "binary_prep_mode",
            "manual_gray_thresh",
            "manual_threshold_invert",
            "binary_split_cuts",
            "junction_cuts",
        ):
            current = merged.get(key, None)
            if JetAnalyzerV9Qt._is_missing_session_value(current) and key in analysis_payload:
                merged[key] = analysis_payload.get(key)
        return merged

    def _load_startup_session_from_json(self, path: str) -> Dict[str, object]:
        try:
            loaded = load_analysis_session(path)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            replay_source = loaded.get("replay_snapshot", None)
            has_replay_snapshot = isinstance(replay_source, dict)
            looks_like_analysis = (
                has_replay_snapshot
                or "flux_reconstruction" in loaded
                or "measurement_result" in loaded
                or "ridgeline" in loaded
                or "trend_report" in loaded
                or "calibration" in loaded
            )
            if looks_like_analysis:
                analysis_payload = dict(loaded)
                replay_payload = self._merge_analysis_replay_defaults(
                    dict(replay_source if has_replay_snapshot else {}),
                    analysis_payload,
                )
                expected_image = self.image_path if self._simple_replay_strict_image_match() else None
                try:
                    replay_session = normalize_replay_session_payload(
                        replay_payload,
                        expected_image_path=expected_image,
                    )
                except ValueError:
                    saved_image = str(replay_payload.get("image_path", "") or analysis_payload.get("image_path", "") or "")
                    if not self._image_path_matches_or_moved_copy(saved_image, self.image_path):
                        raise
                    print(
                        "-> Analysis JSON image path differs from the current image path, "
                        "but the image name/size matches; loading as a moved copy."
                    )
                    replay_session = normalize_replay_session_payload(
                        replay_payload,
                        expected_image_path=None,
                    )
                self._startup_loaded_analysis_payload = analysis_payload
                self._startup_loaded_analysis_path = str(path)
                self._loaded_replay_json_path = str(path)
                return replay_session
            if "roi_points_xy" in loaded:
                replay_session = normalize_replay_session_payload(
                    loaded,
                    expected_image_path=self.image_path if self._simple_replay_strict_image_match() else None,
                )
                self._startup_loaded_analysis_payload = None
                self._startup_loaded_analysis_path = ""
                self._startup_cached_reconstruction_payload = None
                self._loaded_replay_json_path = str(path)
                return replay_session
        replay_session = load_replay_session(
            path,
            expected_image_path=self.image_path if self._simple_replay_strict_image_match() else None,
        )
        self._startup_loaded_analysis_payload = None
        self._startup_loaded_analysis_path = ""
        self._startup_cached_reconstruction_payload = None
        self._loaded_replay_json_path = str(path)
        return replay_session

    def _load_replay_session_from_config(self) -> Optional[Dict[str, object]]:
        path = self._simple_replay_json_load_path()
        if not path:
            return None
        return self._load_startup_session_from_json(path)

    @staticmethod
    def _resolve_cache_path(cache_info: Dict[str, object], source_path: str) -> Optional[Path]:
        raw = str(cache_info.get("path", "") or cache_info.get("npz_path", "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        if source_path:
            return Path(source_path).parent / path
        return path

    @staticmethod
    def _float_matches(value: object, expected: object, *, atol: float = 1e-9) -> bool:
        if value is None and expected is None:
            return True
        try:
            val = float(value)
            exp = float(expected)
        except Exception:
            return False
        return bool(np.isfinite(val) and np.isfinite(exp) and abs(val - exp) <= atol)

    @staticmethod
    def _custom_values_match(value: object, expected: object) -> bool:
        if value is None:
            saved = []
        elif isinstance(value, (list, tuple)):
            try:
                saved = [float(v) for v in value]
            except Exception:
                return False
        else:
            return False
        if expected is None:
            exp = []
        elif isinstance(expected, (list, tuple)):
            try:
                exp = [float(v) for v in expected]
            except Exception:
                return False
        else:
            return False
        if len(saved) != len(exp):
            return False
        if not saved:
            return True
        return bool(np.allclose(np.asarray(saved, dtype=float), np.asarray(exp, dtype=float), rtol=0.0, atol=1e-9))

    def _cache_metadata_matches_payload(self, metadata: Dict[str, object], payload: Dict[str, object]) -> bool:
        flux_rec = payload.get("flux_reconstruction", {})
        if not isinstance(flux_rec, dict):
            return False
        checks = (
            ("background_flux", "background_flux"),
            ("l1_flux", "l1_flux"),
            ("level_ratio", "level_ratio"),
            ("smooth_sigma_px", "sigma_px"),
        )
        for meta_key, payload_key in checks:
            expected = flux_rec.get(payload_key, None)
            if expected is None:
                continue
            if not self._float_matches(metadata.get(meta_key, None), expected):
                return False
        if not self._custom_values_match(
            metadata.get("custom_contour_values", None),
            flux_rec.get("custom_contour_values", None),
        ):
            return False
        expected_transition = _normalize_l0_l1_transition_mode(
            flux_rec.get("l0_l1_transition_mode", "gaussian")
        )
        stored_transition = _normalize_l0_l1_transition_mode(
            metadata.get("l0_l1_transition_mode", "flat")
        )
        if stored_transition != expected_transition:
            return False
        if expected_transition == "gaussian":
            expected_alpha = flux_rec.get("l0_l1_transition_alpha", 3.0)
            if not self._float_matches(metadata.get("l0_l1_transition_alpha", 3.0), expected_alpha):
                return False
            expected_width = flux_rec.get("l0_l1_transition_width_px", None)
            if expected_width is not None:
                try:
                    expected_width_f = float(expected_width)
                except Exception:
                    expected_width_f = float("nan")
                if np.isfinite(expected_width_f) and expected_width_f > 0.0:
                    if not self._float_matches(metadata.get("l0_l1_transition_width_px", None), expected_width_f):
                        return False
        image_path = str(payload.get("image_path", "") or "")
        if image_path and bool(self._simple_replay_strict_image_match()):
            if not self._image_path_matches_or_moved_copy(image_path, self.image_path):
                return False
        return True

    def _try_run_cached_reconstruction_preview(
        self,
        *,
        replay_session: Dict[str, object],
        source_path: str,
    ) -> bool:
        self._startup_cached_reconstruction_payload = None
        payload = self._startup_loaded_analysis_payload
        if not isinstance(payload, dict):
            return False
        flux_rec = payload.get("flux_reconstruction", {})
        if not isinstance(flux_rec, dict):
            return False
        cache_info = flux_rec.get("cache_npz", None)
        if not isinstance(cache_info, dict):
            return False
        cache_path = self._resolve_cache_path(cache_info, source_path)
        if cache_path is None or not cache_path.exists():
            return False

        try:
            cache = load_reconstruction_cache(str(cache_path))
        except Exception as exc:
            print(f"-> Cached reconstruction unavailable: {exc}")
            return False

        metadata = dict(cache.get("metadata", {}) or {})
        if not self._cache_metadata_matches_payload(metadata, payload):
            print("-> Cached reconstruction metadata does not match analysis JSON; falling back to replay.")
            return False

        flux = np.asarray(cache.get("flux_map", []), dtype=np.float32)
        valid_mask = (np.asarray(cache.get("valid_mask", []), dtype=np.uint8) > 0).astype(np.uint8)
        depth = cache.get("region_depth_map", None)
        roi_mask = cache.get("roi_mask", None)
        thin_mask = cache.get("thin_mask", None)
        if depth is None or roi_mask is None or thin_mask is None:
            print("-> Cached reconstruction is missing level-map arrays; falling back to replay.")
            return False
        depth = np.asarray(depth, dtype=np.int32)
        roi_mask = (np.asarray(roi_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        thin_mask = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
        if (
            flux.ndim != 2
            or depth.ndim != 2
            or roi_mask.ndim != 2
            or thin_mask.ndim != 2
            or valid_mask.ndim != 2
            or flux.shape[:2] != depth.shape[:2]
            or flux.shape[:2] != roi_mask.shape[:2]
            or flux.shape[:2] != thin_mask.shape[:2]
            or flux.shape[:2] != valid_mask.shape[:2]
        ):
            print("-> Cached reconstruction array shapes are inconsistent; falling back to replay.")
            return False

        points = [
            (int(pt[0]), int(pt[1]))
            for pt in list(replay_session.get("roi_points_xy", []))
            if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]
        if len(points) < 3:
            return False
        self.roi_polygon_points = list(points)
        self.points = list(points)
        try:
            (x, y, w, h), roi_img, _roi_mask_from_points = _build_roi_from_polygon(self.img, self.points)
            x, y, w, h = int(x), int(y), int(w), int(h)
        except Exception:
            x, y, w, h = 0, 0, int(depth.shape[1]), int(depth.shape[0])
            roi_img = np.zeros((int(depth.shape[0]), int(depth.shape[1]), 3), dtype=np.uint8)
        if roi_mask.shape[:2] != roi_img.shape[:2]:
            print("-> Cached reconstruction ROI shape does not match replay ROI; falling back to replay.")
            return False

        cleaned_mask = ((depth > 0) & (roi_mask > 0)).astype(np.uint8) * 255
        roi_view = self._masked_roi_gray_view(roi_img=roi_img, roi_mask=roi_mask)
        level_overlay_map = self._render_level_region_overlay(
            region_depth_map=depth,
            thin_mask=thin_mask,
            roi_mask=roi_mask,
        )
        preview = self._compose_simple_preview(
            roi_view=roi_view,
            cleaned_mask=cleaned_mask,
            thin_mask_for_levels=thin_mask,
            level_overlay_map=level_overlay_map,
            level_status_text=["Loaded cached reconstruction"],
        )

        max_level = int(np.nanmax(depth[(roi_mask > 0) & np.isfinite(depth)])) if np.any(roi_mask > 0) else 0
        self.final_result_img = np.asarray(preview, dtype=np.uint8).copy()
        self.simple_roi_view = np.asarray(roi_view, dtype=np.uint8).copy()
        self.simple_roi_mask = roi_mask.copy()
        self.simple_final_thin_mask = thin_mask.copy()
        self.simple_region_depth_map = depth.copy()
        self.simple_binary_prep_mode = str(replay_session.get("binary_prep_mode", self.simple_binary_prep_mode))
        self.results["roi_bbox_xywh"] = [int(x), int(y), int(w), int(h)]
        self.results["line_maps_coordinate_frame"] = "roi"
        self.results["line_maps_roi_bbox_xywh"] = [int(x), int(y), int(w), int(h)]
        self.results["line_maps_original_image_shape_hw"] = [int(self.img.shape[0]), int(self.img.shape[1])]
        self.results["simple_status"] = "cached_reconstruction"
        self.results["simple_message"] = f"Loaded cached reconstruction: {cache_path}"
        self.results["simple_clean_px"] = int(np.count_nonzero(cleaned_mask))
        self.results["simple_thin_px"] = int(np.count_nonzero(thin_mask))
        self.results["simple_final_thin_px"] = int(np.count_nonzero(thin_mask))
        self.results["simple_l1_px"] = int(np.count_nonzero((depth == 1) & (roi_mask > 0)))
        self.results["simple_l2_px"] = int(np.count_nonzero((depth == 2) & (roi_mask > 0)))
        self.results["simple_residual_px"] = 0
        self.results["simple_detected_level_count"] = int(len([v for v in np.unique(depth[roi_mask > 0]) if int(v) > 0]))
        self.results["simple_max_detected_level"] = int(max_level)
        self.results["simple_level_assignment_mode"] = "cached_reconstruction_npz"
        self.results["simple_replay_payload"] = dict(replay_session)
        self.results["simple_binary_prep_mode"] = self.simple_binary_prep_mode
        self.results["simple_replay_json_loaded"] = True
        self.results["simple_replay_json_loaded_path"] = str(source_path)
        self.results["simple_manual_binary_split_cuts"] = list(replay_session.get("binary_split_cuts", []))
        self.results["simple_manual_junction_cuts"] = list(replay_session.get("junction_cuts", []))
        self.results["v8_simple_mode"] = True
        self.results["direct_contour_pipeline_mode"] = "cached_reconstruction_npz"
        self._startup_cached_reconstruction_payload = {
            "flux_map": flux,
            "valid_mask": valid_mask,
            "metadata": metadata,
            "cache_path": str(cache_path),
        }

        print(f"-> Using cached reconstruction NPZ: {cache_path}")
        action = self._show_final_preview_window()
        if action == "back_to_junction":
            self._startup_cached_reconstruction_payload = None
            self._prepare_back_to_junction_from_preview()
            print("-> Back requested from cached preview; falling back to replay processing.")
            return False
        return True

    def _run_manual_threshold_tuner(self, roi_img: np.ndarray, roi_mask: np.ndarray) -> bool:
        result = qt_v9_mod.run_manual_threshold_dialog_qt(
            roi_img=roi_img,
            roi_mask=roi_mask,
            preview_callback=lambda threshold, invert: self._compose_manual_threshold_tuning_preview(
                roi_img=roi_img,
                roi_mask=roi_mask,
                gray_thresh=int(threshold),
                invert=bool(invert),
            ),
            initial_threshold=int(self.params.get("SIMPLE_MANUAL_GRAY_THRESH", 180)),
            initial_invert=bool(self.params.get("SIMPLE_MANUAL_THRESHOLD_INVERT", False)),
            max_width=self._simple_window_max_width(),
            max_height=self._simple_window_max_height(),
        )
        if result is None:
            self._last_cancel_reason = "Manual threshold tuning cancelled."
            return False
        threshold, invert = result
        self.params["SIMPLE_MANUAL_GRAY_THRESH"] = int(threshold)
        self.params["SIMPLE_MANUAL_THRESHOLD_INVERT"] = bool(invert)
        return True

    def _build_simple_payload(
        self,
        roi_img: np.ndarray,
        roi_mask: np.ndarray,
    ) -> Dict[str, object]:
        orig_binary_editor = cv_v8_mod.edit_binary_mask_splits
        orig_junction_editor = cv_v8_mod.edit_thin_mask_junctions
        cv_v8_mod.edit_binary_mask_splits = qt_v9_mod.edit_binary_mask_splits_qt
        cv_v8_mod.edit_thin_mask_junctions = qt_v9_mod.edit_thin_mask_junctions_qt
        try:
            return super()._build_simple_payload(roi_img=roi_img, roi_mask=roi_mask)
        finally:
            cv_v8_mod.edit_binary_mask_splits = orig_binary_editor
            cv_v8_mod.edit_thin_mask_junctions = orig_junction_editor

    def _show_final_preview_window(self) -> str:
        if self.final_result_img is None:
            return "close"
        reconstruction_context = None
        if self.simple_region_depth_map is not None and self.simple_roi_mask is not None and self.simple_final_thin_mask is not None:
            reconstruction_context = {
                "region_depth_map": np.asarray(self.simple_region_depth_map, dtype=np.int32),
                "roi_mask": np.asarray(self.simple_roi_mask, dtype=np.uint8),
                "thin_mask": np.asarray(self.simple_final_thin_mask, dtype=np.uint8),
                "max_detected_level": int(self.results.get("simple_max_detected_level", 0)),
                "max_width": int(self._simple_window_max_width()),
                "max_height": int(self._simple_window_max_height()),
                "analysis_context": {
                    "image_path": str(self.image_path),
                    "loaded_replay_json_path": str(self._loaded_replay_json_path),
                    "loaded_analysis_json_path": str(self._startup_loaded_analysis_path),
                    "roi_bbox_xywh": list(self.results.get("roi_bbox_xywh", [])),
                    "roi_points_xy": [[int(pt[0]), int(pt[1])] for pt in list(self.roi_polygon_points or self.points or []) if isinstance(pt, (list, tuple)) and len(pt) >= 2],
                    "binary_prep_mode": str(self.results.get("simple_binary_prep_mode", self.simple_binary_prep_mode)),
                    "replay_snapshot": dict(self.results.get("simple_replay_payload", {})),
                    "startup_loaded_analysis_payload": self._startup_loaded_analysis_payload,
                    "startup_cached_reconstruction_payload": self._startup_cached_reconstruction_payload,
                },
            }
        return qt_v9_mod.show_final_preview_qt(
            image=np.asarray(self.final_result_img, dtype=np.uint8),
            max_width=self._simple_window_max_width(),
            max_height=self._simple_window_max_height(),
            reconstruction_context=reconstruction_context,
        )

    def run(self):
        print(_startup_banner(self.image_path), flush=True)
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
            if self._try_run_cached_reconstruction_preview(
                replay_session=dict(replay_session),
                source_path=str(self._loaded_replay_json_path),
            ):
                return
            print("Workflow: replay JSON -> threshold binary -> manual binary split -> thin -> spur prune -> manual junction split -> level map [Qt GUI]")
            (x, y, w, h), roi_img, roi_mask = _build_roi_from_polygon(self.img, self.points)
            print(" -> Generating v9 Qt level-map preview ...")
            self._run_roi_processing_loop(
                roi_xywh=(int(x), int(y), int(w), int(h)),
                roi_img=roi_img,
                roi_mask=roi_mask,
            )
            return
        if self._simple_use_full_image_roi():
            print("Workflow: full image ROI -> threshold binary -> manual binary split -> thin -> spur prune -> manual junction split -> level map [Qt GUI]")
            if self.params.get("DEBUG_MODE", False):
                print(f"[DEBUG] Params: {self.params}")
            roi_mask = np.ones(self.img.shape[:2], dtype=np.uint8) * 255
            self._run_roi_processing_loop(
                roi_xywh=(0, 0, int(self.img.shape[1]), int(self.img.shape[0])),
                roi_img=self.img.copy(),
                roi_mask=roi_mask,
            )
            return

        print("Workflow: polygon ROI -> threshold binary -> manual binary split -> thin -> spur prune -> manual junction split -> level map [Qt GUI]")
        if self.params.get("DEBUG_MODE", False):
            print(f"[DEBUG] Params: {self.params}")
        selection = qt_v9_mod.select_polygon_roi_or_load_json_qt(
            image=self.clone_base,
            max_width=self._simple_window_max_width(),
            max_height=self._simple_window_max_height(),
        )
        if not isinstance(selection, dict):
            self.results["simple_status"] = "cancelled"
            self.results["simple_message"] = "ROI selection cancelled."
            print("-> ROI selection cancelled.")
            return
        action = str(selection.get("action", "cancel"))
        if action == "load_json":
            path = str(selection.get("path", "") or "")
            if not path:
                self.results["simple_status"] = "cancelled"
                self.results["simple_message"] = "JSON load cancelled."
                print("-> JSON load cancelled.")
                return
            replay_session = self._load_startup_session_from_json(path)
            print(f"-> Loading startup JSON: {path}")
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
                raise ValueError("Loaded JSON must contain at least 3 ROI points.")
        else:
            points = list(selection.get("points", []))

        if not points or len(points) < 3:
            self.results["simple_status"] = "cancelled"
            self.results["simple_message"] = "ROI selection cancelled."
            print("-> ROI selection cancelled.")
            return
        self.roi_polygon_points = list(points)
        self.points = list(points)
        if action == "load_json" and self._try_run_cached_reconstruction_preview(
            replay_session=dict(replay_session),
            source_path=str(self._loaded_replay_json_path),
        ):
            return
        print(" -> Generating v9 Qt level-map preview ...")
        (x, y, w, h), roi_img, roi_mask = _build_roi_from_polygon(self.img, self.points)
        self._run_roi_processing_loop(
            roi_xywh=(int(x), int(y), int(w), int(h)),
            roi_img=roi_img,
            roi_mask=roi_mask,
        )


def run_v9_preview(image_path: str, config: Optional[dict] = None) -> JetAnalyzerV9Qt:
    qt_v9_mod.ensure_qt_app()
    analyzer = JetAnalyzerV9Qt(image_path=image_path, config=config)
    analyzer.run()
    return analyzer
