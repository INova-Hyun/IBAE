from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = ROOT / "data" / "tests"


def _install_local_package_alias() -> None:
    """Run regression tests against this checkout even when old IBAE exists."""
    init_path = (ROOT / "__init__.py").resolve()
    existing = sys.modules.get("IBAE")
    existing_file = getattr(existing, "__file__", None)
    if existing_file and Path(existing_file).resolve() == init_path:
        return
    for name in list(sys.modules):
        if name == "IBAE" or name.startswith("IBAE."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location("IBAE", init_path, submodule_search_locations=[str(ROOT)])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create IBAE package spec from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["IBAE"] = module
    spec.loader.exec_module(module)


_install_local_package_alias()

IMAGE_PATH = TEST_DATA_DIR / "ngc1052_mojave.png"
ANALYSIS_PATH = TEST_DATA_DIR / "ngc1052_mojave_ridgeline_analysis.json"
RECONSTRUCTION_PATH = TEST_DATA_DIR / "ngc1052_mojave_ridgeline_analysis_reconstruction.npz"


def _require_fixture_files() -> None:
    missing = [str(path.name) for path in (IMAGE_PATH, ANALYSIS_PATH, RECONSTRUCTION_PATH) if not path.exists()]
    if missing:
        pytest.fail("Missing ngc1052_mojave regression fixture file(s): " + ", ".join(missing))


def _load_payload():
    _require_fixture_files()
    from IBAE.session_analysis import load_analysis_session

    return load_analysis_session(str(ANALYSIS_PATH))


def _load_cache():
    _require_fixture_files()
    from IBAE.session_analysis import load_reconstruction_cache

    return load_reconstruction_cache(str(RECONSTRUCTION_PATH))


def _as_float_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _assert_array_close(name: str, actual: object, expected: object, *, atol: float = 1e-4) -> None:
    actual_arr = _as_float_array(actual)
    expected_arr = _as_float_array(expected)
    assert actual_arr.shape == expected_arr.shape, f"{name} shape changed: {actual_arr.shape} != {expected_arr.shape}"
    np.testing.assert_allclose(actual_arr, expected_arr, rtol=0.0, atol=atol, equal_nan=True, err_msg=name)


def _assert_scalar_close(name: str, actual: object, expected: object, *, atol: float = 1e-8) -> None:
    actual_f = float(actual)
    expected_f = float(expected)
    if np.isnan(expected_f):
        assert np.isnan(actual_f), f"{name} expected NaN, got {actual_f}"
    else:
        assert abs(actual_f - expected_f) <= atol, f"{name} changed: {actual_f} != {expected_f}"


def test_default_transverse_gaussian_baseline_is_fixed_zero():
    from IBAE.ridgeline import fit_transverse_gaussian

    x = np.linspace(-6.0, 6.0, 121)
    y = 0.2 + 3.0 * np.exp(-0.5 * (x / 1.25) ** 2)
    fit = fit_transverse_gaussian({"valid_x": x, "valid_y": y}, mu_bound_px=6.0)

    assert fit["success"]
    params = dict(fit["params"])
    assert params["baseline"] == 0.0
    assert params["baseline_mode"] == "fixed_zero"
    assert params["baseline_fixed"] is True


def test_legacy_transverse_gaussian_bounded_baseline_wrapper():
    from IBAE.legacy import fit_transverse_gaussian_legacy_bounded

    x = np.linspace(-6.0, 6.0, 121)
    y = 0.2 + 3.0 * np.exp(-0.5 * (x / 1.25) ** 2)
    fit = fit_transverse_gaussian_legacy_bounded(
        {"valid_x": x, "valid_y": y},
        mu_bound_px=6.0,
        baseline_l1_flux=0.2,
        baseline_noise_sigma_flux=0.05,
    )

    assert fit["success"]
    params = dict(fit["params"])
    assert params["baseline_mode"] == "legacy_bounded_l1_noise"
    assert params["baseline_fixed"] is False
    assert -0.05 <= float(params["baseline"]) <= 0.2


def test_l0_l1_gaussian_transition_only_modifies_l0_band():
    pytest.importorskip("cv2")

    from IBAE.flux import reconstruct_flux_from_levels

    depth = np.zeros((9, 15), dtype=np.int32)
    roi = np.ones_like(depth, dtype=np.uint8)
    depth[3:6, 4:11] = 1
    depth[4, 6:9] = 2

    gaussian = reconstruct_flux_from_levels(
        region_depth_map=depth,
        roi_mask=roi,
        background_flux=0.0,
        l1_flux=10.0,
        level_ratio=2.0,
        smooth_sigma_px=0.0,
        l0_l1_transition_width_px=3.0,
        l0_l1_transition_alpha=3.0,
    )
    target = np.asarray(gaussian["target_flux_map"], dtype=np.float32)
    transition = dict(gaussian["l0_l1_transition"])

    assert transition["mode"] == "gaussian"
    assert transition["width_source"] == "requested"
    assert int(transition["applied_pixel_count"]) > 0
    assert target[4, 3] == pytest.approx(10.0)
    assert 0.0 < float(target[4, 2]) < 10.0
    assert target[4, 0] == pytest.approx(0.0)
    assert target[4, 7] >= 10.0

    legacy = reconstruct_flux_from_levels(
        region_depth_map=depth,
        roi_mask=roi,
        background_flux=0.0,
        l1_flux=10.0,
        level_ratio=2.0,
        smooth_sigma_px=0.0,
        l0_l1_transition_mode="flat",
    )
    legacy_target = np.asarray(legacy["target_flux_map"], dtype=np.float32)
    assert np.all(legacy_target[depth == 0] == 0.0)
    assert dict(legacy["l0_l1_transition"])["mode"] == "flat"


@pytest.mark.regression
def test_v9_startup_loader_reads_analysis_json_roi_snapshot():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")

    payload = _load_payload()
    from IBAE.analyzer_v9_qt import JetAnalyzerV9Qt

    analyzer = JetAnalyzerV9Qt(str(IMAGE_PATH))
    session = analyzer._load_startup_session_from_json(str(ANALYSIS_PATH))

    assert analyzer._startup_loaded_analysis_path == str(ANALYSIS_PATH)
    assert len(session["roi_points_xy"]) == len(payload["roi_points_xy"])
    assert session["roi_points_xy"][:3] == payload["roi_points_xy"][:3]


@pytest.mark.regression
def test_v9_startup_loader_rejects_non_session_json(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")

    settings_path = tmp_path / "trend_compare_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "session_paths": [str(ANALYSIS_PATH)],
                "auto_x_range": True,
                "plot_style": {},
            }
        ),
        encoding="utf-8",
    )
    from IBAE.analyzer_v9_qt import JetAnalyzerV9Qt

    analyzer = JetAnalyzerV9Qt(str(IMAGE_PATH))
    with pytest.raises(ValueError, match="roi_points_xy|sessions"):
        analyzer._load_startup_session_from_json(str(settings_path))


def _synthetic_diagonal_jet():
    yy, xx = np.mgrid[0:120, 0:140].astype(np.float32)
    core = np.asarray([18.0, 22.0], dtype=np.float32)
    tail = np.asarray([118.0, 92.0], dtype=np.float32)
    vec = tail - core
    length = float(np.hypot(float(vec[0]), float(vec[1])))
    unit = vec / length
    rel_x = xx - core[0]
    rel_y = yy - core[1]
    along = rel_x * unit[0] + rel_y * unit[1]
    perp = np.abs(rel_x * unit[1] - rel_y * unit[0])
    flux = np.exp(-0.5 * (perp / 3.0) ** 2) * np.exp(-np.maximum(along, 0.0) / 130.0)
    support = ((along >= 0.0) & (along <= length) & (perp <= 8.0)).astype(np.uint8)
    flux = (flux * support).astype(np.float32)
    depth = support.astype(np.int32)
    return flux, support, depth, (int(core[0]), int(core[1])), (int(tail[0]), int(tail[1]))


def test_mojave_polar_ridgeline_tracks_flux_spine():
    pytest.importorskip("cv2")
    from IBAE.ridgeline import extract_ridgeline

    flux, support, _depth, core, tail = _synthetic_diagonal_jet()
    result = extract_ridgeline(
        flux,
        support,
        core,
        tail,
        mode="mojave_polar",
        polar_step_px=2.0,
        polar_pa_step_deg=1.0,
        polar_sector_width_deg=60.0,
        polar_component_mode="peak",
        include_debug_maps=False,
    )

    ridge = np.asarray(result["ridge_xy"], dtype=np.float32)
    assert result["extraction_mode"] == "mojave_polar"
    assert len(ridge) > 30
    line = np.asarray(tail, dtype=np.float32) - np.asarray(core, dtype=np.float32)
    line_len = float(np.hypot(float(line[0]), float(line[1])))
    distances = np.abs(
        ((ridge[:, 0] - core[0]) * line[1]) - ((ridge[:, 1] - core[1]) * line[0])
    ) / line_len
    assert float(np.nanmedian(distances)) < 2.5


def test_legacy_cost_path_ridgeline_mode_remains_callable():
    pytest.importorskip("cv2")
    from IBAE.legacy import extract_ridgeline_legacy_cost_path
    from IBAE.ridgeline import extract_ridgeline

    flux, support, depth, core, tail = _synthetic_diagonal_jet()
    direct = extract_ridgeline_legacy_cost_path(
        flux,
        support,
        core,
        tail,
        region_depth_map=depth,
        smooth_window=3,
        bspline_smoothing=0.0,
        include_debug_maps=False,
    )
    routed = extract_ridgeline(
        flux,
        support,
        core,
        tail,
        region_depth_map=depth,
        mode="legacy_cost_path",
        smooth_window=3,
        bspline_smoothing=0.0,
        include_debug_maps=False,
    )

    assert direct["extraction_mode"] == "legacy_cost_path"
    assert routed["extraction_mode"] == "legacy_cost_path"
    assert len(np.asarray(routed["ridge_xy"])) >= 2


@pytest.mark.regression
def test_ngc1052_fixture_files_and_cache_metadata_are_consistent():
    _require_fixture_files()
    from IBAE.session_analysis import array_sha256

    payload = _load_payload()
    cache = _load_cache()
    metadata = dict(cache["metadata"])

    assert payload["image_path"] == IMAGE_PATH.name
    assert metadata["image_path"] == IMAGE_PATH.name
    assert metadata["shape"] == list(np.asarray(cache["flux_map"]).shape)
    assert metadata["region_depth_hash"] == array_sha256(cache["region_depth_map"])
    assert metadata["roi_mask_hash"] == array_sha256(cache["roi_mask"])
    thin_mask = (np.asarray(cache["thin_mask"], dtype=np.uint8) > 0).astype(np.uint8)
    assert metadata["thin_mask_hash"] in {
        array_sha256(thin_mask),
        array_sha256(thin_mask * np.uint8(255)),
    }
    assert int(np.count_nonzero(cache["valid_mask"])) == int(np.count_nonzero(cache["roi_mask"]))


@pytest.mark.regression
def test_ngc1052_flux_reconstruction_matches_cached_npz():
    pytest.importorskip("cv2")

    payload = _load_payload()
    cache = _load_cache()
    from IBAE.flux import reconstruct_flux_from_levels

    flux_cfg = dict(payload["flux_reconstruction"])
    transition_width = flux_cfg.get("l0_l1_transition_width_px", None)
    transition_width_px = None
    if transition_width is not None:
        transition_width = float(transition_width)
        if np.isfinite(transition_width) and transition_width > 0.0:
            transition_width_px = float(transition_width)
    result = reconstruct_flux_from_levels(
        region_depth_map=cache["region_depth_map"],
        roi_mask=cache["roi_mask"],
        background_flux=float(flux_cfg["background_flux"]),
        l1_flux=float(flux_cfg["l1_flux"]),
        level_ratio=float(flux_cfg["level_ratio"]),
        smooth_sigma_px=float(flux_cfg["sigma_px"]),
        contour_values=flux_cfg.get("custom_contour_values"),
        include_target_flux_map=False,
        l0_l1_transition_mode=str(flux_cfg.get("l0_l1_transition_mode", "flat")),
        l0_l1_transition_width_px=transition_width_px,
        l0_l1_transition_alpha=float(flux_cfg.get("l0_l1_transition_alpha", 3.0)),
    )

    actual = np.asarray(result["smoothed_flux_map"], dtype=np.float32)
    expected = np.asarray(cache["flux_map"], dtype=np.float32)
    assert actual.shape == expected.shape
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-4, equal_nan=True)


@pytest.mark.regression
def test_ngc1052_ridgeline_width_measurement_matches_saved_baseline():
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")

    payload = _load_payload()
    cache = _load_cache()
    from IBAE.ridgeline import measure_ridgeline_fwhm

    ridge_info = dict(payload["ridgeline"])
    baseline = dict(payload["measurement_result"])
    calibration = dict(payload["calibration"])
    support_mask = ((np.asarray(cache["region_depth_map"]) > 0) & (np.asarray(cache["roi_mask"]) > 0)).astype(np.uint8)
    tangent_half_window = None if bool(baseline.get("tangent_half_window_auto", False)) else int(baseline["tangent_half_window"])
    pa_sweep = dict(baseline.get("pa_sweep", {}))

    result = measure_ridgeline_fwhm(
        flux_map=cache["flux_map"],
        support_mask=support_mask,
        ridge_xy=np.asarray(ridge_info["ridgeline_xy"], dtype=np.int32),
        n_slices=int(baseline["requested_n_slices"]),
        trim_start_frac=float(baseline["trim_start_frac"]),
        trim_end_frac=float(baseline["trim_end_frac"]),
        tangent_half_window=tangent_half_window,
        profile_step_px=float(baseline["requested_profile_step_px"]),
        scale_mas_per_px=float(calibration["scale_mas_per_px"]),
        beam_major_mas=float(calibration["beam_major_mas"]),
        beam_minor_mas=float(calibration["beam_minor_mas"]),
        core_separation_px=float(baseline["core_separation_px"]),
        pa_sweep_enabled=bool(pa_sweep.get("enabled", False)),
        pa_sweep_range_deg=float(pa_sweep.get("range_deg", 15.0)),
        pa_sweep_step_deg=float(pa_sweep.get("step_deg", 1.0)),
        pa_sweep_cache=baseline,
        gaussian_baseline_mode=str(baseline["gaussian_baseline_mode"]),
        gaussian_baseline_l1_flux=float(baseline["gaussian_baseline_l1_flux"]),
        gaussian_baseline_noise_sigma_flux=float(baseline["gaussian_baseline_noise_sigma_flux"]),
    )

    assert int(result["valid_count"]) == int(baseline["valid_count"])
    assert int(result["slice_count"]) == int(baseline["slice_count"])
    _assert_array_close("slice_indices", result["slice_indices"], baseline["slice_indices"], atol=0.0)
    _assert_array_close("fwhm_px", result["fwhm_px"], baseline["fwhm_px"])
    _assert_array_close("intrinsic_fwhm_px", result["intrinsic_fwhm_px"], baseline["intrinsic_fwhm_px"])
    _assert_array_close("opening_angle_deg", result["opening_angle_deg"], baseline["opening_angle_deg"])
    _assert_scalar_close("median_half_opening_angle_deg", result["median_half_opening_angle_deg"], baseline["median_half_opening_angle_deg"])


@pytest.mark.regression
def test_ngc1052_trend_report_matches_saved_baseline():
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")

    payload = _load_payload()
    measurement = dict(payload["measurement_result"])
    trend = dict(payload["trend_report"])
    saved = dict(trend["trend_result"])
    from IBAE.reports import build_gaussian_report_rows, find_opening_angle_plateau, fit_power_law_from_rows

    rows = build_gaussian_report_rows(
        measurement,
        scale_mas_per_px=float(payload["calibration"]["scale_mas_per_px"]),
        use_raw_width=bool(trend.get("use_raw_width", False)),
    )
    distance_unit = str(saved.get("distance_unit", trend.get("x_range_unit", "auto")))
    cut_unit = str(trend.get("cut_min_separation_unit", trend.get("x_range_unit", "auto")))
    if cut_unit == "auto":
        cut_unit = distance_unit
    cut_key = "distance_from_core_mas" if cut_unit == "mas" else "distance_from_core_px"
    cut_min = max(0.0, float(trend.get("cut_min_separation_value", 0.0)))
    cut_max = max(0.0, float(trend.get("cut_max_separation_value", 0.0)))
    if cut_min > 0.0 or cut_max > 0.0:
        rows = [
            row for row in rows
            if np.isfinite(float(row.get(cut_key, float("nan"))))
            and float(row.get(cut_key, float("nan"))) >= cut_min
            and (cut_max <= 0.0 or float(row.get(cut_key, float("nan"))) <= cut_max)
        ]
    if bool(trend.get("filter_unstable_gaussian", False)):
        rows = [row for row in rows if not bool(row.get("gaussian_unstable", False))]

    if str(saved.get("fit_mode", trend.get("fit_mode", ""))) == "opening_plateau":
        result = find_opening_angle_plateau(rows, distance_unit=distance_unit)
    else:
        result = fit_power_law_from_rows(
            rows,
            start_slice_order=int(trend["fit_start_slice"]),
            end_slice_order=int(trend["fit_end_slice"]),
            distance_unit=distance_unit,
        )

    for key in (
        "fit_mode",
        "distance_unit",
        "plateau_start_slice_order",
        "plateau_end_slice_order",
    ):
        if key in saved:
            assert result.get(key) == saved.get(key)
    for key in (
        "k_fit",
        "opening_fit_median_deg",
        "half_opening_fit_median_deg",
        "plateau_start_distance",
        "plateau_end_distance",
    ):
        if key in saved:
            _assert_scalar_close(key, result[key], saved[key], atol=1e-6)
    _assert_array_close("trend x_all", result["x_all"], saved["x_all"], atol=1e-6)
