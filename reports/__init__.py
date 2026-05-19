"""Report and trend-analysis implementation block."""

from .trends import (
    auto_select_k_near_one_range,
    build_gaussian_report_rows,
    compute_local_k,
    find_opening_angle_plateau,
    fit_power_law_from_rows,
    paper_fig7_eastern_broken_power_law,
)

__all__ = [
    "auto_select_k_near_one_range",
    "build_gaussian_report_rows",
    "compute_local_k",
    "find_opening_angle_plateau",
    "fit_power_law_from_rows",
    "paper_fig7_eastern_broken_power_law",
]
