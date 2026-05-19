"""Ridgeline extraction, transverse profiles, and width measurements."""

from . import analysis as _analysis

_REPORT_EXPORTS = {
    "auto_select_k_near_one_range",
    "build_gaussian_report_rows",
    "compute_local_k",
    "find_opening_angle_plateau",
    "fit_power_law_from_rows",
    "paper_fig7_eastern_broken_power_law",
}

for _name in dir(_analysis):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_analysis, _name)


def __getattr__(name: str):
    if name in _REPORT_EXPORTS:
        from ..reports import trends as _trends

        value = getattr(_trends, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    name
    for name in globals()
    if not name.startswith("__") and name not in {"_analysis", "_REPORT_EXPORTS"}
] + sorted(_REPORT_EXPORTS)
