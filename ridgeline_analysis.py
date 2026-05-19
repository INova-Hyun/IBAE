from __future__ import annotations

"""Compatibility shim for legacy ridgeline-analysis imports.

New code should import width-measurement helpers from ``IBAE.ridgeline`` and
report/trend helpers from ``IBAE.reports``. This module preserves the older
flat namespace for notebooks and scripts that still import
``IBAE.ridgeline_analysis`` directly.
"""

from .ridgeline import analysis as _impl
from .reports import trends as _report_impl

for _module in (_impl, _report_impl):
    for _name in dir(_module):
        if not (_name.startswith("__") and _name.endswith("__")):
            globals()[_name] = getattr(_module, _name)

__all__ = [
    name
    for name in globals()
    if not name.startswith("__") and name not in {"_impl", "_report_impl", "_module"}
]
