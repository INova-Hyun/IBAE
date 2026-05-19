from __future__ import annotations

"""Compatibility shim for contour iterative peel assignment."""

from .contours import iterative_peel as _impl

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in globals() if not name.startswith("__") and name != "_impl"]
