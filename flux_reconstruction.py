from __future__ import annotations

"""Compatibility shim for the flux reconstruction block.

New code should import from ``IBAE.flux`` or ``IBAE.flux.reconstruction``.
This module preserves the original public and private helper names.
"""

from .flux import reconstruction as _impl

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in globals() if not name.startswith("__") and name != "_impl"]
