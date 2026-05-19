from __future__ import annotations

"""Compatibility shim for legacy transverse-peak ridgeline refinement.

이 기능은 과거 ridgeline을 transverse peak 후보로 반복 보정하는 상황을
해결하기 위해 ``ridgeline_refine_legacy``로 만들어졌으나, 현재 기본
ridgeline extraction 및 FWHM 측정 workflow와 맞지 않아 더 이상 기본
경로에서 쓰이지 않음으로 ``ridgeline_refine_legacy``는 legacy로
이동되었다.
"""

from .legacy import ridgeline_refine as _impl

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in globals() if not name.startswith("__") and name != "_impl"]
