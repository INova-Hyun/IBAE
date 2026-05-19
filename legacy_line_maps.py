from __future__ import annotations

"""Compatibility shim for legacy line-map attributes.

이 기능은 과거 notebook/script가 ``line_*`` 및 ``simple_*`` debug map 속성에
직접 접근하는 상황을 해결하기 위해 ``legacy_line_maps``로 만들어졌으나,
현재 Qt 분석 workflow가 payload/cache 중심으로 동작하는 상황과 맞지 않아
더 이상 기본 경로에서 쓰이지 않음으로 ``legacy_line_maps``는 legacy로
이동되었다.
"""

from .legacy.line_maps import *  # noqa: F401,F403
