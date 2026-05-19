"""Source-tree shim for importing this checkout as ``IBAE``.

When notebooks are launched from this refactor directory, Python otherwise may
fall through to a sibling ``IBAE`` checkout on ``sys.path``.  This package keeps
``from IBAE...`` pointed at the files in this directory without requiring an
editable install.
"""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_ROOT_INIT = _PACKAGE_ROOT / "__init__.py"

__file__ = str(_ROOT_INIT)
__path__ = [str(_PACKAGE_ROOT)]
if __spec__ is not None and __spec__.submodule_search_locations is not None:
    __spec__.submodule_search_locations[:] = __path__

exec(compile(_ROOT_INIT.read_text(encoding="utf-8"), str(_ROOT_INIT), "exec"), globals(), globals())
