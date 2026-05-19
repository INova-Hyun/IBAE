"""Legacy ridgeline extraction entry points.

The default ridgeline extractor now follows the MOJAVE-style polar sampling
workflow.  The previous contour-depth cost-path extractor is kept here for
reproducibility and for comparing old saved sessions.
"""

from __future__ import annotations

from ..ridgeline.analysis import extract_ridgeline_legacy_cost_path

extract_ridgeline_legacy = extract_ridgeline_legacy_cost_path

__all__ = ["extract_ridgeline_legacy", "extract_ridgeline_legacy_cost_path"]
