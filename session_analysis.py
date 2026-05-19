from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


RECONSTRUCTION_CACHE_FORMAT = "ibae_level_map_reconstruction_npz"
RECONSTRUCTION_CACHE_VERSION = 1


def _to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_analysis_session(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {
        "format": "ibae_ridgeline_analysis",
        "version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": _to_jsonable(payload),
    }
    target.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2), encoding="utf-8")


def load_analysis_session(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Analysis JSON root must be an object.")
    if str(data.get("format", "")) == "ibae_ridgeline_analysis":
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("Analysis JSON payload must be an object.")
        return payload
    if "payload" in data and isinstance(data.get("payload"), dict):
        return dict(data["payload"])
    return data


def array_sha256(arr: Any) -> str:
    np_arr = np.ascontiguousarray(np.asarray(arr))
    hasher = hashlib.sha256()
    hasher.update(str(tuple(int(v) for v in np_arr.shape)).encode("utf-8"))
    hasher.update(str(np_arr.dtype).encode("utf-8"))
    hasher.update(np_arr.view(np.uint8))
    return hasher.hexdigest()


def default_reconstruction_cache_path(analysis_json_path: str) -> Path:
    target = Path(analysis_json_path)
    return target.with_name(f"{target.stem}_reconstruction.npz")


def save_reconstruction_cache(
    path: str,
    *,
    flux_map: Any,
    valid_mask: Any,
    region_depth_map: Any,
    roi_mask: Any,
    thin_mask: Any = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flux = np.asarray(flux_map, dtype=np.float32)
    valid = (np.asarray(valid_mask, dtype=np.uint8) > 0).astype(np.uint8)
    depth = np.asarray(region_depth_map, dtype=np.int32)
    roi = (np.asarray(roi_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if flux.shape[:2] != valid.shape[:2] or flux.shape[:2] != depth.shape[:2] or flux.shape[:2] != roi.shape[:2]:
        raise ValueError("Reconstruction cache arrays must have matching 2D shapes.")

    meta = dict(metadata or {})
    meta.update(
        {
            "format": RECONSTRUCTION_CACHE_FORMAT,
            "version": RECONSTRUCTION_CACHE_VERSION,
            "shape": [int(flux.shape[0]), int(flux.shape[1])],
            "flux_dtype": str(flux.dtype),
            "valid_mask_dtype": str(valid.dtype),
            "region_depth_hash": array_sha256(depth),
            "roi_mask_hash": array_sha256(roi),
            "thin_mask_hash": None if thin_mask is None else array_sha256(np.asarray(thin_mask, dtype=np.uint8)),
        }
    )
    arrays: Dict[str, Any] = {
        "flux_map": flux,
        "valid_mask": valid,
        "region_depth_map": depth,
        "roi_mask": roi,
        "metadata_json": np.asarray(json.dumps(_to_jsonable(meta), ensure_ascii=False)),
    }
    if thin_mask is not None:
        arrays["thin_mask"] = (np.asarray(thin_mask, dtype=np.uint8) > 0).astype(np.uint8)
    np.savez_compressed(target, **arrays)
    return meta


def load_reconstruction_cache(path: str) -> Dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise ValueError("Reconstruction cache is missing metadata_json.")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if str(metadata.get("format", "")) != RECONSTRUCTION_CACHE_FORMAT:
            raise ValueError("Unsupported reconstruction cache format.")
        flux = np.asarray(data["flux_map"], dtype=np.float32).copy()
        valid = np.asarray(data["valid_mask"], dtype=np.uint8).copy()
        depth = np.asarray(data["region_depth_map"], dtype=np.int32).copy() if "region_depth_map" in data else None
        roi = np.asarray(data["roi_mask"], dtype=np.uint8).copy() if "roi_mask" in data else None
        thin = np.asarray(data["thin_mask"], dtype=np.uint8).copy() if "thin_mask" in data else None
    return {
        "metadata": metadata,
        "flux_map": flux,
        "valid_mask": valid,
        "region_depth_map": depth,
        "roi_mask": roi,
        "thin_mask": thin,
    }
