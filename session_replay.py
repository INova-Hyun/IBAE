from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _normalize_points(points: Sequence[Sequence[int]]) -> List[List[int]]:
    out: List[List[int]] = []
    for pt in points:
        if len(pt) < 2:
            continue
        out.append([int(pt[0]), int(pt[1])])
    return out


def _normalize_cuts(cuts: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for cut in cuts:
        if not isinstance(cut, dict):
            continue
        start = cut.get("start")
        end = cut.get("end", start)
        if not isinstance(start, (list, tuple)) or len(start) < 2:
            continue
        if not isinstance(end, (list, tuple)) or len(end) < 2:
            end = start
        out.append(
            {
                "kind": str(cut.get("kind", "line")),
                "start": [int(start[0]), int(start[1])],
                "end": [int(end[0]), int(end[1])],
                "width": int(max(1, cut.get("width", 1))),
            }
        )
    return out


def build_replay_session_payload(
    image_path: str,
    roi_points_xy: Sequence[Sequence[int]],
    binary_prep_mode: str,
    manual_gray_thresh: Optional[int],
    manual_invert: Optional[bool],
    binary_split_cuts: Sequence[Dict[str, object]],
    junction_cuts: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "format": "ibae_manual_replay",
        "version": 1,
        "image_path": str(Path(image_path).resolve()),
        "image_basename": str(Path(image_path).name),
        "roi_points_xy": _normalize_points(roi_points_xy),
        "binary_prep_mode": str(binary_prep_mode),
        "manual_gray_thresh": None if manual_gray_thresh is None else int(manual_gray_thresh),
        "manual_threshold_invert": None if manual_invert is None else bool(manual_invert),
        "binary_split_cuts": _normalize_cuts(binary_split_cuts),
        "junction_cuts": _normalize_cuts(junction_cuts),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _normalize_replay_payload(data: Dict[str, object]) -> Dict[str, object]:
    return {
        "format": str(data.get("format", "ibae_manual_replay")),
        "version": int(data.get("version", 1)),
        "image_path": str(data.get("image_path", "")),
        "image_basename": str(data.get("image_basename", Path(str(data.get("image_path", ""))).name)),
        "roi_points_xy": _normalize_points(data.get("roi_points_xy", [])),
        "binary_prep_mode": str(data.get("binary_prep_mode", "manual_threshold")),
        "manual_gray_thresh": None if data.get("manual_gray_thresh", None) is None else int(data.get("manual_gray_thresh")),
        "manual_threshold_invert": None
        if data.get("manual_threshold_invert", None) is None
        else bool(data.get("manual_threshold_invert")),
        "binary_split_cuts": _normalize_cuts(data.get("binary_split_cuts", [])),
        "junction_cuts": _normalize_cuts(data.get("junction_cuts", [])),
        "updated_at_utc": str(data.get("updated_at_utc", "")),
    }


def _pick_legacy_session(
    sessions: Dict[str, object],
    expected_image_path: Optional[str],
) -> Dict[str, object]:
    expected_resolved = str(Path(expected_image_path).resolve()) if expected_image_path else ""
    candidates: List[Dict[str, object]] = []
    for key, value in sessions.items():
        if not isinstance(value, dict):
            continue
        meta = value.get("session_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        image_path = str(meta.get("image_path", ""))
        payload = {
            "format": "ibae_manual_replay_legacy_import",
            "version": 1,
            "image_path": image_path,
            "image_basename": str(Path(image_path).name),
            "roi_points_xy": meta.get("roi_points_xy", []),
            "binary_prep_mode": meta.get("binary_prep_mode", "manual_threshold"),
            "manual_gray_thresh": meta.get("manual_gray_thresh", None),
            "manual_threshold_invert": meta.get("manual_threshold_invert", None),
            "binary_split_cuts": value.get("binary_split_cuts", []),
            "junction_cuts": value.get("junction_cuts", []),
            "updated_at_utc": value.get("updated_at_utc", ""),
            "legacy_session_key": str(key),
        }
        candidates.append(_normalize_replay_payload(payload))
    if not candidates:
        raise ValueError("Legacy sessions JSON does not contain any usable sessions.")
    if expected_resolved:
        exact = [c for c in candidates if str(Path(str(c.get('image_path', ''))).resolve()) == expected_resolved]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            exact.sort(key=lambda item: str(item.get("updated_at_utc", "")), reverse=True)
            return exact[0]
        expected_name = Path(expected_resolved).name
        by_name = [c for c in candidates if str(c.get("image_basename", "")) == expected_name]
        if len(by_name) == 1:
            return by_name[0]
        if len(by_name) > 1:
            by_name.sort(key=lambda item: str(item.get("updated_at_utc", "")), reverse=True)
            return by_name[0]
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: str(item.get("updated_at_utc", "")), reverse=True)
    return candidates[0]


def load_replay_session(path: str, expected_image_path: Optional[str] = None) -> Dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Replay JSON root must be an object.")
    if "sessions" in data and isinstance(data.get("sessions"), dict):
        return _pick_legacy_session(data["sessions"], expected_image_path=expected_image_path)
    if "roi_points_xy" not in data:
        raise ValueError("Replay JSON must contain 'roi_points_xy' or legacy 'sessions'.")
    return _normalize_replay_payload(data)


def normalize_replay_session_payload(
    payload: Dict[str, object],
    expected_image_path: Optional[str] = None,
) -> Dict[str, object]:
    session = _normalize_replay_payload(dict(payload or {}))
    image_path = str(session.get("image_path", "") or "")
    if expected_image_path:
        expected = str(Path(expected_image_path).resolve())
        if image_path and image_path != expected:
            raise ValueError(f"Replay JSON image mismatch: {image_path} != {expected}")
    return session


def save_replay_session(path: str, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
