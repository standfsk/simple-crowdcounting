from __future__ import annotations

import json
from typing import Any, List


def _as_points(obj: Any) -> List[List[float]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        # either [[x, y], ...] or [{"x":..., "y":...}, ...]
        if not obj:
            return []
        if isinstance(obj[0], (list, tuple)):
            out: List[List[float]] = []
            for p in obj:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise ValueError(f"Invalid point entry: {p!r}")
                out.append([float(p[0]), float(p[1])])
            return out
        if isinstance(obj[0], dict):
            out = []
            for p in obj:
                if "x" not in p or "y" not in p:
                    raise ValueError(f"Invalid point dict entry: {p!r}")
                out.append([float(p["x"]), float(p["y"])])
            return out
    raise ValueError("Unsupported JSON point format")


def load_annotation(annotation_path: str) -> List[List[float]]:
    """
    JSON adapter.

    Supported formats:
      - {"points": [[x,y], ...]}
      - [[x,y], ...]
      - {"annotations": [{"x":..., "y":...}, ...]}  (fallback)
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "points" in payload:
            return _as_points(payload["points"])
        if "annotations" in payload:
            return _as_points(payload["annotations"])
    return _as_points(payload)

