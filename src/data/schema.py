from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_points(points: Any) -> List[List[float]]:
    if points is None:
        raise ValueError("points is required")
    if not isinstance(points, list):
        raise ValueError(f"points must be a list, got {type(points).__name__}")

    normalized: List[List[float]] = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"points[{i}] must be [x, y], got {p!r}")
        x, y = p
        if not (_is_number(x) and _is_number(y)):
            raise ValueError(f"points[{i}] must be numeric [x, y], got {p!r}")
        normalized.append([float(x), float(y)])
    return normalized


def validate_resolution(resolution: Any) -> Optional[Tuple[int, int]]:
    if resolution is None:
        return None
    if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
        raise ValueError(f"meta.resolution must be [width, height], got {resolution!r}")
    w, h = resolution
    if not (isinstance(w, int) and isinstance(h, int)):
        raise ValueError(f"meta.resolution must be integers, got {resolution!r}")
    if w <= 0 or h <= 0:
        raise ValueError(f"meta.resolution must be positive, got {resolution!r}")
    return int(w), int(h)


@dataclass(frozen=True)
class CanonicalAnnotation:
    image_path: str
    points: List[List[float]]
    count: int
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_parts(
        *,
        image_path: str,
        points: Sequence[Sequence[float]],
        count: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "CanonicalAnnotation":
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("image_path must be a non-empty string")

        pts = validate_points(list(points))
        derived_count = len(pts)
        if count is None:
            count_ = derived_count
        else:
            if not isinstance(count, int):
                raise ValueError(f"count must be int if provided, got {type(count).__name__}")
            if count != derived_count:
                raise ValueError(f"count ({count}) != len(points) ({derived_count})")
            count_ = count

        meta_ = None
        if meta is not None:
            if not isinstance(meta, dict):
                raise ValueError(f"meta must be a dict if provided, got {type(meta).__name__}")
            meta_ = dict(meta)
            if "resolution" in meta_:
                res = validate_resolution(meta_["resolution"])
                meta_["resolution"] = list(res) if res is not None else None

        return CanonicalAnnotation(image_path=image_path, points=pts, count=count_, meta=meta_)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "image_path": self.image_path,
            "points": self.points,
            "count": self.count,
        }
        if self.meta is not None:
            out["meta"] = self.meta
        return out

