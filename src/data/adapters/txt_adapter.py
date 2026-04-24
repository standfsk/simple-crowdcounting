from __future__ import annotations

from typing import List


def load_annotation(annotation_path: str) -> List[List[float]]:
    """
    TXT adapter.

    Supported line formats:
      - "x y"
      - "x,y"
      - "x y ..." (extra columns ignored)
    """
    points: List[List[float]] = []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) < 2:
                continue
            x = float(parts[0])
            y = float(parts[1])
            points.append([x, y])
    return points

