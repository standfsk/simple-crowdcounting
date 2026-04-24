from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def _extract_point(ann: Dict[str, Any]) -> Tuple[float, float]:
    # Common patterns (best-effort):
    # - {"point": [x, y]}
    # - {"keypoints": [x, y, v, ...]} (first keypoint)
    if "point" in ann and isinstance(ann["point"], (list, tuple)) and len(ann["point"]) >= 2:
        return float(ann["point"][0]), float(ann["point"][1])
    if "keypoints" in ann and isinstance(ann["keypoints"], list) and len(ann["keypoints"]) >= 2:
        return float(ann["keypoints"][0]), float(ann["keypoints"][1])
    raise ValueError("COCO annotation missing a usable point field (expected 'point' or 'keypoints').")


def load_coco(coco_json_path: str) -> Dict[str, List[List[float]]]:
    """
    COCO-style adapter.

    Returns mapping: file_name -> [[x,y], ...]
    Notes:
      - This is a best-effort implementation for point-based COCO variants.
      - It expects each 'annotation' to contain a point (see _extract_point()).
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    id_to_file: Dict[int, str] = {}
    for img in images:
        if "id" in img and "file_name" in img:
            id_to_file[int(img["id"])] = str(img["file_name"])

    file_to_points: Dict[str, List[List[float]]] = {fn: [] for fn in id_to_file.values()}

    for ann in annotations:
        if "image_id" not in ann:
            continue
        image_id = int(ann["image_id"])
        if image_id not in id_to_file:
            continue
        file_name = id_to_file[image_id]
        x, y = _extract_point(ann)
        file_to_points[file_name].append([x, y])

    return file_to_points

