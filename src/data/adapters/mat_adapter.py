from __future__ import annotations

from typing import List

import numpy as np
from scipy.io import loadmat


def _to_points(arr: np.ndarray) -> List[List[float]]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in arr]


def load_annotation(annotation_path: str) -> List[List[float]]:
    """
    MAT adapter.

    Heuristics for common crowd-counting datasets:
      - ShanghaiTech: mat["image_info"][0][0][0][0][0] -> Nx2
      - NWPU/UCF-QNRF variants: mat["annPoints"] -> Nx2
    """
    mat = loadmat(annotation_path)

    if "annPoints" in mat:
        return _to_points(mat["annPoints"])

    if "image_info" in mat:
        try:
            pts = mat["image_info"][0][0][0][0][0]
            return _to_points(pts)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to parse ShanghaiTech-style image_info from {annotation_path}") from e

    # Fallback: find any (N,2) numeric array
    for _, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2 and np.issubdtype(v.dtype, np.number):
            return _to_points(v)

    raise ValueError(f"Unsupported .mat structure in {annotation_path}")

