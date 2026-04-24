from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_density_map(
    points: List[List[float]],
    height: int,
    width: int,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Generate a (1, H, W) density map from point annotations.

    Notes:
      - Density maps are derived artifacts and must be reproducible from points.
      - Points are clamped to image bounds.
    """
    density = np.zeros((1, int(height), int(width)), dtype=np.float32)
    if points:
        arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        xs = np.clip(arr[:, 0].round().astype(np.int64), 0, width - 1)
        ys = np.clip(arr[:, 1].round().astype(np.int64), 0, height - 1)
        density[0, ys, xs] = 1.0
    if sigma is not None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        density = gaussian_filter(density, sigma=float(sigma))
    return density

