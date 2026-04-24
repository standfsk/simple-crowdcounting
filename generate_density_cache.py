from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.density import generate_density_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate reproducible density map cache from canonical JSON points.")
    p.add_argument("--processed", required=True, help="Processed dataset root (expects images/ and annotations/).")
    p.add_argument("--out", required=True, help="Output cache root (e.g. data/cache/density_maps).")
    p.add_argument("--sigma", type=float, default=4.0, help="Gaussian sigma (default: 4.0).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    return p.parse_args()


def _load_annotation_points(ann_path: Path) -> list[list[float]]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "points" not in payload:
        raise ValueError(f"Invalid canonical annotation (missing points): {ann_path}")
    pts = payload["points"] or []
    return [[float(x), float(y)] for x, y in pts]


def _resolve_image_path(processed_root: Path, ann_payload: dict) -> Path:
    image_rel = ann_payload.get("image_path")
    if not isinstance(image_rel, str) or not image_rel:
        raise ValueError("annotation JSON missing image_path")
    return processed_root / Path(image_rel)


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed)
    ann_root = processed_root / "annotations"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ann_paths = sorted([p for p in ann_root.rglob("*.json") if p.is_file()])
    for ann_path in tqdm(ann_paths, desc="density-cache"):
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        points = _load_annotation_points(ann_path)

        # Mirror relative structure under annotations/ into cache root.
        rel = ann_path.relative_to(ann_root)
        out_path = (out_root / rel).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            continue

        w_h = payload.get("meta", {}).get("resolution") if isinstance(payload.get("meta"), dict) else None
        if isinstance(w_h, list) and len(w_h) == 2 and all(isinstance(v, int) for v in w_h):
            width, height = int(w_h[0]), int(w_h[1])
        else:
            img_path = _resolve_image_path(processed_root, payload)
            with Image.open(img_path) as im:
                width, height = im.size

        density = generate_density_map(points, height=height, width=width, sigma=args.sigma)
        np.save(out_path, density)


if __name__ == "__main__":
    main()

