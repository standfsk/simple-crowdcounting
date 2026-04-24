import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import random


def _collect_images(root: Path, split: str, exts: List[str]) -> List[Path]:
    candidates = list(root.glob(f"**/{split}/*"))
    out: List[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def _split_train_valid(
    train_paths: List[Path],
    valid_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if valid_ratio <= 0 or not train_paths:
        return train_paths, []

    if not (0 < valid_ratio < 1):
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}")

    n = len(train_paths)
    if n <= 1:
        return train_paths, []

    valid_count = int(round(n * valid_ratio))
    valid_count = max(1, min(n - 1, valid_count))

    idxs = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(idxs)
    valid_idxs = set(idxs[:valid_count])

    new_train = [p for i, p in enumerate(train_paths) if i not in valid_idxs]
    new_valid = [p for i, p in enumerate(train_paths) if i in valid_idxs]
    return new_train, new_valid


def mktxt(
    root: str,
    out_dir: str,
    exts: List[str],
    auto_valid_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    root_p = Path(root)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    exts_ = [e if e.startswith(".") else f".{e}" for e in exts]
    exts_ = [e.lower() for e in exts_]

    # Support both "valid" and "val" folder names, but always write "valid.txt".
    train = _collect_images(root_p, "train", exts_)
    valid = _collect_images(root_p, "valid", exts_)
    if not valid:
        valid = _collect_images(root_p, "val", exts_)
    test = _collect_images(root_p, "test", exts_)

    # If no explicit valid split exists, optionally sample it from train.
    if not valid and auto_valid_ratio > 0:
        train, valid = _split_train_valid(train, auto_valid_ratio, seed)

    subsets = [("train", train), ("valid", valid), ("test", test)]
    for subset, image_paths in subsets:
        out_path = out_dir_p / f"{subset}.txt"
        with open(out_path, "w", encoding="utf-8") as txt_file:
            for image_path in image_paths:
                txt_file.write(f"{os.path.abspath(image_path)}\n")
        print(f"Wrote {out_path} ({len(image_paths)} images)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train/valid/test .txt manifests (absolute image paths).")
    p.add_argument("--root", default=".", help="Root folder to scan for split subfolders (default: current dir).")
    p.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for train.txt/valid.txt/test.txt (default: current dir).",
    )
    p.add_argument(
        "--exts",
        nargs="+",
        default=["jpg"],
        help="Image extensions to include (default: jpg).",
    )
    p.add_argument(
        "--auto-valid-ratio",
        type=float,
        default=0.0,
        help="If no valid/val split exists, sample this fraction from train to create valid.txt (e.g. 0.1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --auto-valid-ratio split (default: 42).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mktxt(args.root, args.out_dir, args.exts, auto_valid_ratio=args.auto_valid_ratio, seed=args.seed)
