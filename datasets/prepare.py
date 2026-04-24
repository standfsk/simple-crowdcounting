import argparse
import os
from pathlib import Path
from typing import Iterable, List


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


def mktxt(root: str, out_dir: str, exts: List[str]) -> None:
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mktxt(args.root, args.out_dir, args.exts)
