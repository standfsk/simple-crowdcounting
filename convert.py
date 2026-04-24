from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.data.converter import convert_dataset


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("data_converter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert crowd-counting datasets into canonical JSON (points) format.")
    p.add_argument("--input", required=True, help="Input dataset root (expects images/ and annotations/).")
    p.add_argument("--output", required=True, help="Output root (will create images/ and annotations/).")
    p.add_argument(
        "--type",
        required=True,
        choices=["txt", "mat", "json", "coco"],
        help="Dataset annotation format type.",
    )
    p.add_argument("--dataset", default=None, help="Dataset name to store into meta.dataset (optional).")
    p.add_argument("--images-subdir", default="images", help="Images subdir under --input (default: images).")
    p.add_argument(
        "--annotations-subdir",
        default="annotations",
        help="Annotations subdir under --input (default: annotations). For coco, may be a file path.",
    )
    p.add_argument(
        "--annotation-ext",
        default=None,
        help="Annotation file extension override (e.g. .txt, .mat). Defaults by --type.",
    )
    p.add_argument(
        "--image-mode",
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help="How to place images into processed output.",
    )
    p.add_argument("--workers", type=int, default=8, help="Number of conversion worker threads.")
    p.add_argument("--no-stats", action="store_true", help="Do not write conversion_stats.json.")
    p.add_argument("--ignore-errors", action="store_true", help="Continue on per-file conversion errors.")
    p.add_argument(
        "--log-file",
        default=None,
        help="Log file path (default: <output>/conversion.log).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output)
    log_path = Path(args.log_file) if args.log_file else (output_root / "conversion.log")
    logger = _setup_logger(log_path)

    logger.info("Starting conversion: type=%s input=%s output=%s", args.type, args.input, args.output)
    stats = convert_dataset(
        dataset_type=args.type,
        input_root=args.input,
        output_root=args.output,
        dataset_name=args.dataset,
        images_subdir=args.images_subdir,
        annotations_subdir=args.annotations_subdir,
        annotation_ext=args.annotation_ext,
        image_mode=args.image_mode,
        workers=args.workers,
        write_stats=not args.no_stats,
        ignore_errors=args.ignore_errors,
        logger=logger,
    )
    logger.info("Done. Converted=%s Failed=%s Total=%s", stats.get("converted"), stats.get("failed"), stats.get("images_total"))

    if stats.get("failed", 0) > 0 and not args.ignore_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    # On Windows, hardlinks/symlinks can require special permissions; surface a helpful hint.
    if os.name == "nt":
        pass
    main()

