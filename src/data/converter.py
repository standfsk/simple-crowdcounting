from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from src.data.adapters import json_adapter, mat_adapter, txt_adapter
from src.data.adapters.coco_adapter import load_coco
from src.data.schema import CanonicalAnnotation


def _posix_relpath(path: Path) -> str:
    # JSON stores paths in a portable form.
    return path.as_posix()


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _safe_mkdir(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    raise ValueError(f"Unsupported image handling mode: {mode}")


def _image_resolution(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)


def _default_annotation_ext(dataset_type: str) -> str:
    if dataset_type == "txt":
        return ".txt"
    if dataset_type == "mat":
        return ".mat"
    if dataset_type == "json":
        return ".json"
    if dataset_type == "coco":
        return ".json"
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


@dataclass(frozen=True)
class ConvertItem:
    raw_image_path: Path
    raw_annotation_path: Optional[Path]  # None for coco (single file handled differently)
    image_rel: Path  # relative to raw images root


def _discover_images(images_root: Path, image_exts: Sequence[str]) -> List[Path]:
    exts = {e.lower() for e in image_exts}
    out: List[Path] = []
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


def _build_annotation_index(annotations_root: Path, annotation_ext: str) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in annotations_root.rglob(f"*{annotation_ext}"):
        if not p.is_file():
            continue
        idx.setdefault(p.stem, []).append(p)
    return idx


def _match_annotation(
    *,
    image_path: Path,
    images_root: Path,
    annotations_root: Path,
    annotation_ext: str,
    ann_by_stem: Dict[str, List[Path]],
) -> Path:
    # Primary strategy: mirror relative path and swap root.
    rel = image_path.relative_to(images_root)
    mirrored = (annotations_root / rel).with_suffix(annotation_ext)
    if mirrored.exists():
        return mirrored

    # Fallback: unique stem match.
    stem = image_path.stem
    candidates = ann_by_stem.get(stem, [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous annotation match for {image_path.name}: {candidates}")
    raise FileNotFoundError(f"No annotation found for image {image_path} (stem={stem})")


def _load_points(dataset_type: str, annotation_path: Path) -> List[List[float]]:
    if dataset_type == "txt":
        return txt_adapter.load_annotation(str(annotation_path))
    if dataset_type == "mat":
        return mat_adapter.load_annotation(str(annotation_path))
    if dataset_type == "json":
        return json_adapter.load_annotation(str(annotation_path))
    raise ValueError(f"Unsupported per-file dataset type: {dataset_type}")


def convert_dataset(
    *,
    dataset_type: str,
    input_root: str,
    output_root: str,
    dataset_name: Optional[str] = None,
    images_subdir: str = "images",
    annotations_subdir: str = "annotations",
    image_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
    annotation_ext: Optional[str] = None,
    image_mode: str = "copy",  # copy|hardlink|symlink
    workers: int = 8,
    write_stats: bool = True,
    ignore_errors: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """
    Convert a raw dataset (any format) into canonical JSON-per-image format.

    Output layout:
      <output_root>/
        images/...
        annotations/... (one JSON per image, mirrored paths)
    """
    dataset_type = dataset_type.lower().strip()
    if dataset_name is None:
        dataset_name = dataset_type
    if annotation_ext is None:
        annotation_ext = _default_annotation_ext(dataset_type)

    input_root_p = Path(input_root)
    output_root_p = Path(output_root)
    images_root = input_root_p / images_subdir
    annotations_root = input_root_p / annotations_subdir
    out_images_root = output_root_p / "images"
    out_annotations_root = output_root_p / "annotations"

    if logger is None:
        logger = logging.getLogger("data_converter")

    if dataset_type != "coco":
        if not images_root.is_dir():
            raise FileNotFoundError(f"Missing images dir: {images_root}")
        if not annotations_root.is_dir():
            raise FileNotFoundError(f"Missing annotations dir: {annotations_root}")
    else:
        if not images_root.is_dir():
            raise FileNotFoundError(f"Missing images dir: {images_root}")
        # For COCO, annotations_subdir can be a file or dir containing a single JSON.
        if annotations_root.is_file():
            coco_json_path = annotations_root
        else:
            candidates = list(annotations_root.glob("*.json"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"For coco, expected exactly 1 json under {annotations_root}, got {len(candidates)}"
                )
            coco_json_path = candidates[0]

        file_to_points = load_coco(str(coco_json_path))
        images = _discover_images(images_root, image_exts=image_exts)

        stats = {"images_total": len(images), "converted": 0, "failed": 0, "skipped": 0}
        _safe_mkdir(out_images_root)
        _safe_mkdir(out_annotations_root)

        for raw_image_path in images:
            rel = raw_image_path.relative_to(images_root)
            out_image_path = out_images_root / rel
            out_ann_path = (out_annotations_root / rel).with_suffix(".json")

            try:
                _link_or_copy(raw_image_path, out_image_path, image_mode)
                w, h = _image_resolution(out_image_path)
                # COCO file_name uses posix; match by leaf name and also by relative posix.
                key_candidates = {
                    raw_image_path.name,
                    _posix_relpath(rel),
                    _posix_relpath(rel).lstrip("./"),
                }
                points = None
                for k in key_candidates:
                    if k in file_to_points:
                        points = file_to_points[k]
                        break
                if points is None:
                    points = []
                ann = CanonicalAnnotation.from_parts(
                    image_path=_posix_relpath(Path("images") / rel),
                    points=points,
                    count=None,
                    meta={"dataset": dataset_name, "resolution": [w, h]},
                )
                _safe_mkdir(out_ann_path.parent)
                out_ann_path.write_text(json.dumps(ann.to_dict(), ensure_ascii=False), encoding="utf-8")
                stats["converted"] += 1
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                logger.exception("Failed converting %s: %s", raw_image_path, e)
                if not ignore_errors:
                    raise

        if write_stats:
            (output_root_p / "conversion_stats.json").write_text(
                json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return stats

    images = _discover_images(images_root, image_exts=image_exts)
    ann_by_stem = _build_annotation_index(annotations_root, annotation_ext)

    items: List[ConvertItem] = []
    for raw_image_path in images:
        rel = raw_image_path.relative_to(images_root)
        ann_path = _match_annotation(
            image_path=raw_image_path,
            images_root=images_root,
            annotations_root=annotations_root,
            annotation_ext=annotation_ext,
            ann_by_stem=ann_by_stem,
        )
        items.append(ConvertItem(raw_image_path=raw_image_path, raw_annotation_path=ann_path, image_rel=rel))

    stats = {"images_total": len(images), "converted": 0, "failed": 0, "skipped": 0}
    _safe_mkdir(out_images_root)
    _safe_mkdir(out_annotations_root)

    def _process_one(item: ConvertItem) -> None:
        out_image_path = out_images_root / item.image_rel
        out_ann_path = (out_annotations_root / item.image_rel).with_suffix(".json")

        _link_or_copy(item.raw_image_path, out_image_path, image_mode)
        w, h = _image_resolution(out_image_path)
        points = _load_points(dataset_type, item.raw_annotation_path)  # type: ignore[arg-type]
        ann = CanonicalAnnotation.from_parts(
            image_path=_posix_relpath(Path("images") / item.image_rel),
            points=points,
            count=None,
            meta={"dataset": dataset_name, "resolution": [w, h]},
        )
        _safe_mkdir(out_ann_path.parent)
        out_ann_path.write_text(json.dumps(ann.to_dict(), ensure_ascii=False), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_process_one, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                fut.result()
                stats["converted"] += 1
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                logger.exception("Failed converting %s: %s", it.raw_image_path, e)
                if not ignore_errors:
                    raise

    if write_stats:
        (output_root_p / "conversion_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return stats

