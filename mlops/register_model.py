from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _maybe_run(cmd: list[str]) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception:  # noqa: BLE001
        return


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    version: str
    network: str
    created_at_utc: str
    git_sha: Optional[str]
    checkpoint_source: str
    notes: Optional[str]
    extra: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register a trained checkpoint into a local model registry.")
    p.add_argument("--network", required=True, help="Model family (e.g. apgcc, dmcount, clip_ebc).")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint file (e.g. output/train/exp/best.pt).")
    p.add_argument("--version", required=True, help="Version tag (e.g. 2026-04-23_001, v1, v1.0.0).")
    p.add_argument("--name", default="crowdcounting", help="Logical product/model name.")
    p.add_argument("--notes", default=None, help="Optional notes.")
    p.add_argument("--extra-json", default=None, help="Optional JSON dict to attach as metadata.")
    p.add_argument("--registry", default="registry", help="Registry root folder (default: registry).")
    p.add_argument("--dvc", action="store_true", help="Also `dvc add` the registered version (best-effort).")
    p.add_argument("--push", action="store_true", help="If --dvc, also `dvc push` (best-effort).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    registry_root = Path(args.registry)
    version_dir = registry_root / "models" / args.network / args.version
    version_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_src = Path(args.checkpoint)
    if not checkpoint_src.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_src}")

    checkpoint_dst = version_dir / "model.pt"
    shutil.copy2(checkpoint_src, checkpoint_dst)

    extra: Dict[str, Any] = {}
    if args.extra_json:
        payload = json.loads(args.extra_json)
        if not isinstance(payload, dict):
            raise ValueError("--extra-json must be a JSON object")
        extra = payload

    meta = ModelMetadata(
        name=str(args.name),
        version=str(args.version),
        network=str(args.network),
        created_at_utc=_utc_now_iso(),
        git_sha=_git_sha(),
        checkpoint_source=str(checkpoint_src.as_posix()),
        notes=args.notes,
        extra=extra,
    )
    (version_dir / "metadata.json").write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep big artifacts out of Git by default; users can still DVC them.
    (registry_root / ".gitkeep").touch(exist_ok=True)

    if args.dvc:
        _maybe_run(["dvc", "add", str(version_dir)])
        if args.push:
            _maybe_run(["dvc", "push"])

    print(f"Registered: {version_dir}")
    print(f"  - checkpoint: {checkpoint_dst}")
    print(f"  - metadata:   {version_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

