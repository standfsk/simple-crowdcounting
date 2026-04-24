from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _maybe_run(cmd: list[str]) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception:  # noqa: BLE001
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote a registered model version to the active serving slot.")
    p.add_argument("--network", required=True, help="Model family (e.g. apgcc).")
    p.add_argument("--version", required=True, help="Version to activate (must exist under registry/models/<network>/).")
    p.add_argument("--registry", default="registry", help="Registry root folder (default: registry).")
    p.add_argument("--active-dir", default="registry/active", help="Active serving dir (default: registry/active).")
    p.add_argument("--dvc", action="store_true", help="Also `dvc add` active artifacts (best-effort).")
    p.add_argument("--push", action="store_true", help="If --dvc, also `dvc push` (best-effort).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry_root = Path(args.registry)
    version_dir = registry_root / "models" / args.network / args.version
    model_pt = version_dir / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"Registered model not found: {model_pt}")

    active_dir = Path(args.active_dir)
    active_dir.mkdir(parents=True, exist_ok=True)
    active_model = active_dir / "model.pt"
    shutil.copy2(model_pt, active_model)

    pointer = {
        "network": args.network,
        "version": args.version,
        "activated_at_utc": _utc_now_iso(),
    }
    (active_dir / "active.json").write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dvc:
        _maybe_run(["dvc", "add", str(active_dir)])
        if args.push:
            _maybe_run(["dvc", "push"])

    print(f"Activated model for serving: {active_model}")
    print(f"Active pointer: {active_dir / 'active.json'}")


if __name__ == "__main__":
    main()

