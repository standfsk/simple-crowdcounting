from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _is_dist_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main_process() -> bool:
    if not _is_dist_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:  # noqa: BLE001
        return None


def _git_commit_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return sha or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment: str = "crowd-counting"
    run_name: Optional[str] = None
    tags_json: Optional[str] = None


class MlflowRun:
    def __init__(self) -> None:
        self._mlflow = None
        self._active = False

    def configure(self, cfg: MlflowConfig) -> None:
        self._cfg = cfg

    def start(self, params: Optional[Dict[str, Any]] = None) -> None:
        if not getattr(self, "_cfg", None) or not self._cfg.enabled:
            return
        if not is_main_process():
            return

        mlflow = _try_import_mlflow()
        if mlflow is None:
            raise RuntimeError(
                "MLflow is enabled but not installed. Install it (e.g. `pip install mlflow`) or run without --mlflow."
            )
        self._mlflow = mlflow

        if self._cfg.tracking_uri:
            mlflow.set_tracking_uri(self._cfg.tracking_uri)
        mlflow.set_experiment(self._cfg.experiment)

        tags: Dict[str, str] = {"git_sha": _git_commit_sha() or "unknown"}
        if self._cfg.tags_json:
            try:
                extra = json.loads(self._cfg.tags_json)
                if isinstance(extra, dict):
                    tags.update({str(k): str(v) for k, v in extra.items()})
            except Exception:  # noqa: BLE001
                # Keep it best-effort; do not fail training for malformed tags.
                tags["mlflow_tags_parse_error"] = "true"

        mlflow.start_run(run_name=self._cfg.run_name, tags=tags)
        self._active = True

        if params:
            # MLflow params must be strings.
            safe_params = {str(k): str(v) for k, v in params.items()}
            # Avoid hard failure on too many params; log in chunks.
            for k, v in safe_params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:  # noqa: BLE001
                    continue

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
        if not self._active or not is_main_process():
            return
        mlflow = self._mlflow
        if mlflow is None:
            return
        for k, v in metrics.items():
            try:
                mlflow.log_metric(f"{prefix}{k}", float(v), step=step)
            except Exception:  # noqa: BLE001
                continue

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        if not self._active or not is_main_process():
            return
        mlflow = self._mlflow
        if mlflow is None:
            return
        try:
            if artifact_path:
                mlflow.log_artifact(path, artifact_path=artifact_path)
            else:
                mlflow.log_artifact(path)
        except Exception:  # noqa: BLE001
            return

    def end(self) -> None:
        if not self._active or not is_main_process():
            return
        mlflow = self._mlflow
        if mlflow is None:
            return
        try:
            mlflow.end_run()
        finally:
            self._active = False


_GLOBAL_RUN = MlflowRun()


def configure_from_config(config: Any) -> None:
    """
    Expects (optional) attributes on config:
      - mlflow (bool)
      - mlflow_uri (str)
      - mlflow_experiment (str)
      - mlflow_run_name (str)
      - mlflow_tags (json str)
    """
    enabled = bool(getattr(config, "mlflow", False)) or os.environ.get("MLOPS_MLFLOW", "") in {"1", "true", "TRUE"}
    cfg = MlflowConfig(
        enabled=enabled,
        tracking_uri=getattr(config, "mlflow_uri", None),
        experiment=getattr(config, "mlflow_experiment", "crowd-counting"),
        run_name=getattr(config, "mlflow_run_name", None),
        tags_json=getattr(config, "mlflow_tags", None),
    )
    _GLOBAL_RUN.configure(cfg)


def start_run(params: Optional[Dict[str, Any]] = None) -> None:
    _GLOBAL_RUN.start(params=params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None, prefix: str = "") -> None:
    _GLOBAL_RUN.log_metrics(metrics, step=step, prefix=prefix)


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    _GLOBAL_RUN.log_artifact(path, artifact_path=artifact_path)


def end_run() -> None:
    _GLOBAL_RUN.end()

