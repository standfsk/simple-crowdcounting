from __future__ import annotations

import io
import importlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import yaml

from core.evaluation import sliding_window_predict


def _flatten_dict_no_prefix(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten_dict_no_prefix(v))
        else:
            out[str(k)] = v
    return out


class _Config:
    def __init__(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _device_from_str(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    # allow "0" -> cuda:0
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state:
        return state
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def _load_state_dict(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return _strip_module_prefix(obj["state_dict"])
    if isinstance(obj, dict):
        # assume it is already a state_dict
        return _strip_module_prefix(obj)
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, RGB, [0,1]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
    t = (t - mean) / std
    return t


def _resize_if_needed(img: Image.Image, size: Optional[Tuple[int, int]]) -> Image.Image:
    if size is None:
        return img
    w, h = size
    if img.size == (w, h):
        return img
    return img.resize((w, h))


def _apgcc_points(output: Dict[str, torch.Tensor], threshold: float) -> Tuple[List[List[float]], int]:
    output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1][0]
    output_points = output["pred_points"][0]
    keep = output_scores > threshold
    pts = output_points[keep].detach().cpu().numpy().reshape(-1, 2).tolist()
    return pts, int(keep.detach().cpu().numpy().sum())


def _cltr_points(output: Dict[str, torch.Tensor], original_hw: Tuple[int, int], crop_size: int, num_queries: int) -> Tuple[List[List[float]], int]:
    from models.cltr.utils import get_points

    # get_points expects an ndarray only for shape; content unused for count.
    h, w = original_hw
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    pred_points = get_points(dummy, output["pred_logits"], output["pred_points"], crop_size, num_queries)
    pts = [[float(x), float(y)] for x, y in pred_points]
    return pts, len(pts)


def _density_count(pred_density: torch.Tensor) -> float:
    return float(pred_density.detach().cpu().numpy().sum())


@dataclass
class Predictor:
    network: str
    device: torch.device
    model: torch.nn.Module
    config: Any
    version_sha: Optional[str]

    @staticmethod
    def create(
        *,
        network: str,
        checkpoint: str,
        device: str,
        input_size: Optional[Tuple[int, int]] = None,
        threshold: float = 0.5,
        sliding_window: bool = False,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> "Predictor":
        cfg_path = f"configs/{network}.yml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Invalid YAML config: {cfg_path}")
        config_dict = _flatten_dict_no_prefix(raw_cfg)
        config_dict.update(
            {
                "network": network,
                "checkpoint": checkpoint,
                "device": device,
                "threshold": threshold,
            }
        )
        if input_size is not None:
            # Some configs expect scalar; keep both.
            config_dict["input_size"] = input_size[0] if input_size[0] == input_size[1] else max(input_size)
            config_dict["input_size_hw"] = input_size
        config_dict["sliding_window"] = sliding_window
        config_dict["window_size"] = window_size
        config_dict["stride"] = stride

        config = _Config(config_dict)

        # Some networks rely on these at inference time (mirrors existing scripts).
        if network in {"dmcount", "fusioncount", "clip_ebc"}:
            config.bins = [[0.0, 0.0], [1.0, 1.0], [2.0, float("inf")]]
            config.anchor_points = [0.0, 1.0, 2.10737]

        module = importlib.import_module(f"models.{network}")
        if not hasattr(module, "__all__") or not module.__all__:
            raise ValueError(f"models.{network} does not expose a model class via __all__")
        class_name = module.__all__[0]
        model_cls = getattr(module, class_name)
        model = model_cls(config)
        state_dict = _load_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # Keep it best-effort: serving should still start for compatible checkpoints.
            pass

        dev = _device_from_str(device)
        model.to(dev)
        model.eval()

        return Predictor(network=network, device=dev, model=model, config=config, version_sha=_git_sha())

    @torch.inference_mode()
    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes))
        original_w, original_h = img.size
        img = _resize_if_needed(img, getattr(self.config, "input_size_hw", None))
        input_tensor = _pil_to_tensor_rgb(img).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)

        # Point-based
        if isinstance(output, dict) and "pred_logits" in output and "pred_points" in output:
            if self.network == "cltr":
                crop_size = int(getattr(self.config, "crop_size"))
                num_queries = int(getattr(self.config, "num_queries"))
                points, count = _cltr_points(output, (original_h, original_w), crop_size, num_queries)
            else:
                points, count = _apgcc_points(output, float(getattr(self.config, "threshold", 0.5)))
            return {
                "network": self.network,
                "count": int(count),
                "points": points,
                "meta": {"original_resolution": [int(original_w), int(original_h)], "git_sha": self.version_sha},
            }

        # Density-based
        pred_density: Optional[torch.Tensor] = None
        if torch.is_tensor(output):
            pred_density = output
        elif isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
            pred_density = output[0]
        elif isinstance(output, (list, tuple)) and len(output) >= 2 and torch.is_tensor(output[0]):
            pred_density = output[0]

        if pred_density is None:
            raise ValueError(f"Unsupported model output type for serving: {type(output).__name__}")

        if getattr(self.config, "sliding_window", False):
            if self.network != "clip_ebc":
                raise ValueError("Sliding-window serving is only supported for clip_ebc in this implementation.")
            if self.config.window_size is None or self.config.stride is None:
                raise ValueError("SERVE_WINDOW_SIZE and SERVE_STRIDE are required when SERVE_SLIDING_WINDOW=true.")
            pred_density = sliding_window_predict(self.model, input_tensor, self.config.window_size, self.config.stride)

        if self.network == "steerer":
            density_factor = float(getattr(self.config, "density_factor"))
            pred_density = pred_density / density_factor

        count = _density_count(pred_density)
        return {
            "network": self.network,
            "count": float(count),
            "meta": {"original_resolution": [int(original_w), int(original_h)], "git_sha": self.version_sha},
        }
