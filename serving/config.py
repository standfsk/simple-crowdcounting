from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_input_size(v: Optional[str]) -> Optional[Tuple[int, int]]:
    if not v:
        return None
    s = v.strip().lower().replace(" ", "")
    if "x" in s:
        w, h = s.split("x", 1)
        return int(w), int(h)
    n = int(s)
    return n, n


@dataclass(frozen=True)
class ServeSettings:
    network: str
    checkpoint: str
    device: str
    input_size: Optional[Tuple[int, int]]
    threshold: float
    sliding_window: bool
    window_size: Optional[int]
    stride: Optional[int]

    @staticmethod
    def from_env() -> "ServeSettings":
        network = os.environ.get("SERVE_NETWORK", "apgcc").strip().lower()
        checkpoint = os.environ.get("SERVE_CHECKPOINT", "").strip()
        if not checkpoint:
            raise ValueError("SERVE_CHECKPOINT is required (path inside container/host).")

        device = os.environ.get("SERVE_DEVICE", "cpu").strip().lower()
        input_size = _parse_input_size(os.environ.get("SERVE_INPUT_SIZE"))
        threshold = float(os.environ.get("SERVE_THRESHOLD", "0.5"))

        sliding_window = _parse_bool(os.environ.get("SERVE_SLIDING_WINDOW", "false"))
        window_size = os.environ.get("SERVE_WINDOW_SIZE")
        stride = os.environ.get("SERVE_STRIDE")

        return ServeSettings(
            network=network,
            checkpoint=checkpoint,
            device=device,
            input_size=input_size,
            threshold=threshold,
            sliding_window=sliding_window,
            window_size=int(window_size) if window_size else None,
            stride=int(stride) if stride else None,
        )

