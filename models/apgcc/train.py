import math
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import gather_pred_gt_counts, reduce_dict
from core.metrics import calculate_metrics
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
    clip_max_norm: float = 0.1,
    threshold: float = 0.5
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    loss_fn.train()
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = {"loss_value": [], "loss_ce": [], "loss_points": [], "loss_ce_scaled": [], "loss_points_scaled": []}
    pred_counts_tensors: list[torch.Tensor] = []
    target_counts_tensors: list[torch.Tensor] = []

    use_amp = grad_scaler is not None
    for image, target_points, _, path, original_image in data_iter:
        input_image = image.to(device, non_blocking=True)

        # counts (keep on device; gather at epoch end)
        target_counts_local = torch.tensor([len(p) for p in target_points], device=device, dtype=torch.float32)
        target_counts_tensors.append(target_counts_local)

        targets = [
            {"point": p.to(device, non_blocking=True), "labels": torch.ones(p.shape[0], dtype=int, device=device)}
            for p in target_points
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            if use_amp:
                with autocast(enabled=grad_scaler.is_enabled()):  # type: ignore[union-attr]
                    output = model(input_image)
                    output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1]
                    pred_counts_local = (output_scores > threshold).sum(dim=1).float()
                    loss_dict = loss_fn(output, targets)
                    weight_dict = loss_fn.weight_dict
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            else:
                output = model(input_image)
                output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1]
                pred_counts_local = (output_scores > threshold).sum(dim=1).float()
                loss_dict = loss_fn(output, targets)
                weight_dict = loss_fn.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        pred_counts_tensors.append(pred_counts_local.detach())

        # reduced loss info for logging (rank0 consumes it)
        loss_dict_reduced = reduce_dict(loss_dict, nprocs) if ddp else loss_dict
        for k, v in loss_dict_reduced.items():
            loss_info[k].append(float(v.item()))
            loss_info[k + "_scaled"].append(float(v.item()) * float(weight_dict[k]))
        step_loss_value = sum(float(loss_dict_reduced[k].item()) * float(weight_dict[k]) for k in loss_dict_reduced.keys() if k in weight_dict)
        loss_info["loss_value"].append(step_loss_value)

        if not math.isfinite(step_loss_value):
            print(f"Loss is {step_loss_value}, stopping training")
            sys.exit(1)

        if use_amp:
            grad_scaler.scale(loss).backward()  # type: ignore[union-attr]
            if clip_max_norm > 0:
                grad_scaler.unscale_(optimizer)  # type: ignore[union-attr]
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            grad_scaler.step(optimizer)  # type: ignore[union-attr]
            grad_scaler.update()  # type: ignore[union-attr]
        else:
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

    # metric info
    pred_counts_local = torch.cat(pred_counts_tensors, dim=0) if pred_counts_tensors else torch.empty((0,), device=device)
    target_counts_local = torch.cat(target_counts_tensors, dim=0) if target_counts_tensors else torch.empty((0,), device=device)
    pred_all, gt_all = gather_pred_gt_counts(pred_counts_local, target_counts_local) if ddp else (pred_counts_local, target_counts_local)

    if (not ddp) or rank == 0:
        pred_np = pred_all.detach().cpu().numpy()
        gt_np = gt_all.detach().cpu().numpy()
        assert len(pred_np) == len(gt_np), f"Length of predictions and ground truths should be equal, but got {len(pred_np)} and {len(gt_np)}"
        metric_info = calculate_metrics(pred_np, gt_np)
    else:
        metric_info = {}

    # organize infos
    info = {k: round(float(np.mean(v)), 8) for k,v in loss_info.items()}
    info.update(metric_info)

    return model, optimizer, info
