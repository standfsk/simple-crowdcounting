from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import gather_pred_gt_counts, reduce_mean
from core.logging import update_loss_info
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
) -> Tuple[nn.Module, Optimizer, GradScaler, Dict[str, float]]:
    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1
    regression = (model.module.bins is None) if ddp else (model.bins is None)

    pred_counts_tensors: list[torch.Tensor] = []
    target_counts_tensors: list[torch.Tensor] = []
    for image, target_points, target_density, path, original_image in data_iter:
        input_image = image.to(device, non_blocking=True)
        target_counts_tensors.append(torch.tensor([len(p) for p in target_points], device=device, dtype=torch.float32))
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)
        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    if not regression:
                        pred_class, pred_density = model(input_image)
                        loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
                    else:
                        pred_density = model(input_image)
                        loss, loss_info = loss_fn(pred_density, target_density, target_points)
            else:
                if not regression:
                    pred_class, pred_density = model(input_image)
                    loss, loss_info = loss_fn(pred_class, pred_density, target_density, target_points)
                else:
                    pred_density = model(input_image)
                    loss, loss_info = loss_fn(pred_density, target_density, target_points)
            pred_counts_tensors.append(pred_density.sum(dim=(1, 2, 3)).detach())

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # loss info
        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in loss_info.items()}
        info = update_loss_info(info, loss_info)

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
    info = {k: round(float(np.mean(v)), 8) for k,v in info.items()}
    info.update(metric_info)

    return model, optimizer, grad_scaler, info
