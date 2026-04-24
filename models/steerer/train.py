from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import gather_pred_gt_counts
from core.metrics import calculate_metrics
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import reshape_train_data


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = []
    pred_counts_tensors: list[torch.Tensor] = []
    target_counts_tensors: list[torch.Tensor] = []
    for image, target_points, target_density, path, original_image in data_iter:
        input_image = image.to(device, non_blocking=True)
        target_counts_tensors.append(torch.tensor([len(p) for p in target_points], device=device, dtype=torch.float32))
        input_image, target_points = reshape_train_data(input_image, target_points)

        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    output = model(input_image)
                    loss_dict = loss_fn(output, target_points)
                    pred_density = loss_dict['pred_den']['1']
                    pred_counts_tensors.append(pred_density.sum(dim=(1, 2, 3)).detach())

                    loss = loss_dict['loss'].mean()
                    loss_info.append(loss.detach().cpu().item())
            else:
                output = model(input_image)
                loss_dict = loss_fn(output, target_points)
                pred_density = loss_dict['pred_den']['1']
                pred_counts_tensors.append(pred_density.sum(dim=(1, 2, 3)).detach())

                loss = loss_dict['loss'].mean()
                loss_info.append(loss.detach().cpu().item())

        optimizer.zero_grad(set_to_none=True)
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
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
    info = {"loss": round(float(np.mean(loss_info)), 8)}
    info.update(metric_info)

    return model, optimizer, info
