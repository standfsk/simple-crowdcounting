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
    num_queries: int,
    threshold: float,
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    loss_fn.train()
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = []
    pred_counts_tensors: list[torch.Tensor] = []
    target_counts_tensors: list[torch.Tensor] = []
    for image, target_points, _, path, original_image in data_iter:
        input_image = image.to(device, non_blocking=True)
        input_image, label = reshape_train_data(input_image, target_points)
        target_counts_tensors.append(torch.tensor([x["points"].shape[0] for x in label], device=device, dtype=torch.float32))

        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    output = model(input_image)
                    out_logits = output["pred_logits"]
                    prob = out_logits.sigmoid().view(1, -1, 2)
                    out_logits = out_logits.view(1, -1, 2)
                    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                                           input_image.shape[0] * num_queries, dim=1)
                    for k in range(topk_values.shape[0]):
                        sub_count = (topk_values[k, :] >= threshold).sum().float()
                        pred_counts_tensors.append(sub_count.unsqueeze(0))

                    loss_dict = loss_fn(output, label)
                    weight_dict = loss_fn.weight_dict
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    loss_info.append(loss.detach().cpu().item())
            else:
                output = model(input_image)
                out_logits = output["pred_logits"]
                prob = out_logits.sigmoid().view(1, -1, 2)
                out_logits = out_logits.view(1, -1, 2)
                topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                                       input_image.shape[0] * num_queries, dim=1)
                for k in range(topk_values.shape[0]):
                    sub_count = (topk_values[k, :] >= threshold).sum().float()
                    pred_counts_tensors.append(sub_count.unsqueeze(0))

                loss_dict = loss_fn(output, label)
                weight_dict = loss_fn.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
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
    info = {"loss_value": round(float(np.mean(loss_info)), 8)}
    info.update(metric_info)

    return model, optimizer, info
