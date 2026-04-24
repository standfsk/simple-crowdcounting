from typing import Dict
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def reduce_mean(tensor: Tensor, nprocs: int) -> Tensor:
    if not is_dist_avail_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def reduce_dict(input_dict: Dict[str, Tensor], nprocs: int) -> dict:
    if nprocs < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        values = reduce_mean(values, nprocs)
        reduced_dict = {k:v for k,v in zip(names, values)}
    return reduced_dict


def all_gather_1d_tensor(t: Tensor, pad_value: float = -1.0) -> Tensor:
    """
    All-gather variable-length 1D tensors across ranks.

    Returns the concatenated tensor on rank 0, and an empty tensor on other ranks.

    Notes:
      - For NCCL backends, tensors must be on CUDA.
      - Use small tensors only (counts/metrics), not large activations.
    """
    if not is_dist_avail_and_initialized():
        return t

    if t.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape={tuple(t.shape)}")

    world_size = dist.get_world_size()
    device = t.device

    local_len = torch.tensor([t.numel()], device=device, dtype=torch.int64)
    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    lens_i = [int(x.item()) for x in lens]
    max_len = max(lens_i) if lens_i else 0

    if max_len == 0:
        out = torch.empty((0,), device=device, dtype=t.dtype)
        return out if get_rank() == 0 else out[:0]

    if t.numel() < max_len:
        pad = torch.full((max_len - t.numel(),), float(pad_value), device=device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    gathered = [torch.empty_like(t_pad) for _ in range(world_size)]
    dist.all_gather(gathered, t_pad)

    if get_rank() != 0:
        return torch.empty((0,), device=device, dtype=t.dtype)

    chunks = []
    for gi, li in zip(gathered, lens_i):
        if li <= 0:
            continue
        chunks.append(gi[:li])
    if not chunks:
        return torch.empty((0,), device=device, dtype=t.dtype)
    return torch.cat(chunks, dim=0)


def gather_pred_gt_counts(
    pred_counts: Tensor,
    gt_counts: Tensor,
    *,
    pad_value: float = -1.0,
) -> tuple[Tensor, Tensor]:
    """
    Gather prediction/gt count vectors to rank0 (DDP), keeping variable lengths.

    On non-DDP runs, returns inputs unchanged.
    On DDP runs, returns (pred_all, gt_all) on rank0 and empty tensors on other ranks.
    """
    if pred_counts.shape != gt_counts.shape:
        raise ValueError(f"pred_counts and gt_counts must have same shape, got {pred_counts.shape} vs {gt_counts.shape}")
    pred_all = all_gather_1d_tensor(pred_counts, pad_value=pad_value)
    gt_all = all_gather_1d_tensor(gt_counts, pad_value=pad_value)
    return pred_all, gt_all

def setup(local_rank: int, nprocs: int) -> None:
    if nprocs > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12366"
        dist.init_process_group("nccl", rank=local_rank, world_size=nprocs)
    else:
        print("Single process. No need to setup dist.")


def cleanup(ddp: bool = True) -> None:
    if ddp:
        dist.destroy_process_group()


def init_seeds(seed: int, cuda_deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, not reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def barrier(ddp: bool = True) -> None:
    if ddp:
        dist.barrier()
