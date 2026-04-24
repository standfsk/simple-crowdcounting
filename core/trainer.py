from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from core.checkpoint import load_checkpoint, save_checkpoint
from core.data import get_dataloader
from core.distributed import barrier, cleanup
from core.logging import get_config, get_logger, get_writer, log, update_eval_result, update_train_result


BuildModelFn = Callable[[Any, torch.device], nn.Module]
BuildLossFn = Callable[[Any, torch.device], nn.Module]
BuildOptimFn = Callable[[Any, nn.Module, int], Tuple[Optimizer, Optional[Any]]]
TrainEpochFn = Callable[
    [nn.Module, Any, nn.Module, Optimizer, Optional[GradScaler], torch.device, int, int],
    Tuple[nn.Module, Optimizer, Optional[GradScaler], Dict[str, float]],
]
EvalFn = Callable[[nn.Module, Any, torch.device], Dict[str, float]]
SchedulerStepFn = Callable[[Any, int], None]


@dataclass(frozen=True)
class TrainRecipe:
    build_model: BuildModelFn
    build_loss_fn: BuildLossFn
    build_optimizer_and_scheduler: BuildOptimFn
    train_one_epoch: TrainEpochFn
    evaluate: EvalFn
    scheduler_step: Optional[SchedulerStepFn] = None
    ddp_find_unused_parameters: bool = False


def _get_recipe(network: str, config: Any) -> TrainRecipe:
    """
    Centralized model training wiring.

    This replaces per-model `models/<net>/trainer.py` modules.
    """
    network = network.lower()

    if network == "apgcc":
        m_model = importlib.import_module("models.apgcc.model")
        m_utils = importlib.import_module("models.apgcc.utils")
        m_train = importlib.import_module("models.apgcc.train")
        m_eval = importlib.import_module("models.apgcc.eval")

        APGCC = getattr(m_model, "APGCC")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return APGCC(cfg, sync_bn=False, last_pool=False)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "clip_ebc":
        m_model = importlib.import_module("models.clip_ebc.model")
        m_utils = importlib.import_module("models.clip_ebc.utils")
        m_train = importlib.import_module("models.clip_ebc.train")
        m_eval = importlib.import_module("models.clip_ebc.eval")

        clip_model = getattr(m_model, "_clip_ebc")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return clip_model(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, grad_scaler, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device, config.sliding_window, config.input_size, config.stride)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "cltr":
        m_model = importlib.import_module("models.cltr.model")
        m_utils = importlib.import_module("models.cltr.utils")
        m_train = importlib.import_module("models.cltr.train")
        m_eval = importlib.import_module("models.cltr.eval")

        CLTR = getattr(m_model, "CLTR")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return CLTR(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(
                model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs, config.num_queries, config.threshold
            )
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device, config.num_queries, config.threshold, config.crop_size)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "dmcount":
        m_model = importlib.import_module("models.dmcount.model")
        m_utils = importlib.import_module("models.dmcount.utils")
        m_train = importlib.import_module("models.dmcount.train")
        m_eval = importlib.import_module("models.dmcount.eval")

        DMCount = getattr(m_model, "DMCount")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return DMCount(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(config=cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "fusioncount":
        m_model = importlib.import_module("models.fusioncount.model")
        m_utils = importlib.import_module("models.fusioncount.utils")
        m_train = importlib.import_module("models.fusioncount.train")
        m_eval = importlib.import_module("models.fusioncount.eval")

        FusionCount = getattr(m_model, "FusionCount")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return FusionCount(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "ffnet":
        m_model = importlib.import_module("models.ffnet.model")
        m_utils = importlib.import_module("models.ffnet.utils")
        m_train = importlib.import_module("models.ffnet.train")
        m_eval = importlib.import_module("models.ffnet.eval")

        FFNet = getattr(m_model, "FFNet")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return FFNet(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
        )

    if network == "steerer":
        m_model = importlib.import_module("models.steerer.model")
        m_utils = importlib.import_module("models.steerer.utils")
        m_train = importlib.import_module("models.steerer.train")
        m_eval = importlib.import_module("models.steerer.eval")

        STEERER = getattr(m_model, "STEERER")
        get_loss_fn = getattr(m_utils, "get_loss_fn")
        get_optimizer = getattr(m_utils, "get_optimizer")
        train = getattr(m_train, "train")
        evaluate = getattr(m_eval, "evaluate")

        def build_model(cfg: Any, device: torch.device) -> nn.Module:
            return STEERER(cfg)

        def build_loss_fn(cfg: Any, device: torch.device) -> nn.Module:
            return get_loss_fn(cfg)

        def build_optim(cfg: Any, model: nn.Module, train_loader_len: int):
            return get_optimizer(cfg, model, train_loader_len)

        def train_one_epoch(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs):
            model, optimizer, info = train(model, train_loader, loss_fn, optimizer, grad_scaler, device, rank, nprocs)
            return model, optimizer, grad_scaler, info

        def eval_fn(model, val_loader, device):
            return evaluate(model, val_loader, device, config.density_factor)

        def scheduler_step(scheduler, epoch: int) -> None:
            scheduler.step(epoch)

        return TrainRecipe(
            build_model=build_model,
            build_loss_fn=build_loss_fn,
            build_optimizer_and_scheduler=build_optim,
            train_one_epoch=train_one_epoch,
            evaluate=eval_fn,
            scheduler_step=scheduler_step,
            ddp_find_unused_parameters=True,
        )

    raise ValueError(f"Unsupported network for training: {network}")


def run(
    local_rank: int,
    nprocs: int,
    config: Any,
) -> None:
    recipe = _get_recipe(getattr(config, "network"), config)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    ddp = nprocs > 1

    model = recipe.build_model(config, device).to(device)
    grad_scaler = GradScaler() if getattr(config, "amp", False) else None
    loss_fn = recipe.build_loss_fn(config, device).to(device)

    if local_rank == 0:
        model_without_ddp = model
        writer = get_writer(config.save_path)
        logger = get_logger(os.path.join(config.save_path, "train.log"))
        logger.info(get_config(config.to_dict(), mute=False))
        val_loader = get_dataloader(config, split="val", ddp=False)
    else:
        model_without_ddp = None
        writer = None
        logger = None
        val_loader = None

    config.batch_size = int(config.batch_size / nprocs)
    train_loader, sampler = get_dataloader(config, split="train", ddp=ddp)

    optimizer, scheduler = recipe.build_optimizer_and_scheduler(config, model, len(train_loader))
    model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_val_scores, best_val_scores = load_checkpoint(
        config, model, optimizer, scheduler, grad_scaler
    )

    if ddp:
        model = DDP(
            nn.SyncBatchNorm.convert_sync_batchnorm(model),
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=recipe.ddp_find_unused_parameters,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model, optimizer, grad_scaler, loss_info = recipe.train_one_epoch(
            model, train_loader, loss_fn, optimizer, grad_scaler, device, local_rank, nprocs
        )

        if scheduler is not None:
            if recipe.scheduler_step is not None:
                recipe.scheduler_step(scheduler, epoch)
            else:
                scheduler.step()

        barrier(ddp)

        if local_rank == 0:
            assert writer is not None
            assert logger is not None
            assert val_loader is not None
            assert model_without_ddp is not None

            eval_enabled = (epoch >= config.eval_start) and ((epoch - config.eval_start) % config.eval_freq == 0)
            update_train_result(epoch, loss_info, writer)
            log(logger, epoch, config.epochs, loss_info=loss_info)

            if eval_enabled:
                state_dict = model.module.state_dict() if ddp else model.state_dict()
                model_without_ddp.load_state_dict(state_dict)
                curr_val_scores = recipe.evaluate(model_without_ddp, val_loader, device)
                hist_val_scores, best_val_scores = update_eval_result(
                    epoch,
                    curr_val_scores,
                    hist_val_scores,
                    best_val_scores,
                    writer,
                    state_dict,
                    config.save_path,
                )
                log(logger, epoch, config.epochs, None, curr_val_scores, best_val_scores)

            if epoch % config.save_freq == 0:
                save_checkpoint(
                    epoch + 1,
                    model.module.state_dict() if ddp else model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict() if scheduler is not None else None,
                    loss_info,
                    hist_val_scores,
                    best_val_scores,
                    config.save_path,
                    grad_scaler.state_dict() if grad_scaler is not None else None,
                )

        barrier(ddp)

    if local_rank == 0:
        assert writer is not None
        writer.close()
        print("Training completed. Best scores:")
        print(best_val_scores)

    cleanup(ddp)

