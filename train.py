import argparse
import os

import torch
import torch.multiprocessing as mp

from core.distributed import init_seeds, setup
from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--save-path', type=str, required=True, help="save path")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--input-size', type=int, default=512, help="input image size")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'steerer', 'ffnet'],
                        help="Model architecture to use.")
    parser.add_argument('--eval-start', type=int, default=0, help="Epoch to start evaluation.")
    parser.add_argument('--eval-freq', type=int, default=1, help="Frequency (in epochs) to run evaluation.")
    parser.add_argument('--save-freq', type=int, default=1, help="Frequency (in epochs) to save.")
    parser.add_argument('--local-rank', type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action="store_true")

    # Optional MLOps (MLflow)
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking (optional).")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (optional).")
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="crowd-counting",
        help="MLflow experiment name (optional).",
    )
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="MLflow run name (optional).")
    parser.add_argument(
        "--mlflow-tags",
        type=str,
        default=None,
        help='Extra MLflow tags as JSON string (e.g. \'{"stage":"dev"}\').',
    )

    args = parser.parse_args()

    # Set up DDP
    args.nprocs = torch.cuda.device_count()
    print(f"Using {args.nprocs} GPUs.")

    # save path
    args.save_path = os.path.join("output", "train", args.save_path)

    # update_config
    config = update_config(args).flatten()

    # run
    if args.nprocs > 1:
        config.lr = config.lr * args.nprocs
        mp.spawn(run, nprocs=args.nprocs, args=(args.nprocs, config))
    else:
        run(0, 1, config)


def run(local_rank: int, nprocs: int, config: object) -> None:
    from core import trainer as core_trainer

    # Optional MLflow run (rank 0 only).
    try:
        from core import mlops

        # Map CLI args (mlflow-*) into the config attributes expected by core.mlops.
        config.mlflow = getattr(config, "mlflow", False)
        config.mlflow_uri = getattr(config, "mlflow_uri", None)
        config.mlflow_experiment = getattr(config, "mlflow_experiment", "crowd-counting")
        config.mlflow_run_name = getattr(config, "mlflow_run_name", None)
        config.mlflow_tags = getattr(config, "mlflow_tags", None)

        mlops.configure_from_config(config)
        # Convert config params to a flat dict for MLflow params.
        params = config.to_dict() if hasattr(config, "to_dict") else None
        mlops.start_run(params=params)
        cfg_path = os.path.join("configs", f"{config.network.lower()}.yml")
        if os.path.isfile(cfg_path):
            mlops.log_artifact(cfg_path, artifact_path="configs")
    except Exception:
        # Keep training runnable even without MLflow installed.
        pass

    if nprocs > 1:
        print(f"Rank {local_rank} process among {nprocs} processes.")
        init_seeds(config.seed + local_rank)
        setup(local_rank, nprocs)
        print(f"Initialized successfully. Training with {nprocs} GPUs.")
    try:
        core_trainer.run(local_rank, nprocs, config)
    finally:
        try:
            from core import mlops

            mlops.end_run()
        except Exception:
            pass


if __name__ == '__main__':
    main()
