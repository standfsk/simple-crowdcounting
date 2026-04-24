import argparse
import importlib
import os

from core.utils import update_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--save-path', type=str, required=True, help="save path")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument('--network', type=str, required=True,
                        choices=['apgcc', 'clip_ebc', 'cltr', 'dmcount', 'fusioncount', 'steerer', 'ffnet'],
                        help="Model architecture to use.")
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--device', default="cpu", help="device to use. either gpu_id or cpu")
    parser.add_argument('--save', action="store_true", help="save result image")

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

    # save path
    args.save_path = os.path.join("output", "test", args.save_path)

    # load config
    config = update_config(args).flatten()

    # run
    # Optional MLflow run.
    try:
        from core import mlops

        config.mlflow = getattr(config, "mlflow", False)
        config.mlflow_uri = getattr(config, "mlflow_uri", None)
        config.mlflow_experiment = getattr(config, "mlflow_experiment", "crowd-counting")
        config.mlflow_run_name = getattr(config, "mlflow_run_name", None)
        config.mlflow_tags = getattr(config, "mlflow_tags", None)

        mlops.configure_from_config(config)
        params = config.to_dict() if hasattr(config, "to_dict") else None
        mlops.start_run(params=params)
        cfg_path = os.path.join("configs", f"{config.network.lower()}.yml")
        if os.path.isfile(cfg_path):
            mlops.log_artifact(cfg_path, artifact_path="configs")
        if os.path.isfile(config.checkpoint):
            mlops.log_artifact(config.checkpoint, artifact_path="checkpoints")
    except Exception:
        pass

    try:
        run(config)
    finally:
        try:
            from core import mlops

            mlops.end_run()
        except Exception:
            pass

def run(config: object) -> None:
    test_module = importlib.import_module(f'models.{config.network}.test')
    test_module.test(config)

if __name__ == '__main__':
    main()
