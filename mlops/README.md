## MLOps starter (local-first)

This repository includes a minimal MLOps scaffold focused on:

- **Canonical data format**: 1 JSON annotation per image (`data/processed/annotations/*.json`)
- **Reproducible conversion**: `convert.py`
- **Data versioning (local-only)**: **DVC** with a filesystem remote
- **Optional experiment tracking**: **MLflow** (disabled by default)

### 0) DVC local storage (recommended)

Initialize + configure a local filesystem remote (stored in `.dvc/config.local`, not committed):

PowerShell:
```
powershell -ExecutionPolicy Bypass -File .\\mlops\\dvc_setup_local.ps1
```

Then version raw data:
```
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc .dvcignore dvc.yaml
dvc push
```

To restore on a new machine:
```
powershell -ExecutionPolicy Bypass -File .\\mlops\\dvc_setup_local.ps1
dvc pull
```

### Local-only “full stack” (MLflow + Serving + Monitoring)
Start everything locally via Docker Compose:
```
powershell -ExecutionPolicy Bypass -File .\\mlops\\start_stack.ps1
```

Endpoints:
- MLflow UI: `http://localhost:5000`
- Serving API: `http://localhost:8000` (`/docs`, `/health`, `/predict`, `/metrics`)
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

Typical flow:
1) Train with MLflow tracking:
   - `python train.py ... --mlflow --mlflow-uri http://localhost:5000`
2) Register a checkpoint:
   - `python mlops/register_model.py --network apgcc --checkpoint output/train/<run>/best.pt --version v1`
3) Promote to serving slot:
   - `python mlops/promote_model.py --network apgcc --version v1`
4) Restart API container (or restart stack) to pick up the new file.

### 1) Convert raw dataset into canonical format

Input layout:
```
data/raw/images
data/raw/annotations
```

Run conversion:
```
python convert.py --input data/raw --output data/processed --type txt
```

Generate manifests used by training:
```
python datasets/prepare.py --root data/processed/images --out-dir datasets
```

### 2) (Optional) density-map cache
```
python generate_density_cache.py --processed data/processed --out data/cache/density_maps --sigma 4.0
```

### 3) MLflow tracking

Start MLflow UI (local file store `./mlruns`):
```
mlflow ui --port 5000
```

Run training with MLflow enabled:
```
python train.py --save-path exp1 --network apgcc --mlflow --mlflow-experiment crowdcounting
```

Notes:
- In multi-GPU/DDP, **rank 0 only** logs metrics/artifacts.
- Training metrics are logged via `core/logging.py` hooks.
