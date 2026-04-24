## Data layout (recommended)

This repo supports a standardized crowd-counting data pipeline:

```
data/
  raw/
    images/
    annotations/
  processed/
    images/
    annotations/   # JSON (1 per image)
  cache/
    density_maps/
```

Canonical annotation schema (per image JSON):

```json
{
  "image_path": "images/train/img_001.jpg",
  "points": [[x1, y1], [x2, y2]],
  "count": 2,
  "meta": { "dataset": "sha", "resolution": [1024, 768] }
}
```

Conversion CLI:

```
python convert.py --input data/raw --output data/processed --type txt
```

## DVC data versioning (local-only)
This repo is configured for DVC-based data versioning with a **local filesystem remote**.

One-time setup (Windows PowerShell):
```
powershell -ExecutionPolicy Bypass -File .\\mlops\\dvc_setup_local.ps1
```

Version raw data and push into the local DVC remote:
```
dvc add data/raw
git add data/raw.dvc data/.gitignore
dvc push
```

Optional: generate reproducible density-map cache:

```
python generate_density_cache.py --processed data/processed --out data/cache/density_maps --sigma 4.0
```
