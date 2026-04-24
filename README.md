
## Introduction
This repository provides an integrated process for training and evaluating multiple crowd counting models, each with their own original implementations and licenses.

#### Key features:
- Multi-gpu training support
- YAML-based model config files
- Modular structure for easy addition of new models

## Supported Models
| Model       | License       | Source |
|-------------|---------------|--------|
| APGCC       | MIT  | [APGCC](https://github.com/AaronCIH/APGCC) |
| CLIP-EBC    | MIT  | [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC) |
| CLTR        | MIT  | [CLTR](https://github.com/dk-liang/CLTR) |
| DMCount     | MIT  | [DMCount](https://github.com/cvlab-stonybrook/DM-Count) |
| FusionCount | MIT  | [FusionCount](https://github.com/Yiming-M/FusionCount) |
| STEERER     | MIT  | [STEERER](https://github.com/taohan10200/STEERER) |
| FFNet     | Unavailable  | [FFNet](https://github.com/erdongsanshi/Fuss-Free-structure) |

## Supported Datasets
- ShanghaiTech A & B
- NWPU
- UCF-QNRF
- JHU-Crowd

## Installation
### 1. Clone repository
```
https://github.com/standfsk/crowd-counting-framework.git
cd crowd-counting-framework
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

## Prepare dataset
```
cd datasets
python prepare.py
```

## Standardized data pipeline (MLOps-friendly)
This project supports converting any raw dataset format into a single canonical format:
**Image + point annotations** stored as **one JSON per image**.

Recommended layout:
```
data/raw/images
data/raw/annotations
data/processed/images
data/processed/annotations  # JSON (1 per image)
```

Convert raw -> processed:
```
python convert.py --input data/raw --output data/processed --type txt
```

Create `datasets/train.txt`, `datasets/valid.txt`, `datasets/test.txt` pointing to processed images:
```
python datasets/prepare.py --root data/processed/images --out-dir datasets
```

## Train
```
python train.py --save-path train --network apgcc
```

## Performance notes
- AMP: add `--amp` to `train.py` for faster training on GPU.
- DDP workers: `--num-workers` is treated as **per-process** (do not divide by GPU count).
- DDP metrics: training metrics (MAE/RMSE/precision/recall/F1/accuracy) are computed at **epoch end** by gathering counts to rank 0 (avoids per-batch GPU↔CPU sync).
- Density maps: point-based models skip density-map generation in the dataloader to reduce CPU overhead.

## (Optional) MLflow tracking
```
python train.py --save-path exp1 --network apgcc --mlflow --mlflow-experiment crowdcounting
mlflow ui --port 5000
```

## Test
```
python test.py --save-path test --network apgcc --checkpoint output/train/best.pt --device 0 --save
```

## Export
```
python export.py --save-path apgcc.onnx --network apgcc --backbone vgg16_bn --checkpoint output/train/best.pt 
```

## Serve (FastAPI + Docker)
See `serving/README.md`.

## Acknowledgement
This project builds upon the work of many researchers in the field of crowd counting.<br>
Full credit goes to original authors of supported models
- [APGCC](https://github.com/AaronCIH/APGCC)
- [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC)
- [CLTR](https://github.com/dk-liang/CLTR)
- [DMCount](https://github.com/cvlab-stonybrook/DM-Count)
- [FusionCount](https://github.com/Yiming-M/FusionCount)
- [STEERER](https://github.com/taohan10200/STEERER)
- [FFNet](https://github.com/erdongsanshi/Fuss-Free-structure)




