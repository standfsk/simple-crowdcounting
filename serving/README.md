## FastAPI serving (Docker)

### Prereqs
- A trained checkpoint file (e.g. `output/train/<run>/best.pt`)

### Run with docker-compose

1) Set `SERVE_CHECKPOINT` mount target:
- Default compose mounts `./output/train` -> `/models`
- Put your checkpoint at `./output/train/best.pt` or update the env var

2) Start:
```
docker compose up --build
```

3) Health:
```
curl http://localhost:8000/health
```

4) Predict:
```
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@path/to/image.jpg"
```

### Environment variables
- `SERVE_NETWORK` (default: `apgcc`)
- `SERVE_CHECKPOINT` (required)
- `SERVE_DEVICE` (default: `cpu`, or `0` for `cuda:0`)
- `SERVE_INPUT_SIZE` (optional: `512` or `640x640`)
- `SERVE_THRESHOLD` (apgcc point threshold, default: `0.5`)
- `SERVE_SLIDING_WINDOW` / `SERVE_WINDOW_SIZE` / `SERVE_STRIDE` (clip_ebc only, optional)

