from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from serving.config import ServeSettings
from serving.metrics import PREDICTIONS, metrics_response, prometheus_middleware
from serving.predictor import Predictor


app = FastAPI(title="simple-crowdcounting", version="0.1.0")
app.middleware("http")(prometheus_middleware)
_predictor: Predictor | None = None


@app.on_event("startup")
def _startup() -> None:
    global _predictor
    settings = ServeSettings.from_env()
    _predictor = Predictor.create(
        network=settings.network,
        checkpoint=settings.checkpoint,
        device=settings.device,
        input_size=settings.input_size,
        threshold=settings.threshold,
        sliding_window=settings.sliding_window,
        window_size=settings.window_size,
        stride=settings.stride,
    )


@app.get("/health")
def health() -> dict:
    if _predictor is None:
        return {"status": "starting"}
    return {"status": "ok", "network": _predictor.network}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    image_bytes = await file.read()
    try:
        out = _predictor.predict_image_bytes(image_bytes)
        PREDICTIONS.labels(network=_predictor.network).inc()
        return out
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
