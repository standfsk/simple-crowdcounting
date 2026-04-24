from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency (seconds)", ["method", "path"])
PREDICTIONS = Counter("predictions_total", "Total prediction requests", ["network"])


async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    path = request.url.path
    method = request.method
    status = str(response.status_code)

    LATENCY.labels(method=method, path=path).observe(duration)
    REQUESTS.labels(method=method, path=path, status=status).inc()
    return response


def metrics_response() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

