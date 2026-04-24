#!/usr/bin/env bash
set -euo pipefail

echo "Starting local MLOps stack (MLflow + API)..."
docker compose -f ./docker-compose.stack.yml up --build

