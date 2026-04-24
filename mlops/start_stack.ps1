$ErrorActionPreference = "Stop"

Write-Host "Starting local MLOps stack (MLflow + API)..."
docker compose -f .\\docker-compose.stack.yml up --build

