#!/usr/bin/env bash
set -euo pipefail

REMOTE_DIR="${1:-$HOME/.dvcstore/simple-crowdcounting}"

echo "Using local DVC remote: ${REMOTE_DIR}"

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC not found. Installing via pip..."
  python -m pip install dvc
fi

if [ ! -d ".dvc" ]; then
  echo "Initializing DVC..."
  dvc init -q
fi

mkdir -p "${REMOTE_DIR}"

set +e
dvc remote add -d localstore "${REMOTE_DIR}" >/dev/null 2>&1
set -e
dvc remote modify --local localstore url "${REMOTE_DIR}" >/dev/null 2>&1

echo "Done."
echo "Next:"
echo "  - Track raw data: dvc add data/raw"
echo "  - Push to local remote: dvc push"

