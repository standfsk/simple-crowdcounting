Param(
  [string]$RemoteDir = "$env:USERPROFILE\\.dvcstore\\simple-crowdcounting"
)

$ErrorActionPreference = "Stop"

Write-Host "Using local DVC remote: $RemoteDir"

if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
  Write-Host "DVC not found. Installing via pip..."
  python -m pip install dvc
}

if (-not (Test-Path ".dvc")) {
  Write-Host "Initializing DVC..."
  dvc init -q
}

New-Item -ItemType Directory -Force -Path $RemoteDir | Out-Null

# Keep machine-specific path in .dvc/config.local (not committed).
try {
  dvc remote add -d localstore $RemoteDir | Out-Null
} catch {
  # If remote exists, keep going.
}

dvc remote modify --local localstore url $RemoteDir | Out-Null

Write-Host "Done."
Write-Host "Next:"
Write-Host "  - Track raw data: dvc add data/raw"
Write-Host "  - Push to local remote: dvc push"

