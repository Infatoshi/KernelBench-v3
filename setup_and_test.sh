#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log() {
  echo "[setup] $1"
}

if ! command -v uv >/dev/null 2>&1; then
  log "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  log "uv installed"
else
  log "uv already available"
fi

if [[ ! -d .venv ]]; then
  log "Creating virtual environment via uv"
  uv venv .venv
else
  log "Virtual environment exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing project dependencies"
uv pip install -r requirements.txt
uv pip install -e .

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "CUDA-compatible GPU not detected (nvidia-smi missing)."
  log "Refer to https://docs.nvidia.com/cuda/ for installation instructions."
  exit 1
fi

log "GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  log "GROQ_API_KEY not set. Export GROQ_API_KEY before running benchmarks."
  exit 1
fi

log "Running raw Triton smoke test"
uv run python eval.py --config configs/smoke_raw.yaml --verbose || {
  log "Raw Triton smoke test failed"
  exit 1
}

log "Running raw CUDA smoke test"
uv run python eval.py --config configs/smoke_raw_cuda.yaml --verbose || {
  log "Raw CUDA smoke test failed"
  exit 1
}

log "Running agentic Triton smoke test"
uv run python eval.py --config configs/smoke_agentic.yaml --verbose || {
  log "Agentic Triton smoke test failed"
  exit 1
}

log "Running agentic CUDA smoke test"
uv run python eval.py --config configs/smoke_agentic_cuda.yaml --verbose || {
  log "Agentic CUDA smoke test failed"
  exit 1
}

log "Environment setup and smoke tests completed successfully"

