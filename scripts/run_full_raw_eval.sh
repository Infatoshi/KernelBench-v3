#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG_PATH="${REPO_ROOT}/configs/all_providers_raw.yaml"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/full_raw_${timestamp}.log"

echo "[run_full_raw_eval] configuration: ${CONFIG_PATH}"
echo "[run_full_raw_eval] log file: ${log_file}"
echo "[run_full_raw_eval] starting at ${timestamp}"

cd "${REPO_ROOT}"

uv run python eval.py --config "${CONFIG_PATH}" "$@" | tee "${log_file}"

echo "[run_full_raw_eval] completed at $(date +%Y%m%d_%H%M%S)"
