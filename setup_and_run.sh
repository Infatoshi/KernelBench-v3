#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash setup_and_run.sh <config-path>" >&2
  exit 1
fi

CONFIG_PATH_INPUT=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${SCRIPT_DIR}
CONFIG_ABS=$(cd "${REPO_ROOT}" && realpath "${CONFIG_PATH_INPUT}")
if [ ! -f "${CONFIG_ABS}" ]; then
  echo "[setup] Config not found: ${CONFIG_ABS}" >&2
  exit 1
fi
CONFIG_REL=$(python3 - <<'PY' "${CONFIG_ABS}" "${REPO_ROOT}"
import os, sys
abs_path, repo_root = sys.argv[1], sys.argv[2]
print(os.path.relpath(abs_path, repo_root))
PY
)

GENERATION_REQUIREMENT=$(uv run python - "${CONFIG_ABS}" <<'PY'
import sys, yaml
config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
models = data.get("models") or []
defaults = data.get("defaults") or {}
defaults_generation = defaults.get("generation") or {}
global_generation = data.get("generation") or {}
need_keys = False
for model in models:
    merged = {}
    merged.update(defaults_generation)
    merged.update(global_generation)
    merged.update(model.get("generation") or {})
    mode = (merged.get("mode") or "local")
    if str(mode).strip().lower() != "local":
        need_keys = True
        break
print("require_keys" if need_keys else "skip_keys")
PY
)
if [ "${GENERATION_REQUIREMENT}" = "skip_keys" ]; then
  SKIP_PROVIDER_KEYS=1
  echo "[setup] Provider keys not required (using local kernel generation)."
else
  SKIP_PROVIDER_KEYS=0
fi

UV_TIMEOUT=${UV_TIMEOUT:-120}
MODAL_TOKEN_TIMEOUT=${MODAL_TOKEN_TIMEOUT:-120}

with_timeout() {
  local seconds=$1
  shift
  if [ "${seconds}" -eq 0 ]; then
    "$@"
    return $?
  fi
  timeout "${seconds}s" "$@"
  local status=$?
  if [ $status -eq 124 ]; then
    echo "[setup] Command timed out after ${seconds}s: $*" >&2
  fi
  return $status
}

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv (timeout ${UV_TIMEOUT}s)..."
  with_timeout "${UV_TIMEOUT}" sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[setup] uv already present at $(command -v uv)"
fi

cd "${REPO_ROOT}"
echo "[setup] Synchronizing project dependencies with uv..."
with_timeout "${UV_TIMEOUT}" uv sync

if [ "${SKIP_PROVIDER_KEYS}" != "1" ]; then
  echo "[setup] Validating provider API credentials for ${CONFIG_REL}..."
  if ! with_timeout "${UV_TIMEOUT}" uv run python - <<'PY' "${CONFIG_REL}" "${REPO_ROOT}"; then
import sys
from pathlib import Path

import yaml


CONFIG_REL = Path(sys.argv[1])
REPO_ROOT = Path(sys.argv[2])
CONFIG_PATH = (REPO_ROOT / CONFIG_REL).resolve()

SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from providers import (
    provider_api_key_env_candidates,
    resolve_provider_api_key,
)


def _collect_providers(node, accumulator):
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("provider"):
                _record_provider(value, accumulator)
            elif key == "providers":
                _record_provider(value, accumulator)
            _collect_providers(value, accumulator)
    elif isinstance(node, list):
        for item in node:
            _collect_providers(item, accumulator)


def _record_provider(value, accumulator):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            accumulator.add(normalized)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _record_provider(item, accumulator)


with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config_payload = yaml.safe_load(handle) or {}

providers = set()
_collect_providers(config_payload, providers)

missing = []
for provider in sorted(providers):
    if resolve_provider_api_key(provider) is None:
        missing.append((provider, provider_api_key_env_candidates(provider)))

if missing:
    print("[setup] Missing provider API keys:")
    for provider, candidates in missing:
        choices = ", ".join(candidates) if candidates else "KB3_LLM_API_KEY"
        print(f"  - {provider}: set one of [{choices}]")
    raise SystemExit(1)

if providers:
    print("[setup] Provider API keys detected for:", ", ".join(sorted(providers)))
else:
    print("[setup] No provider API keys required by configuration.")
PY
    echo "[setup] Provider API credential check failed." >&2
    exit 1
  fi
  echo "[setup] Provider API credentials verified."
else
  echo "[setup] Skipping provider API credential validation."
fi

echo "[setup] Checking Modal auth token..."
if with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run python - <<'PY'
import modal.config, sys
profiles = modal.config.config_profiles()
sys.exit(0 if profiles else 1)
PY
then
  echo "[setup] Modal token already configured."
else
  echo "[setup] No Modal token detected; launching interactive 'modal token new'."
  with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run modal token new
fi

if [ "${SKIP_PROVIDER_KEYS}" != "1" ]; then
  echo "[setup] Ensuring Modal secret kb3-llm-keys exists..."
  for var in GEMINI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY XAI_API_KEY GROQ_API_KEY; do
    if [ -z "${!var:-}" ]; then
      echo "[setup] Missing required env var: ${var}" >&2
      exit 1
    fi
  done
  if with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run modal secret get kb3-llm-keys >/dev/null 2>&1; then
    echo "[setup] Updating existing Modal secret kb3-llm-keys..."
  else
    echo "[setup] Creating Modal secret kb3-llm-keys..."
  fi
  with_timeout "${MODAL_TOKEN_TIMEOUT}" uv run -- modal secret create --force kb3-llm-keys \
    "GEMINI_API_KEY=${GEMINI_API_KEY}" \
    "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" \
    "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
    "XAI_API_KEY=${XAI_API_KEY}" \
    "GROQ_API_KEY=${GROQ_API_KEY}"
else
  echo "[setup] Skipping Modal secret provisioning."
fi

export KB3_MODAL_TIMEOUT_SECONDS="${KB3_MODAL_TIMEOUT_SECONDS:-0}"

echo "[setup] Precomputing local kernels for ${CONFIG_REL}..."
LOCAL_KERNEL_OUTPUT=$(uv run python scripts/generate_local_kernels.py "${CONFIG_REL}")
echo "${LOCAL_KERNEL_OUTPUT}"
KB3_LOCAL_KERNEL_DIR=$(printf '%s\n' "${LOCAL_KERNEL_OUTPUT}" | awk -F= '/^LOCAL_KERNEL_DIR=/{print $2}')
if [ -z "${KB3_LOCAL_KERNEL_DIR}" ]; then
  echo "[setup] Failed to determine local kernel directory." >&2
  exit 1
fi
KB3_LOCAL_KERNEL_DIR=$(python3 - "${KB3_LOCAL_KERNEL_DIR}" "${REPO_ROOT}" <<'PY'
import os
import sys

path = sys.argv[1]
root = sys.argv[2]
try:
    rel = os.path.relpath(path, root)
    if rel.startswith("../"):
        raise ValueError
    print(rel)
except Exception:
    print(path)
PY
)
export KB3_LOCAL_KERNEL_DIR

echo "[setup] Launching Modal raw evaluation for ${CONFIG_REL}..."
KB3_MODAL_CONFIG_PATH="${CONFIG_REL}" uv run modal run tools/modal_raw.py
