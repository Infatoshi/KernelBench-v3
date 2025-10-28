"""GPU hardware specification helpers loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class GpuSpec:
    """Normalized GPU hardware metadata."""

    name: str
    architecture: str
    memory_gb: int | None = None
    memory_type: str | None = None
    tensor_core_generation: str | None = None


def _config_path() -> Path:
    """Return the path to the GPU spec YAML file."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "hardware.yaml"


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    path = _config_path()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, found {type(payload).__name__}")
    return payload


def default_gpu_name() -> str:
    """Return the default GPU name configured for Modal runs."""
    data = _load_config()
    value = data.get("default_gpu", "A100")
    if not isinstance(value, str):
        raise ValueError("`default_gpu` must be a string in config/hardware.yaml")
    return value


def gpu_spec(name: str | None) -> GpuSpec:
    """Lookup the GPU specification for the given Modal GPU name."""
    data = _load_config()
    inventory = data.get("gpus") or {}
    if not isinstance(inventory, dict):
        raise ValueError("`gpus` must be a mapping in config/hardware.yaml")

    candidate = (name or "").strip()
    if not candidate:
        candidate = default_gpu_name()

    record = inventory.get(candidate)
    canonical_name = candidate
    if record is None:
        # Attempt case-insensitive match
        lowered = {key.lower(): key for key in inventory.keys()}
        key = lowered.get(candidate.lower())
        if key:
            record = inventory.get(key)
            canonical_name = key

    if not isinstance(record, dict):
        return GpuSpec(name=canonical_name, architecture="unknown")

    return GpuSpec(
        name=canonical_name,
        architecture=str(record.get("architecture", "unknown")),
        memory_gb=_safe_int(record.get("memory_gb")),
        memory_type=_safe_str(record.get("memory_type")),
        tensor_core_generation=_safe_str(record.get("tensor_core_generation")),
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
