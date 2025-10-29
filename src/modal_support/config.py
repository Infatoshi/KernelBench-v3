"""Helpers for reading Modal raw-run configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from hardware.specs import default_gpu_name, gpu_spec

@dataclass(frozen=True)
class ModalImageSpec:
    """Image settings for Modal GPU runs."""

    cuda_version: str
    os_tag: str
    python_version: str = "3.12"

    @property
    def registry_tag(self) -> str:
        """Return CUDA registry tag, e.g. 12.8.1-devel-ubuntu24.04."""
        return f"{self.cuda_version}-devel-{self.os_tag}"


@dataclass(frozen=True)
class ModalGpuSpec:
    """GPU allocation for Modal jobs."""

    name: str
    count: int = 1

    def to_modal_argument(self) -> str:
        """Return value suitable for the modal.function gpu= argument."""
        if self.count != 1:
            raise ValueError("Modal GPU counts greater than 1 are not supported yet.")
        return self.name


MODAL_FUNCTION_TIMEOUT_MIN = 10
MODAL_FUNCTION_TIMEOUT_MAX = 24 * 60 * 60  # 24 hours


@dataclass(frozen=True)
class ModalTimeoutSpec:
    """Timeout settings for Modal subprocess management."""

    process_seconds: int = 0

    def normalized(self) -> "ModalTimeoutSpec":
        """Coerce negative values to zero while leaving positive values unchanged."""
        seconds = int(self.process_seconds)
        if seconds <= 0:
            return ModalTimeoutSpec(process_seconds=0)
        return ModalTimeoutSpec(process_seconds=seconds)

    def modal_function_timeout(self) -> int:
        """Return a Modal-compatible execution timeout in seconds."""
        seconds = int(self.process_seconds)
        if seconds <= 0:
            return MODAL_FUNCTION_TIMEOUT_MAX
        bounded = max(MODAL_FUNCTION_TIMEOUT_MIN, min(seconds, MODAL_FUNCTION_TIMEOUT_MAX))
        return bounded


@dataclass(frozen=True)
class ModalRawConfig:
    """Aggregate Modal configuration for raw kernel execution."""

    yaml_path: Path
    image: ModalImageSpec
    gpu: ModalGpuSpec
    timeouts: ModalTimeoutSpec

    @classmethod
    def load(cls, yaml_path: Path) -> "ModalRawConfig":
        data = _load_yaml(yaml_path)
        modal_data = data.get("modal") or {}
        if not isinstance(modal_data, dict):
            raise ValueError(f"'modal' section must be a mapping in {yaml_path}")

        image_data = modal_data.get("image") or {}
        gpu_data = modal_data.get("gpu") or {}
        timeout_data = modal_data.get("timeouts") or {}

        image = _resolve_image(image_data)
        gpu = _resolve_gpu(gpu_data)
        timeouts = _resolve_timeouts(timeout_data)

        return cls(
            yaml_path=yaml_path,
            image=image,
            gpu=gpu,
            timeouts=timeouts,
        )


def _resolve_image(image_data: dict[str, Any]) -> ModalImageSpec:
    cuda_version = str(image_data.get("cuda_version") or "12.8.1")
    os_tag = str(image_data.get("os_tag") or "ubuntu24.04")
    python_version = str(image_data.get("python_version") or "3.12")
    return ModalImageSpec(
        cuda_version=cuda_version,
        os_tag=os_tag,
        python_version=python_version,
    )


def _resolve_gpu(gpu_data: dict[str, Any]) -> ModalGpuSpec:
    env_gpu = os.environ.get("KB3_MODAL_GPU_NAME")
    configured_name = gpu_data.get("name") or env_gpu or default_gpu_name()
    spec = gpu_spec(str(configured_name))

    count_value = gpu_data.get("count") or os.environ.get("KB3_MODAL_GPU_COUNT") or spec_count_default()
    try:
        count = int(count_value)
    except (TypeError, ValueError):
        count = 1
    if count <= 0:
        count = 1

    return ModalGpuSpec(name=spec.name, count=count)


def _resolve_timeouts(timeout_data: dict[str, Any]) -> ModalTimeoutSpec:
    process_seconds = timeout_data.get("process_seconds")
    if process_seconds is None:
        env_override = os.environ.get("KB3_MODAL_TIMEOUT_SECONDS")
        process_seconds = env_override if env_override is not None else 0
    try:
        seconds = int(process_seconds)
    except (TypeError, ValueError):
        seconds = 0
    return ModalTimeoutSpec(process_seconds=seconds).normalized()


def spec_count_default() -> int:
    env_default = os.environ.get("KB3_MODAL_GPU_COUNT_DEFAULT")
    if env_default is not None:
        try:
            value = int(env_default)
            if value > 0:
                return value
        except ValueError:
            pass
    return 1


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, found {type(payload).__name__}")
    return payload
