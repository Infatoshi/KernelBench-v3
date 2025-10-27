"""Helpers for reading Modal raw-run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


@dataclass(frozen=True)
class ModalTimeoutSpec:
    """Timeout settings for Modal subprocess management."""

    process_seconds: int = 120

    def clamp(self) -> "ModalTimeoutSpec":
        """Enforce an upper bound of 120 seconds by default."""
        # Modal workloads can be long, but scripts must default to <=120 seconds.
        seconds = int(self.process_seconds)
        if seconds <= 0:
            return ModalTimeoutSpec(process_seconds=0)
        bounded = max(1, min(seconds, 120))
        return ModalTimeoutSpec(process_seconds=bounded)


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
        modal_data = data.get("modal")
        if not isinstance(modal_data, dict):
            raise ValueError(f"'modal' section missing in {yaml_path}")

        image_data = modal_data.get("image") or {}
        gpu_data = modal_data.get("gpu") or {}
        timeout_data = modal_data.get("timeouts") or {}

        image = ModalImageSpec(
            cuda_version=str(image_data.get("cuda_version", "12.8.1")),
            os_tag=str(image_data.get("os_tag", "ubuntu24.04")),
            python_version=str(image_data.get("python_version", "3.12")),
        )
        gpu = ModalGpuSpec(
            name=str(gpu_data.get("name", "H100")),
            count=int(gpu_data.get("count", 1)),
        )
        timeouts = ModalTimeoutSpec(
            process_seconds=int(timeout_data.get("process_seconds", 120))
        ).clamp()

        return cls(
            yaml_path=yaml_path,
            image=image,
            gpu=gpu,
            timeouts=timeouts,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}, found {type(payload).__name__}")
    return payload
