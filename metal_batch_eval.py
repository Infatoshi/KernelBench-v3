#!/usr/bin/env python3
"""
Batch Evaluation Runner for MetalBench.

Wraps the existing batch_eval pipeline and routes execution through
metal_eval.run_agent_on_modal (Metal prompts + MPS benchmark path).
"""

from __future__ import annotations

import sys

import batch_eval as base_batch_eval
import metal_eval
from src.config.runtime_validation import (
    ERROR_MESSAGES as RUNTIME_ERROR_MESSAGES,
    ensure_gpu_arg,
    parse_requested_gpus,
    validate_gpus as validate_allowed_gpus,
    validate_platform,
)

ALLOWED_GPUS = ["M4MAX"]
BENCHMARK_NAME = "MetalBench"
GPU_REASON = "MetalBench targets Apple Silicon Metal runtime (M4MAX only)."
ERROR_MESSAGES = dict(RUNTIME_ERROR_MESSAGES)

# Patch evaluation entrypoints used by base_batch_eval.
base_batch_eval.MODELS = metal_eval.MODELS
base_batch_eval.find_problems = metal_eval.find_problems
base_batch_eval.get_model_config = metal_eval.get_model_config
base_batch_eval.run_agent_on_modal = metal_eval.run_agent_on_modal
base_batch_eval.GPUS = ALLOWED_GPUS
base_batch_eval.ALLOWED_GPUS = ALLOWED_GPUS
base_batch_eval.BENCHMARK_NAME = BENCHMARK_NAME
base_batch_eval.GPU_REASON = GPU_REASON


def validate_gpus(requested: list[str]) -> None:
    validate_allowed_gpus(requested, ALLOWED_GPUS, BENCHMARK_NAME, GPU_REASON)


def validate_benchmark_platform(requested: list[str]) -> None:
    validate_platform(requested, BENCHMARK_NAME)


def main() -> None:
    requested = parse_requested_gpus(sys.argv[1:], ALLOWED_GPUS)
    validate_gpus(requested)
    validate_benchmark_platform(requested)
    sys.argv = [sys.argv[0], *ensure_gpu_arg(sys.argv[1:], requested)]
    original_hardware_filter = base_batch_eval.get_problem_hardware_required
    # Problem HARDWARE_REQUIRED metadata in shared L1-L4 files is primarily CUDA-oriented.
    # MetalBench validates these same problem definitions on M4MAX, so do not filter by
    # HARDWARE_REQUIRED here.
    base_batch_eval.get_problem_hardware_required = lambda _problem_path: None
    try:
        base_batch_eval.main()
    finally:
        base_batch_eval.get_problem_hardware_required = original_hardware_filter


if __name__ == "__main__":
    main()
