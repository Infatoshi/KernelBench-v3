#!/usr/bin/env python3
"""
Batch Evaluation Runner for CuTeBench.

Wraps batch_eval pipeline and routes execution through cute_eval.
"""

from __future__ import annotations

import sys

import batch_eval as base_batch_eval
import cute_eval
from src.config.runtime_validation import (
    ERROR_MESSAGES as RUNTIME_ERROR_MESSAGES,
    ensure_gpu_arg,
    parse_requested_gpus,
    validate_gpus as validate_allowed_gpus,
    validate_platform,
)

ALLOWED_GPUS = ["H100", "B200"]
BENCHMARK_NAME = "CuTeBench"
GPU_REASON = "CuTeBench targets Hopper/Blackwell tensor-core paths (H100, B200)."
ERROR_MESSAGES = dict(RUNTIME_ERROR_MESSAGES)

# Patch evaluation entrypoints used by base_batch_eval.
base_batch_eval.MODELS = cute_eval.MODELS
base_batch_eval.find_problems = cute_eval.find_problems
base_batch_eval.get_model_config = cute_eval.get_model_config
base_batch_eval.run_agent_on_modal = cute_eval.run_agent_on_modal
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
    base_batch_eval.main()


if __name__ == "__main__":
    main()
