#!/usr/bin/env python3
"""
CuTile Python KernelBench evaluation.

Reuses modal_eval pipeline with CuTile-Python-specific prompts and
CuTile-specific problem discovery (KernelBench/cutile).

Note: with CUDA 13.1 tileiras in this runtime, CuTile compilation is
Blackwell-only (B200).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import modal_eval as cuda_eval
from src.config.benchmark_problems import find_problems_for_benchmark
from src.prompts.cutile_system import (
    get_cutile_reasoning_system_prompt,
    get_cutile_system_prompt,
)

# Re-export shared structures
ModelConfig = cuda_eval.ModelConfig
EvalResult = cuda_eval.EvalResult
MODELS = cuda_eval.MODELS
GPU_SPECS = cuda_eval.GPU_SPECS
get_model_config = cuda_eval.get_model_config

PROJECT_ROOT = Path(__file__).parent


def find_problems(levels: list[int]):
    """Find CuTile-specific problems only."""
    return find_problems_for_benchmark(PROJECT_ROOT, benchmark="cutile", levels=levels)


@contextmanager
def _cutile_eval_overrides() -> Iterator[None]:
    """Enable CuTile prompts for modal_eval execution."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt

    cuda_eval.get_system_prompt = get_cutile_system_prompt
    cuda_eval.get_reasoning_system_prompt = get_cutile_reasoning_system_prompt
    try:
        yield
    finally:
        cuda_eval.get_system_prompt = original_system_prompt
        cuda_eval.get_reasoning_system_prompt = original_reasoning_prompt


def run_agent_on_modal(
    model_config: ModelConfig,
    gpu: str,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20,
    **kwargs,
) -> EvalResult:
    """Run agent with CuTile prompts."""
    if gpu != "B200":
        raise ValueError("CuTileBench currently supports B200 only (tileiras on CUDA 13.1).")

    # CuTile is very new; force reasoning mode for consistent code extraction.
    if not model_config.reasoning_mode:
        model_config = replace(model_config, reasoning_mode=True)

    with _cutile_eval_overrides():
        return cuda_eval.run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_name,
            level=level,
            max_turns=max_turns,
            backend="cutile",
            **kwargs,
        )


def main() -> None:
    """Run modal_eval CLI with CuTile prompts."""
    with _cutile_eval_overrides():
        cuda_eval.main()


if __name__ == "__main__":
    main()
