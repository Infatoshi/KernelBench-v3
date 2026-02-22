#!/usr/bin/env python3
"""
CUTLASS-based KernelBench Evaluation.

Reuses modal_eval pipeline while swapping prompts to request CUTLASS 3.x kernels.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import modal_eval as cuda_eval
from src.config.benchmark_problems import find_problems_for_benchmark
from src.prompts.cutlass_system import (
    get_cutlass_reasoning_system_prompt,
    get_cutlass_system_prompt,
)

# Re-export shared data structures and helpers used by batch runners.
ModelConfig = cuda_eval.ModelConfig
EvalResult = cuda_eval.EvalResult
MODELS = cuda_eval.MODELS
GPU_SPECS = cuda_eval.GPU_SPECS
get_model_config = cuda_eval.get_model_config
PROJECT_ROOT = Path(__file__).parent


def find_problems(levels: list[int]):
    """CUTLASSBench uses tile-specialized problems only."""
    return find_problems_for_benchmark(PROJECT_ROOT, benchmark="cutlass", levels=levels)


@contextmanager
def _cutlass_eval_overrides() -> Iterator[None]:
    """Enable CUTLASS prompts for modal_eval execution."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt

    cuda_eval.get_system_prompt = get_cutlass_system_prompt
    cuda_eval.get_reasoning_system_prompt = get_cutlass_reasoning_system_prompt
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
) -> EvalResult:
    """Run an LLM agent using CUTLASS prompts on existing sandbox paths."""
    if not model_config.reasoning_mode:
        model_config = replace(model_config, reasoning_mode=True)

    with _cutlass_eval_overrides():
        return cuda_eval.run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_name,
            level=level,
            max_turns=max_turns,
            backend="cutlass",
        )


def main() -> None:
    """Run modal_eval CLI with CUTLASS prompts enabled."""
    with _cutlass_eval_overrides():
        cuda_eval.main()


if __name__ == "__main__":
    main()
