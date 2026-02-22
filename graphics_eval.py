#!/usr/bin/env python3
"""
GraphicsBench Evaluation.

Uses modal_eval execution loop with graphics-specific prompts and problem discovery.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator, List, Tuple

import modal_eval as cuda_eval
from src.prompts.graphics_triton_system import (
    get_graphics_triton_reasoning_system_prompt,
    get_graphics_triton_system_prompt,
)

# Re-export shared data structures and helpers used by batch runners.
ModelConfig = cuda_eval.ModelConfig
EvalResult = cuda_eval.EvalResult
MODELS = cuda_eval.MODELS
GPU_SPECS = cuda_eval.GPU_SPECS
get_model_config = cuda_eval.get_model_config

PROJECT_ROOT = Path(__file__).parent
GRAPHICS_DIR = PROJECT_ROOT / "KernelBench" / "graphics"


def _read_graphics_level(path: Path) -> int:
    """Extract GRAPHICS_LEVEL = int from file, default 1."""
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return 1

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GRAPHICS_LEVEL":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        return node.value.value
    return 1


def find_problems(levels: List[int]) -> List[Tuple[int, Path]]:
    """Find graphics problems for specified graphics levels."""
    if not GRAPHICS_DIR.exists():
        return []

    problems: List[Tuple[int, Path]] = []
    for problem_file in sorted(GRAPHICS_DIR.glob("*.py")):
        if problem_file.name.startswith("_"):
            continue
        level = _read_graphics_level(problem_file)
        if level in levels:
            problems.append((level, problem_file))
    return problems


def _prompt_funcs_for_level(level: int):
    """GraphicsBench currently uses Triton prompts for all levels."""
    _ = level
    return get_graphics_triton_system_prompt, get_graphics_triton_reasoning_system_prompt


@contextmanager
def _graphics_eval_overrides(level: int) -> Iterator[None]:
    """Enable graphics prompts for selected level."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt

    system_prompt_fn, reasoning_prompt_fn = _prompt_funcs_for_level(level)
    cuda_eval.get_system_prompt = system_prompt_fn
    cuda_eval.get_reasoning_system_prompt = reasoning_prompt_fn
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
    """Run an LLM agent using graphics prompts."""
    if not model_config.reasoning_mode:
        model_config = replace(model_config, reasoning_mode=True)

    with _graphics_eval_overrides(level):
        return cuda_eval.run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_name,
            level=level,
            max_turns=max_turns,
            backend="graphics",
        )


def main() -> None:
    """Run modal_eval CLI with graphics prompts enabled."""
    # CLI path does not supply problem level before dispatch. Default to level 1 prompts.
    with _graphics_eval_overrides(level=1):
        cuda_eval.main()


if __name__ == "__main__":
    main()
