"""Graphics backend -- Triton compute kernels for graphics workloads."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

from src.backends import Backend, register
from src.backends.triton import _run_triton_benchmark
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_graphics
from src.prompts.graphics_triton_system import (
    get_graphics_triton_reasoning_system_prompt,
    get_graphics_triton_system_prompt,
)


def _read_graphics_level(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return 1
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "GRAPHICS_LEVEL":
            try:
                return ast.literal_eval(node.value)
            except Exception:
                pass
    return 1


@register("graphics")
class GraphicsBackend(Backend):
    name = "graphics"
    benchmark_name = "GraphicsBench"
    allowed_gpus = ["RTX3090"]
    gpu_reason = "GraphicsBench uses Triton on RTX3090 for graphics compute workloads."

    @property
    def force_reasoning_mode(self) -> bool:
        return True

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_graphics_triton_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_graphics_triton_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        graphics_dir = project_root / "problems" / "graphics"
        problems: List[Tuple[int, Path]] = []
        if not graphics_dir.exists():
            return problems
        requested = set(levels)
        for f in sorted(graphics_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            level = _read_graphics_level(f)
            if level in requested:
                problems.append((level, f))
        return problems

    def validate_solution(self, solution_code):
        return validate_graphics(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("graphics")

    def build_template_solution(self):
        return _build_template_solution("graphics")

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        return _run_triton_benchmark(sandbox, solution_path, **kwargs)
