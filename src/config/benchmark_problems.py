"""Benchmark-to-problem mapping and problem metadata helpers."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BENCHMARK_PROBLEMS: Dict[str, Dict[str, Any]] = {
    "cuda": {
        "dirs": ["level1", "level2", "level3", "level4"],
        "hardware": ["RTX3090", "H100", "B200"],
    },
    "triton": {
        "dirs": ["level1", "level2", "level3", "level4"],
        "hardware": ["RTX3090", "H100", "B200"],
    },
    "cutlass": {
        "dirs": ["tile_specialized"],
        "hardware": ["H100", "B200"],
    },
    "cute": {
        "dirs": ["tile_specialized"],
        "hardware": ["H100", "B200"],
    },
    "cutile": {
        "dirs": ["cutile"],
        "hardware": ["B200"],
    },
    "metal": {
        "dirs": ["level1", "level2", "level3", "level4"],
        "hardware": ["M4MAX"],
    },
    "graphics": {
        "dirs": ["graphics"],
        "hardware": ["RTX3090"],
    },
}


@lru_cache(maxsize=2048)
def _read_module_assignments(path: Path) -> Dict[str, Any]:
    """Read top-level constant assignments from a python problem file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    values: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return values


def get_problem_hardware_required(problem_path: Path) -> Optional[List[str]]:
    """Return HARDWARE_REQUIRED list if present, else None."""
    raw = _read_module_assignments(problem_path).get("HARDWARE_REQUIRED")
    if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
        return raw
    return None


def _problem_level(problem_path: Path, default_level: int) -> int:
    assignments = _read_module_assignments(problem_path)
    for key in ("GRAPHICS_LEVEL", "SPECIALIZED_LEVEL", "CUTILE_LEVEL"):
        value = assignments.get(key)
        if isinstance(value, int):
            return value
    return default_level


def _discover_dir(dir_path: Path, default_level: int, allowed_levels: set[int]) -> List[Tuple[int, Path]]:
    problems: List[Tuple[int, Path]] = []
    if not dir_path.exists():
        return problems

    for problem_file in sorted(dir_path.glob("*.py")):
        if problem_file.name.startswith("_"):
            continue
        level = _problem_level(problem_file, default_level=default_level)
        if level in allowed_levels:
            problems.append((level, problem_file))
    return problems


def find_problems_for_benchmark(project_root: Path, benchmark: str, levels: List[int]) -> List[Tuple[int, Path]]:
    """Find benchmark-specific problem files for selected logical levels."""
    config = BENCHMARK_PROBLEMS.get(benchmark)
    if config is None:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    kernelbench_dir = project_root / "KernelBench"
    requested = set(levels)
    problems: List[Tuple[int, Path]] = []

    for dirname in config["dirs"]:
        default_level = 1
        if dirname.startswith("level"):
            try:
                default_level = int(dirname.replace("level", ""))
            except ValueError:
                default_level = 1
        problems.extend(_discover_dir(kernelbench_dir / dirname, default_level=default_level, allowed_levels=requested))

    return problems
