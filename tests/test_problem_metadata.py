from __future__ import annotations

import ast
from pathlib import Path


REQUIRED_FIELDS = ("OP_TYPE", "SUPPORTED_PRECISIONS", "HARDWARE_REQUIRED")


def _top_level_assignments(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def test_all_problems_have_required_metadata() -> None:
    problems = sorted(Path("KernelBench").rglob("*.py"))
    missing: list[str] = []

    for problem in problems:
        if problem.name.startswith("_"):
            continue
        assignments = _top_level_assignments(problem)
        for field in REQUIRED_FIELDS:
            if field not in assignments:
                missing.append(f"{problem}: missing {field}")

    assert not missing, "Problems missing metadata:\\n" + "\\n".join(missing)
