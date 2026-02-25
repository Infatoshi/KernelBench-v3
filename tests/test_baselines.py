from __future__ import annotations

from pathlib import Path

import torch

from src.validate import run_local_validation


def test_cpu_baseline_smoke() -> None:
    """CI smoke test: baseline forward pass should work for representative CPU problems."""
    problems = [
        Path("problems/level1/23_Softmax.py"),
        Path("problems/level1/26_GELU_.py"),
        Path("problems/level1/40_LayerNorm.py"),
    ]

    results = run_local_validation(problems, device=torch.device("cpu"), platform="cpu-ci")
    failures = [result for result in results if not result.passed]
    assert not failures, "\n".join(f"{f.problem}: {f.error}" for f in failures)
