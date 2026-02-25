"""EvalResult dataclass and metric helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    model: str
    gpu: str
    problem: str
    level: int
    compiled: bool = False
    correct: bool = False
    speedup: Optional[float] = None
    ref_ms: Optional[float] = None
    sol_ms: Optional[float] = None
    ref_mean_ms: Optional[float] = None
    sol_mean_ms: Optional[float] = None
    ref_std_ms: Optional[float] = None
    sol_std_ms: Optional[float] = None
    ref_p10_ms: Optional[float] = None
    ref_p90_ms: Optional[float] = None
    sol_p10_ms: Optional[float] = None
    sol_p90_ms: Optional[float] = None
    turns: int = 0
    submitted: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    solution_code: Optional[str] = None
    solution_path: Optional[str] = None
    solution_hash: Optional[str] = None
    ref_kernels: Optional[int] = None
    sol_kernels: Optional[int] = None
    correctness_seeds: Optional[List[int]] = None
    benchmark_seed: Optional[int] = None
    baseline_type: Optional[str] = None
    precision: Optional[str] = None
    precision_used: Optional[str] = None
    valid_precisions: Optional[List[str]] = None
    precision_supported: Optional[bool] = None
    tolerance_atol: float = 0.05
    tolerance_rtol: float = 0.02
    has_nan: bool = False
    has_inf: bool = False
    is_deterministic: bool = True
    achieved_tflops: Optional[float] = None
    ref_tflops: Optional[float] = None
    pct_of_peak: Optional[float] = None
    ref_pct_of_peak: Optional[float] = None


def attach_solution_metadata(result: EvalResult, solution_path: Optional[str], sandbox) -> None:
    """Load solution code and attach hash/path metadata to result."""
    if not solution_path:
        return
    sol_path = solution_path if solution_path.startswith("/") else f"/workspace/{solution_path}"
    solution_code = sandbox.read_file(sol_path.replace("/workspace/", ""))
    result.solution_code = solution_code
    result.solution_path = solution_path
    if solution_code is not None:
        result.solution_hash = hashlib.sha256(solution_code.encode("utf-8")).hexdigest()[:16]


def apply_benchmark_metrics(result: EvalResult, benchmark_result: Dict[str, Any]) -> None:
    """Populate EvalResult fields from benchmark output."""
    result.compiled = benchmark_result.get("compiled", False)
    result.correct = benchmark_result.get("correct", False)
    result.speedup = benchmark_result.get("speedup")
    result.ref_ms = benchmark_result.get("ref_ms")
    result.sol_ms = benchmark_result.get("sol_ms")
    result.ref_mean_ms = benchmark_result.get("ref_mean_ms")
    result.sol_mean_ms = benchmark_result.get("sol_mean_ms")
    result.ref_std_ms = benchmark_result.get("ref_std_ms")
    result.sol_std_ms = benchmark_result.get("sol_std_ms")
    result.ref_p10_ms = benchmark_result.get("ref_p10_ms")
    result.ref_p90_ms = benchmark_result.get("ref_p90_ms")
    result.sol_p10_ms = benchmark_result.get("sol_p10_ms")
    result.sol_p90_ms = benchmark_result.get("sol_p90_ms")
    result.ref_kernels = benchmark_result.get("ref_kernels")
    result.sol_kernels = benchmark_result.get("sol_kernels")
    result.correctness_seeds = benchmark_result.get("correctness_seeds")
    result.benchmark_seed = benchmark_result.get("benchmark_seed")
    result.baseline_type = benchmark_result.get("baseline_type")
    result.precision = benchmark_result.get("precision") or benchmark_result.get("precision_used")
    result.precision_used = benchmark_result.get("precision_used") or benchmark_result.get("precision")
    result.valid_precisions = benchmark_result.get("valid_precisions")
    result.precision_supported = benchmark_result.get("precision_supported")
    result.tolerance_atol = benchmark_result.get("tolerance_atol", result.tolerance_atol)
    result.tolerance_rtol = benchmark_result.get("tolerance_rtol", result.tolerance_rtol)
    result.has_nan = benchmark_result.get("has_nan", False)
    result.has_inf = benchmark_result.get("has_inf", False)
    result.is_deterministic = benchmark_result.get("is_deterministic", True)
    result.achieved_tflops = benchmark_result.get("achieved_tflops")
    result.ref_tflops = benchmark_result.get("ref_tflops")
    result.pct_of_peak = benchmark_result.get("pct_of_peak")
    result.ref_pct_of_peak = benchmark_result.get("ref_pct_of_peak")
    if benchmark_result.get("error"):
        result.error = benchmark_result["error"]
