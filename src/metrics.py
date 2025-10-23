"""Lightweight metric helpers for KernelBench outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_jsonl_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Parse JSONL results file into a list of dicts."""
    results: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        return results

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but continue processing the rest.
                continue
    return results


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def compute_core_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics from raw result records."""
    as_list = list(results)
    total = len(as_list)
    if total == 0:
        return {
            "total_problems": 0,
            "compilation_rate": 0.0,
            "correctness_rate": 0.0,
            "fast_1_rate": 0.0,
            "mean_runtime": 0.0,
            "correct_count": 0,
            "compiled_count": 0,
        }

    compiled = sum(1 for r in as_list if r.get("compiled"))
    correct = sum(1 for r in as_list if r.get("correctness"))

    runtimes = [r.get("runtime") for r in as_list if r.get("correctness") and r.get("runtime")]

    fast_1 = 0
    for record in as_list:
        if record.get("fast_1") is True:
            fast_1 += 1
        elif record.get("correctness") and (record.get("runtime") or 0) > 0:
            fast_1 += 1

    metrics = {
        "total_problems": total,
        "compilation_rate": _safe_rate(compiled, total),
        "correctness_rate": _safe_rate(correct, compiled) if compiled else 0.0,
        "fast_1_rate": _safe_rate(fast_1, total),
        "mean_runtime": (sum(runtimes) / len(runtimes)) if runtimes else 0.0,
        "correct_count": correct,
        "compiled_count": compiled,
    }

    return metrics


def load_metrics(payload_path: Path) -> Dict[str, Any] | None:
    if not payload_path.exists():
        return None
    try:
        with payload_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def save_metrics(payload_path: Path, payload: Dict[str, Any]) -> None:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
