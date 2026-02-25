"""Batch evaluation orchestrator for KernelBench benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from src.backends import Backend, get_backend
from src.config.benchmark_problems import get_problem_hardware_required
from src.eval.agent import run_eval
from src.eval.results import EvalResult
from src.models import get_model_config


def get_all_tasks(
    backend: Backend,
    project_root: Path,
    models: List[str],
    gpus: List[str],
    levels: List[int],
    problems_per_level: Optional[int] = None,
) -> List[Tuple[str, str, int, Path]]:
    """Generate all (model, gpu, level, problem_path) combinations."""
    problems = backend.find_problems(project_root, levels)
    tasks = []
    for model_key in models:
        for gpu in gpus:
            gpu_compatible = []
            for level, problem_path in problems:
                required_hardware = get_problem_hardware_required(problem_path)
                if backend.name == "metal":
                    required_hardware = None
                if required_hardware and gpu not in required_hardware:
                    continue
                gpu_compatible.append((level, problem_path))

            if problems_per_level is not None:
                filtered = []
                level_counts: dict = {}
                for level, problem_path in gpu_compatible:
                    level_counts[level] = level_counts.get(level, 0) + 1
                    if level_counts[level] <= problems_per_level:
                        filtered.append((level, problem_path))
                gpu_compatible = filtered

            for level, problem_path in gpu_compatible:
                tasks.append((model_key, gpu, level, problem_path))
    return tasks


def load_completed(run_dir: Path) -> set:
    completed = set()
    results_file = run_dir / "results.jsonl"
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    task_id = f"{result['model']}_{result['gpu']}_{result['problem']}"
                    completed.add(task_id)
                except Exception:
                    pass
    return completed


def save_kernel(run_dir: Path, result: EvalResult, problem_path: Path) -> None:
    if not result.solution_code:
        return
    model_safe = result.model.replace(" ", "_").replace("/", "-")
    problem_safe = problem_path.stem
    kernel_dir = run_dir / "kernels" / f"{model_safe}_{result.gpu}_{problem_safe}"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    with open(problem_path) as f:
        problem_code = f.read()
    with open(kernel_dir / "problem.py", "w") as f:
        f.write(problem_code)
    with open(kernel_dir / "solution.py", "w") as f:
        f.write(result.solution_code)
    metadata = {
        "model": result.model, "gpu": result.gpu,
        "problem": result.problem, "level": result.level,
        "compiled": result.compiled, "correct": result.correct,
        "speedup": result.speedup, "ref_ms": result.ref_ms,
        "sol_ms": result.sol_ms, "ref_kernels": result.ref_kernels,
        "sol_kernels": result.sol_kernels,
    }
    with open(kernel_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def run_single_eval(
    backend_name: str,
    model_key: str,
    gpu: str,
    level: int,
    problem_path: Path,
    max_turns: Optional[int] = None,
    turn_artifact_dir: Optional[Path] = None,
) -> EvalResult:
    """Worker function for batch evaluation."""
    backend = get_backend(backend_name)
    model_config = get_model_config(model_key)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_key}")

    if max_turns is None:
        max_turns = backend.max_turns(level)

    with open(problem_path) as f:
        problem_code = f.read()

    print(f"[START] {model_config.name} | {gpu} | {problem_path.name}", flush=True)

    prev_turn_dir = os.environ.get("KB_TURN_ARTIFACT_DIR")
    try:
        if turn_artifact_dir is not None:
            turn_artifact_dir.mkdir(parents=True, exist_ok=True)
            os.environ["KB_TURN_ARTIFACT_DIR"] = str(turn_artifact_dir)
        elif "KB_TURN_ARTIFACT_DIR" in os.environ:
            del os.environ["KB_TURN_ARTIFACT_DIR"]

        result = run_eval(
            backend=backend,
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_path.name,
            level=level,
            max_turns=max_turns,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = EvalResult(model=model_config.name, gpu=gpu, problem=problem_path.name, level=level, error=str(e))
    finally:
        if prev_turn_dir is None:
            os.environ.pop("KB_TURN_ARTIFACT_DIR", None)
        else:
            os.environ["KB_TURN_ARTIFACT_DIR"] = prev_turn_dir

    if turn_artifact_dir is not None and result.solution_code:
        turn_artifact_dir.mkdir(parents=True, exist_ok=True)
        final_solution = turn_artifact_dir / "final_solution.py"
        with open(final_solution, "w", encoding="utf-8") as f:
            f.write(result.solution_code)
        result.solution_path = str(Path("turns") / turn_artifact_dir.name / "final_solution.py")
        if not result.solution_hash:
            result.solution_hash = hashlib.sha256(result.solution_code.encode("utf-8")).hexdigest()[:16]

    status = "OK" if result.correct else ("FAIL" if result.compiled else "ERR")
    speedup = f"{result.speedup:.2f}x" if result.speedup else "N/A"
    kernels = f"k:{result.ref_kernels}->{result.sol_kernels}" if result.ref_kernels is not None else ""
    print(f"[{status}] {model_config.name} | {gpu} | {problem_path.name} | {speedup} {kernels}", flush=True)
    return result


def run_batch_sequential(
    backend_name: str,
    tasks: List[Tuple[str, str, int, Path]],
    run_dir: Path,
    completed: set,
    max_turns: Optional[int] = None,
) -> None:
    backend = get_backend(backend_name)
    results_file = run_dir / "results.jsonl"
    for i, (model_key, gpu, level, problem_path) in enumerate(tasks):
        model_config = get_model_config(model_key)
        task_id = f"{model_config.name}_{gpu}_{problem_path.name}"
        if task_id in completed:
            print(f"[SKIP] {task_id} (already completed)")
            continue
        effective_turns = max_turns if max_turns is not None else backend.max_turns(level)
        print(f"\n[{i+1}/{len(tasks)}] {task_id} (max {effective_turns} turns)")
        turn_artifact_dir = run_dir / "turns" / task_id.replace("/", "-").replace(" ", "_")
        result = run_single_eval(backend_name, model_key, gpu, level, problem_path, max_turns, turn_artifact_dir)
        save_kernel(run_dir, result, problem_path)
        result_dict = asdict(result)
        result_dict.pop("solution_code", None)
        with open(results_file, "a") as f:
            f.write(json.dumps(result_dict) + "\n")


def run_batch_parallel(
    backend_name: str,
    tasks: List[Tuple[str, str, int, Path]],
    run_dir: Path,
    completed: set,
    max_turns: Optional[int] = None,
    max_workers: int = 4,
) -> None:
    backend = get_backend(backend_name)
    results_file = run_dir / "results.jsonl"

    pending = []
    for task in tasks:
        model_key, gpu, level, problem_path = task
        model_config = get_model_config(model_key)
        task_id = f"{model_config.name}_{gpu}_{problem_path.name}"
        if task_id not in completed:
            pending.append(task)

    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Max workers: {max_workers}")
    turn_limits = {lv: backend.max_turns(lv) for lv in range(1, 5)}
    if max_turns is not None:
        print(f"Max turns override: {max_turns}")
    else:
        print(f"Per-level turns: L1={turn_limits[1]}, L2={turn_limits[2]}, L3={turn_limits[3]}, L4={turn_limits[4]}")
    print()

    if not pending:
        print("All tasks completed!")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task in pending:
            model_key, gpu, level, problem_path = task
            model_config = get_model_config(model_key)
            task_id = f"{model_config.name}_{gpu}_{problem_path.name}" if model_config else f"{model_key}_{gpu}_{problem_path.name}"
            turn_artifact_dir = run_dir / "turns" / task_id.replace("/", "-").replace(" ", "_")
            future = executor.submit(run_single_eval, backend_name, model_key, gpu, level, problem_path, max_turns, turn_artifact_dir)
            futures[future] = task

        completed_count = len(completed)
        for future in as_completed(futures):
            task = futures[future]
            model_key, gpu, level, problem_path = task
            try:
                result = future.result()
            except Exception as e:
                model_config = get_model_config(model_key)
                result = EvalResult(model=model_config.name if model_config else model_key, gpu=gpu, problem=problem_path.name, level=level, error=str(e))

            save_kernel(run_dir, result, problem_path)
            result_dict = asdict(result)
            result_dict.pop("solution_code", None)
            with open(results_file, "a") as f:
                f.write(json.dumps(result_dict) + "\n")
            completed_count += 1
            print(f"Progress: {completed_count}/{len(tasks)} ({100*completed_count/len(tasks):.1f}%)")


def aggregate_results(run_dir: Path) -> dict:
    results_file = run_dir / "results.jsonl"
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    by_model: dict = {}
    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = {"total": 0, "compiled": 0, "correct": 0, "speedups": [], "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_model[model]["total"] += 1
        if r["compiled"]:
            by_model[model]["compiled"] += 1
        if r["correct"]:
            by_model[model]["correct"] += 1
        if r.get("speedup"):
            by_model[model]["speedups"].append(r["speedup"])
        input_toks = r.get("input_tokens", 0)
        output_toks = r.get("output_tokens", 0)
        cost = r.get("estimated_cost_usd", 0) or 0
        by_model[model]["input_tokens"] += input_toks
        by_model[model]["output_tokens"] += output_toks
        by_model[model]["cost_usd"] += cost
        total_input_tokens += input_toks
        total_output_tokens += output_toks
        total_cost += cost

    for model, stats in by_model.items():
        if stats["speedups"]:
            stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"])
            stats["max_speedup"] = max(stats["speedups"])
        else:
            stats["avg_speedup"] = None
            stats["max_speedup"] = None
        del stats["speedups"]
        stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

    by_gpu: dict = {}
    for r in results:
        gpu = r["gpu"]
        if gpu not in by_gpu:
            by_gpu[gpu] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        by_gpu[gpu]["total"] += 1
        if r["compiled"]:
            by_gpu[gpu]["compiled"] += 1
        if r["correct"]:
            by_gpu[gpu]["correct"] += 1
        if r.get("speedup"):
            by_gpu[gpu]["speedups"].append(r["speedup"])
    for gpu, stats in by_gpu.items():
        stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"]) if stats["speedups"] else None
        del stats["speedups"]

    by_level: dict = {}
    for r in results:
        level = r["level"]
        if level not in by_level:
            by_level[level] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        by_level[level]["total"] += 1
        if r["compiled"]:
            by_level[level]["compiled"] += 1
        if r["correct"]:
            by_level[level]["correct"] += 1
        if r.get("speedup"):
            by_level[level]["speedups"].append(r["speedup"])
    for level, stats in by_level.items():
        stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"]) if stats["speedups"] else None
        del stats["speedups"]

    return {
        "total_runs": len(results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost_usd": round(total_cost, 2),
        "by_model": by_model, "by_gpu": by_gpu, "by_level": by_level,
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\nTotal runs: {summary['total_runs']}")
    if summary.get("total_tokens"):
        print(f"Total tokens: {summary['total_input_tokens']:,} in / {summary['total_output_tokens']:,} out / {summary['total_tokens']:,} total")
    if summary.get("total_cost_usd"):
        print(f"Total cost: ${summary['total_cost_usd']:.2f}")

    print("\n--- BY MODEL ---")
    print(f"{'Model':<25} {'Total':>6} {'Compiled':>8} {'Correct':>8} {'Speedup':>10} {'Tokens':>12} {'Cost':>10}")
    print("-" * 90)
    for model, stats in sorted(summary["by_model"].items()):
        speedup = f"{stats['avg_speedup']:.2f}x" if stats.get("avg_speedup") else "N/A"
        tokens = f"{stats.get('total_tokens', 0):,}" if stats.get("total_tokens") else "N/A"
        cost = f"${stats.get('cost_usd', 0):.2f}" if stats.get("cost_usd") else "N/A"
        print(f"{model:<25} {stats['total']:>6} {stats['compiled']:>8} {stats['correct']:>8} {speedup:>10} {tokens:>12} {cost:>10}")

    print("\n--- BY GPU ---")
    print(f"{'GPU':<15} {'Total':>8} {'Compiled':>10} {'Correct':>10} {'Avg Speedup':>12}")
    print("-" * 55)
    for gpu, stats in sorted(summary["by_gpu"].items()):
        speedup = f"{stats['avg_speedup']:.2f}x" if stats["avg_speedup"] else "N/A"
        print(f"{gpu:<15} {stats['total']:>8} {stats['compiled']:>10} {stats['correct']:>10} {speedup:>12}")

    print("\n--- BY LEVEL ---")
    print(f"{'Level':>8} {'Total':>8} {'Compiled':>10} {'Correct':>10} {'Avg Speedup':>12}")
    print("-" * 50)
    for level, stats in sorted(summary["by_level"].items()):
        speedup = f"{stats['avg_speedup']:.2f}x" if stats["avg_speedup"] else "N/A"
        print(f"{level:>8} {stats['total']:>8} {stats['compiled']:>10} {stats['correct']:>10} {speedup:>12}")
