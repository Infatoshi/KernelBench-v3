#!/usr/bin/env python3
"""
Batch Evaluation Runner for KernelBench

Runs all models across all GPUs on all problems in parallel.
Handles resumption, progress tracking, and result aggregation.

Usage:
    # Full evaluation (9 models x 4 GPUs x 86 problems = 3096 runs)
    uv run python batch_eval.py --all

    # Subset for testing
    uv run python batch_eval.py --models claude-opus-4.5 --gpus H100 --levels 4

    # Resume interrupted run
    uv run python batch_eval.py --resume outputs/batch_eval/run_20260103_123456
"""

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modal_eval import (
    MODELS, ModelConfig, EvalResult, run_agent_on_modal,
    GPU_SPECS, find_problems
)

# GPU types available on Modal
GPUS = ["L40S", "A100", "H100", "B200"]

# Concurrency limits
MAX_CONCURRENT_MODAL = 8  # Modal sandbox limit
MAX_CONCURRENT_API = 4    # API rate limit per provider

# Per-level turn limits (lower for simpler problems, saves tokens/cost)
TURN_LIMITS = {
    1: 10,  # Simple operators
    2: 12,  # Fused operations
    3: 15,  # Model architectures
    4: 15,  # Real models
}


def get_all_tasks(
    models: List[str],
    gpus: List[str],
    levels: List[int],
    problems_per_level: Optional[int] = None
) -> List[Tuple[str, str, int, Path]]:
    """Generate all (model, gpu, level, problem_path) combinations."""
    problems = find_problems(levels)

    # Limit problems per level if specified
    if problems_per_level is not None:
        limited = []
        level_counts = {}
        for level, problem_path in problems:
            level_counts[level] = level_counts.get(level, 0) + 1
            if level_counts[level] <= problems_per_level:
                limited.append((level, problem_path))
        problems = limited

    tasks = []
    for model_key in models:
        for gpu in gpus:
            for level, problem_path in problems:
                tasks.append((model_key, gpu, level, problem_path))

    return tasks


def load_completed(run_dir: Path) -> set:
    """Load completed task IDs from results file."""
    completed = set()
    results_file = run_dir / "results.jsonl"

    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    task_id = f"{result['model']}_{result['gpu']}_{result['problem']}"
                    completed.add(task_id)
                except:
                    pass

    return completed


def save_kernel(
    run_dir: Path,
    result: EvalResult,
    problem_path: Path
):
    """Save submitted kernel and problem to kernels directory."""
    if not result.solution_code:
        return  # No solution to save

    # Create kernel directory: kernels/Model_GPU_Problem/
    # Sanitize names for filesystem
    model_safe = result.model.replace(" ", "_").replace("/", "-")
    problem_safe = problem_path.stem  # Remove .py extension
    kernel_dir = run_dir / "kernels" / f"{model_safe}_{result.gpu}_{problem_safe}"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Save the original problem
    with open(problem_path) as f:
        problem_code = f.read()
    with open(kernel_dir / "problem.py", "w") as f:
        f.write(problem_code)

    # Save the submitted solution
    with open(kernel_dir / "solution.py", "w") as f:
        f.write(result.solution_code)

    # Save metadata
    metadata = {
        "model": result.model,
        "gpu": result.gpu,
        "problem": result.problem,
        "level": result.level,
        "compiled": result.compiled,
        "correct": result.correct,
        "speedup": result.speedup,
        "ref_ms": result.ref_ms,
        "sol_ms": result.sol_ms,
        "ref_kernels": result.ref_kernels,
        "sol_kernels": result.sol_kernels,
    }
    with open(kernel_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def run_single_eval(
    model_key: str,
    gpu: str,
    level: int,
    problem_path: Path,
    max_turns: Optional[int] = None
) -> EvalResult:
    """Run single evaluation (worker function)."""
    model_config = MODELS[model_key]

    # Use level-specific turn limit if not overridden
    if max_turns is None:
        max_turns = TURN_LIMITS.get(level, 15)

    with open(problem_path) as f:
        problem_code = f.read()

    print(f"[START] {model_config.name} | {gpu} | {problem_path.name}", flush=True)

    try:
        result = run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_path.name,
            level=level,
            max_turns=max_turns
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = EvalResult(
            model=model_config.name,
            gpu=gpu,
            problem=problem_path.name,
            level=level,
            error=str(e)
        )

    status = "OK" if result.correct else ("FAIL" if result.compiled else "ERR")
    speedup = f"{result.speedup:.2f}x" if result.speedup else "N/A"
    kernels = f"k:{result.ref_kernels}->{result.sol_kernels}" if result.ref_kernels is not None else ""
    print(f"[{status}] {model_config.name} | {gpu} | {problem_path.name} | {speedup} {kernels}", flush=True)

    return result


def run_batch_sequential(
    tasks: List[Tuple[str, str, int, Path]],
    run_dir: Path,
    completed: set,
    max_turns: Optional[int] = None
):
    """Run evaluations sequentially (for debugging)."""
    results_file = run_dir / "results.jsonl"

    for i, (model_key, gpu, level, problem_path) in enumerate(tasks):
        task_id = f"{MODELS[model_key].name}_{gpu}_{problem_path.name}"

        if task_id in completed:
            print(f"[SKIP] {task_id} (already completed)")
            continue

        # Use level-specific turns unless overridden
        effective_turns = max_turns if max_turns is not None else TURN_LIMITS.get(level, 15)
        print(f"\n[{i+1}/{len(tasks)}] {task_id} (max {effective_turns} turns)")

        result = run_single_eval(model_key, gpu, level, problem_path, max_turns)

        # Save kernel if submitted
        save_kernel(run_dir, result, problem_path)

        # Append result (exclude solution_code from JSONL to keep it small)
        result_dict = asdict(result)
        result_dict.pop("solution_code", None)
        with open(results_file, "a") as f:
            f.write(json.dumps(result_dict) + "\n")


def run_batch_parallel(
    tasks: List[Tuple[str, str, int, Path]],
    run_dir: Path,
    completed: set,
    max_turns: Optional[int] = None,
    max_workers: int = 4
):
    """Run evaluations in parallel using ProcessPoolExecutor."""
    results_file = run_dir / "results.jsonl"

    # Filter completed
    pending = []
    for task in tasks:
        model_key, gpu, level, problem_path = task
        task_id = f"{MODELS[model_key].name}_{gpu}_{problem_path.name}"
        if task_id not in completed:
            pending.append(task)

    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Max workers: {max_workers}")
    if max_turns is not None:
        print(f"Max turns override: {max_turns}")
    else:
        print(f"Per-level turns: L1={TURN_LIMITS[1]}, L2={TURN_LIMITS[2]}, L3={TURN_LIMITS[3]}, L4={TURN_LIMITS[4]}")
    print()

    if not pending:
        print("All tasks completed!")
        return

    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for task in pending:
            model_key, gpu, level, problem_path = task
            future = executor.submit(
                run_single_eval, model_key, gpu, level, problem_path, max_turns
            )
            futures[future] = task

        completed_count = len(completed)
        for future in as_completed(futures):
            task = futures[future]
            model_key, gpu, level, problem_path = task

            try:
                result = future.result()
            except Exception as e:
                result = EvalResult(
                    model=MODELS[model_key].name,
                    gpu=gpu,
                    problem=problem_path.name,
                    level=level,
                    error=str(e)
                )

            # Save kernel if submitted
            save_kernel(run_dir, result, problem_path)

            # Append result (exclude solution_code from JSONL to keep it small)
            result_dict = asdict(result)
            result_dict.pop("solution_code", None)
            with open(results_file, "a") as f:
                f.write(json.dumps(result_dict) + "\n")

            completed_count += 1
            total = len(tasks)
            print(f"Progress: {completed_count}/{total} ({100*completed_count/total:.1f}%)")


def aggregate_results(run_dir: Path) -> dict:
    """Aggregate results into summary statistics."""
    results_file = run_dir / "results.jsonl"

    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    # Total token/cost tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    # Aggregate by model
    by_model = {}
    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = {
                "total": 0, "compiled": 0, "correct": 0, "speedups": [],
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }

        by_model[model]["total"] += 1
        if r["compiled"]:
            by_model[model]["compiled"] += 1
        if r["correct"]:
            by_model[model]["correct"] += 1
        if r.get("speedup"):
            by_model[model]["speedups"].append(r["speedup"])

        # Token tracking
        input_toks = r.get("input_tokens", 0)
        output_toks = r.get("output_tokens", 0)
        cost = r.get("estimated_cost_usd", 0) or 0
        by_model[model]["input_tokens"] += input_toks
        by_model[model]["output_tokens"] += output_toks
        by_model[model]["cost_usd"] += cost
        total_input_tokens += input_toks
        total_output_tokens += output_toks
        total_cost += cost

    # Compute averages
    for model, stats in by_model.items():
        if stats["speedups"]:
            stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"])
            stats["max_speedup"] = max(stats["speedups"])
        else:
            stats["avg_speedup"] = None
            stats["max_speedup"] = None
        del stats["speedups"]
        stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

    # Aggregate by GPU
    by_gpu = {}
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
        if stats["speedups"]:
            stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"])
        else:
            stats["avg_speedup"] = None
        del stats["speedups"]

    # Aggregate by level
    by_level = {}
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
        if stats["speedups"]:
            stats["avg_speedup"] = sum(stats["speedups"]) / len(stats["speedups"])
        else:
            stats["avg_speedup"] = None
        del stats["speedups"]

    return {
        "total_runs": len(results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost_usd": round(total_cost, 2),
        "by_model": by_model,
        "by_gpu": by_gpu,
        "by_level": by_level
    }


def print_summary(summary: dict):
    """Print summary in table format."""
    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)

    print(f"\nTotal runs: {summary['total_runs']}")

    # Token/cost summary if available
    if summary.get("total_tokens"):
        print(f"Total tokens: {summary['total_input_tokens']:,} in / {summary['total_output_tokens']:,} out / {summary['total_tokens']:,} total")
    if summary.get("total_cost_usd"):
        print(f"Total cost: ${summary['total_cost_usd']:.2f}")

    print("\n--- BY MODEL ---")
    print(f"{'Model':<25} {'Total':>6} {'Compiled':>8} {'Correct':>8} {'Speedup':>10} {'Tokens':>12} {'Cost':>10}")
    print("-" * 90)
    for model, stats in sorted(summary["by_model"].items()):
        speedup = f"{stats['avg_speedup']:.2f}x" if stats.get("avg_speedup") else "N/A"
        tokens = f"{stats.get('total_tokens', 0):,}" if stats.get('total_tokens') else "N/A"
        cost = f"${stats.get('cost_usd', 0):.2f}" if stats.get('cost_usd') else "N/A"
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


def main():
    parser = argparse.ArgumentParser(description="Batch KernelBench Evaluation")

    # Selection
    parser.add_argument("--models", type=str, help="Comma-separated model keys (or 'all')")
    parser.add_argument("--gpus", type=str, default="H100", help="Comma-separated GPU types (or 'all')")
    parser.add_argument("--levels", type=str, default="1,2,3,4", help="Comma-separated levels")
    parser.add_argument("--problems-per-level", type=int, default=None, help="Limit problems per level (for testing)")
    parser.add_argument("--all", action="store_true", help="Run all models on all GPUs")

    # Execution
    parser.add_argument("--max-turns", type=int, default=None, help="Override max turns (default: per-level L1=10, L2=12, L3=15, L4=15)")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (for debugging)")

    # Resume
    parser.add_argument("--resume", type=str, help="Resume from existing run directory")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/batch_eval", help="Output directory")
    parser.add_argument("--summary-only", type=str, help="Only print summary for existing run")

    # Info
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-problems", action="store_true", help="List all problems")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")

    args = parser.parse_args()

    # List models
    if args.list_models:
        print("Available models:")
        for key, cfg in MODELS.items():
            print(f"  {key}: {cfg.name} ({cfg.provider})")
        return

    # List problems
    if args.list_problems:
        problems = find_problems([1, 2, 3, 4])
        print(f"Total problems: {len(problems)}")
        for level, path in problems:
            print(f"  L{level}: {path.name}")
        return

    # Summary only
    if args.summary_only:
        run_dir = Path(args.summary_only)
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}")
            return
        summary = aggregate_results(run_dir)
        print_summary(summary)
        return

    # Parse selections
    if args.all:
        models = list(MODELS.keys())
        gpus = GPUS
    else:
        if not args.models:
            parser.print_help()
            print("\nError: --models required (or use --all)")
            return
        models = args.models.split(",") if args.models != "all" else list(MODELS.keys())
        gpus = args.gpus.split(",") if args.gpus != "all" else GPUS

    levels = [int(l) for l in args.levels.split(",")]

    # Validate
    for m in models:
        if m not in MODELS:
            print(f"Unknown model: {m}")
            print(f"Available: {list(MODELS.keys())}")
            return

    for g in gpus:
        if g not in GPUS:
            print(f"Unknown GPU: {g}")
            print(f"Available: {GPUS}")
            return

    # Generate tasks
    tasks = get_all_tasks(models, gpus, levels, args.problems_per_level)

    if args.dry_run:
        print(f"Models: {models}")
        print(f"GPUs: {gpus}")
        print(f"Levels: {levels}")
        print(f"Total tasks: {len(tasks)}")
        print("\nFirst 10 tasks:")
        for model_key, gpu, level, problem_path in tasks[:10]:
            print(f"  {MODELS[model_key].name} | {gpu} | L{level} | {problem_path.name}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    # Setup output directory
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"Resume directory not found: {run_dir}")
            return
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KERNELBENCH BATCH EVALUATION")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Models: {models}")
    print(f"GPUs: {gpus}")
    print(f"Levels: {levels}")
    print(f"Total tasks: {len(tasks)}")
    if args.max_turns is not None:
        print(f"Max turns: {args.max_turns} (override)")
    else:
        print(f"Max turns: L1={TURN_LIMITS[1]}, L2={TURN_LIMITS[2]}, L3={TURN_LIMITS[3]}, L4={TURN_LIMITS[4]}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    # Load completed
    completed = load_completed(run_dir)

    # Save config
    config = {
        "models": models,
        "gpus": gpus,
        "levels": levels,
        "max_turns_override": args.max_turns,
        "turn_limits": TURN_LIMITS if args.max_turns is None else None,
        "workers": args.workers,
        "started": datetime.now().isoformat()
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run
    start_time = time.time()

    if args.sequential:
        run_batch_sequential(tasks, run_dir, completed, args.max_turns)
    else:
        run_batch_parallel(tasks, run_dir, completed, args.max_turns, args.workers)

    elapsed = time.time() - start_time

    # Aggregate and print summary
    summary = aggregate_results(run_dir)
    summary["elapsed_seconds"] = elapsed

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
