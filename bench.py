#!/usr/bin/env python3
"""KernelBench v3 -- GPU kernel optimization benchmark.

Single entry point for all benchmark backends.

Usage:
    uv run python bench.py batch cuda --models deepseek/deepseek-v3.2 --gpus H100,B200 --levels 1,2,3,4
    uv run python bench.py eval cuda --model deepseek/deepseek-v3.2 --gpu H100 --level 1 --problem 1_Square_matrix_multiplication_.py
    uv run python bench.py list-models
    uv run python bench.py list-problems --backend cuda
    uv run python bench.py summary outputs/batch_eval/run_XXX
    uv run python bench.py validate --platforms cpu --max-problems 5
    uv run python bench.py export
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_batch(args):
    """Run batch evaluation across models/GPUs/problems."""
    from src.backends import get_backend
    from src.batch import (
        aggregate_results,
        get_all_tasks,
        load_completed,
        print_summary,
        run_batch_parallel,
        run_batch_sequential,
    )
    from src.config.runtime_validation import validate_gpus as validate_allowed_gpus, validate_platform
    from src.models import MODELS, get_model_config

    backend = get_backend(args.backend)

    if args.models == "all":
        models = list(MODELS.keys())
    else:
        models = args.models.split(",")

    gpus = args.gpus.split(",") if args.gpus else backend.allowed_gpus
    levels = [int(x) for x in args.levels.split(",")]

    for m in models:
        if get_model_config(m) is None:
            print(f"Unknown model: {m}")
            print("Use 'bench.py list-models' to see available models.")
            return

    validate_allowed_gpus(gpus, backend.allowed_gpus, backend.benchmark_name, backend.gpu_reason)
    validate_platform(gpus, backend.benchmark_name)

    tasks = get_all_tasks(backend, PROJECT_ROOT, models, gpus, levels, args.problems_per_level)

    if args.dry_run:
        print(f"Backend: {backend.benchmark_name}")
        print(f"Models: {models}")
        print(f"GPUs: {gpus}")
        print(f"Levels: {levels}")
        print(f"Total tasks: {len(tasks)}")
        print("\nFirst 10 tasks:")
        for model_key, gpu, level, problem_path in tasks[:10]:
            mc = get_model_config(model_key)
            print(f"  {mc.name} | {gpu} | L{level} | {problem_path.name}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

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
    print(f"KERNELBENCH BATCH EVALUATION -- {backend.benchmark_name}")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Models: {models}")
    print(f"GPUs: {gpus}")
    print(f"Levels: {levels}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    completed = load_completed(run_dir)

    config = {
        "backend": args.backend,
        "models": models, "gpus": gpus, "levels": levels,
        "max_turns_override": args.max_turns,
        "workers": args.workers,
        "started": datetime.now().isoformat(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    start_time = time.time()
    if args.sequential:
        run_batch_sequential(args.backend, tasks, run_dir, completed, args.max_turns)
    else:
        run_batch_parallel(args.backend, tasks, run_dir, completed, args.max_turns, args.workers)

    elapsed = time.time() - start_time
    summary = aggregate_results(run_dir)
    summary["elapsed_seconds"] = elapsed
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
    print(f"Results saved to: {run_dir}")


def cmd_eval(args):
    """Run single problem evaluation."""
    from src.backends import get_backend
    from src.eval.agent import run_eval
    from src.models import get_model_config

    backend = get_backend(args.backend)
    model_config = get_model_config(args.model)
    if model_config is None:
        print(f"Unknown model: {args.model}")
        return

    problem_path = PROJECT_ROOT / "problems" / args.problem
    if not problem_path.exists():
        print(f"Problem not found: {problem_path}")
        return

    with open(problem_path) as f:
        problem_code = f.read()

    level = int(args.problem.split("/")[0].replace("level", "").replace("metal_level", "").replace("graphics", "1"))

    print("=" * 60)
    print(f"Backend: {backend.benchmark_name}")
    print(f"Model: {model_config.name}")
    print(f"GPU: {args.gpu}")
    print(f"Problem: {args.problem}")
    print("=" * 60)

    result = run_eval(
        backend=backend, model_config=model_config, gpu=args.gpu,
        problem_code=problem_code, problem_name=Path(args.problem).name,
        level=level, max_turns=args.max_turns or backend.max_turns(level),
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Compiled: {result.compiled}")
    print(f"Correct: {result.correct}")
    if result.speedup:
        print(f"Speedup: {result.speedup:.2f}x")
    print(f"Turns: {result.turns}")
    print(f"Time: {result.elapsed_seconds:.1f}s")
    if result.error:
        print(f"Error: {result.error}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"{args.model.replace('/', '-')}_{args.gpu}_{Path(args.problem).stem}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nResult saved: {result_path}")


def cmd_list_models(args):
    from src.models import MODELS
    print(f"\n{'Key':<40s} {'Name':<30s} {'Provider'}")
    print("-" * 85)
    for key, cfg in sorted(MODELS.items()):
        print(f"{key:<40s} {cfg.name:<30s} {cfg.provider}")
    print(f"\nTotal: {len(MODELS)} models")
    print("\nAny valid OpenRouter model ID also works dynamically.")


def cmd_list_problems(args):
    from src.backends import get_backend
    backend = get_backend(args.backend)
    levels = [int(x) for x in args.levels.split(",")]
    problems = backend.find_problems(PROJECT_ROOT, levels)
    print(f"Backend: {backend.benchmark_name}")
    print(f"Total problems: {len(problems)}")
    for level, path in problems:
        print(f"  L{level}: {path.name}")


def cmd_list_backends(args):
    from src.backends import BACKENDS
    print(f"\n{'Backend':<12s} {'Benchmark':<16s} {'Allowed GPUs'}")
    print("-" * 55)
    for name in sorted(BACKENDS):
        b = BACKENDS[name]
        gpus = ", ".join(b.allowed_gpus)
        print(f"{name:<12s} {b.benchmark_name:<16s} {gpus}")


def cmd_summary(args):
    from src.batch import aggregate_results, print_summary
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return
    summary = aggregate_results(run_dir)
    print_summary(summary)


def cmd_validate(args):
    from src.validate import main as validate_main
    sys.argv = ["validate"]
    if args.platforms:
        sys.argv.extend(["--platforms", args.platforms])
    if args.max_problems:
        sys.argv.extend(["--max-problems", str(args.max_problems)])
    validate_main()


def cmd_export(args):
    from src.export import main as export_main
    export_main()


def main():
    parser = argparse.ArgumentParser(
        description="KernelBench v3 -- GPU kernel optimization benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    p_batch = subparsers.add_parser("batch", help="Run batch evaluation")
    p_batch.add_argument("backend", choices=["cuda", "triton", "cutlass", "cute", "cutile", "graphics", "metal"])
    p_batch.add_argument("--models", required=True, help="Comma-separated model keys or 'all'")
    p_batch.add_argument("--gpus", help="Comma-separated GPU names (default: backend's allowed GPUs)")
    p_batch.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels")
    p_batch.add_argument("--problems-per-level", type=int, help="Limit problems per level")
    p_batch.add_argument("--max-turns", type=int, help="Override max turns per problem")
    p_batch.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p_batch.add_argument("--sequential", action="store_true", help="Run sequentially")
    p_batch.add_argument("--resume", help="Resume from existing run directory")
    p_batch.add_argument("--output-dir", default="outputs/batch_eval", help="Output directory")
    p_batch.add_argument("--dry-run", action="store_true", help="Show tasks without executing")
    p_batch.set_defaults(func=cmd_batch)

    p_eval = subparsers.add_parser("eval", help="Run single problem evaluation")
    p_eval.add_argument("backend", choices=["cuda", "triton", "cutlass", "cute", "cutile", "graphics", "metal"])
    p_eval.add_argument("--model", required=True, help="Model key")
    p_eval.add_argument("--gpu", required=True, help="GPU type")
    p_eval.add_argument("--problem", required=True, help="Problem path (e.g., level1/23_Softmax.py)")
    p_eval.add_argument("--max-turns", type=int, help="Max turns")
    p_eval.add_argument("--output-dir", default="outputs/eval", help="Output directory")
    p_eval.set_defaults(func=cmd_eval)

    p_lm = subparsers.add_parser("list-models", help="List available models")
    p_lm.set_defaults(func=cmd_list_models)

    p_lp = subparsers.add_parser("list-problems", help="List problems for a backend")
    p_lp.add_argument("--backend", default="cuda", choices=["cuda", "triton", "cutlass", "cute", "cutile", "graphics", "metal"])
    p_lp.add_argument("--levels", default="1,2,3,4")
    p_lp.set_defaults(func=cmd_list_problems)

    p_lb = subparsers.add_parser("list-backends", help="List available backends")
    p_lb.set_defaults(func=cmd_list_backends)

    p_sum = subparsers.add_parser("summary", help="Show summary for a completed run")
    p_sum.add_argument("run_dir", help="Path to run directory")
    p_sum.set_defaults(func=cmd_summary)

    p_val = subparsers.add_parser("validate", help="Validate problem baselines")
    p_val.add_argument("--platforms", default="cpu", help="Comma-separated: cpu,cuda,metal")
    p_val.add_argument("--max-problems", type=int, help="Limit problems per platform")
    p_val.set_defaults(func=cmd_validate)

    p_exp = subparsers.add_parser("export", help="Export results for website")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
