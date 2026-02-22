#!/usr/bin/env python3
"""Export KernelBench results for website with solution files."""

import json
import csv
import shutil
from pathlib import Path

# Paths
BATCH_EVAL_DIR = Path("outputs/batch_eval")
WEBSITE_DATA_DIR = Path.home() / "elliotarledge.com/public/data/kernelbench-v3"
SOLUTIONS_DIR = WEBSITE_DATA_DIR / "solutions"

def sanitize_for_dir(name: str) -> str:
    """Convert model name to directory format (spaces -> underscores, keep dots)."""
    return name.replace(" ", "_")

def sanitize_filename(name: str) -> str:
    """Make filename safe."""
    return name.replace(" ", "_").replace("/", "_").replace(".", "_")

def get_problem_num_name(problem: str) -> str:
    """Extract problem number and name from filename like '26_GELU_.py'."""
    return problem.replace(".py", "")

def main():
    # Create output directories
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # First pass: build index of all kernel directories
    kernel_index = {}  # (model_dir_name, gpu, problem_base) -> solution_path

    for run_dir in sorted(BATCH_EVAL_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        kernels_dir = run_dir / "kernels"
        if not kernels_dir.exists():
            continue

        for kernel_dir in kernels_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            solution_file = kernel_dir / "solution.py"
            if solution_file.exists():
                # Parse directory name: Model_Name_GPU_problem_name
                parts = kernel_dir.name.split("_")
                # Find GPU (H100 or B200)
                gpu_idx = None
                for i, p in enumerate(parts):
                    if p in ("H100", "B200"):
                        gpu_idx = i
                        break
                if gpu_idx is None:
                    continue

                model_parts = parts[:gpu_idx]
                gpu = parts[gpu_idx]
                problem_parts = parts[gpu_idx + 1:]

                model_dir = "_".join(model_parts)
                problem_base = "_".join(problem_parts)

                key = (model_dir, gpu, problem_base)
                kernel_index[key] = solution_file

    print(f"Indexed {len(kernel_index)} kernel directories with solutions")

    # Second pass: collect all results
    all_results = []

    for run_dir in sorted(BATCH_EVAL_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    result["_run_dir"] = run_dir.name
                    all_results.append(result)

    print(f"Found {len(all_results)} total results across all runs")

    # Deduplicate - keep latest run for each (model, gpu, problem)
    seen = {}
    for r in all_results:
        key = (r.get("model"), r.get("gpu"), r.get("problem"))
        run_dir = r.get("_run_dir", "")
        if key not in seen or run_dir > seen[key].get("_run_dir", ""):
            seen[key] = r

    results = list(seen.values())
    print(f"After dedup: {len(results)} unique (model, gpu, problem) combinations")

    # Build CSV and copy solution files
    csv_rows = []
    solutions_copied = 0

    for r in results:
        model = r.get("model", "unknown")
        gpu = r.get("gpu", "unknown")
        problem = r.get("problem", "unknown")
        problem_base = get_problem_num_name(problem)

        # Find solution file
        model_dir = sanitize_for_dir(model)
        key = (model_dir, gpu, problem_base)

        solution_link = ""
        is_passed = r.get("passed") or (r.get("compiled") and r.get("correct"))
        if key in kernel_index and is_passed:
            src = kernel_index[key]
            # Naming: model_gpu_problem.txt
            dest_name = f"{sanitize_filename(model)}_{gpu}_{sanitize_filename(problem_base)}.txt"
            dest = SOLUTIONS_DIR / dest_name
            shutil.copy(src, dest)
            solution_link = f"/data/kernelbench-v3/solutions/{dest_name}"
            solutions_copied += 1

        # Baseline link to GitHub repo
        level = r.get("level", 0)
        baseline_link = f"https://github.com/Infatoshi/KernelBench-v3/blob/master/KernelBench/level{level}/{problem}"

        csv_rows.append({
            "model": model,
            "gpu": gpu,
            "level": level,
            "problem": problem,
            "problem_name": r.get("problem_name", ""),
            "problem_category": r.get("problem_category", ""),
            "provider": r.get("provider", ""),
            "model_tier": r.get("model_tier", ""),
            "compiled": r.get("compiled", False),
            "correct": r.get("correct", False),
            "passed": r.get("passed") or (r.get("compiled") and r.get("correct")),
            "speedup": r.get("speedup") if is_passed else None,
            "beats_baseline": r.get("beats_baseline") or (is_passed and r.get("speedup", 0) and r.get("speedup", 0) > 1.0),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "total_tokens": r.get("total_tokens", 0),
            "turns": r.get("turns", 0),
            "estimated_cost_usd": r.get("estimated_cost_usd"),
            "baseline_link": baseline_link,
            "solution_link": solution_link,
        })

    print(f"Copied {solutions_copied} solution files to {SOLUTIONS_DIR}")

    # Sort by model, gpu, level, problem
    csv_rows.sort(key=lambda x: (x["model"], x["gpu"], x["level"], x["problem"]))

    # Write CSV
    csv_path = WEBSITE_DATA_DIR / "results.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {len(csv_rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
