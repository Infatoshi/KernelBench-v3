#!/usr/bin/env python3
"""Precompute local kernel payloads for raw benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_runner import expand_run_matrix, load_yaml_config, yaml_to_benchmark_config
from raw.runner import (
    GenerationConfig as RawGenerationPlan,
    load_reference,
    local_candidate_json_path,
    resolve_problems,
    PROJECT_ROOT,
)


def _write_payload(path: Path, payload: Dict[str, object], reuse_existing: bool) -> bool:
    if reuse_existing and path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return True


def _generate_for_config(config, base_dir: Path, reuse_existing: bool) -> int:
    total_written = 0
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    for level in getattr(config.problems, "levels", [1]):
        plan = RawGenerationPlan(
            level=level,
            problem_ids=getattr(config.problems, "problem_ids", None),
            max_problems=getattr(config.problems, "max_problems", 100),
        )
        for problem_id in resolve_problems(plan):
            ref_src, problem_name = load_reference(level, problem_id)
            payload = {
                "level": level,
                "problem_id": problem_id,
                "problem_name": problem_name,
                "generated_code": ref_src,
                "raw_completion": ref_src,
                "formatted_completion": None,
                "prompt": None,
                "metadata": {
                    "source": "reference",
                    "generated_at": timestamp,
                },
            }
            target_path = local_candidate_json_path(base_dir, config, level, problem_id)
            if _write_payload(target_path, payload, reuse_existing):
                total_written += 1
    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute local kernels for raw benchmark runs.")
    parser.add_argument("config", type=str, help="Path to a KernelBench YAML configuration file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit directory for generated kernels (defaults to tmp/local_kernels/<timestamp>).",
    )
    args = parser.parse_args()

    yaml_path = Path(args.config)
    yaml_data = load_yaml_config(yaml_path)
    run_plan = expand_run_matrix(yaml_data)

    if args.output_dir:
        base_dir = Path(args.output_dir)
        if not base_dir.is_absolute():
            base_dir = PROJECT_ROOT / base_dir
    else:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        base_dir = PROJECT_ROOT / "tmp" / "local_kernels" / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    reuse_existing_default = True
    total_written = 0
    total_written = 0
    for mode, language, model_entry in run_plan:
        if str(mode).lower() != "raw":
            continue
        config = yaml_to_benchmark_config(yaml_data, model_entry, mode, language, cli_overrides={})
        reuse_existing = getattr(getattr(config, "generation", None), "reuse_existing", reuse_existing_default)
        written = _generate_for_config(config, base_dir, reuse_existing)
        total_written += written

    print(f"[local-generation] Wrote {total_written} kernel payload(s) for {yaml_path} into {base_dir}.")
    print(f"LOCAL_KERNEL_DIR={base_dir}")


if __name__ == "__main__":
    main()
