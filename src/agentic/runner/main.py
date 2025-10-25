"""Agentic benchmark runner for KernelBench-v3."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, TYPE_CHECKING

from agentic.agents.OptimAgent import OptimAgent
from agentic.dataloaders.TritonBench import TritonBench
from providers import ProviderConfig, create_provider, resolve_provider_api_key

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from config import BenchmarkConfig
    from agentic.dataloaders.ProblemState import ProblemState


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRITONBENCH_ROOT = PROJECT_ROOT / "data" / "TritonBench"
TRITONBENCH_DATA_DIR = TRITONBENCH_ROOT / "data"
TRITONBENCH_PERF_G_DIR = TRITONBENCH_ROOT / "performance_metrics" / "perf_G"


def _find_existing_file(*relative_names: str) -> Path:
    for name in relative_names:
        candidate = TRITONBENCH_DATA_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required resource not found in any of: {relative_names}")


@dataclass(frozen=True)
class TritonBenchPaths:
    statis_path: Path
    py_folder: Path
    instruction_path: Path
    corpus_path: Path
    golden_metrics: Path
    perf_G_path: Path


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required resource not found: {path}")
    return path


def resolve_tritonbench_paths() -> TritonBenchPaths:
    statis_path = _find_existing_file(
        "TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json",
        "TritonBench_G_comp_alpac_v1.json",
        "TritonBench_G_v1.json",
    )
    instruction_path = _find_existing_file(
        "TritonBench_G_comp_alpac_v1.json",
        "TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json",
        "TritonBench_G_v1.json",
    )

    return TritonBenchPaths(
        statis_path=statis_path,
        py_folder=ensure_exists(TRITONBENCH_DATA_DIR / "TritonBench_G_v1"),
        instruction_path=instruction_path,
        corpus_path=ensure_exists(TRITONBENCH_DATA_DIR / "train_crawl.json"),
        golden_metrics=ensure_exists(TRITONBENCH_PERF_G_DIR / "golden_metrics"),
        perf_G_path=ensure_exists(TRITONBENCH_PERF_G_DIR),
    )


def parse_model_spec(spec: str) -> tuple[str, str]:
    default_provider = "openai"
    if ":" in spec:
        provider, model_id = spec.split(":", 1)
    else:
        provider, model_id = default_provider, spec
    provider = provider.strip().lower()
    model_id = model_id.strip()
    if not provider or not model_id:
        raise ValueError(f"Invalid generator_model specification: '{spec}'")
    return provider, model_id


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def extract_problem_id(filename: str) -> int | None:
    stem = Path(filename).stem
    token = stem.split("_")[0]
    try:
        return int(token)
    except ValueError:
        return None


def _parse_json_lines(path: Path) -> List[dict]:
    records: List[dict] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _extract_iteration(stem: str, marker: str) -> int:
    try:
        return int(stem.split(marker)[-1])
    except (ValueError, IndexError):
        return -1


def build_agentic_summary(
    run_dir: Path,
    base_output: Path,
    problem_states: Iterable["ProblemState"],
) -> tuple[Path, Dict[str, int]]:
    """Aggregate per-iteration agentic outputs into a single JSONL summary."""

    prefix = base_output.stem
    summary_path = run_dir / "results.jsonl"

    mem_files = sorted(
        (p for p in run_dir.glob(f"{prefix}_mem_*.json") if p.is_file()),
        key=lambda p: _extract_iteration(p.stem, "_mem_"),
    )

    if not mem_files:
        raise FileNotFoundError(
            f"No agentic memory files found for prefix '{prefix}' in {run_dir}."
        )

    latest_mem_path = mem_files[-1]
    latest_iteration = _extract_iteration(latest_mem_path.stem, "_mem_")
    mem_records = _parse_json_lines(latest_mem_path)
    mem_snapshot: Dict[str, Dict] = mem_records[0] if mem_records else {}

    iteration_records: Dict[str, dict] = {}
    if latest_iteration >= 0:
        iter_path = run_dir / f"{prefix}_{latest_iteration}.jsonl"
        iteration_records = {
            record.get("filename", ""): record
            for record in _parse_json_lines(iter_path)
            if isinstance(record, dict)
        }

    order_map: Dict[str, int] = {
        ps.filename: index
        for index, ps in enumerate(problem_states, start=1)
    }

    records: List[dict] = []
    seen_files = set()

    for filename, info in sorted(mem_snapshot.items()):
        seen_files.add(filename)
        iteration_entry = iteration_records.get(filename, {})

        metadata = {
            "call_error": info.get("call_err_msg"),
            "exe_error": info.get("exe_err_msg"),
            "ms": info.get("ms"),
            "efficiency": info.get("efficiency"),
            "iteration": latest_iteration,
        }

        if iteration_entry:
            metadata["instruction"] = iteration_entry.get("instruction")
            metadata["reference_label"] = iteration_entry.get("label")
            metadata["prediction"] = iteration_entry.get("predict")

        records.append(
            {
                "problem_index": order_map.get(filename),
                "problem_name": filename,
                "compiled": bool(info.get("pass_call")),
                "correctness": bool(info.get("pass_exe")),
                "fast_1": bool(info.get("pass_perf")),
                "metadata": metadata,
            }
        )

    for ps in problem_states:
        if ps.filename in seen_files:
            continue
        records.append(
            {
                "problem_index": order_map.get(ps.filename),
                "problem_name": ps.filename,
                "compiled": False,
                "correctness": False,
                "fast_1": False,
                "metadata": {
                    "note": "No agentic result generated for this problem",
                    "iteration": latest_iteration,
                },
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle)
            handle.write("\n")

    return summary_path, {"records": len(records), "latest_iteration": latest_iteration}


def filter_problem_states(problem_states: Iterable[ProblemState], config: "BenchmarkConfig") -> list[ProblemState]:
    selected = list(problem_states)
    if config.problems.problem_ids:
        target_ids = {int(pid) for pid in config.problems.problem_ids}
        selected = [ps for ps in selected if extract_problem_id(ps.filename) in target_ids]

    max_count = config.problems.max_problems or len(selected)
    if config.num_runs and config.num_runs != "all":
        try:
            max_count = min(max_count, int(config.num_runs))
        except ValueError:
            pass
    return selected[:max_count]


def build_provider(config: "BenchmarkConfig"):
    provider_name, model_id = parse_model_spec(config.generator_model)
    provider_name = config.provider or provider_name

    api_key = resolve_provider_api_key(provider_name)
    base_url = config.provider_base_url

    provider_config = ProviderConfig(
        provider=provider_name,
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )
    print(f"[Agentic] Using provider='{provider_config.provider}' model='{provider_config.model}'")
    return create_provider(provider_config)


def derive_output_paths(config: "BenchmarkConfig", provider: str, model_id: str) -> tuple[Path, Path]:
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    provider_safe = sanitize_filename(provider)
    model_safe = sanitize_filename(model_id)
    run_dir = outputs_dir / f"agentic_{provider_safe}_{model_safe}"
    run_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = run_dir / f"results_{timestamp}.jsonl"
    latest = run_dir / "results.jsonl"
    return timestamped, latest


def run(config: "BenchmarkConfig") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.hardware.gpu_id))

    paths = resolve_tritonbench_paths()
    provider_wrapper = build_provider(config)

    dataset = TritonBench(
        statis_path=str(paths.statis_path),
        py_folder=str(paths.py_folder),
        instruction_path=str(paths.instruction_path),
        py_interpreter=sys.executable,
        golden_metrics=str(paths.golden_metrics),
        perf_ref_folder=None,
        perf_G_path=str(paths.perf_G_path),
        result_path=None,
        target_kernels=None,
    )

    original_count = len(dataset)
    dataset.problem_states = filter_problem_states(dataset.problem_states, config)
    if not dataset.problem_states:
        raise ValueError("No problems selected for agentic evaluation. Adjust config.problems.")

    selected_count = len(dataset)
    if selected_count != original_count:
        print(f"[Agentic] Filtered problems: {selected_count} of {original_count}")
    else:
        print(f"[Agentic] Using full TritonBench dataset: {selected_count} problems")

    agent = OptimAgent(
        model=provider_wrapper,
        dataset=dataset,
        corpus_path=str(paths.corpus_path),
        max_perf_debug_num=config.agentic.max_debug_attempts,
        mem_file=None,
    )

    output_path, latest_output_path = derive_output_paths(
        config, provider_wrapper.config.provider, provider_wrapper.config.model
    )

    temperature = 0.0
    iteration_num = max(1, int(config.agentic.max_optimization_cycles))
    ancestor_num = max(1, int(config.agentic.max_debug_attempts))

    print("[Agentic] Writing results to", output_path)
    agent.run(
        output_path=str(output_path),
        multi_thread=True,
        datalen=selected_count,
        iteration_num=iteration_num,
        temperature=temperature,
        ancestor_num=ancestor_num,
        gpu_id=config.hardware.gpu_id,
    )

    print("[Agentic] Completed agentic benchmark run")

    summary_path, summary_stats = build_agentic_summary(
        latest_output_path.parent,
        output_path,
        dataset.problem_states,
    )

    if summary_stats.get("records"):
        print(
            f"[Agentic] Aggregated {summary_stats['records']} problem(s) from iteration {summary_stats['latest_iteration']}."
        )

    if summary_path.exists() and output_path != summary_path:
        try:
            shutil.copy2(summary_path, output_path)
        except Exception as exc:
            print(
                f"[Agentic] Warning: failed to persist timestamped summary at {output_path}: {exc}"
            )

    resolved_results = summary_path

    return {
        "results_path": str(resolved_results),
        "timestamped_path": str(output_path),
        "run_dir": str(latest_output_path.parent),
    }
