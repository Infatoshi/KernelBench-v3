"""Agentic benchmark runner for KernelBench-v3."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, TYPE_CHECKING
from time import perf_counter

import yaml

from agentic.agents.OptimAgent import OptimAgent
from agentic.dataloaders.TritonBench import TritonBench
from providers import ProviderConfig, create_provider, resolve_provider_api_key

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from config import BenchmarkConfig
    from agentic.dataloaders.ProblemState import ProblemState


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
TRITONBENCH_ROOT = PROJECT_ROOT / "data" / "TritonBench"
TRITONBENCH_DATA_DIR = TRITONBENCH_ROOT / "data"
TRITONBENCH_PERF_G_DIR = TRITONBENCH_ROOT / "performance_metrics" / "perf_G"

DEFAULT_FORMATTER_PROVIDER = "groq"
DEFAULT_FORMATTER_MODEL = "moonshotai/kimi-k2-instruct-0905"


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


def _write_text(path: Path, content: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "" if content is None else str(content)
    path.write_text(text, encoding="utf-8")


def _write_yaml(path: Path, data: Dict[str, any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _kernel_dir_name(problem_name: str) -> str:
    return sanitize_filename(problem_name)


def _print_formatted_preview(title: str, file_path: Path, directory: Path) -> None:
    return


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





def filter_problem_states(problem_states: Iterable["ProblemState"], config: "BenchmarkConfig") -> list["ProblemState"]:
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


def derive_output_paths(
    config: "BenchmarkConfig",
    provider: str,
    model_id: str,
    timestamp: datetime | None = None,
) -> tuple[Path, str, datetime]:
    provider_safe = sanitize_filename(provider)
    model_safe = sanitize_filename(model_id)
    language_safe = sanitize_filename(config.language or "cuda")
    timestamp = timestamp or datetime.utcnow()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp_str}_{config.mode}_{language_safe}_{provider_safe}_{model_safe}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id, timestamp


def _problem_dir_name(index: int, filename: str) -> str:
    problem_id = extract_problem_id(filename)
    if problem_id is None:
        problem_id = index
    problem_token = f"{problem_id:03d}"
    return f"kernel_{problem_token}_{_kernel_dir_name(filename)}"


def _final_code(mem) -> str:
    candidates = [
        getattr(mem.ps, "solution", None),
        getattr(mem, "exe_candidate", None),
        getattr(mem, "call_candidate", None),
    ]
    raw_code = getattr(mem, "raw_code", None)
    if raw_code and isinstance(raw_code, list) and raw_code:
        candidates.append(raw_code[0])
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def _render_history_entry(entry: Dict[str, Any]) -> str:
    lines: List[str] = []
    error = entry.get("error")
    request_text = entry.get("request_text")
    kwargs_text = entry.get("kwargs_text")
    response_text = entry.get("response_text")

    if request_text:
        lines.append("=== REQUEST ===")
        lines.append(str(request_text))
    if kwargs_text:
        lines.append("=== KWARGS ===")
        lines.append(str(kwargs_text))
    if response_text:
        lines.append("=== RESPONSE ===")
        lines.append(str(response_text))
    if entry.get("call_pass") is not None:
        lines.append("=== EXECUTION ===")
        lines.append(f"call_pass: {entry['call_pass']}")
        lines.append(f"exe_pass: {entry.get('exe_pass')}")
        if entry.get("call_stdout"):
            lines.append("call_stdout:")
            lines.append(str(entry["call_stdout"]))
        if entry.get("call_stderr"):
            lines.append("call_stderr:")
            lines.append(str(entry["call_stderr"]))
        if entry.get("exe_stdout"):
            lines.append("exe_stdout:")
            lines.append(str(entry["exe_stdout"]))
        if entry.get("exe_stderr"):
            lines.append("exe_stderr:")
            lines.append(str(entry["exe_stderr"]))
    if entry.get("pass_perf") is not None:
        lines.append("=== PERFORMANCE ===")
        lines.append(f"pass_perf: {entry['pass_perf']}")
        if entry.get("latency_ms") is not None:
            lines.append(f"latency_ms: {entry['latency_ms']}")
        if entry.get("efficiency") is not None:
            lines.append(f"efficiency: {entry['efficiency']}")
        if entry.get("details"):
            lines.append(f"details: {entry['details']}")
    if error:
        lines.append("=== ERROR ===")
        lines.append(str(error))

    if not lines:
        lines.append("(no additional details)")

    return "\n".join(lines) + "\n"


def emit_agentic_artifacts(
    config: "BenchmarkConfig",
    run_dir: Path,
    run_id: str,
    timestamp: datetime,
    problem_states: Iterable["ProblemState"],
    memories,
    iteration_num: int,
    temperature: float,
    ancestor_num: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    kernel_summaries: List[Dict[str, Any]] = []
    total = 0
    call_pass_count = 0
    exe_pass_count = 0
    perf_pass_count = 0

    for index, mem in enumerate(memories, start=1):
        ps = getattr(mem, "ps", None)
        if ps is None:
            continue

        total += 1
        call_pass = bool(getattr(mem, "pass_call", False))
        exe_pass = bool(getattr(mem, "pass_exe", False))
        perf_pass = bool(getattr(mem, "pass_perf", False))
        if call_pass:
            call_pass_count += 1
        if exe_pass:
            exe_pass_count += 1
        if perf_pass:
            perf_pass_count += 1

        filename = ps.filename
        dir_name = _problem_dir_name(index, filename)
        problem_dir = run_dir / dir_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        summary_lines = [
            f"filename: {filename}",
            f"iterations_requested: {iteration_num}",
            f"call_pass: {'yes' if call_pass else 'no'}",
            f"exe_pass: {'yes' if exe_pass else 'no'}",
            f"perf_pass: {'yes' if perf_pass else 'no'}",
        ]
        if getattr(mem, "call_err_msg", None):
            summary_lines.append(f"call_error: {mem.call_err_msg}")
        if getattr(mem, "exe_err_msg", None):
            summary_lines.append(f"exe_error: {mem.exe_err_msg}")
        if getattr(mem, "ms", None) is not None:
            summary_lines.append(f"latency_ms: {mem.ms}")
        if getattr(mem, "efficiency", None) is not None:
            summary_lines.append(f"efficiency: {mem.efficiency}")

        _write_text(problem_dir / "summary.txt", "\n".join(summary_lines) + "\n")

        instruction = ps.instruction or ""
        _write_text(problem_dir / "instruction.txt", instruction)

        reference = ps.label or ""
        _write_text(problem_dir / "reference_solution.py", reference)

        test_code = ps.test_code or ""
        _write_text(problem_dir / "test_harness.py", test_code)

        final_code = _final_code(mem)
        _write_text(problem_dir / "final_code.py", final_code)

        raw_candidates = getattr(mem, "raw_code", None)
        if raw_candidates and isinstance(raw_candidates, list):
            _write_text(problem_dir / "last_raw_candidate.py", raw_candidates[0] or "")

        _print_formatted_preview(
            f"[Agentic] Formatted output for {problem_dir.name}",
            problem_dir / "final_code.py",
            problem_dir,
        )

        history_entries = getattr(mem, "history", []) or []
        stage_counts: Dict[tuple[int, str], int] = {}
        for entry in history_entries:
            iteration = int(entry.get("iteration", 0))
            stage = str(entry.get("stage", "stage"))
            key = (iteration, stage)
            stage_counts[key] = stage_counts.get(key, 0) + 1
            suffix = stage_counts[key] - 1
            file_name = stage if suffix == 0 else f"{stage}_{suffix}"
            history_dir = problem_dir / "history" / f"iteration_{iteration:02d}"
            history_dir.mkdir(parents=True, exist_ok=True)
            stage_path = history_dir / f"{file_name}.txt"
            _write_text(stage_path, _render_history_entry(entry))

        kernel_summaries.append(
            {
                "filename": filename,
                "path": problem_dir.relative_to(run_dir).as_posix(),
                "call_pass": call_pass,
                "exe_pass": exe_pass,
                "perf_pass": perf_pass,
                "latency_ms": getattr(mem, "ms", None),
                "efficiency": getattr(mem, "efficiency", None),
                "iterations": iteration_num,
            }
        )

    metrics_bundle = {
        "total": total,
        "compiled_count": call_pass_count,
        "correct_count": exe_pass_count,
        "performance_count": perf_pass_count,
        "compilation_rate": round((call_pass_count / total) * 100, 3) if total else 0.0,
        "correctness_rate": round((exe_pass_count / call_pass_count) * 100, 3)
        if call_pass_count
        else 0.0,
        "performance_rate": round((perf_pass_count / exe_pass_count) * 100, 3)
        if exe_pass_count
        else 0.0,
    }

    environment: Dict[str, Any] | None = None
    hw = getattr(config, "hardware", None)
    if hw:
        environment = {
            "hardware": {
                "gpu_architecture": getattr(hw, "gpu_architecture", None),
                "gpu_id": getattr(hw, "gpu_id", None),
            }
        }

    manifest = {
        "run_id": run_id,
        "timestamp": timestamp.replace(microsecond=0).isoformat() + "Z",
        "mode": config.mode,
        "language": config.language,
        "provider": config.provider,
        "model": config.generator_model,
        "elapsed_seconds": elapsed_seconds,
        "settings": {
            "num_runs": config.num_runs,
            "iterations": iteration_num,
            "temperature": temperature,
            "ancestor_num": ancestor_num,
            "formatter": {
                "provider": getattr(config, "formatter_provider", DEFAULT_FORMATTER_PROVIDER),
                "model": getattr(config, "formatter_model", DEFAULT_FORMATTER_MODEL),
            },
        },
        "environment": environment,
        "kernels": kernel_summaries,
    }

    manifest_path = run_dir / "manifest.yaml"
    _write_yaml(manifest_path, manifest)

    print(
        f"[Agentic] Completed run for {total} problem(s) in {elapsed_seconds:.1f}s. Results saved to {manifest_path}"
    )

    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "metrics": metrics_bundle,
        "elapsed_seconds": elapsed_seconds,
    }


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

    run_timestamp = datetime.utcnow()
    run_dir, run_id, timestamp_obj = derive_output_paths(
        config,
        provider_wrapper.config.provider,
        provider_wrapper.config.model,
        timestamp=run_timestamp,
    )

    temperature = 0.0
    iteration_num = max(1, int(config.agentic.max_optimization_cycles))
    ancestor_num = max(1, int(config.agentic.max_debug_attempts))

    run_start = perf_counter()

    agent.run(
        output_path=None,
        multi_thread=True,
        datalen=selected_count,
        iteration_num=iteration_num,
        temperature=temperature,
        ancestor_num=ancestor_num,
        gpu_id=config.hardware.gpu_id,
    )

    elapsed_seconds = round(perf_counter() - run_start, 1)

    artifacts = emit_agentic_artifacts(
        config,
        run_dir,
        run_id,
        timestamp_obj,
        dataset.problem_states,
        agent.memories,
        iteration_num,
        temperature,
        ancestor_num,
        elapsed_seconds,
    )

    return artifacts
