from __future__ import annotations

import json
from collections import Counter
import multiprocessing
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Dict, Iterable, TYPE_CHECKING

from prompt_constructor import (
    prompt_generate_custom_cuda_from_prompt_template,
    prompt_generate_custom_triton_from_template,
    build_formatter_messages,
)
from utils import extract_first_code, set_gpu_arch
from dataset import construct_kernelbench_dataset
from raw.eval import eval_kernel_against_ref_auto, KernelExecResult
from providers import ProviderConfig, create_provider, resolve_provider_api_key

if TYPE_CHECKING:  # pragma: no cover
    from config import BenchmarkConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

_CPU_STATE: Dict[str, Any] = {}


def _render_progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[ ]"
    completed = max(0, min(completed, total))
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"


def _update_progress(progress_state: Dict[str, Any]) -> None:
    if not progress_state:
        return

    lock: threading.Lock = progress_state["lock"]
    with lock:
        progress_state["completed"] += 1
        completed = progress_state["completed"]
        total = progress_state["total"]
        label = progress_state["label"]
        prefix = progress_state["prefix"]
        bar = _render_progress_bar(completed, total)
        sys.stdout.write(f"\r{prefix} {bar} {completed}/{total}")
        sys.stdout.flush()
        if completed >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def sanitize_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def build_provider(config: "BenchmarkConfig"):
    provider_name, model_id = config.provider, config.generator_model
    api_key = resolve_provider_api_key(provider_name)
    base_url = config.provider_base_url

    provider_config = ProviderConfig(
        provider=provider_name,
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )
    return create_provider(provider_config)


def build_formatter_provider(config: "BenchmarkConfig"):
    provider_name = getattr(config, "formatter_provider", None)
    if not provider_name:
        return None

    model_id = getattr(config, "formatter_model", None)
    if not model_id:
        raise ValueError("formatter_model must be set when formatter_provider is configured.")

    api_key = resolve_provider_api_key(provider_name)
    base_url = getattr(config, "formatter_base_url", None)

    provider_config = ProviderConfig(
        provider=provider_name,
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )
    return create_provider(provider_config)


@dataclass
class GenerationConfig:
    level: int
    problem_ids: Iterable[int] | None
    max_problems: int


def load_reference(level: int, problem_id: int) -> tuple[str, str]:
    dataset = construct_kernelbench_dataset(level)
    index = problem_id - 1
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Problem id {problem_id} out of range for level {level}")
    path = Path(dataset[index])
    return path.read_text(), path.name


def resolve_problems(config: GenerationConfig) -> list[int]:
    dataset = construct_kernelbench_dataset(config.level)
    all_ids = list(range(1, len(dataset) + 1))
    if config.problem_ids:
        filtered = [pid for pid in config.problem_ids if 1 <= pid <= len(dataset)]
    else:
        filtered = all_ids
    return filtered[: config.max_problems]


def build_inference_callable(config: "BenchmarkConfig"):
    provider = build_provider(config)
    generation_limit = getattr(config, "generation_max_tokens", 4096)
    if generation_limit is not None and generation_limit <= 0:
        generation_limit = None

    def _inference(prompt: str | list[dict]):
        messages = []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        return provider.generate(messages, max_tokens=generation_limit)

    return _inference


def build_formatter_callable(config: "BenchmarkConfig"):
    formatter_provider = build_formatter_provider(config)
    if formatter_provider is None:
        return None
    formatter_limit = getattr(config, "formatter_max_tokens", None)
    if formatter_limit is not None and formatter_limit <= 0:
        formatter_limit = None

    def _formatter(prompt: str | list[dict]):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        return formatter_provider.generate(messages, max_tokens=formatter_limit)

    return _formatter


def _cpu_initializer(config: "BenchmarkConfig", generation_level: int, job_queue, result_store, build_root: Path):
    """Set up CPU worker state for prompt generation."""
    global _CPU_STATE
    _CPU_STATE = {
        "config": config,
        "generation_level": generation_level,
        "queue": job_queue,
        "results": result_store,
        "profile_enabled": config.profile_stages,
        "build_root": build_root,
    }
    _CPU_STATE["inference_fn"] = build_inference_callable(config)
    _CPU_STATE["formatter_fn"] = build_formatter_callable(config)


def _prepare_problem(level: int, problem_id: int, config: "BenchmarkConfig", inference_fn) -> Dict[str, Any]:
    ref_arch_src, problem_name = load_reference(level, problem_id)
    generated: str | None
    raw_completion: str | None = None
    formatted_completion: str | None = None

    lang = (config.language or "cuda").lower()
    if lang == "triton":
        prompt = prompt_generate_custom_triton_from_template(ref_arch_src)
    else:
        prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    completion = inference_fn(prompt)
    raw_completion = completion
    formatter_fn = _CPU_STATE.get("formatter_fn")
    if formatter_fn:
        try:
            messages = build_formatter_messages(
                language=lang,
                reference_architecture=ref_arch_src,
                original_prompt=prompt,
                raw_completion=completion,
            )
            formatted_completion = formatter_fn(messages)
        except Exception:  # noqa: BLE001
            formatted_completion = None
    candidate_source = formatted_completion or completion
    generated = (
        extract_first_code(candidate_source, ["python", "cpp"])
        or extract_first_code(completion, ["python", "cpp"])
        or candidate_source
    )

    return {
        "level": level,
        "problem_id": problem_id,
        "problem_name": problem_name,
        "ref_arch_src": ref_arch_src,
        "generated_code": generated,
        "raw_completion": raw_completion,
        "formatted_completion": formatted_completion,
    }


def _cpu_prepare_problem(problem_id: int) -> Dict[str, Any]:
    if not _CPU_STATE:
        raise RuntimeError("CPU worker state has not been initialized")

    config: "BenchmarkConfig" = _CPU_STATE["config"]
    level = _CPU_STATE["generation_level"]
    queue = _CPU_STATE["queue"]
    results_store = _CPU_STATE["results"]
    inference_fn = _CPU_STATE.get("inference_fn")
    profile_enabled = _CPU_STATE.get("profile_enabled", False)
    build_root: Path = _CPU_STATE.get("build_root")

    cpu_profile: Dict[str, float] | None = {} if profile_enabled else None
    cpu_start = perf_counter()

    try:
        if inference_fn is None:
            raise RuntimeError("Inference function not initialized")

        prepare_start = perf_counter()
        candidate = _prepare_problem(level, problem_id, config, inference_fn)
        if cpu_profile is not None:
            cpu_profile["cpu_prepare_s"] = perf_counter() - prepare_start
        if cpu_profile is not None:
            cpu_profile["cpu_total_s"] = perf_counter() - cpu_start
            candidate["profile"] = {
                "cpu": cpu_profile,
                "enqueue_ts": perf_counter(),
            }
        else:
            candidate["profile"] = None
        if build_root is not None:
            candidate["build_dir"] = str(Path(build_root) / f"problem_{problem_id}")
        queue.put(candidate)
        return {"problem_id": problem_id, "queued": True}

    except Exception as exc:  # noqa: BLE001
        results_store[problem_id] = {
            "level": level,
            "problem_id": problem_id,
            "problem_name": None,
            "compiled": False,
            "correctness": False,
            "runtime": None,
            "metadata": {
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "stage": "generation",
            },
            "generated_code": None,
        }
        if cpu_profile is not None:
            cpu_profile["cpu_total_s"] = perf_counter() - cpu_start
            cpu_total = cpu_profile.get("cpu_total_s", 0.0)
            times = {
                "cpu_total_s": cpu_total,
                "total_s": cpu_total,
            }
            percentages = {}
            if cpu_total > 0:
                percentages["cpu_total_pct"] = 100.0
            results_store[problem_id]["profiling"] = {
                "times_s": {k: round(v, 6) for k, v in times.items()},
                "percentages": percentages,
            }
        return {"problem_id": problem_id, "queued": False}


def _run_gpu_evaluation(candidate: Dict[str, Any], config: "BenchmarkConfig", profile: Dict[str, float] | None = None) -> Dict[str, Any]:
    result = eval_kernel_against_ref_auto(
        original_model_src=candidate["ref_arch_src"],
        custom_model_src=candidate["generated_code"],
        measure_performance=config.fast_p_threshold is not None,
        verbose=False,
        num_correct_trials=1,
        num_perf_trials=5,
        profile=profile,
        build_dir=candidate.get("build_dir"),
    )

    metadata = result.metadata if isinstance(result, KernelExecResult) else {}
    return {
        "level": candidate["level"],
        "problem_id": candidate["problem_id"],
        "problem_name": candidate["problem_name"],
        "compiled": bool(result and result.compiled),
        "correctness": bool(result and result.correctness),
        "runtime": getattr(result, "runtime", None),
        "metadata": metadata,
        "generated_code": candidate["generated_code"],
    }


def _evaluate_candidate(
    candidate: Dict[str, Any],
    config: "BenchmarkConfig",
    build_root: Path | None = None,
) -> Dict[str, Any]:
    profile_enabled = config.profile_stages
    cpu_profile = None
    enqueue_ts = None
    gpu_profile: Dict[str, float] | None = {} if profile_enabled else None

    if profile_enabled:
        candidate_profile = candidate.get("profile") or {}
        cpu_profile = candidate_profile.get("cpu")
        enqueue_ts = candidate_profile.get("enqueue_ts")

    start_time = perf_counter()
    if gpu_profile is not None and enqueue_ts is not None:
        gpu_profile["queue_wait_s"] = start_time - enqueue_ts

    problem_id = candidate["problem_id"]
    build_dir = candidate.get("build_dir")
    if build_dir is None and build_root is not None:
        build_dir = str(Path(build_root) / f"problem_{problem_id}")
    if build_dir is not None:
        Path(build_dir).mkdir(parents=True, exist_ok=True)

    candidate_for_eval = dict(candidate)
    if build_dir is not None:
        candidate_for_eval["build_dir"] = build_dir

    try:
        result = _run_gpu_evaluation(candidate_for_eval, config, profile=gpu_profile)

        if gpu_profile is not None and "gpu_total_s" not in gpu_profile:
            gpu_profile["gpu_total_s"] = perf_counter() - start_time

        if profile_enabled:
            stage_times: Dict[str, float] = {}
            cpu_total = 0.0
            queue_wait = 0.0
            gpu_total = 0.0

            if cpu_profile:
                cpu_total = cpu_profile.get("cpu_total_s", 0.0)
                stage_times["cpu_total_s"] = cpu_total
                for key, value in cpu_profile.items():
                    if key != "cpu_total_s":
                        stage_times[key] = value

            if gpu_profile:
                queue_wait = gpu_profile.get("queue_wait_s", 0.0)
                gpu_total = gpu_profile.get("gpu_total_s", 0.0)
                stage_times["queue_wait_s"] = queue_wait
                stage_times["gpu_total_s"] = gpu_total
                for key, value in gpu_profile.items():
                    if key not in {"queue_wait_s", "gpu_total_s"} and key.endswith("_s"):
                        stage_times[key] = value

            total_time = cpu_total + queue_wait + gpu_total
            stage_times["total_s"] = total_time

            percentages = {}
            if total_time > 0:
                if cpu_total:
                    percentages["cpu_total_pct"] = round((cpu_total / total_time) * 100, 4)
                if queue_wait:
                    percentages["queue_wait_pct"] = round((queue_wait / total_time) * 100, 4)
                if gpu_total:
                    percentages["gpu_total_pct"] = round((gpu_total / total_time) * 100, 4)

            result["profiling"] = {
                "times_s": {k: round(v, 6) for k, v in stage_times.items()},
                "percentages": percentages,
            }

        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "level": candidate.get("level"),
            "problem_id": candidate.get("problem_id"),
            "problem_name": candidate.get("problem_name"),
            "compiled": False,
            "correctness": False,
            "runtime": None,
            "metadata": {
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "stage": "evaluation",
            },
            "generated_code": candidate.get("generated_code"),
        }


def _gpu_process_entry(
    candidate: Dict[str, Any],
    config: "BenchmarkConfig",
    build_root: str | None,
) -> tuple[int, Dict[str, Any]]:
    problem_id = candidate.get("problem_id")
    build_root_path = Path(build_root) if build_root else None

    try:
        if config.hardware.gpu_architecture:
            set_gpu_arch([config.hardware.gpu_architecture])
        result = _evaluate_candidate(candidate, config, build_root_path)
    except Exception as exc:  # noqa: BLE001
        result = {
            "level": candidate.get("level"),
            "problem_id": problem_id,
            "problem_name": candidate.get("problem_name"),
            "compiled": False,
            "correctness": False,
            "runtime": None,
            "metadata": {
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "stage": "evaluation",
            },
            "generated_code": candidate.get("generated_code"),
        }

    return problem_id, result


def run(config: "BenchmarkConfig") -> None:
    if config.language not in {"triton", "cuda"}:
        print(f"[Raw] Warning: '{config.language}' kernels are not fully supported; falling back to CUDA evaluation workflow.")

    if config.raw_max_jobs:
        os.environ.setdefault("MAX_JOBS", str(config.raw_max_jobs))

    if config.hardware.gpu_architecture:
        set_gpu_arch([config.hardware.gpu_architecture])

    max_problems = config.problems.max_problems
    if config.num_runs and config.num_runs != "all":
        try:
            max_problems = min(max_problems, int(config.num_runs))
        except ValueError:
            pass

    gen_cfg = GenerationConfig(
        level=max(1, config.problems.levels[0]),
        problem_ids=config.problems.problem_ids,
        max_problems=max_problems,
    )

    problems = resolve_problems(gen_cfg)
    if not problems:
        raise ValueError("No problems selected for raw evaluation")

    provider_safe = sanitize_component(config.provider)
    model_safe = sanitize_component(config.generator_model.replace("/", "_"))
    language_safe = sanitize_component(config.language or "cuda")
    run_dir = RUNS_DIR / f"{config.mode}_{language_safe}_{provider_safe}_{model_safe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    build_root = run_dir / "build_cache"
    build_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, config.raw_concurrency or 1)
    use_parallel = worker_count > 1 and len(problems) > 1

    if config.verbose:
        mode = "parallel" if use_parallel else "sequential"
        print(
            f"[Raw] Evaluating {len(problems)} problem(s) in {mode} mode "
            f"using {worker_count if use_parallel else 1} CPU worker(s) with queued GPU execution."
        )

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    job_queue = ctx.Queue(maxsize=worker_count * 2 if use_parallel else 1)
    shared_results: Dict[int, Dict[str, Any]] = manager.dict()

    progress_state: Dict[str, Any] = {
        "lock": threading.Lock(),
        "completed": 0,
        "total": len(problems),
        "label": f"{config.provider}/{config.generator_model}",
        "prefix": f"[{config.mode.upper()}|{config.language.upper()}]",
    }

    gpu_worker_count = max(1, config.raw_gpu_concurrency or 1)

    process_executor_kwargs = {
        "max_workers": gpu_worker_count,
        "mp_context": ctx,
    }
    try:
        gpu_executor = ProcessPoolExecutor(max_tasks_per_child=1, **process_executor_kwargs)
    except TypeError:
        gpu_executor = ProcessPoolExecutor(**process_executor_kwargs)

    pending_futures_lock = threading.Lock()
    pending_futures: set = set()
    dispatch_done = threading.Event()

    def _handle_gpu_future(future):
        try:
            problem_id, result = future.result()
        except Exception as exc:  # noqa: BLE001
            if config.verbose:
                print(f"[Raw] GPU worker raised unexpected error: {exc}")
            return

        if problem_id is None:
            return

        shared_results[problem_id] = result
        _update_progress(progress_state)
        with pending_futures_lock:
            pending_futures.discard(future)

    def _gpu_dispatch_loop() -> None:
        sentinel_seen = 0
        while True:
            candidate = job_queue.get()
            if candidate is None:
                sentinel_seen += 1
                if sentinel_seen >= gpu_worker_count:
                    break
                continue

            future = gpu_executor.submit(
                _gpu_process_entry,
                candidate,
                config,
                str(build_root),
            )
            with pending_futures_lock:
                pending_futures.add(future)
            future.add_done_callback(_handle_gpu_future)

        dispatch_done.set()

    dispatch_thread = threading.Thread(
        target=_gpu_dispatch_loop,
        name="raw-gpu-dispatch",
        daemon=True,
    )
    dispatch_thread.start()

    if use_parallel:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_cpu_initializer,
            initargs=(config, gen_cfg.level, job_queue, shared_results, build_root),
        ) as executor:
            future_map = {executor.submit(_cpu_prepare_problem, pid): pid for pid in problems}

            for future in as_completed(future_map):
                pid = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    shared_results[pid] = {
                        "level": gen_cfg.level,
                        "problem_id": pid,
                        "problem_name": None,
                        "compiled": False,
                        "correctness": False,
                        "runtime": None,
                        "metadata": {
                            "error": str(exc),
                            "exception_type": type(exc).__name__,
                            "stage": "generation",
                        },
                        "generated_code": None,
                    }
                    _update_progress(progress_state)
                    if config.verbose:
                        print(f"[Raw] Prepared level {gen_cfg.level} problem {pid}")
                else:
                    if config.verbose:
                        print(f"[Raw] Prepared level {gen_cfg.level} problem {pid}")
                    if result and not result.get("queued", True):
                        _update_progress(progress_state)
    else:
        _cpu_initializer(config, gen_cfg.level, job_queue, shared_results, build_root)
        for pid in problems:
            if config.verbose:
                print(f"[Raw] Preparing level {gen_cfg.level} problem {pid}")
            result_info = _cpu_prepare_problem(pid)
            if result_info and not result_info.get("queued", True):
                _update_progress(progress_state)

    for _ in range(gpu_worker_count):
        job_queue.put(None)

    dispatch_thread.join()
    dispatch_done.wait()
    gpu_executor.shutdown(wait=True)

    results = {pid: shared_results.get(pid) for pid in problems}

    # Ensure every problem has a result entry
    for pid in problems:
        if results[pid] is None:
            results[pid] = {
                "level": gen_cfg.level,
                "problem_id": pid,
                "problem_name": None,
                "compiled": False,
                "correctness": False,
                "runtime": None,
                "metadata": {
                    "error": "Result missing after evaluation",
                    "exception_type": "MissingResultError",
                },
                "generated_code": None,
            }

    with results_path.open("w", encoding="utf-8") as f:
        for pid in problems:
            f.write(json.dumps(results[pid]) + "\n")
            f.flush()

    total = len(problems)
    compiled_count = sum(
        1 for record in results.values() if record and record.get("compiled")
    )
    correct_count = sum(
        1 for record in results.values() if record and record.get("correctness")
    )
    failure_breakdown: Counter[str] = Counter()
    stage_breakdown: Counter[str] = Counter()

    for record in results.values():
        if not record:
            continue
        metadata = record.get("metadata") or {}
        if not record.get("compiled", False):
            reason = str(
                metadata.get("error_category")
                or metadata.get("compilation_error")
                or metadata.get("error")
                or "compilation_failed"
            )
            failure_breakdown[reason] += 1
            stage_breakdown["compilation"] += 1
        elif not record.get("correctness", False):
            reason = str(
                metadata.get("error_category")
                or metadata.get("correctness_issue")
                or metadata.get("runtime_error")
                or metadata.get("error")
                or "correctness_failed"
            )
            stage = str(metadata.get("stage") or "correctness")
            failure_breakdown[reason] += 1
            stage_breakdown[stage] += 1

    print(f"[Raw] Compiled: {compiled_count}/{total} | Correct: {correct_count}/{total}")
    if failure_breakdown:
        print("[Raw] Failure breakdown:")
        for reason, count in failure_breakdown.most_common():
            label = reason.replace("_", " ")
            print(f"  - {label}: {count}")
        if stage_breakdown:
            print("[Raw] Failure stages:")
            for stage, count in stage_breakdown.most_common():
                label = stage.replace("_", " ")
                print(f"  - {label}: {count}")

    if config.verbose:
        print(f"[Raw] Completed run for {len(problems)} problems. Results saved to {results_path}")

    manager.shutdown()

    return {
        "results_path": str(results_path),
        "run_dir": str(run_dir),
        "evaluated_problems": len(problems),
    }
