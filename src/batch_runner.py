"""Batch runner for evaluating multiple models from YAML configuration."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from config import AgenticConfig, BenchmarkConfig, HardwareConfig, ProblemSetConfig, GenerationConfig
from hardware.specs import default_gpu_name, gpu_spec
from metrics import load_metrics, save_metrics
from providers import verify_model_responds_hello

GREEN = "\033[32m"
RESET = "\033[0m"
HOST_CPU_COUNT = max(1, os.cpu_count() or 1)

DEFAULT_FORMATTER_PROVIDER = "groq"
DEFAULT_FORMATTER_MODEL = "moonshotai/kimi-k2-instruct-0905"


def load_yaml_config(yaml_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file as a dictionary."""
    with open(yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def sanitize_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def expand_run_matrix(yaml_data: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return the (mode, language, model) combinations requested by the config."""
    models = yaml_data.get("models", [])
    if not models:
        raise ValueError("No models defined in YAML config under 'models'.")

    modes = yaml_data.get("modes")
    if not modes:
        legacy_mode = yaml_data.get("mode")
        modes = [legacy_mode] if legacy_mode else ["raw", "agentic"]

    languages = yaml_data.get("languages")
    if not languages:
        legacy_language = yaml_data.get("language")
        languages = [legacy_language] if legacy_language else ["cuda", "triton"]

    combinations: List[Tuple[str, str, Dict[str, Any]]] = []
    for mode, language, model_entry in product(modes, languages, models):
        combinations.append((str(mode).lower(), str(language).lower(), model_entry))
    return combinations


def _verify_provider_models(
    run_plan: List[Tuple[str, str, Dict[str, Any]]],
    yaml_data: Dict[str, Any],
    cli_overrides: Dict[str, Any],
) -> None:
    """Ensure every provider/model pair responds to a simple hello prompt."""
    seen: set[tuple[str, str, str]] = set()
    errors: List[str] = []

    if not run_plan:
        return
    unique_targets: List[Tuple[str, str, str | None]] = []
    for mode, language, model_entry in run_plan:
        config = yaml_to_benchmark_config(yaml_data, model_entry, mode, language, cli_overrides)
        if getattr(config.generation, "mode", "llm") == "local":
            continue
        key = (
            config.provider.lower(),
            config.generator_model,
            config.provider_base_url or "",
        )
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append((config.provider, config.generator_model, config.provider_base_url))

    total_targets = len(unique_targets)

    def _render_progress_bar(completed: int, total: int, width: int = 30) -> str:
        if total <= 0:
            return "[ ]"
        completed = max(0, min(completed, total))
        filled = int(width * completed / total)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    print("[Preflight] Verifying provider/model combinations...")
    for index, (provider_name, model_name, base_url) in enumerate(unique_targets, start=1):
        try:
            verify_model_responds_hello(
                provider_name,
                model_name,
                base_url,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider_name}/{model_name}: {exc}")
        progress_bar = _render_progress_bar(index, total_targets)
        sys.stdout.write(
            f"\r[Preflight] Verifying provider/model combinations... {progress_bar} {index}/{total_targets}"
        )
        sys.stdout.flush()

    if total_targets:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if errors:
        print("[Preflight] Provider/model validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("[Preflight] All provider/model combinations responded with 'Hello.'")


def _resolve_hardware(yaml_data: Dict[str, Any]) -> HardwareConfig:
    modal_cfg = yaml_data.get("modal") or {}
    gpu_cfg = modal_cfg.get("gpu") or {}
    hw_cfg = yaml_data.get("hardware") or {}

    gpu_name = gpu_cfg.get("name") or hw_cfg.get("gpu_name") or default_gpu_name()
    spec = gpu_spec(gpu_name)

    architecture = spec.architecture
    if architecture == "unknown":
        architecture = hw_cfg.get("gpu_architecture", "unknown")

    return HardwareConfig(
        gpu_name=spec.name,
        gpu_architecture=architecture,
        gpu_id=int(hw_cfg.get("gpu_id", 0) or 0),
        gpu_memory_gb=spec.memory_gb,
        gpu_memory_type=spec.memory_type,
        tensor_core_generation=spec.tensor_core_generation,
    )


def _resolve_problems(yaml_data: Dict[str, Any]) -> ProblemSetConfig:
    prob_data = yaml_data.get("problems", {})
    return ProblemSetConfig(
        levels=prob_data.get("levels", [1, 2]),
        problem_ids=prob_data.get("problem_ids"),
        max_problems=prob_data.get("max_problems", 100),
    )


def _resolve_agentic(yaml_data: Dict[str, Any], overrides: Dict[str, Any]) -> AgenticConfig:
    ag_defaults = yaml_data.get("defaults", {}).get("agentic", {})
    ag_data = yaml_data.get("agentic", {})
    ag_model = {
        "max_debug_attempts": ag_data.get(
            "max_debug_attempts",
            ag_defaults.get("max_debug_attempts", 3),
        ),
        "max_optimization_cycles": ag_data.get(
            "max_optimization_cycles",
            ag_defaults.get("max_optimization_cycles", 2),
        ),
        "reflector_model": ag_data.get(
            "reflector_model",
            ag_defaults.get("reflector_model"),
        ),
        "optimizer_model": ag_data.get(
            "optimizer_model",
            ag_defaults.get("optimizer_model"),
        ),
    }

    ag_model.update(overrides.get("agentic", {}))
    return AgenticConfig(**ag_model)


def yaml_to_benchmark_config(
    yaml_data: Dict[str, Any],
    model_entry: Dict[str, Any],
    mode: str,
    language: str,
    cli_overrides: Dict[str, Any] | None = None,
) -> BenchmarkConfig:
    """Convert YAML + model entry into a BenchmarkConfig instance."""

    cli_overrides = cli_overrides or {}
    defaults = yaml_data.get("defaults", {})

    provider = model_entry.get("provider") or yaml_data.get("provider", "openai")
    model_id = model_entry.get("model") or yaml_data.get("generator_model", "gpt-4-turbo")
    base_url = model_entry.get("base_url") or yaml_data.get("provider_base_url")

    formatter_defaults = defaults.get("formatter", {})
    formatter_cfg = model_entry.get("formatter") or {}
    if not isinstance(formatter_cfg, dict):
        formatter_cfg = {}

    formatter_provider = formatter_cfg.get("provider") or yaml_data.get("formatter_provider")
    if not formatter_provider:
        formatter_provider = formatter_defaults.get("provider") or DEFAULT_FORMATTER_PROVIDER

    formatter_model = formatter_cfg.get("model") or yaml_data.get("formatter_model")
    if not formatter_model:
        formatter_model = formatter_defaults.get("model") or DEFAULT_FORMATTER_MODEL

    formatter_base_url = formatter_cfg.get("base_url") or yaml_data.get("formatter_base_url")
    if not formatter_base_url:
        formatter_base_url = formatter_defaults.get("base_url")

    num_runs = model_entry.get("num_runs", yaml_data.get("num_runs", defaults.get("num_runs")))
    if "num_runs" in cli_overrides:
        num_runs = cli_overrides["num_runs"]
    profile_stages = True

    modal_gpu_cfg = (yaml_data.get("modal") or {}).get("gpu") or {}
    modal_gpu_count = int(modal_gpu_cfg.get("count") or 1)
    env_gpu_count = os.environ.get("KB3_MODAL_GPU_COUNT")
    if env_gpu_count:
        try:
            modal_gpu_count = max(modal_gpu_count, int(env_gpu_count))
        except ValueError:
            pass

    raw_concurrency = HOST_CPU_COUNT
    raw_gpu_concurrency = max(1, modal_gpu_count)
    raw_max_jobs = HOST_CPU_COUNT

    generation_max_tokens = model_entry.get("generation_max_tokens")
    if generation_max_tokens is None:
        generation_max_tokens = yaml_data.get(
            "generation_max_tokens",
            defaults.get("generation_max_tokens"),
        )
    if generation_max_tokens is not None:
        generation_max_tokens = int(generation_max_tokens)

    formatter_max_tokens = model_entry.get("formatter_max_tokens")
    if formatter_max_tokens is None:
        formatter_max_tokens = yaml_data.get(
            "formatter_max_tokens",
            defaults.get("formatter_max_tokens"),
        )
    if formatter_max_tokens is not None:
        formatter_max_tokens = int(formatter_max_tokens)

    generation_defaults = defaults.get("generation", {}) if isinstance(defaults, dict) else {}
    yaml_generation = yaml_data.get("generation", {}) or {}
    if not isinstance(yaml_generation, dict):
        yaml_generation = {}
    if not isinstance(generation_defaults, dict):
        generation_defaults = {}
    model_generation = model_entry.get("generation", {}) or {}
    if not isinstance(model_generation, dict):
        model_generation = {}

    def _first_value(*candidates):
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    generation_mode_value = _first_value(
        model_generation.get("mode"),
        yaml_generation.get("mode"),
        generation_defaults.get("mode"),
    )
    if generation_mode_value is None:
        generation_mode_value = GenerationConfig().mode
    generation_mode = str(generation_mode_value).lower().strip()
    if generation_mode not in {"llm", "local"}:
        generation_mode = "llm"

    generation_local_dir_value = _first_value(
        model_generation.get("local_dir"),
        yaml_generation.get("local_dir"),
        generation_defaults.get("local_dir"),
    )
    if generation_local_dir_value is None:
        generation_local_dir_value = GenerationConfig().local_dir
    generation_local_dir = (
        str(generation_local_dir_value)
        if generation_local_dir_value is not None and str(generation_local_dir_value).strip()
        else None
    )

    reuse_existing_value = _first_value(
        model_generation.get("reuse_existing"),
        yaml_generation.get("reuse_existing"),
        generation_defaults.get("reuse_existing"),
    )
    if reuse_existing_value is None:
        reuse_existing_value = GenerationConfig().reuse_existing
    reuse_existing = bool(reuse_existing_value)

    if "formatter_provider" in cli_overrides:
        formatter_provider = cli_overrides["formatter_provider"]
    if "formatter_model" in cli_overrides:
        formatter_model = cli_overrides["formatter_model"]
    if "formatter_base_url" in cli_overrides:
        formatter_base_url = cli_overrides["formatter_base_url"]

    hardware = _resolve_hardware(yaml_data)
    problems = _resolve_problems(yaml_data)
    agentic_cfg = _resolve_agentic(yaml_data, cli_overrides)

    config_kwargs: Dict[str, Any] = dict(
        mode=mode,
        language=language,
        provider=provider,
        provider_base_url=base_url,
        formatter_provider=formatter_provider,
        formatter_model=formatter_model,
        formatter_base_url=formatter_base_url,
        generator_model=model_id,
        verbose=False,
        num_runs=num_runs,
        profile_stages=profile_stages,
        hardware=hardware,
        problems=problems,
        agentic=agentic_cfg,
        generation=GenerationConfig(
            mode="local" if generation_mode == "local" else "llm",
            local_dir=generation_local_dir,
            reuse_existing=reuse_existing,
        ),
        fast_p_threshold=yaml_data.get("fast_p_threshold", defaults.get("fast_p_threshold")),
        raw_concurrency=raw_concurrency,
        raw_gpu_concurrency=raw_gpu_concurrency,
        raw_max_jobs=raw_max_jobs,
    )

    if generation_max_tokens is not None:
        config_kwargs["generation_max_tokens"] = generation_max_tokens
    if formatter_max_tokens is not None:
        config_kwargs["formatter_max_tokens"] = formatter_max_tokens

    config = BenchmarkConfig(**config_kwargs)

    if config.agentic:
        config.agentic.reflector_model = config.generator_model
        config.agentic.optimizer_model = config.generator_model

    return config


def fingerprint_config(config: BenchmarkConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_cache_path(
    json_dir: Path,
    mode: str,
    language: str,
    provider: str,
    model_id: str,
) -> Path:
    provider_safe = sanitize_component(provider)
    model_safe = sanitize_component(model_id.replace("/", "_"))
    language_safe = sanitize_component(language)
    mode_safe = sanitize_component(mode)
    filename = f"{mode_safe}_{language_safe}_{provider_safe}_{model_safe}.json"
    return json_dir / filename

def _run_and_collect(
    config: BenchmarkConfig,
    run_fn,
) -> Dict[str, Any]:
    run_artifacts = run_fn(config) or {}
    run_dir = run_artifacts.get("run_dir")
    if run_dir is None:
        raise RuntimeError("Runner did not return a run directory."
                           " Ensure run(config) returns artifact metadata.")
    return {
        "run_dir": Path(run_dir),
        "manifest": Path(run_artifacts.get("manifest", Path(run_dir) / "manifest.yaml")),
        "metrics": run_artifacts.get("metrics", {}),
        "elapsed_seconds": run_artifacts.get("elapsed_seconds"),
    }


def run_batch_benchmark(
    yaml_path: str | Path,
    cli_overrides: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Run benchmarks for all combinations defined in a YAML config."""

    from raw.runner import run as run_raw
    from agentic.runner.main import run as run_agentic

    cli_overrides = cli_overrides or {}
    yaml_data = load_yaml_config(yaml_path)
    run_plan = expand_run_matrix(yaml_data)

    artifacts_cfg = yaml_data.get("artifacts", {}) or {}
    outputs_root = Path(artifacts_cfg.get("root", "outputs"))
    json_dir = outputs_root / artifacts_cfg.get("json_dir", "json")
    plots_dir = outputs_root / artifacts_cfg.get("plots_dir", artifacts_cfg.get("plots", "plots"))

    yaml_data.setdefault("artifacts", {})
    yaml_data["artifacts"]["root"] = str(outputs_root)
    yaml_data["artifacts"]["json_dir"] = str(json_dir)
    yaml_data["artifacts"]["plots_dir"] = str(plots_dir)
    viz_defaults = yaml_data.setdefault("visualization", {})
    if not isinstance(viz_defaults, dict):
        viz_defaults = {}
        yaml_data["visualization"] = viz_defaults
    viz_defaults.setdefault("enabled", True)

    total_runs = len(run_plan)
    print(f"\n{'=' * 70}")
    print("KernelBench-v3 Batch Runner")
    print(f"{'=' * 70}")
    print(f"Running {total_runs} target(s) from {yaml_path}")
    print(f"{'=' * 70}\n")

    _verify_provider_models(run_plan, yaml_data, cli_overrides)

    results: List[Dict[str, Any]] = []

    mode_priority = {"raw": 0, "agentic": 1}
    language_priority = {"cuda": 0, "triton": 1}

    def _sort_key(entry: tuple[str, str, Dict[str, Any]]) -> tuple[int, int, str, str]:
        mode_key = mode_priority.get(entry[0], 99)
        language_key = language_priority.get(entry[1], 99)
        provider = str(entry[2].get("provider", ""))
        model = str(entry[2].get("model", ""))
        return (mode_key, language_key, provider, model)

    run_plan.sort(key=_sort_key)

    for index, (mode, language, model_entry) in enumerate(run_plan, start=1):
        provider = model_entry.get("provider", yaml_data.get("provider", "unknown"))
        model_id = model_entry.get("model", yaml_data.get("generator_model", "unknown"))
        header = f"{mode.lower()} + {language.lower()} + {provider}/{model_id}"
        print(f"{GREEN}{header}{RESET}")

        config = yaml_to_benchmark_config(yaml_data, model_entry, mode, language, cli_overrides)
        fingerprint = fingerprint_config(config)

        cache_path = json_cache_path(json_dir, mode, language, provider, model_id)
        cached_payload = load_metrics(cache_path)

        if cached_payload and cached_payload.get("config_fingerprint") == fingerprint:
            metrics = cached_payload.get("metrics", {})
            artifacts = cached_payload.get("artifacts", {})
            print("  ↪ Using cached metrics (config fingerprint match).")
            results.append({
                "provider": provider,
                "model": model_id,
                "mode": mode,
                "language": language,
                "status": "cached",
                "metrics": metrics,
                "metrics_path": str(cache_path),
                "artifacts": artifacts,
                "timestamp": cached_payload.get("timestamp"),
                "config_fingerprint": fingerprint,
                "elapsed_seconds": cached_payload.get("elapsed_seconds"),
            })
            print(f"✅ Cached: {provider}/{model_id}\n")
            continue

        try:
            if mode == "raw":
                os.environ.setdefault("MAX_JOBS", str(config.raw_max_jobs))
                artifact_info = _run_and_collect(config, run_raw)
            elif mode == "agentic":
                artifact_info = _run_and_collect(config, run_agentic)
            else:
                raise ValueError(f"Unsupported mode '{mode}' in run matrix.")

            metrics_bundle = artifact_info.get("metrics", {})
            payload = {
                "provider": provider,
                "model": model_id,
                "mode": mode,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_bundle,
                "config_fingerprint": fingerprint,
                "artifacts": {
                    "manifest": str(artifact_info.get("manifest")),
                    "run_dir": str(artifact_info["run_dir"]),
                },
                "elapsed_seconds": artifact_info.get("elapsed_seconds"),
            }
            save_metrics(cache_path, payload)

            results.append({
                **payload,
                "status": "completed",
                "metrics_path": str(cache_path),
            })
            print(f"✅ Completed: {provider}/{model_id}")

        except Exception as exc:  # noqa: BLE001
            print(f"❌ Failed: {provider}/{model_id}")
            print(f"   Error: {exc}\n")
            results.append({
                "provider": provider,
                "model": model_id,
                "mode": mode,
                "language": language,
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            })
            continue

        print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = json_dir / f"batch_summary_{timestamp}.json"
    json_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "config": str(yaml_path),
        "total_targets": total_runs,
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "cached": sum(1 for r in results if r["status"] == "cached"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
        "generated_at": datetime.now().isoformat(),
        "artifacts": {
            "json_dir": str(json_dir),
            "plots_dir": str(plots_dir),
        },
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    print(f"\n{'=' * 70}")
    print("Batch run complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 70}\n")

    viz_config = yaml_data.get("visualization", {})
    if viz_config.get("enabled", False):
        print("Generating visualizations...")
        from visualization import generate_comparison_charts

        generate_comparison_charts(
            yaml_data,
            results,
        )

    return results
