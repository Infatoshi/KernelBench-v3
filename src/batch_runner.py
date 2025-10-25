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

from config import AgenticConfig, BenchmarkConfig, HardwareConfig, ProblemSetConfig
from metrics import compute_core_metrics, load_metrics, parse_jsonl_results, save_metrics
from providers import verify_model_responds_hello


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
        legacy_mode = yaml_data.get("mode", "raw")
        modes = [legacy_mode]

    languages = yaml_data.get("languages")
    if not languages:
        legacy_language = yaml_data.get("language", "triton")
        languages = [legacy_language]

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
    hw_data = yaml_data.get("hardware", {})
    return HardwareConfig(
        gpu_architecture=hw_data.get("gpu_architecture", "Ampere"),
        gpu_id=hw_data.get("gpu_id", 0),
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
            ag_defaults.get("reflector_model", "gpt-4-turbo"),
        ),
        "optimizer_model": ag_data.get(
            "optimizer_model",
            ag_defaults.get("optimizer_model", "gpt-4-turbo"),
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
    raw_defaults = defaults.get("raw", {})

    provider = model_entry.get("provider") or yaml_data.get("provider", "openai")
    model_id = model_entry.get("model") or yaml_data.get("generator_model", "gpt-4-turbo")
    base_url = model_entry.get("base_url") or yaml_data.get("provider_base_url")
    formatter_defaults = defaults.get("formatter", {})
    formatter_cfg = model_entry.get("formatter", {})
    formatter_provider = formatter_cfg.get(
        "provider",
        yaml_data.get("formatter_provider", formatter_defaults.get("provider")),
    )
    formatter_model = formatter_cfg.get(
        "model",
        yaml_data.get("formatter_model", formatter_defaults.get("model")),
    )
    formatter_base_url = formatter_cfg.get(
        "base_url",
        yaml_data.get("formatter_base_url", formatter_defaults.get("base_url")),
    )

    num_runs = model_entry.get("num_runs", yaml_data.get("num_runs", defaults.get("num_runs")))
    if "num_runs" in cli_overrides:
        num_runs = cli_overrides["num_runs"]
    profile_stages = model_entry.get(
        "profile_stages",
        yaml_data.get("profile_stages", defaults.get("profile_stages", False)),
    )
    if cli_overrides.get("profile_stages") is True:
        profile_stages = True

    raw_concurrency = model_entry.get(
        "raw_concurrency",
        yaml_data.get("raw_concurrency", raw_defaults.get("cpu_concurrency", 8)),
    )
    raw_gpu_concurrency = model_entry.get(
        "raw_gpu_concurrency",
        yaml_data.get("raw_gpu_concurrency", raw_defaults.get("gpu_concurrency", 1)),
    )
    raw_max_jobs = model_entry.get(
        "raw_max_jobs",
        yaml_data.get("raw_max_jobs", raw_defaults.get("max_jobs", 8)),
    )

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
        verbose=yaml_data.get("verbose", False),
        num_runs=num_runs,
        profile_stages=profile_stages,
        hardware=hardware,
        problems=problems,
        agentic=agentic_cfg,
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


def collect_metrics(results_path: Path) -> Dict[str, Any]:
    records = parse_jsonl_results(results_path)
    metrics = compute_core_metrics(records)
    return {
        "metrics": metrics,
        "total_records": len(records),
    }


def _run_and_collect(
    config: BenchmarkConfig,
    run_fn,
) -> Dict[str, Any]:
    run_artifacts = run_fn(config) or {}
    results_path = run_artifacts.get("results_path")
    if results_path is None:
        raise RuntimeError("Runner did not return a results_path."
                           " Ensure run(config) returns artifact metadata.")
    return {
        "results_path": Path(results_path),
        "run_dir": Path(run_artifacts.get("run_dir", Path(results_path).parent)),
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

    artifacts_cfg = yaml_data.get("artifacts", {})
    json_dir = Path(artifacts_cfg.get("json_dir", "json"))
    plots_dir = Path(artifacts_cfg.get("plots_dir", artifacts_cfg.get("plots", "plots")))

    yaml_data.setdefault("artifacts", {})
    yaml_data["artifacts"]["json_dir"] = str(json_dir)
    yaml_data["artifacts"]["plots_dir"] = str(plots_dir)

    total_runs = len(run_plan)
    print(f"\n{'=' * 70}")
    print("KernelBench-v3 Batch Runner")
    print(f"{'=' * 70}")
    print(f"Running {total_runs} target(s) from {yaml_path}")
    print(f"{'=' * 70}\n")

    _verify_provider_models(run_plan, yaml_data, cli_overrides)

    results: List[Dict[str, Any]] = []

    for index, (mode, language, model_entry) in enumerate(run_plan, start=1):
        provider = model_entry.get("provider", yaml_data.get("provider", "unknown"))
        model_id = model_entry.get("model", yaml_data.get("generator_model", "unknown"))

        print(f"[{index}/{total_runs}] {mode.upper()} | {language.upper()} :: {provider}/{model_id}")
        print("-" * 70)

        config = yaml_to_benchmark_config(yaml_data, model_entry, mode, language, cli_overrides)
        fingerprint = fingerprint_config(config)

        cache_path = json_cache_path(json_dir, mode, language, provider, model_id)
        cached_payload = load_metrics(cache_path)

        if cached_payload and cached_payload.get("config_fingerprint") == fingerprint:
            metrics = cached_payload.get("metrics", {})
            print("  ↪ Using cached metrics (config fingerprint match).")
            results.append({
                "provider": provider,
                "model": model_id,
                "mode": mode,
                "language": language,
                "status": "cached",
                "metrics": metrics,
                "metrics_path": str(cache_path),
                "results_path": cached_payload.get("artifacts", {}).get("results_jsonl"),
                "timestamp": cached_payload.get("timestamp"),
                "config_fingerprint": fingerprint,
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

            metrics_bundle = collect_metrics(artifact_info["results_path"])
            payload = {
                "provider": provider,
                "model": model_id,
                "mode": mode,
                "language": language,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_bundle["metrics"],
                "config_fingerprint": fingerprint,
                "artifacts": {
                    "results_jsonl": str(artifact_info["results_path"]),
                    "run_dir": str(artifact_info["run_dir"]),
                },
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
