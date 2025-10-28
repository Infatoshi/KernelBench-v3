"""Visualization tools for KernelBench-v3 results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import json
import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: uv pip install matplotlib")

from metrics import compute_core_metrics, load_metrics, parse_jsonl_results


def parse_jsonl_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Parse JSONL results file."""
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute benchmark metrics from results."""
    total = len(results)
    if total == 0:
        return {}
    
    compiled = sum(1 for r in results if r.get("compiled", False))
    correct = sum(1 for r in results if r.get("correctness", False))
    
    # Speedup metrics
    runtimes = [r.get("runtime") for r in results if r.get("correctness") and r.get("runtime")]
    speedups = []
    
    # Count fast_1, fast_2, fast_3 (kernels faster than baseline)
    # Note: In fake mode, runtime is None, so we just count correct ones
    fast_1 = sum(1 for r in results if r.get("correctness") and (r.get("runtime") or 0) > 0)
    fast_2 = sum(1 for r in results if r.get("correctness") and (r.get("runtime") or 0) > 0)  # placeholder
    
    metrics = {
        "total_problems": total,
        "compilation_rate": (compiled / total * 100) if total > 0 else 0,
        "correctness_rate": (correct / compiled * 100) if compiled > 0 else 0,
        "correct_count": correct,
        "compiled_count": compiled,
        "fast_1_rate": (fast_1 / total * 100) if total > 0 else 0,
        "mean_runtime": sum(runtimes) / len(runtimes) if runtimes else 0,
    }

    return metrics


def _load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path or not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except (yaml.YAMLError, OSError):
        return None


def _metrics_from_manifest(kernels: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not kernels:
        return None

    total = len(kernels)
    compiled = 0
    correct = 0
    fast_1 = 0
    runtimes: List[float] = []

    for kernel in kernels:
        compiled_flag = kernel.get("compiled")
        if compiled_flag is None:
            compiled_flag = kernel.get("call_pass")
        correct_flag = kernel.get("correctness")
        if correct_flag is None:
            correct_flag = kernel.get("exe_pass")
        fast_flag = kernel.get("fast_1")
        if fast_flag is None:
            fast_flag = kernel.get("perf_pass")

        runtime = kernel.get("runtime_ms")
        if runtime is None:
            runtime = kernel.get("latency_ms")

        if compiled_flag:
            compiled += 1
        if correct_flag:
            correct += 1
        if fast_flag:
            fast_1 += 1
        if runtime and runtime > 0:
            try:
                runtimes.append(float(runtime))
            except (TypeError, ValueError):
                continue

    metrics = {
        "total_problems": total,
        "compiled_count": compiled,
        "correct_count": correct,
        "compilation_rate": (compiled / total * 100) if total else 0.0,
        "correctness_rate": (correct / compiled * 100) if compiled else 0.0,
        "fast_1_rate": (fast_1 / total * 100) if total else 0.0,
        "mean_runtime": (sum(runtimes) / len(runtimes)) if runtimes else 0.0,
    }

    return metrics
def sanitize_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def ensure_agentic_results(provider: str, model: str) -> Optional[Path]:
    """Return path to agentic results.jsonl, migrating legacy locations if needed."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None

    provider_safe = sanitize_component(provider)
    model_safe = sanitize_component(model.replace("/", "_"))
    run_dir = outputs_dir / f"agentic_{provider_safe}_{model_safe}"
    latest_path = run_dir / "results.jsonl"

    if latest_path.exists():
        return latest_path

    if run_dir.exists():
        timestamped = sorted(run_dir.glob("results_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if timestamped:
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(timestamped[0], latest_path)
            except Exception as exc:
                print(f"  Warning: failed to refresh latest results for {provider}/{model}: {exc}")
                return timestamped[0]
            return latest_path

    legacy_candidates = sorted(
        outputs_dir.glob(f"agentic_{provider_safe}_{model_safe}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if legacy_candidates:
        run_dir.mkdir(parents=True, exist_ok=True)
        latest = legacy_candidates[0]
        legacy_copy = run_dir / latest.name
        if not legacy_copy.exists():
            try:
                shutil.copy2(latest, legacy_copy)
            except Exception as exc:
                print(f"  Warning: failed to migrate legacy results for {provider}/{model}: {exc}")
                return latest
        try:
            shutil.copy2(legacy_copy, latest_path)
        except Exception as exc:
            print(f"  Warning: failed to create latest results for {provider}/{model}: {exc}")
            return legacy_copy
        return latest_path

    return None


def _cache_path(json_dir: Path, mode: str, language: str, provider: str, model: str) -> Path:
    provider_safe = sanitize_component(provider or "unknown")
    model_safe = sanitize_component((model or "unknown").replace("/", "_"))
    mode_safe = sanitize_component(mode or "raw")
    language_safe = sanitize_component(language or "cuda")
    return json_dir / f"{mode_safe}_{language_safe}_{provider_safe}_{model_safe}.json"


def collect_metrics_for_visualization(
    yaml_config: Dict[str, Any],
    batch_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    artifacts_cfg = yaml_config.get("artifacts", {})
    json_dir = Path(artifacts_cfg.get("json_dir", "json"))

    metrics_records: List[Dict[str, Any]] = []

    for entry in batch_results:
        status = entry.get("status", "unknown")
        provider = entry.get("provider", "unknown")
        model = entry.get("model", "unknown")
        mode = entry.get("mode", yaml_config.get("mode", "raw"))
        language = entry.get("language", yaml_config.get("language", "cuda"))

        metrics = entry.get("metrics")
        metrics_loaded = False
        if not metrics and status in {"completed", "cached"}:
            metrics_path = entry.get("metrics_path")
            payload = load_metrics(Path(metrics_path)) if metrics_path else None
            if payload is None:
                cache_candidate = _cache_path(json_dir, mode, language, provider, model)
                payload = load_metrics(cache_candidate)
            if payload:
                metrics = payload.get("metrics")
                metrics_loaded = metrics is not None

        artifacts = entry.get("artifacts") or {}
        manifest_path = artifacts.get("manifest")
        if not metrics and manifest_path:
            manifest = _load_manifest(Path(manifest_path))
            if manifest:
                manifest_metrics = _metrics_from_manifest(manifest.get("kernels", []))
                if manifest_metrics:
                    metrics = manifest_metrics
                    metrics_loaded = True

        if not metrics and status in {"completed", "cached"}:
            results_path = entry.get("results_path")
            records = []
            if results_path:
                rp = Path(results_path)
                if rp.exists():
                    records = parse_jsonl_results(rp)
            elif mode == "agentic":
                agentic_path = ensure_agentic_results(provider, model)
                if agentic_path:
                    records = parse_jsonl_results(agentic_path)

            if records and not metrics_loaded:
                metrics = compute_core_metrics(records)
                metrics_loaded = metrics is not None

        if not metrics:
            if status in {"completed", "cached"} and not metrics_loaded:
                print(f"  Warning: Missing metrics for {provider}/{model} [{mode}/{language}]")
            total = entry.get("evaluated_problems") or 0
            metrics = {
                "total_problems": total,
                "compiled_count": 0,
                "correct_count": 0,
                "compilation_rate": 0.0,
                "correctness_rate": 0.0,
                "fast_1_rate": 0.0,
                "mean_runtime": 0.0,
            }

        label = f"{mode}-{language} :: {provider}/{model}"
        if status not in {"completed", "cached"}:
            label = f"{label} [{status.upper()}]"

        metrics_records.append(
            {
                "model": label,
                "provider": provider,
                "model_name": model,
                "mode": mode,
                "language": language,
                "status": status,
                **metrics,
            }
        )

    return metrics_records


def generate_comparison_charts(yaml_config: Dict[str, Any], batch_results: List[Dict[str, Any]]) -> None:
    """Generate comparison visualizations for batch benchmark results."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization: matplotlib not installed")
        return
    
    viz_config = yaml_config.get("visualization", {})
    artifacts_cfg = yaml_config.get("artifacts", {})

    plots_dir = Path(artifacts_cfg.get("plots_dir", viz_config.get("output_dir", "plots")))
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = collect_metrics_for_visualization(yaml_config, batch_results)

    if not model_metrics:
        print("No results found for visualization")
        return

    generate_bar_chart(model_metrics, plots_dir, viz_config)
    generate_summary_table(model_metrics, plots_dir)

    print(f"✅ Visualizations saved to: {plots_dir}")


def _infer_metrics_to_plot(model_metrics: List[Dict[str, Any]]) -> List[str]:
    """Return sorted metric keys suitable for visualization."""
    numeric_keys: List[str] = []
    excluded = {"model", "provider", "model_name", "mode", "language", "status"}
    for record in model_metrics:
        for key, value in record.items():
            if key in excluded:
                continue
            if isinstance(value, (int, float)):
                if key not in numeric_keys:
                    numeric_keys.append(key)
    return numeric_keys


def generate_bar_chart(model_metrics: List[Dict[str, Any]], output_dir: Path, viz_config: Dict[str, Any]) -> None:
    """Generate bar chart comparing models across key metrics."""
    
    if not model_metrics:
        return
    
    configured_metrics = viz_config.get("metrics")
    if configured_metrics:
        metrics_to_plot = configured_metrics
    else:
        metrics_to_plot = _infer_metrics_to_plot(model_metrics)
    if not metrics_to_plot:
        return
    
    models = [m["model"] for m in model_metrics]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for metric_name in metrics_to_plot:
        values = [float(m.get(metric_name, 0.0)) for m in model_metrics]
        fig, ax = plt.subplots(figsize=(max(6, 2 + len(models)), 6))

        ax.bar(range(len(models)), values, color="steelblue")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

        if metric_name.endswith("_rate") or metric_name.endswith("_pct"):
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim([0, 100])
        else:
            ax.set_ylabel(metric_name.replace("_", " ").title())

        for idx, value in enumerate(values):
            if ax.get_ylabel() == "Percentage (%)":
                label = f"{value:.1f}%"
                offset = 2
            else:
                label = f"{value:.2f}"
                offset = max(0.05 * max(values) if values else 0.1, 0.1)
            ax.text(idx, value + offset, label, ha="center", va="bottom")

        plt.tight_layout()
        output_path = output_dir / f"benchmark_{metric_name}_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")


def generate_summary_table(model_metrics: List[Dict[str, Any]], output_dir: Path) -> None:
    """Generate markdown summary table of results."""
    
    if not model_metrics:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"summary_table_{timestamp}.md"
    
    with open(output_path, "w") as f:
        f.write("# KernelBench-v3 Results Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Table header
        f.write("| Model | Total | Compiled | Correct | Compile% | Correct% | Fast@1% |\n")
        f.write("|-------|-------|----------|---------|----------|----------|----------|\n")
        
        # Table rows
        for m in model_metrics:
            f.write(f"| {m['model']} ")
            f.write(f"| {m.get('total_problems', 0)} ")
            f.write(f"| {m.get('compiled_count', 0)} ")
            f.write(f"| {m.get('correct_count', 0)} ")
            f.write(f"| {m.get('compilation_rate', 0):.1f}% ")
            f.write(f"| {m.get('correctness_rate', 0):.1f}% ")
            f.write(f"| {m.get('fast_1_rate', 0):.1f}% |\n")
    
    print(f"  Saved summary table: {output_path}")
