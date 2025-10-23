from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure local src package is importable when run as script
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import BenchmarkConfig
from providers import verify_model_responds_hello


def load_config() -> BenchmarkConfig:
    """Load BenchmarkConfig instance from config.py."""
    # config.py is executed when imported; simply instantiate BenchmarkConfig
    return BenchmarkConfig()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelBench-v3 unified benchmark entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch run from YAML config
  python eval.py --config configs/example_batch.yaml
  
  # Quick test config
  python eval.py --config configs/quick_test.yaml
  
  # Single model run (legacy mode)
  python eval.py --mode raw --provider groq --model llama-3.3-70b-versatile --num-runs 5
        """
    )
    
    # Primary mode: YAML config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file for batch runs",
    )
    
    # Legacy single-run flags (for backwards compatibility)
    parser.add_argument(
        "--mode",
        choices=["raw", "agentic"],
        help="Evaluation mode (only used if --config not provided)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="LLM provider (only used if --config not provided)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model identifier (only used if --config not provided)",
    )
    parser.add_argument(
        "--formatter-provider",
        type=str,
        dest="formatter_provider",
        help="Optional provider to post-process model outputs into the KernelBench contract.",
    )
    parser.add_argument(
        "--formatter-model",
        type=str,
        dest="formatter_model",
        help="Model identifier for the formatter provider.",
    )
    parser.add_argument(
        "--formatter-base-url",
        type=str,
        dest="formatter_base_url",
        help="Override base URL for the formatter provider (OpenAI-compatible APIs).",
    )
    parser.add_argument(
        "--groq-formatter-beta",
        action="store_true",
        help="Enable the Groq moonshotai/kimi-k2-instruct-0905 formatter beta to enforce structured output.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        dest="base_url",
        help="Override base URL for provider API",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--num-runs",
        type=str,
        default="all",
        help="Number of problems to evaluate or 'all'",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Collect per-stage timing during evaluation",
    )
    
    cli_args = parser.parse_args()

    # Batch mode: run from YAML config
    if cli_args.config:
        from batch_runner import run_batch_benchmark

        overrides: dict[str, Any] = {}
        if cli_args.profile_stages:
            overrides["profile_stages"] = True
        if cli_args.num_runs and cli_args.num_runs != "all":
            overrides["num_runs"] = cli_args.num_runs
        if cli_args.formatter_provider:
            overrides["formatter_provider"] = cli_args.formatter_provider
        if cli_args.formatter_model:
            overrides["formatter_model"] = cli_args.formatter_model
        if cli_args.formatter_base_url:
            overrides["formatter_base_url"] = cli_args.formatter_base_url
        if cli_args.groq_formatter_beta:
            overrides["formatter_provider"] = "groq"
            overrides["formatter_model"] = "moonshotai/kimi-k2-instruct-0905"
            overrides.setdefault("formatter_base_url", None)

        run_batch_benchmark(cli_args.config, overrides)
        return
    
    # Single-run mode: use CLI flags
    config = load_config()
    if cli_args.mode:
        config.mode = cli_args.mode  # type: ignore[assignment]
    if cli_args.provider:
        config.provider = cli_args.provider  # type: ignore[assignment]
    if cli_args.model:
        config.generator_model = cli_args.model  # type: ignore[assignment]
    if cli_args.base_url:
        config.provider_base_url = cli_args.base_url  # type: ignore[assignment]
    if cli_args.formatter_provider:
        config.formatter_provider = cli_args.formatter_provider  # type: ignore[assignment]
    if cli_args.formatter_model:
        config.formatter_model = cli_args.formatter_model  # type: ignore[assignment]
    if cli_args.formatter_base_url:
        config.formatter_base_url = cli_args.formatter_base_url  # type: ignore[assignment]
    if cli_args.groq_formatter_beta:
        config.formatter_provider = "groq"  # type: ignore[assignment]
        config.formatter_model = "moonshotai/kimi-k2-instruct-0905"  # type: ignore[assignment]
        if not config.formatter_base_url:
            config.formatter_base_url = None  # type: ignore[assignment]
    if cli_args.verbose:
        config.verbose = True  # type: ignore[assignment]
    config.num_runs = cli_args.num_runs  # type: ignore[assignment]
    if cli_args.profile_stages:
        config.profile_stages = True  # type: ignore[assignment]

    try:
        verify_model_responds_hello(
            config.provider,
            config.generator_model,
            config.provider_base_url,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Preflight] Provider/model validation failed: {exc}")
        sys.exit(1)

    if config.mode == "raw":
        from raw.runner import run as run_raw

        run_raw(config)
    elif config.mode == "agentic":
        from agentic.runner.main import run as run_agentic

        run_agentic(config)
    else:
        raise ValueError(f"Unsupported mode '{config.mode}'")


if __name__ == "__main__":
    main()
