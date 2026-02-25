#!/usr/bin/env python3
"""Validate KernelBench problem baselines across CPU, CUDA (Modal), and Metal (SSH)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from src.config.benchmark_problems import (
    BENCHMARK_PROBLEMS,
    find_problems_for_benchmark,
    get_problem_hardware_required,
)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
NVIDIA_GPUS = {"RTX3090", "H100", "B200", "A100", "L40S", "LOCAL"}
REMOTE_RESULT_PREFIX = "RESULT_JSON:"


@dataclass
class ValidationResult:
    """Per-problem baseline validation result."""

    platform: str
    problem: str
    passed: bool
    error: str = ""
    duration_ms: float = 0.0


def _normalize_problem_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _all_problem_paths() -> list[Path]:
    preferred = [
        "level1",
        "level2",
        "level3",
        "level4",
        "graphics",
        "tile_specialized",
        "cutile",
    ]
    discovered = {d for cfg in BENCHMARK_PROBLEMS.values() for d in cfg["dirs"]}
    dirs = [d for d in preferred if d in discovered]
    dirs.extend(sorted(discovered - set(dirs)))
    paths: list[Path] = []
    for dirname in dirs:
        for path in sorted((PROJECT_ROOT / "KernelBench" / dirname).glob("*.py")):
            if path.name.startswith("_"):
                continue
            paths.append(path)
    return paths


def _cuda_problem_paths(all_paths: Iterable[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in all_paths:
        required = get_problem_hardware_required(path)
        if required is None:
            selected.append(path)
            continue
        if any(gpu in NVIDIA_GPUS for gpu in required):
            selected.append(path)
    return selected


def _metal_problem_paths() -> list[Path]:
    metal = find_problems_for_benchmark(PROJECT_ROOT, benchmark="metal", levels=[1, 2, 3, 4])
    return [path for _, path in metal]


def _load_problem_module(path: Path):
    module_name = f"baseline_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _find_invalid_output(value: Any, prefix: str = "output") -> str | None:
    if isinstance(value, torch.Tensor):
        if torch.isnan(value).any():
            return f"{prefix} contains NaN"
        if torch.isinf(value).any():
            return f"{prefix} contains Inf"
        return None
    if isinstance(value, list):
        for i, child in enumerate(value):
            err = _find_invalid_output(child, f"{prefix}[{i}]")
            if err:
                return err
        return None
    if isinstance(value, tuple):
        for i, child in enumerate(value):
            err = _find_invalid_output(child, f"{prefix}[{i}]")
            if err:
                return err
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            err = _find_invalid_output(child, f"{prefix}.{key}")
            if err:
                return err
        return None
    return None


def _deterministic_equal(lhs: Any, rhs: Any) -> bool:
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, torch.Tensor):
        return torch.equal(lhs, rhs)
    if isinstance(lhs, list):
        return len(lhs) == len(rhs) and all(_deterministic_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, tuple):
        return len(lhs) == len(rhs) and all(_deterministic_equal(a, b) for a, b in zip(lhs, rhs))
    if isinstance(lhs, dict):
        if lhs.keys() != rhs.keys():
            return False
        return all(_deterministic_equal(lhs[k], rhs[k]) for k in lhs)
    return lhs == rhs


def validate_problem(path: Path, device: torch.device, platform: str) -> ValidationResult:
    path = _normalize_problem_path(path)
    start = time.perf_counter()
    rel_path = path.relative_to(PROJECT_ROOT).as_posix()
    try:
        module = _load_problem_module(path)
        if not hasattr(module, "Model"):
            raise RuntimeError("Missing Model class")
        if not hasattr(module, "get_inputs"):
            raise RuntimeError("Missing get_inputs()")

        init_inputs = []
        if hasattr(module, "get_init_inputs"):
            init_inputs = module.get_init_inputs()
        if init_inputs is None:
            init_inputs = []
        if not isinstance(init_inputs, (list, tuple)):
            raise RuntimeError("get_init_inputs() must return list/tuple")

        raw_inputs = module.get_inputs()
        if not isinstance(raw_inputs, (list, tuple)):
            raise RuntimeError("get_inputs() must return list/tuple")
        if len(raw_inputs) == 0:
            raise RuntimeError("get_inputs() returned no inputs")

        model = module.Model(*init_inputs)
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model = model.to(device)

        inputs = [_move_to_device(v, device) for v in raw_inputs]

        with torch.no_grad():
            out1 = model(*inputs)
            _sync_device(device)
            out2 = model(*inputs)
            _sync_device(device)

        invalid_err = _find_invalid_output(out1, prefix="output")
        if invalid_err:
            raise RuntimeError(invalid_err)
        if not _deterministic_equal(out1, out2):
            raise RuntimeError("Non-deterministic output across repeated runs")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ValidationResult(platform=platform, problem=rel_path, passed=True, duration_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ValidationResult(
            platform=platform,
            problem=rel_path,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=elapsed_ms,
        )


def run_local_validation(
    problem_paths: list[Path], device: torch.device, platform: str
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    total = len(problem_paths)
    for index, path in enumerate(problem_paths, start=1):
        rel = _normalize_problem_path(path).relative_to(PROJECT_ROOT).as_posix()
        print(f"[{platform}] [{index}/{total}] {rel}", flush=True)
        results.append(validate_problem(path, device=device, platform=platform))
    return results


REMOTE_VALIDATOR_SCRIPT = textwrap.dedent(
    """
    import argparse
    import importlib.util
    import json
    import time
    from pathlib import Path
    from typing import Any

    import torch

    REMOTE_RESULT_PREFIX = "RESULT_JSON:"

    def _load_problem_module(path: Path):
        module_name = f"remote_baseline_{path.stem}_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [_move_to_device(v, device) for v in value]
        if isinstance(value, tuple):
            return tuple(_move_to_device(v, device) for v in value)
        if isinstance(value, dict):
            return {k: _move_to_device(v, device) for k, v in value.items()}
        return value

    def _sync_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    def _find_invalid_output(value: Any, prefix: str = "output") -> str | None:
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                return f"{prefix} contains NaN"
            if torch.isinf(value).any():
                return f"{prefix} contains Inf"
            return None
        if isinstance(value, list):
            for i, child in enumerate(value):
                err = _find_invalid_output(child, f"{prefix}[{i}]")
                if err:
                    return err
            return None
        if isinstance(value, tuple):
            for i, child in enumerate(value):
                err = _find_invalid_output(child, f"{prefix}[{i}]")
                if err:
                    return err
            return None
        if isinstance(value, dict):
            for key, child in value.items():
                err = _find_invalid_output(child, f"{prefix}.{key}")
                if err:
                    return err
            return None
        return None

    def _deterministic_equal(lhs: Any, rhs: Any) -> bool:
        if type(lhs) is not type(rhs):
            return False
        if isinstance(lhs, torch.Tensor):
            return torch.equal(lhs, rhs)
        if isinstance(lhs, list):
            return len(lhs) == len(rhs) and all(_deterministic_equal(a, b) for a, b in zip(lhs, rhs))
        if isinstance(lhs, tuple):
            return len(lhs) == len(rhs) and all(_deterministic_equal(a, b) for a, b in zip(lhs, rhs))
        if isinstance(lhs, dict):
            if lhs.keys() != rhs.keys():
                return False
            return all(_deterministic_equal(lhs[k], rhs[k]) for k in lhs)
        return lhs == rhs

    def validate_problem(path: Path, device: torch.device, platform: str) -> dict:
        start = time.perf_counter()
        try:
            module = _load_problem_module(path)
            if not hasattr(module, "Model"):
                raise RuntimeError("Missing Model class")
            if not hasattr(module, "get_inputs"):
                raise RuntimeError("Missing get_inputs()")

            init_inputs = []
            if hasattr(module, "get_init_inputs"):
                init_inputs = module.get_init_inputs()
            if init_inputs is None:
                init_inputs = []
            if not isinstance(init_inputs, (list, tuple)):
                raise RuntimeError("get_init_inputs() must return list/tuple")

            raw_inputs = module.get_inputs()
            if not isinstance(raw_inputs, (list, tuple)):
                raise RuntimeError("get_inputs() must return list/tuple")
            if len(raw_inputs) == 0:
                raise RuntimeError("get_inputs() returned no inputs")

            model = module.Model(*init_inputs)
            if hasattr(model, "eval"):
                model.eval()
            if hasattr(model, "to"):
                model = model.to(device)

            inputs = [_move_to_device(v, device) for v in raw_inputs]
            with torch.no_grad():
                out1 = model(*inputs)
                _sync_device(device)
                out2 = model(*inputs)
                _sync_device(device)

            invalid_err = _find_invalid_output(out1, prefix="output")
            if invalid_err:
                raise RuntimeError(invalid_err)
            if not _deterministic_equal(out1, out2):
                raise RuntimeError("Non-deterministic output across repeated runs")

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "platform": platform,
                "problem": str(path),
                "passed": True,
                "error": "",
                "duration_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "platform": platform,
                "problem": str(path),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": elapsed_ms,
            }

    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--platform", required=True, choices=["cuda", "metal"])
        parser.add_argument("--manifest", required=True)
        args = parser.parse_args()

        with open(args.manifest, "r", encoding="utf-8") as f:
            problem_paths = json.load(f)["problem_paths"]

        if args.platform == "cuda":
            if not torch.cuda.is_available():
                payload = {
                    "platform": "cuda",
                    "results": [],
                    "error": "CUDA not available",
                }
                print(REMOTE_RESULT_PREFIX + json.dumps(payload))
                return
            device = torch.device("cuda")
        else:
            if not torch.backends.mps.is_available():
                payload = {
                    "platform": "metal",
                    "results": [],
                    "error": "MPS not available",
                }
                print(REMOTE_RESULT_PREFIX + json.dumps(payload))
                return
            device = torch.device("mps")

        results = []
        total = len(problem_paths)
        for index, problem_path in enumerate(problem_paths, start=1):
            print(f"[{args.platform}] [{index}/{total}] {problem_path}", flush=True)
            results.append(validate_problem(Path(problem_path), device=device, platform=args.platform))

        payload = {"platform": args.platform, "results": results}
        print(REMOTE_RESULT_PREFIX + json.dumps(payload))

    if __name__ == "__main__":
        main()
    """
).strip()


def _copy_problem_files_to_sandbox(sandbox: Any, problem_paths: list[Path]) -> None:
    for path in problem_paths:
        path = _normalize_problem_path(path)
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        content = path.read_text(encoding="utf-8")
        ok = sandbox.write_file(rel, content)
        if not ok:
            raise RuntimeError(f"Failed to copy problem file to remote sandbox: {rel}")


def _parse_remote_payload(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        if line.startswith(REMOTE_RESULT_PREFIX):
            return json.loads(line[len(REMOTE_RESULT_PREFIX) :])
    raise RuntimeError(f"Remote validator output did not contain {REMOTE_RESULT_PREFIX}")


def run_cuda_validation(problem_paths: list[Path], gpu: str) -> list[ValidationResult]:
    from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig

    manifest = {
        "problem_paths": [
            _normalize_problem_path(path).relative_to(PROJECT_ROOT).as_posix() for path in problem_paths
        ],
    }
    sandbox = ModalSandbox(problem_code="# baseline validation", config=ModalSandboxConfig(gpu=gpu))
    try:
        sandbox.start()
        _copy_problem_files_to_sandbox(sandbox, problem_paths)
        sandbox.write_file("_validate_baselines_remote.py", REMOTE_VALIDATOR_SCRIPT)
        sandbox.write_file("_validate_baselines_manifest.json", json.dumps(manifest))
        cmd = "python _validate_baselines_remote.py --platform cuda --manifest _validate_baselines_manifest.json"
        remote = sandbox.run_command(cmd, timeout=7200)
        if remote["returncode"] != 0:
            raise RuntimeError(f"Remote CUDA validation failed: {remote['stderr']}")
        payload = _parse_remote_payload(remote["stdout"])
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return [
            ValidationResult(
                platform="cuda",
                problem=item["problem"],
                passed=item["passed"],
                error=item.get("error", ""),
                duration_ms=float(item.get("duration_ms", 0.0)),
            )
            for item in payload["results"]
        ]
    finally:
        sandbox.stop()


def run_metal_validation(problem_paths: list[Path], ssh_host: str) -> list[ValidationResult]:
    from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig

    remote_home = _resolve_remote_home_dir(ssh_host)
    metalbench_dir = f"{remote_home}/MetalBench"

    manifest = {
        "problem_paths": [
            _normalize_problem_path(path).relative_to(PROJECT_ROOT).as_posix() for path in problem_paths
        ],
    }
    sandbox = MetalSandbox(
        problem_code="# baseline validation",
        config=MetalSandboxConfig(ssh_host=ssh_host, workdir=metalbench_dir, cleanup=False),
    )
    try:
        sandbox.start()
        setup_info = _ensure_metalbench_uv_environment(sandbox)
        print(
            "[METAL] setup action="
            f"{setup_info['action']} torch={setup_info['torch']} "
            f"mlx={setup_info['mlx']} numpy={setup_info['numpy']} "
            f"mlx_device={setup_info['mlx_device']}"
        )

        _copy_problem_files_to_sandbox(sandbox, problem_paths)
        sandbox.write_file("_validate_baselines_remote.py", REMOTE_VALIDATOR_SCRIPT)
        sandbox.write_file("_validate_baselines_manifest.json", json.dumps(manifest))
        cmd = (
            'export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH" && '
            "uv run python _validate_baselines_remote.py --platform metal "
            "--manifest _validate_baselines_manifest.json"
        )
        remote = sandbox.run_command(cmd, timeout=7200)
        if remote["returncode"] != 0:
            raise RuntimeError(f"Remote Metal validation failed: {remote['stderr']}")
        payload = _parse_remote_payload(remote["stdout"])
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        return [
            ValidationResult(
                platform="metal",
                problem=item["problem"],
                passed=item["passed"],
                error=item.get("error", ""),
                duration_ms=float(item.get("duration_ms", 0.0)),
            )
            for item in payload["results"]
        ]
    finally:
        sandbox.stop()


def _resolve_remote_home_dir(ssh_host: str) -> str:
    completed = subprocess.run(
        ["ssh", ssh_host, 'printf %s "$HOME"'],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to resolve remote home directory: {completed.stderr.strip()}")
    remote_home = completed.stdout.strip()
    if not remote_home:
        raise RuntimeError("Failed to resolve remote home directory: empty output")
    return remote_home


def _ensure_metalbench_uv_environment(sandbox: Any) -> dict[str, str]:
    setup_script = """
set -euo pipefail
cd /workspace
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "METALBENCH_SETUP_ERROR=uv_not_found_after_bootstrap"
  exit 2
fi

if [ ! -f pyproject.toml ]; then
  uv init --name metalbench --no-readme >/dev/null 2>&1
fi

SETUP_ACTION="cached"
if [ ! -d .venv ]; then
  SETUP_ACTION="created"
fi

if ! uv run python - <<'PY' >/dev/null 2>&1
import torch
import numpy
import mlx.core as mx
PY
then
  SETUP_ACTION="installed"
fi

if [ "${SETUP_ACTION}" != "cached" ]; then
  uv add torch mlx numpy >/dev/null
fi

echo "METALBENCH_SETUP_ACTION=${SETUP_ACTION}"
uv run python - <<'PY'
import importlib.metadata as md
import numpy as np
import torch
import mlx.core as mx
print("METALBENCH_TORCH_VERSION=" + torch.__version__)
print("METALBENCH_MLX_VERSION=" + md.version("mlx"))
print("METALBENCH_NUMPY_VERSION=" + np.__version__)
print("METALBENCH_MLX_DEVICE=" + str(mx.default_device()))
PY
"""
    result = sandbox.run_command(setup_script, timeout=3600)
    if result["returncode"] != 0:
        stderr = result["stderr"].strip() or result["stdout"].strip()
        raise RuntimeError(f"Failed to setup ~/MetalBench uv environment: {stderr}")

    info: dict[str, str] = {
        "action": "unknown",
        "torch": "unknown",
        "mlx": "unknown",
        "numpy": "unknown",
        "mlx_device": "unknown",
    }
    for raw_line in result["stdout"].splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("METALBENCH_SETUP_ACTION="):
            info["action"] = line.split("=", 1)[1]
        elif line.startswith("METALBENCH_TORCH_VERSION="):
            info["torch"] = line.split("=", 1)[1]
        elif line.startswith("METALBENCH_MLX_VERSION="):
            info["mlx"] = line.split("=", 1)[1]
        elif line.startswith("METALBENCH_NUMPY_VERSION="):
            info["numpy"] = line.split("=", 1)[1]
        elif line.startswith("METALBENCH_MLX_DEVICE="):
            info["mlx_device"] = line.split("=", 1)[1]
    return info


def _print_summary(platform: str, results: list[ValidationResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    print(f"\n[{platform.upper()}] total={total} passed={passed} failed={failed}")
    for result in results:
        if result.passed:
            continue
        print(f"  - {result.problem}: {result.error}")


def _slice_problems(paths: list[Path], max_problems: int | None) -> list[Path]:
    if max_problems is None:
        return paths
    return paths[:max_problems]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate baseline problems across CPU/CUDA/Metal")
    parser.add_argument(
        "--platforms",
        type=str,
        default="cpu,cuda,metal",
        help="Comma-separated platforms to run: cpu,cuda,metal",
    )
    parser.add_argument("--cuda-gpu", type=str, default="H100", help="Modal GPU for CUDA validation")
    parser.add_argument("--metal-host", type=str, default="macbook", help="SSH host for Metal validation")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems per platform")
    parser.add_argument("--json-output", type=str, default="", help="Optional path to write JSON summary")
    args = parser.parse_args()

    selected_platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    allowed_platforms = {"cpu", "cuda", "metal"}
    invalid_platforms = [p for p in selected_platforms if p not in allowed_platforms]
    if invalid_platforms:
        print(f"Invalid platforms: {invalid_platforms}. Allowed: {sorted(allowed_platforms)}")
        sys.exit(1)

    all_paths = _all_problem_paths()
    cpu_paths = _slice_problems(all_paths, args.max_problems)
    cuda_paths = _slice_problems(_cuda_problem_paths(all_paths), args.max_problems)
    metal_paths = _slice_problems(_metal_problem_paths(), args.max_problems)

    all_results: list[ValidationResult] = []
    platform_errors: dict[str, str] = {}

    if "cpu" in selected_platforms:
        cpu_results = run_local_validation(cpu_paths, device=torch.device("cpu"), platform="cpu")
        all_results.extend(cpu_results)
        _print_summary("cpu", cpu_results)

    if "cuda" in selected_platforms:
        try:
            cuda_results = run_cuda_validation(cuda_paths, gpu=args.cuda_gpu)
            all_results.extend(cuda_results)
            _print_summary("cuda", cuda_results)
        except Exception as exc:
            err = str(exc)
            platform_errors["cuda"] = err
            print("\n[CUDA] validation failed before per-problem results:")
            print(err)

    if "metal" in selected_platforms:
        try:
            metal_results = run_metal_validation(metal_paths, ssh_host=args.metal_host)
            all_results.extend(metal_results)
            _print_summary("metal", metal_results)
        except Exception as exc:
            err = str(exc)
            platform_errors["metal"] = err
            print("\n[METAL] validation failed before per-problem results:")
            print(err)

    if args.json_output:
        payload = {
            "results": [asdict(r) for r in all_results],
            "platform_errors": platform_errors,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    any_failed_results = any(not result.passed for result in all_results)
    if platform_errors or any_failed_results:
        sys.exit(1)

    print("\nAll baseline validations passed.")


if __name__ == "__main__":
    main()
