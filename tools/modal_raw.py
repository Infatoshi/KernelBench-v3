#!/usr/bin/env python3
"""Modal entrypoint for running KernelBench raw evaluations on remote GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import modal

# Ensure local packages are importable when Modal loads this module.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modal_support.config import ModalRawConfig

CONFIG_PATH = REPO_ROOT / "configs" / "modal_raw.yaml"
MODAL_CONFIG = ModalRawConfig.load(CONFIG_PATH)
TIMEOUT = MODAL_CONFIG.timeouts.process_seconds
_GROQ_SECRET_ATTACHED = False


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    """Run a subprocess with the configured timeout."""
    printable_cmd = " ".join(command)
    print(f"[modal/raw] running: {printable_cmd}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    timeout_kwargs = {}
    if TIMEOUT > 0:
        timeout_kwargs["timeout"] = TIMEOUT

    if capture:
        output = subprocess.check_output(
            command,
            cwd=str(cwd) if cwd else None,
            env=run_env,
            text=True,
            **timeout_kwargs,
        )
        print(output)
        return output

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=run_env,
        check=True,
        **timeout_kwargs,
    )
    return ""


def _rsync_repo(source: Path, destination: Path) -> None:
    """Mirror the repository into a writable workspace."""
    destination.mkdir(parents=True, exist_ok=True)
    rsync_cmd: list[str] = [
        "rsync",
        "-a",
        "--delete",
        "--exclude",
        ".venv",
        "--exclude",
        "runs",
        "--exclude",
        "logs",
        f"{source}/",
        f"{destination}/",
    ]
    _run_subprocess(rsync_cmd)


def _ensure_uv_installed_commands() -> Iterable[str]:
    """Commands that install uv inside the Modal image."""
    return (
        "sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh && ln -sf /root/.local/bin/uv /usr/local/bin/uv'",
    )


image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{MODAL_CONFIG.image.registry_tag}",
        add_python=MODAL_CONFIG.image.python_version,
    )
    .apt_install("build-essential", "rsync", "curl")
    .run_commands(*_ensure_uv_installed_commands())
    .add_local_dir(str(REPO_ROOT), remote_path="/workspace_src")
)

app = modal.App("kernelbench-modal-raw")

_function_kwargs: dict[str, object] = {
    "image": image,
    "gpu": MODAL_CONFIG.gpu.to_modal_argument(),
}

if "GROQ_API_KEY" in os.environ:
    try:
        _function_kwargs["secrets"] = [modal.Secret.from_local_environ(["GROQ_API_KEY"])]
        _GROQ_SECRET_ATTACHED = True
    except Exception as exc:  # noqa: BLE001
        print(f"[modal/raw] WARNING: failed to attach GROQ_API_KEY secret: {exc}")


@app.function(**_function_kwargs)
def nvcc_version() -> str:
    """Return nvcc version to validate the CUDA toolchain."""
    return _run_subprocess(["nvcc", "--version"], capture=True)


@app.function(**_function_kwargs)
def run_raw_eval() -> None:
    """Synchronize the repo, install deps with uv, and execute the raw benchmark."""
    repo_src = Path("/workspace_src")
    workdir = Path("/tmp/kernelbench")

    _rsync_repo(repo_src, workdir)
    if os.environ.get("GROQ_API_KEY"):
        print("[modal/raw] GROQ_API_KEY detected in environment for remote run.")
    else:
        print("[modal/raw] WARNING: GROQ_API_KEY is not set inside the Modal task.")

    _run_subprocess(["uv", "sync"], cwd=workdir)
    _run_subprocess(
        ["uv", "run", "python", "eval.py", "--config", "configs/modal_raw.yaml"],
        cwd=workdir,
    )
    print("[modal/raw] run complete; artifacts are in /tmp/kernelbench/runs")


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint invoked via `modal run tools/modal_raw.py`."""
    print("[modal/raw] Modal configuration loaded from", CONFIG_PATH)
    print(
        "[modal/raw] image tag:",
        f"nvidia/cuda:{MODAL_CONFIG.image.registry_tag} (Python {MODAL_CONFIG.image.python_version})",
    )
    print("[modal/raw] gpu:", MODAL_CONFIG.gpu.to_modal_argument())
    if _GROQ_SECRET_ATTACHED:
        print("[modal/raw] GROQ_API_KEY secret forwarded to Modal execution.")
    else:
        print(
            "[modal/raw] WARNING: GROQ_API_KEY secret not forwarded; set GROQ_API_KEY before running."
        )
    nvcc_version.remote()
    run_raw_eval.remote()
