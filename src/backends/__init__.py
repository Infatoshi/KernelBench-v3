"""Backend registry for KernelBench benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


class Backend:
    """Base class for benchmark backends."""
    name: str = "base"
    benchmark_name: str = "BaseBench"
    allowed_gpus: List[str] = []
    gpu_reason: str = ""

    @property
    def gpu_specs(self) -> dict:
        return {
            "L40S": ("L40S", 48),
            "A100": ("A100", 40),
            "H100": ("H100", 80),
            "B200": ("B200", 192),
            "RTX3090": ("RTX 3090", 24),
            "LOCAL": ("Local CUDA GPU", 24),
        }

    @property
    def local_gpus(self) -> set:
        return {"RTX3090", "LOCAL"}

    @property
    def force_reasoning_mode(self) -> bool:
        return False

    def get_system_prompt(self, gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
        raise NotImplementedError

    def get_reasoning_prompt(self, gpu_name: str, vram_gb: int) -> str:
        raise NotImplementedError

    def find_problems(self, project_root: Path, levels: List[int]) -> List[Tuple[int, Path]]:
        raise NotImplementedError

    def max_turns(self, level: int) -> int:
        return {1: 10, 2: 12, 3: 15, 4: 15}.get(level, 15)

    def run_benchmark(self, sandbox, solution_path: str, **kwargs) -> dict:
        raise NotImplementedError

    def validate_solution(self, solution_code: str) -> Optional[str]:
        return None

    def build_api_reference(self) -> str:
        return ""

    def build_template_solution(self) -> str:
        return ""

    def create_sandbox(self, gpu: str, problem_code: str):
        from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig
        from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig
        if gpu in self.local_gpus:
            return LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
        return ModalSandbox(problem_code, ModalSandboxConfig(gpu=gpu, timeout=300, sandbox_timeout=3600))


BACKENDS: dict = {}


def register(name: str):
    def decorator(cls):
        BACKENDS[name] = cls()
        return cls
    return decorator


def get_backend(name: str) -> Backend:
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return BACKENDS[name]


from src.backends import cuda, triton, cutlass, cute, cutile, graphics, metal  # noqa: E402, F401
