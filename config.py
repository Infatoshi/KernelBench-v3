from dataclasses import dataclass, field
from typing import List, Literal

EvaluationMode = Literal["raw", "agentic"]
KernelLanguage = Literal["cuda", "triton"]


@dataclass
class HardwareConfig:
    gpu_architecture: str = "Ampere"
    gpu_id: int = 0


@dataclass
class ProblemSetConfig:
    levels: List[int] = field(default_factory=lambda: [1, 2])
    problem_ids: List[int] | None = None
    max_problems: int = 100


@dataclass
class AgenticConfig:
    max_debug_attempts: int = 3
    max_optimization_cycles: int = 2
    reflector_model: str = "gpt-4-turbo"
    optimizer_model: str = "gpt-4-turbo"


@dataclass
class BenchmarkConfig:
    mode: EvaluationMode = "raw"
    language: KernelLanguage = "triton"
    provider: str = "openai"
    provider_base_url: str | None = None
    formatter_provider: str | None = None
    formatter_model: str | None = None
    formatter_base_url: str | None = None
    generator_model: str = "gpt-4-turbo"
    verbose: bool = False
    num_runs: str | int | None = None
    profile_stages: bool = False
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    problems: ProblemSetConfig = field(default_factory=ProblemSetConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    fast_p_threshold: float | None = None
    raw_concurrency: int = 8
    raw_gpu_concurrency: int = 1
    raw_max_jobs: int = 8
    generation_max_tokens: int = 4096
    formatter_max_tokens: int | None = None
