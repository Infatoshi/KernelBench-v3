"""CuTe backend -- uses CuTe (CUTLASS 3.x) abstractions."""

from __future__ import annotations


from src.backends import Backend, register
from src.backends.cuda import _run_cuda_benchmark
from src.config.benchmark_problems import find_problems_for_benchmark
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_cuda
from src.prompts.cute_system import get_cute_reasoning_system_prompt, get_cute_system_prompt


@register("cute")
class CuTeBackend(Backend):
    name = "cute"
    benchmark_name = "CuTeBench"
    allowed_gpus = ["H100", "B200"]
    gpu_reason = "CuTeBench requires Hopper+ for CuTe tile-level APIs."

    @property
    def force_reasoning_mode(self) -> bool:
        return True

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_cute_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_cute_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        return find_problems_for_benchmark(project_root, benchmark="cute", levels=levels)

    def validate_solution(self, solution_code):
        return validate_cuda(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("cute")

    def build_template_solution(self):
        return _build_template_solution("cute")

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        return _run_cuda_benchmark(sandbox, solution_path, **kwargs)
