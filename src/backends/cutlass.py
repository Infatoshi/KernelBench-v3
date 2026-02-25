"""CUTLASS backend -- uses CUTLASS 3.x headers via load_inline."""

from __future__ import annotations


from src.backends import Backend, register
from src.backends.cuda import _run_cuda_benchmark
from src.config.benchmark_problems import find_problems_for_benchmark
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_cuda
from src.prompts.cutlass_system import get_cutlass_reasoning_system_prompt, get_cutlass_system_prompt


@register("cutlass")
class CutlassBackend(Backend):
    name = "cutlass"
    benchmark_name = "CUTLASSBench"
    allowed_gpus = ["H100", "B200"]
    gpu_reason = "CUTLASSBench requires Hopper+ for CUTLASS 3.x tile-level APIs."

    @property
    def force_reasoning_mode(self) -> bool:
        return True

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_cutlass_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_cutlass_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        return find_problems_for_benchmark(project_root, benchmark="cutlass", levels=levels)

    def validate_solution(self, solution_code):
        return validate_cuda(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("cutlass")

    def build_template_solution(self):
        return _build_template_solution("cutlass")

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        return _run_cuda_benchmark(sandbox, solution_path, **kwargs)
