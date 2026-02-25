"""CuTile backend -- uses cuda.tile Python DSL."""

from __future__ import annotations


from src.backends import Backend, register
from src.backends.cuda import _run_cuda_benchmark
from src.config.benchmark_problems import find_problems_for_benchmark
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_cuda
from src.prompts.cutile_system import get_cutile_reasoning_system_prompt, get_cutile_system_prompt


@register("cutile")
class CuTileBackend(Backend):
    name = "cutile"
    benchmark_name = "CuTileBench"
    allowed_gpus = ["B200"]
    gpu_reason = "CuTileBench requires Blackwell (B200) for cuda.tile DSL support."

    @property
    def force_reasoning_mode(self) -> bool:
        return True

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_cutile_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_cutile_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        return find_problems_for_benchmark(project_root, benchmark="cutile", levels=levels)

    def validate_solution(self, solution_code):
        return validate_cuda(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("cutile")

    def build_template_solution(self):
        return _build_template_solution("cutile")

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        hardware = kwargs.get("hardware", "UNKNOWN")
        if hardware != "B200":
            return {
                "compiled": False,
                "correct": False,
                "speedup": None,
                "error": f"CuTile requires B200 GPU, got {hardware}",
            }
        return _run_cuda_benchmark(sandbox, solution_path, **kwargs)
