from pathlib import Path

from src.backends import get_backend
from src.prompts.cute_system import get_cute_system_prompt
from src.prompts.cutile_system import get_cutile_system_prompt
from src.prompts.cutlass_system import get_cutlass_system_prompt
from src.prompts.graphics_triton_system import get_graphics_triton_system_prompt

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def test_cutlass_scaffold_imports() -> None:
    backend = get_backend("cutlass")
    assert backend.name == "cutlass"
    assert backend.benchmark_name == "CUTLASSBench"
    assert callable(backend.find_problems)
    prompt = get_cutlass_system_prompt("H100", 80)
    assert "CUTLASS 3.x" in prompt
    assert "cutlass/gemm/device/gemm_universal.h" in prompt
    assert "WORKING CUTLASS 3.x GEMM EXAMPLE" in prompt


def test_cute_scaffold_imports() -> None:
    backend = get_backend("cute")
    assert backend.name == "cute"
    assert backend.benchmark_name == "CuTeBench"
    assert callable(backend.find_problems)
    prompt = get_cute_system_prompt("H100", 80)
    assert "cute/tensor.hpp" in prompt
    assert "make_layout" in prompt
    assert "INTERFACE CONTRACT" in prompt


def test_cutile_scaffold_imports() -> None:
    backend = get_backend("cutile")
    assert backend.name == "cutile"
    assert backend.benchmark_name == "CuTileBench"
    assert callable(backend.find_problems)
    prompt = get_cutile_system_prompt("H100", 80)
    assert "CuTile Python" in prompt
    assert "import cuda.tile as ct" in prompt
    assert "@ct.kernel" in prompt
    assert "Do NOT use `torch.utils.cpp_extension.load_inline`" in prompt


def test_graphics_scaffold_imports_and_problem_discovery() -> None:
    backend = get_backend("graphics")
    assert backend.name == "graphics"
    assert backend.benchmark_name == "GraphicsBench"
    assert callable(backend.find_problems)

    prompt = get_graphics_triton_system_prompt("RTX 3090", 24)
    assert "@triton.jit" in prompt
    assert "DO NOT USE:" in prompt
    assert "OpenGL" in prompt
    assert "Vulkan" in prompt

    problems = backend.find_problems(PROJECT_ROOT, [1])
    names = {p.name for _, p in problems}
    assert "bloom.py" in names
    assert "particles.py" in names
    for _, path in problems:
        assert Path(path).exists()
