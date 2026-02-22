from pathlib import Path

import cutlass_batch_eval
import cutlass_eval
import cutile_batch_eval
import cutile_eval
import cute_batch_eval
import cute_eval
import graphics_batch_eval
import graphics_eval
from src.prompts.graphics_triton_system import get_graphics_triton_system_prompt
from src.prompts.cutile_system import get_cutile_system_prompt
from src.prompts.cutlass_system import get_cutlass_system_prompt
from src.prompts.cute_system import get_cute_system_prompt


def test_cutlass_scaffold_imports() -> None:
    assert callable(cutlass_eval.run_agent_on_modal)
    assert callable(cutlass_batch_eval.main)
    prompt = get_cutlass_system_prompt("H100", 80)
    assert "CUTLASS 3.x" in prompt
    assert "cutlass/gemm/device/gemm_universal.h" in prompt
    assert "WORKING CUTLASS 3.x GEMM EXAMPLE" in prompt


def test_cute_scaffold_imports() -> None:
    assert callable(cute_eval.run_agent_on_modal)
    assert callable(cute_batch_eval.main)
    prompt = get_cute_system_prompt("H100", 80)
    assert "cute/tensor.hpp" in prompt
    assert "make_layout" in prompt
    assert "Your solution function MUST have this exact signature" in prompt


def test_cutile_scaffold_imports() -> None:
    assert callable(cutile_eval.run_agent_on_modal)
    assert callable(cutile_batch_eval.main)
    prompt = get_cutile_system_prompt("H100", 80)
    assert "CuTile Python" in prompt
    assert "import cuda.tile as ct" in prompt
    assert "@ct.kernel" in prompt
    assert "Do NOT use `torch.utils.cpp_extension.load_inline`" in prompt


def test_graphics_scaffold_imports_and_problem_discovery() -> None:
    assert callable(graphics_eval.run_agent_on_modal)
    assert callable(graphics_batch_eval.main)

    prompt = get_graphics_triton_system_prompt("RTX 3090", 24)
    assert "@triton.jit" in prompt
    assert "DO NOT USE:" in prompt
    assert "OpenGL" in prompt
    assert "Vulkan" in prompt

    problems = graphics_eval.find_problems([1])
    names = {p.name for _, p in problems}
    assert "bloom.py" in names
    assert "particles.py" in names
    for _, path in problems:
        assert Path(path).exists()
