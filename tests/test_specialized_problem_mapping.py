from pathlib import Path

import batch_eval
import cutlass_eval
import cute_eval
from src.config.benchmark_problems import BENCHMARK_PROBLEMS, get_problem_hardware_required


def test_cutlass_and_cute_use_tile_specialized_directory() -> None:
    cutlass_problems = cutlass_eval.find_problems([1, 2])
    cute_problems = cute_eval.find_problems([1, 2])

    assert cutlass_problems
    assert cute_problems
    assert all("KernelBench/tile_specialized/" in str(path) for _, path in cutlass_problems)
    assert all("KernelBench/tile_specialized/" in str(path) for _, path in cute_problems)


def test_hardware_required_metadata_is_read() -> None:
    fp4_path = Path("KernelBench/tile_specialized/gemm_fp4.py")
    fp8_path = Path("KernelBench/tile_specialized/gemm_fp8.py")
    bf16_path = Path("KernelBench/tile_specialized/gemm_bf16.py")

    assert get_problem_hardware_required(fp4_path) == ["B200"]
    assert get_problem_hardware_required(fp8_path) == ["H100", "B200"]
    assert get_problem_hardware_required(bf16_path) == ["RTX3090", "A100", "H100", "B200"]


def test_batch_eval_skips_incompatible_hardware() -> None:
    original_find_problems = batch_eval.find_problems
    try:
        sample = [
            (1, Path("KernelBench/tile_specialized/gemm_fp4.py")),
            (1, Path("KernelBench/tile_specialized/gemm_fp8.py")),
            (1, Path("KernelBench/tile_specialized/gemm_bf16.py")),
        ]
        batch_eval.find_problems = lambda _levels: sample
        tasks = batch_eval.get_all_tasks(
            models=["dummy-model"],
            gpus=["H100", "B200"],
            levels=[1],
            problems_per_level=None,
        )
    finally:
        batch_eval.find_problems = original_find_problems

    by_gpu = {}
    for _model, gpu, _level, path in tasks:
        by_gpu.setdefault(gpu, set()).add(path.name)

    assert "gemm_fp4.py" not in by_gpu["H100"]
    assert "gemm_fp4.py" in by_gpu["B200"]


def test_cutile_is_blackwell_only() -> None:
    assert BENCHMARK_PROBLEMS["cutile"]["hardware"] == ["B200"]
    assert get_problem_hardware_required(Path("KernelBench/cutile/persistent_gemm.py")) == ["B200"]
