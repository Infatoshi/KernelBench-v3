from pathlib import Path
from unittest.mock import patch

from src.backends import get_backend
from src.batch import get_all_tasks
from src.config.benchmark_problems import BENCHMARK_PROBLEMS, get_problem_hardware_required

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def test_cutlass_and_cute_use_tile_specialized_directory() -> None:
    cutlass_backend = get_backend("cutlass")
    cute_backend = get_backend("cute")
    cutlass_problems = cutlass_backend.find_problems(PROJECT_ROOT, [1, 2])
    cute_problems = cute_backend.find_problems(PROJECT_ROOT, [1, 2])

    assert cutlass_problems
    assert cute_problems
    assert all("problems/tile_specialized/" in str(path) for _, path in cutlass_problems)
    assert all("problems/tile_specialized/" in str(path) for _, path in cute_problems)


def test_hardware_required_metadata_is_read() -> None:
    fp4_path = Path("problems/tile_specialized/gemm_fp4.py")
    fp8_path = Path("problems/tile_specialized/gemm_fp8.py")
    bf16_path = Path("problems/tile_specialized/gemm_bf16.py")

    assert get_problem_hardware_required(fp4_path) == ["B200"]
    assert get_problem_hardware_required(fp8_path) == ["H100", "B200"]
    assert get_problem_hardware_required(bf16_path) == ["RTX3090", "A100", "H100", "B200"]


def test_batch_eval_skips_incompatible_hardware() -> None:
    cuda_backend = get_backend("cuda")
    sample = [
        (1, Path("problems/tile_specialized/gemm_fp4.py")),
        (1, Path("problems/tile_specialized/gemm_fp8.py")),
        (1, Path("problems/tile_specialized/gemm_bf16.py")),
    ]
    with patch.object(cuda_backend, "find_problems", return_value=sample):
        tasks = get_all_tasks(
            backend=cuda_backend,
            project_root=PROJECT_ROOT,
            models=["dummy-model"],
            gpus=["H100", "B200"],
            levels=[1],
            problems_per_level=None,
        )

    by_gpu: dict = {}
    for _model, gpu, _level, path in tasks:
        by_gpu.setdefault(gpu, set()).add(path.name)

    assert "gemm_fp4.py" not in by_gpu["H100"]
    assert "gemm_fp4.py" in by_gpu["B200"]


def test_cutile_is_blackwell_only() -> None:
    assert BENCHMARK_PROBLEMS["cutile"]["hardware"] == ["B200"]
    assert get_problem_hardware_required(Path("problems/cutile/persistent_gemm.py")) == ["B200"]
