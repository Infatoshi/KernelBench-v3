from src.eval.guardrails import validate_metal, validate_triton


def test_triton_guardrails_allow_valid_triton_solution() -> None:
    solution = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)

class Model:
    def __call__(self, x):
        return x
"""
    assert validate_triton(solution) is None


def test_triton_guardrails_reject_torch_matmul_fallback() -> None:
    solution = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel():
    pass

class Model:
    def __call__(self, a, b):
        return torch.matmul(a, b)
"""
    error = validate_triton(solution)
    assert error is not None
    assert "Forbidden Python fallback" in error


def test_triton_guardrails_require_triton_jit_kernel() -> None:
    solution = """
import triton
import triton.language as tl

class Model:
    def __call__(self, x):
        return x
"""
    error = validate_triton(solution)
    assert error is not None
    assert "@triton.jit" in error


def test_metal_guardrails_allow_valid_mlx_solution() -> None:
    solution = """
import mlx.core as mx

def solution(a, b):
    return mx.matmul(a, b)
"""
    assert validate_metal(solution) is None


def test_metal_guardrails_reject_torch_import() -> None:
    solution = """
import torch
import mlx.core as mx

def solution(a, b):
    return mx.matmul(a, b)
"""
    error = validate_metal(solution)
    assert error is not None
    assert "torch" in error.lower()


def test_metal_guardrails_require_mlx_import() -> None:
    solution = """
def solution(a, b):
    return a @ b
"""
    error = validate_metal(solution)
    assert error is not None
    assert "Missing MLX import" in error
