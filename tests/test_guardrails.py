"""Tests for unified guardrails."""

from src.eval.guardrails import validate_nvidia, validate_metal, validate_solution


def test_nvidia_blocks_torch_matmul():
    code = "output = torch.matmul(a, b)"
    err = validate_nvidia(code)
    assert err is not None
    assert "fallback" in err.lower() or "Forbidden" in err


def test_nvidia_blocks_f_linear():
    code = "out = F.linear(x, weight)"
    err = validate_nvidia(code)
    assert err is not None


def test_nvidia_blocks_torch_compile():
    code = "model = torch.compile(model)"
    err = validate_nvidia(code)
    assert err is not None


def test_nvidia_allows_cuda_kernel():
    code = '''
import torch
from torch.utils.cpp_extension import load_inline
cuda_source = """__global__ void kernel() {}"""
'''
    err = validate_nvidia(code)
    assert err is None


def test_nvidia_allows_triton():
    code = '''
import triton
import triton.language as tl
@triton.jit
def my_kernel(): pass
'''
    err = validate_nvidia(code)
    assert err is None


def test_metal_blocks_torch():
    code = "import torch"
    err = validate_metal(code)
    assert err is not None


def test_metal_requires_mlx():
    code = "def solution(*inputs): pass"
    err = validate_metal(code)
    assert err is not None
    assert "MLX" in err


def test_metal_valid():
    code = '''import mlx.core as mx

def solution(*inputs):
    return mx.matmul(inputs[0], inputs[1])
'''
    err = validate_metal(code)
    assert err is None


def test_validate_solution_dispatch():
    cuda_code = "output = torch.matmul(a, b)"
    assert validate_solution(cuda_code, is_metal=False) is not None
    assert validate_solution(cuda_code, is_metal=True) is not None
