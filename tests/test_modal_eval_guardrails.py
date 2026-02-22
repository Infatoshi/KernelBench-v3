import modal_eval


def test_validate_solution_guardrails_allows_custom_kernel_path() -> None:
    solution = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = '''
__global__ void my_kernel(float* out, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i];
}
torch::Tensor my_op(torch::Tensor x) {
  auto out = torch::empty_like(x);
  my_kernel<<<(x.numel() + 255) / 256, 256>>>(out.data_ptr<float>(), x.data_ptr<float>(), x.numel());
  return out;
}
'''

class Model(nn.Module):
    def forward(self, x):
        return x
"""
    assert modal_eval.validate_solution_guardrails(solution) is None


def test_validate_solution_guardrails_rejects_python_torch_matmul() -> None:
    solution = """
import torch
class Model:
    def forward(self, a, b):
        return torch.matmul(a, b)
"""
    error = modal_eval.validate_solution_guardrails(solution)
    assert error is not None
    assert "Forbidden Python fallback" in error


def test_validate_solution_guardrails_rejects_cpp_wrapper_torch_mm() -> None:
    solution = """
cuda_source = '''
torch::Tensor my_op(torch::Tensor a, torch::Tensor b) {
    return torch::mm(a, b);
}
'''
"""
    error = modal_eval.validate_solution_guardrails(solution)
    assert error is not None
    assert "Forbidden C++ wrapper fallback" in error


def test_validate_solution_guardrails_rejects_torch_compile() -> None:
    solution = """
import torch
class Model:
    def forward(self, x):
        return torch.compile(lambda y: y)(x)
"""
    error = modal_eval.validate_solution_guardrails(solution)
    assert error is not None
    assert "torch.compile" in error


def test_validate_solution_guardrails_graphics_requires_triton_kernel() -> None:
    solution = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x
"""
    error = modal_eval.validate_solution_guardrails(solution, backend="graphics")
    assert error is not None
    assert "Missing Triton import" in error


def test_validate_solution_guardrails_graphics_rejects_reference_fallback() -> None:
    solution = """
import triton
import triton.language as tl
import reference

@triton.jit
def k():
    pass

class Model:
    def forward(self, *args):
        return reference.Model()(*args)
"""
    error = modal_eval.validate_solution_guardrails(solution, backend="graphics")
    assert error is not None
    assert "Forbidden direct fallback to reference model" in error
