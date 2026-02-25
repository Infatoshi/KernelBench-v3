"""Solution validation guardrails for all backends."""

from __future__ import annotations

import re
from typing import Optional

CUDA_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"torch::\s*(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden C++ wrapper fallback to PyTorch operator",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden Python fallback to PyTorch operator",
    ),
    (
        re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("),
        "Forbidden Python fallback via torch.nn.functional",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.compile\s*\("),
        "Forbidden use of torch.compile",
    ),
    (
        re.compile(r"@torch\.jit\.script"),
        "Forbidden use of torch.jit.script",
    ),
]

TRITON_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])torch\.compile\s*\("),
        "Forbidden use of torch.compile in Triton backend",
    ),
    (
        re.compile(r"@torch\.jit\.script"),
        "Forbidden use of torch.jit.script in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden Python fallback to PyTorch operator in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("),
        "Forbidden Python fallback via torch.nn.functional in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])load_inline\s*\("),
        "Forbidden CUDA C++ extension fallback in Triton backend",
    ),
    (
        re.compile(r"torch\.utils\.cpp_extension"),
        "Forbidden CUDA C++ extension import in Triton backend",
    ),
]

METAL_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])import\s+torch\b"),
        "Forbidden import: torch is not allowed in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\."),
        "Forbidden PyTorch usage in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])import\s+triton\b"),
        "Forbidden import: triton is not allowed in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])triton\."),
        "Forbidden Triton usage in Metal backend",
    ),
    (
        re.compile(r"torch\.utils\.cpp_extension|(?:^|[^\w])load_inline\s*\("),
        "Forbidden CUDA extension usage in Metal backend",
    ),
]

GRAPHICS_REQUIRED_PATTERNS = [
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+triton\b|from\s+triton\b)"),
        "Missing Triton import for GraphicsBench",
    ),
    (
        re.compile(r"@triton\.jit"),
        "Missing `@triton.jit` kernel for GraphicsBench",
    ),
    (
        re.compile(r"[A-Za-z_]\w*\s*\[[^\]]+\]\s*\("),
        "Missing Triton kernel launch `kernel[grid](...)` for GraphicsBench",
    ),
]

GRAPHICS_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])reference\.Model(?:\W|$)"),
        "Forbidden direct fallback to reference model in GraphicsBench",
    ),
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+OpenGL\b|from\s+OpenGL\b)"),
        "Forbidden OpenGL runtime path in GraphicsBench",
    ),
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+moderngl\b|from\s+moderngl\b)"),
        "Forbidden Moderngl runtime path in GraphicsBench",
    ),
]


def validate_cuda(solution_code: str) -> Optional[str]:
    for pattern, message in CUDA_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    return None


def validate_triton(solution_code: str) -> Optional[str]:
    for pattern, message in TRITON_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    if "@triton.jit" not in solution_code:
        return "Missing Triton kernel definition: expected `@triton.jit`"
    if "import triton" not in solution_code:
        return "Missing Triton import: expected `import triton`"
    return None


def validate_metal(solution_code: str) -> Optional[str]:
    for pattern, message in METAL_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    if "import mlx.core as mx" not in solution_code:
        return "Missing MLX import: expected `import mlx.core as mx`"
    if "def solution(" not in solution_code:
        return "Missing required benchmark interface: `def solution(*inputs)`"
    return None


def validate_graphics(solution_code: str) -> Optional[str]:
    base_err = validate_cuda(solution_code)
    if base_err:
        return base_err
    for pattern, message in GRAPHICS_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    for pattern, message in GRAPHICS_REQUIRED_PATTERNS:
        if not pattern.search(solution_code):
            return message
    return None


def validate_solution(solution_code: str, backend: str = "cuda") -> Optional[str]:
    """Dispatch to backend-specific validation. Returns error string or None."""
    validators = {
        "cuda": validate_cuda,
        "cutlass": validate_cuda,
        "cute": validate_cuda,
        "cutile": validate_cuda,
        "triton": validate_triton,
        "metal": validate_metal,
        "graphics": validate_graphics,
    }
    validator = validators.get(backend, validate_cuda)
    return validator(solution_code)
