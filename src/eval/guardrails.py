"""Unified solution validation guardrails."""

from __future__ import annotations

import re
from typing import Optional

NVIDIA_FORBIDDEN = [
    (re.compile(r"torch::\s*(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("), "Forbidden C++ PyTorch fallback"),
    (re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("), "Forbidden Python PyTorch fallback"),
    (re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("), "Forbidden torch.nn.functional fallback"),
    (re.compile(r"(?:^|[^\w])torch\.compile\s*\("), "Forbidden torch.compile"),
    (re.compile(r"@torch\.jit\.script"), "Forbidden torch.jit.script"),
]

METAL_FORBIDDEN = [
    (re.compile(r"(?:^|[^\w])import\s+torch\b"), "Forbidden: torch in Metal solution"),
    (re.compile(r"(?:^|[^\w])torch\."), "Forbidden: PyTorch usage in Metal solution"),
    (re.compile(r"(?:^|[^\w])import\s+triton\b"), "Forbidden: triton in Metal solution"),
    (re.compile(r"torch\.utils\.cpp_extension|(?:^|[^\w])load_inline\s*\("), "Forbidden: CUDA extension in Metal solution"),
]


def validate_nvidia(code: str) -> Optional[str]:
    for pattern, message in NVIDIA_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    return None


def validate_metal(code: str) -> Optional[str]:
    for pattern, message in METAL_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    if "import mlx.core as mx" not in code:
        return "Missing MLX import: expected `import mlx.core as mx`"
    if "def solution(" not in code:
        return "Missing required interface: `def solution(*inputs)`"
    return None


def validate_solution(code: str, is_metal: bool = False) -> Optional[str]:
    return validate_metal(code) if is_metal else validate_nvidia(code)
