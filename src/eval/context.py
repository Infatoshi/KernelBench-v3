"""Context building for agent workspace: reference metadata, environment, templates."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict


def safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_reference_metadata(reference_code: str) -> Dict[str, Any]:
    """Extract static metadata from reference.py without executing it."""
    metadata: Dict[str, Any] = {
        "op_type": "unknown",
        "supported_precisions": [],
        "hardware_required": [],
        "has_model_class": False,
        "has_get_inputs": False,
        "has_get_init_inputs": False,
    }
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return metadata

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            metadata["has_model_class"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            metadata["has_get_inputs"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            metadata["has_get_init_inputs"] = True
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "OP_TYPE":
                value = safe_literal_eval(node.value)
                if isinstance(value, str):
                    metadata["op_type"] = value
            elif target.id == "SUPPORTED_PRECISIONS":
                value = safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["supported_precisions"] = [str(x) for x in value]
            elif target.id == "HARDWARE_REQUIRED":
                value = safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["hardware_required"] = [str(x) for x in value]
    return metadata


def backend_self_check_command(backend: str) -> str:
    if backend == "metal":
        return (
            'python -c "import mlx.core as mx, solution; '
            "a=mx.random.normal((128,128), dtype=mx.float32); "
            "b=mx.random.normal((128,128), dtype=mx.float32); "
            "y=solution.solution(a,b); mx.eval(y); print('OK')\""
        )
    return (
        'python -c "import reference, solution, torch; '
        "m=solution.Model(*reference.get_init_inputs()).cuda().eval(); "
        "inputs=[x.cuda() if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]; "
        "_=m(*inputs); print('OK')\""
    )


def augment_system_prompt(system_prompt: str, backend: str) -> str:
    """Append environment contract and anti-probing policy to any prompt."""
    self_check = backend_self_check_command(backend)
    return (
        system_prompt
        + f"""

EXECUTION ENVIRONMENT CONTRACT:
- You are in an isolated benchmark sandbox shell, not a general SSH host session.
- Working directory is `/workspace`.
- Preloaded files: `/workspace/reference.py`, `/workspace/ENVIRONMENT.md`, `/workspace/BACKEND_API.md`, `/workspace/TEMPLATE_solution.py`, `/workspace/TASK_CONTEXT.md`.
- Internet access and package installation are not part of this task. Do not run package managers.

ANTI-PROBING POLICY:
- At most one lightweight environment probe command in turn 1.
- Do not spend turns on `pip list`, `which python`, `sys.path`, or filesystem crawling unless a compile/runtime error explicitly requires it.
- Move to implementing `solution.py` immediately after reading context files.

REQUIRED PRE-SUBMIT SELF-CHECK:
- Run exactly:
`{self_check}`
- If this prints `OK`, call submit immediately and stop editing.
"""
    )


def inject_workspace_context(system_prompt: str, context_bundle: Dict[str, str]) -> str:
    """Inline workspace context content directly into the prompt."""
    environment_md = context_bundle.get("environment_md", "").strip()
    backend_api_md = context_bundle.get("backend_api_md", "").strip()
    task_context_md = context_bundle.get("task_context_md", "").strip()
    template_solution_py = context_bundle.get("template_solution_py", "").strip()

    return (
        system_prompt
        + "\n\nINLINE WORKSPACE CONTEXT (authoritative; do not re-discover):\n"
        + "\n\n[ENVIRONMENT.md]\n"
        + environment_md
        + "\n\n[BACKEND_API.md]\n"
        + backend_api_md
        + "\n\n[TASK_CONTEXT.md]\n"
        + task_context_md
        + "\n\n[TEMPLATE_solution.py]\n```python\n"
        + template_solution_py
        + "\n```\n"
    )


def collect_runtime_environment(sandbox, backend: str, gpu_name: str, vram_gb: int, level: int) -> str:
    """Collect runtime environment facts and return markdown."""
    probe_cmd = """python - <<'PY'
import importlib
import json
import platform
import sys

modules = {}
for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
    try:
        mod = importlib.import_module(name)
        modules[name] = getattr(mod, "__version__", "unknown")
    except Exception:
        modules[name] = None

payload = {
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "modules": modules,
}
print(json.dumps(payload))
PY"""
    probe_result = sandbox.run_command(probe_cmd, timeout=45)
    payload: Dict[str, Any] = {}
    if probe_result.get("returncode") == 0:
        for line in reversed((probe_result.get("stdout") or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    modules = payload.get("modules", {})
    lines = [
        "# Environment Snapshot",
        "",
        f"- backend: `{backend}`",
        f"- target_gpu: `{gpu_name}`",
        f"- target_vram_gb: `{vram_gb}`",
        f"- level: `{level}`",
        "- working_directory: `/workspace`",
        f"- python_version: `{payload.get('python_version', 'unknown')}`",
        f"- platform: `{payload.get('platform', 'unknown')}`",
        "",
        "## Module Availability",
    ]
    for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
        version = modules.get(name)
        lines.append(f"- {name}: `{version if version else 'unavailable'}`")
    lines.extend(
        [
            "",
            "## Constraints",
            "- Do not run package installers during solving.",
            "- Use preinstalled toolchain and files in `/workspace`.",
        ]
    )
    return "\n".join(lines)


def build_task_context(
    problem_name: str,
    level: int,
    gpu_name: str,
    backend: str,
    metadata: Dict[str, Any],
) -> str:
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    hardware_required = ", ".join(metadata.get("hardware_required", [])) or "unknown"
    return f"""# Task Context

- problem: `{problem_name}`
- level: `{level}`
- backend: `{backend}`
- gpu: `{gpu_name}`
- op_type: `{metadata.get("op_type", "unknown")}`
- supported_precisions: `{precisions}`
- hardware_required: `{hardware_required}`
- has_model_class: `{metadata.get("has_model_class", False)}`
- has_get_inputs: `{metadata.get("has_get_inputs", False)}`
- has_get_init_inputs: `{metadata.get("has_get_init_inputs", False)}`

Read `/workspace/reference.py` and preserve API compatibility.
"""


def prepare_workspace_context(
    backend: str,
    gpu_name: str,
    vram_gb: int,
    level: int,
    problem_name: str,
    metadata: Dict[str, Any],
    sandbox,
    backend_obj=None,
) -> Dict[str, str]:
    """Build helper context bundle used for files and inline prompt injection."""
    if backend_obj is not None:
        api_ref = backend_obj.build_api_reference()
        template = backend_obj.build_template_solution()
    else:
        api_ref = _build_backend_api_reference(backend)
        template = _build_template_solution(backend)

    return {
        "environment_md": collect_runtime_environment(
            sandbox=sandbox,
            backend=backend,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            level=level,
        ),
        "backend_api_md": api_ref,
        "template_solution_py": template,
        "task_context_md": build_task_context(problem_name, level, gpu_name, backend, metadata),
    }


def seed_workspace_context(sandbox, context_bundle: Dict[str, str]) -> None:
    """Write helper context files into /workspace."""
    try:
        sandbox.write_file("ENVIRONMENT.md", context_bundle.get("environment_md", ""))
        sandbox.write_file("BACKEND_API.md", context_bundle.get("backend_api_md", ""))
        sandbox.write_file("TEMPLATE_solution.py", context_bundle.get("template_solution_py", ""))
        sandbox.write_file("TASK_CONTEXT.md", context_bundle.get("task_context_md", ""))
    except Exception as exc:
        print(f"Warning: failed to seed workspace context files: {exc}", flush=True)


def build_initial_user_message(
    backend: str,
    problem_name: str,
    level: int,
    gpu_name: str,
    max_turns: int,
    reference_code: str,
    metadata: Dict[str, Any],
) -> str:
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    hardware_required = ", ".join(metadata.get("hardware_required", [])) or "unknown"
    return f"""Optimize the benchmark task and produce `/workspace/solution.py`.

Task summary:
- benchmark backend: `{backend}`
- problem: `{problem_name}`
- level: `{level}`
- target GPU: `{gpu_name}`
- OP_TYPE: `{metadata.get("op_type", "unknown")}`
- SUPPORTED_PRECISIONS: `{precisions}`
- HARDWARE_REQUIRED: `{hardware_required}`

Mirrored workspace files (same content already provided inline in system context):
- `/workspace/reference.py`
- `/workspace/ENVIRONMENT.md`
- `/workspace/BACKEND_API.md`
- `/workspace/TEMPLATE_solution.py`
- `/workspace/TASK_CONTEXT.md`

Rules:
- Preserve `Model`, `get_inputs`, and `get_init_inputs` compatibility with `reference.py`.
- Run one compile/import/forward self-check, then submit immediately.
- Avoid environment probing beyond one lightweight check.

Turn budget (hard cap):
- You have exactly `{max_turns}` turns.
- Use turn 1 to inspect/reference and write a complete draft.
- Use intermediate turns only to fix concrete compile/runtime errors.
- On the final turn, if a `submit` tool is available, submit your best current `solution.py` even if still imperfect.
- Do not finish without either submitting or clearly reporting a blocking compile error.

Reference code:
```python
{reference_code}
```
"""


# ---------------------------------------------------------------------------
# Fallback helpers (used when no Backend object is available)
# ---------------------------------------------------------------------------

def _build_backend_api_reference(backend: str) -> str:
    backend_key = backend.lower()
    refs = {
        "triton": """# Backend API Quick Reference: Triton

- Required imports:
  - `import triton`
  - `import triton.language as tl`
- Required kernel pattern:
  - `@triton.jit`
  - `kernel_name[grid](...)`
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT use `torch.utils.cpp_extension.load_inline` in Triton backend.
""",
        "cutlass": """# Backend API Quick Reference: CUTLASS

- Required includes from `/opt/cutlass/include`:
  - `cutlass/gemm/device/gemm_universal.h`
- Use `from torch.utils.cpp_extension import load_inline`.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT fallback to raw PyTorch ops in the wrapper path.
""",
        "cute": """# Backend API Quick Reference: CuTe

- CuTe headers path: `/opt/cutlass/include`.
- Required includes:
  - `cute/tensor.hpp`
  - `cute/layout.hpp`
- Use `from torch.utils.cpp_extension import load_inline`.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Use exact include path `/opt/cutlass/include` (not `/opt/cutlass/include/include`).
""",
        "cutile": """# Backend API Quick Reference: CuTile Python

- Required import:
  - `import cuda.tile as ct`
- Use CuTile Python kernels with `@ct.kernel`.
- Launch signature:
  - `ct.launch(stream, grid, kernel, kernel_args_tuple)`
- Use compile-time constants for tile shapes.
- Do NOT use `load_inline` or C++ extension code in CuTile backend.
""",
        "metal": """# Backend API Quick Reference: Metal (MLX)

- Required import:
  - `import mlx.core as mx`
- Implement `solution(a, b)` for MLX arrays.
- Use MLX operations / kernel APIs, not PyTorch CUDA extensions.
- Do NOT use `torch.utils.cpp_extension.load_inline`.
""",
        "graphics": """# Backend API Quick Reference: Graphics (CUDA Triton)

- Required imports:
  - `import triton`
  - `import triton.language as tl`
- Required kernel pattern:
  - `@triton.jit`
  - `kernel_name[grid](...)`
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT use OpenGL/Vulkan/Metal runtime code paths.
""",
    }
    return refs.get(
        backend_key,
        """# Backend API Quick Reference: CUDA

- Use `from torch.utils.cpp_extension import load_inline`.
- Write actual `__global__` CUDA kernels and call them from wrappers.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT fallback to raw PyTorch/cuBLAS calls in wrapper path.
""",
    )


def _build_template_solution(backend: str) -> str:
    backend_key = backend.lower()
    templates = {
        "triton": """import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 256),)
        _kernel[grid](x, y, n, BLOCK=256)
        return y
""",
        "graphics": """import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _graphics_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 256),)
        _graphics_kernel[grid](x, y, n, BLOCK=256)
        return y
""",
        "cutile": """import torch
import torch.nn as nn
import cuda.tile as ct

@ct.kernel
def _kernel(x, y, n):
    pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (ct.cdiv(n, 256),)
        ct.launch(torch.cuda.current_stream(), grid, _kernel, (x, y, n))
        return y
""",
        "metal": """import mlx.core as mx

def solution(a, b):
    return mx.matmul(a, b)
""",
    }
    if backend_key in templates:
        return templates[backend_key]
    # cuda / cutlass / cute share the same C++ extension template
    return """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void identity_kernel(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx];
}

torch::Tensor launch_identity(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  identity_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
'''

ext = load_inline(
    name='kb_template_ext',
    cpp_sources='torch::Tensor launch_identity(torch::Tensor x);',
    cuda_sources=cuda_source,
    functions=['launch_identity'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.launch_identity(x)
"""
