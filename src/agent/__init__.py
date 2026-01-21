"""
Agentic Kernel Optimization Module

Provides an agentic evaluation framework where the model can:
- Execute bash commands in a sandbox
- Read/write files
- Introspect models to understand their structure
- Iterate on solutions with real feedback
"""

from .modal_sandbox import ModalSandbox, ModalSandboxConfig, GPUType, create_modal_sandbox

__all__ = [
    "ModalSandbox",
    "ModalSandboxConfig",
    "GPUType",
    "create_modal_sandbox",
]
