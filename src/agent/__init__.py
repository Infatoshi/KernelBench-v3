"""
Agentic Kernel Optimization Module

Provides an agentic evaluation framework where the model can:
- Execute bash commands in a sandbox
- Read/write files
- Introspect models to understand their structure
- Iterate on solutions with real feedback
"""

from .sandbox import DockerSandbox, LocalSandbox, create_sandbox, SandboxConfig
from .executor import AgentExecutor, AgentConfig, AgentResult, run_agent_evaluation
from .prompts import TOOLS, TOOLS_OPENAI, get_system_prompt, get_initial_user_message

__all__ = [
    # Sandbox
    "DockerSandbox",
    "LocalSandbox",
    "create_sandbox",
    "SandboxConfig",
    # Executor
    "AgentExecutor",
    "AgentConfig",
    "AgentResult",
    "run_agent_evaluation",
    # Prompts
    "TOOLS",
    "TOOLS_OPENAI",
    "get_system_prompt",
    "get_initial_user_message",
]
