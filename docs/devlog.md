# Development Log

Key design decisions and milestones in reverse chronological order.

## 2026-02-25: Repo Consolidation

Decomposed 15 root-level eval scripts (modal_eval.py at 2818 lines, 7 *_eval.py, 7 *_batch_eval.py) into modular `src/` with single `bench.py` CLI. Eliminated monkey-patching in favor of proper Backend class dispatch. Renamed `KernelBench/` to `problems/`. Removed stale files (audit/, paper/, SPEC.md, CLAUDE.md). Moved docs to `docs/`.

Root went from 20 Python files to 1 (`bench.py`).

## 2026-02-23: Agent Toolset Upgrade

Added `read_file`, `write_file`, `edit_file` tools matching Claude Code's tool interface. Trimmed system prompts from prescriptive step-by-step workflows to goal-oriented constraints. Stopped inlining workspace context into prompts (models read files themselves). Research-informed: mechanical enforcement > suggestions, 2-3 constraints beat a manual.

## 2026-02-23: Problem Shapes and Compiled Baseline

Resized undersized L1/L2 problem shapes to realistic workloads (e.g., matmuls from 128x64 to 128x4096, convs from 32x32 to 256x256). Added `torch.compile(mode='reduce-overhead')` as the CUDA/Triton baseline — models now compete against the compiler, not naive Python.

## 2026-02-22: Metal-Specific Problems

Added 26 Metal-specific problems across 4 levels: image/signal processing (gaussian blur, bilateral filter, FFT, etc.), physics simulation (N-body, SPH, cloth), rendering (bloom, SSAO, raymarching), and scientific compute (prefix sum, radix sort, sparse matvec, KNN).

## 2026-02-22: MetalBench Harness Fix

Fixed MetalBench to evaluate real problems from `reference.py` instead of synthetic fixed matmul. Updated contract to `solution(*inputs)` with dynamic arity. Aligned correctness checking against `reference.Model(*get_init_inputs())` output.

## 2026-02-22: Model Registry Update

Replaced stale model entries with current OpenRouter IDs. Added Claude Opus/Sonnet 4.6, GPT-5.2/5.3 Codex, Gemini 3.1 Pro, Qwen3 Coder Next, Qwen3.5. Removed deprecated short-key entries (grok-4.1, minimax-m2.1, etc.). All models route through OpenRouter for unified pricing.

## 2026-02-21: GraphicsBench

Added GraphicsBench for Triton compute kernels on graphics workloads (bloom effect, particle simulation). RTX3090 only. Uses AST-based level parsing from problem files.

## Design Principles

- **One problem, one file**: Each problem is a self-contained `.py` with `Model`, `get_inputs()`, `get_init_inputs()`.
- **Static shapes**: One fixed shape per problem. No dynamic shape testing — avoids agent turn waste on edge cases.
- **Backend guardrails**: Regex-based validation prevents reward hacking (e.g., falling back to `torch.matmul` instead of writing a kernel).
- **Prompt caching**: System prompts use Anthropic/OpenRouter cache_control for 90% savings on multi-turn conversations.
- **Kernel persistence**: All submitted solutions are saved to `outputs/.../kernels/` for website display and analysis.
