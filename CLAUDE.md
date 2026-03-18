# AGENTS.md - KernelBench-v3 Handoff

Last updated: 2026-03-17

## Snapshot
- Repository: `/home/infatoshi/cuda/KernelBench-v3`
- Branch: `master`
- Remote: `origin https://github.com/Infatoshi/KernelBench-v3.git`
- Local run artifacts: `/home/infatoshi/cuda/KernelBench-v3/outputs/batch_eval/`
- M4 Max benchmark: separate copy on macbook at `~/MetalBench`

## Non-Negotiable Project Rules
- Use UV only: `uv run ...`, `uv add ...`, `uv pip install ...`
- Do not use bare `python` or `pip`
- Before closing work, run: `uv run ruff check . --fix` and `uv run pytest`

## Architecture
Single repo with hardware target dispatch. Entry point: `bench.py`.

```
bench.py                    # CLI entry point
src/
  models.py                 # Model registry, pricing, provider clients
  api.py                    # API calls, token usage, cost estimation
  tools.py                  # Agent tools (bash, read_file, write_file, edit_file, submit) + guardrails
  prompts.py                # Per-architecture system prompts (RTX3090/H100/B200/M4Max)
  batch.py                  # Batch evaluation orchestration
  parsing.py                # Code extraction from LLM responses
  eval/
    agent.py                # Multi-turn agent loop (standard, gemini, reasoning modes)
    benchmark.py            # Performance benchmarking with adaptive torch.compile baseline
    context.py              # Workspace context, self-check commands
    fingerprint.py          # GPU/system metadata collection
    guardrails.py           # Solution validation (forbidden patterns)
    results.py              # EvalResult dataclass
  hardware/
    __init__.py             # HardwareTarget base, registry
    rtx3090.py              # RTX 3090 — local sandbox, 24GB, 43 problems
    h100.py                 # H100 — Modal sandbox, 80GB, 54 problems
    b200.py                 # B200 — Modal sandbox, 192GB, 58 problems (includes FP4, cutile)
    m4max.py                # M4 Max — Metal sandbox, 128GB, 63 problems
  agent/
    local_sandbox.py        # Local GPU execution
    modal_sandbox.py        # Modal cloud GPU execution
    metal_sandbox.py        # macOS Metal execution
problems/
  level1/                   # 15 simple ops (matmul, softmax, conv, norms)
  level2/                   # 15 fused ops (matmul+activation chains)
  level3/                   # 3 architecture blocks (attention, transformer)
  level4/                   # 9 novel layers (MLA, MoE, GQA, FP8, INT4, FP4, etc.)
  graphics/                 # 2 graphics problems (bloom, particles) — RTX3090 only
  tile_specialized/         # 13 GEMM variants — H100/B200
  cutile/                   # 3 cuTile problems — B200 only
  metal_level1-4/           # 26 Metal-specific problems — M4Max only
```

## Hardware Targets
| Target | GPU | VRAM | Problems | Execution |
|--------|-----|------|----------|-----------|
| rtx3090 | RTX 3090 (Ampere SM86) | 24GB | 43 | Local |
| h100 | H100 (Hopper SM90) | 80GB | 54 | Modal |
| b200 | B200 (Blackwell SM100) | 192GB | 58 | Modal |
| m4max | M4 Max | 128GB unified | 63 | macbook local |

## CUDA Versions
- `/usr/local/cuda` symlink → `/usr/local/cuda-13.2` (default)
- `/usr/local/cuda-12.6` also installed (PATH currently resolves here)
- cuTile supported on Ampere as of CUDA 13.2
- Driver: 595.45.04

## Model Registry (src/models.py)
All models route through OpenRouter except OpenAI direct (gpt-5.3, gpt-5.4):

| Key | Provider | Notes |
|-----|----------|-------|
| anthropic/claude-opus-4.6 | openrouter | Expensive ($5/$25 per M) |
| anthropic/claude-sonnet-4.6 | openrouter | ($3/$15 per M) |
| openai/gpt-5.3 | openai direct | model_id=gpt-5.3-chat-latest |
| openai/gpt-5.4 | openai direct | |
| openai/gpt-5.4-low | openai direct | reasoning_effort="low" |
| openai/gpt-5.4-high | openai direct | reasoning_effort="high" |
| google/gemini-3-flash-preview | openrouter | ($0.50/$3 per M) |
| google/gemini-3.1-pro-preview | openrouter | ($2/$12 per M) |
| deepseek/deepseek-v3.2 | openrouter | ($0.26/$0.38) — scores 0%, skip |
| z-ai/glm-5 | openrouter | ($0.72/$2.30) |
| minimax/minimax-m2.5 | openrouter | ($0.27/$0.95) |
| moonshotai/kimi-k2.5 | openrouter | ($0.45/$2.20), reasoning_mode=True |
| qwen/qwen3.5-397b-a17b | openrouter | ($0.39/$2.34) |

## Completed Evaluation Runs

### Coverage Matrix (correct/total)
| Model | RTX 3090 | H100 | B200 |
|-------|----------|------|------|
| GPT-5.4 | 33/43 (77%) | 42/54 (78%) | 50/58 (86%) |
| GPT-5.3 | 28/43 (65%) | 40/54 (74%) | 49/58 (84%) |
| Gemini 3 Flash | 32/43 (74%) | 41/54 (76%) | 46/58 (79%) |
| Kimi K2.5 | 22/43 (51%) | 27/54 (50%) | 35/58 (60%) |
| GLM-5 | 19/43 (44%) | 31/54 (57%) | 31/58 (53%) |
| Claude Opus 4.6 | 27/43 (63%) | 37/54 (69%) | 11/58 (19%) |
| Qwen3.5-397B | 13/43 (30%) | 22/54 (41%) | 25/58 (43%) |
| Gemini 3.1 Pro | 16/43 (37%) | 13/54 (24%) | 22/58 (38%) |
| Claude Sonnet 4.6 | 25/43 (58%) | 19/54 (35%) | 18/58 (31%) |
| MiniMax M2.5 | 35/43 (77%*) | 9/54 (17%) | 12/58 (21%) |
| MiniMax M2.7 | 9/43 (21%) | 14/54 (26%) | 8/58 (14%) |

*MiniMax RTX3090 had 129 results from possible multi-run merge

### Run Directory → Model/GPU Mapping
```
run_20260226_235356  Gemini 3 Flash    RTX3090  43 results  32 correct
run_20260227_030206  Gemini 3 Flash    H100     54 results  41 correct
run_20260227_035338  Gemini 3 Flash    B200     58 results  46 correct
run_20260227_044818  Claude Opus 4.6   RTX3090  43 results  27 correct
run_20260301_120228  DeepSeek V3.2     B200     58 results   2 correct
run_20260301_123244  GLM-5             B200     58 results  31 correct
run_20260302_204111  Kimi K2.5         B200     58 results  35 correct
run_20260309_032138  Qwen3 Coder Next  B200     58 results   5 correct
run_20260309_181804  Qwen3.5-35B-A3B   H100     54 results   0 correct
run_20260310_041756  Qwen3.5-122B-A10B H100     54 results  17 correct
run_20260311_133649  GLM-5             RTX3090  31 results  18 correct (INCOMPLETE)
run_20260311_213917  Kimi K2.5         RTX3090  43 results  22 correct
run_20260313_025234  Claude Sonnet 4.6 RTX3090  43 results  25 correct
run_20260313_033440  GPT-5.4           RTX3090  43 results  33 correct
run_20260313_034511  GPT-5.3           RTX3090  43 results  28 correct
run_20260313_040306  Gemini 3.1 Pro    RTX3090  43 results  16 correct
run_20260313_045040  DeepSeek V3.2     RTX3090  43 results   0 correct
run_20260313_234022  MiniMax M2.5      RTX3090  23 results   4 correct (partial)
run_20260314_004831  MiniMax M2.5      RTX3090 129 results  35 correct
run_20260314_023431  MiniMax M2.5      H100     54 results   9 correct
run_20260314_055031  GLM-5             H100    162 results  89 correct
run_20260315_065251  Kimi K2.5         H100     54 results  27 correct
run_20260315_105800  Qwen3.5-397B      H100     54 results  22 correct
run_20260316_095221  MiniMax M2.5      B200     58 results  12 correct
run_20260316_180349  Qwen3.5-397B      B200     58 results  25 correct
run_20260317_072632  GLM-5             RTX3090  43 results  19 correct
run_20260317_072633  GPT-5.4           H100     54 results  42 correct
run_20260317_084945  Qwen3.5 397B      RTX3090  43 results  13 correct
run_20260317_084946  GPT-5.3           H100     54 results  40 correct
run_20260317_091603  Claude Opus 4.6   H100     54 results  37 correct
run_20260317_101252  Claude Sonnet 4.6 H100     54 results  19 correct
run_20260317_110358  Gemini 3.1 Pro    H100     54 results  13 correct
run_20260317_121922  GPT-5.4           B200     58 results  50 correct
run_20260317_130201  GPT-5.3           B200     58 results  49 correct
run_20260317_132246  Claude Opus 4.6   B200     58 results  11 correct
run_20260317_134109  Claude Sonnet 4.6 B200     58 results  18 correct
run_20260317_142816  Gemini 3.1 Pro    B200     58 results  22 correct
run_20260318_095508  MiniMax M2.7      RTX3090  43 results   9 correct
run_20260318_095510  MiniMax M2.7      H100     54 results  14 correct
run_20260318_111835  MiniMax M2.7      B200     58 results   8 correct
```

## ALL RUNS COMPLETE
Coverage matrix is fully populated. No remaining runs.

## Recent Bug Fixes (this session)
1. **Bash guardrail bypass**: Both Gemini and standard agent loops in `src/eval/agent.py` handled `bash` as a special case calling `sandbox.run_command()` directly, bypassing `BLOCKED_COMMANDS` check in `_dispatch_tool`. Fixed by adding `BLOCKED_COMMANDS.search(cmd)` check inline in both paths.
2. **Overly aggressive `rm -rf` regex**: `rm\s+-rf\s+/` was blocking legitimate cache clears like `rm -rf /root/.cache/torch_extensions/...`. Narrowed to only block `rm -rf /` (root) and `rm -rf /workspace`, `/home`, `/usr`, etc.
3. **MiniMax M2.5 reward hacking**: Model attempted `pkill -f python` on RTX 3090 (first run), killing the evaluation process. Guardrail fix prevented this on subsequent runs.

## Key Design Decisions
- **Adaptive baseline**: `src/eval/benchmark.py` tries `torch.compile(mode='reduce-overhead')` and uses it only if >=5% faster than eager PyTorch
- **Weight sharing**: `sol_model.load_state_dict(ref_model.state_dict(), strict=False)` ensures fair comparison for models with learned params
- **Self-check**: Models run `torch.allclose` check before submitting, with `atol=1e-2, rtol=1e-2`
- **Hardware fingerprinting**: Every result includes GPU model, driver version, CUDA version via `src/eval/fingerprint.py`
- **Per-architecture prompts**: `src/prompts.py` injects WMMA (Ampere), WGMMA (Hopper), tcgen05 (Blackwell) guidance

## API Keys
All in `~/.env_vars`: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
Modal: `~/.modal.toml` (profile: elliot-2)

## Quick Commands
```bash
# List hardware targets
uv run python bench.py list-hardware

# List models
uv run python bench.py list-models

# Run eval
uv run python bench.py run <hardware> --models <model> --levels 1,2,3,4 --workers 4

# View summary
uv run python bench.py summary outputs/batch_eval/run_XXXXXXXX

# Linting
uv run --with ruff ruff check . --fix

# Tests
uv run pytest
```

## Cautions
- `outputs/` is large and gitignored — do not expect git to preserve run artifacts
- NVIDIA ComputeEval cloned at `/home/infatoshi/cuda/compute-eval/` for reference only (not integrated)
- M4 Max runs from separate `~/MetalBench` on macbook via ssh
