# AGENTS.md - KernelBench v3

## Project Location

```
/home/infatoshi/cuda/KernelBench-v3
```

---

## Compute Infrastructure

| Name | Hardware | Access | Use Case |
|------|----------|--------|----------|
| `theodolos` | RTX 3090, Ryzen 9 9950X, 96GB RAM | Local (current machine) | CUDA/Triton dev, 3090 baselines |
| `macbook` | Apple M4 Max | `ssh macbook` | Metal benchmarks |
| Modal | H100 SXM5 80GB | `modal run` | H100 baselines |
| Modal | B200 | `modal run` | Blackwell baselines, FP4 validation |

### Access Commands
```bash
# Metal/Apple Silicon testing
ssh macbook

# H100/B200 evaluation entrypoints
uv run python modal_eval.py --help
uv run python batch_eval.py --help
```

## Hardware Status

| Target | Status | Path |
|--------|--------|------|
| RTX 3090 | ✅ Working | LocalSandbox |
| H100 | ✅ Working | ModalSandbox |
| B200 | ✅ Working | ModalSandbox |
| M4 Max | ❌ Not implemented | Needs MetalSandbox |
| L40S | Untested | ModalSandbox |
| A100 | Untested | ModalSandbox |

---

## Backends

| Backend | Baseline | Extension |
|---------|----------|-----------|
| CUDA | PyTorch eager + torch.compile | `.cu` |
| Triton | PyTorch eager | `.py` |
| CuTe DSL | CUTLASS | `.cu` |
| CUTLASS | cuBLAS | `.cu` |
| Metal | MPS | `.metal` |
| PTX | cuBLAS | `.ptx` |

### Future (TODO)
- ROCm/HIP (AMD MI300)
- oneAPI/SYCL (Intel Arc)
- Vulkan Compute

---

## Problem Levels

| Level | Description | Count |
|-------|-------------|-------|
| 1 | Single ops (GEMM, conv, norm, activation) | 100 |
| 2 | Fused sequences (matmul+bias+act) | 100 |
| 3 | Full architectures (transformer blocks) | 50 |
| Graphics | Post-processing, particles, neural rendering | New |

### Graphics Problems (New)
- Bloom, DoF, motion blur, SSAO, TAA (compute shaders)
- Particle simulation/sorting
- Software rasterization
- DLSS-style CNN/Swin upscaling
- Ray tracing denoiser

---

## Precision Modes

| Precision | Hardware | Notes |
|-----------|----------|-------|
| FP32 | All | Default |
| FP16/BF16 | All modern | Common inference/training |
| TF32 | Ampere+ | cuBLAS default |
| FP8 | Hopper+ | Transformer Engine |
| **FP4 (NVFP4)** | **Blackwell** | **Requires CUTLASS 3.x baseline validation** |
| INT8/INT4 | Hopper+ | Quantized |

### FP4 Validation (Critical)
```
1. Run FP32 PyTorch reference → golden output
2. Run CUTLASS FP4 → compare to golden with tolerance
3. If CUTLASS passes, use as baseline for LLM kernels
```
Do NOT trust FP4 results without this validation step.

---

## Validation Protocol

**Before full eval, run ONE problem per level per backend per hardware.**

Use this minimal validation pattern for planning checks:

```bash
uv run python batch_eval.py --models gemini-3-flash --gpus H100 --levels 1 --problems-per-level 1 --dry-run
```

Key flags:
- `--dry-run` avoids compute spend while verifying task selection and wiring
- `--problems-per-level 1` keeps validation minimal

| Level | Validation Problem |
|-------|-------------------|
| 1 | `gemm_nn` (basic matmul) |
| 2 | `matmul_bias_relu` |
| 3 | `transformer_block` |
| Graphics | `bloom` |

### Validation Checklist
- [ ] PyTorch reference runs on all hardware
- [ ] Baseline timing collected
- [ ] CUDA backend works on theodolos (3090)
- [ ] CUDA backend works on Modal H100
- [ ] CUDA backend works on Modal B200
- [ ] Triton backend works on theodolos
- [ ] Metal backend works via `ssh macbook`
- [ ] CuTe/CUTLASS works on Hopper+
- [ ] FP4 validation passes on B200
- [ ] Correctness checker catches errors
- [ ] Timing variance is low
- [ ] Modal billing checked

**Do NOT run full eval until all boxes checked.**

## Baseline Reference Times (Level 1 GEMM)

| Hardware | ref_ms | Source |
|----------|--------|--------|
| RTX 3090 | 0.7175 | Local run |
| H100 | 0.3464 | Modal run |
| B200 | 0.3127 | Modal run |

---

## Metrics Schema

```python
@dataclass
class KernelSubmission:
    # Identity
    task_id: str
    backend: str  # cuda, triton, cute, cutlass, metal, ptx
    level: int
    category: str  # gemm, conv, attention, bloom, etc.
    hardware: str  # h100, b200, rtx3090, m4max
    dtype: str  # fp32, bf16, fp8, fp4
    
    # Code
    reference_code: str
    generated_code: str
    input_shapes: List[Tuple]
    
    # Generation
    model_used: str
    prompt_config: dict
    turn_number: int
    feedback_given: Optional[str]
    trajectory_id: Optional[str]
    
    # Compilation
    compiled: bool
    compile_error: Optional[str]
    compile_time_ms: float
    
    # Correctness
    correct: bool
    max_abs_error: float
    mean_abs_error: float
    pct_within_tolerance: float
    
    # Performance
    kernel_time_ms: float
    baseline_eager_time_ms: float
    baseline_compile_time_ms: float
    speedup_vs_eager: float
    speedup_vs_compile: float
    achieved_tflops: Optional[float]
    achieved_bandwidth_gbps: Optional[float]
    peak_utilization_pct: Optional[float]
```

### Aggregate Metrics
```python
# fast_p: fraction correct AND speedup > p
fast_p = correct_and_faster_than_p / total

# fast_p@k: best of k attempts
fast_p_at_k = any_of_k_correct_and_faster / total_problems
```

---

## Modal Configuration

```python
import modal

app = modal.App("kernelbench-v3")

@app.function(gpu="H100", timeout=600)
def run_h100_eval(problem_id: str, generated_code: str) -> dict:
    ...

@app.function(gpu="B200", timeout=600)
def run_b200_eval(problem_id: str, generated_code: str, dtype: str = "fp32") -> dict:
    if dtype == "fp4":
        validate_fp4_baseline(problem_id)  # REQUIRED
    ...
```

---

## Prompting Strategy

Few-shot with:
- Problem specification (PyTorch reference)
- Input shapes and dtypes
- Target hardware specs (SMs, bandwidth, peak TFLOPS)
- Few-shot examples (tiling, fusion, shared memory, tensor cores)
- Profiler tools available

Variable turns per problem with feedback loop.

### Hardware Specs Format
```yaml
name: NVIDIA H100 SXM5 80GB
compute_capability: 9.0
sm_count: 132
shared_memory_per_sm_kb: 228
memory_bandwidth_gbps: 3350
fp32_tflops: 67
tf32_tflops: 989
fp16_tflops: 1979
fp8_tflops: 3958
```

---

## Iterative Refinement Loop

```python
for turn in range(max_turns):
    generated_code = llm_generate(context)
    result = evaluate_kernel(problem_id, generated_code, ...)
    
    if result.correct and result.speedup_vs_eager > 1.0:
        break
    
    # Build feedback
    if not result.compiled:
        feedback = f"Compile error:\n{result.compile_error}"
    elif not result.correct:
        feedback = f"Incorrect. Max error: {result.max_abs_error}"
    else:
        feedback = f"Correct but slow. Speedup: {result.speedup_vs_eager:.2f}x\n{profiler_output}"
    
    context = update_context(context, generated_code, feedback)
```

---

## Quick Commands

```bash
# Validate infrastructure wiring (run first, no remote execution)
uv run python batch_eval.py --models gemini-3-flash --gpus H100 --levels 1 --problems-per-level 1 --dry-run

# Validate one task per level (still dry-run)
uv run python batch_eval.py --models gemini-3-flash --gpus H100 --levels 1,2,3,4 --problems-per-level 1 --dry-run

# List available options
uv run python batch_eval.py --list-models
uv run python batch_eval.py --list-problems

# Single problem eval (actual execution)
uv run python modal_eval.py --model gemini-3-flash --gpu H100 --problem level1/1_Square_matrix_multiplication_.py

# Small batch eval (actual execution)
uv run python batch_eval.py --models gemini-3-flash --gpus H100 --levels 1 --problems-per-level 1

# Resume interrupted run
uv run python batch_eval.py --resume outputs/batch_eval/run_YYYYMMDD_HHMMSS

# Check Modal
modal app list
modal app logs kernelbench-v3
```

## Frontier Models (OpenRouter)

Validated working:
- deepseek/deepseek-chat (DeepSeek V3)
- anthropic/claude-sonnet-4

Needs manual validation (run outside Codex):
- google/gemini-2.5-pro-preview
- google/gemini-2.5-flash-preview
- anthropic/claude-opus-4
- moonshotai/kimi-k2
- zhipu/glm-4-plus
- z-ai/glm-5
- openrouter/aurora-alpha

To find correct model IDs:
```bash
curl -s "https://openrouter.ai/api/v1/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | \
  jq '.data[].id' | grep -i "gemini\|claude\|kimi\|glm\|deepseek"
```

### CUTLASS/CuTe Model Viability (H100, Level 1 GEMM)

| Model | CUTLASS compiled | CUTLASS speedup | CuTe compiled | CuTe speedup |
|-------|------------------|-----------------|---------------|--------------|
| anthropic/claude-sonnet-4 | true | 0.9943x | true | N/A (incorrect, max_diff=243.18) |
| google/gemini-2.5-pro-preview | false | N/A | false | N/A |
| z-ai/glm-5 | false | N/A | timeout/no submission | N/A |
| openrouter/aurora-alpha | false | N/A | false | N/A |
| deepseek/deepseek-chat | false | N/A | false | N/A |

Notes:
- Models that produced compiling CUTLASS submissions in this table: `anthropic/claude-sonnet-4`.
- Models that produced compiling CuTe submissions in this table: `anthropic/claude-sonnet-4` only, but result was incorrect (max_diff=243.18).
- `z-ai/glm-5` did not reproduce the prior Pony CUTLASS result and failed CUTLASS compilation in latest validation.
- CUTLASS/CuTe viability is model-dependent; DeepSeek, Gemini, and Aurora are currently unreliable on these template-heavy APIs.

### CuTe GEMM Benchmark Status

**Difficulty: Frontier (No current model passes)**

Best attempt: `anthropic/claude-sonnet-4`
- `compiled=true`
- `correct=false` (`max_diff=243.18`)

#### Identified Capability Gaps
1. **TiledMMA not used**: Models generate basic GEMM loops instead of using CuTe `TiledMMA` for warp-level tensor-core computation.
2. **TiledCopy not properly used**: Attempts are superficial; models do not reliably apply `get_slice` + `partition_S`/`partition_D` for cooperative thread mapping.
3. **Invalid launch configurations**: Models emit launches that exceed CUDA limits (for example `dim3(64,64)=4096` threads, max is 1024).
4. **Missing `partition_A/B/C` idiom**: The `sgemm_2.cu` accumulator partitioning pattern is not reliably reproduced.
5. **No CUDA error checking**: Launch failures are often not checked, so kernels can return zero-initialized output silently.

#### What Would Be Needed To Pass
- `TiledMMA` with valid MMA atom selection (for example `SM90_16x8x16_F32F16F16F32`-class atoms).
- `TiledCopy` for cooperative global-to-shared loads.
- Correct thread partitioning via `get_slice()`.
- Block size derived from `size(mmaC)`, not hardcoded.
- Correct accumulator fragment lifecycle (init, accumulate, writeback).

#### Tracking Progress
This benchmark measures whether models can:
1. Understand modern GPU programming abstractions.
2. Reason about thread/warp/block hierarchies.
3. Apply complex C++ template APIs correctly.
4. Generate numerically correct parallel algorithms.

Future models that pass CuTe GEMM indicate significant capability advancement.

#### Diagnostic Note
Current diagnostics indicate both Claude submissions produced mostly-zero outputs from fundamental kernel errors (partial writes and/or invalid launch), which explains the consistent `max_diff` near `243` across runs.

#### Benchmark Policy
- Do not simplify CuTe GEMM for pass-rate optimization.
- Do not add easier CuTe stepping-stone tasks in place of this benchmark.
- Do not reduce matrix size for this capability gate.
- Keep CuTe GEMM as a hard benchmark for future-model capability measurement.

### CuTe Prompt Sync Policy
- Keep `src/prompts/cute_system.py` aligned with current CUTLASS tutorial idioms (`/opt/cutlass/examples/cute/tutorial/sgemm_1.cu`, `sgemm_2.cu`, `sgemm_sm80.cu`).
- When CUTLASS is bumped in `src/agent/modal_sandbox.py`, re-extract tutorial files and refresh the CuTe few-shot snippet before benchmark sweeps.
- Pin CuTe include path usage to `/opt/cutlass/include` in prompt templates and generated wrappers.

---

## Development Phases

### Phase 1: Infrastructure
1. CUDA backend runner on theodolos
2. Modal integration for H100/B200
3. Run validation suite
4. Fix issues

### Phase 2: Baselines
1. RTX 3090 baselines (local)
2. H100 baselines (Modal)
3. M4 Max baselines (`ssh macbook`)
4. FP4 validation on B200

### Phase 3: Additional Backends
1. Triton
2. Metal
3. CuTe/CUTLASS

### Phase 4: Problem Expansion
1. Graphics/post-processing
2. Neural rendering (DLSS variants)
3. Additional Level 1-3 problems

### Phase 5: Release
1. HuggingFace dataset
2. Leaderboard
3. Documentation

---

## TODO

### Implemented ✅
- [x] Local GPU runner (RTX 3090)
- [x] Modal GPU runner (H100, B200)
- [x] OpenRouter integration
- [x] CUDA backend
- [x] Level 1/2/3 problem structure
- [x] Triton-specific backend runner
- [x] Metal backend runner (MLX-based prototype)
- [x] CUTLASSBench scaffold
- [x] CuTeBench scaffold
- [x] GraphicsBench scaffold (bloom + particles, level 1)

### Not Implemented
- [ ] Full graphics problem suite (DoF, motion blur, TAA, SSAO, DLSS)
- [ ] CuTe/CUTLASS expert baselines (curated, hand-tuned reference kernels)
- [ ] ROCm/HIP (AMD)

### Needs Manual Validation
- [ ] Frontier model OpenRouter IDs
- [ ] FP4 precision on B200
- [ ] CUTLASSBench end-to-end compile+correct on H100
- [ ] CuTeBench end-to-end compile+correct on H100
- [x] GraphicsBench level 1 POC (bloom) compile+correct on RTX3090

### Detailed Backlog (Legacy)

### Infrastructure
- [ ] CUDA backend runner
- [ ] Modal H100 dispatch
- [ ] Modal B200 dispatch
- [ ] Metal runner via SSH
- [ ] Validation script

### Baselines
- [ ] RTX 3090 Level 1-3
- [ ] H100 Level 1-3
- [ ] B200 Level 1-3 + FP4
- [ ] M4 Max Level 1-3

### Backends
- [x] Triton backend
- [x] Metal backend (MLX prototype)
- [x] CuTe DSL backend scaffold
- [x] CUTLASS backend scaffold

### Problems
- [x] Graphics: bloom
- [ ] Graphics: motion blur
- [ ] Graphics: SSAO
- [ ] Graphics: TAA
- [x] Graphics: particles
- [ ] Neural rendering: DLSS CNN
- [ ] Neural rendering: DLSS Swin

### Hardware (Future)
- [ ] AMD MI300 (ROCm/HIP)
- [ ] Intel Arc (oneAPI)

### Dataset
- [ ] HuggingFace schema
- [ ] Submission pipeline
- [ ] Trajectory storage

---

## Deployment and Maintenance

### GPU Restrictions (Release Policy)

| Benchmark | Allowed GPUs | Why |
|-----------|--------------|-----|
| CUDABench (`batch_eval.py`) | `RTX3090`, `H100`, `B200` | Validated CUDA paths only |
| TritonBench (`triton_batch_eval.py`) | `RTX3090`, `H100`, `B200` | Triton validated on NVIDIA CUDA targets |
| CUTLASSBench (`cutlass_batch_eval.py`) | `H100`, `B200` | Hopper/Blackwell tensor-core tuning target |
| CuTeBench (`cute_batch_eval.py`) | `H100`, `B200` | Hopper/Blackwell tensor-core abstractions |
| CuTileBench (`cutile_batch_eval.py`) | `B200` | Current tileiras runtime constraint |
| MetalBench (`metal_batch_eval.py`) | `M4MAX` | Apple Metal/MLX only |
| GraphicsBench (`graphics_batch_eval.py`) | `RTX3090` | CUDA/Triton graphics compute workloads |

Platform enforcement:
- `M4MAX` must run on `Darwin`.
- NVIDIA benchmark paths are Linux-only in this harness.

### How To Add A New GPU
1. Update benchmark GPU policies in relevant `*_batch_eval.py` (`ALLOWED_GPUS` and `GPU_REASON`).
2. Update benchmark mapping in `src/config/benchmark_problems.py` (`BENCHMARK_PROBLEMS[...]["hardware"]`).
3. Add/update GPU specs in `modal_eval.py` (`GPU_SPECS`) and any sandbox routing.
4. Validate with `--dry-run` first, then one real single-problem run.

### How To Add A New Benchmark
1. Create `<name>_eval.py` with prompt overrides and problem discovery.
2. Create `<name>_batch_eval.py` with explicit `ALLOWED_GPUS` and runtime validation.
3. Register benchmark problem dirs/hardware in `src/config/benchmark_problems.py`.
4. Add benchmark to `run_all_benchmarks.py` (`BENCHMARK_CONFIG`).
5. Update `README.md` and `problem_inventory.md`.

### How To Add A New Model
1. Add model entry to `MODELS` in `modal_eval.py`.
2. Ensure provider routing/API key path works.
3. Validate with `uv run python batch_eval.py --models <model> --gpus <valid_gpu> --levels 1 --problems-per-level 1 --dry-run`.
4. Update `README.md` model list if this is a release-facing model.

### Common Failure Modes and Fixes
- `GPU not supported`:
  Use benchmark-specific allowed GPUs shown by the script error output.
- `platform mismatch`:
  - Move Metal runs to macOS (`M4MAX`).
  - Move NVIDIA runs to Linux environment/Modal.
- `No solution submitted`:
  Increase `--max-turns`, inspect `outputs/.../turns/` artifacts.
- `Compilation errors`:
  Verify prompt/backend alignment (CUDA vs Triton vs CUTLASS vs CuTe vs CuTile vs Metal).
- `Incorrect output`:
  Check tolerance/precision metadata and inspect `max_diff` plus per-turn compile logs.

## Critical Reminders

1. **Run validation before full eval** - saves compute credits
2. **FP4 requires CUTLASS validation** - don't trust raw results
3. **Check Modal billing** after test runs
4. **Metal may need adjusted tolerances** - different numerical behavior
5. **Keep trajectory data** for multi-turn analysis

---

## Repository

`/home/infatoshi/cuda/KernelBench-v3`

Owner: Infatoshi (Elliot Arledge)
