# 2025-10-22 Ensure Provider Preflight

## rationale
Prevent expensive benchmarks from starting with missing or misconfigured API credentials, and harden provider adapters (especially Gemini) so preflight probes succeed while surfacing actionable failures.

## patch
```diff
diff --git a/src/providers/base.py b/src/providers/base.py
@@
-    #: Whether this provider requires an API key to operate.
-    requires_api_key: bool = True
-    #: Default message used for the lightweight preflight ping.
-    preflight_message: str = "KernelBench preflight ping"
+    #: Whether this provider requires an API key to operate.
+    requires_api_key: bool = True
+    #: Default message used for the lightweight preflight ping.
+    preflight_message: str = "KernelBench preflight ping"
+    #: Maximum tokens requested during preflight checks.
+    preflight_max_output_tokens: int | None = 1
@@
-        """Execute the default preflight chat completion."""
-        self.generate(
-            [{"role": "user", "content": self.preflight_message}],
-            temperature=0.0,
-            max_tokens=1,
-        )
+        """Execute the default preflight chat completion."""
+        kwargs: Dict[str, Any] = {"temperature": 0.0}
+        if self.preflight_max_output_tokens is not None:
+            kwargs["max_tokens"] = self.preflight_max_output_tokens
+        self.generate(
+            [{"role": "user", "content": self.preflight_message}],
+            **kwargs,
+        )
diff --git a/src/providers/__init__.py b/src/providers/__init__.py
@@
-from typing import Dict, Type
+import os
+from typing import Dict, Iterable, Type
@@
 PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
@@
     "ollama": OllamaProvider,
 }
 
 
+_PROVIDER_API_KEY_ALIASES: Dict[str, Iterable[str]] = {
+    "openai": ("OPENAI_API_KEY",),
+    "groq": ("GROQ_API_KEY",),
+    "anthropic": ("ANTHROPIC_API_KEY",),
+    "xai": ("XAI_API_KEY",),
+    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"),
+    "cerebras": ("CEREBRAS_API_KEY",),
+    "vllm": ("VLLM_API_KEY",),
+    "sglang": ("SGLANG_API_KEY",),
+    "ollama": ("OLLAMA_API_KEY",),
+}
+
+
+def resolve_provider_api_key(provider: str) -> str | None:
+    """Resolve provider API keys from common environment variable names."""
+    provider_slug = provider.lower()
+    scoped_key = f"KB3_LLM_API_KEY_{provider_slug.upper()}"
+    candidates = list(_PROVIDER_API_KEY_ALIASES.get(provider_slug, ()))
+    candidates.extend([scoped_key, "KB3_LLM_API_KEY"])
+
+    for env_name in candidates:
+        value = os.getenv(env_name)
+        if value:
+            return value
+    return None
+
+
 def create_provider(config: ProviderConfig) -> BaseProvider:
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-from providers import ProviderConfig, create_provider
+from providers import ProviderConfig, create_provider, resolve_provider_api_key
@@
-    provider_name, model_id = config.provider, config.generator_model
-    api_key = os.getenv("KB3_LLM_API_KEY") or os.getenv(f"KB3_LLM_API_KEY_{provider_name.upper()}")
+    provider_name, model_id = config.provider, config.generator_model
+    api_key = resolve_provider_api_key(provider_name)
diff --git a/src/agentic/runner/main.py b/src/agentic/runner/main.py
@@
-from providers import ProviderConfig, create_provider
+from providers import ProviderConfig, create_provider, resolve_provider_api_key
@@
-    api_key = os.getenv("KB3_LLM_API_KEY") or os.getenv(f"KB3_LLM_API_KEY_{provider_name.upper()}")
+    api_key = resolve_provider_api_key(provider_name)
diff --git a/src/providers/sglang_provider.py b/src/providers/sglang_provider.py
@@
 class SGLangProvider(BaseProvider):
-    def __init__(self, config: ProviderConfig) -> None:
+    requires_api_key = False
+
+    def __init__(self, config: ProviderConfig) -> None:
diff --git a/src/providers/vllm_provider.py b/src/providers/vllm_provider.py
@@
 class VLLMProvider(BaseProvider):
-    def __init__(self, config: ProviderConfig) -> None:
+    requires_api_key = False
+
+    def __init__(self, config: ProviderConfig) -> None:
diff --git a/src/providers/ollama_provider.py b/src/providers/ollama_provider.py
@@
 class OllamaProvider(BaseProvider):
-    def __init__(self, config: ProviderConfig) -> None:
+    requires_api_key = False
+
+    def __init__(self, config: ProviderConfig) -> None:
diff --git a/src/providers/anthropic_provider.py b/src/providers/anthropic_provider.py
@@
-    def preflight(self) -> None:
-        try:
-            self.client.messages.create(
-                model=self.config.model,
-                system="",
-                messages=[{"role": "user", "content": "ping"}],
-                max_tokens=1,
-            )
-        except Exception as exc:  # noqa: BLE001
-            raise RuntimeError(
-                f"Anthropic preflight check failed for model '{self.config.model}': {exc}"
-            ) from exc
+    def _perform_preflight_request(self) -> None:
+        self.client.messages.create(
+            model=self.config.model,
+            system="",
+            messages=[{"role": "user", "content": "ping"}],
+            max_tokens=1,
+        )
diff --git a/src/providers/groq_provider.py b/src/providers/groq_provider.py
@@
-    def preflight(self) -> None:
-        try:
-            self.client.chat.completions.create(
-                model=self.config.model,
-                messages=[{"role": "user", "content": "ping"}],
-                max_tokens=1,
-                temperature=0.0,
-            )
-        except Exception as exc:  # noqa: BLE001
-            raise RuntimeError(
-                f"Groq preflight check failed for model '{self.config.model}': {exc}"
-            ) from exc
+    def _perform_preflight_request(self) -> None:
+        self.client.chat.completions.create(
+            model=self.config.model,
+            messages=[{"role": "user", "content": "ping"}],
+            max_tokens=1,
+            temperature=0.0,
+        )
diff --git a/src/providers/gemini_provider.py b/src/providers/gemini_provider.py
@@
-class GeminiProvider(BaseProvider):
-    def __init__(self, config: ProviderConfig) -> None:
+class GeminiProvider(BaseProvider):
+    preflight_max_output_tokens = None
+
+    def __init__(self, config: ProviderConfig) -> None:
@@
-        max_tokens: int = 1024,
+        max_tokens: int | None = None,
@@
-        response = self.model.generate_content(
-            prompt,
-            generation_config=genai.types.GenerationConfig(
-                temperature=temperature,
-                max_output_tokens=max_tokens,
-                **kwargs,
-            ),
-        )
+        generation_config = genai.types.GenerationConfig(
+            temperature=temperature,
+            **kwargs,
+        )
+        if max_tokens is not None:
+            generation_config.max_output_tokens = max_tokens
+
+        response = self.model.generate_content(
+            prompt,
+            generation_config=generation_config,
+        )
@@
-            raise RuntimeError(
-                "Gemini response contained no textual parts; "
-                f"finish_reasons={finish_reasons}, safety_ratings={safety_blocks}"
-            )
+            raise RuntimeError(
+                "Gemini response contained no textual parts; "
+                f"finish_reasons={finish_reasons}, safety_ratings={safety_blocks}"
+            )
diff --git a/configs/openai_groq.yaml b/configs/openai_groq.yaml
new file mode 100644
@@
+description: OpenAI vs Groq comparison (CUDA + Triton, raw mode default)
+
+modes:
+  - raw
+
+languages:
+  - cuda
+  - triton
+
+models:
+  - provider: gemini
+    model: gemini-2.5-flash
+    raw_concurrency: 4
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 4
+  - provider: groq
+    model: moonshotai/kimi-k2-instruct-0905
+    raw_concurrency: 4
+    raw_gpu_concurrency: 4
+    raw_max_jobs: 4
+
+defaults:
+  num_runs: 100
+  profile_stages: false
+  raw:
+    cpu_concurrency: 4
+    gpu_concurrency: 4
+    max_jobs: 4
+
+verbose: false
+profile_stages: false
+fast_p_threshold: null
+raw_concurrency: 4
+raw_gpu_concurrency: 4
+raw_max_jobs: 4
+
+hardware:
+  gpu_architecture: Ampere
+  gpu_id: 0
+
+problems:
+  levels: [1, 2]
+  problem_ids: null
+  max_problems: 100
+
+agentic:
+  max_debug_attempts: 3
+  max_optimization_cycles: 2
+  reflector_model: gpt-4-turbo
+  optimizer_model: gpt-4-turbo
+
+artifacts:
+  json_dir: json
+  plots_dir: plots
+
+visualization:
+  enabled: false
```

## validate
- `uv run python eval.py --config configs/openai_groq.yaml ` (Gemini and Groq both preflight, runs hit cache, confirming env alias resolution and Gemini fallback logic).

# 2025-10-23 Isolate Raw GPU Evaluations

## rationale
Prevent a single illegal CUDA memory access from corrupting the shared GPU context and cascading across the rest of the benchmark by executing each candidate evaluation inside a short-lived worker process and preserving profiling data.

## patch
```diff
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-def _gpu_consumer(
-    queue,
-    results_store: Dict[int, Dict[str, Any]],
-    config: "BenchmarkConfig",
-    progress_state: Dict[str, Any] | None = None,
-    build_root: Path | None = None,
-) -> None:
-    if config.hardware.gpu_architecture and not FAKE_LLM:
-        set_gpu_arch([config.hardware.gpu_architecture])
-
-    while True:
-        candidate = queue.get()
-        if candidate is None:
-            break
-
-        problem_id = candidate["problem_id"]
-        try:
-            profile_enabled = config.profile_stages
-            cpu_profile = None
-            enqueue_ts = None
-            gpu_profile: Dict[str, float] | None = {} if profile_enabled else None
-
-            candidate_profile = candidate.get("profile") if profile_enabled else None
-            if candidate_profile:
-                cpu_profile = candidate_profile.get("cpu")
-                enqueue_ts = candidate_profile.get("enqueue_ts")
-
-            start_time = perf_counter()
-            if gpu_profile is not None and enqueue_ts is not None:
-                gpu_profile["queue_wait_s"] = start_time - enqueue_ts
-
-            build_dir = candidate.get("build_dir")
-            if build_dir is None and build_root is not None:
-                build_dir = str(Path(build_root) / f"problem_{problem_id}")
-            if build_dir is not None:
-                Path(build_dir).mkdir(parents=True, exist_ok=True)
-
-            result = _run_gpu_evaluation(candidate, config, profile=gpu_profile)
-
-            if gpu_profile is not None and "gpu_total_s" not in gpu_profile:
-                gpu_profile["gpu_total_s"] = perf_counter() - start_time
-
-            if profile_enabled:
-                stage_times: Dict[str, float] = {}
-                cpu_total = 0.0
-                queue_wait = 0.0
-                gpu_total = 0.0
-
-                if cpu_profile:
-                    cpu_total = cpu_profile.get("cpu_total_s", 0.0)
-                    stage_times["cpu_total_s"] = cpu_total
-                    for key, value in cpu_profile.items():
-                        if key != "cpu_total_s":
-                            stage_times[key] = value
-
-                if gpu_profile:
-                    queue_wait = gpu_profile.get("queue_wait_s", 0.0)
-                    gpu_total = gpu_profile.get("gpu_total_s", 0.0)
-                    stage_times["queue_wait_s"] = queue_wait
-                    stage_times["gpu_total_s"] = gpu_total
-                    for key, value in gpu_profile.items():
-                        if key not in {"queue_wait_s", "gpu_total_s"} and key.endswith("_s"):
-                            stage_times[key] = value
-
-                total_time = cpu_total + queue_wait + gpu_total
-                stage_times["total_s"] = total_time
-
-                percentages = {}
-                if total_time > 0:
-                    for name in ("cpu_total_s", "queue_wait_s", "gpu_total_s"):
-                        value = stage_times.get(name)
-                        if value is None:
-                            continue
-                        percentages[f"{name[:-2]}_pct"] = round((value / total_time) * 100, 4)
-
-                result["profiling"] = {
-                    "times_s": {k: round(v, 6) for k, v in stage_times.items()},
-                    "percentages": percentages,
-                }
-
-            results_store[problem_id] = result
-        except Exception as exc:  # noqa: BLE001
-            results_store[problem_id] = {
-                "level": candidate["level"],
-                "problem_id": problem_id,
-                "problem_name": candidate["problem_name"],
-                "compiled": False,
-                "correctness": False,
-                "runtime": None,
-                "metadata": {
-                    "error": str(exc),
-                    "exception_type": type(exc).__name__,
-                    "stage": "evaluation",
-                },
-                "generated_code": candidate["generated_code"],
-            }
-        finally:
-            _update_progress(progress_state)
+def _evaluate_candidate(
+    candidate: Dict[str, Any],
+    config: "BenchmarkConfig",
+    build_root: Path | None = None,
+) -> Dict[str, Any]:
+    profile_enabled = config.profile_stages
+    cpu_profile = None
+    enqueue_ts = None
+    gpu_profile: Dict[str, float] | None = {} if profile_enabled else None
+
+    if profile_enabled:
+        candidate_profile = candidate.get("profile") or {}
+        cpu_profile = candidate_profile.get("cpu")
+        enqueue_ts = candidate_profile.get("enqueue_ts")
+
+    start_time = perf_counter()
+    if gpu_profile is not None and enqueue_ts is not None:
+        gpu_profile["queue_wait_s"] = start_time - enqueue_ts
+
+    problem_id = candidate["problem_id"]
+    build_dir = candidate.get("build_dir")
+    if build_dir is None and build_root is not None:
+        build_dir = str(Path(build_root) / f"problem_{problem_id}")
+    if build_dir is not None:
+        Path(build_dir).mkdir(parents=True, exist_ok=True)
+
+    candidate_for_eval = dict(candidate)
+    if build_dir is not None:
+        candidate_for_eval["build_dir"] = build_dir
+
+    try:
+        result = _run_gpu_evaluation(candidate_for_eval, config, profile=gpu_profile)
+
+        if gpu_profile is not None and "gpu_total_s" not in gpu_profile:
+            gpu_profile["gpu_total_s"] = perf_counter() - start_time
+
+        if profile_enabled:
+            stage_times: Dict[str, float] = {}
+            cpu_total = 0.0
+            queue_wait = 0.0
+            gpu_total = 0.0
+
+            if cpu_profile:
+                cpu_total = cpu_profile.get("cpu_total_s", 0.0)
+                stage_times["cpu_total_s"] = cpu_total
+                for key, value in cpu_profile.items():
+                    if key != "cpu_total_s":
+                        stage_times[key] = value
+
+            if gpu_profile:
+                queue_wait = gpu_profile.get("queue_wait_s", 0.0)
+                gpu_total = gpu_profile.get("gpu_total_s", 0.0)
+                stage_times["queue_wait_s"] = queue_wait
+                stage_times["gpu_total_s"] = gpu_total
+                for key, value in gpu_profile.items():
+                    if key not in {"queue_wait_s", "gpu_total_s"} and key.endswith("_s"):
+                        stage_times[key] = value
+
+            total_time = cpu_total + queue_wait + gpu_total
+            stage_times["total_s"] = total_time
+
+            percentages = {}
+            if total_time > 0:
+                if cpu_total:
+                    percentages["cpu_total_pct"] = round((cpu_total / total_time) * 100, 4)
+                if queue_wait:
+                    percentages["queue_wait_pct"] = round((queue_wait / total_time) * 100, 4)
+                if gpu_total:
+                    percentages["gpu_total_pct"] = round((gpu_total / total_time) * 100, 4)
+
+            result["profiling"] = {
+                "times_s": {k: round(v, 6) for k, v in stage_times.items()},
+                "percentages": percentages,
+            }
+
+        return result
+    except Exception as exc:  # noqa: BLE001
+        return {
+            "level": candidate.get("level"),
+            "problem_id": candidate.get("problem_id"),
+            "problem_name": candidate.get("problem_name"),
+            "compiled": False,
+            "correctness": False,
+            "runtime": None,
+            "metadata": {
+                "error": str(exc),
+                "exception_type": type(exc).__name__,
+                "stage": "evaluation",
+            },
+            "generated_code": candidate.get("generated_code"),
+        }
+
+
+def _gpu_process_entry(
+    candidate: Dict[str, Any],
+    config: "BenchmarkConfig",
+    build_root: str | None,
+) -> tuple[int, Dict[str, Any]]:
+    problem_id = candidate.get("problem_id")
+    build_root_path = Path(build_root) if build_root else None
+
+    try:
+        if config.hardware.gpu_architecture and not FAKE_LLM:
+            set_gpu_arch([config.hardware.gpu_architecture])
+        result = _evaluate_candidate(candidate, config, build_root_path)
+    except Exception as exc:  # noqa: BLE001
+        result = {
+            "level": candidate.get("level"),
+            "problem_id": problem_id,
+            "problem_name": candidate.get("problem_name"),
+            "compiled": False,
+            "correctness": False,
+            "runtime": None,
+            "metadata": {
+                "error": str(exc),
+                "exception_type": type(exc).__name__,
+                "stage": "evaluation",
+            },
+            "generated_code": candidate.get("generated_code"),
+        }
+
+    return problem_id, result
@@
-    gpu_threads = []
-    for idx in range(gpu_worker_count):
-        thread = threading.Thread(
-            target=_gpu_consumer,
-            name=f"raw-gpu-consumer-{idx}",
-            args=(job_queue, shared_results, config, progress_state, build_root),
-            daemon=True,
-        )
-        thread.start()
-        gpu_threads.append(thread)
+    process_executor_kwargs = {
+        "max_workers": gpu_worker_count,
+        "mp_context": ctx,
+    }
+    try:
+        gpu_executor = ProcessPoolExecutor(max_tasks_per_child=1, **process_executor_kwargs)
+    except TypeError:
+        gpu_executor = ProcessPoolExecutor(**process_executor_kwargs)
+
+    pending_futures_lock = threading.Lock()
+    pending_futures: set = set()
+    dispatch_done = threading.Event()
+
+    def _handle_gpu_future(future):
+        try:
+            problem_id, result = future.result()
+        except Exception as exc:  # noqa: BLE001
+            if config.verbose:
+                print(f"[Raw] GPU worker raised unexpected error: {exc}")
+            return
+
+        if problem_id is None:
+            return
+
+        shared_results[problem_id] = result
+        _update_progress(progress_state)
+        with pending_futures_lock:
+            pending_futures.discard(future)
+
+    def _gpu_dispatch_loop() -> None:
+        sentinel_seen = 0
+        while True:
+            candidate = job_queue.get()
+            if candidate is None:
+                sentinel_seen += 1
+                if sentinel_seen >= gpu_worker_count:
+                    break
+                continue
+
+            future = gpu_executor.submit(
+                _gpu_process_entry,
+                candidate,
+                config,
+                str(build_root),
+            )
+            with pending_futures_lock:
+                pending_futures.add(future)
+            future.add_done_callback(_handle_gpu_future)
+
+        dispatch_done.set()
+
+    dispatch_thread = threading.Thread(
+        target=_gpu_dispatch_loop,
+        name="raw-gpu-dispatch",
+        daemon=True,
+    )
+    dispatch_thread.start()
@@
-    for _ in range(gpu_worker_count):
-        job_queue.put(None)
-
-    for thread in gpu_threads:
-        thread.join()
+    for _ in range(gpu_worker_count):
+        job_queue.put(None)
+
+    dispatch_thread.join()
+    dispatch_done.wait()
+    gpu_executor.shutdown(wait=True)
```

## validate
- `uv run python eval.py --config configs/openai_groq.yaml  --profile-stages --verbose`

# 2025-10-23 Fused Provider Config

## rationale
Unify all target providers into a single raw benchmark configuration (CUDA and Triton) to drive one-pass credential smoke tests before committing to full KernelBench sweeps.

## patch
```diff
diff --git a/configs/all_providers.yaml b/configs/all_providers.yaml
new file mode 100644
+description: Unified benchmark config covering Google, Anthropic, OpenAI, and OpenRouter models.
+
+modes:
+  - raw
+
+languages:
+  - cuda
+  - triton
+
+models:
+  - provider: gemini
+    model: gemini-2.5-pro
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: gemini
+    model: gemini-2.5-flash
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: anthropic
+    model: claude-sonnet-4-5
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: anthropic
+    model: claude-haiku-4-5
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: openai
+    model: gpt-5
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: openai
+    model: gpt-5-nano
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: openai
+    model: gpt-5-mini
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+  - provider: openrouter
+    model: z-ai/glm-4.6
+    raw_concurrency: 1
+    raw_gpu_concurrency: 1
+    raw_max_jobs: 1
+
+defaults:
+  num_runs: 1
+  profile_stages: true
+  raw:
+    cpu_concurrency: 1
+    gpu_concurrency: 1
+    max_jobs: 1
+
+verbose: true
+profile_stages: true
+fast_p_threshold: null
+raw_concurrency: 1
+raw_gpu_concurrency: 1
+raw_max_jobs: 1
+
+hardware:
+  gpu_architecture: Ampere
+  gpu_id: 0
+
+problems:
+  levels: [1, 2]
+  problem_ids: null
+  max_problems: 100
+
+agentic:
+  max_debug_attempts: 3
+  max_optimization_cycles: 2
+  reflector_model: gpt-4-turbo
+  optimizer_model: gpt-4-turbo
+
+artifacts:
+  json_dir: json
+  plots_dir: plots
+
+visualization:
+  enabled: false
```

## validate
- `source ~/.bashrc && export GEMINI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GROQ_API_KEY CEREBRAS_API_KEY XAI_API_KEY && uv run eval.py --config configs/all_providers.yaml ` *(fails fast when API keys aren’t exported; ready to re-run once credentials are in the shell env)*

# 2025-10-23 OpenRouter Provider

## rationale
Expose an adapter for OpenRouter’s OpenAI-compatible API so that OpenRouter-hosted models (e.g., z-ai/glm-4.6) can participate in KernelBench raw runs alongside first-party providers.

## patch
```diff
diff --git a/src/providers/openrouter_provider.py b/src/providers/openrouter_provider.py
new file mode 100644
+from __future__ import annotations
+
+import os
+from typing import Any, Dict, List
+
+from openai import OpenAI
+
+from .base import BaseProvider, ProviderConfig
+
+
+class OpenRouterProvider(BaseProvider):
+    """Provider adapter for OpenRouter's OpenAI-compatible API."""
+
+    default_base_url = "https://openrouter.ai/api/v1"
+
+    def __init__(self, config: ProviderConfig) -> None:
+        super().__init__(config)
+        base_url = config.base_url or self.default_base_url
+
+        self.client = OpenAI(
+            api_key=config.api_key,
+            base_url=base_url,
+            timeout=config.timeout,
+        )
+
+        extra = config.extra or {}
+        referer = extra.get("http_referer") or os.getenv("OR_SITE_URL")
+        title = extra.get("x_title") or os.getenv("OR_APP_NAME")
+
+        headers: Dict[str, str] = {}
+        if referer:
+            headers["HTTP-Referer"] = referer
+        if title:
+            headers["X-Title"] = title
+
+        self._default_headers = headers if headers else None
+
+    def generate(
+        self,
+        messages: List[Dict[str, Any]],
+        temperature: float = 0.0,
+        max_tokens: int = 1024,
+        **kwargs: Any,
+    ) -> str:
+        response = self.client.chat.completions.create(
+            model=self.config.model,
+            messages=messages,
+            temperature=temperature,
+            max_tokens=max_tokens,
+            extra_headers=self._default_headers,
+            **kwargs,
+        )
+        return response.choices[0].message.content
```
```diff
diff --git a/src/providers/__init__.py b/src/providers/__init__.py
@@
-    "ollama": OllamaProvider,
+    "ollama": OllamaProvider,
+    "openrouter": OpenRouterProvider,
@@
-    "ollama": ("OLLAMA_API_KEY",),
+    "ollama": ("OLLAMA_API_KEY",),
+    "openrouter": ("OPENROUTER_API_KEY", "OR_API_KEY"),
```

## validate
- `uv run python - <<'PY' ...` *(inspected ~/.bashrc for API key variable names without echoing secrets to ensure OpenRouter key needs to be exported before running the benchmark)*

# 2025-10-23 Enforce GPT-5 Default Temperature

## rationale
OpenAI GPT-5 chat completions reject temperature overrides, so preflight checks and benchmarks failed when KernelBench tried to force `temperature=0.0`. Omitting the field lets the API fall back to its required default (1.0) without changing behavior for earlier models.

## patch
```diff
diff --git a/src/providers/openai_provider.py b/src/providers/openai_provider.py
@@
-from typing import Any, List, Dict
+from typing import Any, Dict, List
@@
-        self._uses_max_completion_tokens = model_lower.startswith("gpt-5")
+        uses_gpt5_family = model_lower.startswith("gpt-5")
+        self._uses_max_completion_tokens = uses_gpt5_family
+        self._requires_default_temperature = uses_gpt5_family
+        if uses_gpt5_family:
+            # GPT-5 responses often exceed a single token, so widen the preflight budget.
+            if getattr(self, "preflight_max_output_tokens", None) is not None:
+                self.preflight_max_output_tokens = max(
+                    32, self.preflight_max_output_tokens
+                )
@@
-        request: Dict[str, Any] = {
-            "model": self.config.model,
-            "messages": messages,
-            "temperature": temperature,
-        }
-
-        request.update(kwargs)
+        temperature = kwargs.pop("temperature", temperature)
+
+        request: Dict[str, Any] = {
+            "model": self.config.model,
+            "messages": messages,
+        }
+
+        request.update(kwargs)
+
+        if not self._requires_default_temperature and temperature is not None:
+            request["temperature"] = temperature
```

## validate
- `uv run python -m compileall src/providers/openai_provider.py`
- `uv run eval.py --config configs/all_providers.yaml ` *(requires valid OpenAI GPT-5 credentials; run once keys are available to confirm preflight succeeds without temperature override failures).*

# 2025-10-23 Tighten CUDA/Triton Prompt Contracts

## rationale
Only Gemini models were returning compilable kernels. The updated prompts now force every provider to emit a single runnable Python block with explicit imports, kernel definitions, and a `ModelNew` implementation, reducing prose-only responses that previously failed compilation.

## patch
```diff
diff --git a/src/prompt_constructor.py b/src/prompt_constructor.py
@@
-PROBLEM_INSTRUCTION = """
-Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
-Always include these imports exactly once at the top of the code block:
-import torch
-import torch.nn as nn
-from torch.utils.cpp_extension import load_inline
-Never reference symbols that you have not imported.
-Define a complete `ModelNew` class that mirrors Model's constructor signature and uses the generated kernels.
-Do not emit placeholder text, TODOs, or comments that indicate missing implementation.
-"""
+PROBLEM_INSTRUCTION = """
+Optimize the architecture named Model with custom CUDA operators and emit a drop-in replacement called ModelNew. Follow this contract exactly:
+1. Reply with a single Markdown code block labeled `python` and no additional prose before or after it.
+2. Begin the block with these imports exactly once: `import torch`, `import torch.nn as nn`, `from torch.utils.cpp_extension import load_inline`.
+3. Define at least one CUDA kernel string plus a `functions` dictionary that calls `load_inline` to compile it, and expose Python wrapper functions that launch the kernels with correct grid and block dimensions.
+4. Implement a complete `ModelNew` class whose `__init__` signature matches `Model` and whose `forward` uses the wrapper functions. Preserve all tensor shapes returned by the original model.
+5. Do not include unit tests, benchmarking harnesses, placeholder comments, or explanatory text. Only runnable production code is allowed.
+6. Never reference symbols you did not define or import within the block.
+"""
@@
-PROBLEM_INSTRUCTION_TRITON = """
-Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. When you respond:
-- Provide a single Python code block.
-- Import `torch`, `torch.nn as nn`, `triton`, and `triton.language as tl` at the top.
-- Define any Triton kernels with `@triton.jit` and supply a Python wrapper that launches them with proper grids and stride calculations.
-- Keep the original constructor signature for ModelNew and ensure it produces outputs identical in shape to Model.
-- Do not include test code or explanatory prose outside the code block.
-- Avoid `triton.autotune` or any decorator that changes the kernel signature dynamically; return a single deterministic kernel implementation.
-- Never reference symbols you have not imported.
-- Ensure the emitted code is fully runnable without edits and contains a complete `ModelNew` definition that invokes the Triton kernel.
-"""
+PROBLEM_INSTRUCTION_TRITON = """
+Optimize the architecture named Model with custom Triton kernels and output a drop-in replacement called ModelNew. When you respond:
+1. Emit exactly one Markdown code block labeled `python` and nothing else.
+2. Start with these imports in order: `import torch`, `import torch.nn as nn`, `import triton`, `import triton.language as tl`.
+3. Implement each custom kernel with `@triton.jit`, provide launch-time wrappers that compute grid sizes and strides, and ensure the wrappers are invoked from ModelNew.
+4. Keep ModelNew's constructor signature identical to Model's and preserve all output tensor shapes.
+5. Avoid tests, benchmarking, or explanatory text; only include runnable library code.
+6. Do not use dynamic decorators such as `triton.autotune`; return a single deterministic kernel implementation per operation.
+7. Reference only symbols defined or imported inside the code block.
+"""
@@
-PROBLEM_INSTRUCTION_CLEANED = """
-Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
-"""
+PROBLEM_INSTRUCTION_CLEANED = """
+Optimize the architecture named Model with custom CUDA operators and emit a drop-in replacement called ModelNew. Follow this contract exactly:
+1. Reply with a single Markdown code block labeled `python` and no additional prose before or after it.
+2. Begin the block with these imports exactly once: `import torch`, `import torch.nn as nn`, `from torch.utils.cpp_extension import load_inline`.
+3. Define at least one CUDA kernel string plus a `functions` dictionary that calls `load_inline` to compile it, and expose Python wrapper functions that launch the kernels with correct grid and block dimensions.
+4. Implement a complete `ModelNew` class whose `__init__` signature matches `Model` and whose `forward` uses the wrapper functions. Preserve all tensor shapes returned by the original model.
+5. Do not include unit tests, benchmarking harnesses, placeholder comments, or explanatory text. Only runnable production code is allowed.
+6. Never reference symbols you did not define or import within the block.
+"""
```

## validate
- `uv run python -m compileall src/prompt_constructor.py`
- (Optional) `uv run eval.py --config configs/all_providers.yaml ` once provider API keys are configured, to observe improved compilation rates across non-Gemini models.

# 2025-10-23 Formatter Beta & Triton Guardrails

## rationale
Stricter CUDA/Triton prompt contracts still let some providers return prose; added an optional Groq-based formatter stage that normalizes completions into the required `ModelNew` code block. Batch configs and CLI now surface formatter settings, and the Triton prompt includes an explicit example. `extract_first_code` was hardened to accept uppercase language tags so we stop losing code blocks.

## patch
```diff
diff --git a/config.py b/config.py
@@
 class BenchmarkConfig:
     mode: EvaluationMode = "raw"
     language: KernelLanguage = "triton"
     provider: str = "openai"
     provider_base_url: str | None = None
+    formatter_provider: str | None = None
+    formatter_model: str | None = None
+    formatter_base_url: str | None = None

diff --git a/eval.py b/eval.py
@@
     parser.add_argument("--formatter-provider", type=str, dest="formatter_provider", help="Optional provider to post-process model outputs into the KernelBench contract.")
     parser.add_argument("--formatter-model", type=str, dest="formatter_model", help="Model identifier for the formatter provider.")
     parser.add_argument("--formatter-base-url", type=str, dest="formatter_base_url", help="Override base URL for the formatter provider (OpenAI-compatible APIs).")
     parser.add_argument("--groq-formatter-beta", action="store_true", help="Enable the Groq moonshotai/kimi-k2-instruct-0905 formatter beta to enforce structured output.")
@@
         if cli_args.groq_formatter_beta:
             overrides["formatter_provider"] = "groq"
             overrides["formatter_model"] = "moonshotai/kimi-k2-instruct-0905"
             overrides.setdefault("formatter_base_url", None)

diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
     formatter_defaults = defaults.get("formatter", {})
     formatter_cfg = model_entry.get("formatter", {})
     formatter_provider = formatter_cfg.get("provider", yaml_data.get("formatter_provider", formatter_defaults.get("provider")))
     formatter_model = formatter_cfg.get("model", yaml_data.get("formatter_model", formatter_defaults.get("model")))
     formatter_base_url = formatter_cfg.get("base_url", yaml_data.get("formatter_base_url", formatter_defaults.get("base_url")))
@@
     config = BenchmarkConfig(
         mode=mode,
         language=language,
         provider=provider,
         provider_base_url=base_url,
         formatter_provider=formatter_provider,
         formatter_model=formatter_model,
         formatter_base_url=formatter_base_url,
         generator_model=model_id,

diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-from prompt_constructor import (
-    prompt_generate_custom_cuda_from_prompt_template,
-    prompt_generate_custom_triton_from_template,
-)
+from prompt_constructor import (
+    prompt_generate_custom_cuda_from_prompt_template,
+    prompt_generate_custom_triton_from_template,
+    build_formatter_messages,
+)
@@
 def build_formatter_provider(config: "BenchmarkConfig"):
     if FAKE_LLM:
         return None
     provider_name = getattr(config, "formatter_provider", None)
     if not provider_name:
         return None
     model_id = getattr(config, "formatter_model", None)
     if not model_id:
         raise ValueError("formatter_model must be set when formatter_provider is configured.")
     api_key = resolve_provider_api_key(provider_name)
     base_url = getattr(config, "formatter_base_url", None) or os.getenv("KB3_LLM_FORMATTER_BASE_URL")
     provider_config = ProviderConfig(provider=provider_name, model=model_id, api_key=api_key, base_url=base_url)
     return create_provider(provider_config)
@@
     if not FAKE_LLM:
         _CPU_STATE["inference_fn"] = build_inference_callable(config)
         _CPU_STATE["formatter_fn"] = build_formatter_callable(config)
@@
-        completion = inference_fn(prompt)
-        generated = extract_first_code(completion, ["python", "cpp"]) or completion
+        completion = inference_fn(prompt)
+        raw_completion = completion
+        formatter_fn = _CPU_STATE.get("formatter_fn")
+        if formatter_fn:
+            try:
+                messages = build_formatter_messages(language=lang, reference_architecture=ref_arch_src, original_prompt=prompt, raw_completion=completion)
+                formatted_completion = formatter_fn(messages)
+            except Exception:  # noqa: BLE001
+                formatted_completion = None
+        candidate_source = formatted_completion or completion
+        generated = (
+            extract_first_code(candidate_source, ["python", "cpp"])
+            or extract_first_code(completion, ["python", "cpp"])
+            or candidate_source
+        )
@@
         "generated_code": generated,
-        "raw_completion": completion if not FAKE_LLM else None,
-        "formatted_completion": formatted_completion if not FAKE_LLM else None,
+        "raw_completion": raw_completion,
+        "formatted_completion": formatted_completion,
     }

diff --git a/src/prompt_constructor.py b/src/prompt_constructor.py
@@
 PROBLEM_INSTRUCTION_TRITON = """
 Optimize the architecture named Model with custom Triton kernels and output a drop-in replacement called ModelNew. When you respond:
 1. Emit exactly one Markdown code block labeled `python` and nothing else.
@@
 def prompt_generate_custom_triton_from_template(ref_arch_src: str) -> str:
     """Generate a Triton-focused prompt mirroring the CUDA template flow."""
 
     prompt = PROBLEM_STATEMENT_TRITON
     example_arch_path = os.path.join(REPO_TOP_PATH, "src/prompts/model_ex_add.py")
     example_triton_path = os.path.join(REPO_TOP_PATH, "src/prompts/model_new_ex_add_triton.py")
     if os.path.exists(example_arch_path) and os.path.exists(example_triton_path):
         example_arch = read_file(example_arch_path)
         example_triton = read_file(example_triton_path)
         prompt += "\nHere is an example that meets the contract:\n```\n"
         prompt += example_arch
         prompt += "\n```\nbecomes\n```\n"
         prompt += example_triton
         prompt += "\n```\n"
@@
 FORMATTER_SYSTEM_PROMPT = (
     "You are a GPU kernel formatting assistant. Always output a single runnable "
     "KernelBench solution as a python fenced code block that obeys the ModelNew contract."
 )
@@
 def build_formatter_messages(...):
     """Construct messages for a formatting LLM that enforces KernelBench contracts."""
     contract = PROBLEM_INSTRUCTION if language == "cuda" else PROBLEM_INSTRUCTION_TRITON
     completion_text = raw_completion or ""
     user_content = f"""Rewrite the raw completion so it satisfies the KernelBench contract.
@@
     return [
         {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
         {"role": "user", "content": user_content},
     ]

diff --git a/src/utils.py b/src/utils.py
@@
-        for code_type in code_language_types:
-            if code.startswith(code_type):
-                code = code[len(code_type) :].strip()
+        for code_type in code_language_types:
+            pattern = re.compile(rf"^{code_type}\s*", re.IGNORECASE)
+            if pattern.match(code):
+                code = pattern.sub("", code, count=1).lstrip("\n")
+                break
```

## validate
- `uv run python -m compileall config.py eval.py src/batch_runner.py src/raw/runner.py src/prompt_constructor.py src/utils.py`
- `uv run python eval.py --config configs/all_providers.yaml  --profile-stages --verbose --groq-formatter-beta`
- `uv run python - <<'PY'` (case-insensitive code fence check)```
import sys
sys.path.append('src')
from utils import extract_first_code
assert extract_first_code(\"\"\"```PYTHON\\nprint('ok')\\n```\"\"\", ['python']) == \"print('ok')\"
PY
```
 
# 2025-10-23 README Formatter Docs

## rationale
Documented the formatter beta workflow so users can enable the Groq cleanup pass via environment variables, CLI flags, or YAML.

## patch
```diff
diff --git a/README.md b/README.md
@@
 export GROQ_API_KEY="your-groq-key"  # required for smoke tests & Groq runs
 # export OPENAI_API_KEY="your-openai-key"
 # export ANTHROPIC_API_KEY="your-anthropic-key"
 # export GEMINI_API_KEY="your-gemini-key"
 ```

 Formatter overrides (beta) are configured via CLI/YAML—see [Formatter Beta (Optional)](#formatter-beta-optional).
@@
 ### Formatter Beta (Optional)
 
 - Enable the Groq-based formatter on the CLI with `--groq-formatter-beta`, or provide explicit overrides:
   - `--formatter-provider groq`
   - `--formatter-model moonshotai/kimi-k2-instruct-0905`
   - `--formatter-base-url https://api.groq.com/openai/v1`
 - YAML configs support a `formatter` block on each model:
 
   ```yaml
   defaults:
     formatter:
       provider: groq
       model: moonshotai/kimi-k2-instruct-0905
   models:
     - provider: openai
       model: gpt-5
       formatter:
         provider: groq
         model: moonshotai/kimi-k2-instruct-0905
   ```
 
   The formatter performs a second LLM pass to enforce a single fenced `python` block containing a complete `ModelNew` implementation, complementing the stricter CUDA/Triton prompts in `src/prompt_constructor.py`.
```

## validate
- `uv run python eval.py --config configs/test_visualization.yaml  --profile-stages --verbose --groq-formatter-beta`

# 2025-10-24 Lift Raw Token Budget

## rationale
Raw CUDA/Triton prompts routinely exceed the 1k-token default, leading to truncated Gemini/OpenAI completions and compilation failures. Explicitly budgeting 4k tokens (with YAML overrides) for both generators and optional formatters removes that bottleneck before the next evaluation sweep finishes.

## patch
```diff
diff --git a/config.py b/config.py
@@
-    raw_max_jobs: int = 8
+    raw_max_jobs: int = 8
+    generation_max_tokens: int = 4096
+    formatter_max_tokens: int | None = None
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
     raw_max_jobs = model_entry.get(
         "raw_max_jobs",
         yaml_data.get("raw_max_jobs", raw_defaults.get("max_jobs", 8)),
     )
 
+    generation_max_tokens = model_entry.get("generation_max_tokens")
+    if generation_max_tokens is None:
+        generation_max_tokens = yaml_data.get(
+            "generation_max_tokens",
+            defaults.get("generation_max_tokens"),
+        )
+    if generation_max_tokens is not None:
+        generation_max_tokens = int(generation_max_tokens)
+
+    formatter_max_tokens = model_entry.get("formatter_max_tokens")
+    if formatter_max_tokens is None:
+        formatter_max_tokens = yaml_data.get(
+            "formatter_max_tokens",
+            defaults.get("formatter_max_tokens"),
+        )
+    if formatter_max_tokens is not None:
+        formatter_max_tokens = int(formatter_max_tokens)
+
@@
-    config = BenchmarkConfig(
+    config_kwargs: Dict[str, Any] = dict(
         mode=mode,
         language=language,
         provider=provider,
         provider_base_url=base_url,
         formatter_provider=formatter_provider,
@@
-        fast_p_threshold=yaml_data.get("fast_p_threshold", defaults.get("fast_p_threshold")),
-        raw_concurrency=raw_concurrency,
-        raw_gpu_concurrency=raw_gpu_concurrency,
-        raw_max_jobs=raw_max_jobs,
-    )
-
-    return config
+        fast_p_threshold=yaml_data.get("fast_p_threshold", defaults.get("fast_p_threshold")),
+        raw_concurrency=raw_concurrency,
+        raw_gpu_concurrency=raw_gpu_concurrency,
+        raw_max_jobs=raw_max_jobs,
+    )
+
+    if generation_max_tokens is not None:
+        config_kwargs["generation_max_tokens"] = generation_max_tokens
+    if formatter_max_tokens is not None:
+        config_kwargs["formatter_max_tokens"] = formatter_max_tokens
+
+    config = BenchmarkConfig(**config_kwargs)
+
+    return config
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-    provider = build_provider(config)
+    provider = build_provider(config)
+    generation_limit = getattr(config, "generation_max_tokens", 4096)
+    if generation_limit is not None and generation_limit <= 0:
+        generation_limit = None
@@
-        return provider.generate(messages)
+        return provider.generate(messages, max_tokens=generation_limit)
@@
-    formatter_provider = build_formatter_provider(config)
-    if formatter_provider is None:
-        return None
+    formatter_provider = build_formatter_provider(config)
+    if formatter_provider is None:
+        return None
+    formatter_limit = getattr(config, "formatter_max_tokens", None)
+    if formatter_limit is not None and formatter_limit <= 0:
+        formatter_limit = None
@@
-        return formatter_provider.generate(messages)
+        return formatter_provider.generate(messages, max_tokens=formatter_limit)
 ```

## validate
- `uv run python eval.py --config configs/all_providers.yaml ` (after exporting required API keys)

# 2025-10-24 Widen Provider Handshake Tokens

## rationale
OpenAI GPT-5 uses the Responses API and spends tokens on hidden reasoning before emitting final text. The 8-token sanity check in `verify_model_responds_hello` starved it, yielding an empty reply and aborting batch preflight. Matching the provider’s preflight token allowance fixes the handshake without relaxing the exact-response assertion.

## patch
```diff
diff --git a/src/providers/__init__.py b/src/providers/__init__.py
@@
-    response = instance.generate(messages, max_tokens=8)
+    handshake_tokens = 16
+    preflight_cap = getattr(instance, "preflight_max_output_tokens", None)
+    if isinstance(preflight_cap, int) and preflight_cap > 0:
+        handshake_tokens = max(handshake_tokens, preflight_cap)
+    response = instance.generate(messages, max_tokens=handshake_tokens)
```

## validate
- `uv run python eval.py --config configs/all_providers.yaml `

# 2025-10-24 Revert Context Telemetry Monitor

## rationale
Rollback the live context-character counter because it proved unreliable during long sweeps and cluttered stdout. Restored providers/raw/batch runners to their prior behavior and removed the telemetry module.

## patch
```diff
diff --git a/src/telemetry/context_tracker.py b/src/telemetry/context_tracker.py
deleted file mode 100644
diff --git a/src/telemetry/__init__.py b/src/telemetry/__init__.py
deleted file mode 100644
diff --git a/src/providers/__init__.py b/src/providers/__init__.py
@@
-from .openrouter_provider import OpenRouterProvider
-from telemetry.context_tracker import context_tracker
+from .openrouter_provider import OpenRouterProvider
@@
-class ContextTrackingProvider(BaseProvider):
-    """Proxy that records prompt/response character usage for telemetry."""
-
-    def __init__(self, delegate: BaseProvider) -> None:
-        super().__init__(delegate.config)
-        self._delegate = delegate
-
-    def __getattr__(self, name: str):
-        return getattr(self._delegate, name)
-
-    def generate(self, *args, **kwargs):
-        ...
-        return response
-
-
 def create_provider(config: ProviderConfig) -> BaseProvider:
@@
-    instance = provider_cls(config)
-    instance.preflight()
-    return ContextTrackingProvider(instance)
+    instance = provider_cls(config)
+    instance.preflight()
+    return instance
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-from providers import ProviderConfig, create_provider, resolve_provider_api_key
-from telemetry.context_tracker import context_tracker, configure_shared_from_handles
+from providers import ProviderConfig, create_provider, resolve_provider_api_key
@@
-def _cpu_initializer(
-    config: "BenchmarkConfig",
-    generation_level: int,
-    job_queue,
-    result_store,
-    build_root: Path,
-    tracker_handles,
-):
-    """Set up CPU worker state for prompt generation."""
+def _cpu_initializer(config: "BenchmarkConfig", generation_level: int, job_queue, result_store, build_root: Path):
+    """Set up CPU worker state for prompt generation."""
@@
-    manager = ctx.Manager()
-    job_queue = ctx.Queue(maxsize=worker_count * 2 if use_parallel else 1)
-    shared_results: Dict[int, Dict[str, Any]] = manager.dict()
-    tracker_handles = context_tracker.create_shared_handles(ctx)
+    manager = ctx.Manager()
+    job_queue = ctx.Queue(maxsize=worker_count * 2 if use_parallel else 1)
+    shared_results: Dict[int, Dict[str, Any]] = manager.dict()
@@
-            initializer=_cpu_initializer,
-            initargs=(config, gen_cfg.level, job_queue, shared_results, build_root, tracker_handles),
+            initializer=_cpu_initializer,
+            initargs=(config, gen_cfg.level, job_queue, shared_results, build_root),
@@
-        _cpu_initializer(
-            config,
-            gen_cfg.level,
-            job_queue,
-            shared_results,
-            build_root,
-            tracker_handles,
-        )
+        _cpu_initializer(config, gen_cfg.level, job_queue, shared_results, build_root)
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-from providers import verify_model_responds_hello
-from telemetry.context_tracker import context_tracker
+from providers import verify_model_responds_hello
@@
-    for index, (mode, language, model_entry) in enumerate(run_plan, start=1):
-        provider = model_entry.get("provider", yaml_data.get("provider", "unknown"))
-        model_id = model_entry.get("model", yaml_data.get("generator_model", "unknown"))
-
-        label = f"[{index}/{total_runs}] {mode.upper()} | {language.upper()} :: {provider}/{model_id}"
-
-        print()
-        print("-" * 70)
+    for index, (mode, language, model_entry) in enumerate(run_plan, start=1):
+        provider = model_entry.get("provider", yaml_data.get("provider", "unknown"))
+        model_id = model_entry.get("model", yaml_data.get("generator_model", "unknown"))
+
+        print(f"[{index}/{total_runs}] {mode.upper()} | {language.upper()} :: {provider}/{model_id}")
+        print("-" * 70)
@@
-        if cached_payload and cached_payload.get("config_fingerprint") == fingerprint:
-            context_tracker.pause()
+        if cached_payload and cached_payload.get("config_fingerprint") == fingerprint:
@@
-        try:
+        try:
-            results.append({
-                **payload,
-                "status": "completed",
-                "metrics_path": str(cache_path),
-            })
-            context_tracker.pause()
+            results.append({
+                **payload,
+                "status": "completed",
+                "metrics_path": str(cache_path),
+            })
@@
-        except Exception as exc:  # noqa: BLE001
-            context_tracker.pause()
+        except Exception as exc:  # noqa: BLE001
```

## validate
- not run (telemetry rollback only)

# 2025-10-24 Preflight Progress Bar

## rationale
Batch runs looked stalled during provider verification. Adding a live progress bar over the unique provider/model pairs makes clear that preflight is advancing even when many credentials must be checked.

## patch
```diff
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-    print("[Preflight] Verifying provider/model combinations...")
-    for mode, language, model_entry in run_plan:
-        config = yaml_to_benchmark_config(yaml_data, model_entry, mode, language, cli_overrides)
-        key = (
-            config.provider.lower(),
-            config.generator_model,
-            config.provider_base_url or "",
-        )
-        if key in seen:
-            continue
-        seen.add(key)
-        try:
-            verify_model_responds_hello(
-                config.provider,
-                config.generator_model,
-                config.provider_base_url,
-            )
-        except Exception as exc:  # noqa: BLE001
-            errors.append(f"{config.provider}/{config.generator_model}: {exc}")
-
-    if errors:
+    print("[Preflight] Verifying provider/model combinations...")
+    for index, (provider_name, model_name, base_url) in enumerate(unique_targets, start=1):
+        try:
+            verify_model_responds_hello(
+                provider_name,
+                model_name,
+                base_url,
+            )
+        except Exception as exc:  # noqa: BLE001
+            errors.append(f"{provider_name}/{model_name}: {exc}")
+        progress_bar = _render_progress_bar(index, total_targets)
+        sys.stdout.write(
+            f"\r[Preflight] Verifying provider/model combinations... {progress_bar} {index}/{total_targets}"
+        )
+        sys.stdout.flush()
+
+    if total_targets:
+        sys.stdout.write("\n")
+        sys.stdout.flush()
+
+    if errors:
         print("[Preflight] Provider/model validation failed:")
         for err in errors:
             print(f"  - {err}")
         sys.exit(1)
```

## validate
- `uv run python eval.py --config configs/all_providers.yaml `

# 2025-10-23 Fix solve_and_add_scaled_vector filename

## why
Rename the TritonBench golden result for `solve_and_add_scaled_vector` to avoid the `..json` basename that breaks croc transfers, and update the associated performance harness to emit the corrected path.

## patch
```diff
--- a/data/TritonBench/performance_metrics/perf_T/golden_metrics/solve_and_add_scaled_vector_perf.py
+++ b/data/TritonBench/performance_metrics/perf_T/golden_metrics/solve_and_add_scaled_vector_perf.py
@@
-        super().__init__('solve_and_add_scaled_vector.', dtype=dtype, is_backward=is_backward, **kwargs)
+        super().__init__('solve_and_add_scaled_vector', dtype=dtype, is_backward=is_backward, **kwargs)
```

```diff
--- a/data/TritonBench/performance_metrics/perf_T/golden_results/solve_and_add_scaled_vector..json
+++ b/data/TritonBench/performance_metrics/perf_T/golden_results/solve_and_add_scaled_vector.json
```

## validate
- No runtime validation required; this is a rename plus constructor literal fix.

# 2025-10-24 README benchmark overview graphic

## why
Surface the latest raw CUDA and Triton compile/correctness results directly in the README so visitors immediately see real benchmark outcomes.

## patch
```diff
--- a/README.md
+++ b/README.md
@@
-![Raw and Triton compilation/correctness hit rates for recent runs](docs/media/model_outcomes.svg)
-
-Unified benchmark for evaluating LLM-generated GPU kernels (CUDA & Triton) in both raw and agentic workflows.
+![Raw and agentic compilation/correctness hit rates for recent runs](docs/media/model_outcomes.svg)
+
+`—` indicates the agentic pipeline has not yet completed a fresh sweep for that provider/model pair.
+
+Unified benchmark for evaluating LLM-generated GPU kernels (CUDA & Triton) in both raw and agentic workflows.
```

```diff
--- /dev/null
+++ b/docs/media/model_outcomes.svg
@@
+<svg xmlns='http://www.w3.org/2000/svg' width='920' height='1304' viewBox='0 0 920 1304'>
+<style> text { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 15px; } .header { font-weight: 600; } .mode { font-weight: 600; } .model { font-weight: 500; } </style>
+<rect x='0' y='0' width='920' height='48' fill='#0B1F33' rx='8' ry='8' />
+<text x='110.0' y='30.0' fill='#FFFFFF' text-anchor='middle' class='header'>Mode</text>
+<text x='390.0' y='30.0' fill='#FFFFFF' text-anchor='middle' class='header'>Model</text>
+<text x='620.0' y='30.0' fill='#FFFFFF' text-anchor='middle' class='header'>Compiled</text>
+<text x='740.0' y='30.0' fill='#FFFFFF' text-anchor='middle' class='header'>Numerically Correct</text>
+<text x='860.0' y='30.0' fill='#FFFFFF' text-anchor='middle' class='header'>Fast@1%</text>
+<rect x='0' y='48' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='48' x2='920' y2='48' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='71.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='71.0' fill='#213547' class='model'>Gemini 2.5 Pro</text>
+<text x='620.0' y='71.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='71.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='860.0' y='71.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='84' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='84' x2='920' y2='84' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='107.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='107.0' fill='#213547' class='model'>Gemini 2.5 Flash</text>
+<text x='620.0' y='107.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='107.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='860.0' y='107.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='120' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='120' x2='920' y2='120' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='143.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='143.0' fill='#213547' class='model'>Claude Sonnet 4.5</text>
+<text x='620.0' y='143.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='143.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='860.0' y='143.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='156' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='156' x2='920' y2='156' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='179.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='179.0' fill='#213547' class='model'>Claude Haiku 4.5</text>
+<text x='620.0' y='179.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='179.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='860.0' y='179.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='192' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='192' x2='920' y2='192' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='215.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='215.0' fill='#213547' class='model'>GPT-5</text>
+<text x='620.0' y='215.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='215.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='215.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='228' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='228' x2='920' y2='228' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='251.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='251.0' fill='#213547' class='model'>GPT-5 Nano</text>
+<text x='620.0' y='251.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='251.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='251.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='264' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='264' x2='920' y2='264' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='287.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='287.0' fill='#213547' class='model'>GPT-5 Mini</text>
+<text x='620.0' y='287.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='287.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='287.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='300' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='300' x2='920' y2='300' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='323.0' fill='#213547' text-anchor='middle'>Raw CUDA</text>
+<text x='232' y='323.0' fill='#213547' class='model'>OpenRouter GLM-4.6</text>
+<text x='620.0' y='323.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='323.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='323.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<line x1='0' y1='336' x2='920' y2='336' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='360' fill='#213547' text-anchor='middle' class='mode'>Raw Triton</text>
+<rect x='0' y='372' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='372' x2='920' y2='372' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='395.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='395.0' fill='#213547' class='model'>Gemini 2.5 Pro</text>
+<text x='620.0' y='395.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='395.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='395.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='408' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='408' x2='920' y2='408' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='431.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='431.0' fill='#213547' class='model'>Gemini 2.5 Flash</text>
+<text x='620.0' y='431.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='431.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='431.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='444' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='444' x2='920' y2='444' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='467.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='467.0' fill='#213547' class='model'>Claude Sonnet 4.5</text>
+<text x='620.0' y='467.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='467.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='467.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='480' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='480' x2='920' y2='480' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='503.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='503.0' fill='#213547' class='model'>Claude Haiku 4.5</text>
+<text x='620.0' y='503.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='503.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='503.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='516' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='516' x2='920' y2='516' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='539.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='539.0' fill='#213547' class='model'>GPT-5</text>
+<text x='620.0' y='539.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='539.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='539.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='552' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='552' x2='920' y2='552' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='575.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='575.0' fill='#213547' class='model'>GPT-5 Nano</text>
+<text x='620.0' y='575.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='575.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='575.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='588' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='588' x2='920' y2='588' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='611.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='611.0' fill='#213547' class='model'>GPT-5 Mini</text>
+<text x='620.0' y='611.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='740.0' y='611.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='611.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<rect x='0' y='624' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='624' x2='920' y2='624' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='647.0' fill='#213547' text-anchor='middle'>Raw Triton</text>
+<text x='232' y='647.0' fill='#213547' class='model'>OpenRouter GLM-4.6</text>
+<text x='620.0' y='647.0' fill='#2E7D32' text-anchor='middle'>100%</text>
+<text x='740.0' y='647.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<text x='860.0' y='647.0' fill='#B71C1C' text-anchor='middle'>0%</text>
+<line x1='0' y1='660' x2='920' y2='660' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='684' fill='#213547' text-anchor='middle' class='mode'>Agentic CUDA</text>
+<rect x='0' y='696' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='696' x2='920' y2='696' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='719.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='719.0' fill='#213547' class='model'>Gemini 2.5 Pro</text>
+<text x='620.0' y='719.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='719.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='719.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='732' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='732' x2='920' y2='732' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='755.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='755.0' fill='#213547' class='model'>Gemini 2.5 Flash</text>
+<text x='620.0' y='755.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='755.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='755.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='768' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='768' x2='920' y2='768' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='791.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='791.0' fill='#213547' class='model'>Claude Sonnet 4.5</text>
+<text x='620.0' y='791.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='791.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='791.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='804' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='804' x2='920' y2='804' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='827.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='827.0' fill='#213547' class='model'>Claude Haiku 4.5</text>
+<text x='620.0' y='827.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='827.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='827.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='840' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='840' x2='920' y2='840' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='863.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='863.0' fill='#213547' class='model'>GPT-5</text>
+<text x='620.0' y='863.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='863.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='863.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='876' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='876' x2='920' y2='876' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='899.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='899.0' fill='#213547' class='model'>GPT-5 Nano</text>
+<text x='620.0' y='899.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='899.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='899.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='912' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='912' x2='920' y2='912' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='935.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='935.0' fill='#213547' class='model'>GPT-5 Mini</text>
+<text x='620.0' y='935.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='935.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='935.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='948' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='948' x2='920' y2='948' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='971.0' fill='#213547' text-anchor='middle'>Agentic CUDA</text>
+<text x='232' y='971.0' fill='#213547' class='model'>OpenRouter GLM-4.6</text>
+<text x='620.0' y='971.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='971.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='971.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<line x1='0' y1='984' x2='920' y2='984' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1008' fill='#213547' text-anchor='middle' class='mode'>Agentic Triton</text>
+<rect x='0' y='1020' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='1020' x2='920' y2='1020' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1043.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1043.0' fill='#213547' class='model'>Gemini 2.5 Pro</text>
+<text x='620.0' y='1043.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1043.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1043.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1056' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='1056' x2='920' y2='1056' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1079.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1079.0' fill='#213547' class='model'>Gemini 2.5 Flash</text>
+<text x='620.0' y='1079.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1079.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1079.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1092' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='1092' x2='920' y2='1092' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1115.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1115.0' fill='#213547' class='model'>Claude Sonnet 4.5</text>
+<text x='620.0' y='1115.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1115.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1115.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1128' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='1128' x2='920' y2='1128' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1151.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1151.0' fill='#213547' class='model'>Claude Haiku 4.5</text>
+<text x='620.0' y='1151.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1151.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1151.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1164' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='1164' x2='920' y2='1164' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1187.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1187.0' fill='#213547' class='model'>GPT-5</text>
+<text x='620.0' y='1187.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1187.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1187.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1200' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='1200' x2='920' y2='1200' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1223.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1223.0' fill='#213547' class='model'>GPT-5 Nano</text>
+<text x='620.0' y='1223.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1223.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1223.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1236' width='920' height='36' fill='#F4F7FB'/>
+<line x1='0' y1='1236' x2='920' y2='1236' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1259.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1259.0' fill='#213547' class='model'>GPT-5 Mini</text>
+<text x='620.0' y='1259.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1259.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1259.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<rect x='0' y='1272' width='920' height='36' fill='#FFFFFF'/>
+<line x1='0' y1='1272' x2='920' y2='1272' stroke='#D0D7E3' stroke-width='1'/>
+<text x='110.0' y='1295.0' fill='#213547' text-anchor='middle'>Agentic Triton</text>
+<text x='232' y='1295.0' fill='#213547' class='model'>OpenRouter GLM-4.6</text>
+<text x='620.0' y='1295.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='740.0' y='1295.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<text x='860.0' y='1295.0' fill='#5C6F82' text-anchor='middle'>—</text>
+<line x1='0' y1='1308' x2='920' y2='1308' stroke='#D0D7E3' stroke-width='1'/>
+</svg>
```
## validate
- Visual check: open `docs/media/model_outcomes.svg` in a browser and confirm the table renders with raw (percent) and agentic (dash) sections.

# 2025-10-23 Expand All-Providers Config with XAI

## rationale
Ensure the shared batch config benchmarks X.AI Grok models alongside other providers.

## patch
```diff
diff --git a/configs/all_providers.yaml b/configs/all_providers.yaml
@@
   - provider: openrouter
     model: z-ai/glm-4.6
     raw_concurrency: 1
     raw_gpu_concurrency: 1
     raw_max_jobs: 1
   - provider: xai
     model: grok-code-fast-1
     raw_concurrency: 1
     raw_gpu_concurrency: 1
     raw_max_jobs: 1
   - provider: xai
     model: grok-4-fast-reasoning
     raw_concurrency: 1
     raw_gpu_concurrency: 1
     raw_max_jobs: 1
   - provider: xai
     model: grok-4-fast-non-reasoning
     raw_concurrency: 1
     raw_gpu_concurrency: 1
     raw_max_jobs: 1
   - provider: xai
     model: grok-4-0709
     raw_concurrency: 1
     raw_gpu_concurrency: 1
     raw_max_jobs: 1
```

## validate
- `uv run python eval.py --config configs/all_providers.yaml  --profile-stages --verbose --groq-formatter-beta`

# 2025-10-23 Provider Hello Preflight & Remove KB3 Fallbacks

## rationale
Ensure every provider/model pair is callable before starting a run and drop legacy `KB3_*` environment shortcuts so evaluations always hit live APIs.

## patch
```diff
diff --git a/src/providers/__init__.py b/src/providers/__init__.py
@@
-    scoped_key = f"KB3_LLM_API_KEY_{provider_slug.upper()}"
-    candidates = list(_PROVIDER_API_KEY_ALIASES.get(provider_slug, ()))
-    candidates.extend([scoped_key, "KB3_LLM_API_KEY"])
+    candidates = list(_PROVIDER_API_KEY_ALIASES.get(provider_slug, ()))
@@
 def create_provider(config: ProviderConfig) -> BaseProvider:
     provider_cls = PROVIDER_REGISTRY.get(config.provider.lower())
     if provider_cls is None:
         raise ValueError(f"Unsupported provider: {config.provider}")
     instance = provider_cls(config)
     instance.preflight()
     return instance
+
+def verify_model_responds_hello(provider: str, model: str, base_url: str | None = None) -> None:
+    provider_config = ProviderConfig(
+        provider=provider,
+        model=model,
+        api_key=resolve_provider_api_key(provider),
+        base_url=base_url,
+    )
+    instance = create_provider(provider_config)
+    messages = [
+        {"role": "system", "content": "You are a helpful assistant."},
+        {"role": "user", "content": "Reply with exactly the word Hello."},
+    ]
+    response = instance.generate(messages, max_tokens=8)
+    if response is None or response.strip() not in {"Hello.", "Hello"}:
+        raise RuntimeError(
+            f"Unexpected response from provider '{provider}' model '{model}': {response!r}"
+        )

diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-    print(f"Running {total_runs} target(s) from {yaml_path}")
-    print(f"{'=' * 70}\n")
+    print(f"Running {total_runs} target(s) from {yaml_path}")
+    print(f"{'=' * 70}\n")
+
+    _verify_provider_models(run_plan, yaml_data, cli_overrides)

diff --git a/eval.py b/eval.py
@@
-from config import BenchmarkConfig
+from config import BenchmarkConfig
+from providers import verify_model_responds_hello
@@
     if cli_args.profile_stages:
         config.profile_stages = True  # type: ignore[assignment]
 
+    try:
+        verify_model_responds_hello(
+            config.provider,
+            config.generator_model,
+            config.provider_base_url,
+        )
+    except Exception as exc:  # noqa: BLE001
+        print(f"[Preflight] Provider/model validation failed: {exc}")
+        sys.exit(1)
+
 diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-FAKE_LLM = os.getenv("KB3_FAKE_LLM")
@@
-    provider_name, model_id = config.provider, config.generator_model
-    api_key = resolve_provider_api_key(provider_name)
-    base_url = config.provider_base_url or os.getenv("KB3_LLM_BASE_URL")
+    provider_name, model_id = config.provider, config.generator_model
+    api_key = resolve_provider_api_key(provider_name)
+    base_url = config.provider_base_url
@@
-def build_formatter_provider(config: "BenchmarkConfig"):
-    if FAKE_LLM:
-        return None
+def build_formatter_provider(config: "BenchmarkConfig"):
@@
-    base_url = getattr(config, "formatter_base_url", None) or os.getenv(
-        "KB3_LLM_FORMATTER_BASE_URL"
-    )
+    base_url = getattr(config, "formatter_base_url", None)
@@
-    if FAKE_LLM:
-        generated = ref_arch_src
-    else:
-        lang = (config.language or "cuda").lower()
-        if lang == "triton":
-            prompt = prompt_generate_custom_triton_from_template(ref_arch_src)
-        else:
-            prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
-        completion = inference_fn(prompt)
-        raw_completion = completion
-        formatter_fn = _CPU_STATE.get("formatter_fn")
-        if formatter_fn:
-            try:
-                messages = build_formatter_messages(
-                    language=lang,
-                    reference_architecture=ref_arch_src,
-                    original_prompt=prompt,
-                    raw_completion=completion,
-                )
-                formatted_completion = formatter_fn(messages)
-            except Exception:  # noqa: BLE001
-                formatted_completion = None
-        candidate_source = formatted_completion or completion
-        generated = (
-            extract_first_code(candidate_source, ["python", "cpp"])
-            or extract_first_code(completion, ["python", "cpp"])
-            or candidate_source
-        )
+    lang = (config.language or "cuda").lower()
+    if lang == "triton":
+        prompt = prompt_generate_custom_triton_from_template(ref_arch_src)
+    else:
+        prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
+    completion = inference_fn(prompt)
+    raw_completion = completion
+    formatter_fn = _CPU_STATE.get("formatter_fn")
+    if formatter_fn:
+        try:
+            messages = build_formatter_messages(
+                language=lang,
+                reference_architecture=ref_arch_src,
+                original_prompt=prompt,
+                raw_completion=completion,
+            )
+            formatted_completion = formatter_fn(messages)
+        except Exception:  # noqa: BLE001
+            formatted_completion = None
+    candidate_source = formatted_completion or completion
+    generated = (
+        extract_first_code(candidate_source, ["python", "cpp"])
+        or extract_first_code(completion, ["python", "cpp"])
+        or candidate_source
+    )
@@
-    if config.hardware.gpu_architecture and not FAKE_LLM:
-        set_gpu_arch([config.hardware.gpu_architecture])
+    if config.hardware.gpu_architecture:
+        set_gpu_arch([config.hardware.gpu_architecture])
@@
-    workers_override = os.getenv("KB3_RAW_WORKERS")
-    worker_count = config.raw_concurrency or 1
-    if workers_override:
-        try:
-            worker_count = int(workers_override)
-        except ValueError:
-            if config.verbose:
-                print(f"[Raw] Ignoring invalid KB3_RAW_WORKERS value '{workers_override}'")
-
-    worker_count = max(1, worker_count)
+    worker_count = max(1, config.raw_concurrency or 1)
@@
-    gpu_workers_override = os.getenv("KB3_RAW_GPU_WORKERS")
-    gpu_worker_count = config.raw_gpu_concurrency or 1
-    if gpu_workers_override:
-        try:
-            gpu_worker_count = int(gpu_workers_override)
-        except ValueError:
-            if config.verbose:
-                print(f"[Raw] Ignoring invalid KB3_RAW_GPU_WORKERS value '{gpu_workers_override}'")
-
-    gpu_worker_count = max(1, gpu_worker_count)
+    gpu_worker_count = max(1, config.raw_gpu_concurrency or 1)

diff --git a/src/agentic/runner/main.py b/src/agentic/runner/main.py
@@
-    default_provider = os.getenv("KB3_LLM_PROVIDER", "openai")
+    default_provider = "openai"
@@
-    api_key = resolve_provider_api_key(provider_name)
-    base_url = config.provider_base_url or os.getenv("KB3_LLM_BASE_URL")
+    api_key = resolve_provider_api_key(provider_name)
+    base_url = config.provider_base_url
@@
-    temperature = float(os.getenv("KB3_AGENTIC_TEMPERATURE", "0.0"))
+    temperature = 0.0
```

## validate
- `uv run python eval.py --config configs/test_visualization.yaml  --profile-stages --verbose --groq-formatter-beta`
# 2025-10-25 Unify logging + artifacts

## rationale
Provide identical plain-text logging and manifest-driven artifacts for raw and agentic runs, enforce shared concurrency defaults, and document the new workflow so debugging stays consistent across providers.

## patch
```diff
diff --git a/src/agentic/runner/main.py b/src/agentic/runner/main.py
@@
-    output_path, latest_output_path, run_dir, run_id, timestamp_obj = derive_output_paths(
-        config,
-        provider_wrapper.config.provider,
-        provider_wrapper.config.model,
-        timestamp=run_timestamp,
-    )
+    run_dir, run_id, timestamp_obj = derive_output_paths(
+        config,
+        provider_wrapper.config.provider,
+        provider_wrapper.config.model,
+        timestamp=run_timestamp,
+    )
@@
-    print("[Agentic] Writing results to", output_path)
-    agent.run(
-        output_path=str(output_path),
+    agent.run(
+        output_path=None,
@@
-    print("[Agentic] Completed agentic benchmark run")
-
-    summary_path, summary_stats = build_agentic_summary(
-        latest_output_path.parent,
-        output_path,
-        dataset.problem_states,
-    )
-
-    if summary_stats.get("records"):
-        print(
-            f"[Agentic] Aggregated {summary_stats['records']} problem(s) from iteration {summary_stats['latest_iteration']}."
-        )
-
-    if summary_path.exists() and output_path != summary_path:
-        try:
-            shutil.copy2(summary_path, output_path)
-        except Exception as exc:
-            print(
-                f"[Agentic] Warning: failed to persist timestamped summary at {output_path}: {exc}"
-            )
-
-    resolved_results = summary_path
-
-    return {
-        "results_path": str(resolved_results),
-        "timestamped_path": str(output_path),
-        "run_dir": str(latest_output_path.parent),
-    }
+    print("[Agentic] Completed agentic benchmark run")
+
+    artifacts = emit_agentic_artifacts(
+        config,
+        run_dir,
+        run_id,
+        timestamp_obj,
+        dataset.problem_states,
+        agent.memories,
+        iteration_num,
+        temperature,
+        ancestor_num,
+    )
+
+    return artifacts
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-    raw_concurrency = model_entry.get(
-        "raw_concurrency",
-        yaml_data.get("raw_concurrency", raw_defaults.get("cpu_concurrency", 8)),
-    )
-    raw_gpu_concurrency = model_entry.get(
-        "raw_gpu_concurrency",
-        yaml_data.get("raw_gpu_concurrency", raw_defaults.get("gpu_concurrency", 1)),
-    )
-    raw_max_jobs = model_entry.get(
-        "raw_max_jobs",
-        yaml_data.get("raw_max_jobs", raw_defaults.get("max_jobs", 8)),
-    )
+    raw_settings = yaml_data.get("raw", {}) or {}
+
+    raw_concurrency = cli_overrides.get("raw_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = yaml_data.get("raw_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = raw_settings.get("cpu_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = raw_defaults.get("cpu_concurrency", 1)
+    raw_concurrency = int(raw_concurrency)
+
+    raw_gpu_concurrency = cli_overrides.get("raw_gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = yaml_data.get("raw_gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = raw_settings.get("gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = raw_defaults.get("gpu_concurrency", 1)
+    raw_gpu_concurrency = int(raw_gpu_concurrency)
+
+    raw_max_jobs = cli_overrides.get("raw_max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = yaml_data.get("raw_max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = raw_settings.get("max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = raw_defaults.get("max_jobs", 8)
+    raw_max_jobs = int(raw_max_jobs)
diff --git a/README.md b/README.md
@@
-├── outputs/                 # Agentic run outputs (raw JSONL traces)
-├── runs/                    # Raw run outputs (per-problem JSONL)
+├── outputs/                 # Legacy agentic outputs (JSONL, kept for back-compat)
+├── runs/                    # Raw & agentic run artifacts (manifests + per-problem logs)
```

## validate
- `uv sync`
- `uv run python eval.py --config configs/quick_test.yaml `
- Inspect `runs/<timestamp>_*` to confirm manifest, summaries, and history files exist for both raw and agentic modes.

# 2025-10-25 Increase raw concurrency defaults

## rationale
Run every target as fast as possible by raising the shared CPU/GPU worker pools so a single-problem sweep finishes quickly when `num_runs=1`.

## patch
```diff
diff --git a/configs/all_providers.yaml b/configs/all_providers.yaml
@@
-  raw:
-    cpu_concurrency: 4
-    gpu_concurrency: 1
-    max_jobs: 8
+  raw:
+    cpu_concurrency: 16
+    gpu_concurrency: 2
+    max_jobs: 16
@@
-raw_concurrency: 4
-raw_gpu_concurrency: 1
-raw_max_jobs: 8
+raw_concurrency: 16
+raw_gpu_concurrency: 2
+raw_max_jobs: 16
diff --git a/README.md b/README.md
@@
-- Raw runs now parallelize LLM prompting and compilation prep across multiple CPU workers (default: 4) that feed a shared GPU evaluation queue.
-- Override CPU fan-out globally via YAML (`raw_concurrency: 4`) or CLI overrides—per-model overrides are disabled so every target uses the same worker pool.
-- Tune GPU queue throughput with `raw_gpu_concurrency` (default 1 consumer thread) if the device has headroom.
+- Raw runs now parallelize LLM prompting and compilation prep across multiple CPU workers (default: 16) that feed a shared GPU evaluation queue.
+- Override CPU fan-out globally via YAML (`raw_concurrency: 16`) or CLI overrides—per-model overrides are disabled so every target uses the same worker pool.
+- Tune GPU queue throughput with `raw_gpu_concurrency` (default 2 consumer threads) if the device has headroom.
```

## validate
- `uv run python eval.py --config configs/quick_test.yaml `
- Confirm job logs show 16 CPU workers dispatched with GPU queue size 2.

# 2025-10-25 Add quick_test config

## rationale
Provide a checked-in smoke-test YAML so users can immediately run a three-problem raw sweep with the new concurrency defaults.

## patch
```diff
diff --git a/configs/quick_test.yaml b/configs/quick_test.yaml
new file mode 100644
@@
+description: Minimal smoke-test sweep (single provider/model, three TritonBench problems).
+
+modes:
+  - raw
+
+languages:
+  - triton
+
+models:
+  - provider: groq
+    model: llama-3.1-8b-instant
+
+defaults:
+  num_runs: 1
+  profile_stages: false
+
+verbose: true
+profile_stages: false
+fast_p_threshold: null
+
+hardware:
+  gpu_architecture: Ampere
+  gpu_id: 0
+
+problems:
+  levels: [1]
+  problem_ids: null
+  max_problems: 3
+
+artifacts:
+  json_dir: json
+  plots_dir: plots
+
+visualization:
+  enabled: false
```

## validate
- `uv run python eval.py --config configs/quick_test.yaml `
- Ensure run directory `runs/<timestamp>_raw_triton_groq_llama-3_1-8b-instant/` exists with manifest and three problem folders.

# 2025-10-25 Add test_providers config

## rationale
Ship a multi-provider smoke-test config equivalent to `all_providers.yaml`, but intended for one-pass validation runs.

## patch
```diff
diff --git a/configs/test_providers.yaml b/configs/test_providers.yaml
new file mode 100644
@@
+description: Multi-provider smoke test mirroring all_providers (single run each).
+
+modes:
+  - raw
+  - agentic
+
+languages:
+  - cuda
+  - triton
+
+models:
+  - provider: gemini
+    model: gemini-2.5-pro
+  - provider: gemini
+    model: gemini-2.5-flash
+  - provider: anthropic
+    model: claude-sonnet-4-5
+  - provider: anthropic
+    model: claude-haiku-4-5
+  - provider: openai
+    model: gpt-5
+  - provider: openrouter
+    model: z-ai/glm-4.6
+  - provider: xai
+    model: grok-code-fast-1
+  - provider: xai
+    model: grok-4-fast-reasoning
+  - provider: xai
+    model: grok-4-0709
+  - provider: groq
+    model: moonshotai/kimi-k2-instruct-0905
+
+defaults:
+  num_runs: 1
+  profile_stages: true
+  raw:
+    cpu_concurrency: 16
+    gpu_concurrency: 2
+    max_jobs: 16
+
+verbose: true
+profile_stages: true
+fast_p_threshold: null
+raw_concurrency: 16
+raw_gpu_concurrency: 2
+raw_max_jobs: 16
+
+hardware:
+  gpu_architecture: Ampere
+  gpu_id: 0
+
+problems:
+  levels: [1, 2]
+  problem_ids: null
+  max_problems: 100
+
+agentic:
+  max_debug_attempts: 3
+  max_optimization_cycles: 2
+  reflector_model: gpt-4-turbo
+  optimizer_model: gpt-4-turbo
+
+artifacts:
+  json_dir: json
+  plots_dir: plots
+
+visualization:
+  enabled: true
```

## validate
- `uv run python eval.py --config configs/test_providers.yaml`
- Confirm `runs/` contains raw + agentic manifests for each provider/model entry.

# 2025-10-26 Truncate formatted output previews

## rationale
When printing results to the terminal we only need a glimpse of the final formatted kernel. Show the first three lines, an ellipsis, and the last three lines, then point users to the on-disk artifact for full inspection.

## patch
```diff
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
 def _problem_directory_name(problem_id: int, problem_name: str | None) -> str:
     base = problem_name or f"problem_{problem_id}"
     return f"kernel_{problem_id}_{sanitize_component(base)}"


def _print_error(message: str) -> None:
@@
def _print_message_block(title: str, content: str | list) -> None:
@@
def _print_formatted_preview(title: str, file_path: Path, directory: Path) -> None:
@@
        _write_text(problem_dir / "reference.py", record.get("ref_arch_src"))
        _write_text(problem_dir / "generated_code.py", record.get("generated_code"))
        _print_formatted_preview(
            f"[Raw] Formatted output for {problem_dir.name}",
            problem_dir / "generated_code.py",
            problem_dir,
        )
diff --git a/src/agentic/agents/OptimAgent.py b/src/agentic/agents/OptimAgent.py
@@
def _truncate_lines(lines: List[str], head: int = 10, tail: int = 5) -> List[str]:
@@
def _format_messages(messages: Iterable[Dict[str, Any]]) -> str:
@@
def _stringify_response(response: Any) -> str:
@@
def _print_error(message: str) -> None:
@@
diff --git a/src/agentic/runner/main.py b/src/agentic/runner/main.py
@@
def _kernel_dir_name(problem_name: str) -> str:
    return sanitize_filename(problem_name)


def _print_formatted_preview(title: str, file_path: Path, directory: Path) -> None:
@@
        final_code = _final_code(mem)
        _write_text(problem_dir / "final_code.py", final_code)

        raw_candidates = getattr(mem, "raw_code", None)
        if raw_candidates and isinstance(raw_candidates, list):
            _write_text(problem_dir / "last_raw_candidate.py", raw_candidates[0] or "")

        _print_formatted_preview(
            f"[Agentic] Formatted output for {problem_dir.name}",
            problem_dir / "final_code.py",
            problem_dir,
        )
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-    run_plan.sort(key=lambda entry: (0 if entry[0] == "agentic" else 1, entry[1], entry[2].get("provider", "")))
+    run_plan.sort(key=lambda entry: 0 if entry[0] == "agentic" else 1)
```

## validate
- `uv run python eval.py --config configs/quick_test.yaml`
- Confirm console output shows truncated previews plus absolute paths for each kernel's formatted code.

# 2025-10-26 Record run elapsed seconds

## rationale
Surface the total wall-clock for each provider/model sweep so slow runs (e.g., GROK-4) are obvious without external profiling.

## patch
- Track run start/end in `src/raw/runner.py`, print the elapsed time, write `elapsed_seconds` into `manifest.yaml`, and return it with the artifact bundle.
- Mirror the same timing logic for agentic runs (`src/agentic/runner/main.py`, `emit_agentic_artifacts`) so manifests capture comparable numbers.
- Bubble the timing through `_run_and_collect` and batch caching (`src/batch_runner.py`) so cached payloads retain the measurement.

## validate
- `uv run python eval.py --config configs/test_providers.yaml`
- Observe `Completed run ... in X.Ys` messages and confirm the same values appear in each run directory’s `manifest.yaml` under `elapsed_seconds`.

# 2025-10-26 Disable grok-4 models in smoke test

## rationale
XAI grok-4 requests stalled for >30 minutes; removing them from the smoke suite keeps quick validation fast.

## patch
- Drop `xai/grok-4-0709` and `xai/grok-4-fast-reasoning` from `configs/test_providers.yaml` so only responsive providers remain.
- Update the README config table to reflect the new model count (8).

## validate
- `uv run python eval.py --config configs/test_providers.yaml`
- Confirm the run skips grok-4 entries and completes without long waits.

# 2025-10-26 Align agentic models with generator

## rationale
Ensure agentic runs evaluate the requested provider/model end-to-end by reusing the generator model for optimizer/reflector phases instead of hard-coded `gpt-4-turbo`.

## patch
- Default `AgenticConfig` reflective/optimizer fields to `None` (`config.py`).
- When loading YAML, stop injecting explicit model names and override the agentic config to the current `generator_model` in `yaml_to_benchmark_config` (`src/batch_runner.py`).
- Drop `reflector_model` / `optimizer_model` entries from shipped configs and documentation.

## validate
- `uv run python eval.py --config configs/only_kimi.yaml`
- Inspect manifest metadata to confirm the agentic runs hit Groq Kimi instead of a secondary model.

# 2025-10-26 Tighten agentic log truncation

## rationale
Ensure every agentic API request/response stays readable by limiting each block to 3 leading lines, `...`, and 3 trailing lines (total ≤7 rows) to eliminate console spam from large prompts.

## patch
- Update `_truncate_lines` in `src/agentic/agents/OptimAgent.py` to use defaults `head=3`, `tail=3` so message and response logs follow the 7-line preview format.

## validate
- Trigger an agentic run (e.g., `uv run python eval.py --config configs/test_providers.yaml`) and confirm each `=== AGENTIC LLM REQUEST/RESPONSE ===` block shows at most 3 lines, an ellipsis, and 3 lines of content.

# 2025-10-27 Suppress agentic prompt/response bodies

## rationale
Keep the console focused on phase transitions by hiding full prompts/responses for agentic requests.

## patch
- Replace `_format_messages`, `_format_kwargs`, and `_stringify_response` to emit placeholder strings rather than the full payload in `src/agentic/agents/OptimAgent.py`.

## validate
- `uv run python eval.py --config configs/test_providers.yaml`
- Observe that each agentic stage prints only the phase banner (request/response) without message content.

# 2025-10-27 Restore only_kimi config

## rationale
Provide a dedicated Groq Kimi baseline to measure infrastructure overhead with a single dependable provider.

## patch
- Reintroduce `configs/only_kimi.yaml` covering raw + agentic (cuda/triton) runs for `moonshotai/kimi-k2-instruct-0905`.
- Document the preset again in README’s configuration table.

## validate
- `uv run python eval.py --config configs/only_kimi.yaml`
- Confirm both raw and agentic manifests contain `elapsed_seconds` and use the Kimi model in every stage.

# 2025-10-26 Harmonize Progress Output

## rationale
- Remove per-kernel spam and make raw evaluation emit only the requested Generation/Evaluation progress bars so the Kimi run stays readable.
- Give the agentic loop the same treatment: stage-scoped progress bars, silent iteration logs, and no per-problem request chatter.
- Freeze batch ordering (raw before agentic, CUDA before Triton) and disable verbose mode in configs so the quiet output is consistent.

## patch
```diff
diff --git a/src/batch_runner.py b/src/batch_runner.py
@@
-        verbose=True,
+        verbose=False,
@@
-    run_plan.sort(key=lambda entry: 0 if entry[0] == "agentic" else 1)
+    mode_priority = {"raw": 0, "agentic": 1}
+    language_priority = {"cuda": 0, "triton": 1}
+
+    def _sort_key(entry: tuple[str, str, Dict[str, Any]]) -> tuple[int, int, str, str]:
+        mode_key = mode_priority.get(entry[0], 99)
+        language_key = language_priority.get(entry[1], 99)
+        provider = str(entry[2].get("provider", ""))
+        model = str(entry[2].get("model", ""))
+        return (mode_key, language_key, provider, model)
+
+    run_plan.sort(key=_sort_key)
diff --git a/src/raw/runner.py b/src/raw/runner.py
@@
-def _update_progress(progress_state: Dict[str, Any]) -> None:
-    if not progress_state:
-        return
-
-    lock: threading.Lock = progress_state["lock"]
-    progress_state["completed"] += 1
-    completed = progress_state["completed"]
-    total = progress_state["total"]
-    label = progress_state["label"]
-    prefix = progress_state["prefix"]
-    bar = _render_progress_bar(completed, total)
-    sys.stdout.write(f"\r{prefix} {bar} {completed}/{total}")
-    sys.stdout.flush()
-    if completed >= total:
-        sys.stdout.write("\n")
-        sys.stdout.flush()
+def _update_progress(progress_state: Dict[str, Any]) -> None:
+    if not progress_state:
+        return
+
+    lock: threading.Lock = progress_state["lock"]
+    bar: tqdm = progress_state["bar"]
+    with lock:
+        bar.update(1)
+
+
+def _close_bar(bar: tqdm | None) -> None:
+    if bar is None:
+        return
+    try:
+        bar.close()
+    except Exception:  # noqa: BLE001
+        pass
@@
-    progress_bar = tqdm(
-        total=len(problems),
-        desc=f"{config.mode}-{config.language}-{config.provider}",
-        unit="kernel",
-        leave=False,
-    )
-    progress_state: Dict[str, Any] = {
-        "lock": threading.Lock(),
-        "total": len(problems),
-        "completed": 0,
-        "label": f"{config.mode}-{config.language}",
-        "prefix": "[Raw]",
-    }
+    cpu_bar = tqdm(
+        total=len(problems),
+        desc="Generation",
+        unit="kernel",
+        leave=True,
+        dynamic_ncols=True,
+    )
+    cpu_progress_state: Dict[str, Any] = {
+        "lock": threading.Lock(),
+        "bar": cpu_bar,
+    }
+
+    gpu_worker_count = max(1, config.raw_gpu_concurrency or 1)
+    gpu_bar = tqdm(
+        total=len(problems),
+        desc="Evaluation",
+        unit="kernel",
+        leave=True,
+        dynamic_ncols=True,
+    )
+    gpu_progress_state: Dict[str, Any] = {
+        "lock": threading.Lock(),
+        "bar": gpu_bar,
+    }
+
+    print(
+        f"[Raw] CPU workers: {worker_count} | GPU workers: {gpu_worker_count}"
+    )
+
+    manager = None
@@
-        shared_results[problem_id] = result
-        _update_progress(progress_state)
+        shared_results[problem_id] = result
+        _update_progress(gpu_progress_state)
@@
-            initargs=(config, gen_cfg.level, job_queue, shared_results, build_root),
+            initargs=(
+                config,
+                gen_cfg.level,
+                job_queue,
+                shared_results,
+                build_root,
+                cpu_progress_state,
+            ),
@@
-                    _update_progress(progress_state)
-                    print(f"[Raw] Prepared level {gen_cfg.level} problem {pid}")
-                else:
-                    print(f"[Raw] Prepared level {gen_cfg.level} problem {pid}")
-                    if result and not result.get("queued", True):
-                        _update_progress(progress_state)
+                    _update_progress(cpu_progress_state)
+                    _update_progress(gpu_progress_state)
+                    _print_error(f"[Raw] CPU future failed for problem {pid}: {exc}")
+                    continue
+
+                queued = bool(result.get("queued", True))
+                if not queued:
+                    _update_progress(gpu_progress_state)
@@
-        _cpu_initializer(config, gen_cfg.level, job_queue, shared_results, build_root)
+        _cpu_initializer(
+            config,
+            gen_cfg.level,
+            job_queue,
+            shared_results,
+            build_root,
+            cpu_progress_state,
+        )
         for pid in problems:
-            print(f"[Raw] Preparing level {gen_cfg.level} problem {pid}")
             result_info = _cpu_prepare_problem(pid)
-                if result_info and not result_info.get("queued", True):
-                    _update_progress(progress_state)
+            if result_info and not result_info.get("queued", True):
+                _update_progress(gpu_progress_state)
@@
-    print(f"[Raw] Compiled: {compiled_count}/{total} | Correct: {correct_count}/{total}")
-    if failure_breakdown:
-        print("[Raw] Failure breakdown:")
-        ...
+    verbose = bool(getattr(config, "verbose", False))
+    if verbose:
+        print(f"[Raw] Compiled: {compiled_count}/{total} | Correct: {correct_count}/{total}")
+        if failure_breakdown:
+            print("[Raw] Failure breakdown:")
+            ...
@@
-    print(f"[Raw] Completed run for {len(problems)} problems in {elapsed_seconds:.1f}s. Results saved to {manifest_path}")
-
-    manager.shutdown()
+    if verbose:
+        print(f"[Raw] Completed run for {len(problems)} problems in {elapsed_seconds:.1f}s. Results saved to {manifest_path}")
+
+    if manager is not None:
+        manager.shutdown()
diff --git a/src/agentic/agents/OptimAgent.py b/src/agentic/agents/OptimAgent.py
@@
-from typing import Any, Dict, Iterable, List
+from typing import Any, Dict, Iterable, List
+
+from tqdm.auto import tqdm
@@
-        self._iteration_count = iteration_num
-        iteration_range = range(start_iter, start_iter + iteration_num)
-        iter_bar = tqdm(iteration_range, desc=self.progress_desc, unit="iter", leave=False)
-        for iter in iteration_range:
+        self._iteration_count = iteration_num
+        iteration_range = range(start_iter, start_iter + iteration_num)
+        with tqdm(
+            total=iteration_num,
+            desc=self.progress_desc,
+            unit="iter",
+            leave=False,
+            dynamic_ncols=True,
+        ) as iter_bar:
+            for iter in iteration_range:
                 self._active_iteration = iter
                 if output_path is not None:
                     root, extension = os.path.splitext(output_path)
                     iter_path = f"{root}_{iter}{extension}"
                     mem_output_path = f"{root}_mem_{iter}.json"
-
-            for mem in self.memories[start_idx:(start_idx + data_len)]:
-                self.generate_solution(mem, temperature=temperature)
+                with tqdm(
+                    total=data_len,
+                    desc="Solutions",
+                    unit="kernel",
+                    leave=False,
+                    dynamic_ncols=True,
+                ) as solution_bar:
+                    for mem in self.memories[start_idx:(start_idx + data_len)]:
+                        self.generate_solution(mem, temperature=temperature)
+                        solution_bar.update(1)
@@
-            for mem in self.memories[start_idx:(start_idx + data_len)]:
-                try:
-                    pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr = self.dataset.test_opt_correctness(mem.raw_code[0], mem.ps.filename, tmp_dir, exe_dir=exe_dir)
-                except Exception as e:
-                    _print_error(f"failed to test the code for {mem.ps.filename}: {e}")
-                    ...
-                    continue
-
-                ...
+            with tqdm(
+                total=data_len,
+                desc="Execution",
+                unit="kernel",
+                leave=False,
+                dynamic_ncols=True,
+            ) as execution_bar:
+                for mem in self.memories[start_idx:(start_idx + data_len)]:
+                    try:
+                        (
+                            pass_call,
+                            pass_exe,
+                            call_stdout,
+                            call_stderr,
+                            exe_stdout,
+                            exe_stderr,
+                        ) = self.dataset.test_opt_correctness(mem.raw_code[0], mem.ps.filename, tmp_dir, exe_dir=exe_dir)
+                    except Exception as e:
+                        _print_error(f"failed to test the code for {mem.ps.filename}: {e}")
+                        ...
+                        execution_bar.update(1)
+                        continue
+
+                    ...
+                    execution_bar.update(1)
@@
-            for mem in self.memories[start_idx:(start_idx + data_len)]:
-                self.generate_reflexion(mem, temperature=temperature)
+            with tqdm(
+                total=data_len,
+                desc="Reflections",
+                unit="kernel",
+                leave=False,
+                dynamic_ncols=True,
+            ) as reflection_bar:
+                for mem in self.memories[start_idx:(start_idx + data_len)]:
+                    self.generate_reflexion(mem, temperature=temperature)
+                    reflection_bar.update(1)
@@
-        try:
-            request_payload = {
-                "stage": "generate_solution",
-                ...
-            }
-            print(
-                f"[Agentic] Iter {self._active_iteration} request "
-                f"(solution) -> {mem.ps.filename}"
-            )
-            response_obj = self.model.generate(msg, temperature=temperature, max_tokens=15000)
+        try:
+            response_obj = self.model.generate(msg, temperature=temperature, max_tokens=15000)
@@
-        print(
-            f"[Agentic] Iter {self._active_iteration} response "
-            f"(solution) <- {mem.ps.filename}"
-        )
@@
-        request_payload = {
-            "stage": "generate_reflection",
-            ...
-        }
-        print(
-            f"[Agentic] Iter {self._active_iteration} request "
-            f"(reflection) -> {mem.ps.filename}"
-        )
-
         try:
             response_obj = self.model.generate(reflect_msg, temperature=temperature)
@@
-        print(
-            f"[Agentic] Iter {self._active_iteration} response "
-            f"(reflection) <- {mem.ps.filename}"
-        )
```

## validation
- `uv run python -m compileall src/batch_runner.py src/raw/runner.py src/agentic/agents/OptimAgent.py`
- Manual log review: ensure raw runs show only Generation/Evaluation bars and agentic runs display stage bars without per-problem chatter.

# 2025-10-26 Fix Raw Progress Multiprocessing

## rationale
- `uv run python eval.py --config configs/only_kimi.yaml` crashed because the raw runner passed a `threading.Lock` inside the progress state to worker processes, which `spawn` cannot pickle.
- Keep the dual progress bars while ensuring only the parent process touches the locks/bars.

## patch
- `src/raw/runner.py`: stop storing progress state in the child initializer, move `_update_progress` calls to the parent loop (both success and failure paths), and update the sequential path accordingly.

## validation
- `uv run python -m compileall src/raw/runner.py`
- Re-run `uv run python eval.py --config configs/only_kimi.yaml` (expect progress bars without the pickling failure; execution still depends on provider access).

# 2025-10-26 Rate Limit Backoff

## rationale
- Raw runs routinely hit Groq’s 429 rate limits when we hammer 100 kernels in parallel; we received errors telling us to “try again in 268.8ms” and currently abort the kernel entirely.
- Insert a simple retry loop (up to 3 attempts) with a 10s pause so temporary rate spikes can clear without losing the batch.
- Apply the same logic to the formatter call path since it also consumes LLM quota.

## patch
- `src/raw/runner.py`: wrap provider and formatter `generate` calls in retry loops that catch rate-limit signatures (`"rate limit"`/`"429"`) and sleep before retrying; keep existing verbose logging.

## validation
- `uv run python -m compileall src/raw/runner.py`
- Re-run `uv run python eval.py --config configs/only_kimi.yaml` to confirm the rate-limit error now pauses and continues rather than failing the target.

# 2025-10-26 TritonBench Expansion Blueprint & Full Eval Utilities

## rationale
- Document a concrete plan for importing the full TritonBench corpus so agentic benchmarks can cover every kernel.
- Provide a scripted entry point (`scripts/run_full_tritonbench_eval.sh`) and config (`configs/all_providers_full.yaml`) to run all providers across the expanded dataset with logging and visualization.

## patch
- Added `docs/tritonbench_corpus_expansion.md` outlining sourcing, normalization, integration, and validation steps.
- Created `configs/all_providers_full.yaml` that mirrors the multi-provider matrix while removing the problem cap and routing plots to `plots/full/`.
- Added `scripts/run_full_tritonbench_eval.sh` to tee command output into timestamped logs and accept extra CLI overrides.

## validate
- `chmod +x scripts/run_full_tritonbench_eval.sh`
- `uv run python -m compileall src/raw/runner.py` (ensures code touched indirectly still compiles post-change).
- Smoke test (optional) `scripts/run_full_tritonbench_eval.sh --num-runs 5 --mode agentic --language cuda` once the full dataset is staged.

# 2025-10-26 Raw-Only Mega Eval Harness

## rationale
- Provide a turnkey configuration/script for bulk raw-only evaluation while the agentic corpus remains sparse.
- Dedicate plot output to `plots/raw/` to avoid mixing with agentic experiments.

## patch
- Added `configs/all_providers_raw.yaml` (raw-only, cuda+triton languages, `max_problems: 100`).
- Added `scripts/run_full_raw_eval.sh` that mirrors the full-run helper but targets the raw-only config.

## validate
- `chmod +x scripts/run_full_raw_eval.sh`
- Optional smoke: `scripts/run_full_raw_eval.sh --num-runs 5 --language cuda`.
# 2025-10-27 Modal Raw GPU Runner

## rationale
- Introduce a Modal-managed path for raw kernel runs with configurable CUDA image, GPU tier, and strict default timeouts.
- Default raw CPU worker settings now honor the host's logical core count via a `max` sentinel across every YAML config.
- Forward `GROQ_API_KEY` into Modal jobs while documenting setup, hard-failing token issues, and preserving the three-command quickstart.

## patch
```diff
diff --git a/README.md b/README.md
index 4cb6d78..8554074 100644
--- a/README.md
+++ b/README.md
@@ -42,8 +42,8 @@ KernelBench-v3/
 ├── data/TritonBench/        # TritonBench dataset & metrics (copied in repo)
 ├── json/                    # Cached aggregate metrics per mode/language/model
 ├── plots/                   # Visualization artifacts (PNG only)
-├── outputs/                 # Agentic run outputs (raw JSONL traces)
-├── runs/                    # Raw run outputs (per-problem JSONL)
+├── outputs/                 # Legacy agentic outputs (JSONL, kept for back-compat)
+├── runs/                    # Raw & agentic run artifacts (manifests + per-problem logs)
 ├── scripts/                 # Legacy scripts (generate, eval, analysis)
 ├── src/
 │   ├── providers/           # Provider wrappers (OpenAI, Groq, etc.)
@@ -67,6 +67,15 @@ cd /home/infatoshi/gpu_benchmarks/KernelBench-v3
 ```
 
 ### 2. Automated Setup & Smoke Test
+
+The repository is now an [uv](https://github.com/astral-sh/uv) project. After cloning you can bring up a fresh environment with:
+
+```bash
+uv sync
+source .venv/bin/activate
+```
+
+This creates `.venv` and installs every dependency declared in `pyproject.toml`.
 ```bash
 bash setup_and_test.sh
 ```
@@ -85,7 +94,25 @@ export GROQ_API_KEY="your-groq-key"  # required for smoke tests & Groq runs
 # export GEMINI_API_KEY="your-gemini-key"
 ```
 
-Formatter overrides (beta) are configured via CLI/YAML—see [Formatter Beta](#formatter-beta-optional).
+Formatter overrides are configured via CLI/YAML—see [Formatter (Always On)](#formatter-always-on).
+
+### Modal Raw GPU Runs (Experimental)
+
+Modal orchestration currently supports **raw kernels only**; the agentic pipeline remains offline to avoid runaway GPU costs.
+
+#### Quickstart
+
+```bash
+git clone https://github.com/Infatoshi/KernelBench-v3.git && cd KernelBench-v3
+bash setup.sh
+bash run.sh
+```
+
+`setup.sh` installs `uv`, synchronizes the project environment, and prompts you to create a Modal token via `modal token new`. `run.sh` submits the Modal job (`tools/modal_raw.py`) that provisions the CUDA image, syncs the repo in a writable workspace, and launches the raw benchmark.
+
+Configuration lives in `configs/modal_raw.yaml`. Adjust the `modal.image` block to change the CUDA version or Ubuntu tag (`ubuntu24.04` by default), and update `modal.gpu.name` to target a different GPU tier. The default subprocess timeout is 120 seconds; set `MODAL_RUN_TIMEOUT=0` and `modal.timeouts.process_seconds: 0` if you need to disable the safeguard.
+
+`run.sh` forwards `GROQ_API_KEY` to Modal automatically when the variable is set locally; the task logs whether the secret was attached before submitting the job. If Modal credentials are missing, the script aborts with instructions to rerun `uv run modal token new`.
 
 ---
 
@@ -114,34 +141,29 @@ BenchmarkConfig(
 
 You can modify defaults directly or override via CLI flags.
 
-### Formatter Beta (Optional)
+### Formatter (Always On)
 
-- Enable the Groq-based formatter on the CLI with `--groq-formatter-beta`, or provide explicit overrides:
-  - `--formatter-provider groq`
-  - `--formatter-model moonshotai/kimi-k2-instruct-0905`
-  - `--formatter-base-url https://api.groq.com/openai/v1`
-- YAML configs support a `formatter` block on each model:
+- The Groq formatter now runs for every generation—there is no concise/disabled mode.
+- Override provider/model globally via CLI (`--formatter-provider`, `--formatter-model`, `--formatter-base-url`) or YAML defaults:
 
   ```yaml
   defaults:
     formatter:
       provider: groq
       model: moonshotai/kimi-k2-instruct-0905
-  models:
-    - provider: openai
-      model: gpt-5
-      formatter:
-        provider: groq
-        model: moonshotai/kimi-k2-instruct-0905
+      base_url: https://api.groq.com/openai/v1
   ```
 
+- Per-model overrides remain available, but leaving them unset inherits the global defaults so that every kernel gets a formatted candidate.
+
 The formatter performs a second LLM pass to enforce a single fenced `python` block containing a complete `ModelNew` implementation, complementing the stricter CUDA/Triton prompts in `src/prompt_constructor.py`.
 
 ### Raw Concurrency
 
-- Raw runs now parallelize LLM prompting and compilation prep across multiple CPU workers (default: 8) that feed a shared GPU evaluation queue.
-- Override CPU fan-out via YAML (`raw_concurrency: 4`).
-- Tune GPU queue throughput with `raw_gpu_concurrency` (default 1 consumer thread) if the device has headroom.
+- Raw runs now parallelize LLM prompting and compilation prep across as many CPU workers as your host exposes (default equals `os.cpu_count()`), feeding a shared GPU evaluation queue.
+- Override CPU fan-out globally via YAML with `raw_concurrency: max` (default) or any explicit integer; CLI overrides accept the same token.
+- `raw.max_jobs` follows the same convention, defaulting to the CPU thread count so job queues never bottleneck below available workers.
+- Tune GPU queue throughput with `raw_gpu_concurrency` (default 2 consumer threads) if the device has headroom.
 - Set both to `1` for fully sequential debugging, or raise CPU workers while keeping GPU slots low to overlap compilation with execution.
 
 Performance profiling is disabled by default (no GPU timing trials). Re-enable it per run with `fast_p_threshold: 1.2` or via CLI `--fast-p-threshold 1.2` if you need speedup metrics.
@@ -150,6 +172,48 @@ Stage profiling (CPU prep, queue wait, GPU compile/correctness/perf) is off by d
 
 ---
 
+## 📦 Run Artifact Layout
+
+Every invocation now writes a single directory under `runs/` with a timestamped slug:
+
+```
+runs/
+  20251025_012633_agentic_triton_openai_gpt-5/
+    manifest.yaml
+    kernel_001_context_attn_nopad.py/
+      summary.txt
+      instruction.txt
+      reference_solution.py
+      test_harness.py
+      final_code.py
+      last_raw_candidate.py
+      history/
+        iteration_00/solution.txt
+        iteration_00/execution.txt
+        iteration_00/performance.txt
+        iteration_00/reflection.txt
+        iteration_01/...
+    kernel_002_...
+```
+
+- **manifest.yaml** captures provider/model metadata, concurrency settings, formatter configuration, and a per-kernel summary (call/exec/perf pass flags and runtime hints).
+- The manifest also records `elapsed_seconds` (wall-clock runtime rounded to 0.1s) so you can compare how long each provider/model/problem sweep took.
+- **summary.txt** highlights the outcome for the kernel along with any errors.
+- **history/** holds plain-text logs for each iteration and stage. Requests, kwargs, responses, execution stdout/stderr, and performance diagnostics are all written without JSON so you can skim directly in the terminal.
+- Raw runs emit the same structure, with `prompt.txt`, `response_raw.txt`, `response_formatted.txt`, and `metrics.yaml` mirroring the agentic history.
+
+Legacy JSONL traces remain untouched inside `outputs/` for backwards compatibility with older analysis scripts, but new tooling should rely on the `runs/` layout.
+
+### Logging
+
+- Verbose mode is always enabled. Batch runner headers render as `mode + language + provider/model` in green, and no concise mode is exposed.
+- Raw and agentic runners dump every LLM exchange in plain text—system prompts, user prompts, kwargs, and responses—without JSON framing so the terminal output stays readable.
+- Agentic iterations record the same information under each kernel's `history/` folder for offline inspection.
+- To keep the console manageable, LLM prompts/responses and formatted kernels are truncated to head/tail slices with an ellipsis and a loud error banner is emitted whenever a stage fails.
+- Agentic runs reuse the evaluated provider/model for every phase (generation, optimizer, reflection) so metrics reflect that single LLM end-to-end.
+
+---
+
 ## 🚀 Unified CLI Usage (`eval.py`)
 
 ### Batch Mode (Recommended)
@@ -211,8 +275,7 @@ For quick one-off tests:
 ```bash
 uv run python eval.py --mode raw \
   --provider groq \
-  --model llama-3.3-70b-versatile \
-  --num-runs 5
+  --model llama-3.3-70b-versatile
 ```
 
 ### Examples
@@ -446,8 +509,7 @@ problems:
 agentic:                     # Only used if mode=agentic
   max_debug_attempts: 3
   max_optimization_cycles: 2
-  reflector_model: gpt-4-turbo
-  optimizer_model: gpt-4-turbo
+  # reflector_model and optimizer_model automatically reuse the generator model.
 
 fast_p_threshold: 1.2        # Speedup threshold
 
@@ -477,6 +539,8 @@ visualization:
 | Config | Description | Models | Problems |
 |--------|-------------|--------|----------|
 | `quick_test.yaml` | Single model, 3 problems | Groq Llama | 3 |
+| `test_providers.yaml` | Multi-provider smoke test | 8 models | 1 |
+| `only_kimi.yaml` | Kimi K2 across all modes/languages | 1 model | full |
 | `test_visualization.yaml` | Two models for viz testing | 2× Groq | 2 |
 | `example_batch.yaml` | Full multi-provider comparison | 5 models | 10 |
 | `multi_provider_benchmark.yaml` | Production benchmark | 6 models | 20 |
@@ -497,7 +561,7 @@ Outputs saved to `visualizations/` as PNG/PDF with timestamp.
 Run this command (preflight + benchmark) once your API keys are exported:
 
 ```bash
-uv run python eval.py --config configs/test_visualization.yaml --num-runs 1 --profile-stages --verbose --groq-formatter-beta
+uv run python eval.py --config configs/test_visualization.yaml --profile-stages --verbose
 ```
 
 > The entrypoint automatically pings each provider/model with a "Hello." request before starting the benchmark and aborts if any credentials or model names are misconfigured.
@@ -557,4 +621,3 @@ MIT License (see `LICENSE`).
 - KernelBench-v2 (kernelbench-v1 + triton)
 - MultiKernelBench (i dont have the hardware yet… so pause)
 - TritonBench (covered in geak-eval and kernelbench-v2)
-
diff --git a/config.py b/config.py
index 8b4ac96..d8dd370 100644
--- a/config.py
+++ b/config.py
@@ -1,5 +1,6 @@
 from dataclasses import dataclass, field
 from typing import List, Literal
+import os
 
 EvaluationMode = Literal["raw", "agentic"]
 KernelLanguage = Literal["cuda", "triton"]
@@ -22,8 +23,13 @@ class ProblemSetConfig:
 class AgenticConfig:
     max_debug_attempts: int = 3
     max_optimization_cycles: int = 2
-    reflector_model: str = "gpt-4-turbo"
-    optimizer_model: str = "gpt-4-turbo"
+    reflector_model: str | None = None
+    optimizer_model: str | None = None
+
+
+def _cpu_worker_default() -> int:
+    """Return the maximum available CPU worker count."""
+    return max(1, os.cpu_count() or 1)
 
 
 @dataclass
@@ -43,8 +49,8 @@ class BenchmarkConfig:
     problems: ProblemSetConfig = field(default_factory=ProblemSetConfig)
     agentic: AgenticConfig = field(default_factory=AgenticConfig)
     fast_p_threshold: float | None = None
-    raw_concurrency: int = 8
+    raw_concurrency: int = field(default_factory=_cpu_worker_default)
     raw_gpu_concurrency: int = 1
-    raw_max_jobs: int = 8
+    raw_max_jobs: int = field(default_factory=_cpu_worker_default)
     generation_max_tokens: int = 4096
     formatter_max_tokens: int | None = None
diff --git a/configs/all_providers.yaml b/configs/all_providers.yaml
index 85ab82d..87814ea 100644
--- a/configs/all_providers.yaml
+++ b/configs/all_providers.yaml
@@ -11,79 +11,38 @@ languages:
 models:
   - provider: gemini
     model: gemini-2.5-pro
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: gemini
     model: gemini-2.5-flash
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: anthropic
     model: claude-sonnet-4-5
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: anthropic
     model: claude-haiku-4-5
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: openai
     model: gpt-5
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
-  - provider: openai
-    model: gpt-5-nano
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
-  - provider: openai
-    model: gpt-5-mini
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: openrouter
     model: z-ai/glm-4.6
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: xai
     model: grok-code-fast-1
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: xai
     model: grok-4-fast-reasoning
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
-  - provider: xai
-    model: grok-4-fast-non-reasoning
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
   - provider: xai
     model: grok-4-0709
-    raw_concurrency: 1
-    raw_gpu_concurrency: 1
-    raw_max_jobs: 1
+  - provider: groq
+    model: moonshotai/kimi-k2-instruct-0905
 
 defaults:
-  num_runs: 1
   profile_stages: true
   raw:
-    cpu_concurrency: 1
-    gpu_concurrency: 1
-    max_jobs: 1
+    cpu_concurrency: max
+    gpu_concurrency: 2
+    max_jobs: max
 
 verbose: true
 profile_stages: true
 fast_p_threshold: null
-raw_concurrency: 1
-raw_gpu_concurrency: 1
-raw_max_jobs: 1
+raw_concurrency: max
+raw_gpu_concurrency: 2
+raw_max_jobs: max
 
 hardware:
   gpu_architecture: Ampere
@@ -97,8 +56,6 @@ problems:
 agentic:
   max_debug_attempts: 3
   max_optimization_cycles: 2
-  reflector_model: gpt-4-turbo
-  optimizer_model: gpt-4-turbo
 
 artifacts:
   json_dir: json
diff --git a/src/batch_runner.py b/src/batch_runner.py
index d4bdc0d..dd95287 100644
--- a/src/batch_runner.py
+++ b/src/batch_runner.py
@@ -15,11 +15,15 @@ from typing import Any, Dict, List, Tuple
 import yaml
 
 from config import AgenticConfig, BenchmarkConfig, HardwareConfig, ProblemSetConfig
-from metrics import compute_core_metrics, load_metrics, parse_jsonl_results, save_metrics
+from metrics import load_metrics, save_metrics
 from providers import verify_model_responds_hello
 
 GREEN = "\033[32m"
 RESET = "\033[0m"
+HOST_CPU_COUNT = max(1, os.cpu_count() or 1)
+
+DEFAULT_FORMATTER_PROVIDER = "groq"
+DEFAULT_FORMATTER_MODEL = "moonshotai/kimi-k2-instruct-0905"
 
 
 def load_yaml_config(yaml_path: str | Path) -> Dict[str, Any]:
@@ -32,6 +36,28 @@ def sanitize_component(value: str) -> str:
     return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
 
 
+def _resolve_cpu_concurrency(value: object | None) -> int:
+    """Coerce CPU worker counts, honoring the 'max' sentinel."""
+    if value is None:
+        return HOST_CPU_COUNT
+    if isinstance(value, str):
+        if value.strip().lower() == "max":
+            return HOST_CPU_COUNT
+        raise ValueError(f"Unsupported cpu concurrency token: {value}")
+    return max(1, int(value))
+
+
+def _resolve_max_jobs(value: object | None) -> int:
+    """Allow YAML to specify 'max' for job fan-out."""
+    if isinstance(value, str):
+        if value.strip().lower() == "max":
+            return HOST_CPU_COUNT
+        raise ValueError(f"Unsupported max_jobs token: {value}")
+    if value is None:
+        return HOST_CPU_COUNT
+    return max(1, int(value))
+
+
 def expand_run_matrix(yaml_data: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
     """Return the (mode, language, model) combinations requested by the config."""
     models = yaml_data.get("models", [])
@@ -147,11 +173,11 @@ def _resolve_agentic(yaml_data: Dict[str, Any], overrides: Dict[str, Any]) -> Ag
         ),
         "reflector_model": ag_data.get(
             "reflector_model",
-            ag_defaults.get("reflector_model", "gpt-4-turbo"),
+            ag_defaults.get("reflector_model"),
         ),
         "optimizer_model": ag_data.get(
             "optimizer_model",
-            ag_defaults.get("optimizer_model", "gpt-4-turbo"),
+            ag_defaults.get("optimizer_model"),
         ),
     }
 
@@ -175,20 +201,23 @@ def yaml_to_benchmark_config(
     provider = model_entry.get("provider") or yaml_data.get("provider", "openai")
     model_id = model_entry.get("model") or yaml_data.get("generator_model", "gpt-4-turbo")
     base_url = model_entry.get("base_url") or yaml_data.get("provider_base_url")
+
     formatter_defaults = defaults.get("formatter", {})
-    formatter_cfg = model_entry.get("formatter", {})
-    formatter_provider = formatter_cfg.get(
-        "provider",
-        yaml_data.get("formatter_provider", formatter_defaults.get("provider")),
-    )
-    formatter_model = formatter_cfg.get(
-        "model",
-        yaml_data.get("formatter_model", formatter_defaults.get("model")),
-    )
-    formatter_base_url = formatter_cfg.get(
-        "base_url",
-        yaml_data.get("formatter_base_url", formatter_defaults.get("base_url")),
-    )
+    formatter_cfg = model_entry.get("formatter") or {}
+    if not isinstance(formatter_cfg, dict):
+        formatter_cfg = {}
+
+    formatter_provider = formatter_cfg.get("provider") or yaml_data.get("formatter_provider")
+    if not formatter_provider:
+        formatter_provider = formatter_defaults.get("provider") or DEFAULT_FORMATTER_PROVIDER
+
+    formatter_model = formatter_cfg.get("model") or yaml_data.get("formatter_model")
+    if not formatter_model:
+        formatter_model = formatter_defaults.get("model") or DEFAULT_FORMATTER_MODEL
+
+    formatter_base_url = formatter_cfg.get("base_url") or yaml_data.get("formatter_base_url")
+    if not formatter_base_url:
+        formatter_base_url = formatter_defaults.get("base_url")
 
     num_runs = model_entry.get("num_runs", yaml_data.get("num_runs", defaults.get("num_runs")))
     if "num_runs" in cli_overrides:
@@ -200,18 +229,34 @@ def yaml_to_benchmark_config(
     if cli_overrides.get("profile_stages") is True:
         profile_stages = True
 
-    raw_concurrency = model_entry.get(
-        "raw_concurrency",
-        yaml_data.get("raw_concurrency", raw_defaults.get("cpu_concurrency", 8)),
-    )
-    raw_gpu_concurrency = model_entry.get(
-        "raw_gpu_concurrency",
-        yaml_data.get("raw_gpu_concurrency", raw_defaults.get("gpu_concurrency", 1)),
-    )
-    raw_max_jobs = model_entry.get(
-        "raw_max_jobs",
-        yaml_data.get("raw_max_jobs", raw_defaults.get("max_jobs", 8)),
-    )
+    raw_settings = yaml_data.get("raw", {}) or {}
+
+    raw_concurrency = cli_overrides.get("raw_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = yaml_data.get("raw_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = raw_settings.get("cpu_concurrency")
+    if raw_concurrency is None:
+        raw_concurrency = raw_defaults.get("cpu_concurrency")
+    raw_concurrency = _resolve_cpu_concurrency(raw_concurrency)
+
+    raw_gpu_concurrency = cli_overrides.get("raw_gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = yaml_data.get("raw_gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = raw_settings.get("gpu_concurrency")
+    if raw_gpu_concurrency is None:
+        raw_gpu_concurrency = raw_defaults.get("gpu_concurrency", 1)
+    raw_gpu_concurrency = int(raw_gpu_concurrency)
+
+    raw_max_jobs = cli_overrides.get("raw_max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = yaml_data.get("raw_max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = raw_settings.get("max_jobs")
+    if raw_max_jobs is None:
+        raw_max_jobs = raw_defaults.get("max_jobs")
+    raw_max_jobs = _resolve_max_jobs(raw_max_jobs)
 
     generation_max_tokens = model_entry.get("generation_max_tokens")
     if generation_max_tokens is None:
@@ -251,7 +296,7 @@ def yaml_to_benchmark_config(
         formatter_model=formatter_model,
         formatter_base_url=formatter_base_url,
         generator_model=model_id,
-        verbose=yaml_data.get("verbose", False),
+        verbose=False,
         num_runs=num_runs,
         profile_stages=profile_stages,
         hardware=hardware,
@@ -270,6 +315,10 @@ def yaml_to_benchmark_config(
 
     config = BenchmarkConfig(**config_kwargs)
 
+    if config.agentic:
+        config.agentic.reflector_model = config.generator_model
+        config.agentic.optimizer_model = config.generator_model
+
     return config
 
 
@@ -292,28 +341,20 @@ def json_cache_path(
     filename = f"{mode_safe}_{language_safe}_{provider_safe}_{model_safe}.json"
     return json_dir / filename
 
-
-def collect_metrics(results_path: Path) -> Dict[str, Any]:
-    records = parse_jsonl_results(results_path)
-    metrics = compute_core_metrics(records)
-    return {
-        "metrics": metrics,
-        "total_records": len(records),
-    }
-
-
 def _run_and_collect(
     config: BenchmarkConfig,
     run_fn,
 ) -> Dict[str, Any]:
     run_artifacts = run_fn(config) or {}
-    results_path = run_artifacts.get("results_path")
-    if results_path is None:
-        raise RuntimeError("Runner did not return a results_path."
+    run_dir = run_artifacts.get("run_dir")
+    if run_dir is None:
+        raise RuntimeError("Runner did not return a run directory."
                            " Ensure run(config) returns artifact metadata.")
     return {
-        "results_path": Path(results_path),
-        "run_dir": Path(run_artifacts.get("run_dir", Path(results_path).parent)),
+        "run_dir": Path(run_dir),
+        "manifest": Path(run_artifacts.get("manifest", Path(run_dir) / "manifest.yaml")),
+        "metrics": run_artifacts.get("metrics", {}),
+        "elapsed_seconds": run_artifacts.get("elapsed_seconds"),
     }
 
 
@@ -349,6 +390,18 @@ def run_batch_benchmark(
 
     results: List[Dict[str, Any]] = []
 
+    mode_priority = {"raw": 0, "agentic": 1}
+    language_priority = {"cuda": 0, "triton": 1}
+
+    def _sort_key(entry: tuple[str, str, Dict[str, Any]]) -> tuple[int, int, str, str]:
+        mode_key = mode_priority.get(entry[0], 99)
+        language_key = language_priority.get(entry[1], 99)
+        provider = str(entry[2].get("provider", ""))
+        model = str(entry[2].get("model", ""))
+        return (mode_key, language_key, provider, model)
+
+    run_plan.sort(key=_sort_key)
+
     for index, (mode, language, model_entry) in enumerate(run_plan, start=1):
         provider = model_entry.get("provider", yaml_data.get("provider", "unknown"))
         model_id = model_entry.get("model", yaml_data.get("generator_model", "unknown"))
@@ -363,6 +416,7 @@ def run_batch_benchmark(
 
         if cached_payload and cached_payload.get("config_fingerprint") == fingerprint:
             metrics = cached_payload.get("metrics", {})
+            artifacts = cached_payload.get("artifacts", {})
             print("  ↪ Using cached metrics (config fingerprint match).")
             results.append({
                 "provider": provider,
@@ -372,9 +426,10 @@ def run_batch_benchmark(
                 "status": "cached",
                 "metrics": metrics,
                 "metrics_path": str(cache_path),
-                "results_path": cached_payload.get("artifacts", {}).get("results_jsonl"),
+                "artifacts": artifacts,
                 "timestamp": cached_payload.get("timestamp"),
                 "config_fingerprint": fingerprint,
+                "elapsed_seconds": cached_payload.get("elapsed_seconds"),
             })
             print(f"✅ Cached: {provider}/{model_id}\n")
             continue
@@ -388,19 +443,20 @@ def run_batch_benchmark(
             else:
                 raise ValueError(f"Unsupported mode '{mode}' in run matrix.")
 
-            metrics_bundle = collect_metrics(artifact_info["results_path"])
+            metrics_bundle = artifact_info.get("metrics", {})
             payload = {
                 "provider": provider,
                 "model": model_id,
                 "mode": mode,
                 "language": language,
                 "timestamp": datetime.now().isoformat(),
-                "metrics": metrics_bundle["metrics"],
+                "metrics": metrics_bundle,
                 "config_fingerprint": fingerprint,
                 "artifacts": {
-                    "results_jsonl": str(artifact_info["results_path"]),
+                    "manifest": str(artifact_info.get("manifest")),
                     "run_dir": str(artifact_info["run_dir"]),
                 },
+                "elapsed_seconds": artifact_info.get("elapsed_seconds"),
             }
             save_metrics(cache_path, payload)
 
```

## validate
- `bash setup.sh`
- `MODAL_RUN_TIMEOUT=0 bash run.sh` (fails: Modal token missing)
- `uv run python -m compileall config.py src/batch_runner.py tools/modal_raw.py src/modal_support/config.py`

