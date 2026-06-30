# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience is a **cross-model KV Cache reuse patch framework** for open-source
serving stacks. The active disk-reuse baseline is **vLLM + LMCache MP + filesystem L2**;
**SGLang + LMCache** is retained as a legacy control path.

The new boundary is deliberately narrow:

- vLLM owns default model loading, scheduling, decoding, and inference correctness.
- LMCache owns KV storage, lookup, offload, eviction, and prefetch mechanics.
- GoldenExperience adds the control plane for **reusing KV Cache across models**.
- If a reuse plan is not safe or not calibrated, the stack falls back to normal engine
  prefill behavior.

## What This Project Focuses On

GoldenExperience is no longer trying to be an inference engine or a KV offload system.
It is intended to be carried as a small patch on top of LMCache, with runtime metadata
flowing from serving requests into LMCache lookup and retrieve paths.

The research/development target is three GoldenExperience reuse tracks:

| Track Name | Scenario | Goal | Default Strategy | Safety Gate |
| --- | --- | --- | --- | --- |
| GoldenLoRA | Base model <-> LoRA model | Reuse KV between a model and its LoRA fine-tuned variant | Adapter-delta gated aliasing | Same base, tokenizer, KV layout, LoRA drift probe |
| GoldenScale | Same model, different parameter sizes | Reuse KV across variants such as 7B <-> 14B | Direct alias if shapes match; otherwise layerwise projection | Layer/head mapping and projection calibration |
| GoldenBridge | Different base models | Explore broader cross-base reuse | Learned translator | Explicit calibration set, tokenizer bridge, task allowlist |

The names map to the implementation scenarios as follows: `GoldenLoRA` targets
`model_lora_mutual_reuse`, `GoldenScale` targets `same_model_different_parameter_size`,
and `GoldenBridge` targets `different_base_model`.

## Architecture

```text
vLLM request/session
        |
        | model refs, prefix hash, experiment flags
        v
GoldenExperience planner
        |
        | ReusePlan: scenario, strategy, confidence, gates
        v
LMCache patch surface
        |
        | secondary lookup -> materialize/transform -> quality accounting
        v
LMCache storage/offload + vLLM inference remain upstream-owned
```

The patch surface is described by `PatchManifest.default()`:

1. `serving_request_metadata`: attach `ModelRef` and prefix metadata before LMCache lookup.
2. `lmcache_cross_model_lookup`: on a same-model miss, query cross-model candidates.
3. `goldenexperience_materializer`: alias/project/translate KV before returning it.
4. `quality_gate_accounting`: record confidence, calibration, and fallback reasons.

## Repository Layout

```text
goldenexperience/
  reuse/             ModelRef, KVShape, ReuseRequest, ReusePlan, scenario planner.
  lmcache_patch/     Patch manifest and sidecar key metadata for LMCache deltas.
  vllm_lmcache_runtime/ Dependency checks and namespaced env helpers for wrappers.
  cache_core/        Legacy in-repo cache block metadata utilities for tests/prototypes.
  tiered_store/      Legacy synthetic tiering prototype; not the product runtime path.
  engine_adapter/    Legacy adapter experiments; vLLM is now the default runtime target.
docs/                Design, experiment matrix, artifact, and paper planning notes.
configs/             Cross-model reuse experiment configuration.
examples/            Minimal planning examples.
scripts/             Optional bootstrap helpers.
tests/               Unit tests for the current framework and legacy utilities.
```

## Quick Start

Create a Python 3.10+ environment and install the local project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
```

Run the planner smoke test:

```bash
python3 scripts/smoke_cross_model_plan.py
```

Build the Qwen2.5 7B <-> 14B GoldenScale calibration scaffold:

```bash
golden-scale-collect --output artifacts/golden_scale/prompts.json
golden-scale-fit \
  --direction bidirectional \
  --prompt-manifest artifacts/golden_scale/prompts.json \
  --output-dir artifacts/golden_scale
golden-scale-validate artifacts/golden_scale/qwen25_7b_to_14b_projection_v0.json
golden-scale-bench artifacts/golden_scale/qwen25_14b_to_7b_projection_v0.json
```

## Deployment Flow

GoldenExperience deploys as a Python package inside the same environment as the serving
runtime. The default same-model disk-reuse baseline is now **vLLM + LMCache MP +
filesystem L2**. `SGLang + LMCache` is retained as a legacy control path.

```text
client -> vLLM OpenAI-compatible server
             |
             | LMCacheMPConnector
             v
        LMCache MP server
             |
             | filesystem L2
             v
        disk KV store
```

Runtime ownership:

- vLLM starts the default inference server and owns request scheduling and generation.
- LMCache MP owns KV lookup, storage, offload, eviction, prefetch, and filesystem L2.
- GoldenExperience owns `ModelRef`, `ReuseRequest`, `ReusePlan`, patch metadata, and
  quality/fallback accounting.
- SGLang remains available through `GE_KV_BACKEND=legacy GE_ENGINE=sglang` for comparison.

### 1. Install Runtime Packages

Use package mode when you only need to run the stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode package
```

Use source mode when you need to patch vLLM or LMCache internals:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode source
```

`--mode source` clones vLLM and LMCache into `third_party/` and installs editable copies.
Override the defaults when using forks:

```bash
GE_THIRD_PARTY_DIR=third_party \
GE_VLLM_REPO_URL=https://github.com/vllm-project/vllm.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
./scripts/install_runtime.sh --mode source
```

Install only GoldenExperience if vLLM and LMCache are already available:

```bash
./scripts/install_runtime.sh --mode golden-only
```

The script prefers `uv pip install` when `uv` is installed; otherwise it falls back to
`python3 -m pip install`. Add `--with-legacy-sglang` only when you need the SGLang
control path. Runtime install details should still be checked against upstream docs when
changing CUDA, Python, or package versions:

- vLLM docs: <https://docs.vllm.ai/>
- SGLang docs: <https://docs.sglang.ai/>
- LMCache docs: <https://docs.lmcache.ai/>

### 2. Verify Planner and Imports

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime
```

Expected planner output includes three rows:

- `model_lora_mutual_reuse`: ready base/LoRA plan.
- `same_model_different_parameter_size`: calibrated GoldenScale projection plan.
- `different_base_model`: conservative unready cross-base plan.

If `--check-runtime` reports missing `vllm` or `lmcache`, install the runtime stack before
starting model-backed serving. Use `--check-legacy-sglang` to include the optional SGLang
control path in the import check.

### 3. Generate Patch Manifest

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

### 4. Send a Request

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

### 5. Run the vLLM + LMCache MP Filesystem-L2 Baseline

Use this baseline after vLLM, LMCache, and GoldenExperience are installed in the same
Python environment. The script starts a standalone LMCache MP server with filesystem L2,
starts vLLM with `LMCacheMPConnector`, sends an offload request, restarts only vLLM, sends
the same prompt again, and records evidence for disk reuse.

```bash
source .venv/bin/activate

GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tensor-parallel-size 1
```

Default outputs are written under `artifacts/kv_baseline/<run_id>/`:

- `metadata.json`: model, prompt, mode, cache path, and runtime settings.
- `runtime.json`: generated command/config state used by the lightweight shell runner.
- `lmc_config.yaml`: recorded LMCache MP and filesystem-L2 configuration.
- `requests/offload.json`: first request output, usage, end-to-end latency, and TTFT.
- `requests/reuse.json`: second request with the same prompt after restart.
- `logs/offload_server.log`, `logs/reuse_server.log`, and LMCache MP log slices.
- `summary.json`: request deltas plus disk/reuse/PID evidence.

When `GE_FORCE_DISK_OFFLOAD=1` and no prompt file is provided, the script writes a long
deterministic prompt into the run directory so repo configs are not modified.

Useful overrides:

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_RUN_ID=qwen3_8b_gsm8k_restart_001 \
GE_REQUIRE_REUSE_EVIDENCE=1 \
GE_FORCE_DISK_OFFLOAD=1 \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tensor-parallel-size 1
```

Interpretation checklist:

1. Confirm the first run finished and `requests/offload.json` contains the expected answer.
2. Confirm `summary.json` has `offload_engine_pid != reuse_engine_pid`.
3. Confirm `summary.json` has a non-null `lmcache_mp_pid`.
4. Confirm `cache.file_count > 0` and `cache.total_bytes > 0`.
5. Confirm `evidence.reuse_has_cache_evidence=true` and `evidence.disk_reuse_success=true`.
6. Treat TTFT deltas as performance context only; they are not sufficient reuse proof.

If your installed LMCache version does not rebuild local-disk metadata across process
restart, run a diagnostic same-process baseline instead:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_BASELINE_MODE=same-process \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tensor-parallel-size 1
```

### 6. Legacy SGLang Control Path

`scripts/start_sglang_lmcache.sh` is retained for legacy experiments. The old baseline
filename is now a wrapper that forwards to the vLLM-named runner. To run the old in-process
SGLang + LMCache control path:

```bash
GE_KV_BACKEND=legacy \
GE_ENGINE=sglang \
GE_MODEL_PATH=/models/Qwen3-8B \
scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tp 1
```

### Current Runtime Limitation

The repository currently contains the GoldenExperience planner, metadata model, patch
manifest, and deployment wrappers. The real LMCache hook implementation is the next step.
Until `lmcache_cross_model_lookup` and `goldenexperience_materializer` are wired into an
LMCache patch or fork, the vLLM + LMCache MP baseline validates same-model disk reuse and
GoldenExperience can validate plans/metadata, but accepted cross-model KV reuse will not yet
be executed inside LMCache.

## GoldenScale Reuse

The first GoldenScale MVP targets bidirectional `Qwen/Qwen2.5-7B-Instruct` and
`Qwen/Qwen2.5-14B-Instruct` reuse. GoldenExperience treats each direction as an independent
artifact because 7B->14B and 14B->7B need different layer maps, projection specs, cost
estimates, and quality gates.

The artifact flow is:

```text
shared prompts
  -> golden-scale-collect
  -> golden-scale-fit
  -> CalibrationManifest JSON per direction
  -> golden-scale-validate
  -> planner READY only when calibration/artifact gates pass
```

The MVP artifact contains:

- `LayerMap`: covers every target layer and maps it to source layer ids.
- `ProjectionSpec`: source/target KV width, KV heads, head dim, method, and projection id.
- `QualityGateResult`: offline/shadow gate metrics such as KV cosine and perplexity drift.
- sidecar ids: `pair_id`, `direction`, `calibration_id`, `layer_map_id`, `projection_id`,
  source/target config hashes, and fallback reason.

Runtime behavior remains conservative:

- Prefix token ids must match exactly; chunk alignment is required.
- The materializer must output full target-shaped KV for every target layer.
- `estimated_materialization_ms` must be <= 70% of target prefill cost.
- Any tokenizer, RoPE, config hash, artifact, layer-map, projection, or quality mismatch
  falls back to the original serving-engine + LMCache target prefill path.

## Minimal Planner Example

```python
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest

shape = KVShape(num_layers=32, num_key_value_heads=8, head_dim=128)
base = ModelRef(
    model_id="qwen2.5-7b",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="qwen2.5",
    parameter_count_b=7,
    kv_shape=shape,
)
lora = ModelRef(
    model_id="qwen2.5-7b-lora-math",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="qwen2.5",
    parameter_count_b=7,
    base_model_id="qwen2.5-7b",
    lora_adapter_id="math-adapter",
    kv_shape=shape,
)

plan = CrossModelReusePlanner().plan(
    ReuseRequest(source=base, target=lora, prefix_hash="shared-system-prompt")
)
print(plan.scenario.value, plan.strategy.value, plan.status.value)
```

Render the LMCache patch contract:

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

## Development Roadmap

- M0: Lock the vLLM + LMCache MP filesystem-L2 same-model disk reuse baseline.
- M1: Implement LMCache secondary lookup sidecar for base/LoRA mutual reuse.
- M2: Add layer/head mapping and calibrated projection for same-model size variants.
- M3: Add experimental learned translator interface for different-base reuse.
- M4: Build vLLM model-backed benchmarks plus SGLang legacy controls.
- M5: Keep the patch small enough to rebase on upstream LMCache.

See `docs/design.md`, `docs/experiment_matrix.md`, and `docs/artifact.md` for the detailed
framework plan.
