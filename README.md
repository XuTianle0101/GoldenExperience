# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience is a **cross-model KV Cache reuse patch framework** for an open-source
serving stack built from **SGLang** + **LMCache**.

The new boundary is deliberately narrow:

- SGLang owns model loading, scheduling, decoding, and inference correctness.
- LMCache owns KV storage, lookup, offload, eviction, and prefetch mechanics.
- GoldenExperience adds the control plane for **reusing KV Cache across models**.
- If a reuse plan is not safe or not calibrated, the stack falls back to the original
  SGLang + LMCache behavior.

## What This Project Focuses On

GoldenExperience is no longer trying to be an inference engine or a KV offload system.
It is intended to be carried as a small patch on top of LMCache, with runtime metadata
flowing from SGLang requests into LMCache lookup and retrieve paths.

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
SGLang request/session
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
LMCache storage/offload + SGLang inference remain upstream-owned
```

The patch surface is described by `PatchManifest.default()`:

1. `sglang_request_metadata`: attach `ModelRef` and prefix metadata before LMCache lookup.
2. `lmcache_cross_model_lookup`: on a same-model miss, query cross-model candidates.
3. `goldenexperience_materializer`: alias/project/translate KV before returning it.
4. `quality_gate_accounting`: record confidence, calibration, and fallback reasons.

## Repository Layout

```text
goldenexperience/
  reuse/             ModelRef, KVShape, ReuseRequest, ReusePlan, scenario planner.
  lmcache_patch/     Patch manifest and sidecar key metadata for LMCache deltas.
  sglang_runtime/    Dependency checks and namespaced env helpers for wrappers.
  cache_core/        Legacy in-repo cache block metadata utilities for tests/prototypes.
  tiered_store/      Legacy synthetic tiering prototype; not the product runtime path.
  engine_adapter/    Legacy adapter experiments; SGLang is now the runtime target.
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

GoldenExperience deploys as a Python package inside the same environment as SGLang and
LMCache. It is not a standalone daemon.

```text
client -> SGLang OpenAI-compatible server
             |
             | --enable-lmcache
             v
          LMCache
             |
             | GoldenExperience patch hooks and planner metadata
             v
          fallback or cross-model KV reuse
```

Runtime ownership:

- SGLang starts the inference server and owns request scheduling and generation.
- LMCache owns KV lookup, storage, offload, eviction, and prefetch.
- GoldenExperience owns `ModelRef`, `ReuseRequest`, `ReusePlan`, patch metadata, and
  quality/fallback accounting.

### 1. Install Runtime Packages

Use package mode when you only need to run the stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode package
```

Use source mode when you need to patch LMCache or debug SGLang/LMCache internals:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode source
```

`--mode source` clones SGLang and LMCache into `third_party/` and installs editable copies.
Override the defaults when using forks:

```bash
GE_THIRD_PARTY_DIR=third_party \
GE_SGLANG_REPO_URL=https://github.com/sgl-project/sglang.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
./scripts/install_runtime.sh --mode source
```

Install only GoldenExperience if SGLang and LMCache are already available:

```bash
./scripts/install_runtime.sh --mode golden-only
```

The script prefers `uv pip install` when `uv` is installed; otherwise it falls back to
`python3 -m pip install`. SGLang and LMCache install details should still be checked
against their upstream docs when changing CUDA, Python, or package versions:

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

If `--check-runtime` reports missing `sglang` or `lmcache`, install the runtime stack before
starting model-backed serving.

### 3. Generate Patch Manifest

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

The manifest is also generated automatically by `scripts/start_sglang_lmcache.sh`.

### 4. Start SGLang With LMCache

The default launch command starts a SGLang OpenAI-compatible server with LMCache enabled:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
./scripts/start_sglang_lmcache.sh
```

Common overrides:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_HOST=0.0.0.0 \
GE_PORT=30000 \
GE_LMCACHE_CONFIG_FILE=artifacts/runtime/lmc_config.yaml \
GE_LMCACHE_CHUNK_SIZE=256 \
GE_LMCACHE_LOCAL_CPU_GB=10 \
./scripts/start_sglang_lmcache.sh --tp 1
```

The start script does the following before `exec python3 -m sglang.launch_server`:

1. Writes an LMCache config if `GE_LMCACHE_CONFIG_FILE` does not exist.
2. Renders `docs/patch_manifest.md`.
3. Checks that `sglang`, `lmcache`, and `goldenexperience` are importable.
4. Exports GoldenExperience metadata variables:
   - `GE_ENABLE_CROSS_MODEL_REUSE=1`
   - `GE_PATCH_MANIFEST=docs/patch_manifest.md`
   - `GE_LMCACHE_CONFIG=configs/lmcache.example.yaml`
   - `GE_SGLANG_MODEL_ID=$GE_MODEL_PATH`
5. Starts SGLang with `--enable-lmcache`.

The generated LMCache config is intentionally small:

```yaml
chunk_size: 256
local_cpu: true
use_layerwise: true
max_local_cpu_size: 10
```

Set `GE_OVERWRITE_LMCACHE_CONFIG=1` to regenerate an existing config.

### 5. Send a Request

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

### 6. Run the First KV Offload/Reuse Baseline

Use this baseline after SGLang, LMCache, and GoldenExperience are installed in the same
Python environment. The script starts one SGLang + LMCache server, sends a deterministic
GSM8K-style prompt to populate/offload KV, restarts SGLang with the same LMCache disk
directory, sends the same prompt again, and records timing/log evidence for reuse.

```bash
source .venv/bin/activate

GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

Default outputs are written under `artifacts/kv_baseline/<run_id>/`:

- `metadata.json`: model, prompt, mode, cache path, and runtime settings.
- `lmc_config.yaml`: generated LMCache config with local CPU plus persistent local disk.
- `requests/offload.json`: first request output, usage, end-to-end latency, and TTFT.
- `requests/reuse.json`: second request with the same prompt after restart.
- `logs/offload_server.log` and `logs/reuse_server.log`: SGLang/LMCache evidence.
- `summary.json`: request deltas and best-effort log counters for store/retrieve events.

The default prompt lives in `configs/kv_baseline_prompts.json` and uses the classic GSM8K
Natalia clips question. The default `GE_KV_CHUNK_SIZE=16` is intentionally small so this
short prompt crosses at least one LMCache chunk. For longer workloads, raise it back toward
the production-style value used in your LMCache experiments.

Useful overrides:

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_RUN_ID=qwen3_8b_gsm8k_restart_001 \
GE_KV_LOCAL_CPU_GB=20 \
GE_KV_LOCAL_DISK_GB=200 \
GE_PROMPT_ID=gsm8k_natalia_clips \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

Interpretation checklist:

1. Confirm the first run finished and `requests/offload.json` contains the expected answer.
2. Confirm `logs/offload_server.log` contains LMCache store/offload messages.
3. Confirm the second run starts from a fresh process and `logs/reuse_server.log` contains
   LMCache lookup/retrieve/hit evidence.
4. Compare `summary.json` fields `reuse_minus_offload_ttft_ms` and
   `reuse_minus_offload_e2e_ms`; negative values indicate the second request was faster.
5. Keep the whole `artifacts/kv_baseline/<run_id>/` directory as the same-model baseline for
   later cross-model KV reuse experiments.

If your installed LMCache version does not rebuild local-disk metadata across process
restart, run a diagnostic same-process baseline instead:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_BASELINE_MODE=same-process \
scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1
```

`GE_DISABLE_RADIX_CACHE=1` is enabled by default so the baseline is not hidden by SGLang's
in-process radix cache. Set `GE_DISABLE_RADIX_CACHE=0` only when you want to measure the
combined SGLang radix cache plus LMCache behavior.

### Current Runtime Limitation

The repository currently contains the GoldenExperience planner, metadata model, patch
manifest, and deployment wrappers. The real LMCache hook implementation is the next step.
Until `lmcache_cross_model_lookup` and `goldenexperience_materializer` are wired into an
LMCache patch or fork, SGLang + LMCache will run normally and GoldenExperience can validate
plans/metadata, but accepted cross-model KV reuse will not yet be executed inside LMCache.

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
  falls back to the original SGLang + LMCache target prefill path.

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

- M0: Lock the project boundary around SGLang + LMCache + GoldenExperience patch metadata.
- M1: Implement LMCache secondary lookup sidecar for base/LoRA mutual reuse.
- M2: Add layer/head mapping and calibrated projection for same-model size variants.
- M3: Add experimental learned translator interface for different-base reuse.
- M4: Build SGLang model-backed benchmarks and quality/fallback accounting.
- M5: Keep the patch small enough to rebase on upstream LMCache.

See `docs/design.md`, `docs/experiment_matrix.md`, and `docs/artifact.md` for the detailed
framework plan.
