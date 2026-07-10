# GoldenExperience

[English](README.md) | [中文](README.zh-CN.md)

GoldenExperience is a **cross-model KV Cache reuse framework** for the shared KV serving
substrate built from **vLLM + LMCache MP + Mooncake Store**.

The runtime boundary is deliberately narrow:

- vLLM owns model loading, scheduling, decoding, and inference correctness.
- LMCache MP owns shared KV lookup, offload, eviction, and prefetch orchestration.
- Mooncake Store owns the persistent L2 metadata/storage root used across engine restarts.
- GoldenExperience adds the control plane for **reusing KV Cache across models**.
- If a reuse plan is not safe or not calibrated, the stack falls back to the original
  vLLM + LMCache MP behavior.

## What This Project Focuses On

GoldenExperience is not an inference engine and does not replace cache storage. It is
intended to be carried as a small LMCache MP patch plus Python control-plane library, with
runtime metadata flowing through the vLLM/LMCache MP connector path into lookup and retrieve
logic.

The research/development target is three GoldenExperience reuse tracks:

| Track Name | Scenario | Goal | Default Strategy | Safety Gate |
| --- | --- | --- | --- | --- |
| GoldenLoRA | Base model <-> LoRA model | Reuse KV between a model and its LoRA fine-tuned variant | Adapter-delta gated aliasing | Same base, tokenizer, KV layout, LoRA drift probe |
| GoldenScale | Same model, different parameter sizes | Reuse KV across variants such as 8B <-> 14B | Direct alias if shapes match; otherwise hidden-state bridge | Layer/head mapping and hidden bridge calibration |
| GoldenBridge | Different base models | Explore broader cross-base reuse | Learned translator | Explicit calibration set, tokenizer bridge, task allowlist |

The names map to the implementation scenarios as follows: `GoldenLoRA` targets
`model_lora_mutual_reuse`, `GoldenScale` targets `same_model_different_parameter_size`,
and `GoldenBridge` targets `different_base_model`.

## Architecture

```text
client -> vLLM OpenAI-compatible server
             |
             | LMCacheMPConnector
             v
      standalone LMCache MP server
             |
             | L2 adapter: type=mooncake_store
             v
      Mooncake Store on local TCP + SSD
             |
             | metadata sidecars, secondary lookup, materialization, accounting
             v
      GoldenExperience planner and LMCache patch surface
```

The patch surface is described by `PatchManifest.default()`:

1. `engine_request_metadata`: attach `ModelRef` and prefix metadata before LMCache MP lookup.
2. `lmcache_cross_model_lookup`: on a same-model miss, query cross-model candidates.
3. `goldenexperience_materializer`: alias/project/translate KV before returning it.
4. `quality_gate_accounting`: record confidence, calibration, and fallback reasons.

## Repository Layout

The source tree is organized in the same spirit as C2C: a core Python package, thin script
entrypoints, configs, runnable recipes, docs, examples, tests, and artifacts. C2C's repo
separates `rosetta/` as the package from `script/`, `bash/`, `recipe/`, and `resource/`;
GoldenExperience follows that shape with runtime orchestration under `goldenexperience/`,
thin launchers under `scripts/`, and reproducible run overlays under `recipes/`.

```text
goldenexperience/
  runtime/           vLLM + LMCache MP + Mooncake runtime checks and baseline scheduler.
  reuse/             ModelRef, KVShape, ReuseRequest, ReusePlan, scenario planner.
  lmcache_patch/     Patch manifest and sidecar key metadata for LMCache MP deltas.
  size_variant/      GoldenScale calibration, layer mapping, and projection scaffolds.
  benchmarks/        Synthetic and model-backed benchmark harnesses.
  cache_core/        Legacy in-repo cache block metadata utilities for tests/prototypes.
  tiered_store/      Legacy synthetic tiering prototype; not the product runtime path.
scripts/
  kv_baseline/       Thin shell launchers plus stdlib OpenAI-compatible client/summarizer.
recipes/             Source-able env overlays for reproducible runtime launches.
docs/                Design, shared KV substrate, experiment matrix, artifact, paper notes.
configs/             Runtime env examples and cross-model reuse experiment configuration.
examples/            Minimal planning examples.
tests/               Unit tests for planner, runtime config, and baseline generation.
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

Build the Qwen3 8B <-> 14B GoldenScale calibration scaffold:

```bash
golden-scale-collect --output artifacts/golden_scale/prompts.json
golden-scale-fit \
  --direction bidirectional \
  --prompt-manifest artifacts/golden_scale/prompts.json \
  --output-dir artifacts/golden_scale
golden-scale-validate artifacts/golden_scale/qwen3_8b_to_14b_hidden_bridge_v0.json
golden-scale-bench artifacts/golden_scale/qwen3_14b_to_8b_hidden_bridge_v0.json
```

## Deployment Flow

GoldenExperience deploys as a Python package inside the same environment as vLLM, LMCache,
and Mooncake. It is not a standalone daemon.

Runtime ownership:

- vLLM starts the OpenAI-compatible inference server and owns request scheduling/generation.
- LMCache MP owns shared KV lookup, storage policy, offload, eviction, and prefetch.
- Mooncake Store owns persistent L2 metadata and SSD-backed objects.
- GoldenExperience owns `ModelRef`, `ReuseRequest`, `ReusePlan`, patch metadata, and
  quality/fallback accounting.

### 1. Install Runtime Packages

Use package mode when you only need to run the stack and Mooncake binaries are already
available on `PATH`:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode package
```

Use source mode when you need to patch LMCache or debug vLLM/LMCache/Mooncake internals:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_runtime.sh --mode source
```

`--mode source` clones vLLM, LMCache, and Mooncake into `third_party/`. It installs editable
vLLM and LMCache copies with `BUILD_MOONCAKE=1` by default; build Mooncake according to
upstream instructions and ensure `mooncake_master` plus `mooncake_http_metadata_server` are
on `PATH`. Override defaults when using forks:

```bash
GE_THIRD_PARTY_DIR=third_party \
GE_VLLM_REPO_URL=https://github.com/vllm-project/vllm.git \
GE_LMCACHE_REPO_URL=https://github.com/LMCache/LMCache.git \
GE_MOONCAKE_REPO_URL=https://github.com/kvcache-ai/Mooncake.git \
./scripts/install_runtime.sh --mode source
```

Install only GoldenExperience if the runtime stack is already available:

```bash
./scripts/install_runtime.sh --mode golden-only
```

The script prefers `uv pip install` when `uv` is installed; otherwise it falls back to
`python3 -m pip install`. After installing dependencies it runs
`scripts/patch_lmcache_mooncake_runtime.py` by default. That reproducibility patch creates
the Mooncake `libmooncake_store.so` alias expected by LMCache, selects the Python
`MooncakeDistributedStore` SET/GET adapter by default, and bypasses the native
`batchIsExist` lookup crash path with an LMCache MP in-process key index. Set
`GE_PATCH_MOONCAKE_RUNTIME=0` only if you intentionally want the unpatched upstream path.
It runs a strict `vLLM`/`LMCache`/`Mooncake` runtime check for package and golden-only
modes; source mode defaults to a warning because Mooncake still needs its upstream build
step. Use `--runtime-check strict|warn|skip` to override this. Runtime install details
should still be checked against upstream vLLM, LMCache, and Mooncake docs when changing
CUDA, Python, or package versions.

### 2. Verify Planner and Runtime

```bash
python3 scripts/smoke_cross_model_plan.py --check-runtime --strict-runtime
python3 scripts/patch_lmcache_mooncake_runtime.py --check
```

Expected planner output includes three rows:

- `model_lora_mutual_reuse`: ready base/LoRA plan.
- `same_model_different_parameter_size`: calibrated GoldenScale projection plan.
- `different_base_model`: conservative unready cross-base plan.

The runtime check reports `vLLM`, `LMCache`, and `Mooncake`. If any are missing, install the
runtime stack before starting model-backed serving.

### 3. Generate Patch Manifest

```bash
golden-patch-manifest --output docs/patch_manifest.md
```

### 4. Run the Shared KV Baseline

The recommended launch path is the engineered Python scheduler exposed by the thin shell
wrapper:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

You can also call the console entry point directly after installation:

```bash
GE_MODEL_PATH=Qwen/Qwen3-8B golden-kv-baseline -- --tensor-parallel-size 1
```

### 5. Run the Same-Model KV Offload/Reuse Baseline

Use this baseline after vLLM, LMCache, Mooncake, and GoldenExperience are installed in the
same Python environment. The default path is now `vLLM + LMCache MP + Mooncake Store`:

1. start Mooncake master plus HTTP metadata service on local TCP,
2. start a standalone LMCache MP server with `type=mooncake_store` L2,
3. start vLLM with `LMCacheMPConnector`,
4. send an offload request, restart only vLLM, then send the same reuse request.

This keeps the shared KV substrate outside the inference process. Cross-model work can later
plug into LMCache MP persistent L2 instead of depending on engine-local caches.

```bash
source .venv/bin/activate
source recipes/kv_baseline_mooncake_local.env

GE_MODEL_PATH=Qwen/Qwen3-8B \
GE_PORT=30000 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

Default local TCP + SSD settings:

- `GE_KV_BACKEND=mp`, `GE_ENGINE=vllm`, `GE_LMCACHE_MP_L2_ADAPTER_TYPE=mooncake_store`.
- `GE_MOONCAKE_MASTER_HOST=127.0.0.1`, `GE_MOONCAKE_MASTER_PORT=50051`.
- `GE_MOONCAKE_METADATA_PORT=8080`, producing `http://127.0.0.1:8080/metadata`.
- `GE_LMCACHE_MP_HTTP_PORT=8081`, so LMCache MP HTTP does not collide with Mooncake metadata.
- `GE_MOONCAKE_PROTOCOL=tcp`, `GE_MOONCAKE_STORAGE_ROOT=$GE_KV_CACHE_DIR/mooncake`.
- `GE_MOONCAKE_PER_OP_WORKERS_JSON='{"lookup":2,"retrieve":8,"store":4}'`.
- `LMCACHE_MOONCAKE_PYTHON_ADAPTER=1`, using Python Mooncake Store SET/GET while keeping
  `LMCACHE_MOONCAKE_NATIVE_EXISTS=0` to avoid native `batchIsExist`.
- Mooncake master defaults include `--client_ttl=600`, `--root_fs_dir` from the storage
  root, and `--cluster_id` from `GE_RUN_ID`; override with `GE_MOONCAKE_MASTER_EXTRA_ARGS`
  when needed.

Default outputs are written under `artifacts/kv_baseline/<run_id>/`:

- `metadata.json`: model, prompt, MP connector, Mooncake endpoints, adapter JSON, pids, and logs.
- `lmc_config.yaml`: generated run config with `LMCacheMPConnector` and Mooncake Store L2.
- `requests/offload.json`: first request output, usage, end-to-end latency, and TTFT.
- `requests/reuse.json`: second request with the same prompt after vLLM restart.
- `logs/lmcache_mp_server.log`: persistent LMCache MP server evidence.
- `logs/mooncake_master.log` and `logs/mooncake_metadata_server.log`: Mooncake service evidence.
- `metrics/offload.prom` and `metrics/reuse.prom`: vLLM external KV transfer counters.
- `summary.json`: request deltas plus store/retrieve/L2/Mooncake counters.

Raw KV baseline run directories are ignored by Git. Keep only curated manifests under
`artifacts/kv_baseline/manifests/` and store large KV seed payloads in an external artifact
store:

```bash
python3 scripts/kv_baseline/export_kv_seed_manifest.py \
  artifacts/kv_baseline/<run_id> \
  --artifact-uri s3://bucket/ge-kv-seeds/<artifact_id>.tar.zst \
  --output artifacts/kv_baseline/manifests/<run_id>.json
```

The script generates a long deterministic disk-offload prompt by default when
`GE_FORCE_DISK_OFFLOAD=1`. Set `GE_PROMPT_FILE` and `GE_PROMPT_ID` to use your own prompt
manifest, or tune `GE_DISK_PROMPT_REPEAT` if the generated prompt exceeds the model context.

Useful overrides:

```bash
GE_MODEL_PATH=/models/Qwen3-8B \
GE_MODEL_NAME=/models/Qwen3-8B \
GE_RUN_ID=qwen3_8b_mooncake_restart_001 \
GE_MOONCAKE_STORAGE_ROOT=/ssd/ge-kv/mooncake \
GE_MOONCAKE_GLOBAL_SEGMENT_SIZE=4294967296 \
GE_MOONCAKE_LOCAL_BUFFER_SIZE=4294967296 \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

Interpretation checklist:

1. Confirm `requests/offload.json` contains the expected answer.
2. Confirm `summary.json` reports `offload_has_disk_evidence=true`,
   `reuse_has_cache_evidence=true`, and `disk_reuse_success=true`.
3. Confirm `metrics/reuse.prom` has `vllm:external_prefix_cache_hits_total > 0` and
   `vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} > 0`, while the
   offload phase is dominated by `source="local_compute"`.
4. Confirm `logs/lmcache_mp_server.log` includes Python Mooncake Store evidence:
   `MooncakeStore SET`, `MooncakeStore EXISTS`, `MooncakeStore GET`, and
   `L2 prefetch load completed`.
5. Confirm `metadata.json` records `mooncake.enabled=true`, `l2_adapter_type=mooncake_store`,
   the Mooncake storage root, and distinct offload/reuse vLLM service pids.
6. Confirm the Mooncake storage root contains non-empty files, then keep the whole
   `artifacts/kv_baseline/<run_id>/` directory as the same-model baseline for
   later cross-model KV reuse experiments.

Compatibility and diagnostics:

```bash
# Use the older MP filesystem L2 adapter instead of Mooncake Store.
GE_LMCACHE_MP_L2_ADAPTER_TYPE=fs \
scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh -- --tensor-parallel-size 1
```

Set `GE_KEEP_LMCACHE_MP_AFTER_RUN=1` or `GE_KEEP_MOONCAKE_AFTER_RUN=1` when you want to
inspect live services after a run.

The default Mooncake baseline intentionally uses the Python Mooncake Store SET/GET path
because the native C++ `batchIsExist` path has shown compatibility crashes on missing keys
in the package stack used for the local baseline. To test the native path explicitly, set
`LMCACHE_MOONCAKE_PYTHON_ADAPTER=0` and `LMCACHE_MOONCAKE_NATIVE_EXISTS=1`.

### Current Runtime Status

The runtime now has two cross-size paths:

- `scripts/run_cross_model_runtime.py`: the earlier `native_target_seed` proof. It
  creates target-shaped KV with a target-model prefill, restarts target vLLM, and
  verifies LMCache/Mooncake external KV retrieval.
- `scripts/run_cross_model_hidden_bridge_runtime.py`: the quality-gated hidden-bridge
  path. It runs source offload, performs `lmcache_cross_model_lookup` against source
  Mooncake keys after a target miss, calls `goldenexperience_materializer`, writes
  target-shaped chunks to Mooncake plus a persistent `GE_MOONCAKE_EXTERNAL_INDEX`, and
  lets a fresh target vLLM consume the injected keys only if quality gates pass.

The general Qwen3-8B -> Qwen3-14B low-rank bridge artifact still does **not** pass the
quality gate and correctly falls back. The historical prefix-specific artifact at
`artifacts/cross_model_runtime/manifests/prefix_specific_strict_20260709T0253Z.json`
proves retrieval only: it predates the isolated native-target phase and its recorded task
assertion is false, so it does not satisfy the current strict semantic-success gate.
The comparison against a same-model Qwen3-14B offload -> restart -> reuse baseline is
`artifacts/cross_model_runtime/manifests/prefix_specific_strict_20260709T0253Z_vs_qwen3_14b_same_model_restart_20260709T0223Z.json`.
This remains historical retrieval evidence, not a general-purpose 8B -> 14B bridge.

## GoldenScale Reuse

The first GoldenScale MVP targets bidirectional `Qwen/Qwen3-8B` and
`Qwen/Qwen3-14B` reuse. GoldenExperience treats each direction as an independent
artifact because 8B->14B and 14B->8B need different layer maps, hidden bridge specs, target KV restore specs, cost
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
- `HiddenBridgeSpec`: maps `pre_kv_hidden` from small-model width to large-model width.
- `KVRestoreSpec`: records target-model W_K/W_V/RoPE restore contract and GQA KV layout.
- `ProjectionSpec`: retained as a legacy KV-projection baseline/control artifact.
- `QualityGateResult`: offline/shadow metrics such as hidden cosine, KV cosine, attention proxy cosine, and perplexity drift.
- sidecar ids: `pair_id`, `direction`, `calibration_id`, `layer_map_id`, `hidden_bridge_id`, `restore_id`,
  source/target config hashes, state kind, and fallback reason.

Runtime behavior remains conservative:

- Prefix token ids must match exactly; chunk alignment is required.
- The materializer bridges `h_small -> h_large_hat`, then target-model W_K/W_V/RoPE restores full target-shaped KV.
- `estimated_materialization_ms` must be <= 70% of target prefill cost.
- Any tokenizer, RoPE, config hash, artifact, layer-map, hidden-bridge, restore, or quality mismatch
  falls back to the original vLLM + LMCache MP target prefill path.

## Minimal Planner Example

```python
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest

shape = KVShape(num_layers=36, num_key_value_heads=8, head_dim=128)
base = ModelRef(
    model_id="qwen3-8b",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="qwen3",
    parameter_count_b=8,
    kv_shape=shape,
)
lora = ModelRef(
    model_id="qwen3-8b-lora-math",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="qwen3",
    parameter_count_b=8,
    base_model_id="qwen3-8b",
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

- M0: Lock the project boundary around vLLM + LMCache MP + Mooncake Store + GoldenExperience metadata.
- M1: Implement LMCache secondary lookup sidecar for base/LoRA mutual reuse.
- M2: Add layer/head mapping and calibrated projection for same-model size variants.
- M3: Add experimental learned translator interface for different-base reuse.
- M4: Build vLLM model-backed benchmarks and quality/fallback accounting.
- M5: Keep the patch small enough to rebase on upstream LMCache.

See `docs/design.md`, `docs/experiment_matrix.md`, and `docs/artifact.md` for the detailed
framework plan.
