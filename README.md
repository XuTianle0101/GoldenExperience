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

## Selective Cached-KV v5

The current research path is a fail-closed v5 artifact for same-family scale variants:

- head-aware transport supports different source/target KV-head counts;
- a source-only sidecar and calibrated MLP admit only prefixes whose simultaneous 95%
  one-sided risk bound, Bonferroni-corrected across candidate thresholds, is at most 1%;
- `validation_candidate` and `semantic_approved` artifacts cannot execute runtime reuse;
- final `approved` artifacts use `RETRIEVE_TRANSFORM` to scatter directly into vLLM paged
  KV without publishing target Mooncake objects.

The implementation contract and present evidence boundary are documented in
`docs/selective_kv_v5.md`. No v5 artifact is currently approved; retained rank-512 Qwen3
results are development failures, not production or paper claims.

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
3. `calibrated_risk_gate`: evaluate the source sidecar before source KV is read.
4. `lmcache_retrieve_transform`: batch-read, transform, and atomically scatter paged KV.
5. `goldenexperience_materializer`: retain the read-compatible v4 materializer path.
6. `quality_gate_accounting`: record confidence, calibration, and fallback reasons.

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

Package mode is fail-closed to the verified CUDA 13 stack (`vllm==0.24.0`,
`lmcache==0.4.6`). It never replaces CuPy behind the resolver. Use source mode for CUDA
12 or another runtime matrix until that stack has its own adapter compatibility tests.

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
- `scripts/run_qwen3_cached_kv_runtime.py`: the quality-gated cached-KV path. It binds
  lookup to the current source request,
  reads complete source Mooncake objects, applies a direction-specific safetensors bridge,
  and atomically publishes target keys only after identity, quality, exact-I/O, and runtime
  cost gates pass.

The retired hidden-state and prefix-specific bridge experiments are summarized in
`docs/paper_outline.md`. Their machine-specific manifests and training scripts were
removed after consolidation: the apparent prefix-specific cosine pass failed its runtime
task assertion and was not a general-purpose 8B -> 14B bridge.
The old two-model prefill materializer is now experiment-only and cannot inject Mooncake
objects. No cached-KV bridge is automatically approved until a global held-out artifact
passes both the accuracy and end-to-end cost gates.
Set `GE_CACHED_KV_DIRECTION=8b_to_14b` or `14b_to_8b`; each direction uses a separate
manifest and the runtime swaps the local model defaults accordingly.
The runtime uses deterministic LMCache `blake3` rolling hashes by default. Set
`GE_SOURCE_PROMPT_FILE`/`GE_SOURCE_PROMPT_ID` and
`GE_TARGET_PROMPT_FILE`/`GE_TARGET_PROMPT_ID` to exercise different requests: only the
longest sequence of exact, complete prefix chunks is eligible for materialization, and
the target computes every divergent or partial-tail token locally.

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
- The production materializer consumes `[2, source_layers, chunk_tokens, kv_width]`
  Mooncake objects directly. It inverse-rotates cached Qwen3 keys, applies a learned
  direction-specific per-channel baseline plus joint low-rank and ridge-regularized SiLU
  KV map, reapplies target RoPE at absolute positions, and emits the target layer layout
  without loading or prefilling either model.
- Measured artifact load, exact source read, transform, and target write time must be <=
  70% of the isolated native target prefill cost before target keys are published.
- Long-lived materializer workers keep verified bridge tensors resident. Every cache hit
  rechecks manifest, bridge, model-directory, config, tokenizer, and shard stat identities;
  any change forces complete content verification before reuse.
- Any tokenizer, RoPE, model/config/weight identity, artifact, prompt binding, object
  layout, exact-I/O, quality, or cost mismatch falls back to the original target prefill.

### Cached-KV Training

The checked-in dataset recipe has 256 train, 64 validation, and 64 sealed-test prompts
with explicit, disjoint IDs, groups, and content hashes. Every split covers four task
categories and 32/128/512/2048-token buckets. Prompts use the model's Qwen3 chat template
with thinking explicitly disabled so native and bridged task assertions share a bounded
decode contract. Regenerate it deterministically, then tune against validation without
opening the test split:

```bash
python3 scripts/generate_qwen3_cached_kv_dataset.py
python3 scripts/train_qwen3_cached_kv_bridge.py \
  --direction 8b_to_14b \
  --dataset configs/qwen3_cached_kv_prompts.json \
  --output artifacts/cached_kv/qwen3_8b_to_14b.json
```

The default fit uses rank 512, ridge 1000, and 2048 supervised positions distributed as
32 positions across 64 training prompts. Override both sampling flags together when
running a larger fit; sparse eight-position prompt sampling is known to regress validation.

Run the same command with `--direction 14b_to_8b` for the reverse artifact. Add
`--finalize` only for the selected hyperparameters; this evaluates the sealed 64-prompt
test split and writes safetensors plus a content-addressed manifest. A separate runtime
cost report with measured Mooncake P95 read-transform-write and native-prefill latency is
required for approval. Without it, even perfect offline metrics remain fail closed.

Use `--emit-validation-candidate` to write unapproved safetensors for a non-publishing
runtime cost benchmark. Production loading still rejects this artifact; only the explicit
benchmark loader accepts its fully content-addressed structure without granting approval.

Measure a candidate with real Mooncake source objects and an independently recorded native
target-prefill report. The benchmark writes only unique temporary target keys, verifies
their exact remote sizes, rolls every key back, and never publishes an external index:

```bash
python3 scripts/benchmark_qwen3_cached_kv_cost.py \
  --candidate-manifest artifacts/cached_kv/qwen3_8b_to_14b.candidate.json \
  --source-model /workspace/volume/softdata/models/Qwen3-8B \
  --target-model /workspace/volume/softdata/models/Qwen3-14B \
  --mooncake-setup /path/to/mooncake_setup.json \
  --source-key source-key-from-current-prompt \
  --chunk-size 256 \
  --native-prefill-report /path/to/native_prefill.json \
  --output artifacts/cached_kv/qwen3_8b_to_14b.cost.json
```

Finalization recomputes the report P95 values and accepts them only when the report binds
the exact bridge weights, candidate manifest, direction, validation split, source/target
model weights, real Mooncake backend, and a 20-sample native vLLM prefill report. Both
report SHA values become part of the content-addressed final manifest.

Run a resident materializer worker with one JSON request and compact JSON response per
line. Send `mode=preload_cached_kv_bridge` first to load an already approved artifact
without touching Mooncake, then send normal `mode=cached_kv` requests:

```bash
python3 -m goldenexperience.runtime.cross_model_materializer --serve-jsonl
```

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
