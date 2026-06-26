# GoldenExperience

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

The research/development target is three cross-model reuse cases:

| Scenario | Goal | Default Strategy | Safety Gate |
| --- | --- | --- | --- |
| Base model <-> LoRA model | Reuse KV between a model and its LoRA fine-tuned variant | Adapter-delta gated aliasing | Same base, tokenizer, KV layout, LoRA drift probe |
| Same model, different parameter sizes | Reuse KV across variants such as 7B <-> 14B | Direct alias if shapes match; otherwise layerwise projection | Layer/head mapping and projection calibration |
| Different base models | Explore broader cross-base reuse | Learned translator | Explicit calibration set, tokenizer bridge, task allowlist |

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

Install the project first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
pytest
```

Install SGLang and LMCache from upstream or local clones before running model-backed work.
For a convenience source-install flow:

```bash
./scripts/bootstrap_runtime.sh
```

The helper clones into `third_party/` by default and then runs editable installs. Override
URLs or the target directory with `GE_SGLANG_REPO_URL`, `GE_LMCACHE_REPO_URL`, and
`GE_THIRD_PARTY_DIR` when using forks.

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
