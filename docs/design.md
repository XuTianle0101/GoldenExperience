# GoldenExperience Design

## Goal

GoldenExperience focuses on **KV Cache reuse across models** while delegating serving and
cache mechanics to existing projects:

- vLLM is the default inference engine.
- LMCache MP is the shared KV Cache service.
- Mooncake Store is the persistent L2 storage substrate.
- GoldenExperience is a small LMCache patch plus Python control-plane library for deciding
  when cross-model reuse is safe enough to try.

The core invariant is simple: GoldenExperience must not change model decoding behavior or
replace LMCache offload. It may only enrich metadata, perform secondary cross-model lookup,
materialize compatible KV, and record quality/fallback evidence.

## Runtime Boundary

```text
vLLM owns: request scheduling, model execution, attention kernels, decoding
LMCache MP owns: shared KV lookup, offload, eviction, prefetch
Mooncake Store owns: persistent L2 metadata and SSD-backed object storage
GoldenExperience owns: model identity, reuse planning, transform metadata, quality gates
```

For same-model persistence evidence, the stable shared KV substrate is now
`vLLM + LMCache MP + Mooncake Store`: vLLM is only the inference process, LMCache MP is the
long-lived cross-instance KV service, and Mooncake Store is the inspectable L2. Cross-model
lookup and materialization should be designed around LMCache MP/L2 metadata rather than
engine-local caches. See `docs/shared_kv_substrate.md`.

When a request arrives, the vLLM/LMCache MP connector path should carry enough model and
prefix metadata for the LMCache patch to build a `ReuseRequest`. The
`CrossModelReusePlanner` returns a `ReusePlan`. If the plan is `ready`, LMCache MP may run a
secondary lookup and materialize source KV for the target model. Otherwise, the original
LMCache MP miss path proceeds unchanged.

## Core Abstractions

`KVShape` captures the minimum compatibility surface: layer count, KV heads, head dim,
dtype, RoPE theta, and optional sliding window.

`ModelRef` identifies a base model, a LoRA adapter, or a model-size variant. It records
family, architecture, tokenizer, parameter count, base model id, LoRA adapter id, and KV
shape.

`ReuseRequest` connects a source model, target model, prefix hash, optional calibration id,
and experimental flags.

`ReusePlan` records the selected scenario, strategy, status, confidence, required gates,
transform id, and patch hooks. Plans are metadata-first so they can be attached to LMCache
sidecar records without changing the underlying storage backend.

## Three Reuse Scenarios

### 1. Base Model and LoRA Variant

This is the first implementation target. The source and target must share a base model,
tokenizer, and KV layout. The default strategy is `adapter_delta_gated_alias`: reuse is
allowed only after a cheap LoRA drift or probe-logit gate passes. This path should be a
small LMCache key/metadata patch plus quality accounting.

### 2. Same Model Line, Different Parameter Sizes

This covers pairs such as 8B and 14B variants from the same family/architecture. If KV
layout is identical, direct aliasing can be tested. If layer count, KV heads, or head dim
differs, the plan becomes `layerwise_projection` and must wait for calibration. The main
development work is layer mapping, head mapping, projection materialization, and partial
reuse policy.

The implemented MVP makes this scenario artifact-driven:

- `CalibrationManifest` binds one source model, one target model, one direction, a layer
  map, a projection spec, and quality-gate results.
- `LayerMap` must cover every target layer. The default builder uses linear depth
  interpolation; later calibration can replace the score with CKA/SVCCA alignment.
- `ProjectionSpec` records source/target KV width. The current deterministic materializer
  uses identity-pad-truncate projection as a scaffold for learned per-layer projections.
- Planner execution requires a `calibration_id`; when an `artifact_uri` is supplied, model
  ids, tokenizer/shape contract, config hashes, layer coverage, projection shape, and
  quality gates are validated before returning `ready`.
- Runtime cost gating rejects reuse when estimated materialization cost exceeds 70% of the
  target native prefill cost.

### 3. Different Base Models

This is explicitly experimental. The default plan is not executable unless the caller opts
in with `allow_cross_base=True` and provides a `calibration_id`. The expected strategy is a
learned translator plus tokenizer bridge and task allowlist. Fallback-to-recompute must be
the default for every uncalibrated or low-confidence case.

## LMCache Patch Surface

`PatchManifest.default()` defines four narrow hooks:

1. `engine_request_metadata`: attach source/target model identity and prefix metadata.
2. `lmcache_cross_model_lookup`: query cross-model candidates after normal lookup fails.
3. `goldenexperience_materializer`: alias, project, or translate KV for the target model.
4. `quality_gate_accounting`: record confidence, calibration provenance, and fallback reason.

The patch must preserve these invariants:

- Do not modify vLLM scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache MP storage, offload, eviction, or prefetch implementations.
- Do not replace Mooncake Store; only configure and observe it as persistent L2.
- Always fall back to the original vLLM + LMCache MP path when a plan is not ready.
- Attach scenario, transform id, confidence, and calibration metadata to every reuse attempt.

## Development Shape

The current repository contains legacy synthetic cache-core and tiered-store utilities.
They remain useful for unit tests and paper-prototype thinking, but the product runtime path
is now `reuse/` + `lmcache_patch/` + `runtime/`.

Near-term implementation should avoid building a second cache system. Instead:

- Patch LMCache lookup with a secondary cross-model index.
- Keep model-pair rules in `CrossModelReusePlanner`.
- Keep tensor conversion/materialization behind a small interface selected by `ReusePlan`.
- Record every rejected plan so benchmark results distinguish quality fallback from cache miss.
