# GoldenExperience Design

## Goal

GoldenExperience focuses on **KV Cache reuse across models** while delegating serving and
cache mechanics to existing projects:

- SGLang is the inference engine.
- LMCache is the KV Cache storage/offload layer.
- GoldenExperience is a small LMCache patch plus Python control-plane library for deciding
  when cross-model reuse is safe enough to try.

The core invariant is simple: GoldenExperience must not change model decoding behavior or
replace LMCache offload. It may only enrich metadata, perform secondary cross-model lookup,
materialize compatible KV, and record quality/fallback evidence.

## Runtime Boundary

```text
SGLang owns: request scheduling, model execution, attention kernels, decoding
LMCache owns: cache storage, lookup, offload, eviction, prefetch
GoldenExperience owns: model identity, reuse planning, transform metadata, quality gates
```

When a request arrives, SGLang should pass enough model and prefix metadata for the LMCache
patch to build a `ReuseRequest`. The `CrossModelReusePlanner` returns a `ReusePlan`. If the
plan is `ready`, LMCache may run a secondary lookup and materialize source KV for the target
model. Otherwise, the original LMCache miss path proceeds unchanged.

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

This covers pairs such as 7B and 14B variants from the same family/architecture. If KV
layout is identical, direct aliasing can be tested. If layer count, KV heads, or head dim
differs, the plan becomes `layerwise_projection` and must wait for calibration. The main
development work is layer mapping, head mapping, projection materialization, and partial
reuse policy.

### 3. Different Base Models

This is explicitly experimental. The default plan is not executable unless the caller opts
in with `allow_cross_base=True` and provides a `calibration_id`. The expected strategy is a
learned translator plus tokenizer bridge and task allowlist. Fallback-to-recompute must be
the default for every uncalibrated or low-confidence case.

## LMCache Patch Surface

`PatchManifest.default()` defines four narrow hooks:

1. `sglang_request_metadata`: attach source/target model identity and prefix metadata.
2. `lmcache_cross_model_lookup`: query cross-model candidates after normal lookup fails.
3. `goldenexperience_materializer`: alias, project, or translate KV for the target model.
4. `quality_gate_accounting`: record confidence, calibration provenance, and fallback reason.

The patch must preserve these invariants:

- Do not modify SGLang scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache storage, offload, eviction, or prefetch implementations.
- Always fall back to the original SGLang + LMCache path when a plan is not ready.
- Attach scenario, transform id, confidence, and calibration metadata to every reuse attempt.

## Development Shape

The current repository contains legacy synthetic cache-core and tiered-store utilities.
They remain useful for unit tests and paper-prototype thinking, but the product runtime path
is now `reuse/` + `lmcache_patch/` + `sglang_runtime/`.

Near-term implementation should avoid building a second cache system. Instead:

- Patch LMCache lookup with a secondary cross-model index.
- Keep model-pair rules in `CrossModelReusePlanner`.
- Keep tensor conversion/materialization behind a small interface selected by `ReusePlan`.
- Record every rejected plan so benchmark results distinguish quality fallback from cache miss.
