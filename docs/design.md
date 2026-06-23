# GoldenExperience Design

## Goal

GoldenExperience turns KV cache into an engine-decoupled serving resource. Instead of
keeping KV cache private to a single model instance or inference engine, it exposes:

1. A uniform cache block abstraction.
2. A tiered placement layer across HBM, CPU memory, and NVMe.
3. Same-family cross-model mapping with quality-gated reuse.
4. Thin adapters for engines such as Hugging Face Transformers and vLLM.

## Core Abstractions

`CacheBlock` stores a `KVPayload` plus `CacheBlockMetadata`. Metadata contains model id,
layer id, head id, token range, dtype, tier, shape, checksum, quality score, prefix hash,
session id, reference count, and source mapper fields. The payload remains opaque so the
same store can handle PyTorch tensors, NumPy arrays, or list-backed synthetic payloads.

`TieredKVStore` owns metadata indexing and tier placement. The public methods are:

- `put(block)`
- `get(query, promote_to=None)`
- `get_many(query, limit=None)`
- `get_by_id(block_id, promote_to=None)`
- `offload(block_id, target_tier)`
- `prefetch(plan)`
- `evict(query=None, required_bytes=0)`
- `pin(block_id)` and `release(block_id)`

## Tiered Offload

The default demotion path is:

```text
HBM -> CPU -> NVMe -> remove
```

HBM and CPU are represented by in-process memory backends. The NVMe tier is a pickle-backed
directory so artifact runs can work without a custom C++ runtime. When PyTorch tensors are
present, the utility layer performs best-effort movement to CUDA or CPU pinned memory.

The initial policy is LRU with pinned/ref-count protection and low-quality victim
preference. This is intentionally simple: it gives a clean baseline before adding
prefix-hotness and decode-progress prediction.

## Cross-Model Reuse

`ArchitectureSignature` records the minimal compatibility surface:

- family and architecture
- layer count
- hidden size
- attention heads and KV heads
- head dimension
- RoPE, tokenizer, dtype, and optional metadata

v1 supports three compatibility classes:

- `exact`: direct reuse.
- `shape_compatible`: direct reuse with lower confidence.
- `shape_mismatch`: final-dimension projection for same-family models.

`ReusePolicy` combines compatibility, mapper confidence, prefix similarity, and expected
latency savings. Every failed gate returns fallback recompute or warm-start recompute,
which keeps quality risk explicit.

## Engine Adapters

Adapters are deliberately thin. They convert engine-owned KV state into `CacheBlock`
objects and convert blocks back into engine-specific formats.

- `TransformersAdapter` handles Hugging Face `past_key_values`.
- `VLLMAdapter` exposes extractor/injector callables because vLLM cache internals vary by
  version.
- `MockModelAdapter` supports deterministic tests and synthetic benchmarks.

## Extension Points

- Add a learned projection mapper that fits from paired calibration traces.
- Add prefix-aware prefetch scheduling from request queues and decode progress.
- Replace NVMe pickle backend with memory-mapped tensors or an async I/O backend.
- Add C++/CUDA fast paths without changing the Python public APIs.

