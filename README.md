# GoldenExperience

GoldenExperience is a research infrastructure repository for engine-decoupled KV cache
reuse in LLM serving. The v1 system focuses on:

- Tiered KV cache offload across HBM, CPU memory, and NVMe.
- Cross-model KV cache reuse for same-family LLMs such as Llama or Qwen variants.
- Thin adapters for inference engines instead of engine-owned cache logic.
- Reproducible synthetic and model-backed benchmarks for systems papers.

The first paper target is an MLSys/OSDI-style systems contribution. The main success
metrics are TTFT, throughput, memory pressure, and tail latency, with answer quality used
as a gate for reuse decisions.

## Repository Layout

```text
goldenexperience/
  cache_core/          CacheBlock metadata, indexing, and store-facing APIs.
  tiered_store/        HBM/CPU/NVMe storage backends, offload, prefetch, eviction.
  cross_model_mapper/  Same-family KV mapping, confidence scoring, reuse policy.
  engine_adapter/      Engine-neutral adapter interface plus HF/vLLM thin adapters.
  benchmarks/          Synthetic benchmark harness and metrics.
configs/               Example experiment configuration.
docs/                  Design, paper, and artifact notes.
examples/              Minimal usage examples.
tests/                 Unit tests for core behavior.
```

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
golden-synthetic-benchmark --blocks 64 --tokens-per-block 128
```

The core package has no mandatory PyTorch dependency. PyTorch is optional and needed for
real model integration through Hugging Face or vLLM.

## Minimal Example

```python
from goldenexperience.cache_core import CacheBlock, CacheBlockMetadata, DeviceTier, KVPayload
from goldenexperience.tiered_store import TieredKVStore

store = TieredKVStore(
    capacities={DeviceTier.HBM: 64 * 1024 * 1024, DeviceTier.CPU: 512 * 1024 * 1024},
    nvme_path="artifacts/cache/nvme",
)

payload = KVPayload(key=[[[1.0, 0.0]]], value=[[[0.5, 0.5]]])
block = CacheBlock.from_payload(
    payload=payload,
    model_id="qwen2.5-7b",
    layer_id=0,
    head_id=0,
    token_range=(0, 1),
    dtype="float32",
    device_tier=DeviceTier.HBM,
)

store.put(block)
recovered = store.get_by_id(block.metadata.block_id)
```

## Layerwise Offload

The tiered store also supports LMCache-inspired layerwise transfer. Engine adapters can
emit KV blocks in layer-major order, then the store pipelines one layer at a time:

```python
from goldenexperience.cache_core import CacheQuery, DeviceTier
from goldenexperience.tiered_store import LayerwiseOffloadPlan

list(store.put_layers({0: layer0_blocks, 1: layer1_blocks}, target_tier=DeviceTier.CPU))

results = store.offload_layers(
    LayerwiseOffloadPlan(
        query=CacheQuery(model_id="qwen2.5-7b", prefix_hash="shared-system-prompt"),
        target_tier=DeviceTier.NVME,
    )
)

for layer in store.retrieve_layers(CacheQuery(model_id="qwen2.5-7b"), target_tier=DeviceTier.HBM):
    engine_adapter.inject_kv(layer.blocks)
```

`put_layers` and `retrieve_layers` are generator-style APIs: while one layer is consumed
by the caller, the next layer can already be scheduled in the background.

## Policy-Driven Tiering

`TieredKVStore` separates mechanism from policy:

- Eviction policies choose safe victims while respecting pinned and retained blocks:
  `LRUEvictionPolicy`, `LFUEvictionPolicy`, and `CostAwareEvictionPolicy`.
- `WatermarkOffloadPolicy` demotes blocks when HBM or CPU utilization crosses a high
  watermark.
- Prefetch policies build plans from access context: `DecodeWindowPrefetchPolicy` warms
  upcoming layers, and `PrefixHotnessPrefetchPolicy` warms the hottest matching blocks.

```python
from goldenexperience.cache_core import CacheQuery, DeviceTier
from goldenexperience.tiered_store import PrefetchContext, WatermarkOffloadPolicy

store.enforce_offload_policy()

plan = store.build_prefetch_plan(
    PrefetchContext(
        query=CacheQuery(model_id="qwen2.5-7b", prefix_hash="shared-system-prompt"),
        target_tier=DeviceTier.HBM,
        current_layer_id=7,
        lookahead_layers=2,
        max_blocks=4,
    )
)
store.prefetch(plan)
```

## Research Roadmap

- M0: Repo skeleton, public APIs, synthetic benchmark, paper artifact docs.
- M1: Tiered KV store with eviction, offload, and prefetch policies.
- M2: Hugging Face and vLLM adapters with engine-neutral signatures.
- M3: Same-family cross-model KV mapping and quality-gated reuse.
- M4: Long-context, multi-turn, RAG prefix-sharing, and batch-serving evaluation.
- M5: Paper writing, artifact package, Docker/conda reproduction path.

See `docs/design.md`, `docs/paper_outline.md`, and `docs/artifact.md` for the detailed
research and artifact plan.
