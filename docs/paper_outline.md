# Paper Outline

## Title Candidates

- GoldenExperience: Engine-Decoupled Tiered KV Cache Reuse Across Same-Family LLMs
- Beyond Engine-Local KV Cache: Tiered Offload and Cross-Model Reuse for LLM Serving

## Abstract Shape

1. Problem: KV cache dominates memory and TTFT for long-context and shared-prefix serving.
2. Gap: existing serving stacks keep KV cache tied to one engine and one model instance.
3. Method: tiered cache store plus same-family cross-model KV mapping.
4. Evidence: TTFT, throughput, memory, and quality-gated reuse benchmarks.
5. Contribution: engine-decoupled artifact with adapters and reproducible evaluation.

## Main Contributions

- A reusable cache block API that separates KV metadata, placement, and engine integration.
- A tiered offload system for HBM, CPU memory, and NVMe with policy-controlled demotion and
  prefetch.
- A same-family cross-model reuse path with compatibility signatures, projection mapping,
  confidence scoring, and fallback recompute.
- A benchmark suite for long-context QA, multi-turn chat, RAG prefix sharing, and agent
  workflows.

## Evaluation Questions

- How much does tiered offload reduce HBM pressure under long-context workloads?
- How much TTFT improvement is possible when shared prefixes are reused?
- When can same-family model variants reuse mapped KV without quality loss?
- What are the overheads of projection, prefetch misses, and NVMe movement?
- Does the adapter design work across at least two inference backends?

## Required Figures

- System architecture: engine adapters, cache core, tiered store, mapper, policy layer.
- TTFT and tail latency versus baseline systems.
- GPU memory peak and cache hit rate across prompt lengths.
- Quality versus reuse aggressiveness.
- Ablation: offload only, reuse only, offload plus reuse.

## Limitations to State

- v1 focuses on same-family models and does not claim arbitrary cross-architecture reuse.
- Projection confidence must be calibrated per model family.
- The NVMe prototype is artifact-friendly and should be replaced by async/mmap storage for
  production-grade serving.

