# Experiment Matrix

## Baselines

- No offload: engine default KV cache behavior.
- CPU-only offload: HBM to CPU movement without NVMe.
- Engine prefix cache: vLLM prefix cache or equivalent backend cache.
- PagedAttention default: vLLM default paging policy where applicable.
- GoldenExperience offload only.
- GoldenExperience cross-model reuse only.
- GoldenExperience offload plus cross-model reuse.

## Workloads

- Long-context QA: low prefix sharing, large prompt length, TTFT-sensitive.
- Multi-turn chat: high session locality, repeated prefixes, moderate batch size.
- RAG prefix sharing: shared system prompt and retrieved context across requests.
- Agent workflow: repeated tool traces and planning prefixes.

## Model Pairs

- small to large: same-family model variants such as Qwen 7B to 14B.
- large to small: quality and latency tradeoff for downshifting.
- base to instruct: same architecture with instruction-tuned weights.

## Main Metrics

- TTFT, end-to-end latency, P50, P95, P99.
- Decode throughput and request throughput.
- GPU memory peak and average HBM residency.
- Cache hit rate, prefetch miss rate, and offload bandwidth.
- Mapper latency and projection memory.
- Perplexity drift, exact match/F1, and task quality gate pass rate.

## Ablations

- Tier capacity sweep: HBM, CPU, and NVMe budget.
- Prefix length sweep.
- Projection type: identity, final-dimension projection, learned projection.
- Reuse depth: early layers, late layers, all layers.
- Quality gate thresholds and fallback policy.

