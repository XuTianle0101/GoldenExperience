# Experiment Matrix

## Execution Status

The matrix below describes the implemented end-to-end protocol, not a set of completed success
claims. The registered Qwen3 4B-to-8B screening fit completed, but its full 1,024-prompt
method-dev stage failed the fixed `0.45` oracle-safe coverage gate at `0.138671875` for the
selected rank-64/seed-17 candidate. The union over all nine screened candidates was only
`0.3681640625`. This blocks every selector, calibration, other-direction, validation,
semantic-sealed, and runtime experiment in the present workspace. The sealed split has not
been opened.

The completed screening experiment is therefore reported as a negative result and a boundary
on fixed low-rank affine KV transport. Rows below that require a frozen successful structure
remain prospective and may only be run in a newly preregistered workspace with fresh
development evidence.

## Baselines

- vLLM without external KV reuse.
- vLLM + LMCache MP + Mooncake Store with cross-model reuse disabled.
- Same-model offload/reuse across vLLM restart through LMCache MP and Mooncake L2.
- GoldenScale shadow mode, where projected KV is validated but not injected.
- GoldenExperience base/LoRA reuse enabled.
- GoldenScale reuse enabled.
- GoldenExperience cross-base reuse enabled only for calibrated experiments.

## Reuse Scenarios

| Scenario | Example Pair | Required Evidence | Primary Risk |
| --- | --- | --- | --- |
| Base <-> LoRA | `qwen3-8b` <-> `qwen3-8b-lora-math` | same base, tokenizer, KV shape, adapter drift gate | LoRA changes hidden states enough to hurt quality |
| Size variant | `qwen3-8b` <-> `qwen3-14b` | layer/head map, hidden bridge calibration, prefix match | bridge materialization overhead or quality drift |
| Different base | `qwen3-8b` <-> `llama-3.1-8b` | calibration set, tokenizer bridge, task allowlist | unsafe semantic mismatch |

## Workloads

- Shared system prompts for chat serving.
- RAG prefixes with repeated document context.
- Agent traces with repeated tool schema and planning prefix.
- LoRA multi-tenant serving where several adapters share the same base model.
- Size-variant fallback serving where a small model and a large model share traffic.

## Main Metrics

- TTFT delta versus unmodified vLLM + LMCache MP + Mooncake Store.
- Cross-model cache hit rate and accepted reuse rate.
- Fallback rate by reason: no candidate, not calibrated, shape mismatch, quality gate fail.
- Materialization overhead: alias, projection, or learned translator latency.
- Quality gate metrics: perplexity drift, probe-logit drift, exact match/F1, task score.
- LMCache MP overhead: secondary lookup latency and metadata/index memory.
- Mooncake Store evidence: storage-root bytes, L2 operation logs, and metadata stability.

## Ablations

- Base/LoRA drift threshold sweep.
- Prefix length sweep.
- Layer subset: early layers, middle layers, late layers, all layers.
- Projection type for size variants: direct alias, linear projection, learned projection.
- Direction for size variants: 8B->14B, 14B->8B, and bidirectional serving.
- Runtime gate: materialization cost ratio from 0.3 to 0.9 of target prefill.
- Calibration set size.
- Cross-base task allowlist versus no allowlist.
