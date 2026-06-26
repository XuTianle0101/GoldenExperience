# Experiment Matrix

## Baselines

- SGLang without LMCache.
- SGLang with unmodified LMCache.
- SGLang with LMCache plus GoldenExperience metadata, but cross-model reuse disabled.
- GoldenExperience base/LoRA reuse enabled.
- GoldenExperience same-model size-variant reuse enabled.
- GoldenExperience cross-base reuse enabled only for calibrated experiments.

## Reuse Scenarios

| Scenario | Example Pair | Required Evidence | Primary Risk |
| --- | --- | --- | --- |
| Base <-> LoRA | `qwen2.5-7b` <-> `qwen2.5-7b-lora-math` | same base, tokenizer, KV shape, adapter drift gate | LoRA changes hidden states enough to hurt quality |
| Size variant | `qwen2.5-7b` <-> `qwen2.5-14b` | layer/head map, projection calibration, prefix match | projection overhead or quality drift |
| Different base | `qwen2.5-7b` <-> `llama-3.1-8b` | calibration set, tokenizer bridge, task allowlist | unsafe semantic mismatch |

## Workloads

- Shared system prompts for chat serving.
- RAG prefixes with repeated document context.
- Agent traces with repeated tool schema and planning prefix.
- LoRA multi-tenant serving where several adapters share the same base model.
- Size-variant fallback serving where a small model and a large model share traffic.

## Main Metrics

- TTFT delta versus unmodified SGLang + LMCache.
- Cross-model cache hit rate and accepted reuse rate.
- Fallback rate by reason: no candidate, not calibrated, shape mismatch, quality gate fail.
- Materialization overhead: alias, projection, or learned translator latency.
- Quality gate metrics: perplexity drift, probe-logit drift, exact match/F1, task score.
- LMCache overhead: secondary lookup latency and metadata/index memory.

## Ablations

- Base/LoRA drift threshold sweep.
- Prefix length sweep.
- Layer subset: early layers, middle layers, late layers, all layers.
- Projection type for size variants: direct alias, linear projection, learned projection.
- Calibration set size.
- Cross-base task allowlist versus no allowlist.
