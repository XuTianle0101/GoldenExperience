# Paper Outline

## Title Candidates

- GoldenExperience: Cross-Model KV Cache Reuse on vLLM + LMCache MP + Mooncake Store
- Reusing KV Cache Across Models Without Rebuilding the Serving Stack

## Abstract Shape

1. Problem: KV Cache is valuable across related model deployments, but serving stacks treat
   it as model-local state.
2. Gap: existing cache systems focus on storage/offload and same-model prefix reuse, not
   controlled reuse across LoRA adapters, size variants, or different base models.
3. Method: a small LMCache MP patch driven by model identity, reuse planning,
   materialization, and quality/fallback accounting.
4. Evidence: TTFT improvement and accepted reuse rate under vLLM + LMCache MP + Mooncake
   Store, with quality gates for three model-pair scenarios.
5. Contribution: a narrow, upstream-friendly framework that leaves inference to vLLM and
   shared KV persistence to LMCache MP plus Mooncake Store.

## Main Contributions

- A model-pair taxonomy for cross-model KV reuse: base/LoRA, same-model size variants, and
  different base models.
- A control-plane planner that emits explicit `ReusePlan` metadata, confidence, gates, and
  fallback reasons.
- An LMCache MP patch surface for secondary lookup, materialization, and quality accounting.
- A vLLM-based evaluation path that measures latency gain without modifying inference
  semantics or replacing the cache/offload mechanics.

## Evaluation Questions

- How often can base/LoRA deployments reuse KV safely?
- What TTFT improvement is available when same-model size variants share prefixes?
- Which layer subsets and projection methods are useful for GoldenScale reuse?
- How much overhead does LMCache MP secondary lookup and materialization add?
- When do quality gates reject reuse, and are those rejections predictive of task quality?
- Is cross-base reuse ever useful under strict calibration and task allowlists?

## Current Empirical Record (2026-07-13)

These results are development evidence, not final test-set claims. All cross-parameter
quality numbers below use the 64-prompt validation split (64 task prompts, 1024 evaluated
tokens, and token buckets 32/128/512/2048). The sealed test split has not been opened.

### Why the hidden-state bridge was retired

- A rank sweep from 16 to 192 improved key cosine from 0.8793 to 0.9225 and value cosine
  from 0.4358 to 0.6028, but decode-logit cosine and top-1 agreement remained unstable.
- A five-layer, rank-512 multisource bridge reached key cosine 0.9388 and value cosine
  0.6884 on three held-out prompts, with decode-logit cosine 0.8781. It still failed the
  quality gate.
- A general rank-1024 bridge reached decode-logit cosine 0.9129, but hidden cosine was
  0.7750, value cosine was 0.6347, and decode top-1 match was only 2/3.
- A prefix-specific rank-1024 bridge appeared nearly perfect on three represented-prefix
  probes (key cosine 0.9991, value cosine 0.9940, decode-logit cosine 0.9929). Runtime
  evaluation exposed this as prefix overfitting: the exact-answer assertion failed.

This sequence is useful negative evidence: small cosine probe sets can approve a bridge
that does not preserve free-running task behavior. It motivated exact-answer evaluation,
larger held-out splits, and fail-closed runtime admission.

### Runtime and storage evidence

- Same-model Qwen3-14B Mooncake reuse transferred 1792 external-KV prompt tokens after a
  vLLM restart. TTFT fell from 253.61 ms to 143.32 ms, a 110.29 ms improvement, with 111
  Mooncake GET events. This establishes that the underlying LMCache/Mooncake path works.
- The prefix-specific cross-parameter bridge materialized 111 target chunks and vLLM
  consumed 1776 external-KV prompt tokens, but target TTFT was 25201.50 ms, versus
  143.32 ms for same-model reuse. Synchronous transformation and publication dominated
  the request, and the exact-answer assertion failed.
- The later isolated cost experiment measured 223.83 ms native-target prefill P95 versus
  769.00 ms Mooncake read-transform-put P95, a 3.4357 ratio against the 0.70 limit.
- API rollback removed temporary object keys, but 6.69 GB of backing files remained.
  Physical reclamation is therefore a correctness requirement for a later runtime phase,
  not evidence that can be counted toward approval.

### Current cached-KV bridge

The selected validation configuration is rank 512, source window 3, a scaled SiLU
residual bridge, and four nonlinear-up-only refinement steps at learning rate 3e-6. The
mixed objective combines native-generation loss at weight 1.0 with prompt-tail
distillation at weight 0.25.

| Metric (gate) | 8B->14B before | 8B->14B mixed | 14B->8B before | 14B->8B mixed |
| --- | ---: | ---: | ---: | ---: |
| Key cosine (>=0.95) | 0.931665 | 0.931665 | 0.929008 | 0.929008 |
| Value cosine (>=0.95) | 0.798981 | 0.798946 | 0.792346 | 0.792324 |
| Next-token agreement (>=0.98) | 0.816406 | 0.823242 | 0.743164 | 0.751953 |
| Greedy continuation match (>=0.98) | 0.082031 | 0.837891 | 0.636719 | 0.738281 |
| Perplexity drift (<=2%) | 25.13% | 23.21% | 40.35% | 35.03% |
| Bridge task score (>=0.95) | 0.09375 | 0.90625 | 0.84375 | 0.84375 |
| Task-score drop (<=1%) | 90.625% | 9.375% | 15.625% | 15.625% |

Generation-aligned refinement produces a large forward task/greedy gain. The reverse
direction confirms bidirectional improvements in greedy match, next-token agreement, and
perplexity drift, but it preserves rather than improves task score. Both directions still
fail seven quality checks, so neither bridge is approved.

### Checkpoint-selection and objective ablations

- A four-prompt teacher-forced holdout failed to detect an eight-step exact-answer
  regression. Checkpoint selection now uses 16 train-only prompts, free-runs 16 greedy
  tokens, and ranks checkpoints by task score, greedy match, then teacher objective.
- The 16-prompt free-running holdout selected step 4 in both directions. Forward holdout
  task score moved from 0.25 to 1.0 and greedy match from 0.1719 to 0.8789; reverse task
  score remained 1.0 while greedy match moved from 0.6641 to 0.8203.
- Against pure native-generation refinement in the forward direction, the mixed objective
  preserved task score 0.90625, improved next-token agreement by 0.00586, and reduced
  perplexity drift by another 0.49 percentage points, at a 0.00293 greedy-match cost.

### Evidence boundary and retained provenance

- The authoritative aggregate record is
  `artifacts/cached_kv/bidirectional_mixed_refinement_summary_20260713.json`; direction
  summaries retain raw-result paths and SHA-256 digests.
- The two mixed raw per-prompt result files are retained for analysis of the remaining six
  forward and ten reverse exact-answer failures. Earlier smoke outputs, superseded bridge
  weights, and machine-specific runtime payloads are not publication artifacts.
- Validation failure keeps the sealed test closed. Chunk batching, batch serialization,
  further cost optimization, and Mooncake physical reclamation remain deferred until all
  validation gates pass and the sealed test subsequently succeeds.

## Required Figures

- Architecture: vLLM, LMCache MP, Mooncake Store, and GoldenExperience patch hooks.
- Taxonomy table for the three reuse scenarios.
- TTFT and accepted-reuse rate for base/LoRA serving.
- Quality versus latency for GoldenScale projection strategies.
- Fallback reason breakdown.
- Patch overhead: lookup, materialization, accounting.
- Validation-quality trajectory: hidden-state bridge, generation-aligned refinement, and
  mixed refinement in both directions.
- Native prefill versus Mooncake read-transform-put P95, including the 0.70 admission
  threshold.

## Limitations to State

- GoldenExperience does not claim to improve LMCache MP offload or vLLM inference kernels.
- Cross-base reuse is experimental and disabled without calibration.
- Reuse quality must be evaluated per model pair, task, and prefix distribution.
- Cosine and one-step logit agreement are not sufficient proxies for free-running exact
  answers; checkpoint selection and final admission require free-running task evaluation.
- Upstream vLLM, LMCache, and Mooncake APIs may change, so the patch must remain small and
  rebased.
