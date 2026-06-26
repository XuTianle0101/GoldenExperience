# Paper Outline

## Title Candidates

- GoldenExperience: Cross-Model KV Cache Reuse as an LMCache Patch for SGLang
- Reusing KV Cache Across Models Without Rebuilding the Serving Stack

## Abstract Shape

1. Problem: KV Cache is valuable across related model deployments, but serving stacks treat
   it as model-local state.
2. Gap: existing cache systems focus on storage/offload and same-model prefix reuse, not
   controlled reuse across LoRA adapters, size variants, or different base models.
3. Method: a small LMCache patch driven by model identity, reuse planning, materialization,
   and quality/fallback accounting.
4. Evidence: TTFT improvement and accepted reuse rate under SGLang + LMCache, with quality
   gates for three model-pair scenarios.
5. Contribution: a narrow, upstream-friendly framework that leaves inference and offload to
   SGLang and LMCache.

## Main Contributions

- A model-pair taxonomy for cross-model KV reuse: base/LoRA, same-model size variants, and
  different base models.
- A control-plane planner that emits explicit `ReusePlan` metadata, confidence, gates, and
  fallback reasons.
- An LMCache patch surface for secondary lookup, materialization, and quality accounting.
- A SGLang-based evaluation path that measures latency gain without modifying inference
  semantics or cache offload mechanics.

## Evaluation Questions

- How often can base/LoRA deployments reuse KV safely?
- What TTFT improvement is available when same-model size variants share prefixes?
- Which layer subsets and projection methods are useful for size-variant reuse?
- How much overhead does the LMCache secondary lookup and materializer add?
- When do quality gates reject reuse, and are those rejections predictive of task quality?
- Is cross-base reuse ever useful under strict calibration and task allowlists?

## Required Figures

- Architecture: SGLang, LMCache, and the GoldenExperience patch hooks.
- Taxonomy table for the three reuse scenarios.
- TTFT and accepted-reuse rate for base/LoRA serving.
- Quality versus latency for size-variant projection strategies.
- Fallback reason breakdown.
- Patch overhead: lookup, materialization, accounting.

## Limitations to State

- GoldenExperience does not claim to improve LMCache offload or SGLang inference kernels.
- Cross-base reuse is experimental and disabled without calibration.
- Reuse quality must be evaluated per model pair, task, and prefix distribution.
- Upstream LMCache and SGLang APIs may change, so the patch must remain small and rebased.
