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

## Required Figures

- Architecture: vLLM, LMCache MP, Mooncake Store, and GoldenExperience patch hooks.
- Taxonomy table for the three reuse scenarios.
- TTFT and accepted-reuse rate for base/LoRA serving.
- Quality versus latency for GoldenScale projection strategies.
- Fallback reason breakdown.
- Patch overhead: lookup, materialization, accounting.

## Limitations to State

- GoldenExperience does not claim to improve LMCache MP offload or vLLM inference kernels.
- Cross-base reuse is experimental and disabled without calibration.
- Reuse quality must be evaluated per model pair, task, and prefix distribution.
- Upstream vLLM, LMCache, and Mooncake APIs may change, so the patch must remain small and
  rebased.
