# GoldenExperience Patch Manifest: vLLM + LMCache MP + Mooncake Store

## Hooks
- `engine_request_metadata` (required): OpenAI-compatible request metadata before LMCache MP lookup -- Attach source/target ModelRef, prefix hash, and experiment flags.
- `lmcache_cross_model_lookup` (required): LMCache lookup miss path or secondary index lookup -- Ask the GoldenExperience planner whether a compatible source model entry exists.
- `goldenexperience_materializer` (required): LMCache MP retrieve path before KV is handed back to vLLM -- Alias, project, or translate retrieved KV according to a ReusePlan.
- `quality_gate_accounting` (required): LMCache store/metrics metadata -- Record confidence, fallback reason, and calibration provenance.

## Invariants
- Do not modify vLLM scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache MP storage, offload, eviction, or prefetch implementations.
- Do not replace Mooncake Store; only configure and observe it as persistent L2.
- If a ReusePlan is not ready, fall back to the original vLLM plus LMCache MP path.
- All cross-model reuse must carry scenario, transform_id, confidence, and calibration metadata.

## Notes
- The patch should be small enough to carry as a delta on top of upstream LMCache.
- The default baseline is vLLM + LMCache MP + Mooncake Store; filesystem L2 adapters are diagnostics only.
