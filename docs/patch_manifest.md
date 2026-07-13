# GoldenExperience Patch Manifest: vLLM + LMCache MP + Mooncake Store

## Hooks
- `engine_request_metadata` (required): OpenAI-compatible request metadata before LMCache MP lookup -- Attach source/target ModelRef, prefix hash, and experiment flags.
- `lmcache_cross_model_lookup` (required): LMCache lookup miss path or secondary index lookup -- Ask the GoldenExperience planner whether a compatible source model entry exists.
- `calibrated_risk_gate` (required): LMCache MP candidate path before source KV retrieval -- Evaluate the source-only sidecar and reject missing, OOD, stale, or statistically unsafe prefixes without reading source KV.
- `lmcache_retrieve_transform` (required): LMCache MP RETRIEVE_TRANSFORM connector worker path -- Batch-read accepted source chunks, run head-aware transport, and scatter every layer into registered vLLM paged KV slots.
- `goldenexperience_materializer` (optional): Legacy LMCache MP read-transform-put path for v4 artifacts -- Alias KV, bridge hidden states, or translate retrieved state according to a ReusePlan.
- `quality_gate_accounting` (required): LMCache store/metrics metadata -- Record confidence, fallback reason, and calibration provenance.

## Invariants
- Do not modify vLLM scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache MP storage, offload, eviction, or prefetch implementations.
- Do not replace Mooncake Store; only configure and observe it as persistent L2.
- If a ReusePlan is not ready, fall back to the original vLLM plus LMCache MP path.
- All cross-model reuse must carry scenario, state_kind, transform_id, confidence, and calibration metadata.
- Rejected selective requests must not read source KV or write target objects.
- Accepted v5 requests must publish load-complete only after every paged-KV layer succeeds; partial writes remain invalid for native prefill overwrite.
- Direct selective reuse must never create target Mooncake objects.

## Notes
- The patch should be small enough to carry as a delta on top of upstream LMCache.
- The default baseline is vLLM + LMCache MP + Mooncake Store; filesystem L2 adapters are diagnostics only.
