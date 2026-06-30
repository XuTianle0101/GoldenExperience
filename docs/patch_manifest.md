# GoldenExperience Patch Manifest: vLLM + LMCache

## Hooks
- `serving_request_metadata` (required): Serving request/session metadata before LMCache lookup -- Attach source/target ModelRef, prefix hash, and experiment flags.
- `lmcache_cross_model_lookup` (required): LMCache lookup miss path or secondary index lookup -- Ask the GoldenExperience planner whether a compatible source model entry exists.
- `goldenexperience_materializer` (required): LMCache retrieve path before KV is handed back to the serving engine -- Alias, project, or translate retrieved KV according to a ReusePlan.
- `quality_gate_accounting` (required): LMCache store/metrics metadata -- Record confidence, fallback reason, and calibration provenance.

## Invariants
- Do not modify serving-engine scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache storage, offload, eviction, or prefetch implementations.
- If a ReusePlan is not ready, fall back to the original serving-engine plus LMCache path.
- All cross-model reuse must carry scenario, transform_id, confidence, and calibration metadata.

## Notes
- The patch should be small enough to carry as a delta on top of upstream LMCache.
- Source installs of vLLM and LMCache are supported for development and debugging.
- SGLang remains supported through explicit legacy control scripts.
