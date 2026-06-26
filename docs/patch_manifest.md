# GoldenExperience Patch Manifest: SGLang + LMCache

## Hooks
- `sglang_request_metadata` (required): SGLang request/session metadata before LMCache lookup -- Attach source/target ModelRef, prefix hash, and experiment flags.
- `lmcache_cross_model_lookup` (required): LMCache lookup miss path or secondary index lookup -- Ask the GoldenExperience planner whether a compatible source model entry exists.
- `goldenexperience_materializer` (required): LMCache retrieve path before KV is handed back to SGLang -- Alias, project, or translate retrieved KV according to a ReusePlan.
- `quality_gate_accounting` (required): LMCache store/metrics metadata -- Record confidence, fallback reason, and calibration provenance.

## Invariants
- Do not modify SGLang scheduling, attention kernels, or token generation semantics.
- Do not replace LMCache storage, offload, eviction, or prefetch implementations.
- If a ReusePlan is not ready, fall back to the original SGLang plus LMCache path.
- All cross-model reuse must carry scenario, transform_id, confidence, and calibration metadata.

## Notes
- The patch should be small enough to carry as a delta on top of upstream LMCache.
- Source installs of SGLang and LMCache are supported for development and debugging.
