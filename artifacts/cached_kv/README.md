# Qwen3 Cached-KV Bridge Artifacts

This directory contains small manifests and curated result summaries. Generated
`.safetensors` weights are ignored by Git.

`bidirectional_pipeline_smoke_20260710.json` verifies that both Qwen3 directions execute
the real cached-KV collection, fitting, RoPE translation, and target DynamicCache decode
path. It is deliberately not a calibration artifact: the run used rank 2, four training
positions, one validation prompt, and no sealed test or runtime cost evidence. Both
directions fail the production quality thresholds.

`bidirectional_validation_sweep_20260710.json` records validation-only rank 256/512
results against the bounded non-thinking chat contract. Native task assertions pass, but
both bridge directions remain far below the accuracy thresholds. The sealed test remains
unopened and no candidate weights are checked in.

`scaled_baseline_ab_20260710.json` compares the v1 fixed interpolation baseline with the
v2 learned per-channel baseline under identical rank, data, and prompt settings. v2
improves both directions, but still fails the end-to-end accuracy gates.

Only a `CachedKVBridgeManifest` whose derived `approved` property is true may be used by
the runtime materializer. Missing held-out accuracy or Mooncake cost evidence keeps a
manifest fail closed.
