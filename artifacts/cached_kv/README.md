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

`regularization_capacity_sweep_20260710.json` records the validation-only ridge and rank
sweep after the v2 scaled baseline. Stronger regularization and more supervised positions
substantially improve both directions, but the best candidates remain below every
production output-quality threshold.

`expanded_corpus_nonlinear_ab_20260710.json` records the 256-train/64-validation corpus
and compares the v3 linear map with the v4 SiLU correction. v4 improves generation in
both directions, especially 8B to 14B, but neither direction is approved.

`cka_layer_alignment_ab_20260710.json` compares fixed normalized-depth source windows
with train-only monotonic linear-CKA alignment. CKA improves forward generation metrics
but regresses reverse generation and tensor metrics, so it remains an opt-in experiment.

`validation_candidate_8b_to_14b_20260713.json` records the first full validation
candidate emitted by the production-gated training path. The rank-512 v4 bridge is
bound to content-addressed local Qwen3-8B and Qwen3-14B weights, but it fails every
output-quality gate and remains validation-only. Raw candidate manifests, per-prompt
results, and generated weights stay local and are ignored by Git.

Only a `CachedKVBridgeManifest` whose derived `approved` property is true may be used by
the runtime materializer. Missing held-out accuracy or Mooncake cost evidence keeps a
manifest fail closed.
