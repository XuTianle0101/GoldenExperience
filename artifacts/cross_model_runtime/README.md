# Cross-Model Runtime Artifacts

This directory stores compact manifests for cross-size vLLM + LMCache MP runtime
runs. Raw run directories may contain hundreds of MB of Mooncake KV payloads and
are ignored by Git; keep only curated manifests under `manifests/`.

Current proof modes:

- `native_target_seed`: run the source model to populate source KV metadata, run a
  target-model prefill to materialize target-shaped KV in the same LMCache MP +
  Mooncake runtime, then restart target vLLM and verify external KV retrieval.
- `hidden_bridge`: run source offload, verify a cross-model source candidate exists,
  invoke the learned hidden bridge materializer, inject accepted target-shaped chunks
  through Mooncake plus the persistent external key index, then start target vLLM.
  Strict policy only reuses when the quality gate passes; otherwise target vLLM falls
  back to local prefill.

Latest recorded strict result:

- `manifests/prefix_specific_strict_20260709T0253Z.json`: prefix-specific
  Qwen3-8B -> Qwen3-14B bridge passed strict offline and runtime gates. Source
  lookup found `111/111` chunks, target was a direct miss before materialization
  (`0/111`), materializer injected `111` target-shaped chunks, and target vLLM
  consumed `1776` external KV-transfer prompt tokens with `111` Mooncake GET events.
  This artifact is only valid for calibration prefixes represented during training.
- `manifests/prefix_specific_strict_20260709T0253Z_vs_qwen3_14b_same_model_restart_20260709T0223Z.json`:
  comparison against same-model Qwen3-14B offload -> restart -> reuse. Same-model
  reuse TTFT was `143.32 ms`; prefix-specific cross-model target TTFT was
  `25201.50 ms` because materialization/injection is synchronous and external KV
  transfer dominates the target request path in this MVP.
- `manifests/strict_20260709T0220Z.json`: source Qwen3-8B candidate lookup succeeded
  (`111/111` chunks found) and target Qwen3-14B was a direct miss (`0/111` chunks),
  but the bridge failed the quality gate (`hidden=0.7808`, `value=0.6182`,
  `decode=0.7626`), so no materialized KV was injected and vLLM used fallback.
- `manifests/strict_20260709T0220Z_vs_qwen3_14b_same_model_restart_20260709T0223Z.json`:
  comparison against a same-model Qwen3-14B offload -> restart -> reuse baseline,
  where same-model reuse succeeded with `1792` external KV-transfer prompt tokens and
  `111` Mooncake GET events. The same comparison also records an unsafe shadow smoke
  (`unsafe_shadow4_20260709T0226Z`) that injected 4 materialized chunks and vLLM consumed
  `64` external KV-transfer prompt tokens. That shadow run is explicitly disallowed for
  automatic reuse because the quality gate still fails.
