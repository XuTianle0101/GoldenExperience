# Cross-Model Runtime Artifacts

This directory stores compact manifests for cross-size vLLM + LMCache MP runtime
runs. Raw run directories may contain hundreds of MB of Mooncake KV payloads and
are ignored by Git; keep only curated manifests under `manifests/`.

Current proof mode:

- `native_target_seed`: run the source model to populate source KV metadata, run a
  target-model prefill to materialize target-shaped KV in the same LMCache MP +
  Mooncake runtime, then restart target vLLM and verify external KV retrieval.
- This proves target-shaped runtime reuse through vLLM/LMCache/Mooncake. It does
  not claim hidden-bridge KV injection quality or latency yet.
