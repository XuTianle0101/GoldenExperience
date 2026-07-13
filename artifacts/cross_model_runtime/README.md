# Cross-Model Runtime Artifacts

This directory stores generated evidence for cross-size vLLM + LMCache MP runtime runs.
Raw run directories may contain hundreds of MB of Mooncake KV payloads and are ignored
by Git. Commit a compact manifest only for a result that remains part of the active
validation or sealed-test record.

Current proof modes:

- `native_target_seed`: run the source model to populate source KV metadata, run a
  target-model prefill to materialize target-shaped KV in the same LMCache MP +
  Mooncake runtime, then restart target vLLM and verify external KV retrieval.
- `cached_kv`: run source offload, verify a prompt-bound source candidate exists, invoke
  the resident cached-KV materializer, inject accepted target-shaped chunks
  through Mooncake plus the persistent external key index, then start target vLLM.
  Strict policy only reuses when the quality gate passes; otherwise target vLLM falls
  back to local prefill.

The current harness is `scripts/run_qwen3_cached_kv_runtime.py`. It remains fail closed:
without an approved direction-specific manifest, it records fallback rather than
publishing translated KV.

Historical hidden-state, prefix-specific, same-model comparison, and unsafe-shadow
manifests were consolidated into `docs/paper_outline.md` and removed. They demonstrated
the storage path and fail-closed behavior, but either failed semantic quality or measured
an obsolete synchronous materializer. Git history retains their exact original payloads.
