# Shared KV Substrate

GoldenExperience's cross-model planner does not own KV storage. The same-model baseline
used to validate persistence now treats the shared KV substrate as:

```text
vLLM inference process
        |
        | LMCacheMPConnector
        v
standalone LMCache MP server
        |
        | L2 adapter: type=mooncake_store
        v
Mooncake Store on local TCP + SSD
```

This is the default for `scripts/kv_baseline/run_vllm_lmcache_mooncake_kv_baseline.sh`:

- `GE_KV_BACKEND=mp`
- `GE_ENGINE=vllm`
- `GE_LMCACHE_MP_L2_ADAPTER_TYPE=mooncake_store`
- `GE_MOONCAKE_PROTOCOL=tcp`
- `GE_MOONCAKE_STORAGE_ROOT=$GE_KV_CACHE_DIR/mooncake`

The script starts Mooncake master/metadata services, then LMCache MP, then vLLM. During
validation it stores KV through LMCache MP, restarts only the inference engine, and verifies
that the reuse request can observe cache/L2 evidence from the still-running MP service and
Mooncake-backed storage.

## Why This Boundary

- Engine-local caches are not a stable substrate for cross-instance reuse.
- LMCache MP gives GoldenExperience a persistent process boundary that survives engine
  restarts.
- Mooncake Store supplies an L2 adapter with explicit metadata/master endpoints and a
  storage root that can be inspected for disk evidence.
- Cross-model lookup and materialization should attach to LMCache MP/L2 metadata later; the
  planner semantics do not change in this refactor.

## Diagnostics

Filesystem L2 adapters can still be selected with `GE_LMCACHE_MP_L2_ADAPTER_TYPE=fs` for
MP-server diagnostics, but they are not the project substrate. New same-model evidence for
cross-model work should come from the vLLM + LMCache MP + Mooncake Store path.
