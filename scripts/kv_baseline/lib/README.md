# KV baseline Bash modules

`run_vllm_lmcache_mp_l2_baseline.sh` is the public entrypoint. Keep phase
orchestration there and put reusable Bash mechanics in these modules:

- `common.sh`: command checks, log tailing, and user-facing run summaries.
- `processes.sh`: shared process shutdown and trap cleanup.
- `lmcache_mp.sh`: standalone LMCache MP server startup and readiness checks.
- `engine_server.sh`: vLLM server startup and readiness checks, plus the explicit SGLang legacy path.
- `requests.sh`: warmup/measured requests, metrics fetches, and final summary.

The modules expect the entrypoint to define `python_bin`, `helper`, `client`,
`server_pid`, and `lmcache_pid`, and to source the generated `GE_RUNTIME_*`
environment before loading them.
