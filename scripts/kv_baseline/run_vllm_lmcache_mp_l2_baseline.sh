#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a two-phase same-model KV offload/reuse baseline.

Default path:
  1. start a standalone LMCache MP server,
  2. configure filesystem L2 storage,
  3. start vLLM with LMCacheMPConnector,
  4. send an offload request,
  5. restart only vLLM, then send the reuse request.

SGLang + LMCache remains available as a legacy control path:
  GE_KV_BACKEND=legacy GE_ENGINE=sglang scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh -- --tp 1

Usage:
  GE_MODEL_PATH=/models/Qwen3-8B scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh [-- engine args...]

Important environment:
  PYTHON_BIN                         Python executable. Default: python3.
  GE_KV_BACKEND                      mp or legacy. Default: mp.
  GE_ENGINE                          vllm or sglang. Default: vllm.
  VLLM_BIN                           vLLM executable. Default: vllm.
  LMCACHE_BIN                        LMCache executable. Default: lmcache.
  GE_MODEL_PATH                      Model path or HF model id. Default: Qwen/Qwen3-8B.
  GE_MODEL_NAME                      OpenAI model name in requests. Default: GE_MODEL_PATH.
  GE_RUN_DIR                         Output directory. Default: artifacts/kv_baseline/<UTC timestamp>.
  GE_KV_CACHE_DIR                    Persistent filesystem L2 dir. Default: $GE_RUN_DIR/cache.
  GE_FORCE_DISK_OFFLOAD              Require disk-backed evidence. Default: 1.
  GE_REQUIRE_REUSE_EVIDENCE          Fail if reuse evidence is absent. Default: 1 when disk is forced.
  GE_VLLM_KV_CONNECTOR_MODULE_PATH   Optional vLLM kv_connector_module_path override.
  -h, --help                         Show this help.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if [ "${1:-}" = "--" ]; then
  shift
fi

python_bin="${PYTHON_BIN:-python3}"
helper="scripts/kv_baseline/vllm_lmcache_mp_l2.py"
client="scripts/kv_baseline/kv_baseline_client.py"

runtime_env="$("$python_bin" "$helper" prepare --repo-root "$REPO_ROOT" -- "$@")"
# shellcheck disable=SC1090
source "$runtime_env"

# shellcheck source=scripts/kv_baseline/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=scripts/kv_baseline/lib/processes.sh
source "${SCRIPT_DIR}/lib/processes.sh"
# shellcheck source=scripts/kv_baseline/lib/lmcache_mp.sh
source "${SCRIPT_DIR}/lib/lmcache_mp.sh"
# shellcheck source=scripts/kv_baseline/lib/engine_server.sh
source "${SCRIPT_DIR}/lib/engine_server.sh"
# shellcheck source=scripts/kv_baseline/lib/requests.sh
source "${SCRIPT_DIR}/lib/requests.sh"

server_pid=""
lmcache_pid=""
trap cleanup_baseline_processes EXIT

print_run_context

start_lmcache_mp_server
wait_for_lmcache_mp_ready

start_engine_server "offload"
wait_for_engine_ready "offload"
run_phase_request "offload"

if [ "$GE_RUNTIME_BASELINE_MODE" = "restart" ]; then
  stop_engine_server "offload"
  sleep 3
  start_engine_server "reuse"
  wait_for_engine_ready "reuse"
fi

run_phase_request "reuse"

if [ "$GE_RUNTIME_KEEP_SERVER_AFTER_REUSE" != "1" ]; then
  stop_engine_server "reuse"
fi

write_baseline_summary

if [ "$GE_RUNTIME_KV_BACKEND" = "mp" ] && [ "$GE_RUNTIME_KEEP_LMCACHE_MP_AFTER_RUN" != "1" ]; then
  stop_lmcache_mp_server
fi

print_key_outputs
