#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh is deprecated." >&2
echo "Forwarding to scripts/kv_baseline/run_vllm_lmcache_mp_l2_baseline.sh." >&2
echo "Set GE_KV_BACKEND=legacy GE_ENGINE=sglang to run the old SGLang + LMCache control path." >&2
exec "${SCRIPT_DIR}/run_vllm_lmcache_mp_l2_baseline.sh" "$@"
