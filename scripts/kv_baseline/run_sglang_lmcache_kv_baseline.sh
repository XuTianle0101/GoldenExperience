#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a two-phase same-model KV offload/reuse baseline.

The default mode is the persistent path recommended by LMCache:
  1. start a standalone LMCache multiprocess server,
  2. configure its L2 adapter as a filesystem directory,
  3. start vLLM with LMCacheMPConnector,
  4. send an offload request, restart only the inference engine, then send reuse.

This keeps LMCache alive across engine restart, so the second request can retrieve KV
from the MP server/L2 disk instead of relying on an in-process cache. The older
SGLang in-process path is still available via GE_KV_BACKEND=legacy GE_ENGINE=sglang.

Usage:
  GE_MODEL_PATH=./models/Qwen/Qwen3-8B scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh [-- engine args...]

Examples:
  GE_MODEL_PATH=./models/Qwen/Qwen3-8B scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tensor-parallel-size 1
  GE_KV_BACKEND=legacy GE_ENGINE=sglang GE_MODEL_PATH=./models/Qwen/Qwen3-8B scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh -- --tp 1

Environment:
  PYTHON_BIN                         Python executable. Default: python3.
  GE_KV_BACKEND                      mp or legacy. Default: mp.
  GE_ENGINE                          vllm or sglang. Default: vllm.
  VLLM_BIN                           vLLM executable. Default: vllm.
  LMCACHE_BIN                        LMCache executable. Default: lmcache.
  GE_MODEL_PATH                      Model path or HF model id. Default: Qwen/Qwen3-8B.
  GE_MODEL_NAME                      OpenAI model name in requests. Default: GE_MODEL_PATH.
  GE_HOST                            Engine bind host. Default: 0.0.0.0.
  GE_CLIENT_HOST                     Client host. Default: 127.0.0.1.
  GE_PORT                            Engine OpenAI API port. Default: 30000.
  GE_RUN_ID                          Baseline run id. Default: UTC timestamp.
  GE_RUN_DIR                         Output directory. Default: artifacts/kv_baseline/$GE_RUN_ID.
  GE_KV_CACHE_DIR                    Persistent KV/L2 disk directory. Default: $GE_RUN_DIR/cache.
  GE_LMCACHE_CONFIG_FILE             Recorded config path. Default: $GE_RUN_DIR/lmc_config.yaml.
  GE_KV_CHUNK_SIZE                   LMCache chunk size. Default: 16.
  GE_FORCE_DISK_OFFLOAD              Require disk-backed KV evidence. Default: 1.
  GE_LMCACHE_HASH_ALGORITHM          LMCache prefix hash algorithm. Default: builtin.
  PYTHONHASHSEED                     Fixed for builtin hash stability. Default: 0.

  GE_LMCACHE_MP_HOST                 MP connector host. Default: 127.0.0.1.
  GE_LMCACHE_MP_BIND_HOST            MP server bind host. Default: GE_LMCACHE_MP_HOST.
  GE_LMCACHE_MP_PORT                 MP server port. Default: 6555.
  GE_LMCACHE_MP_HTTP_HOST            MP server HTTP host. Default: 127.0.0.1.
  GE_LMCACHE_MP_HTTP_PORT            MP server HTTP port. Default: 8080.
  GE_LMCACHE_MP_PROMETHEUS_PORT      MP server metrics port. Default: 9090.
  GE_LMCACHE_MP_L1_GB                MP server L1 memory budget. Default: 4.
  GE_LMCACHE_MP_L1_INIT_GB           MP server initial L1 allocation. Default: 1.
  GE_LMCACHE_MP_EVICTION_POLICY      MP L1 eviction policy. Default: noop.
  GE_LMCACHE_MP_L2_STORE_POLICY      MP L2 store policy. Default: skip_l1.
  GE_LMCACHE_MP_L2_ADAPTER_TYPE      MP L2 adapter type. Default: fs.
  GE_LMCACHE_MP_L2_DIR               MP L2 filesystem path. Default: GE_KV_CACHE_DIR.
  GE_LMCACHE_MP_TRANSFER_MODE        vLLM connector transfer mode. Default: auto.
  GE_LMCACHE_MP_CONNECT_HOST         Host value passed to vLLM connector. Default: GE_LMCACHE_MP_HOST.
  GE_LMCACHE_MP_LOOKUP_HASH_LOG      Enable MP lookup hash logs. Default: 1.
  GE_LMCACHE_MP_LOOKUP_HASH_LOG_DIR  Lookup hash log directory. Default: $GE_RUN_DIR/lookup_hashes.
  GE_LMCACHE_MP_EXTRA_ARGS           Extra simple whitespace-split args for `lmcache server`.

  GE_PROMPT_FILE                     Prompt manifest. Default: configs/kv_baseline_prompts.json.
  GE_PROMPT_ID                       Prompt id. Default: gsm8k_natalia_clips.
  GE_DISK_PROMPT_ID                  Generated long prompt id. Default: kv_disk_long_prefix.
  GE_DISK_PROMPT_REPEAT              Repeated context lines in generated prompt. Default: 256.
  GE_DISK_PROMPT_MAX_TOKENS          Generated prompt max_tokens. Default: 128.
  GE_MAX_TOKENS                      Override prompt max_tokens.
  GE_TEMPERATURE                     Override prompt temperature.
  GE_SERVER_START_TIMEOUT_SEC        Wait time for servers. Default: 900.
  GE_REQUEST_TIMEOUT_SEC             Request timeout. Default: 600.
  GE_AFTER_REQUEST_SLEEP_SEC         Delay after each request for cache flush/logs. Default: 10.
  GE_AFTER_WARMUP_SLEEP_SEC          Delay after discarded warmup request. Default: 1.
  GE_BASELINE_MODE                   restart or same-process. Default: restart.
  GE_ENABLE_ENGINE_METRICS           Snapshot engine /metrics. Default: 1.
  GE_INCLUDE_USAGE                   Request stream usage accounting if supported. Default: 1.
  GE_WAIT_FOR_READY_LOG              Wait for ready log. Default: 0 for vLLM, 1 for SGLang.
  GE_READY_LOG_PATTERN               Ready log regex for SGLang legacy.
  GE_WARMUP_BEFORE_MEASURE           Send a discarded warmup request. Default: 1.
  GE_WARMUP_PROMPT_ID                Warmup prompt id. Default: kv_baseline_warmup.
  GE_REQUIRE_REUSE_EVIDENCE          Fail summary if reuse evidence is absent. Default: 1 when disk is forced.
  GE_KEEP_SERVER_AFTER_REUSE         Leave final engine running after reuse. Default: 0.
  GE_KEEP_LMCACHE_MP_AFTER_RUN       Leave MP server running. Default: GE_KEEP_SERVER_AFTER_REUSE.
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
raw_engine_args=("$@")

python_bin="${PYTHON_BIN:-python3}"
kv_backend="${GE_KV_BACKEND:-mp}"
engine="${GE_ENGINE:-vllm}"
vllm_bin="${VLLM_BIN:-vllm}"
lmcache_bin="${LMCACHE_BIN:-lmcache}"
model_path="${GE_MODEL_PATH:-Qwen/Qwen3-8B}"
model_name="${GE_MODEL_NAME:-$model_path}"
host="${GE_HOST:-0.0.0.0}"
client_host="${GE_CLIENT_HOST:-127.0.0.1}"
port="${GE_PORT:-30000}"
run_id="${GE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${GE_RUN_DIR:-artifacts/kv_baseline/${run_id}}"
cache_dir="${GE_KV_CACHE_DIR:-${run_dir}/cache}"
log_dir="${run_dir}/logs"
request_dir="${run_dir}/requests"
metrics_dir="${run_dir}/metrics"
config_file="${GE_LMCACHE_CONFIG_FILE:-${run_dir}/lmc_config.yaml}"
prompt_file_was_set=0
if [ "${GE_PROMPT_FILE+x}" = "x" ]; then
  prompt_file_was_set=1
fi
prompt_file="${GE_PROMPT_FILE:-configs/kv_baseline_prompts.json}"
prompt_id="${GE_PROMPT_ID:-gsm8k_natalia_clips}"
chunk_size="${GE_KV_CHUNK_SIZE:-16}"
force_disk_offload="${GE_FORCE_DISK_OFFLOAD:-1}"
local_cpu_enabled="${GE_KV_LOCAL_CPU_ENABLED:-true}"
local_cpu_gb="${GE_KV_LOCAL_CPU_GB:-10}"
if [ "$force_disk_offload" = "1" ]; then
  local_cpu_enabled="${GE_KV_LOCAL_CPU_ENABLED:-false}"
  local_cpu_gb="${GE_KV_LOCAL_CPU_GB:-0}"
fi
local_disk_gb="${GE_KV_LOCAL_DISK_GB:-100}"
hash_algorithm="${GE_LMCACHE_HASH_ALGORITHM:-builtin}"
disk_prompt_id="${GE_DISK_PROMPT_ID:-kv_disk_long_prefix}"
disk_prompt_repeat="${GE_DISK_PROMPT_REPEAT:-256}"
disk_prompt_max_tokens="${GE_DISK_PROMPT_MAX_TOKENS:-128}"
start_timeout="${GE_SERVER_START_TIMEOUT_SEC:-900}"
request_timeout="${GE_REQUEST_TIMEOUT_SEC:-600}"
after_request_sleep="${GE_AFTER_REQUEST_SLEEP_SEC:-10}"
after_warmup_sleep="${GE_AFTER_WARMUP_SLEEP_SEC:-1}"
baseline_mode="${GE_BASELINE_MODE:-restart}"
disable_radix_cache="${GE_DISABLE_RADIX_CACHE:-1}"
enable_metrics="${GE_ENABLE_ENGINE_METRICS:-${GE_ENABLE_SGLANG_METRICS:-1}}"
save_decode_cache="${GE_SAVE_DECODE_CACHE:-false}"
include_usage="${GE_INCLUDE_USAGE:-1}"
ready_log_pattern="${GE_READY_LOG_PATTERN:-The server is fired up and ready to roll!}"
warmup_before_measure="${GE_WARMUP_BEFORE_MEASURE:-1}"
warmup_prompt_id="${GE_WARMUP_PROMPT_ID:-kv_baseline_warmup}"
require_reuse_evidence="${GE_REQUIRE_REUSE_EVIDENCE:-0}"
if [ "$force_disk_offload" = "1" ]; then
  require_reuse_evidence="${GE_REQUIRE_REUSE_EVIDENCE:-1}"
fi
keep_server_after_reuse="${GE_KEEP_SERVER_AFTER_REUSE:-0}"
keep_lmcache_mp_after_run="${GE_KEEP_LMCACHE_MP_AFTER_RUN:-$keep_server_after_reuse}"
base_url="http://${client_host}:${port}"
server_pid=""
lmcache_pid=""

case "$kv_backend" in
  mp|legacy) ;;
  *)
    echo "GE_KV_BACKEND must be mp or legacy, got: ${kv_backend}" >&2
    exit 2
    ;;
esac

case "$engine" in
  vllm|sglang) ;;
  *)
    echo "GE_ENGINE must be vllm or sglang, got: ${engine}" >&2
    exit 2
    ;;
esac

if [ "$kv_backend" = "mp" ] && [ "$engine" != "vllm" ]; then
  echo "GE_KV_BACKEND=mp currently requires GE_ENGINE=vllm for LMCacheMPConnector." >&2
  echo "Use GE_KV_BACKEND=legacy GE_ENGINE=sglang for the older in-process SGLang path." >&2
  exit 2
fi

if [ "$kv_backend" = "legacy" ] && [ "$engine" != "sglang" ]; then
  echo "GE_KV_BACKEND=legacy only supports GE_ENGINE=sglang." >&2
  exit 2
fi

if [ "${GE_WAIT_FOR_READY_LOG+x}" = "x" ]; then
  wait_for_ready_log="$GE_WAIT_FOR_READY_LOG"
elif [ "$engine" = "sglang" ]; then
  wait_for_ready_log="1"
else
  wait_for_ready_log="0"
fi

case "$baseline_mode" in
  restart|same-process) ;;
  *)
    echo "GE_BASELINE_MODE must be restart or same-process, got: ${baseline_mode}" >&2
    exit 2
    ;;
esac

if [ "$force_disk_offload" = "1" ] && [ "$prompt_file_was_set" = "0" ]; then
  prompt_file="${run_dir}/disk_prompt.json"
  prompt_id="$disk_prompt_id"
fi

lmcache_mp_host="${GE_LMCACHE_MP_HOST:-127.0.0.1}"
lmcache_mp_bind_host="${GE_LMCACHE_MP_BIND_HOST:-$lmcache_mp_host}"
lmcache_mp_connect_host="${GE_LMCACHE_MP_CONNECT_HOST:-$lmcache_mp_host}"
lmcache_mp_port="${GE_LMCACHE_MP_PORT:-6555}"
lmcache_mp_http_host="${GE_LMCACHE_MP_HTTP_HOST:-127.0.0.1}"
lmcache_mp_http_port="${GE_LMCACHE_MP_HTTP_PORT:-8080}"
lmcache_mp_prometheus_port="${GE_LMCACHE_MP_PROMETHEUS_PORT:-9090}"
lmcache_mp_l1_gb="${GE_LMCACHE_MP_L1_GB:-4}"
lmcache_mp_l1_init_gb="${GE_LMCACHE_MP_L1_INIT_GB:-1}"
lmcache_mp_eviction_policy="${GE_LMCACHE_MP_EVICTION_POLICY:-noop}"
lmcache_mp_l2_store_policy="${GE_LMCACHE_MP_L2_STORE_POLICY:-skip_l1}"
lmcache_mp_l2_adapter_type="${GE_LMCACHE_MP_L2_ADAPTER_TYPE:-fs}"
lmcache_mp_l2_dir="${GE_LMCACHE_MP_L2_DIR:-$cache_dir}"
lmcache_mp_l2_use_odirect="${GE_LMCACHE_MP_L2_USE_ODIRECT:-false}"
lmcache_mp_l2_num_workers="${GE_LMCACHE_MP_L2_NUM_WORKERS:-}"
lmcache_mp_transfer_mode="${GE_LMCACHE_MP_TRANSFER_MODE:-auto}"
lmcache_mp_lookup_hash_log="${GE_LMCACHE_MP_LOOKUP_HASH_LOG:-1}"
lmcache_mp_lookup_hash_log_dir="${GE_LMCACHE_MP_LOOKUP_HASH_LOG_DIR:-${run_dir}/lookup_hashes}"
lmcache_log_level="${LMCACHE_LOG_LEVEL:-DEBUG}"

case "$cache_dir" in
  /*) cache_dir_abs="$cache_dir" ;;
  *) cache_dir_abs="${PWD}/${cache_dir}" ;;
esac

case "$lmcache_mp_l2_dir" in
  /*) lmcache_mp_l2_dir_abs="$lmcache_mp_l2_dir" ;;
  *) lmcache_mp_l2_dir_abs="${PWD}/${lmcache_mp_l2_dir}" ;;
esac

if [ "$kv_backend" = "mp" ]; then
  kv_cache_dir_abs="$lmcache_mp_l2_dir_abs"
else
  kv_cache_dir_abs="$cache_dir_abs"
fi

mkdir -p "$cache_dir" "$lmcache_mp_l2_dir" "$log_dir" "$request_dir" "$metrics_dir" "$(dirname "$config_file")"
if [ "$lmcache_mp_lookup_hash_log" = "1" ]; then
  mkdir -p "$lmcache_mp_lookup_hash_log_dir"
fi

l2_adapter_json="$(
  L2_TYPE="$lmcache_mp_l2_adapter_type" \
  L2_PATH="$lmcache_mp_l2_dir_abs" \
  L2_USE_ODIRECT="$lmcache_mp_l2_use_odirect" \
  L2_NUM_WORKERS="$lmcache_mp_l2_num_workers" \
  "$python_bin" - <<'PY'
import json
import os

adapter = {
    "type": os.environ["L2_TYPE"],
    "base_path": os.environ["L2_PATH"],
}
if os.environ.get("L2_USE_ODIRECT", "").lower() in {"1", "true", "yes"}:
    adapter["use_odirect"] = True
workers = os.environ.get("L2_NUM_WORKERS", "")
if workers:
    adapter["num_thread"] = int(workers)
print(json.dumps(adapter, separators=(",", ":")))
PY
)"

vllm_kv_transfer_config="$(
  MP_HOST="$lmcache_mp_connect_host" \
  MP_PORT="$lmcache_mp_port" \
  MP_TRANSFER_MODE="$lmcache_mp_transfer_mode" \
  "$python_bin" - <<'PY'
import json
import os

config = {
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.host": os.environ["MP_HOST"],
        "lmcache.mp.port": int(os.environ["MP_PORT"]),
        "lmcache.mp.mp_transfer_mode": os.environ["MP_TRANSFER_MODE"],
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
)"

if [ "$force_disk_offload" = "1" ] && [ "$prompt_file_was_set" = "0" ]; then
  "$python_bin" - <<PY
import json
from pathlib import Path

repeat = int("${disk_prompt_repeat}")
lines = []
for i in range(repeat):
    lines.append(
        f"Cache persistence calibration line {i:04d}: "
        "GoldenExperience disk-offload validation keeps this deterministic prefix stable. "
        "Natalia April clips equals 48 and May clips equals half of April. "
        "This repeated text exists only to make the prompt large enough for KV caching."
    )

user_content = (
    "Read the following deterministic reference block. Do not summarize the block.\\n\\n"
    + "\\n".join(lines)
    + "\\n\\nQuestion: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April "
    "and May? End with exactly one line formatted as: Final answer: <number>."
)

manifest = {
    "default_prompt_id": "${disk_prompt_id}",
    "prompts": [
        {
            "id": "${disk_prompt_id}",
            "dataset": "synthetic-long-prefix",
            "split": "disk-offload",
            "source": "generated by run_sglang_lmcache_kv_baseline.sh for disk KV reuse",
            "expected_final_answer": "72",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful math solver. Use the reference only as context.",
                },
                {"role": "user", "content": user_content},
            ],
            "generation": {"max_tokens": int("${disk_prompt_max_tokens}"), "temperature": 0},
        },
        {
            "id": "${warmup_prompt_id}",
            "dataset": "synthetic",
            "split": "warmup",
            "source": "short unrelated warmup prompt for engine stabilization",
            "expected_final_answer": "OK",
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            "generation": {"max_tokens": 8, "temperature": 0},
        },
    ],
}
Path("${prompt_file}").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY
fi

if [ "$kv_backend" = "mp" ]; then
  cat > "$config_file" <<EOF
# Generated by scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh
kv_backend: mp
engine: vllm
chunk_size: ${chunk_size}
hash_algorithm: ${hash_algorithm}
lmcache_mp:
  host: ${lmcache_mp_host}
  bind_host: ${lmcache_mp_bind_host}
  port: ${lmcache_mp_port}
  http_host: ${lmcache_mp_http_host}
  http_port: ${lmcache_mp_http_port}
  prometheus_port: ${lmcache_mp_prometheus_port}
  l1_size_gb: ${lmcache_mp_l1_gb}
  l1_init_size_gb: ${lmcache_mp_l1_init_gb}
  eviction_policy: ${lmcache_mp_eviction_policy}
  l2_store_policy: ${lmcache_mp_l2_store_policy}
  l2_adapter_json: '${l2_adapter_json}'
vllm:
  kv_transfer_config: '${vllm_kv_transfer_config}'
EOF
else
  cat > "$config_file" <<EOF
# Generated by scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh
kv_backend: legacy
engine: sglang
chunk_size: ${chunk_size}
local_cpu: ${local_cpu_enabled}
max_local_cpu_size: ${local_cpu_gb}
local_disk: "file://${cache_dir_abs}"
max_local_disk_size: ${local_disk_gb}
use_layerwise: true
save_decode_cache: ${save_decode_cache}
pre_caching_hash_algorithm: ${hash_algorithm}
EOF
fi

L2_ADAPTER_JSON="$l2_adapter_json" VLLM_KV_TRANSFER_CONFIG="$vllm_kv_transfer_config" "$python_bin" - <<PY
import json
import os
from pathlib import Path
metadata = {
    "run_id": "${run_id}",
    "run_dir": "${run_dir}",
    "mode": "${baseline_mode}",
    "kv_backend": "${kv_backend}",
    "engine": "${engine}",
    "model_path": "${model_path}",
    "model_name": "${model_name}",
    "base_url": "${base_url}",
    "prompt_file": "${prompt_file}",
    "prompt_id": "${prompt_id}",
    "lmcache_config_file": "${config_file}",
    "kv_cache_dir": "${kv_cache_dir_abs}",
    "chunk_size": int("${chunk_size}"),
    "force_disk_offload": "${force_disk_offload}" == "1",
    "local_cpu_enabled": "${local_cpu_enabled}" == "true",
    "local_cpu_gb": float("${local_cpu_gb}"),
    "local_disk_gb": float("${local_disk_gb}"),
    "hash_algorithm": "${hash_algorithm}",
    "disk_prompt_id": "${disk_prompt_id}",
    "disk_prompt_repeat": int("${disk_prompt_repeat}"),
    "disk_prompt_max_tokens": int("${disk_prompt_max_tokens}"),
    "generated_disk_prompt": "${force_disk_offload}" == "1" and "${prompt_file_was_set}" == "0",
    "lmcache_mp": {
        "enabled": "${kv_backend}" == "mp",
        "host": "${lmcache_mp_host}",
        "bind_host": "${lmcache_mp_bind_host}",
        "connect_host": "${lmcache_mp_connect_host}",
        "port": int("${lmcache_mp_port}"),
        "http_host": "${lmcache_mp_http_host}",
        "http_port": int("${lmcache_mp_http_port}"),
        "prometheus_port": int("${lmcache_mp_prometheus_port}"),
        "l1_size_gb": float("${lmcache_mp_l1_gb}"),
        "l1_init_size_gb": float("${lmcache_mp_l1_init_gb}"),
        "eviction_policy": "${lmcache_mp_eviction_policy}",
        "l2_store_policy": "${lmcache_mp_l2_store_policy}",
        "l2_adapter_json": os.environ["L2_ADAPTER_JSON"],
        "l2_dir": "${lmcache_mp_l2_dir_abs}",
        "transfer_mode": "${lmcache_mp_transfer_mode}",
        "lookup_hash_log": "${lmcache_mp_lookup_hash_log}" == "1",
        "lookup_hash_log_dir": "${lmcache_mp_lookup_hash_log_dir}",
    },
    "vllm_kv_transfer_config": os.environ["VLLM_KV_TRANSFER_CONFIG"],
    "disable_radix_cache": "${disable_radix_cache}" == "1",
    "enable_engine_metrics": "${enable_metrics}" == "1",
    "include_usage": "${include_usage}" == "1",
    "wait_for_ready_log": "${wait_for_ready_log}" == "1",
    "ready_log_pattern": "${ready_log_pattern}",
    "warmup_before_measure": "${warmup_before_measure}" == "1",
    "warmup_prompt_id": "${warmup_prompt_id}",
    "require_reuse_evidence": "${require_reuse_evidence}" == "1",
    "created_unix": __import__("time").time(),
}
Path("${run_dir}/metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

ensure_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command '${command_name}' was not found in PATH." >&2
    exit 127
  fi
}

normalize_engine_args() {
  extra_engine_args=()
  local arg
  local value
  while [ "$#" -gt 0 ]; do
    arg="$1"
    case "${engine}:${arg}" in
      vllm:--tp)
        if [ "$#" -lt 2 ]; then
          echo "--tp requires a value" >&2
          exit 2
        fi
        value="$2"
        extra_engine_args+=(--tensor-parallel-size "$value")
        shift 2
        ;;
      vllm:--disable-radix-cache|vllm:--enable-metrics)
        shift
        ;;
      sglang:--tensor-parallel-size)
        if [ "$#" -lt 2 ]; then
          echo "--tensor-parallel-size requires a value" >&2
          exit 2
        fi
        value="$2"
        extra_engine_args+=(--tp "$value")
        shift 2
        ;;
      *)
        extra_engine_args+=("$arg")
        shift
        ;;
    esac
  done
}

append_default_engine_args() {
  if [ "$engine" != "sglang" ]; then
    return
  fi
  if [ "$disable_radix_cache" = "1" ]; then
    extra_engine_args+=(--disable-radix-cache)
  fi
  if [ "$enable_metrics" = "1" ]; then
    extra_engine_args+=(--enable-metrics)
  fi
}

start_lmcache_mp_server() {
  if [ "$kv_backend" != "mp" ]; then
    return
  fi

  ensure_command "$lmcache_bin"
  local log_file="${log_dir}/lmcache_mp_server.log"
  local -a mp_args=(
    server
    --host "$lmcache_mp_bind_host"
    --port "$lmcache_mp_port"
    --http-host "$lmcache_mp_http_host"
    --http-port "$lmcache_mp_http_port"
    --prometheus-port "$lmcache_mp_prometheus_port"
    --chunk-size "$chunk_size"
    --hash-algorithm "$hash_algorithm"
    --l1-size-gb "$lmcache_mp_l1_gb"
    --l1-init-size-gb "$lmcache_mp_l1_init_gb"
    --eviction-policy "$lmcache_mp_eviction_policy"
    --l2-store-policy "$lmcache_mp_l2_store_policy"
    --l2-adapter "$l2_adapter_json"
  )
  if [ "$lmcache_mp_lookup_hash_log" = "1" ]; then
    mp_args+=(--lookup-hash-log-dir "$lmcache_mp_lookup_hash_log_dir")
  fi
  if [ -n "${GE_LMCACHE_MP_EXTRA_ARGS:-}" ]; then
    local -a extra_mp_args
    read -r -a extra_mp_args <<< "$GE_LMCACHE_MP_EXTRA_ARGS"
    mp_args+=("${extra_mp_args[@]}")
  fi

  echo "Starting LMCache MP server on ${lmcache_mp_bind_host}:${lmcache_mp_port}; log: ${log_file}"
  (
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export LMCACHE_LOG_LEVEL="$lmcache_log_level"
    exec "$lmcache_bin" "${mp_args[@]}"
  ) >"$log_file" 2>&1 &
  lmcache_pid="$!"
  echo "$lmcache_pid" > "${run_dir}/lmcache_mp.pid"
}

wait_for_lmcache_mp_ready() {
  if [ "$kv_backend" != "mp" ]; then
    return
  fi

  local log_file="${log_dir}/lmcache_mp_server.log"
  if ! MP_HOST="$lmcache_mp_host" MP_PORT="$lmcache_mp_port" TIMEOUT="$start_timeout" "$python_bin" - <<'PY'
import os
import socket
import sys
import time

host = os.environ["MP_HOST"]
if host.startswith("tcp://"):
    host = host[len("tcp://"):]
port = int(os.environ["MP_PORT"])
deadline = time.time() + float(os.environ["TIMEOUT"])
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(1)
sys.exit(1)
PY
  then
    echo "Timed out waiting for LMCache MP server; last log lines:" >&2
    tail -n 120 "$log_file" >&2 || true
    exit 1
  fi

  if ! kill -0 "$lmcache_pid" >/dev/null 2>&1; then
    echo "LMCache MP server exited after port became reachable; last log lines:" >&2
    tail -n 120 "$log_file" >&2 || true
    exit 1
  fi

  echo "LMCache MP server is ready on ${lmcache_mp_host}:${lmcache_mp_port}"
}

stop_lmcache_mp_server() {
  if [ -z "${lmcache_pid:-}" ]; then
    return
  fi
  if ! kill -0 "$lmcache_pid" >/dev/null 2>&1; then
    lmcache_pid=""
    return
  fi
  echo "Stopping LMCache MP server pid=${lmcache_pid}"
  kill "$lmcache_pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$lmcache_pid" >/dev/null 2>&1; then
      lmcache_pid=""
      return
    fi
    sleep 1
  done
  echo "LMCache MP server pid=${lmcache_pid} did not exit after SIGTERM; sending SIGKILL" >&2
  kill -9 "$lmcache_pid" >/dev/null 2>&1 || true
  lmcache_pid=""
}

start_server() {
  local phase="$1"
  local log_file="${log_dir}/${phase}_server.log"
  echo "Starting ${phase} ${engine} server on ${base_url}; log: ${log_file}"

  if [ "$engine" = "vllm" ]; then
    ensure_command "$vllm_bin"
    (
      export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
      export LMCACHE_LOG_LEVEL="$lmcache_log_level"
      exec "$vllm_bin" serve "$model_path" \
        --host "$host" \
        --port "$port" \
        --served-model-name "$model_name" \
        --kv-transfer-config "$vllm_kv_transfer_config" \
        "${extra_engine_args[@]}"
    ) >"$log_file" 2>&1 &
  else
    (
      export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
      export GE_MODEL_PATH="$model_path"
      export GE_HOST="$host"
      export GE_PORT="$port"
      export GE_LMCACHE_CONFIG_FILE="$config_file"
      export GE_OVERWRITE_LMCACHE_CONFIG=0
      export GE_ENABLE_CROSS_MODEL_REUSE=0
      export GE_SGLANG_MODEL_ID="$model_path"
      export LMCACHE_CONFIG_FILE="$config_file"
      export LMCACHE_LOG_LEVEL="$lmcache_log_level"
      exec scripts/start_sglang_lmcache.sh "${extra_engine_args[@]}"
    ) >"$log_file" 2>&1 &
  fi

  server_pid="$!"
  echo "$server_pid" > "${run_dir}/${phase}.pid"
}

wait_for_ready_log_pattern() {
  local phase="$1"
  local log_file="${log_dir}/${phase}_server.log"
  local deadline=$((SECONDS + start_timeout))
  if [ "$wait_for_ready_log" != "1" ]; then
    return
  fi
  while [ "$SECONDS" -lt "$deadline" ]; do
    if grep -E -q -- "$ready_log_pattern" "$log_file" 2>/dev/null; then
      echo "${phase} server emitted ready log: ${ready_log_pattern}"
      return
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "${phase} server exited before ready log; last log lines:" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "Timed out waiting for ${phase} ready log '${ready_log_pattern}'; last log lines:" >&2
  tail -n 80 "$log_file" >&2 || true
  exit 1
}

wait_for_ready() {
  local phase="$1"
  local log_file="${log_dir}/${phase}_server.log"
  local deadline=$((SECONDS + start_timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "${phase} server exited before becoming ready; last log lines:" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    if "$python_bin" scripts/kv_baseline/kv_baseline_client.py wait \
      --base-url "$base_url" \
      --timeout 2 \
      --interval 1 >/dev/null 2>&1; then
      wait_for_ready_log_pattern "$phase"
      echo "${phase} server is ready at ${base_url}"
      return
    fi
    sleep 2
  done
  echo "Timed out waiting for ${phase} server; last log lines:" >&2
  tail -n 80 "$log_file" >&2 || true
  exit 1
}

stop_server() {
  local phase="${1:-server}"
  if [ -z "${server_pid:-}" ]; then
    return
  fi
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    server_pid=""
    return
  fi
  echo "Stopping ${phase} server pid=${server_pid}"
  kill "$server_pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      server_pid=""
      return
    fi
    sleep 1
  done
  echo "Server pid=${server_pid} did not exit after SIGTERM; sending SIGKILL" >&2
  kill -9 "$server_pid" >/dev/null 2>&1 || true
  server_pid=""
}

cleanup() {
  if [ "${keep_server_after_reuse}" != "1" ]; then
    stop_server "active"
  fi
  if [ "$kv_backend" = "mp" ] && [ "${keep_lmcache_mp_after_run}" != "1" ]; then
    stop_lmcache_mp_server
  fi
}
trap cleanup EXIT

mark_lmcache_log_start() {
  if [ "$kv_backend" != "mp" ]; then
    return
  fi
  local phase="$1"
  local source="${log_dir}/lmcache_mp_server.log"
  local offset="0"
  if [ -f "$source" ]; then
    offset="$(wc -c < "$source" | tr -d '[:space:]')"
  fi
  echo "$offset" > "${run_dir}/${phase}_lmcache_log_offset.txt"
}

capture_lmcache_log_delta() {
  if [ "$kv_backend" != "mp" ]; then
    return
  fi
  local phase="$1"
  local source="${log_dir}/lmcache_mp_server.log"
  local offset_path="${run_dir}/${phase}_lmcache_log_offset.txt"
  local dest="${log_dir}/${phase}_lmcache_mp_server.log"
  if [ ! -f "$source" ]; then
    return
  fi
  LOG_SOURCE="$source" LOG_DEST="$dest" LOG_OFFSET_PATH="$offset_path" "$python_bin" - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["LOG_SOURCE"])
dest = Path(os.environ["LOG_DEST"])
offset_path = Path(os.environ["LOG_OFFSET_PATH"])
offset = 0
if offset_path.exists():
    text = offset_path.read_text(encoding="utf-8", errors="replace").strip()
    offset = int(text or "0")
with source.open("rb") as handle:
    handle.seek(max(0, offset))
    data = handle.read()
dest.write_bytes(data)
PY
}

send_request() {
  local phase="$1"
  local prompt_id_for_request="$2"
  local request_output="$3"
  local -a request_args=(
    request
    --base-url "$base_url"
    --model "$model_name"
    --prompt-file "$prompt_file"
    --prompt-id "$prompt_id_for_request"
    --phase "$phase"
    --output "$request_output"
    --timeout "$request_timeout"
  )
  if [ -n "${GE_MAX_TOKENS:-}" ]; then
    request_args+=(--max-tokens "$GE_MAX_TOKENS")
  fi
  if [ -n "${GE_TEMPERATURE:-}" ]; then
    request_args+=(--temperature "$GE_TEMPERATURE")
  fi
  if [ "$include_usage" != "1" ]; then
    request_args+=(--no-include-usage)
  fi
  "$python_bin" scripts/kv_baseline/kv_baseline_client.py "${request_args[@]}"
}

run_phase_request() {
  local phase="$1"
  local request_output="${request_dir}/${phase}.json"
  if [ "$warmup_before_measure" = "1" ]; then
    echo "Sending ${phase} warmup request (${warmup_prompt_id}); output is excluded from timing deltas"
    send_request "${phase}_warmup" "$warmup_prompt_id" "${request_dir}/${phase}_warmup.json"
    sleep "$after_warmup_sleep"
  fi
  mark_lmcache_log_start "$phase"
  echo "Sending ${phase} request"
  send_request "$phase" "$prompt_id" "$request_output"

  if [ "$enable_metrics" = "1" ]; then
    "$python_bin" scripts/kv_baseline/kv_baseline_client.py fetch-metrics \
      --base-url "$base_url" \
      --output "${metrics_dir}/${phase}.prom" \
      --allow-missing
  fi

  sleep "$after_request_sleep"
  capture_lmcache_log_delta "$phase"
}

normalize_engine_args "${raw_engine_args[@]}"
append_default_engine_args

echo "KV baseline run directory: ${run_dir}"
echo "KV backend: ${kv_backend}; engine: ${engine}"
echo "Recorded config: ${config_file}"
echo "Persistent KV cache dir: ${kv_cache_dir_abs}"
if [ "$kv_backend" = "mp" ]; then
  echo "LMCache MP: ${lmcache_mp_bind_host}:${lmcache_mp_port}; L2 adapter: ${l2_adapter_json}"
  echo "vLLM KV transfer config: ${vllm_kv_transfer_config}"
fi
echo "Force disk offload: ${force_disk_offload}"
echo "Prompt: ${prompt_file}#${prompt_id}"
if [ "$force_disk_offload" = "1" ] && [ "$prompt_file_was_set" = "0" ]; then
  echo "Generated disk prompt repeat: ${disk_prompt_repeat}; max_tokens=${disk_prompt_max_tokens}"
fi
echo "Extra engine args: ${extra_engine_args[*]:-(none)}"

start_lmcache_mp_server
wait_for_lmcache_mp_ready

start_server "offload"
wait_for_ready "offload"
run_phase_request "offload"

if [ "$baseline_mode" = "restart" ]; then
  stop_server "offload"
  sleep 3
  start_server "reuse"
  wait_for_ready "reuse"
fi

run_phase_request "reuse"

if [ "${keep_server_after_reuse}" != "1" ]; then
  stop_server "reuse"
fi

summary_args=(
  summarize
  --run-dir "$run_dir"
  --output "${run_dir}/summary.json"
)
if [ "$require_reuse_evidence" = "1" ]; then
  summary_args+=(--require-reuse-evidence)
fi
if [ "$force_disk_offload" = "1" ]; then
  summary_args+=(--require-disk-offload)
fi
"$python_bin" scripts/kv_baseline/kv_baseline_client.py "${summary_args[@]}"

if [ "$kv_backend" = "mp" ] && [ "${keep_lmcache_mp_after_run}" != "1" ]; then
  stop_lmcache_mp_server
fi

echo "Done. Key outputs:"
echo "  ${run_dir}/metadata.json"
echo "  ${run_dir}/lmc_config.yaml"
echo "  ${run_dir}/requests/offload.json"
echo "  ${run_dir}/requests/reuse.json"
echo "  ${run_dir}/summary.json"
