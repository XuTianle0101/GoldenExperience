#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a two-phase same-model KV offload/reuse baseline with SGLang + LMCache.

The default mode starts SGLang, sends one deterministic prompt to populate/offload KV,
stops the server, starts a fresh SGLang process with the same LMCache disk directory,
and sends the same prompt again. This isolates LMCache reuse from SGLang's in-process
radix cache and records request timings plus server logs under artifacts/kv_baseline/.

Usage:
  GE_MODEL_PATH=Qwen/Qwen3-8B scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh [-- extra sglang args...]

Environment:
  PYTHON_BIN                         Python executable. Default: python3.
  GE_MODEL_PATH                      Model path or HF model id. Default: Qwen/Qwen3-8B.
  GE_MODEL_NAME                      OpenAI model name in requests. Default: GE_MODEL_PATH.
  GE_HOST                            SGLang bind host. Default: 0.0.0.0.
  GE_CLIENT_HOST                     Client host. Default: 127.0.0.1.
  GE_PORT                            SGLang port. Default: 30000.
  GE_RUN_ID                          Baseline run id. Default: UTC timestamp.
  GE_RUN_DIR                         Output directory. Default: artifacts/kv_baseline/$GE_RUN_ID.
  GE_KV_CACHE_DIR                    Persistent LMCache disk directory. Default: $GE_RUN_DIR/cache.
  GE_LMCACHE_CONFIG_FILE             LMCache config path. Default: $GE_RUN_DIR/lmc_config.yaml.
  GE_KV_CHUNK_SIZE                   Small chunk size for this short prompt. Default: 16.
  GE_FORCE_DISK_OFFLOAD              Force disk-backed KV baseline. Default: 1.
  GE_KV_LOCAL_CPU_ENABLED            LMCache CPU tier enabled. Default: false when force disk, true otherwise.
  GE_KV_LOCAL_CPU_GB                 LMCache CPU budget. Default: 0 when force disk, 10 otherwise.
  GE_KV_LOCAL_DISK_GB                LMCache disk budget. Default: 100.
  GE_LMCACHE_HASH_ALGORITHM          LMCache prefix hash algorithm. Default: builtin.
  PYTHONHASHSEED                     Fixed for builtin hash stability. Default: 0.
  GE_PROMPT_FILE                     Prompt manifest. Default: configs/kv_baseline_prompts.json.
  GE_PROMPT_ID                       Prompt id. Default: gsm8k_natalia_clips.
  GE_DISK_PROMPT_ID                  Generated long prompt id for force-disk mode. Default: kv_disk_long_prefix.
  GE_DISK_PROMPT_REPEAT              Number of repeated context lines in generated prompt. Default: 512.
  GE_DISK_PROMPT_MAX_TOKENS          Generated prompt max_tokens. Default: 128.
  GE_MAX_TOKENS                      Override prompt max_tokens.
  GE_TEMPERATURE                     Override prompt temperature.
  GE_SERVER_START_TIMEOUT_SEC        Wait time for model server. Default: 900.
  GE_REQUEST_TIMEOUT_SEC             Request timeout. Default: 600.
  GE_AFTER_REQUEST_SLEEP_SEC         Delay after each request for cache flush/logs. Default: 5.
  GE_AFTER_WARMUP_SLEEP_SEC          Delay after discarded warmup request. Default: 1.
  GE_BASELINE_MODE                   restart or same-process. Default: restart.
  GE_DISABLE_RADIX_CACHE             Add --disable-radix-cache when supported. Default: 1.
  GE_ENABLE_SGLANG_METRICS           Add --enable-metrics and snapshot /metrics. Default: 1.
  GE_SAVE_DECODE_CACHE               Write save_decode_cache to LMCache config. Default: false.
  GE_INCLUDE_USAGE                   Request stream usage accounting if supported. Default: 1.
  GE_WAIT_FOR_READY_LOG              Wait for SGLang's ready log before measuring. Default: 1.
  GE_READY_LOG_PATTERN               Ready log regex. Default: The server is fired up and ready to roll!
  GE_WARMUP_BEFORE_MEASURE           Send a discarded warmup request before measured request. Default: 1.
  GE_WARMUP_PROMPT_ID                Warmup prompt id. Default: kv_baseline_warmup.
  GE_REQUIRE_REUSE_EVIDENCE          Fail summary if reuse evidence is absent. Default: 0.
  GE_KEEP_SERVER_AFTER_REUSE         Leave final server running after reuse phase. Default: 0.
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
extra_sglang_args=("$@")

python_bin="${PYTHON_BIN:-python3}"
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
disk_prompt_repeat="${GE_DISK_PROMPT_REPEAT:-512}"
disk_prompt_max_tokens="${GE_DISK_PROMPT_MAX_TOKENS:-128}"
start_timeout="${GE_SERVER_START_TIMEOUT_SEC:-900}"
request_timeout="${GE_REQUEST_TIMEOUT_SEC:-600}"
after_request_sleep="${GE_AFTER_REQUEST_SLEEP_SEC:-5}"
after_warmup_sleep="${GE_AFTER_WARMUP_SLEEP_SEC:-1}"
baseline_mode="${GE_BASELINE_MODE:-restart}"
disable_radix_cache="${GE_DISABLE_RADIX_CACHE:-1}"
enable_metrics="${GE_ENABLE_SGLANG_METRICS:-1}"
save_decode_cache="${GE_SAVE_DECODE_CACHE:-false}"
include_usage="${GE_INCLUDE_USAGE:-1}"
wait_for_ready_log="${GE_WAIT_FOR_READY_LOG:-1}"
ready_log_pattern="${GE_READY_LOG_PATTERN:-The server is fired up and ready to roll!}"
warmup_before_measure="${GE_WARMUP_BEFORE_MEASURE:-1}"
warmup_prompt_id="${GE_WARMUP_PROMPT_ID:-kv_baseline_warmup}"
require_reuse_evidence="${GE_REQUIRE_REUSE_EVIDENCE:-0}"
if [ "$force_disk_offload" = "1" ]; then
  require_reuse_evidence="${GE_REQUIRE_REUSE_EVIDENCE:-1}"
fi
keep_server_after_reuse="${GE_KEEP_SERVER_AFTER_REUSE:-0}"
base_url="http://${client_host}:${port}"
server_pid=""

if [ "$force_disk_offload" = "1" ] && [ "$prompt_file_was_set" = "0" ]; then
  prompt_file="${run_dir}/disk_prompt.json"
  prompt_id="$disk_prompt_id"
fi

case "$cache_dir" in
  /*) cache_dir_abs="$cache_dir" ;;
  *) cache_dir_abs="${PWD}/${cache_dir}" ;;
esac

case "$baseline_mode" in
  restart|same-process) ;;
  *)
    echo "GE_BASELINE_MODE must be restart or same-process, got: ${baseline_mode}" >&2
    exit 2
    ;;
esac

mkdir -p "$cache_dir" "$log_dir" "$request_dir" "$metrics_dir" "$(dirname "$config_file")"

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

cat > "$config_file" <<EOF
# Generated by scripts/kv_baseline/run_sglang_lmcache_kv_baseline.sh
chunk_size: ${chunk_size}
local_cpu: ${local_cpu_enabled}
max_local_cpu_size: ${local_cpu_gb}
local_disk: "file://${cache_dir_abs}"
max_local_disk_size: ${local_disk_gb}
use_layerwise: true
save_decode_cache: ${save_decode_cache}
pre_caching_hash_algorithm: ${hash_algorithm}
EOF

"$python_bin" - <<PY
import json
from pathlib import Path
metadata = {
    "run_id": "${run_id}",
    "run_dir": "${run_dir}",
    "mode": "${baseline_mode}",
    "model_path": "${model_path}",
    "model_name": "${model_name}",
    "base_url": "${base_url}",
    "prompt_file": "${prompt_file}",
    "prompt_id": "${prompt_id}",
    "lmcache_config_file": "${config_file}",
    "kv_cache_dir": "${cache_dir_abs}",
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
    "disable_radix_cache": "${disable_radix_cache}" == "1",
    "enable_sglang_metrics": "${enable_metrics}" == "1",
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

append_default_sglang_args() {
  if [ "$disable_radix_cache" = "1" ]; then
    extra_sglang_args+=(--disable-radix-cache)
  fi
  if [ "$enable_metrics" = "1" ]; then
    extra_sglang_args+=(--enable-metrics)
  fi
}

start_server() {
  local phase="$1"
  local log_file="${log_dir}/${phase}_server.log"
  echo "Starting ${phase} server on ${base_url}; log: ${log_file}"
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
    export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
    exec scripts/start_sglang_lmcache.sh "${extra_sglang_args[@]}"
  ) >"$log_file" 2>&1 &
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
}
trap cleanup EXIT

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
  echo "Sending ${phase} request"
  send_request "$phase" "$prompt_id" "$request_output"

  if [ "$enable_metrics" = "1" ]; then
    "$python_bin" scripts/kv_baseline/kv_baseline_client.py fetch-metrics \
      --base-url "$base_url" \
      --output "${metrics_dir}/${phase}.prom" \
      --allow-missing
  fi

  sleep "$after_request_sleep"
}

append_default_sglang_args

echo "KV baseline run directory: ${run_dir}"
echo "LMCache config: ${config_file}"
echo "Persistent KV cache dir: ${cache_dir}"
echo "Force disk offload: ${force_disk_offload}; local_cpu=${local_cpu_enabled}; local_cpu_gb=${local_cpu_gb}"
echo "Prompt: ${prompt_file}#${prompt_id}"
if [ "$force_disk_offload" = "1" ] && [ "$prompt_file_was_set" = "0" ]; then
  echo "Generated disk prompt repeat: ${disk_prompt_repeat}; max_tokens=${disk_prompt_max_tokens}"
fi
echo "Extra SGLang args: ${extra_sglang_args[*]:-(none)}"

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

echo "Done. Key outputs:"
echo "  ${run_dir}/metadata.json"
echo "  ${run_dir}/lmc_config.yaml"
echo "  ${run_dir}/requests/offload.json"
echo "  ${run_dir}/requests/reuse.json"
echo "  ${run_dir}/summary.json"
