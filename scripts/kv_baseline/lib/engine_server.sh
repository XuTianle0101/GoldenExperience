#!/usr/bin/env bash

start_engine_server() {
  local phase="$1"
  local log_file="${GE_RUNTIME_LOG_DIR}/${phase}_server.log"
  echo "Starting ${phase} ${GE_RUNTIME_ENGINE} server on ${GE_RUNTIME_BASE_URL}; log: ${log_file}"

  if [ "$GE_RUNTIME_ENGINE" = "vllm" ]; then
    ensure_command "$GE_RUNTIME_VLLM_BIN"
    local -a vllm_args
    mapfile -t vllm_args < <("$python_bin" "$helper" args --runtime "$GE_RUNTIME_JSON" vllm)
    (
      export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
      export LMCACHE_LOG_LEVEL="$GE_RUNTIME_LMCACHE_LOG_LEVEL"
      exec "$GE_RUNTIME_VLLM_BIN" "${vllm_args[@]}"
    ) >"$log_file" 2>&1 &
  else
    local -a sglang_args
    mapfile -t sglang_args < <("$python_bin" "$helper" args --runtime "$GE_RUNTIME_JSON" sglang-legacy)
    (
      export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
      export GE_MODEL_PATH="$GE_RUNTIME_MODEL_PATH"
      export GE_HOST="$GE_RUNTIME_HOST"
      export GE_PORT="$GE_RUNTIME_PORT"
      export GE_LMCACHE_CONFIG_FILE="$GE_RUNTIME_CONFIG_FILE"
      export GE_OVERWRITE_LMCACHE_CONFIG=0
      export GE_ENABLE_CROSS_MODEL_REUSE=0
      export GE_SGLANG_MODEL_ID="$GE_RUNTIME_MODEL_PATH"
      export LMCACHE_CONFIG_FILE="$GE_RUNTIME_CONFIG_FILE"
      export LMCACHE_LOG_LEVEL="$GE_RUNTIME_LMCACHE_LOG_LEVEL"
      exec scripts/start_sglang_lmcache.sh "${sglang_args[@]}"
    ) >"$log_file" 2>&1 &
  fi

  server_pid="$!"
  echo "$server_pid" >"${GE_RUNTIME_RUN_DIR}/${phase}.pid"
}

wait_for_ready_log_pattern() {
  local phase="$1"
  local log_file="${GE_RUNTIME_LOG_DIR}/${phase}_server.log"
  local deadline=$((SECONDS + ${GE_RUNTIME_SERVER_START_TIMEOUT%.*}))
  if [ "$GE_RUNTIME_WAIT_FOR_READY_LOG" != "1" ]; then
    return
  fi

  while [ "$SECONDS" -lt "$deadline" ]; do
    if grep -E -q -- "$GE_RUNTIME_READY_LOG_PATTERN" "$log_file" 2>/dev/null; then
      echo "${phase} server emitted ready log: ${GE_RUNTIME_READY_LOG_PATTERN}"
      return
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "${phase} server exited before ready log; last log lines:" >&2
      tail_log "$log_file"
      exit 1
    fi
    sleep 1
  done

  echo "Timed out waiting for ${phase} ready log '${GE_RUNTIME_READY_LOG_PATTERN}'; last log lines:" >&2
  tail_log "$log_file"
  exit 1
}

wait_for_engine_ready() {
  local phase="$1"
  local log_file="${GE_RUNTIME_LOG_DIR}/${phase}_server.log"
  local deadline=$((SECONDS + ${GE_RUNTIME_SERVER_START_TIMEOUT%.*}))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "${phase} server exited before becoming ready; last log lines:" >&2
      tail_log "$log_file"
      exit 1
    fi
    if "$python_bin" "$client" wait \
      --base-url "$GE_RUNTIME_BASE_URL" \
      --timeout 2 \
      --interval 1 >/dev/null 2>&1; then
      wait_for_ready_log_pattern "$phase"
      echo "${phase} server is ready at ${GE_RUNTIME_BASE_URL}"
      return
    fi
    sleep 2
  done

  echo "Timed out waiting for ${phase} server; last log lines:" >&2
  tail_log "$log_file"
  exit 1
}
