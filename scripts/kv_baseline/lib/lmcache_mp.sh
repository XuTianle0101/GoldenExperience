#!/usr/bin/env bash

start_lmcache_mp_server() {
  if [ "$GE_RUNTIME_KV_BACKEND" != "mp" ]; then
    return
  fi

  ensure_command "$GE_RUNTIME_LMCACHE_BIN"
  local log_file="${GE_RUNTIME_LOG_DIR}/lmcache_mp_server.log"
  local -a lmcache_args
  mapfile -t lmcache_args < <("$python_bin" "$helper" args --runtime "$GE_RUNTIME_JSON" lmcache)

  echo "Starting LMCache MP server; log: ${log_file}"
  (
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export LMCACHE_LOG_LEVEL="$GE_RUNTIME_LMCACHE_LOG_LEVEL"
    exec "$GE_RUNTIME_LMCACHE_BIN" "${lmcache_args[@]}"
  ) >"$log_file" 2>&1 &
  lmcache_pid="$!"
  echo "$lmcache_pid" >"${GE_RUNTIME_RUN_DIR}/lmcache_mp.pid"
}

wait_for_lmcache_mp_ready() {
  if [ "$GE_RUNTIME_KV_BACKEND" != "mp" ]; then
    return
  fi

  local log_file="${GE_RUNTIME_LOG_DIR}/lmcache_mp_server.log"
  if ! "$python_bin" "$helper" tcp-wait \
    --host "$GE_RUNTIME_LMCACHE_MP_HOST" \
    --port "$GE_RUNTIME_LMCACHE_MP_PORT" \
    --timeout "$GE_RUNTIME_SERVER_START_TIMEOUT"; then
    echo "Timed out waiting for LMCache MP server; last log lines:" >&2
    tail_log "$log_file"
    exit 1
  fi

  if ! kill -0 "$lmcache_pid" >/dev/null 2>&1; then
    echo "LMCache MP server exited after port became reachable; last log lines:" >&2
    tail_log "$log_file"
    exit 1
  fi
  echo "LMCache MP server is ready on ${GE_RUNTIME_LMCACHE_MP_HOST}:${GE_RUNTIME_LMCACHE_MP_PORT}"
}
