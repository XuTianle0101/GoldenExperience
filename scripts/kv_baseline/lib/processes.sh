#!/usr/bin/env bash

stop_engine_server() {
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

cleanup_baseline_processes() {
  if [ "${GE_RUNTIME_KEEP_SERVER_AFTER_REUSE}" != "1" ]; then
    stop_engine_server "active"
  fi
  if [ "$GE_RUNTIME_KV_BACKEND" = "mp" ] && [ "${GE_RUNTIME_KEEP_LMCACHE_MP_AFTER_RUN}" != "1" ]; then
    stop_lmcache_mp_server
  fi
}
