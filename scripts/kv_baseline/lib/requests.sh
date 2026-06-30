#!/usr/bin/env bash

send_phase_request() {
  local phase="$1"
  local prompt_id="$2"
  local output="$3"
  local -a request_args=(
    request
    --base-url "$GE_RUNTIME_BASE_URL"
    --model "$GE_RUNTIME_MODEL_NAME"
    --prompt-file "$GE_RUNTIME_PROMPT_FILE"
    --prompt-id "$prompt_id"
    --phase "$phase"
    --output "$output"
    --timeout "$GE_RUNTIME_REQUEST_TIMEOUT"
  )
  if [ -n "${GE_MAX_TOKENS:-}" ]; then
    request_args+=(--max-tokens "$GE_MAX_TOKENS")
  fi
  if [ -n "${GE_TEMPERATURE:-}" ]; then
    request_args+=(--temperature "$GE_TEMPERATURE")
  fi
  if [ "$GE_RUNTIME_INCLUDE_USAGE" != "1" ]; then
    request_args+=(--no-include-usage)
  fi
  "$python_bin" "$client" "${request_args[@]}"
}

fetch_phase_metrics() {
  local phase="$1"
  if [ "$GE_RUNTIME_ENABLE_METRICS" != "1" ]; then
    return
  fi

  "$python_bin" "$client" fetch-metrics \
    --base-url "$GE_RUNTIME_BASE_URL" \
    --output "${GE_RUNTIME_METRICS_DIR}/${phase}.prom" \
    --allow-missing

  if [ "$GE_RUNTIME_KV_BACKEND" = "mp" ]; then
    "$python_bin" "$client" fetch-metrics \
      --base-url "http://${GE_RUNTIME_LMCACHE_MP_HTTP_HOST}:${GE_RUNTIME_LMCACHE_MP_PROMETHEUS_PORT}" \
      --output "${GE_RUNTIME_METRICS_DIR}/${phase}_lmcache_mp.prom" \
      --allow-missing
  fi
}

run_phase_request() {
  local phase="$1"
  local request_output="${GE_RUNTIME_REQUEST_DIR}/${phase}.json"
  if [ "$GE_RUNTIME_WARMUP_BEFORE_MEASURE" = "1" ]; then
    echo "Sending ${phase} warmup request (${GE_RUNTIME_WARMUP_PROMPT_ID}); output is excluded from timing deltas"
    send_phase_request "${phase}_warmup" "$GE_RUNTIME_WARMUP_PROMPT_ID" "${GE_RUNTIME_REQUEST_DIR}/${phase}_warmup.json"
    sleep "$GE_RUNTIME_AFTER_WARMUP_SLEEP"
  fi

  "$python_bin" "$helper" mark-log --runtime "$GE_RUNTIME_JSON" --phase "$phase"
  echo "Sending ${phase} request"
  send_phase_request "$phase" "$GE_RUNTIME_PROMPT_ID" "$request_output"
  fetch_phase_metrics "$phase"
  sleep "$GE_RUNTIME_AFTER_REQUEST_SLEEP"
  "$python_bin" "$helper" capture-log --runtime "$GE_RUNTIME_JSON" --phase "$phase"
}

write_baseline_summary() {
  local -a summary_args=(
    summarize
    --run-dir "$GE_RUNTIME_RUN_DIR"
    --output "${GE_RUNTIME_RUN_DIR}/summary.json"
  )
  if [ "$GE_RUNTIME_REQUIRE_REUSE_EVIDENCE" = "1" ]; then
    summary_args+=(--require-reuse-evidence)
  fi
  if [ "$GE_RUNTIME_FORCE_DISK_OFFLOAD" = "1" ]; then
    summary_args+=(--require-disk-offload)
  fi
  "$python_bin" "$client" "${summary_args[@]}"
}
