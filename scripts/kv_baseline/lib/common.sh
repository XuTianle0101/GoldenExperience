#!/usr/bin/env bash

ensure_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command '${command_name}' was not found in PATH." >&2
    exit 127
  fi
}

tail_log() {
  local path="$1"
  if [ -f "$path" ]; then
    tail -n 120 "$path" >&2 || true
  fi
}

print_run_context() {
  echo "KV baseline run directory: ${GE_RUNTIME_RUN_DIR}"
  echo "KV backend: ${GE_RUNTIME_KV_BACKEND}; engine: ${GE_RUNTIME_ENGINE}"
  echo "Recorded config: ${GE_RUNTIME_CONFIG_FILE}"
  echo "Prompt: ${GE_RUNTIME_PROMPT_FILE}#${GE_RUNTIME_PROMPT_ID}"
}

print_key_outputs() {
  echo "Done. Key outputs:"
  echo "  ${GE_RUNTIME_RUN_DIR}/metadata.json"
  echo "  ${GE_RUNTIME_RUN_DIR}/runtime.json"
  echo "  ${GE_RUNTIME_RUN_DIR}/lmc_config.yaml"
  echo "  ${GE_RUNTIME_RUN_DIR}/requests/offload.json"
  echo "  ${GE_RUNTIME_RUN_DIR}/requests/reuse.json"
  echo "  ${GE_RUNTIME_RUN_DIR}/summary.json"
}
