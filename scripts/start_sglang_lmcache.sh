#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start SGLang with LMCache and GoldenExperience metadata enabled.

Usage:
  GE_MODEL_PATH=Qwen/Qwen3-8B scripts/start_sglang_lmcache.sh [extra sglang args...]

Environment:
  PYTHON_BIN                    Python executable. Default: python3.
  GE_MODEL_PATH                 Model path or HF model id. Default: Qwen/Qwen3-8B.
  GE_HOST                       SGLang host. Default: 0.0.0.0.
  GE_PORT                       SGLang port. Default: 30000.
  GE_LMCACHE_CONFIG_FILE        LMCache config path. Default: artifacts/runtime/lmc_config.yaml.
  GE_LMCACHE_CHUNK_SIZE         LMCache chunk_size for generated config. Default: 256.
  GE_LMCACHE_LOCAL_CPU_GB       LMCache max_local_cpu_size for generated config. Default: 10.
  GE_OVERWRITE_LMCACHE_CONFIG   1 to overwrite an existing generated config. Default: 0.
  GE_PATCH_MANIFEST             Patch manifest path. Default: docs/patch_manifest.md.
  GE_EXTRA_SGLANG_ARGS          Additional SGLang args when no positional args are given.
  -h, --help                    Show this help.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

python_bin="${PYTHON_BIN:-python3}"
model_path="${GE_MODEL_PATH:-Qwen/Qwen3-8B}"
host="${GE_HOST:-0.0.0.0}"
port="${GE_PORT:-30000}"
lmcache_config_file="${GE_LMCACHE_CONFIG_FILE:-artifacts/runtime/lmc_config.yaml}"
chunk_size="${GE_LMCACHE_CHUNK_SIZE:-256}"
local_cpu_gb="${GE_LMCACHE_LOCAL_CPU_GB:-10}"
patch_manifest="${GE_PATCH_MANIFEST:-docs/patch_manifest.md}"

generate_lmcache_config() {
  mkdir -p "$(dirname "$lmcache_config_file")"
  if [ -f "$lmcache_config_file" ] && [ "${GE_OVERWRITE_LMCACHE_CONFIG:-0}" != "1" ]; then
    return
  fi
  cat > "$lmcache_config_file" <<EOF
chunk_size: ${chunk_size}
local_cpu: true
use_layerwise: true
max_local_cpu_size: ${local_cpu_gb}
EOF
}

generate_patch_manifest() {
  mkdir -p "$(dirname "$patch_manifest")"
  "$python_bin" -m goldenexperience.cli.patch_manifest --output "$patch_manifest"
}

check_imports() {
  "$python_bin" - <<'PY'
import importlib.util
missing = [name for name in ("sglang", "lmcache", "goldenexperience") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing runtime imports: " + ", ".join(missing))
PY
}

generate_lmcache_config
generate_patch_manifest
check_imports

export LMCACHE_CONFIG_FILE="$lmcache_config_file"
export GE_ENABLE_CROSS_MODEL_REUSE="${GE_ENABLE_CROSS_MODEL_REUSE:-1}"
export GE_PATCH_MANIFEST="$patch_manifest"
export GE_LMCACHE_CONFIG="${GE_LMCACHE_CONFIG:-configs/lmcache.example.yaml}"
export GE_SGLANG_MODEL_ID="$model_path"

extra_args=()
if [ "$#" -gt 0 ]; then
  extra_args=("$@")
elif [ -n "${GE_EXTRA_SGLANG_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=(${GE_EXTRA_SGLANG_ARGS})
fi

echo "Starting SGLang with LMCache"
echo "  model: ${model_path}"
echo "  listen: ${host}:${port}"
echo "  LMCache config: ${LMCACHE_CONFIG_FILE}"
echo "  GoldenExperience manifest: ${GE_PATCH_MANIFEST}"

exec "$python_bin" -m sglang.launch_server \
  --model-path "$model_path" \
  --host "$host" \
  --port "$port" \
  --enable-lmcache \
  "${extra_args[@]}"
