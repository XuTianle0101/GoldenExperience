#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Install the GoldenExperience runtime stack.

Default runtime: vLLM + LMCache MP + Mooncake Store.

Usage:
  scripts/install_runtime.sh [--mode package|source|golden-only] [--no-dev]

Options:
  --mode VALUE          package: install vLLM and LMCache from packages; verify Mooncake.
                        source: clone/install editable vLLM and LMCache, and clone Mooncake.
                        golden-only: install only this repo.
                        Default: package.
  --no-dev             Install GoldenExperience without dev extras.
  --third-party-dir D  Source clone directory for --mode source. Default: third_party.
  --vllm-repo URL      vLLM repository URL for --mode source.
  --lmcache-repo URL   LMCache repository URL for --mode source.
  --mooncake-repo URL  Mooncake repository URL for --mode source clone.
  --runtime-check MODE strict, warn, or skip final runtime dependency check.
                       Default: strict for package/golden-only, warn for source.
  --skip-runtime-check Alias for --runtime-check skip.
  -h, --help           Show this help.

Environment:
  PYTHON_BIN             Python executable. Default: python3.
  GE_USE_UV              1 to prefer uv when available. Default: 1.
  GE_INSTALL_MODE        Same as --mode.
  GE_THIRD_PARTY_DIR     Same as --third-party-dir.
  GE_VLLM_REPO_URL       Same as --vllm-repo.
  GE_LMCACHE_REPO_URL    Same as --lmcache-repo.
  GE_MOONCAKE_REPO_URL   Same as --mooncake-repo.
  GE_LMCACHE_BUILD_MOONCAKE  Value for BUILD_MOONCAKE when installing LMCache source.
                             Default: 1.
  GE_PATCH_MOONCAKE_RUNTIME  1 to apply the LMCache/Mooncake compatibility patch after
                             installing packages; 0 to skip. Default: 1.
  GE_RUNTIME_CHECK       Same as --runtime-check.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mode="${GE_INSTALL_MODE:-package}"
with_dev=1
third_party_dir="${GE_THIRD_PARTY_DIR:-third_party}"
vllm_repo="${GE_VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
lmcache_repo="${GE_LMCACHE_REPO_URL:-https://github.com/LMCache/LMCache.git}"
mooncake_repo="${GE_MOONCAKE_REPO_URL:-https://github.com/kvcache-ai/Mooncake.git}"
python_bin="${PYTHON_BIN:-python3}"
use_uv="${GE_USE_UV:-1}"
lmcache_build_mooncake="${GE_LMCACHE_BUILD_MOONCAKE:-1}"
runtime_check="${GE_RUNTIME_CHECK:-}"
patch_mooncake_runtime="${GE_PATCH_MOONCAKE_RUNTIME:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      mode="${2:?--mode requires a value}"
      shift 2
      ;;
    --no-dev)
      with_dev=0
      shift
      ;;
    --third-party-dir)
      third_party_dir="${2:?--third-party-dir requires a value}"
      shift 2
      ;;
    --vllm-repo)
      vllm_repo="${2:?--vllm-repo requires a value}"
      shift 2
      ;;
    --lmcache-repo)
      lmcache_repo="${2:?--lmcache-repo requires a value}"
      shift 2
      ;;
    --mooncake-repo)
      mooncake_repo="${2:?--mooncake-repo requires a value}"
      shift 2
      ;;
    --runtime-check)
      runtime_check="${2:?--runtime-check requires a value}"
      shift 2
      ;;
    --skip-runtime-check)
      runtime_check="skip"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  package|source|golden-only) ;;
  *)
    echo "Invalid --mode: $mode" >&2
    exit 2
    ;;
esac

if [ -z "$runtime_check" ]; then
  if [ "$mode" = "source" ]; then
    runtime_check="warn"
  else
    runtime_check="strict"
  fi
fi

case "$runtime_check" in
  strict|warn|skip) ;;
  *)
    echo "Invalid --runtime-check: $runtime_check" >&2
    exit 2
    ;;
esac

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required for GoldenExperience runtime experiments.")
PY

if [ "$use_uv" = "1" ] && command -v uv >/dev/null 2>&1; then
  install_cmd=(uv pip install)
  package_runtime_cmd=(uv pip install --upgrade --prerelease=allow vllm "lmcache>=0.4.6")
else
  install_cmd=("$python_bin" -m pip install)
  package_runtime_cmd=("$python_bin" -m pip install --upgrade --pre vllm "lmcache>=0.4.6")
fi

install_goldenexperience() {
  if [ "$with_dev" = "1" ]; then
    "${install_cmd[@]}" -e ".[dev]"
  else
    "${install_cmd[@]}" -e .
  fi
}

install_source_runtime() {
  mkdir -p "$third_party_dir"
  if [ ! -d "$third_party_dir/vllm/.git" ]; then
    git clone "$vllm_repo" "$third_party_dir/vllm"
  fi
  if [ ! -d "$third_party_dir/LMCache/.git" ]; then
    git clone "$lmcache_repo" "$third_party_dir/LMCache"
  fi
  if [ ! -d "$third_party_dir/Mooncake/.git" ]; then
    git clone "$mooncake_repo" "$third_party_dir/Mooncake"
  fi

  "${install_cmd[@]}" -e "$third_party_dir/vllm"
  BUILD_MOONCAKE="$lmcache_build_mooncake" "${install_cmd[@]}" -e "$third_party_dir/LMCache"

  cat <<MSG
Mooncake source cloned to $third_party_dir/Mooncake.
Build/install Mooncake according to its upstream instructions, then ensure
mooncake_master and mooncake_http_metadata_server are on PATH. Build LMCache with
Mooncake support before running the default baseline.
MSG
}

case "$mode" in
  package)
    "${package_runtime_cmd[@]}"
    "$python_bin" -m pip uninstall -y cupy-cuda12x >/dev/null 2>&1 || true
    "$python_bin" -m pip install --force-reinstall --no-deps cupy-cuda13x
    install_goldenexperience
    ;;
  source)
    install_source_runtime
    install_goldenexperience
    ;;
  golden-only)
    install_goldenexperience
    ;;
esac

if [ "$patch_mooncake_runtime" = "1" ]; then
  "$python_bin" scripts/patch_lmcache_mooncake_runtime.py
else
  echo "Skipping LMCache/Mooncake runtime patch."
fi

case "$runtime_check" in
  strict)
    "$python_bin" scripts/smoke_cross_model_plan.py --check-runtime --strict-runtime
    ;;
  warn)
    "$python_bin" scripts/smoke_cross_model_plan.py --check-runtime
    ;;
  skip)
    echo "Skipping runtime dependency check."
    ;;
esac
