#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install the GoldenExperience runtime stack.

Usage:
  scripts/install_runtime.sh [--mode package|source|golden-only] [--no-dev]

Options:
  --mode VALUE          package: install SGLang and LMCache from packages.
                        source: clone/install editable SGLang and LMCache.
                        golden-only: install only this repo.
                        Default: package.
  --no-dev             Install GoldenExperience without dev extras.
  --third-party-dir D  Source clone directory for --mode source. Default: third_party.
  --sglang-repo URL    SGLang repository URL for --mode source.
  --lmcache-repo URL   LMCache repository URL for --mode source.
  -h, --help           Show this help.

Environment:
  PYTHON_BIN           Python executable. Default: python3.
  GE_USE_UV            1 to prefer uv when available. Default: 1.
  GE_INSTALL_MODE      Same as --mode.
  GE_THIRD_PARTY_DIR   Same as --third-party-dir.
  GE_SGLANG_REPO_URL   Same as --sglang-repo.
  GE_LMCACHE_REPO_URL  Same as --lmcache-repo.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mode="${GE_INSTALL_MODE:-package}"
with_dev=1
third_party_dir="${GE_THIRD_PARTY_DIR:-third_party}"
sglang_repo="${GE_SGLANG_REPO_URL:-https://github.com/sgl-project/sglang.git}"
lmcache_repo="${GE_LMCACHE_REPO_URL:-https://github.com/LMCache/LMCache.git}"
python_bin="${PYTHON_BIN:-python3}"
use_uv="${GE_USE_UV:-1}"

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
    --sglang-repo)
      sglang_repo="${2:?--sglang-repo requires a value}"
      shift 2
      ;;
    --lmcache-repo)
      lmcache_repo="${2:?--lmcache-repo requires a value}"
      shift 2
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

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required for the SGLang + LMCache runtime stack.")
PY

if [ "$use_uv" = "1" ] && command -v uv >/dev/null 2>&1; then
  install_cmd=(uv pip install)
  package_runtime_cmd=(uv pip install --prerelease=allow lmcache sglang)
else
  install_cmd=("$python_bin" -m pip install)
  package_runtime_cmd=("$python_bin" -m pip install --pre lmcache sglang)
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
  if [ ! -d "$third_party_dir/sglang/.git" ]; then
    git clone "$sglang_repo" "$third_party_dir/sglang"
  fi
  if [ ! -d "$third_party_dir/LMCache/.git" ]; then
    git clone "$lmcache_repo" "$third_party_dir/LMCache"
  fi

  if [ -d "$third_party_dir/sglang/python" ]; then
    "${install_cmd[@]}" -e "$third_party_dir/sglang/python"
  else
    "${install_cmd[@]}" -e "$third_party_dir/sglang"
  fi
  "${install_cmd[@]}" -e "$third_party_dir/LMCache"
}

case "$mode" in
  package)
    "${package_runtime_cmd[@]}"
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

"$python_bin" scripts/smoke_cross_model_plan.py --check-runtime
