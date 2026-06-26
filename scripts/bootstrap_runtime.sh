#!/usr/bin/env bash
set -euo pipefail

# Optional helper for local development. It clones SGLang and LMCache only when the
# target directories do not exist, then installs both in editable mode.
ROOT_DIR="${GE_THIRD_PARTY_DIR:-third_party}"
SGLANG_REPO="${GE_SGLANG_REPO_URL:-https://github.com/sgl-project/sglang.git}"
LMCACHE_REPO="${GE_LMCACHE_REPO_URL:-https://github.com/LMCache/LMCache.git}"

mkdir -p "${ROOT_DIR}"

if [ ! -d "${ROOT_DIR}/sglang/.git" ]; then
  git clone "${SGLANG_REPO}" "${ROOT_DIR}/sglang"
fi

if [ ! -d "${ROOT_DIR}/LMCache/.git" ]; then
  git clone "${LMCACHE_REPO}" "${ROOT_DIR}/LMCache"
fi

python3 -m pip install -e "${ROOT_DIR}/sglang"
python3 -m pip install -e "${ROOT_DIR}/LMCache"
python3 -m pip install -e ".[dev]"
