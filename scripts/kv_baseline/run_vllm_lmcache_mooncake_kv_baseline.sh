#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

python_bin="${PYTHON_BIN:-python3}"
exec "$python_bin" -m goldenexperience.runtime.kv_baseline "$@"
