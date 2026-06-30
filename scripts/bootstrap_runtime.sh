#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for the default vLLM + LMCache runtime install path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/install_runtime.sh" "$@"
