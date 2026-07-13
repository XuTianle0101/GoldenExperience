#!/usr/bin/env python3
"""Measure non-publishing Mooncake cost for a Qwen3 cached-KV candidate."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.benchmarks.cached_kv_cost import (
    load_native_prefill_evidence,
    run_cached_kv_cost_benchmark,
)
from goldenexperience.size_variant.cached_kv_bridge import Qwen3CachedKVBridge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--mooncake-setup", type=Path, required=True)
    parser.add_argument("--source-key", action="append", required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--native-prefill-report", type=Path, required=True)
    parser.add_argument("--model-identity-cache", type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bridge = Qwen3CachedKVBridge.from_validation_candidate_for_benchmark(
        args.candidate_manifest,
        source_model_path=args.source_model,
        target_model_path=args.target_model,
        device=args.device,
        model_identity_cache_path=args.model_identity_cache,
    )
    setup_config = json.loads(args.mooncake_setup.read_text(encoding="utf-8"))
    if not isinstance(setup_config, dict):
        raise ValueError("--mooncake-setup must contain a JSON object")
    native_evidence = load_native_prefill_evidence(
        args.native_prefill_report,
        bridge=bridge,
        expected_tokens=len(args.source_key) * args.chunk_size,
    )
    report = run_cached_kv_cost_benchmark(
        bridge,
        candidate_manifest_path=args.candidate_manifest,
        setup_config=setup_config,
        source_keys=args.source_key,
        chunk_size=args.chunk_size,
        native_prefill_samples_ms=native_evidence.samples_ms,
        native_prefill_backend=native_evidence.backend,
        native_prefill_eligible=native_evidence.eligible_for_approval,
        native_prefill_report_sha256=native_evidence.report_sha256,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["eligible_for_approval"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
