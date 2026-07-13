"""CLI for the bounded v5 real-model implementation smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldenexperience.size_variant.real_model_smoke import (
    run_real_model_smoke,
    write_smoke_report,
)

DEFAULT_SOURCE = Path("/workspace/volume/softdata/models/Qwen3-4B")
DEFAULT_TARGET = Path("/workspace/volume/softdata/models/Qwen3-8B")
DEFAULT_PROMPT = (
    "GoldenExperience verifies exact cross-scale cached key value transport. " * 20
).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--source-model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--target-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--source-parameter-count-b", type=float, default=4.0)
    parser.add_argument("--target-parameter-count-b", type=float, default=8.0)
    parser.add_argument("--source-revision", default="local-snapshot")
    parser.add_argument("--target-revision", default="local-snapshot")
    parser.add_argument("--direction", default="qwen3_4b_to_8b")
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--max-keys", type=int, default=32)
    parser.add_argument("--rank", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--source-window", type=int, choices=(1, 3), default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--identity-cache",
        type=Path,
        default=Path("artifacts/cache/v5_smoke_model_identities.json"),
    )
    parser.add_argument("--refresh-identity", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_real_model_smoke(
        source_path=args.source_model,
        target_path=args.target_model,
        source_model_id=args.source_model_id,
        target_model_id=args.target_model_id,
        source_parameter_count_b=args.source_parameter_count_b,
        target_parameter_count_b=args.target_parameter_count_b,
        source_revision=args.source_revision,
        target_revision=args.target_revision,
        direction=args.direction,
        prompt=args.prompt,
        source_device=args.source_device,
        target_device=args.target_device,
        max_tokens=args.max_tokens,
        max_queries=args.max_queries,
        max_keys=args.max_keys,
        rank=args.rank,
        source_window=args.source_window,
        seed=args.seed,
        identity_cache_path=args.identity_cache,
        refresh_identity=args.refresh_identity,
        local_files_only=not args.allow_download,
    )
    if args.output is not None:
        write_smoke_report(args.output, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
