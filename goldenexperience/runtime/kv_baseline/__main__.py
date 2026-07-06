"""CLI for the vLLM + LMCache MP + Mooncake KV baseline."""

from __future__ import annotations

import argparse
import sys

from goldenexperience.runtime.kv_baseline.config import BaselineConfig
from goldenexperience.runtime.kv_baseline.runner import run_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="golden-kv-baseline",
        description="Run the same-model KV offload/reuse baseline.",
        epilog=(
            "Engine args are passed after '--', for example: "
            "golden-kv-baseline -- --tensor-parallel-size 1"
        ),
    )
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = BaselineConfig.from_env(args.engine_args)
        return run_baseline(config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
