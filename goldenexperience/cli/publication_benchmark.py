"""Freeze or validate the grouped-prefix publication benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    SPLIT_COUNTS,
    DatasetSource,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(value)
    return records


def freeze_manifest(args: argparse.Namespace) -> PublicationBenchmarkManifest:
    source_payload = json.loads(args.sources.read_text(encoding="utf-8"))
    if not isinstance(source_payload, list):
        raise ValueError("sources file must contain a JSON list")
    sources = tuple(DatasetSource(**item) for item in source_payload)
    records = tuple(GroupedPrefixRecord(**item) for item in _load_jsonl(args.records))
    provisional = PublicationBenchmarkManifest(
        sources=sources,
        records=records,
        split_sha256={},
        tokenizer_sha256=sha256_file(args.tokenizer),
        chat_template_sha256=_sha256_text(args.chat_template),
        sealed_payload_sha256=sha256_file(args.sealed_payload),
        deprecated_synthetic_sealed_sha256=(
            sha256_file(args.deprecated_synthetic_sealed)
            if args.deprecated_synthetic_sealed is not None
            else None
        ),
    )
    manifest = replace(
        provisional,
        split_sha256={split: provisional.compute_split_sha256(split) for split in SPLIT_COUNTS},
    )
    manifest.save(args.output)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze a hash-only benchmark manifest")
    freeze.add_argument("--sources", type=Path, required=True)
    freeze.add_argument("--records", type=Path, required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--chat-template", type=Path, required=True)
    freeze.add_argument("--sealed-payload", type=Path, required=True)
    freeze.add_argument("--deprecated-synthetic-sealed", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an already frozen manifest")
    validate.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        manifest = freeze_manifest(args)
        print(
            json.dumps(
                {
                    "manifest": str(args.output),
                    "sha256": manifest.content_sha256(),
                    "records": len(manifest.records),
                    "sealed_payload_sha256": manifest.sealed_payload_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = PublicationBenchmarkManifest.load(args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "sha256": manifest.content_sha256(),
                "records": len(manifest.records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
