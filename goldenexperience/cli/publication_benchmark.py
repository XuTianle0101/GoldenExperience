"""Freeze or validate the grouped-prefix publication benchmark manifest."""

from __future__ import annotations

import argparse
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
from goldenexperience.benchmarks.publication_builder import (
    PublicationDatasetBuilder,
    PublicationTokenizer,
    publish_publication_build,
)
from goldenexperience.benchmarks.publication_sources import (
    PublicationSourceLock,
    audit_publication_sources,
    load_publication_sources,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    chat_template_sha256,
    sha256_file,
    tokenizer_semantic_sha256,
)


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
        tokenizer_sha256=tokenizer_semantic_sha256(args.tokenizer_model),
        chat_template_sha256=chat_template_sha256(args.tokenizer_model),
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
    freeze.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help="Model directory containing the canonical tokenizer and chat template.",
    )
    freeze.add_argument("--sealed-payload", type=Path, required=True)
    freeze.add_argument("--deprecated-synthetic-sealed", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an already frozen manifest")
    validate.add_argument("manifest", type=Path)
    audit = subparsers.add_parser(
        "audit-sources",
        help="verify every locked source file without constructing benchmark rows",
    )
    _add_source_arguments(audit)
    build = subparsers.add_parser(
        "build",
        help="build all balanced real-data splits and a separately protected sealed payload",
    )
    _add_source_arguments(build)
    build.add_argument("--tokenizer-model", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--sealed-output", type=Path, required=True)
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root for relative paths in the portable source lock.",
    )
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        metavar="DATASET:ROLE=PATH",
        help="Override one locked local path without changing its portable identity.",
    )


def _source_path_overrides(values: list[str]) -> dict[tuple[str, str], Path]:
    overrides: dict[tuple[str, str], Path] = {}
    for value in values:
        binding, separator, path = value.partition("=")
        dataset_id, role_separator, role = binding.partition(":")
        if not separator or not role_separator or not dataset_id or not role or not path:
            raise ValueError("--source-path must use DATASET:ROLE=PATH")
        key = (dataset_id, role)
        if key in overrides:
            raise ValueError(f"duplicate --source-path override for {dataset_id}:{role}")
        overrides[key] = Path(path)
    return overrides


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
    if args.command in {"audit-sources", "build"}:
        source_lock = PublicationSourceLock.load(args.source_lock)
        audited = audit_publication_sources(
            source_lock,
            source_root=args.source_root,
            path_overrides=_source_path_overrides(args.source_path),
        )
        if args.command == "audit-sources":
            print(
                json.dumps(
                    {
                        "source_lock": str(args.source_lock),
                        "source_lock_sha256": source_lock.content_sha256(),
                        "files": len(audited.files),
                        "bytes": sum(item.size_bytes for item in audited.files),
                    },
                    sort_keys=True,
                )
            )
            return 0
        loaded = load_publication_sources(audited)
        tokenizer = PublicationTokenizer.from_model(args.tokenizer_model)
        result = publish_publication_build(
            PublicationDatasetBuilder(
                audited_sources=audited,
                loaded_sources=loaded,
                tokenizer=tokenizer,
            ),
            output_dir=args.output_dir,
            sealed_payload=args.sealed_output,
        )
        print(
            json.dumps(
                {
                    "manifest": str(result.manifest_path),
                    "manifest_sha256": result.manifest.content_sha256(),
                    "records": len(result.manifest.records),
                    "sealed_payload": str(result.sealed_payload),
                    "sealed_payload_sha256": result.manifest.sealed_payload_sha256,
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
