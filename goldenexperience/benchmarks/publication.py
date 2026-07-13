"""Hash-only publication benchmark contracts and semantic sealed guard."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PUBLICATION_BENCHMARK_SCHEMA_VERSION = "goldenexperience.grouped_prefix_benchmark.v1"
SEALED_RECEIPT_SCHEMA_VERSION = "goldenexperience.semantic_sealed_receipt.v1"
PREFIX_BUCKETS = (128, 512, 2048, 8192)
SPLIT_COUNTS = {
    "transport_train": 4096,
    "selector_train": 2048,
    "method_dev": 1024,
    "risk_calibration": 2048,
    "validation": 2048,
    "semantic_sealed_test": 2048,
    "runtime_audit": 512,
}
REQUIRED_DATASETS = frozenset(
    {
        "longbench_hotpotqa",
        "longbench_qasper",
        "longbench_multifieldqa",
        "bfcl",
        "gsm8k",
        "math",
        "humaneval",
        "mbpp",
        "sharegpt",
        "burstgpt",
    }
)
TRACE_ONLY_DATASETS = frozenset({"sharegpt", "burstgpt"})
REQUIRED_QWEN3_DIRECTIONS = frozenset(
    {
        "qwen3_4b_to_8b",
        "qwen3_8b_to_4b",
        "qwen3_8b_to_14b",
        "qwen3_14b_to_8b",
    }
)


class BenchmarkContractError(RuntimeError):
    """Raised when benchmark isolation or a sealed boundary is violated."""


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    revision: str
    content_sha256: str
    license_id: str
    license_uri: str
    source_uri: str
    usage: str = "semantic"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not all(
            (
                self.dataset_id,
                self.revision,
                self.license_id,
                self.license_uri,
                self.source_uri,
            )
        ):
            errors.append(f"dataset source {self.dataset_id!r} has incomplete provenance")
        if not _is_sha256(self.content_sha256):
            errors.append(f"dataset source {self.dataset_id!r} lacks a content hash")
        if self.usage not in {"semantic", "trace_only"}:
            errors.append(f"dataset source {self.dataset_id!r} has invalid usage")
        if self.dataset_id in TRACE_ONLY_DATASETS and self.usage != "trace_only":
            errors.append(f"dataset source {self.dataset_id!r} must be trace_only")
        return errors


@dataclass(frozen=True)
class GroupedPrefixRecord:
    """Hash-only sample metadata; sealed content never enters the public manifest."""

    sample_id: str
    split: str
    dataset_id: str
    prefix_group_id: str
    prefix_sha256: str
    suffix_query_sha256: str
    content_sha256: str
    token_bucket: int
    task: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not all((self.sample_id, self.dataset_id, self.prefix_group_id, self.task)):
            errors.append("benchmark record identifiers are required")
        if self.split not in SPLIT_COUNTS:
            errors.append(f"benchmark record {self.sample_id} has an invalid split")
        if self.token_bucket not in PREFIX_BUCKETS:
            errors.append(f"benchmark record {self.sample_id} has an invalid prefix bucket")
        for name in ("prefix_sha256", "suffix_query_sha256", "content_sha256"):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"benchmark record {self.sample_id} has invalid {name}")
        return errors


@dataclass(frozen=True)
class PublicationBenchmarkManifest:
    sources: tuple[DatasetSource, ...]
    records: tuple[GroupedPrefixRecord, ...]
    split_sha256: Mapping[str, str]
    tokenizer_sha256: str
    chat_template_sha256: str
    sealed_payload_sha256: str
    deprecated_synthetic_sealed_sha256: str | None = None
    schema_version: str = PUBLICATION_BENCHMARK_SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != PUBLICATION_BENCHMARK_SCHEMA_VERSION:
            errors.append("unsupported publication benchmark schema")
        for name in ("tokenizer_sha256", "chat_template_sha256", "sealed_payload_sha256"):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"benchmark {name} must be a SHA-256 digest")
        if self.deprecated_synthetic_sealed_sha256 is not None and not _is_sha256(
            self.deprecated_synthetic_sealed_sha256
        ):
            errors.append("deprecated synthetic sealed hash is invalid")
        source_by_id: dict[str, DatasetSource] = {}
        for source in self.sources:
            errors.extend(source.validate())
            if source.dataset_id in source_by_id:
                errors.append(f"duplicate dataset source {source.dataset_id}")
            source_by_id[source.dataset_id] = source
        missing_sources = REQUIRED_DATASETS - set(source_by_id)
        if missing_sources:
            errors.append(f"benchmark is missing required datasets: {sorted(missing_sources)}")

        ids: set[str] = set()
        contents: set[str] = set()
        counts = {name: 0 for name in SPLIT_COUNTS}
        buckets: dict[str, set[int]] = {name: set() for name in SPLIT_COUNTS}
        transport_groups: set[str] = set()
        later_groups: set[str] = set()
        protected_suffixes: dict[str, str] = {}
        used_sources: set[str] = set()
        group_prefixes: dict[str, str] = {}
        for record in self.records:
            errors.extend(record.validate())
            if record.sample_id in ids:
                errors.append(f"duplicate benchmark sample_id {record.sample_id}")
            ids.add(record.sample_id)
            if record.content_sha256 in contents:
                errors.append(f"duplicate benchmark content {record.sample_id}")
            contents.add(record.content_sha256)
            if record.dataset_id not in source_by_id:
                errors.append(f"record {record.sample_id} references an unknown dataset")
            else:
                used_sources.add(record.dataset_id)
            if record.dataset_id in TRACE_ONLY_DATASETS and record.split != "runtime_audit":
                errors.append(f"trace-only record {record.sample_id} appears in a semantic split")
            if record.split in counts:
                counts[record.split] += 1
                buckets[record.split].add(record.token_bucket)
            if record.split == "transport_train":
                transport_groups.add(record.prefix_group_id)
            else:
                later_groups.add(record.prefix_group_id)
            previous_prefix = group_prefixes.get(record.prefix_group_id)
            if previous_prefix is not None and previous_prefix != record.prefix_sha256:
                errors.append(
                    f"prefix group {record.prefix_group_id} maps to multiple prefix hashes"
                )
            group_prefixes[record.prefix_group_id] = record.prefix_sha256
            if record.split in {"risk_calibration", "validation", "semantic_sealed_test"}:
                previous = protected_suffixes.get(record.suffix_query_sha256)
                if previous is not None and previous != record.split:
                    errors.append(f"suffix/query hash crosses {previous} and {record.split} splits")
                protected_suffixes[record.suffix_query_sha256] = record.split
        overlap = transport_groups & later_groups
        if overlap:
            errors.append("transport train prefix groups overlap later benchmark splits")
        unused_sources = REQUIRED_DATASETS - used_sources
        if unused_sources:
            errors.append(f"benchmark has no records from datasets: {sorted(unused_sources)}")
        for split, expected in SPLIT_COUNTS.items():
            if counts[split] != expected:
                errors.append(f"{split} must contain exactly {expected} records")
            if buckets[split] != set(PREFIX_BUCKETS):
                errors.append(f"{split} does not cover all fixed prefix buckets")
            observed_hash = self.compute_split_sha256(split)
            if self.split_sha256.get(split) != observed_hash:
                errors.append(f"{split} hash does not match its records")
        if set(self.split_sha256) != set(SPLIT_COUNTS):
            errors.append("benchmark split hash map is incomplete")
        return errors

    def compute_split_sha256(self, split: str) -> str:
        records = [
            asdict(record)
            for record in sorted(
                (item for item in self.records if item.split == split),
                key=lambda item: item.sample_id,
            )
        ]
        raw = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def content_sha256(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [asdict(item) for item in self.sources],
            "records": [asdict(item) for item in self.records],
            "split_sha256": dict(self.split_sha256),
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "sealed_payload_sha256": self.sealed_payload_sha256,
            "deprecated_synthetic_sealed_sha256": self.deprecated_synthetic_sealed_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicationBenchmarkManifest:
        return cls(
            sources=tuple(DatasetSource(**item) for item in payload.get("sources", ())),
            records=tuple(GroupedPrefixRecord(**item) for item in payload.get("records", ())),
            split_sha256=dict(payload.get("split_sha256", {})),
            tokenizer_sha256=payload.get("tokenizer_sha256", ""),
            chat_template_sha256=payload.get("chat_template_sha256", ""),
            sealed_payload_sha256=payload.get("sealed_payload_sha256", ""),
            deprecated_synthetic_sealed_sha256=payload.get("deprecated_synthetic_sealed_sha256"),
            schema_version=payload.get("schema_version", ""),
        )

    @classmethod
    def load(cls, path: str | Path, *, validate: bool = True) -> PublicationBenchmarkManifest:
        manifest = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if validate:
            errors = manifest.validate()
            if errors:
                raise BenchmarkContractError("; ".join(errors))
        return manifest

    def save(self, path: str | Path) -> None:
        errors = self.validate()
        if errors:
            raise BenchmarkContractError("; ".join(errors))
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@dataclass(frozen=True)
class DirectionValidationEvidence:
    direction: str
    passed: bool
    report_sha256: str
    code_sha256: str
    transport_weights_sha256: str
    predictor_sha256: str
    threshold_sha256: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.direction not in REQUIRED_QWEN3_DIRECTIONS:
            errors.append(f"unknown Qwen3 validation direction {self.direction}")
        if type(self.passed) is not bool or not self.passed:
            errors.append(f"Qwen3 validation direction {self.direction} did not pass")
        for name in (
            "report_sha256",
            "code_sha256",
            "transport_weights_sha256",
            "predictor_sha256",
            "threshold_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"validation evidence {name} is invalid")
        return errors


@dataclass(frozen=True)
class ValidationGateReceipt:
    benchmark_manifest_sha256: str
    validation_dataset_sha256: str
    directions: tuple[DirectionValidationEvidence, ...]
    schema_version: str = SEALED_RECEIPT_SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != SEALED_RECEIPT_SCHEMA_VERSION:
            errors.append("unsupported validation gate receipt schema")
        if not _is_sha256(self.benchmark_manifest_sha256) or not _is_sha256(
            self.validation_dataset_sha256
        ):
            errors.append("validation gate receipt hashes are invalid")
        directions = {item.direction for item in self.directions}
        if directions != REQUIRED_QWEN3_DIRECTIONS or len(self.directions) != len(directions):
            errors.append("all four Qwen3 main directions must pass before sealed access")
        for item in self.directions:
            errors.extend(item.validate())
        return errors

    def content_sha256(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SemanticSealedGuard:
    """Open a sealed payload once, only after all validation receipts pass."""

    def __init__(self, marker_path: str | Path) -> None:
        self.marker_path = Path(marker_path)

    def open_once(
        self,
        payload_path: str | Path,
        *,
        expected_payload_sha256: str,
        receipt: ValidationGateReceipt,
        expected_manifest_sha256: str,
        expected_validation_sha256: str,
        validate_payload: Callable[[bytes], None] | None = None,
        opened_metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        errors = receipt.validate()
        if receipt.benchmark_manifest_sha256 != expected_manifest_sha256:
            errors.append("validation receipt refers to a different benchmark manifest")
        if receipt.validation_dataset_sha256 != expected_validation_sha256:
            errors.append("validation receipt refers to a different validation split")
        if errors:
            raise BenchmarkContractError("; ".join(errors))
        metadata = dict(opened_metadata or {})
        reserved = {
            "schema_version",
            "state",
            "payload_sha256",
            "validation_receipt_sha256",
            "error_type",
        }
        if set(metadata) & reserved:
            raise BenchmarkContractError("semantic sealed marker metadata uses reserved fields")
        try:
            _json_bytes(metadata)
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError("semantic sealed marker metadata is malformed") from exc
        path = Path(payload_path)
        if path.is_symlink():
            raise BenchmarkContractError("semantic sealed payload cannot be a symbolic link")
        try:
            before = path.stat()
        except OSError as exc:
            raise BenchmarkContractError("semantic sealed payload is unavailable") from exc
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.marker_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
            )
        except FileExistsError as exc:
            raise BenchmarkContractError("semantic sealed payload was already opened") from exc
        receipt_sha256 = receipt.content_sha256()
        try:
            _replace_descriptor(
                descriptor,
                _json_bytes(
                    {
                        "schema_version": "goldenexperience.semantic_sealed_open.v1",
                        "state": "opening",
                        "validation_receipt_sha256": receipt_sha256,
                        **metadata,
                    }
                ),
            )
            os.fsync(descriptor)
            _fsync_directory(self.marker_path.parent)
            try:
                payload = path.read_bytes()
                after = path.stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise BenchmarkContractError("semantic sealed payload changed while opening")
                observed = hashlib.sha256(payload).hexdigest()
                if observed != expected_payload_sha256:
                    raise BenchmarkContractError("semantic sealed payload checksum mismatch")
                if validate_payload is not None:
                    validate_payload(payload)
            except Exception as exc:
                _replace_descriptor(
                    descriptor,
                    _json_bytes(
                        {
                            "schema_version": "goldenexperience.semantic_sealed_open.v1",
                            "state": "failed",
                            "validation_receipt_sha256": receipt_sha256,
                            "error_type": type(exc).__name__,
                            **metadata,
                        }
                    ),
                )
                os.fsync(descriptor)
                if isinstance(exc, BenchmarkContractError):
                    raise
                raise BenchmarkContractError("semantic sealed payload could not be opened") from exc
            _replace_descriptor(
                descriptor,
                _json_bytes(
                    {
                        "schema_version": "goldenexperience.semantic_sealed_open.v1",
                        "state": "opened",
                        "payload_sha256": observed,
                        "validation_receipt_sha256": receipt_sha256,
                        **metadata,
                    }
                ),
            )
            os.fsync(descriptor)
            return payload
        finally:
            os.close(descriptor)


def write_immutable_sealed_report(directory: str | Path, report: Mapping[str, Any]) -> Path:
    """Publish a content-addressed report without an overwrite path."""

    raw = _json_bytes(dict(report), indent=2)
    digest = hashlib.sha256(raw).hexdigest()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest}.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        try:
            _write_all(descriptor, raw)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
            _fsync_directory(root)
        except FileExistsError as exc:
            raise BenchmarkContractError("immutable sealed report already exists") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _json_bytes(value: Mapping[str, Any], *, indent: int | None = None) -> bytes:
    return (json.dumps(dict(value), indent=indent, sort_keys=True) + "\n").encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("failed to write publication artifact")
        remaining = remaining[written:]


def _replace_descriptor(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    _write_all(descriptor, payload)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
