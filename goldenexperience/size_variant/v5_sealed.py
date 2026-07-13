"""One-shot semantic-sealed opening after all four v5 validations pass."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    REQUIRED_QWEN3_DIRECTIONS,
    SPLIT_COUNTS,
    BenchmarkContractError,
    DirectionValidationEvidence,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
    SemanticSealedGuard,
    ValidationGateReceipt,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file
from goldenexperience.size_variant.selective_manifest import ArtifactState
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    load_bound_benchmark,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError, V5PipelineWorkspace
from goldenexperience.size_variant.v5_validation import load_completed_validation

V5_SEMANTIC_OPEN_RECEIPT_SCHEMA = "goldenexperience.v5_semantic_open_receipt.v1"
V5_SEMANTIC_SNAPSHOT_DIRECTORY = ".pipeline/semantic_sealed"
V5_SEMANTIC_OPEN_RECEIPT_PATH = ".pipeline/semantic_sealed_open_receipt.json"


@dataclass(frozen=True)
class V5SealedDirectionBinding:
    direction: str
    validation_manifest_sha256: str
    validation_report_sha256: str
    risk_calibration_manifest_sha256: str
    selective_artifact_id: str
    code_sha256: str
    transport_weights_sha256: str
    predictor_sha256: str
    threshold: float
    threshold_sha256: str
    passed: bool

    def validate(self, *, code_sha256: str) -> list[str]:
        errors: list[str] = []
        if self.direction not in REQUIRED_QWEN3_DIRECTIONS:
            errors.append("semantic open binding has an unknown direction")
        for name in (
            "validation_manifest_sha256",
            "validation_report_sha256",
            "risk_calibration_manifest_sha256",
            "code_sha256",
            "transport_weights_sha256",
            "predictor_sha256",
            "threshold_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"semantic open direction {name} is invalid")
        if self.code_sha256 != code_sha256:
            errors.append("semantic open direction code hash changed")
        if not self.selective_artifact_id.startswith("selective-kv-"):
            errors.append("semantic open direction selective artifact id is invalid")
        if not _finite_probability(self.threshold):
            errors.append("semantic open direction threshold is invalid")
        elif self.threshold_sha256 != _threshold_sha256(
            self.direction,
            self.threshold,
            self.risk_calibration_manifest_sha256,
        ):
            errors.append("semantic open direction threshold binding is inconsistent")
        if type(self.passed) is not bool or not self.passed:
            errors.append("semantic open direction did not pass validation")
        return errors


@dataclass(frozen=True)
class V5SemanticOpenReceipt:
    pipeline_id: str
    benchmark_manifest_sha256: str
    validation_split_sha256: str
    semantic_split_sha256: str
    sealed_payload_sha256: str
    code_sha256: str
    validation_gate_receipt_sha256: str
    snapshot_path: str
    directions: tuple[V5SealedDirectionBinding, ...]
    opened_once: bool = True
    schema_version: str = V5_SEMANTIC_OPEN_RECEIPT_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        gate_receipt: ValidationGateReceipt,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_SEMANTIC_OPEN_RECEIPT_SCHEMA:
            errors.append("unsupported v5 semantic open receipt schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("semantic open receipt belongs to another pipeline")
        expected = {
            "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
            "validation_split_sha256": workspace.config.split_sha256["validation"],
            "semantic_split_sha256": workspace.config.split_sha256["semantic_sealed_test"],
            "sealed_payload_sha256": workspace.config.sealed_payload_sha256,
            "code_sha256": workspace.config.code_sha256,
            "validation_gate_receipt_sha256": gate_receipt.content_sha256(),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"semantic open receipt {name} changed")
        if not all(_is_sha256(value) for value in expected.values()):
            errors.append("semantic open receipt contains invalid hashes")
        expected_snapshot = f"{V5_SEMANTIC_SNAPSHOT_DIRECTORY}/{self.sealed_payload_sha256}.jsonl"
        if self.snapshot_path != expected_snapshot:
            errors.append("semantic open snapshot path is not content-addressed")
        directions = {item.direction for item in self.directions}
        if directions != REQUIRED_QWEN3_DIRECTIONS or len(self.directions) != len(directions):
            errors.append("semantic open receipt requires exactly four validation directions")
        gate_by_direction = {item.direction: item for item in gate_receipt.directions}
        for item in self.directions:
            errors.extend(item.validate(code_sha256=self.code_sha256))
            gate_item = gate_by_direction.get(item.direction)
            if gate_item is None or gate_item != DirectionValidationEvidence(
                direction=item.direction,
                passed=item.passed,
                report_sha256=item.validation_report_sha256,
                code_sha256=item.code_sha256,
                transport_weights_sha256=item.transport_weights_sha256,
                predictor_sha256=item.predictor_sha256,
                threshold_sha256=item.threshold_sha256,
            ):
                errors.append("semantic open direction differs from validation gate receipt")
        if type(self.opened_once) is not bool or not self.opened_once:
            errors.append("semantic open receipt is not one-shot")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> V5SemanticOpenReceipt:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            validation_split_sha256=str(payload["validation_split_sha256"]),
            semantic_split_sha256=str(payload["semantic_split_sha256"]),
            sealed_payload_sha256=str(payload["sealed_payload_sha256"]),
            code_sha256=str(payload["code_sha256"]),
            validation_gate_receipt_sha256=str(payload["validation_gate_receipt_sha256"]),
            snapshot_path=str(payload["snapshot_path"]),
            directions=tuple(
                V5SealedDirectionBinding(**item) for item in payload.get("directions", ())
            ),
            opened_once=payload.get("opened_once", False),
            schema_version=str(payload.get("schema_version", "")),
        )


def open_semantic_sealed_once(
    workspace: V5PipelineWorkspace,
    payload_path: str | Path,
) -> tuple[V5SemanticOpenReceipt, Path]:
    """Validate four directions, then atomically open and snapshot the sealed JSONL once."""

    benchmark = load_bound_benchmark(workspace)
    bindings: list[V5SealedDirectionBinding] = []
    gate_directions: list[DirectionValidationEvidence] = []
    for direction in sorted(REQUIRED_QWEN3_DIRECTIONS):
        validation, selective, _ = load_completed_validation(workspace, direction)
        if selective.state is not ArtifactState.VALIDATION_CANDIDATE:
            raise V5PipelineError("semantic opening requires validation_candidate artifacts")
        threshold_sha256 = _threshold_sha256(
            direction,
            validation.threshold,
            validation.risk_calibration_manifest_sha256,
        )
        binding = V5SealedDirectionBinding(
            direction=direction,
            validation_manifest_sha256=validation.content_sha256(),
            validation_report_sha256=validation.validation_report_sha256,
            risk_calibration_manifest_sha256=validation.risk_calibration_manifest_sha256,
            selective_artifact_id=selective.artifact_id,
            code_sha256=validation.code_sha256,
            transport_weights_sha256=validation.transport_weights_sha256,
            predictor_sha256=validation.predictor_sha256,
            threshold=validation.threshold,
            threshold_sha256=threshold_sha256,
            passed=validation.passed,
        )
        bindings.append(binding)
        gate_directions.append(
            DirectionValidationEvidence(
                direction=direction,
                passed=binding.passed,
                report_sha256=binding.validation_report_sha256,
                code_sha256=binding.code_sha256,
                transport_weights_sha256=binding.transport_weights_sha256,
                predictor_sha256=binding.predictor_sha256,
                threshold_sha256=binding.threshold_sha256,
            )
        )
    gate_receipt = ValidationGateReceipt(
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        validation_dataset_sha256=workspace.config.split_sha256["validation"],
        directions=tuple(gate_directions),
    )
    gate_errors = gate_receipt.validate()
    if gate_errors:
        raise V5PipelineError("; ".join(gate_errors))
    snapshot_relative = (
        f"{V5_SEMANTIC_SNAPSHOT_DIRECTORY}/{workspace.config.sealed_payload_sha256}.jsonl"
    )
    receipt = V5SemanticOpenReceipt(
        pipeline_id=workspace.config.pipeline_id,
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        validation_split_sha256=workspace.config.split_sha256["validation"],
        semantic_split_sha256=workspace.config.split_sha256["semantic_sealed_test"],
        sealed_payload_sha256=workspace.config.sealed_payload_sha256,
        code_sha256=workspace.config.code_sha256,
        validation_gate_receipt_sha256=gate_receipt.content_sha256(),
        snapshot_path=snapshot_relative,
        directions=tuple(bindings),
    )
    errors = receipt.validate(workspace=workspace, gate_receipt=gate_receipt)
    if errors:
        raise V5PipelineError("; ".join(errors))
    snapshot_path = workspace.root / snapshot_relative
    receipt_path = workspace.root / V5_SEMANTIC_OPEN_RECEIPT_PATH
    source_input = Path(payload_path)
    if source_input.is_symlink():
        raise V5PipelineError("semantic sealed source cannot be a symbolic link")
    source_path = source_input.resolve()
    if source_path == snapshot_path.resolve(strict=False):
        raise V5PipelineError("semantic sealed source cannot be the guarded snapshot path")
    if not workspace.sealed_open_path.exists() and (
        snapshot_path.exists()
        or snapshot_path.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise V5PipelineError("semantic sealed snapshot or open receipt already exists")

    def validate_and_publish(payload: bytes) -> None:
        load_semantic_snapshot_bytes(payload, benchmark)
        _write_exclusive_bytes(snapshot_path, payload, mode=0o444)
        _write_exclusive_bytes(
            receipt_path,
            _canonical_json_bytes(receipt.to_dict(), indent=2),
            mode=0o444,
        )

    marker_metadata = {
        "pipeline_id": workspace.config.pipeline_id,
        "code_sha256": workspace.config.code_sha256,
        "semantic_split_sha256": workspace.config.split_sha256["semantic_sealed_test"],
        "snapshot_path": snapshot_relative,
        "open_receipt_sha256": receipt.content_sha256(),
    }
    guard = SemanticSealedGuard(workspace.sealed_open_path)
    try:
        guard.open_once(
            source_path,
            expected_payload_sha256=workspace.config.sealed_payload_sha256,
            receipt=gate_receipt,
            expected_manifest_sha256=workspace.config.benchmark_manifest_sha256,
            expected_validation_sha256=workspace.config.split_sha256["validation"],
            validate_payload=validate_and_publish,
            opened_metadata=marker_metadata,
        )
    except BenchmarkContractError as exc:
        raise V5PipelineError(str(exc)) from exc
    return load_semantic_open_receipt(workspace)


def load_semantic_open_receipt(
    workspace: V5PipelineWorkspace,
) -> tuple[V5SemanticOpenReceipt, Path]:
    """Load and verify the opened marker, receipt, and immutable semantic snapshot."""

    receipt_path = workspace.root / V5_SEMANTIC_OPEN_RECEIPT_PATH
    try:
        if receipt_path.is_symlink() or workspace.sealed_open_path.is_symlink():
            raise V5PipelineError("semantic open receipt and marker cannot be symbolic links")
        receipt_before = _file_signature(receipt_path)
        marker_before = _file_signature(workspace.sealed_open_path)
        if receipt_path.stat().st_mode & 0o222:
            raise V5PipelineError("semantic open receipt must remain read-only")
        receipt = V5SemanticOpenReceipt.from_dict(
            json.loads(receipt_path.read_text(encoding="utf-8"))
        )
        marker = json.loads(workspace.sealed_open_path.read_text(encoding="utf-8"))
        if workspace.sealed_open_path.stat().st_mode & 0o222:
            raise V5PipelineError("semantic opened marker must remain read-only")
        if (
            _file_signature(receipt_path) != receipt_before
            or _file_signature(workspace.sealed_open_path) != marker_before
        ):
            raise V5PipelineError("semantic open receipt or marker changed while reading")
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("semantic open receipt or marker is unavailable") from exc
    gate_receipt = ValidationGateReceipt(
        benchmark_manifest_sha256=receipt.benchmark_manifest_sha256,
        validation_dataset_sha256=receipt.validation_split_sha256,
        directions=tuple(
            DirectionValidationEvidence(
                direction=item.direction,
                passed=item.passed,
                report_sha256=item.validation_report_sha256,
                code_sha256=item.code_sha256,
                transport_weights_sha256=item.transport_weights_sha256,
                predictor_sha256=item.predictor_sha256,
                threshold_sha256=item.threshold_sha256,
            )
            for item in receipt.directions
        ),
    )
    errors = receipt.validate(workspace=workspace, gate_receipt=gate_receipt)
    if errors:
        raise V5PipelineError("; ".join(errors))
    expected_marker = {
        "schema_version": "goldenexperience.semantic_sealed_open.v1",
        "state": "opened",
        "payload_sha256": receipt.sealed_payload_sha256,
        "validation_receipt_sha256": receipt.validation_gate_receipt_sha256,
        "pipeline_id": receipt.pipeline_id,
        "code_sha256": receipt.code_sha256,
        "semantic_split_sha256": receipt.semantic_split_sha256,
        "snapshot_path": receipt.snapshot_path,
        "open_receipt_sha256": receipt.content_sha256(),
    }
    if marker != expected_marker:
        raise V5PipelineError("semantic opened marker binding changed")
    snapshot_path = workspace.root / receipt.snapshot_path
    try:
        before = _file_signature(snapshot_path)
        if snapshot_path.is_symlink() or snapshot_path.stat().st_mode & 0o222:
            raise V5PipelineError("semantic snapshot must be an immutable regular file")
        if sha256_file(snapshot_path) != receipt.sealed_payload_sha256:
            raise V5PipelineError("semantic snapshot checksum mismatch")
        if _file_signature(snapshot_path) != before:
            raise V5PipelineError("semantic snapshot changed while hashing")
    except V5PipelineError:
        raise
    except OSError as exc:
        raise V5PipelineError("semantic snapshot is unavailable") from exc
    return receipt, snapshot_path


def load_semantic_snapshot_bytes(
    payload: bytes,
    benchmark: PublicationBenchmarkManifest,
) -> tuple[tuple[GroupedPrefixRecord, RawBenchmarkSample], ...]:
    """Parse exactly the sealed semantic split without exposing it to generic collection."""

    expected = {
        item.sample_id: item for item in benchmark.records if item.split == "semantic_sealed_test"
    }
    observed: dict[str, RawBenchmarkSample] = {}
    try:
        text = payload.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise V5PipelineError(
                    f"semantic sealed payload line {line_number} is not a JSON object"
                )
            sample = RawBenchmarkSample.from_dict(raw)
            if sample.sample_id in observed:
                raise V5PipelineError(f"duplicate semantic sealed sample {sample.sample_id!r}")
            record = expected.get(sample.sample_id)
            if record is None:
                raise V5PipelineError(
                    f"semantic sealed sample {sample.sample_id!r} is outside the sealed split"
                )
            errors = sample.validate(record)
            if errors:
                raise V5PipelineError("; ".join(errors))
            observed[sample.sample_id] = sample
    except V5PipelineError:
        raise
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise V5PipelineError("semantic sealed payload is malformed") from exc
    if set(observed) != set(expected) or len(observed) != SPLIT_COUNTS["semantic_sealed_test"]:
        raise V5PipelineError("semantic sealed payload does not contain the complete sealed split")
    return tuple((expected[sample_id], observed[sample_id]) for sample_id in sorted(expected))


def _threshold_sha256(
    direction: str,
    threshold: float,
    calibration_manifest_sha256: str,
) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "direction": direction,
                "threshold": threshold,
                "risk_calibration_manifest_sha256": calibration_manifest_sha256,
            }
        )
    )


def _write_exclusive_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("failed to write semantic sealed artifact")
                remaining = remaining[written:]
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError as exc:
            raise V5PipelineError("semantic sealed artifact already exists") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=indent,
                sort_keys=True,
                separators=(",", ":") if indent is None else None,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V5PipelineError("semantic open metadata is not finite canonical JSON") from exc


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_probability(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )
