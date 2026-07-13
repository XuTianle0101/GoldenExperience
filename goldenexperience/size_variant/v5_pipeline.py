"""Resumable, content-addressed workspace for the selective KV v5 pipeline."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    REQUIRED_QWEN3_DIRECTIONS,
    SPLIT_COUNTS,
    PublicationBenchmarkManifest,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    sha256_file,
    sha256_named_files,
)

V5_PIPELINE_CONFIG_SCHEMA = "goldenexperience.v5_pipeline_config.v1"
V5_PIPELINE_STATE_SCHEMA = "goldenexperience.v5_pipeline_state.v1"
V5_PIPELINE_RECEIPT_SCHEMA = "goldenexperience.v5_pipeline_stage_receipt.v1"
V5_PIPELINE_SEALED_LOCK_SCHEMA = "goldenexperience.v5_pipeline_sealed_lock.v1"

COLLECTABLE_SPLITS = frozenset(SPLIT_COUNTS) - {"semantic_sealed_test"}
COLLECT_STAGES = {f"collect_{split}": split for split in COLLECTABLE_SPLITS}
PIPELINE_STAGES = frozenset(
    {
        *COLLECT_STAGES,
        "fit_transport",
        "evaluate_method_dev",
        "fit_risk",
        "calibrate",
        "validate",
        "semantic_sealed",
        "runtime_audit",
    }
)
STAGE_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "fit_transport": ("collect_transport_train",),
    "evaluate_method_dev": ("fit_transport", "collect_method_dev"),
    "fit_risk": ("fit_transport", "collect_selector_train"),
    "calibrate": ("fit_risk", "collect_risk_calibration"),
    "validate": ("evaluate_method_dev", "calibrate", "collect_validation"),
    "runtime_audit": ("semantic_sealed", "collect_runtime_audit"),
}
STAGE_SPLITS: Mapping[str, str] = {
    **COLLECT_STAGES,
    "fit_transport": "transport_train",
    "evaluate_method_dev": "method_dev",
    "fit_risk": "selector_train",
    "calibrate": "risk_calibration",
    "validate": "validation",
    "semantic_sealed": "semantic_sealed_test",
    "runtime_audit": "runtime_audit",
}
_LOGICAL_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SAFE_SUFFIXES = frozenset({".json", ".jsonl", ".safetensors", ".npz", ".csv"})


class V5PipelineError(RuntimeError):
    """Raised when pipeline state, ordering, or content identity is invalid."""


@dataclass(frozen=True)
class V5DirectionConfig:
    direction: str
    source_model_path: str
    target_model_path: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec

    def validate(self, *, tokenizer_sha256: str) -> list[str]:
        errors: list[str] = []
        if self.direction not in REQUIRED_QWEN3_DIRECTIONS:
            errors.append(f"unsupported pipeline direction {self.direction!r}")
        if not self.source_model_path or not self.target_model_path:
            errors.append(f"pipeline direction {self.direction!r} lacks model paths")
        errors.extend(f"{self.direction} source: {item}" for item in self.source.validate())
        errors.extend(f"{self.direction} target: {item}" for item in self.target.validate())
        if self.source.model_id == self.target.model_id:
            errors.append(f"{self.direction} source and target model ids must differ")
        if self.source.architecture != self.target.architecture:
            errors.append(f"{self.direction} source and target architectures differ")
        if self.source.tokenizer_sha256 != self.target.tokenizer_sha256:
            errors.append(f"{self.direction} source and target tokenizers differ")
        if self.source.tokenizer_sha256 != tokenizer_sha256:
            errors.append(f"{self.direction} tokenizer differs from the benchmark")
        if self.source.head_dim != self.target.head_dim:
            errors.append(f"{self.direction} source and target head dimensions differ")
        if self.source.rope_scaling != self.target.rope_scaling:
            errors.append(f"{self.direction} source and target RoPE scaling differs")
        if self.source.sliding_window != self.target.sliding_window:
            errors.append(f"{self.direction} source and target sliding windows differ")
        expected_order = {
            "qwen3_4b_to_8b": -1,
            "qwen3_8b_to_4b": 1,
            "qwen3_8b_to_14b": -1,
            "qwen3_14b_to_8b": 1,
        }.get(self.direction)
        observed_order = (
            -1
            if self.source.parameter_count_b < self.target.parameter_count_b
            else 1
            if self.source.parameter_count_b > self.target.parameter_count_b
            else 0
        )
        if expected_order is not None and observed_order != expected_order:
            errors.append(f"{self.direction} has reversed or equal parameter counts")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "source_model_path": self.source_model_path,
            "target_model_path": self.target_model_path,
            "source": asdict(self.source),
            "target": asdict(self.target),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5DirectionConfig:
        return cls(
            direction=str(payload["direction"]),
            source_model_path=str(payload["source_model_path"]),
            target_model_path=str(payload["target_model_path"]),
            source=CachedKVModelSpec(**payload["source"]),
            target=CachedKVModelSpec(**payload["target"]),
        )


@dataclass(frozen=True)
class V5PipelineConfig:
    benchmark_manifest_uri: str
    benchmark_manifest_sha256: str
    benchmark_manifest_file_sha256: str
    split_sha256: Mapping[str, str]
    tokenizer_sha256: str
    chat_template_sha256: str
    sealed_payload_sha256: str
    code_sha256: str
    directions: tuple[V5DirectionConfig, ...]
    schema_version: str = V5_PIPELINE_CONFIG_SCHEMA

    @property
    def pipeline_id(self) -> str:
        return "v5-pipeline-" + _sha256_bytes(_canonical_json_bytes(self.identity_payload()))[:24]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_PIPELINE_CONFIG_SCHEMA:
            errors.append("unsupported v5 pipeline config schema")
        if not self.benchmark_manifest_uri:
            errors.append("pipeline benchmark manifest URI is required")
        for name in (
            "benchmark_manifest_sha256",
            "benchmark_manifest_file_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "sealed_payload_sha256",
            "code_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"pipeline {name} must be a SHA-256 digest")
        if set(self.split_sha256) != set(SPLIT_COUNTS):
            errors.append("pipeline split hash map is incomplete")
        elif any(not _is_sha256(value) for value in self.split_sha256.values()):
            errors.append("pipeline split hash map contains an invalid digest")
        elif len(set(self.split_sha256.values())) != len(self.split_sha256):
            errors.append("pipeline benchmark split identities must be distinct")
        directions = {item.direction for item in self.directions}
        if directions != REQUIRED_QWEN3_DIRECTIONS:
            errors.append("publication pipeline requires all four Qwen3 directions")
        if len(directions) != len(self.directions):
            errors.append("pipeline directions contain duplicates")
        for direction in self.directions:
            errors.extend(direction.validate(tokenizer_sha256=self.tokenizer_sha256))
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "split_sha256": dict(self.split_sha256),
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "sealed_payload_sha256": self.sealed_payload_sha256,
            "code_sha256": self.code_sha256,
            "directions": [
                {
                    "direction": item.direction,
                    "source": asdict(item.source),
                    "target": asdict(item.target),
                }
                for item in sorted(self.directions, key=lambda value: value.direction)
            ],
        }

    def direction(self, name: str) -> V5DirectionConfig:
        for direction in self.directions:
            if direction.direction == name:
                return direction
        raise V5PipelineError(f"pipeline direction {name!r} is not configured")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_manifest_uri": self.benchmark_manifest_uri,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "benchmark_manifest_file_sha256": self.benchmark_manifest_file_sha256,
            "split_sha256": dict(self.split_sha256),
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "sealed_payload_sha256": self.sealed_payload_sha256,
            "code_sha256": self.code_sha256,
            "directions": [
                item.to_dict()
                for item in sorted(self.directions, key=lambda value: value.direction)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5PipelineConfig:
        return cls(
            benchmark_manifest_uri=str(payload["benchmark_manifest_uri"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            benchmark_manifest_file_sha256=str(payload["benchmark_manifest_file_sha256"]),
            split_sha256=dict(payload["split_sha256"]),
            tokenizer_sha256=str(payload["tokenizer_sha256"]),
            chat_template_sha256=str(payload["chat_template_sha256"]),
            sealed_payload_sha256=str(payload["sealed_payload_sha256"]),
            code_sha256=str(payload["code_sha256"]),
            directions=tuple(
                V5DirectionConfig.from_dict(item) for item in payload.get("directions", ())
            ),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def from_benchmark(
        cls,
        manifest: PublicationBenchmarkManifest,
        *,
        manifest_path: str | Path,
        code_sha256: str,
        directions: Sequence[V5DirectionConfig],
    ) -> V5PipelineConfig:
        path = Path(manifest_path).resolve()
        return cls(
            benchmark_manifest_uri=str(path),
            benchmark_manifest_sha256=manifest.content_sha256(),
            benchmark_manifest_file_sha256=sha256_file(path),
            split_sha256=dict(manifest.split_sha256),
            tokenizer_sha256=manifest.tokenizer_sha256,
            chat_template_sha256=manifest.chat_template_sha256,
            sealed_payload_sha256=manifest.sealed_payload_sha256,
            code_sha256=code_sha256,
            directions=tuple(directions),
        )


@dataclass(frozen=True)
class PipelineArtifact:
    sha256: str
    path: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.sha256):
            errors.append("pipeline artifact digest is invalid")
        if not self.path or Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            errors.append("pipeline artifact path must be workspace-relative")
        if self.size_bytes < 0:
            errors.append("pipeline artifact size is invalid")
        if self.device < 0 or self.inode <= 0 or self.mtime_ns < 0:
            errors.append("pipeline artifact stat identity is invalid")
        if Path(self.path).stem != self.sha256:
            errors.append("pipeline artifact path is not content-addressed")
        return errors


@dataclass(frozen=True)
class PipelineStageRecord:
    direction: str
    stage: str
    status: str
    input_sha256: str
    attempt_id: str
    attempt_count: int
    started_at: str
    completed_at: str | None = None
    receipt_sha256: str | None = None
    receipt_path: str | None = None
    outputs: Mapping[str, PipelineArtifact] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.outputs is not None:
            payload["outputs"] = {
                name: asdict(artifact) for name, artifact in sorted(self.outputs.items())
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineStageRecord:
        raw_outputs = payload.get("outputs")
        outputs = (
            {name: PipelineArtifact(**value) for name, value in raw_outputs.items()}
            if isinstance(raw_outputs, Mapping)
            else None
        )
        return cls(
            direction=str(payload["direction"]),
            stage=str(payload["stage"]),
            status=str(payload["status"]),
            input_sha256=str(payload["input_sha256"]),
            attempt_id=str(payload["attempt_id"]),
            attempt_count=int(payload["attempt_count"]),
            started_at=str(payload["started_at"]),
            completed_at=payload.get("completed_at"),
            receipt_sha256=payload.get("receipt_sha256"),
            receipt_path=payload.get("receipt_path"),
            outputs=outputs,
            error_type=payload.get("error_type"),
            error_message=payload.get("error_message"),
        )


@dataclass(frozen=True)
class PipelineState:
    pipeline_id: str
    config_sha256: str
    stages: Mapping[str, PipelineStageRecord]
    created_at: str
    updated_at: str
    schema_version: str = V5_PIPELINE_STATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "config_sha256": self.config_sha256,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": {key: record.to_dict() for key, record in sorted(self.stages.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineState:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            config_sha256=str(payload["config_sha256"]),
            stages={
                key: PipelineStageRecord.from_dict(value)
                for key, value in payload.get("stages", {}).items()
            },
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            schema_version=str(payload.get("schema_version", "")),
        )


@dataclass(frozen=True)
class StageLease:
    direction: str
    stage: str
    input_sha256: str
    attempt_id: str
    reused: bool = False
    receipt_sha256: str | None = None


class V5PipelineWorkspace:
    """Own immutable config/objects and a lock-serialized mutable stage state."""

    def __init__(self, root: str | Path, config: V5PipelineConfig) -> None:
        self.root = Path(root).resolve()
        self.control = self.root / ".pipeline"
        self.config_path = self.control / "config.json"
        self.state_path = self.control / "state.json"
        self.lock_path = self.control / "state.lock"
        self.sealed_lock_path = self.control / "semantic_sealed.locked.json"
        self.objects_path = self.root / "objects"
        self.receipts_path = self.root / "receipts"
        self.config = config

    @classmethod
    def create(cls, root: str | Path, config: V5PipelineConfig) -> V5PipelineWorkspace:
        errors = config.validate()
        if errors:
            raise V5PipelineError("; ".join(errors))
        workspace = cls(root, config)
        workspace.control.mkdir(parents=True, exist_ok=True)
        workspace.objects_path.mkdir(parents=True, exist_ok=True)
        workspace.receipts_path.mkdir(parents=True, exist_ok=True)
        workspace.lock_path.touch(exist_ok=True)
        config_payload = config.to_dict()
        if workspace.config_path.exists():
            observed = json.loads(workspace.config_path.read_text(encoding="utf-8"))
            if observed != config_payload:
                raise V5PipelineError("pipeline workspace already has a different config")
        else:
            _write_exclusive_json(workspace.config_path, config_payload, mode=0o444)
        sealed_payload = workspace._expected_sealed_lock()
        if workspace.sealed_lock_path.exists():
            observed = json.loads(workspace.sealed_lock_path.read_text(encoding="utf-8"))
            if observed != sealed_payload:
                raise V5PipelineError("pipeline sealed lock is missing or changed")
        else:
            _write_exclusive_json(workspace.sealed_lock_path, sealed_payload, mode=0o444)
        with workspace._locked():
            if not workspace.state_path.exists():
                now = _utc_now()
                workspace._write_state(
                    PipelineState(
                        pipeline_id=config.pipeline_id,
                        config_sha256=config.content_sha256(),
                        stages={},
                        created_at=now,
                        updated_at=now,
                    )
                )
        return workspace.open(root)

    @classmethod
    def open(cls, root: str | Path) -> V5PipelineWorkspace:
        resolved = Path(root).resolve()
        config_path = resolved / ".pipeline" / "config.json"
        try:
            config = V5PipelineConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("pipeline config is missing or invalid") from exc
        errors = config.validate()
        if errors:
            raise V5PipelineError("; ".join(errors))
        workspace = cls(resolved, config)
        if workspace.config_path.stat().st_mode & 0o222:
            raise V5PipelineError("pipeline config must remain read-only")
        workspace._verify_external_bindings()
        try:
            sealed_lock = json.loads(workspace.sealed_lock_path.read_text(encoding="utf-8"))
            sealed_writable = bool(workspace.sealed_lock_path.stat().st_mode & 0o222)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("pipeline semantic sealed lock is unavailable") from exc
        if sealed_writable or sealed_lock != workspace._expected_sealed_lock():
            raise V5PipelineError("pipeline semantic sealed lock is not intact")
        with workspace._locked():
            workspace._load_state()
        return workspace

    def state(self) -> PipelineState:
        with self._locked():
            return self._load_state()

    def begin_stage(
        self,
        direction: str,
        stage: str,
        *,
        parameters: Mapping[str, Any],
        resume: bool = False,
    ) -> StageLease:
        self.config.direction(direction)
        if stage not in PIPELINE_STAGES:
            raise V5PipelineError(f"unknown v5 pipeline stage {stage!r}")
        if stage == "semantic_sealed":
            raise V5PipelineError(
                "semantic sealed access requires the explicit four-direction validation guard"
            )
        _canonical_json_bytes(parameters)
        with self._locked():
            state = self._load_state()
            dependencies: dict[str, str] = {}
            for dependency in STAGE_DEPENDENCIES.get(stage, ()):
                key = _stage_key(direction, dependency)
                record = state.stages.get(key)
                if record is None or record.status != "completed" or not record.receipt_sha256:
                    raise V5PipelineError(
                        f"stage {stage} requires completed dependency {dependency}"
                    )
                dependencies[dependency] = _stage_output_binding(record)
            split = STAGE_SPLITS.get(stage)
            input_payload = {
                "pipeline_id": self.config.pipeline_id,
                "code_sha256": self.config.code_sha256,
                "benchmark_manifest_sha256": self.config.benchmark_manifest_sha256,
                "direction": direction,
                "stage": stage,
                "split": split,
                "split_sha256": self.config.split_sha256.get(split) if split else None,
                "dependencies": dependencies,
                "parameters": dict(parameters),
            }
            input_sha256 = _sha256_bytes(_canonical_json_bytes(input_payload))
            key = _stage_key(direction, stage)
            previous = state.stages.get(key)
            if previous is not None and previous.status == "completed":
                if previous.input_sha256 != input_sha256:
                    raise V5PipelineError(
                        "completed stage input changed; create a new content-bound workspace"
                    )
                return StageLease(
                    direction=direction,
                    stage=stage,
                    input_sha256=input_sha256,
                    attempt_id=previous.attempt_id,
                    reused=True,
                    receipt_sha256=previous.receipt_sha256,
                )
            if previous is not None:
                if not resume:
                    raise V5PipelineError(
                        f"stage {stage} is {previous.status}; pass resume=True to reclaim it"
                    )
                if previous.input_sha256 != input_sha256:
                    raise V5PipelineError("resumed stage input differs from its first attempt")
            attempt_id = uuid.uuid4().hex
            now = _utc_now()
            record = PipelineStageRecord(
                direction=direction,
                stage=stage,
                status="running",
                input_sha256=input_sha256,
                attempt_id=attempt_id,
                attempt_count=(previous.attempt_count + 1 if previous else 1),
                started_at=now,
            )
            stages = dict(state.stages)
            stages[key] = record
            self._write_state(replace(state, stages=stages, updated_at=now))
            return StageLease(direction, stage, input_sha256, attempt_id)

    def complete_stage(
        self,
        lease: StageLease,
        *,
        outputs: Mapping[str, str | Path],
        metadata: Mapping[str, Any],
    ) -> PipelineStageRecord:
        if lease.reused:
            raise V5PipelineError("a reused completed stage cannot be completed again")
        if not outputs:
            raise V5PipelineError("pipeline stage completion requires at least one output")
        _canonical_json_bytes(metadata)
        artifacts = {
            name: self.publish_file(path, logical_name=name) for name, path in outputs.items()
        }
        receipt = {
            "schema_version": V5_PIPELINE_RECEIPT_SCHEMA,
            "pipeline_id": self.config.pipeline_id,
            "direction": lease.direction,
            "stage": lease.stage,
            "input_sha256": lease.input_sha256,
            "attempt_id": lease.attempt_id,
            "completed_at": _utc_now(),
            "outputs": {name: asdict(artifact) for name, artifact in sorted(artifacts.items())},
            "metadata": dict(metadata),
        }
        receipt_raw = _canonical_json_bytes(receipt, indent=2)
        receipt_sha256 = _sha256_bytes(receipt_raw)
        receipt_path = self.receipts_path / lease.direction / lease.stage / f"{receipt_sha256}.json"
        _write_exclusive_bytes(receipt_path, receipt_raw, mode=0o444, allow_identical=True)
        relative_receipt = receipt_path.relative_to(self.root).as_posix()
        with self._locked():
            state = self._load_state()
            key = _stage_key(lease.direction, lease.stage)
            current = state.stages.get(key)
            current = self._require_current_lease(current, lease)
            completed_at = str(receipt["completed_at"])
            completed = replace(
                current,
                status="completed",
                completed_at=completed_at,
                receipt_sha256=receipt_sha256,
                receipt_path=relative_receipt,
                outputs=artifacts,
                error_type=None,
                error_message=None,
            )
            stages = dict(state.stages)
            stages[key] = completed
            self._write_state(replace(state, stages=stages, updated_at=completed_at))
            return completed

    def fail_stage(self, lease: StageLease, error: BaseException) -> PipelineStageRecord:
        if lease.reused:
            raise V5PipelineError("a reused completed stage cannot fail")
        with self._locked():
            state = self._load_state()
            key = _stage_key(lease.direction, lease.stage)
            current = state.stages.get(key)
            current = self._require_current_lease(current, lease)
            now = _utc_now()
            failed = replace(
                current,
                status="failed",
                completed_at=now,
                error_type=type(error).__name__,
                error_message=str(error)[:512],
            )
            stages = dict(state.stages)
            stages[key] = failed
            self._write_state(replace(state, stages=stages, updated_at=now))
            return failed

    def publish_file(self, source: str | Path, *, logical_name: str) -> PipelineArtifact:
        if not _LOGICAL_NAME.fullmatch(logical_name):
            raise V5PipelineError(f"invalid pipeline output name {logical_name!r}")
        path = Path(source)
        if not path.is_file():
            raise V5PipelineError(f"pipeline output {path} is missing")
        digest = sha256_file(path)
        suffix = path.suffix.lower() if path.suffix.lower() in _SAFE_SUFFIXES else ".blob"
        destination = self.objects_path / digest[:2] / f"{digest}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != digest:
                raise V5PipelineError("content-addressed pipeline object is corrupt")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output, path.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256_file(temporary) != digest:
                    raise V5PipelineError("pipeline output changed while publishing")
                os.chmod(temporary, 0o444)
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    if sha256_file(destination) != digest:
                        raise V5PipelineError(
                            "pipeline object publication raced with corruption"
                        ) from None
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
        stat = destination.stat()
        return PipelineArtifact(
            sha256=digest,
            path=destination.relative_to(self.root).as_posix(),
            size_bytes=stat.st_size,
            device=stat.st_dev,
            inode=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )

    def artifact_path(
        self,
        artifact: PipelineArtifact,
        *,
        verify_hash: bool = True,
    ) -> Path:
        errors = artifact.validate()
        if errors:
            raise V5PipelineError("; ".join(errors))
        path = self._workspace_file(artifact.path)
        try:
            stat = path.stat()
        except OSError as exc:
            raise V5PipelineError("pipeline artifact is unavailable") from exc
        observed_stat = (stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns)
        expected_stat = (
            artifact.size_bytes,
            artifact.device,
            artifact.inode,
            artifact.mtime_ns,
        )
        if observed_stat != expected_stat or stat.st_mode & 0o222:
            raise V5PipelineError("pipeline artifact stat identity changed")
        if verify_hash and sha256_file(path) != artifact.sha256:
            raise V5PipelineError("pipeline artifact checksum mismatch")
        return path

    def _verify_external_bindings(self) -> None:
        benchmark = Path(self.config.benchmark_manifest_uri)
        if not benchmark.is_file():
            raise V5PipelineError("bound benchmark manifest is unavailable")
        if sha256_file(benchmark) != self.config.benchmark_manifest_file_sha256:
            raise V5PipelineError("bound benchmark manifest file changed")

    def _load_state(self) -> PipelineState:
        try:
            state = PipelineState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("pipeline state is missing or invalid") from exc
        if state.schema_version != V5_PIPELINE_STATE_SCHEMA:
            raise V5PipelineError("unsupported pipeline state schema")
        if state.pipeline_id != self.config.pipeline_id:
            raise V5PipelineError("pipeline state belongs to a different config")
        if state.config_sha256 != self.config.content_sha256():
            raise V5PipelineError("pipeline state config hash mismatch")
        for key, record in state.stages.items():
            if key != _stage_key(record.direction, record.stage):
                raise V5PipelineError("pipeline state contains an invalid stage key")
            if record.direction not in REQUIRED_QWEN3_DIRECTIONS:
                raise V5PipelineError("pipeline state contains an unknown direction")
            if record.stage not in PIPELINE_STAGES:
                raise V5PipelineError("pipeline state contains an unknown stage")
            if record.status not in {"running", "failed", "completed"}:
                raise V5PipelineError("pipeline state contains an invalid status")
            if not _is_sha256(record.input_sha256) or record.attempt_count <= 0:
                raise V5PipelineError("pipeline stage input or attempt count is invalid")
            if record.status == "completed":
                if not _is_sha256(record.receipt_sha256) or not record.receipt_path:
                    raise V5PipelineError("completed pipeline stage lacks a receipt")
                if record.outputs is None:
                    raise V5PipelineError("completed pipeline stage lacks outputs")
                errors = [error for item in record.outputs.values() for error in item.validate()]
                if errors:
                    raise V5PipelineError("; ".join(errors))
                receipt = self._workspace_file(record.receipt_path)
                if receipt.stat().st_mode & 0o222:
                    raise V5PipelineError("pipeline stage receipt must remain read-only")
                if sha256_file(receipt) != record.receipt_sha256:
                    raise V5PipelineError("pipeline stage receipt checksum mismatch")
                for artifact in record.outputs.values():
                    self.artifact_path(artifact, verify_hash=False)
        return state

    def _write_state(self, state: PipelineState) -> None:
        _write_replace_json(self.state_path, state.to_dict(), mode=0o600)

    def _expected_sealed_lock(self) -> dict[str, Any]:
        return {
            "schema_version": V5_PIPELINE_SEALED_LOCK_SCHEMA,
            "state": "locked",
            "sealed_payload_sha256": self.config.sealed_payload_sha256,
            "required_directions": sorted(REQUIRED_QWEN3_DIRECTIONS),
            "reason": "all four validation stages must pass before one-shot access",
        }

    def _workspace_file(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise V5PipelineError("pipeline state references a path outside the workspace")
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise V5PipelineError("pipeline state path escapes through a symbolic link")
        if not resolved.is_file():
            raise V5PipelineError("pipeline state references a missing file")
        return resolved

    @staticmethod
    def _require_current_lease(
        current: PipelineStageRecord | None,
        lease: StageLease,
    ) -> PipelineStageRecord:
        if current is None or current.status != "running":
            raise V5PipelineError("pipeline stage lease is no longer running")
        if current.attempt_id != lease.attempt_id or current.input_sha256 != lease.input_sha256:
            raise V5PipelineError("pipeline stage lease was superseded")
        return current

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def source_tree_sha256(repository_root: str | Path) -> str:
    """Hash executable source inputs without relying on a clean Git checkout."""

    root = Path(repository_root).resolve()
    paths = [root / "pyproject.toml"]
    paths.extend(sorted((root / "goldenexperience").rglob("*.py")))
    marker = root / "goldenexperience" / "py.typed"
    if marker.is_file():
        paths.append(marker)
    scripts = root / "scripts"
    if scripts.is_dir():
        paths.extend(
            sorted(
                path
                for path in scripts.rglob("*")
                if path.is_file() and path.suffix in {".py", ".sh"}
            )
        )
    return sha256_named_files(paths, root=root)


def _stage_key(direction: str, stage: str) -> str:
    return f"{direction}/{stage}"


def _stage_output_binding(record: PipelineStageRecord) -> str:
    if record.status != "completed" or record.outputs is None:
        raise V5PipelineError("pipeline dependency is not complete")
    payload = {
        "direction": record.direction,
        "stage": record.stage,
        "input_sha256": record.input_sha256,
        "outputs": {
            name: {"sha256": artifact.sha256, "size_bytes": artifact.size_bytes}
            for name, artifact in sorted(record.outputs.items())
        },
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


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
        raise V5PipelineError("pipeline metadata must be finite canonical JSON") from exc


def _write_exclusive_json(path: Path, value: Any, *, mode: int) -> None:
    _write_exclusive_bytes(path, _canonical_json_bytes(value, indent=2), mode=mode)


def _write_exclusive_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    allow_identical: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if not allow_identical or path.read_bytes() != payload:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _write_replace_json(path: Path, value: Any, *, mode: int) -> None:
    payload = _canonical_json_bytes(value, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("failed to write pipeline artifact")
        remaining = remaining[written:]


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
