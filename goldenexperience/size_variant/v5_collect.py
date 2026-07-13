"""Split-isolated real-model trace collection for the selective KV v5 pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from goldenexperience.benchmarks.publication import (
    SPLIT_COUNTS,
    TRACE_ONLY_DATASETS,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.benchmarks.publication_eval import validate_publication_evaluation
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.attention_collection import (
    TargetAttentionCollector,
    causal_sample_mask,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    canonicalize_safetensors_header,
    sha256_file,
    verify_model_path,
)
from goldenexperience.size_variant.head_aware_transport import dynamic_cache_to_head_object
from goldenexperience.size_variant.v5_pipeline import (
    COLLECTABLE_SPLITS,
    PipelineArtifact,
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)

RAW_BENCHMARK_SAMPLE_SCHEMA = "goldenexperience.publication_raw_sample.v1"
V5_TRACE_MANIFEST_SCHEMA = "goldenexperience.v5_trace_manifest.v2"
V5_TRACE_SHARD_SCHEMA = "goldenexperience.v5_trace_shard.v2"
V5_TRACE_CHECKPOINT_SCHEMA = "goldenexperience.v5_trace_checkpoint.v2"
V5_COLLECTOR_ID = "qwen3_grouped_prefix_attention_trace_v2"
REQUIRED_TRACE_TENSORS = frozenset(
    {
        "source_kv",
        "target_kv",
        "target_query",
        "native_attention_output",
        "full_native_attention_output",
        "query_positions",
        "key_positions",
        "causal_mask",
        "constant_losses",
    }
)


@dataclass(frozen=True)
class RawBenchmarkSample:
    sample_id: str
    prefix_text: str
    suffix_query: str
    reference: Any
    evaluation: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema_version: str = RAW_BENCHMARK_SAMPLE_SCHEMA

    def validate(self, record: GroupedPrefixRecord) -> list[str]:
        errors: list[str] = []
        if self.schema_version != RAW_BENCHMARK_SAMPLE_SCHEMA:
            errors.append(f"raw sample {self.sample_id!r} has an unsupported schema")
        if self.sample_id != record.sample_id:
            errors.append("raw sample id does not match its benchmark record")
        if not self.prefix_text:
            errors.append(f"raw sample {self.sample_id!r} has an empty prefix")
        if not self.suffix_query:
            errors.append(f"raw sample {self.sample_id!r} has an empty suffix/query")
        if record.dataset_id not in TRACE_ONLY_DATASETS and self.reference is None:
            errors.append(f"semantic raw sample {self.sample_id!r} lacks a reference")
        if record.dataset_id not in TRACE_ONLY_DATASETS:
            errors.extend(
                f"raw sample {self.sample_id!r} evaluation: {error}"
                for error in validate_publication_evaluation(self.reference, self.evaluation)
            )
        try:
            _canonical_json_bytes(self.reference)
            _canonical_json_bytes(dict(self.evaluation))
            _canonical_json_bytes(dict(self.provenance))
        except V5PipelineError as exc:
            errors.append(f"raw sample {self.sample_id!r} is not canonical JSON: {exc}")
        if _text_sha256(self.prefix_text) != record.prefix_sha256:
            errors.append(f"raw sample {self.sample_id!r} prefix hash mismatch")
        if _text_sha256(self.suffix_query) != record.suffix_query_sha256:
            errors.append(f"raw sample {self.sample_id!r} suffix/query hash mismatch")
        try:
            observed_content = publication_sample_content_sha256(
                prefix_text=self.prefix_text,
                suffix_query=self.suffix_query,
                reference=self.reference,
                evaluation=self.evaluation,
                task=record.task,
            )
            if observed_content != record.content_sha256:
                errors.append(f"raw sample {self.sample_id!r} content hash mismatch")
        except V5PipelineError as exc:
            errors.append(f"raw sample {self.sample_id!r} content is invalid: {exc}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "prefix_text": self.prefix_text,
            "suffix_query": self.suffix_query,
            "reference": self.reference,
            "evaluation": dict(self.evaluation),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RawBenchmarkSample:
        return cls(
            sample_id=str(payload["sample_id"]),
            prefix_text=str(payload["prefix_text"]),
            suffix_query=str(payload["suffix_query"]),
            reference=payload.get("reference"),
            evaluation=dict(payload.get("evaluation", {})),
            provenance=dict(payload.get("provenance", {})),
            schema_version=str(payload.get("schema_version", "")),
        )


def publication_sample_content_sha256(
    *,
    prefix_text: str,
    suffix_query: str,
    reference: Any,
    evaluation: Mapping[str, Any],
    task: str,
) -> str:
    """Hash every semantic field while excluding source-location provenance."""

    payload = {
        "schema_version": RAW_BENCHMARK_SAMPLE_SCHEMA,
        "prefix_text": prefix_text,
        "suffix_query": suffix_query,
        "reference": reference,
        "evaluation": dict(evaluation),
        "task": task,
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def load_raw_sample_store(
    path: str | Path,
    manifest: PublicationBenchmarkManifest,
    *,
    split: str,
) -> tuple[tuple[GroupedPrefixRecord, RawBenchmarkSample], ...]:
    """Load exactly one non-sealed split and verify every raw/hash-only pair."""

    if split not in COLLECTABLE_SPLITS:
        raise V5PipelineError(f"split {split!r} is not available to the generic collector")
    expected = {record.sample_id: record for record in manifest.records if record.split == split}
    observed: dict[str, RawBenchmarkSample] = {}
    store_path = Path(path)
    try:
        with store_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise V5PipelineError(f"{store_path}:{line_number} must contain a JSON object")
                sample = RawBenchmarkSample.from_dict(payload)
                if sample.sample_id in observed:
                    raise V5PipelineError(f"duplicate raw sample {sample.sample_id!r}")
                record = expected.get(sample.sample_id)
                if record is None:
                    raise V5PipelineError(
                        f"raw sample {sample.sample_id!r} is outside requested split {split}"
                    )
                errors = sample.validate(record)
                if errors:
                    raise V5PipelineError("; ".join(errors))
                observed[sample.sample_id] = sample
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        if isinstance(exc, V5PipelineError):
            raise
        raise V5PipelineError("raw sample store is unreadable or malformed") from exc
    missing = set(expected) - set(observed)
    if missing:
        raise V5PipelineError(
            f"raw sample store is missing {len(missing)} records from split {split}"
        )
    if len(observed) != SPLIT_COUNTS[split]:
        raise V5PipelineError(
            f"raw sample store must contain exactly {SPLIT_COUNTS[split]} {split} records"
        )
    return tuple((expected[sample_id], observed[sample_id]) for sample_id in sorted(expected))


@dataclass(frozen=True)
class TraceObjectRef:
    sha256: str
    path: str
    size_bytes: int

    @classmethod
    def from_artifact(cls, artifact: PipelineArtifact) -> TraceObjectRef:
        return cls(artifact.sha256, artifact.path, artifact.size_bytes)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.sha256):
            errors.append("trace object hash is invalid")
        path = Path(self.path)
        if path.is_absolute() or ".." in path.parts or path.stem != self.sha256:
            errors.append("trace object path is not content-addressed")
        if self.size_bytes <= 0:
            errors.append("trace object size must be positive")
        return errors


@dataclass(frozen=True)
class TraceRecord:
    sample_id: str
    prefix_group_id: str
    dataset_id: str
    task: str
    token_bucket: int
    content_sha256: str
    prefix_sha256: str
    suffix_query_sha256: str
    token_ids_sha256: str
    token_count: int
    query_sample_count: int
    key_sample_count: int
    shard: TraceObjectRef

    def validate(self, record: GroupedPrefixRecord) -> list[str]:
        errors = self.shard.validate()
        expected = {
            "sample_id": record.sample_id,
            "prefix_group_id": record.prefix_group_id,
            "dataset_id": record.dataset_id,
            "task": record.task,
            "token_bucket": record.token_bucket,
            "content_sha256": record.content_sha256,
            "prefix_sha256": record.prefix_sha256,
            "suffix_query_sha256": record.suffix_query_sha256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"trace record {self.sample_id!r} has mismatched {name}")
        if not _is_sha256(self.token_ids_sha256):
            errors.append(f"trace record {self.sample_id!r} has an invalid token hash")
        if self.token_count != record.token_bucket:
            errors.append(f"trace record {self.sample_id!r} does not match its token bucket")
        if self.query_sample_count <= 0 or self.key_sample_count <= 0:
            errors.append(f"trace record {self.sample_id!r} has empty attention samples")
        if self.query_sample_count > self.token_count or self.key_sample_count > self.token_count:
            errors.append(f"trace record {self.sample_id!r} exceeds its token count")
        return errors


def _trace_group_binding(record: TraceRecord) -> tuple[Any, ...]:
    return (
        record.prefix_group_id,
        record.prefix_sha256,
        record.token_ids_sha256,
        record.token_count,
        record.query_sample_count,
        record.key_sample_count,
        record.shard,
    )


def trace_shard_metadata(
    record: TraceRecord,
    source: CachedKVModelSpec,
    target: CachedKVModelSpec,
) -> dict[str, str]:
    return {
        "schema_version": V5_TRACE_SHARD_SCHEMA,
        "collector_id": V5_COLLECTOR_ID,
        "prefix_group_id": record.prefix_group_id,
        "prefix_sha256": record.prefix_sha256,
        "token_ids_sha256": record.token_ids_sha256,
        "token_count": str(record.token_count),
        "query_sample_count": str(record.query_sample_count),
        "key_sample_count": str(record.key_sample_count),
        "source_config_sha256": source.config_sha256,
        "source_weights_sha256": source.weights_sha256,
        "target_config_sha256": target.config_sha256,
        "target_weights_sha256": target.weights_sha256,
    }


def load_trace_shard(
    path: str | Path,
    record: TraceRecord,
    *,
    source: CachedKVModelSpec,
    target: CachedKVModelSpec,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Load one immutable shard and verify its full tensor/metadata contract."""

    import torch
    from safetensors import safe_open

    shard_path = Path(path)
    try:
        stat = shard_path.stat()
    except OSError as exc:
        raise V5PipelineError("trace shard is unavailable") from exc
    if stat.st_size != record.shard.size_bytes or stat.st_mode & 0o222:
        raise V5PipelineError("trace shard size or read-only mode changed")
    before = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if verify_hash and sha256_file(shard_path) != record.shard.sha256:
        raise V5PipelineError("trace shard checksum mismatch")
    tensors: dict[str, Any] = {}
    try:
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != REQUIRED_TRACE_TENSORS:
                raise V5PipelineError("trace shard tensor set is invalid")
            if (handle.metadata() or {}) != trace_shard_metadata(record, source, target):
                raise V5PipelineError("trace shard metadata binding is invalid")
            for name in handle.keys():  # noqa: SIM118
                tensors[name] = handle.get_tensor(name)
    except V5PipelineError:
        raise
    except Exception as exc:
        raise V5PipelineError("trace shard safetensors payload is invalid") from exc
    key_count = record.key_sample_count
    query_count = record.query_sample_count
    expected_shapes = {
        "source_kv": (
            2,
            source.num_layers,
            source.num_key_value_heads,
            key_count,
            source.head_dim,
        ),
        "target_kv": (
            2,
            target.num_layers,
            target.num_key_value_heads,
            key_count,
            target.head_dim,
        ),
        "query_positions": (query_count,),
        "key_positions": (key_count,),
        "causal_mask": (1, 1, query_count, key_count),
        "constant_losses": (2,),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise V5PipelineError(f"trace shard {name} shape is invalid")
    query_shape = tuple(tensors["target_query"].shape)
    if (
        len(query_shape) != 4
        or query_shape[0] != target.num_layers
        or query_shape[1] <= 0
        or query_shape[1] % target.num_key_value_heads
        or query_shape[2:] != (query_count, target.head_dim)
    ):
        raise V5PipelineError("trace shard target query shape is invalid")
    for name in ("native_attention_output", "full_native_attention_output"):
        if tuple(tensors[name].shape) != query_shape:
            raise V5PipelineError(f"trace shard {name} shape is invalid")
    expected_dtypes = {
        "source_kv": _torch_dtype(source.dtype),
        "target_kv": _torch_dtype(target.dtype),
        "target_query": _torch_dtype(target.dtype),
        "full_native_attention_output": _torch_dtype(target.dtype),
        "native_attention_output": torch.float32,
        "query_positions": torch.int64,
        "key_positions": torch.int64,
        "causal_mask": torch.bool,
        "constant_losses": torch.float32,
    }
    for name, dtype in expected_dtypes.items():
        if tensors[name].dtype != dtype:
            raise V5PipelineError(f"trace shard {name} dtype is invalid")
    query_positions = tensors["query_positions"].long()
    key_positions = tensors["key_positions"].long()
    if bool((query_positions < 0).any()) or bool((query_positions >= record.token_count).any()):
        raise V5PipelineError("trace shard query positions are out of range")
    if bool((key_positions < 0).any()) or bool((key_positions >= record.token_count).any()):
        raise V5PipelineError("trace shard key positions are out of range")
    expected_mask = causal_sample_mask(query_positions, key_positions)
    if not torch.equal(tensors["causal_mask"].bool(), expected_mask):
        raise V5PipelineError("trace shard causal mask is inconsistent with positions")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise V5PipelineError("trace shard contains non-finite tensors")
    after_stat = shard_path.stat()
    after = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
        after_stat.st_ctime_ns,
    )
    if after != before:
        raise V5PipelineError("trace shard changed while loading")
    if verify_hash and sha256_file(shard_path) != record.shard.sha256:
        raise V5PipelineError("trace shard changed while hashing")
    return tensors


@dataclass(frozen=True)
class V5TraceManifest:
    pipeline_id: str
    direction: str
    split: str
    split_sha256: str
    benchmark_manifest_sha256: str
    code_sha256: str
    raw_sample_store_sha256: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec
    collector: Mapping[str, Any]
    records: tuple[TraceRecord, ...]
    schema_version: str = V5_TRACE_MANIFEST_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        benchmark: PublicationBenchmarkManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_TRACE_MANIFEST_SCHEMA:
            errors.append("unsupported v5 trace manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("trace manifest belongs to a different pipeline")
        if self.direction not in {item.direction for item in workspace.config.directions}:
            errors.append("trace manifest direction is not configured")
        if self.split not in COLLECTABLE_SPLITS:
            errors.append("trace manifest split is not collectable")
        if self.split_sha256 != workspace.config.split_sha256.get(self.split):
            errors.append("trace manifest split hash mismatch")
        if self.benchmark_manifest_sha256 != workspace.config.benchmark_manifest_sha256:
            errors.append("trace manifest benchmark hash mismatch")
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("trace manifest code hash mismatch")
        if not _is_sha256(self.raw_sample_store_sha256):
            errors.append("trace manifest raw sample store hash is invalid")
        try:
            direction = workspace.config.direction(self.direction)
            if self.source != direction.source or self.target != direction.target:
                errors.append("trace manifest model identities differ from the pipeline")
        except V5PipelineError as exc:
            errors.append(str(exc))
        try:
            _canonical_json_bytes(dict(self.collector))
        except V5PipelineError as exc:
            errors.append(str(exc))
        expected = {
            record.sample_id: record for record in benchmark.records if record.split == self.split
        }
        if len(self.records) != SPLIT_COUNTS.get(self.split, -1):
            errors.append("trace manifest record count differs from the registered split")
        observed: set[str] = set()
        grouped: dict[str, tuple[Any, ...]] = {}
        shard_bindings: dict[str, tuple[Any, ...]] = {}
        for trace in self.records:
            if trace.sample_id in observed:
                errors.append(f"duplicate trace sample {trace.sample_id!r}")
                continue
            observed.add(trace.sample_id)
            record = expected.get(trace.sample_id)
            if record is None:
                errors.append(f"trace sample {trace.sample_id!r} is outside the split")
                continue
            errors.extend(trace.validate(record))
            binding = _trace_group_binding(trace)
            previous = grouped.setdefault(trace.prefix_group_id, binding)
            if previous != binding:
                errors.append(
                    f"prefix group {trace.prefix_group_id!r} refers to multiple trace objects"
                )
            previous_shard_binding = shard_bindings.setdefault(trace.shard.sha256, binding)
            if previous_shard_binding != binding:
                errors.append("one trace object is bound to multiple prefix identities")
        if observed != set(expected):
            errors.append("trace manifest sample ids differ from the frozen split")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "direction": self.direction,
            "split": self.split,
            "split_sha256": self.split_sha256,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "code_sha256": self.code_sha256,
            "raw_sample_store_sha256": self.raw_sample_store_sha256,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "collector": dict(self.collector),
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5TraceManifest:
        records = []
        for raw_record in payload.get("records", ()):
            record = dict(raw_record)
            record["shard"] = TraceObjectRef(**record["shard"])
            records.append(TraceRecord(**record))
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            split=str(payload["split"]),
            split_sha256=str(payload["split_sha256"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            code_sha256=str(payload["code_sha256"]),
            raw_sample_store_sha256=str(payload["raw_sample_store_sha256"]),
            source=CachedKVModelSpec(**payload["source"]),
            target=CachedKVModelSpec(**payload["target"]),
            collector=dict(payload["collector"]),
            records=tuple(records),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        benchmark: PublicationBenchmarkManifest,
    ) -> V5TraceManifest:
        try:
            manifest = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("trace manifest is unreadable or malformed") from exc
        errors = manifest.validate(workspace=workspace, benchmark=benchmark)
        if errors:
            raise V5PipelineError("; ".join(errors))
        return manifest


@dataclass(frozen=True)
class CollectedTrace:
    path: Path
    token_ids_sha256: str
    token_count: int
    query_sample_count: int
    key_sample_count: int


class TraceCollector(Protocol):
    def __enter__(self) -> TraceCollector: ...

    def __exit__(self, *_args: object) -> None: ...

    def collect(
        self,
        record: GroupedPrefixRecord,
        sample: RawBenchmarkSample,
        output: Path,
    ) -> CollectedTrace: ...


def run_collect_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    split: str,
    sample_store_path: str | Path,
    collector_parameters: Mapping[str, Any],
    collector_factory: Callable[[], TraceCollector],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Collect one complete split with per-sample immutable checkpoints."""

    if split not in COLLECTABLE_SPLITS:
        raise V5PipelineError(f"split {split!r} is not collectable")
    benchmark = load_bound_benchmark(workspace)
    store_path = Path(sample_store_path)
    before = _stat_signature(store_path)
    store_sha256 = sha256_file(store_path)
    samples = load_raw_sample_store(store_path, benchmark, split=split)
    parameters = {
        "collector": dict(collector_parameters),
        "raw_sample_store_sha256": store_sha256,
        "record_count": len(samples),
    }
    stage = f"collect_{split}"
    lease = workspace.begin_stage(
        direction,
        stage,
        parameters=parameters,
        resume=resume,
    )
    if lease.reused:
        stage_record = workspace.state().stages[f"{direction}/{stage}"]
        return stage_record
    work = workspace.control / "work" / direction / stage
    checkpoints = work / "checkpoints"
    local_shards = work / "shards"
    checkpoints.mkdir(parents=True, exist_ok=True)
    local_shards.mkdir(parents=True, exist_ok=True)
    trace_records: list[TraceRecord] = []
    missing_by_group: dict[
        str,
        list[tuple[GroupedPrefixRecord, RawBenchmarkSample, Path]],
    ] = {}
    group_assets: dict[str, tuple[TraceRecord, PipelineArtifact]] = {}
    verified_artifacts: dict[str, PipelineArtifact] = {}
    try:
        for benchmark_record, sample in samples:
            checkpoint = checkpoints / f"{_text_sha256(benchmark_record.sample_id)}.json"
            restored = _load_trace_checkpoint(
                checkpoint,
                workspace=workspace,
                input_sha256=lease.input_sha256,
                record=benchmark_record,
                verified_artifacts=verified_artifacts,
            )
            if restored is None:
                missing_by_group.setdefault(benchmark_record.prefix_group_id, []).append(
                    (benchmark_record, sample, checkpoint)
                )
            else:
                trace, artifact = restored
                previous_group_asset = group_assets.setdefault(
                    benchmark_record.prefix_group_id,
                    (trace, artifact),
                )
                if _trace_group_binding(previous_group_asset[0]) != _trace_group_binding(trace):
                    raise V5PipelineError("trace checkpoints disagree within one prefix group")
                trace_records.append(trace)
        missing_count = sum(len(items) for items in missing_by_group.values())
        if missing_count:
            needs_collection = any(group not in group_assets for group in missing_by_group)
            completed = 0
            with ExitStack() as stack:
                collector = stack.enter_context(collector_factory()) if needs_collection else None
                for group_id, group_missing in sorted(missing_by_group.items()):
                    template = group_assets.get(group_id)
                    if template is None:
                        if collector is None:
                            raise V5PipelineError(
                                "trace collector was not opened for a missing group"
                            )
                        representative, sample, _ = group_missing[0]
                        shard_path = local_shards / f"{_text_sha256(group_id)}.safetensors"
                        collected = collector.collect(representative, sample, shard_path)
                        artifact = workspace.publish_file(
                            collected.path,
                            logical_name="trace_shard",
                        )
                        token_ids_sha256_value = collected.token_ids_sha256
                        token_count = collected.token_count
                        query_sample_count = collected.query_sample_count
                        key_sample_count = collected.key_sample_count
                    else:
                        previous_trace, artifact = template
                        token_ids_sha256_value = previous_trace.token_ids_sha256
                        token_count = previous_trace.token_count
                        query_sample_count = previous_trace.query_sample_count
                        key_sample_count = previous_trace.key_sample_count
                    for benchmark_record, _sample, checkpoint in group_missing:
                        trace_record = TraceRecord(
                            sample_id=benchmark_record.sample_id,
                            prefix_group_id=benchmark_record.prefix_group_id,
                            dataset_id=benchmark_record.dataset_id,
                            task=benchmark_record.task,
                            token_bucket=benchmark_record.token_bucket,
                            content_sha256=benchmark_record.content_sha256,
                            prefix_sha256=benchmark_record.prefix_sha256,
                            suffix_query_sha256=benchmark_record.suffix_query_sha256,
                            token_ids_sha256=token_ids_sha256_value,
                            token_count=token_count,
                            query_sample_count=query_sample_count,
                            key_sample_count=key_sample_count,
                            shard=TraceObjectRef.from_artifact(artifact),
                        )
                        errors = trace_record.validate(benchmark_record)
                        if errors:
                            raise V5PipelineError("; ".join(errors))
                        registered_group_asset = group_assets.setdefault(
                            group_id,
                            (trace_record, artifact),
                        )
                        if _trace_group_binding(registered_group_asset[0]) != _trace_group_binding(
                            trace_record
                        ):
                            raise V5PipelineError("collected prefix group trace is inconsistent")
                        _write_trace_checkpoint(
                            checkpoint,
                            input_sha256=lease.input_sha256,
                            record=trace_record,
                            artifact=artifact,
                        )
                        trace_records.append(trace_record)
                        completed += 1
                        if progress is not None:
                            progress(completed, missing_count, benchmark_record.sample_id)
        if _stat_signature(store_path) != before or sha256_file(store_path) != store_sha256:
            raise V5PipelineError("raw sample store changed during collection")
        direction_config = workspace.config.direction(direction)
        trace_manifest = V5TraceManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            split=split,
            split_sha256=workspace.config.split_sha256[split],
            benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
            code_sha256=workspace.config.code_sha256,
            raw_sample_store_sha256=store_sha256,
            source=direction_config.source,
            target=direction_config.target,
            collector=dict(collector_parameters),
            records=tuple(sorted(trace_records, key=lambda item: item.sample_id)),
        )
        errors = trace_manifest.validate(workspace=workspace, benchmark=benchmark)
        if errors:
            raise V5PipelineError("; ".join(errors))
        manifest_path = work / "trace_manifest.json"
        _write_json_replace(manifest_path, trace_manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={"trace_manifest": manifest_path},
            metadata={
                "record_count": len(trace_manifest.records),
                "trace_manifest_sha256": trace_manifest.content_sha256(),
                "raw_sample_store_sha256": store_sha256,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


class RealQwenTraceCollector:
    """Collect bounded source/target KV and target-attention tensors from real models."""

    def __init__(
        self,
        *,
        source_path: str | Path,
        target_path: str | Path,
        source: CachedKVModelSpec,
        target: CachedKVModelSpec,
        source_device: str,
        target_device: str,
        identity_cache_path: str | Path | None,
        max_queries: int = 32,
        max_keys: int = 256,
        logit_tail_tokens: int = 8,
        seed: int = 17,
        attention_implementation: str = "sdpa",
    ) -> None:
        self.source_path = Path(source_path).resolve()
        self.target_path = Path(target_path).resolve()
        self.source = source
        self.target = target
        self.source_device = source_device
        self.target_device = target_device
        self.identity_cache_path = identity_cache_path
        self.max_queries = max_queries
        self.max_keys = max_keys
        self.logit_tail_tokens = logit_tail_tokens
        self.seed = seed
        self.attention_implementation = attention_implementation
        self.tokenizer: Any | None = None
        self.source_model: Any | None = None
        self.target_model: Any | None = None

    def parameters(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "collector_id": V5_COLLECTOR_ID,
            "max_queries": self.max_queries,
            "max_keys": self.max_keys,
            "logit_tail_tokens": self.logit_tail_tokens,
            "seed": self.seed,
            "attention_implementation": self.attention_implementation,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "source_device_type": torch.device(self.source_device).type,
            "source_device_name": _device_name(self.source_device),
            "target_device_type": torch.device(self.target_device).type,
            "target_device_name": _device_name(self.target_device),
        }

    def __enter__(self) -> RealQwenTraceCollector:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for label, expected, path in (
            ("source", self.source, self.source_path),
            ("target", self.target, self.target_path),
        ):
            errors = verify_model_path(
                expected,
                path,
                identity_cache_path=self.identity_cache_path,
            )
            if errors:
                raise V5PipelineError(f"{label} model identity mismatch: {'; '.join(errors)}")
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.target_path,
            local_files_only=True,
        )
        self.source_model = AutoModelForCausalLM.from_pretrained(
            self.source_path,
            local_files_only=True,
            dtype=_torch_dtype(self.source.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.source_device},
        ).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_path,
            local_files_only=True,
            dtype=_torch_dtype(self.target.dtype),
            attn_implementation=self.attention_implementation,
            device_map={"": self.target_device},
        ).eval()
        return self

    def __exit__(self, *_args: object) -> None:
        import torch

        self.tokenizer = None
        self.source_model = None
        self.target_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def collect(
        self,
        record: GroupedPrefixRecord,
        sample: RawBenchmarkSample,
        output: Path,
    ) -> CollectedTrace:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import save_file

        if self.tokenizer is None or self.source_model is None or self.target_model is None:
            raise V5PipelineError("real trace collector is not loaded")
        encoded = self.tokenizer(
            sample.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        if int(encoded.shape[1]) < record.token_bucket:
            raise V5PipelineError(
                f"sample {record.sample_id!r} has fewer tokens than bucket {record.token_bucket}"
            )
        input_ids = encoded[:, : record.token_bucket]
        generation_tokens = min(self.logit_tail_tokens, record.token_bucket - 1)
        logits_to_keep = generation_tokens + 1
        with torch.inference_mode():
            source_output = self.source_model(
                input_ids=input_ids.to(self.source_device),
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
        with (
            TargetAttentionCollector(
                self.target_model,
                token_count=record.token_bucket,
                rope_theta=self.target.rope_theta,
                max_queries=self.max_queries,
                max_keys=self.max_keys,
                offload_to_cpu=True,
            ) as attention_collector,
            torch.inference_mode(),
        ):
            target_output = self.target_model(
                input_ids=input_ids.to(self.target_device),
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
        trace = attention_collector.trace()
        source_kv = dynamic_cache_to_head_object(source_output.past_key_values)
        target_kv = dynamic_cache_to_head_object(target_output.past_key_values)
        _validate_cache_shape(source_kv, self.source, record.token_bucket, label="source")
        _validate_cache_shape(target_kv, self.target, record.token_bucket, label="target")
        source_sample = source_kv.index_select(
            3,
            trace.key_positions.to(source_kv.device),
        ).to("cpu")
        target_sample = target_kv.index_select(
            3,
            trace.key_positions.to(target_kv.device),
        ).to("cpu")
        sampled_native_output = _sampled_attention_output(
            trace.queries,
            target_sample[0],
            target_sample[1],
            trace.causal_mask,
        )
        labels = input_ids[:, -generation_tokens:].to(self.target_device).reshape(-1)
        native_generation = functional.cross_entropy(
            target_output.logits[:, :-1]
            .float()
            .reshape(
                -1,
                target_output.logits.shape[-1],
            ),
            labels,
        )
        teacher = target_output.logits[:, -generation_tokens:].float().softmax(dim=-1)
        student_log = (
            source_output.logits[:, -generation_tokens:]
            .to(self.target_device)
            .float()
            .log_softmax(dim=-1)
        )
        prompt_tail = (
            functional.kl_div(student_log, teacher, reduction="batchmean") / generation_tokens
        )
        losses = torch.tensor(
            [native_generation.item(), prompt_tail.item()],
            dtype=torch.float32,
        )
        tensors = {
            "source_kv": source_sample.contiguous(),
            "target_kv": target_sample.contiguous(),
            "target_query": trace.queries.to("cpu").contiguous(),
            "native_attention_output": sampled_native_output.contiguous(),
            "full_native_attention_output": trace.attention_outputs.to("cpu").contiguous(),
            "query_positions": trace.query_positions.to("cpu").long().contiguous(),
            "key_positions": trace.key_positions.to("cpu").long().contiguous(),
            "causal_mask": trace.causal_mask.to("cpu").bool().contiguous(),
            "constant_losses": losses,
        }
        if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
            raise V5PipelineError(f"sample {record.sample_id!r} produced non-finite tensors")
        token_hash = token_ids_sha256(input_ids[0].tolist())
        metadata = {
            "schema_version": V5_TRACE_SHARD_SCHEMA,
            "collector_id": V5_COLLECTOR_ID,
            "prefix_group_id": record.prefix_group_id,
            "prefix_sha256": record.prefix_sha256,
            "token_ids_sha256": token_hash,
            "token_count": str(record.token_bucket),
            "query_sample_count": str(int(trace.query_positions.numel())),
            "key_sample_count": str(int(trace.key_positions.numel())),
            "source_config_sha256": self.source.config_sha256,
            "source_weights_sha256": self.source.weights_sha256,
            "target_config_sha256": self.target.config_sha256,
            "target_weights_sha256": self.target.weights_sha256,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            save_file(tensors, temporary, metadata=metadata)
            canonicalize_safetensors_header(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return CollectedTrace(
            path=output,
            token_ids_sha256=token_hash,
            token_count=record.token_bucket,
            query_sample_count=int(trace.query_positions.numel()),
            key_sample_count=int(trace.key_positions.numel()),
        )


def load_bound_benchmark(workspace: V5PipelineWorkspace) -> PublicationBenchmarkManifest:
    benchmark = PublicationBenchmarkManifest.load(workspace.config.benchmark_manifest_uri)
    if benchmark.content_sha256() != workspace.config.benchmark_manifest_sha256:
        raise V5PipelineError("pipeline benchmark content identity changed")
    if dict(benchmark.split_sha256) != dict(workspace.config.split_sha256):
        raise V5PipelineError("pipeline benchmark split identities changed")
    if benchmark.tokenizer_sha256 != workspace.config.tokenizer_sha256:
        raise V5PipelineError("pipeline benchmark tokenizer identity changed")
    if benchmark.chat_template_sha256 != workspace.config.chat_template_sha256:
        raise V5PipelineError("pipeline benchmark chat template identity changed")
    return benchmark


def load_completed_trace_manifest(
    workspace: V5PipelineWorkspace,
    direction: str,
    split: str,
    benchmark: PublicationBenchmarkManifest,
) -> V5TraceManifest:
    """Load and fully verify one completed non-sealed trace dependency."""

    if split not in COLLECTABLE_SPLITS:
        raise V5PipelineError(f"split {split!r} is not a completed trace dependency")
    state = workspace.state()
    record = state.stages.get(f"{direction}/collect_{split}")
    if record is None or record.status != "completed" or record.outputs is None:
        raise V5PipelineError(f"stage requires completed {split} traces")
    artifact = record.outputs.get("trace_manifest")
    if artifact is None:
        raise V5PipelineError(f"completed {split} trace stage lacks its manifest")
    path = workspace.artifact_path(artifact, verify_hash=True)
    trace = V5TraceManifest.load(path, workspace=workspace, benchmark=benchmark)
    if trace.direction != direction or trace.split != split:
        raise V5PipelineError("completed trace manifest has the wrong direction or split")
    return trace


def _write_trace_checkpoint(
    path: Path,
    *,
    input_sha256: str,
    record: TraceRecord,
    artifact: PipelineArtifact,
) -> None:
    payload = {
        "schema_version": V5_TRACE_CHECKPOINT_SCHEMA,
        "input_sha256": input_sha256,
        "record": asdict(record),
        "artifact": asdict(artifact),
    }
    _write_json_replace(path, payload)


def _load_trace_checkpoint(
    path: Path,
    *,
    workspace: V5PipelineWorkspace,
    input_sha256: str,
    record: GroupedPrefixRecord,
    verified_artifacts: dict[str, PipelineArtifact],
) -> tuple[TraceRecord, PipelineArtifact] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_TRACE_CHECKPOINT_SCHEMA:
            raise V5PipelineError("trace checkpoint schema mismatch")
        if payload.get("input_sha256") != input_sha256:
            raise V5PipelineError("trace checkpoint input binding mismatch")
        artifact = PipelineArtifact(**payload["artifact"])
        verified = verified_artifacts.get(artifact.sha256)
        if verified is None:
            workspace.artifact_path(artifact, verify_hash=True)
            verified_artifacts[artifact.sha256] = artifact
        else:
            if verified != artifact:
                raise V5PipelineError("trace checkpoints disagree on shared artifact identity")
            workspace.artifact_path(artifact, verify_hash=False)
        raw_record = dict(payload["record"])
        raw_record["shard"] = TraceObjectRef(**raw_record["shard"])
        trace = TraceRecord(**raw_record)
        errors = trace.validate(record)
        if errors:
            raise V5PipelineError("; ".join(errors))
        if trace.shard != TraceObjectRef.from_artifact(artifact):
            raise V5PipelineError("trace checkpoint artifact binding mismatch")
        return trace, artifact
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, V5PipelineError):
            raise
        raise V5PipelineError(f"trace checkpoint {path} is malformed") from exc


def _validate_cache_shape(
    value: Any,
    spec: CachedKVModelSpec,
    token_count: int,
    *,
    label: str,
) -> None:
    expected = (
        2,
        spec.num_layers,
        spec.num_key_value_heads,
        token_count,
        spec.head_dim,
    )
    if tuple(value.shape) != expected:
        raise V5PipelineError(f"{label} cache shape {tuple(value.shape)} does not match {expected}")


def _sampled_attention_output(
    query: Any,
    key: Any,
    value: Any,
    causal_mask: Any,
) -> Any:
    import torch

    query_heads = int(query.shape[1])
    key_heads = int(key.shape[1])
    if query_heads % key_heads:
        raise V5PipelineError("target query heads are not divisible by KV heads")
    repeats = query_heads // key_heads
    expanded_key = key.repeat_interleave(repeats, dim=1)
    expanded_value = value.repeat_interleave(repeats, dim=1)
    logits = torch.einsum(
        "lhqd,lhkd->lhqk",
        query.float(),
        expanded_key.float(),
    ) / math.sqrt(int(query.shape[-1]))
    mask = torch.as_tensor(causal_mask, dtype=torch.bool, device=logits.device)
    probabilities = logits.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    output = torch.einsum("lhqk,lhkd->lhqd", probabilities, expanded_value.float())
    if not bool(torch.isfinite(output).all()):
        raise V5PipelineError("sampled native attention output is non-finite")
    return output


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(dict(value), indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
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
        raise V5PipelineError("value is not finite canonical JSON") from exc


def _torch_dtype(name: str) -> Any:
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    try:
        return values[name]
    except KeyError as exc:
        raise V5PipelineError(f"unsupported trace collection dtype {name!r}") from exc


def _device_name(device: str) -> str:
    import torch

    parsed = torch.device(device)
    return torch.cuda.get_device_name(parsed) if parsed.type == "cuda" else parsed.type


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def stderr_progress(every: int = 1) -> Callable[[int, int, str], None]:
    """Return a bounded progress callback for the long-running CLI stage."""

    if every <= 0:
        raise V5PipelineError("progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"collected {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
