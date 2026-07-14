"""Synchronous multi-candidate transport fitting for the selective KV v5 pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from goldenexperience.benchmarks.publication import PublicationBenchmarkManifest
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVModelSpec,
    canonicalize_safetensors_header,
    sha256_file,
)
from goldenexperience.size_variant.head_aware_transport import (
    RIDGE_INITIALIZER_TENSORS,
    TRANSPORT_TRAINING_SEEDS,
    HeadAwareKVTransport,
    attention_distillation_terms,
    attention_preserving_loss,
    build_trainable_head_aware_transport,
    fit_head_aware_normalizers,
    fit_head_aware_ridge_initializer,
    initialize_trainable_from_ridge,
    transport_artifact_metadata,
)
from goldenexperience.size_variant.selective_manifest import (
    TRANSPORT_V1_STRUCTURE_ID,
    TRANSPORT_V2_STRUCTURE_ID,
    TransportLossContract,
    TransportSpec,
)
from goldenexperience.size_variant.v5_collect import (
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
    load_trace_shard,
    trace_shard_metadata,
)
from goldenexperience.size_variant.v5_generation import (
    FULL_PREFIX_SUPERVISION_ID,
    FullPrefixGenerationBackend,
    GenerationSupervisionSpec,
    TraceConstantGenerationBackend,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)

V5_TRANSPORT_FIT_SCHEMA = "goldenexperience.v5_transport_fit.v4"
V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA = "goldenexperience.v5_transport_fit.v3"
V5_RIDGE_TRANSPORT_FIT_SCHEMA = "goldenexperience.v5_transport_fit.v2"
V5_LEGACY_TRANSPORT_FIT_SCHEMA = "goldenexperience.v5_transport_fit.v1"
V5_TRANSPORT_CHECKPOINT_SCHEMA = "goldenexperience.v5_transport_checkpoint.v3"
V5_NORMALIZER_CHECKPOINT_SCHEMA = "goldenexperience.v5_transport_normalizers.v1"
V5_RIDGE_INITIALIZER_CHECKPOINT_SCHEMA = "goldenexperience.v5_ridge_initializer.v1"
SCREENING_DIRECTION = "qwen3_4b_to_8b"
REGISTERED_RANKS = (32, 64, 128)
DEPLOYMENT_SEED = 17
REGISTERED_RIDGE_RATIO = 1e-3
RIDGE_INITIALIZER = "train_only_row_weighted_ridge_svd"
RIDGE_SEED_STRATEGY = "deployment_identity_other_seed_orthogonal_rotation"
FULL_PREFIX_CANDIDATE_MICROBATCH = 3
FULL_PREFIX_ACTIVATION_CHECKPOINT = "non_reentrant"
FULL_PREFIX_BATCHING = "grouped_global_accumulation_v1"
LEGACY_INITIALIZER = "random_gated_silu"
LEGACY_SEED_STRATEGY = "independent_gaussian_down_projection"
_TERM_NAMES = (
    "native_generation",
    "prompt_tail_distillation",
    "attention_logit_kl",
    "attention_output_mse",
    "transformed_kv_anchor",
    "total",
)


@dataclass(frozen=True)
class TransportTrainingParameters:
    ranks: tuple[int, ...] = REGISTERED_RANKS
    seeds: tuple[int, ...] = TRANSPORT_TRAINING_SEEDS
    deployment_seed: int = DEPLOYMENT_SEED
    source_window: int = 3
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 8
    max_grad_norm: float = 1.0
    structure_id: str = TRANSPORT_V2_STRUCTURE_ID
    initializer: str = RIDGE_INITIALIZER
    ridge_ratio: float = REGISTERED_RIDGE_RATIO
    seed_strategy: str = RIDGE_SEED_STRATEGY
    loss: TransportLossContract = field(default_factory=TransportLossContract)
    generation: GenerationSupervisionSpec = field(
        default_factory=GenerationSupervisionSpec.full_prefix
    )
    full_prefix_candidate_microbatch: int = FULL_PREFIX_CANDIDATE_MICROBATCH
    full_prefix_activation_checkpoint: str = FULL_PREFIX_ACTIVATION_CHECKPOINT
    full_prefix_batching: str = FULL_PREFIX_BATCHING

    def validate(self, *, require_registered: bool = True) -> list[str]:
        try:
            errors = self.loss.validate()
        except (TypeError, ValueError):
            errors = ["transport loss contract is malformed"]
        try:
            errors.extend(self.generation.validate(require_registered=require_registered))
        except (TypeError, ValueError):
            errors.append("transport generation supervision is malformed")
        if require_registered and self.ranks != REGISTERED_RANKS:
            errors.append("transport screening must use ranks 32, 64, and 128")
        if require_registered and self.seeds != TRANSPORT_TRAINING_SEEDS:
            errors.append("transport fitting must use registered seeds 17, 29, and 43")
        ranks_valid = bool(self.ranks) and all(type(rank) is int for rank in self.ranks)
        seeds_valid = bool(self.seeds) and all(type(seed) is int for seed in self.seeds)
        if type(self.deployment_seed) is not int:
            errors.append("transport deployment seed must be an integer")
        elif not seeds_valid or self.deployment_seed not in self.seeds:
            errors.append("transport deployment seed is absent from training seeds")
        if require_registered and self.deployment_seed != DEPLOYMENT_SEED:
            errors.append("transport deployment seed must remain 17")
        if not ranks_valid or len(set(self.ranks)) != len(self.ranks):
            errors.append("transport ranks must be non-empty and unique")
        if not seeds_valid or len(set(self.seeds)) != len(self.seeds):
            errors.append("transport seeds must be non-empty and unique")
        if ranks_valid and any(rank not in REGISTERED_RANKS for rank in self.ranks):
            errors.append("transport rank is outside the registered set")
        if type(self.source_window) is not int or self.source_window not in {1, 3}:
            errors.append("transport source window must be 1 or 3")
        if require_registered and self.source_window != 3:
            errors.append("publication transport source window must remain 3")
        if type(self.epochs) is not int or self.epochs <= 0:
            errors.append("transport training epochs must be positive")
        if not _is_finite_number(self.learning_rate) or self.learning_rate <= 0:
            errors.append("transport learning rate must be finite and positive")
        if not _is_finite_number(self.weight_decay) or self.weight_decay < 0:
            errors.append("transport weight decay must be finite and non-negative")
        if type(self.gradient_accumulation) is not int or self.gradient_accumulation <= 0:
            errors.append("transport gradient accumulation must be positive")
        if not _is_finite_number(self.max_grad_norm) or self.max_grad_norm <= 0:
            errors.append("transport max gradient norm must be finite and positive")
        if self.structure_id == TRANSPORT_V2_STRUCTURE_ID:
            if self.initializer != RIDGE_INITIALIZER:
                errors.append("transport v2 must use the registered train-only ridge initializer")
            if (
                not _is_finite_number(self.ridge_ratio)
                or self.ridge_ratio != REGISTERED_RIDGE_RATIO
            ):
                errors.append("transport v2 ridge ratio must remain 1e-3")
            if self.seed_strategy != RIDGE_SEED_STRATEGY:
                errors.append("transport v2 seed strategy differs from the registered contract")
        elif self.structure_id == TRANSPORT_V1_STRUCTURE_ID:
            if self.initializer != LEGACY_INITIALIZER or self.ridge_ratio != 0:
                errors.append("legacy transport initializer contract is invalid")
            if self.seed_strategy != LEGACY_SEED_STRATEGY:
                errors.append("legacy transport seed strategy is invalid")
        else:
            errors.append("transport training structure is unsupported")
        if require_registered and self.structure_id != TRANSPORT_V2_STRUCTURE_ID:
            errors.append("publication transport screening must use transport v2")
        if require_registered and self.loss != TransportLossContract():
            errors.append("publication transport loss weights differ from the frozen contract")
        if self.generation.supervision_id == FULL_PREFIX_SUPERVISION_ID:
            if self.full_prefix_candidate_microbatch != FULL_PREFIX_CANDIDATE_MICROBATCH:
                errors.append("full-prefix candidate microbatch must remain three")
            if self.full_prefix_activation_checkpoint != FULL_PREFIX_ACTIVATION_CHECKPOINT:
                errors.append("full-prefix activation checkpoint contract changed")
            if self.full_prefix_batching != FULL_PREFIX_BATCHING:
                errors.append("full-prefix batching contract changed")
        return errors

    def to_dict(self, *, include_generation: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ranks": list(self.ranks),
            "seeds": list(self.seeds),
            "deployment_seed": self.deployment_seed,
            "source_window": self.source_window,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_accumulation": self.gradient_accumulation,
            "max_grad_norm": self.max_grad_norm,
            "loss": asdict(self.loss),
        }
        if include_generation:
            payload["generation"] = self.generation.to_dict()
        if include_generation and self.generation.supervision_id == FULL_PREFIX_SUPERVISION_ID:
            payload["full_prefix"] = {
                "candidate_microbatch": self.full_prefix_candidate_microbatch,
                "activation_checkpoint": self.full_prefix_activation_checkpoint,
                "batching": self.full_prefix_batching,
            }
        if self.structure_id == TRANSPORT_V2_STRUCTURE_ID:
            payload.update(
                {
                    "structure_id": self.structure_id,
                    "initializer": self.initializer,
                    "ridge_ratio": self.ridge_ratio,
                    "seed_strategy": self.seed_strategy,
                }
            )
        return payload

    def transport_spec(
        self,
        *,
        weights_uri: str,
        weights_sha256: str,
        rank: int,
    ) -> TransportSpec:
        values: dict[str, Any] = {
            "weights_uri": weights_uri,
            "weights_sha256": weights_sha256,
            "rank": rank,
            "source_window": self.source_window,
            "loss": self.loss,
        }
        if self.structure_id == TRANSPORT_V1_STRUCTURE_ID:
            values.update(
                {
                    "projection": "independent_per_head_low_rank_kv",
                    "residual": "gated_silu",
                    "structure_id": TRANSPORT_V1_STRUCTURE_ID,
                }
            )
        return TransportSpec(**values)


@dataclass(frozen=True)
class CandidateTrainingMetrics:
    samples: int
    optimizer_steps: int
    native_generation: float
    prompt_tail_distillation: float
    attention_logit_kl: float
    attention_output_mse: float
    transformed_kv_anchor: float
    total: float

    def validate(self, expected_samples: int, expected_optimizer_steps: int) -> list[str]:
        errors: list[str] = []
        if self.samples != expected_samples:
            errors.append("transport candidate training sample count is inconsistent")
        if self.optimizer_steps != expected_optimizer_steps:
            errors.append("transport candidate optimizer step count is inconsistent")
        for name in _TERM_NAMES:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"transport candidate metric {name} is invalid")
        return errors


@dataclass(frozen=True)
class TransportCandidateArtifact:
    candidate_id: str
    rank: int
    seed: int
    deployment_seed: bool
    weights: TraceObjectRef
    parameter_count: int
    metrics: CandidateTrainingMetrics

    def validate(self, expected_samples: int, expected_optimizer_steps: int) -> list[str]:
        errors = self.weights.validate() + self.metrics.validate(
            expected_samples,
            expected_optimizer_steps,
        )
        if not self.candidate_id:
            errors.append("transport candidate id is required")
        if self.rank not in REGISTERED_RANKS:
            errors.append("transport candidate rank is invalid")
        if self.seed not in TRANSPORT_TRAINING_SEEDS:
            errors.append("transport candidate seed is invalid")
        if self.deployment_seed != (self.seed == DEPLOYMENT_SEED):
            errors.append("transport candidate deployment-seed marker is invalid")
        if Path(self.weights.path).suffix != ".safetensors":
            errors.append("transport candidate weights must use safetensors")
        if self.parameter_count <= 0:
            errors.append("transport candidate parameter count must be positive")
        return errors


@dataclass(frozen=True)
class V5TransportFitManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    transport_train_split_sha256: str
    trace_manifest_sha256: str
    normalizer_sha256: str
    source: CachedKVModelSpec
    target: CachedKVModelSpec
    training: TransportTrainingParameters
    candidates: tuple[TransportCandidateArtifact, ...]
    training_initializer_sha256: str | None = None
    generation_sample_store_sha256: str | None = None
    schema_version: str = V5_TRANSPORT_FIT_SCHEMA

    def validate(self, *, workspace: V5PipelineWorkspace, trace: V5TraceManifest) -> list[str]:
        errors: list[str] = []
        if self.schema_version not in {
            V5_TRANSPORT_FIT_SCHEMA,
            V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA,
            V5_RIDGE_TRANSPORT_FIT_SCHEMA,
            V5_LEGACY_TRANSPORT_FIT_SCHEMA,
        }:
            errors.append("unsupported transport fit manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("transport fit manifest belongs to another pipeline")
        if self.direction != SCREENING_DIRECTION or trace.direction != self.direction:
            errors.append("transport structure screening is restricted to Qwen3 4B to 8B")
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("transport fit code hash mismatch")
        if self.transport_train_split_sha256 != workspace.config.split_sha256["transport_train"]:
            errors.append("transport fit split hash mismatch")
        if self.trace_manifest_sha256 != trace.content_sha256():
            errors.append("transport fit trace manifest hash mismatch")
        if not _is_sha256(self.normalizer_sha256):
            errors.append("transport fit normalizer hash is invalid")
        if self.schema_version in {
            V5_TRANSPORT_FIT_SCHEMA,
            V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA,
            V5_RIDGE_TRANSPORT_FIT_SCHEMA,
        }:
            if not _is_sha256(self.training_initializer_sha256):
                errors.append("transport fit training initializer hash is invalid")
        elif self.training_initializer_sha256 is not None:
            errors.append("legacy transport fit cannot claim a ridge initializer")
        if self.schema_version in {
            V5_TRANSPORT_FIT_SCHEMA,
            V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA,
        }:
            if self.generation_sample_store_sha256 != trace.raw_sample_store_sha256:
                errors.append("transport fit generation sample store hash mismatch")
        elif self.generation_sample_store_sha256 is not None:
            errors.append("pre-v3 transport fit cannot claim generation sample input")
        if self.source != trace.source or self.target != trace.target:
            errors.append("transport fit model identities differ from traces")
        training_errors = self.training.validate(
            require_registered=self.schema_version == V5_TRANSPORT_FIT_SCHEMA
        )
        if (
            self.schema_version == V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA
            and self.training.generation != GenerationSupervisionSpec()
        ):
            training_errors.append("v3 transport fit must use sampled-cache target logits")
        if (
            self.schema_version == V5_RIDGE_TRANSPORT_FIT_SCHEMA
            and self.training.generation != GenerationSupervisionSpec.legacy()
        ):
            training_errors.append("v2 transport fit must use trace-constant reporting losses")
        if (
            self.schema_version == V5_LEGACY_TRANSPORT_FIT_SCHEMA
            and self.training.structure_id != TRANSPORT_V1_STRUCTURE_ID
        ):
            training_errors.append("legacy transport fit must use transport v1")
        errors.extend(training_errors)
        expected_pairs = {
            (rank, seed) for rank in self.training.ranks for seed in self.training.seeds
        }
        observed_pairs = {(item.rank, item.seed) for item in self.candidates}
        if observed_pairs != expected_pairs or len(self.candidates) != len(expected_pairs):
            errors.append("transport fit candidate rank/seed matrix is incomplete")
        ids: set[str] = set()
        expected_samples = len(trace.records) * self.training.epochs
        expected_optimizer_steps = -1
        if not training_errors:
            expected_optimizer_steps = (
                math.ceil(len(trace.records) / self.training.gradient_accumulation)
                * self.training.epochs
            )
        for candidate in self.candidates:
            if candidate.candidate_id in ids:
                errors.append("transport fit candidate ids are not unique")
            ids.add(candidate.candidate_id)
            errors.extend(candidate.validate(expected_samples, expected_optimizer_steps))
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "pipeline_id": self.pipeline_id,
            "direction": self.direction,
            "code_sha256": self.code_sha256,
            "transport_train_split_sha256": self.transport_train_split_sha256,
            "trace_manifest_sha256": self.trace_manifest_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "source": asdict(self.source),
            "target": asdict(self.target),
            "training": self.training.to_dict(
                include_generation=self.schema_version
                in {V5_TRANSPORT_FIT_SCHEMA, V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA}
            ),
            "candidates": [asdict(item) for item in self.candidates],
        }
        if self.schema_version in {
            V5_TRANSPORT_FIT_SCHEMA,
            V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA,
            V5_RIDGE_TRANSPORT_FIT_SCHEMA,
        }:
            payload["training_initializer_sha256"] = self.training_initializer_sha256
        if self.schema_version in {
            V5_TRANSPORT_FIT_SCHEMA,
            V5_TARGET_LOGIT_TRANSPORT_FIT_SCHEMA,
        }:
            payload["generation_sample_store_sha256"] = self.generation_sample_store_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5TransportFitManifest:
        training_payload = dict(payload["training"])
        training_payload["ranks"] = tuple(training_payload["ranks"])
        training_payload["seeds"] = tuple(training_payload["seeds"])
        training_payload["loss"] = TransportLossContract(**training_payload["loss"])
        raw_generation = training_payload.get("generation")
        training_payload["generation"] = (
            GenerationSupervisionSpec(**raw_generation)
            if isinstance(raw_generation, Mapping)
            else GenerationSupervisionSpec.legacy()
        )
        raw_full_prefix = training_payload.pop("full_prefix", None)
        if isinstance(raw_full_prefix, Mapping):
            training_payload.update(
                {
                    "full_prefix_candidate_microbatch": raw_full_prefix.get(
                        "candidate_microbatch"
                    ),
                    "full_prefix_activation_checkpoint": raw_full_prefix.get(
                        "activation_checkpoint"
                    ),
                    "full_prefix_batching": raw_full_prefix.get("batching"),
                }
            )
        if "structure_id" not in training_payload:
            training_payload.update(
                {
                    "structure_id": TRANSPORT_V1_STRUCTURE_ID,
                    "initializer": LEGACY_INITIALIZER,
                    "ridge_ratio": 0.0,
                    "seed_strategy": LEGACY_SEED_STRATEGY,
                }
            )
        candidates = []
        for item in payload.get("candidates", ()):
            candidate = dict(item)
            candidate["weights"] = TraceObjectRef(**candidate["weights"])
            candidate["metrics"] = CandidateTrainingMetrics(**candidate["metrics"])
            candidates.append(TransportCandidateArtifact(**candidate))
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            transport_train_split_sha256=str(payload["transport_train_split_sha256"]),
            trace_manifest_sha256=str(payload["trace_manifest_sha256"]),
            normalizer_sha256=str(payload["normalizer_sha256"]),
            source=CachedKVModelSpec(**payload["source"]),
            target=CachedKVModelSpec(**payload["target"]),
            training=TransportTrainingParameters(**training_payload),
            candidates=tuple(candidates),
            training_initializer_sha256=payload.get("training_initializer_sha256"),
            generation_sample_store_sha256=payload.get("generation_sample_store_sha256"),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
    ) -> V5TransportFitManifest:
        """Load a fit manifest and verify every referenced candidate object."""

        try:
            manifest = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = manifest.validate(workspace=workspace, trace=trace)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("transport fit manifest is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        for candidate in manifest.candidates:
            _verify_workspace_object(workspace, candidate.weights)
        return manifest


class TransportFitManifestLike(Protocol):
    @property
    def direction(self) -> str: ...

    @property
    def source(self) -> CachedKVModelSpec: ...

    @property
    def target(self) -> CachedKVModelSpec: ...

    @property
    def training(self) -> TransportTrainingParameters: ...

    @property
    def candidates(self) -> tuple[TransportCandidateArtifact, ...]: ...


def load_completed_transport_fit(
    workspace: V5PipelineWorkspace,
    direction: str,
    benchmark: PublicationBenchmarkManifest,
) -> tuple[V5TransportFitManifest, V5TraceManifest]:
    """Load the completed fit manifest together with its verified train traces."""

    trace = load_completed_trace_manifest(workspace, direction, "transport_train", benchmark)
    state = workspace.state()
    record = state.stages.get(f"{direction}/fit_transport")
    if record is None or record.status != "completed" or record.outputs is None:
        raise V5PipelineError("stage requires completed transport fitting")
    artifact = record.outputs.get("transport_fit_manifest")
    if artifact is None:
        raise V5PipelineError("completed transport fit lacks its manifest")
    path = workspace.artifact_path(artifact, verify_hash=True)
    return V5TransportFitManifest.load(path, workspace=workspace, trace=trace), trace


def load_fitted_transport(
    workspace: V5PipelineWorkspace,
    manifest: TransportFitManifestLike,
    candidate: TransportCandidateArtifact,
    *,
    device: str,
) -> tuple[HeadAwareKVTransport, TransportSpec, Path]:
    """Load one fitted candidate under the final runtime tensor/metadata contract."""

    from safetensors import safe_open

    if candidate not in manifest.candidates:
        raise V5PipelineError("transport candidate is not part of the fit manifest")
    path = _verify_workspace_object(workspace, candidate.weights)
    spec = manifest.training.transport_spec(
        weights_uri=path.name,
        weights_sha256=candidate.weights.sha256,
        rank=candidate.rank,
    )
    expected_metadata = transport_artifact_metadata(
        direction=manifest.direction,
        source=manifest.source,
        target=manifest.target,
        spec=spec,
    )
    tensors: dict[str, Any] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if (handle.metadata() or {}) != expected_metadata:
                raise V5PipelineError("fitted transport metadata is invalid")
            tensors = {
                name: handle.get_tensor(name)
                for name in handle.keys()  # noqa: SIM118
            }
    except V5PipelineError:
        raise
    except Exception as exc:
        raise V5PipelineError("fitted transport safetensors payload is invalid") from exc
    _verify_workspace_object(workspace, candidate.weights)
    try:
        runtime = HeadAwareKVTransport(
            manifest.source,
            manifest.target,
            spec,
            tensors,
            device=device,
        )
    except Exception as exc:
        raise V5PipelineError("fitted transport tensor contract is invalid") from exc
    if runtime.parameter_count() != candidate.parameter_count:
        raise V5PipelineError("fitted transport parameter count changed")
    return runtime, spec, path


def verify_transport_candidate_object(
    workspace: V5PipelineWorkspace,
    candidate: TransportCandidateArtifact,
) -> Path:
    """Resolve and checksum a fitted candidate's immutable workspace object."""

    return _verify_workspace_object(workspace, candidate.weights)


@dataclass
class _CandidateContext:
    candidate_id: str
    rank: int
    seed: int
    spec: TransportSpec
    module: Any
    optimizer: Any
    metric_sums: dict[str, float]


@dataclass(frozen=True)
class _FittedCandidate:
    candidate_id: str
    rank: int
    seed: int
    path: Path
    parameter_count: int
    metrics: CandidateTrainingMetrics


class _TraceLoader:
    def __init__(self, workspace: V5PipelineWorkspace, trace: V5TraceManifest) -> None:
        self.workspace = workspace
        self.trace = trace
        self.verified: set[str] = set()
        self.cache: dict[
            str,
            tuple[TraceObjectRef, dict[str, str], dict[str, Any]],
        ] = {}

    def load(self, record: TraceRecord) -> dict[str, Any]:
        metadata = trace_shard_metadata(record, self.trace.source, self.trace.target)
        cached = self.cache.get(record.shard.sha256)
        if cached is not None:
            cached_ref, cached_metadata, tensors = cached
            if cached_ref != record.shard or cached_metadata != metadata:
                raise V5PipelineError("shared trace shard has inconsistent identity bindings")
            return tensors
        relative = Path(record.shard.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise V5PipelineError("trace object path escapes the pipeline workspace")
        path = (self.workspace.root / relative).resolve()
        if not path.is_relative_to(self.workspace.root):
            raise V5PipelineError("trace object symlink escapes the pipeline workspace")
        tensors = load_trace_shard(
            path,
            record,
            source=self.trace.source,
            target=self.trace.target,
            verify_hash=record.shard.sha256 not in self.verified,
        )
        self.verified.add(record.shard.sha256)
        self.cache[record.shard.sha256] = (record.shard, metadata, tensors)
        return tensors


class GenerationBackend(Protocol):
    supervision_id: str

    def __enter__(self) -> GenerationBackend: ...

    def __exit__(self, *_args: object) -> None: ...

    def parameters(self) -> Mapping[str, Any]: ...

    def losses(
        self,
        record: TraceRecord,
        tensors: Mapping[str, Any],
        transformed_kv_batch: Any,
    ) -> tuple[Any, Any]: ...


def _row_weighted_unique_records(trace: V5TraceManifest) -> tuple[tuple[TraceRecord, int], ...]:
    grouped: dict[str, tuple[TraceRecord, int]] = {}
    for record in trace.records:
        previous = grouped.get(record.shard.sha256)
        if previous is None:
            grouped[record.shard.sha256] = (record, 1)
            continue
        representative, count = previous
        if representative.shard != record.shard or trace_shard_metadata(
            representative,
            trace.source,
            trace.target,
        ) != trace_shard_metadata(record, trace.source, trace.target):
            raise V5PipelineError("shared trace shard has inconsistent row-weight bindings")
        grouped[record.shard.sha256] = (representative, count + 1)
    return tuple(grouped.values())


def _full_prefix_epoch_order(trace: V5TraceManifest, epoch: int) -> tuple[int, ...]:
    if type(epoch) is not int or epoch < 0:
        raise V5PipelineError("full-prefix epoch must be non-negative")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(trace.records):
        groups.setdefault(record.prefix_group_id, []).append(index)
    group_ids = sorted(groups)
    random.Random(10_000 * epoch + DEPLOYMENT_SEED).shuffle(group_ids)
    order: list[int] = []
    for group_id in group_ids:
        indices = sorted(groups[group_id], key=lambda value: trace.records[value].sample_id)
        seed = int(
            _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "epoch": epoch,
                        "prefix_group_id": group_id,
                        "strategy": FULL_PREFIX_BATCHING,
                    }
                )
            )[:16],
            16,
        )
        random.Random(seed).shuffle(indices)
        order.extend(indices)
    if len(order) != len(trace.records) or set(order) != set(range(len(trace.records))):
        raise V5PipelineError("full-prefix epoch order is not a record permutation")
    return tuple(order)


def _full_prefix_order_sha256(trace: V5TraceManifest, epoch: int) -> str:
    order = _full_prefix_epoch_order(trace, epoch)
    return _sha256_bytes(
        _canonical_json_bytes([trace.records[index].sample_id for index in order])
    )


def _prefix_segments(
    trace: V5TraceManifest,
    indices: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    segments: list[list[int]] = []
    for index in indices:
        if not segments or (
            trace.records[segments[-1][-1]].prefix_group_id
            != trace.records[index].prefix_group_id
        ):
            segments.append([])
        segments[-1].append(index)
    return tuple(tuple(segment) for segment in segments)


class SynchronousTransportTrainer:
    """Train every candidate with one verified tensor load per unique trace shard."""

    def __init__(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        parameters: TransportTrainingParameters,
        device: str,
        generation_backend: GenerationBackend | None = None,
        checkpoint_every_steps: int = 256,
        progress: Callable[[int, int, int, str], None] | None = None,
    ) -> None:
        if checkpoint_every_steps <= 0:
            raise V5PipelineError("transport checkpoint interval must be positive")
        if not trace.records:
            raise V5PipelineError("transport fitting requires at least one trace")
        if trace.split != "transport_train":
            raise V5PipelineError("transport fitting may only read transport-train traces")
        errors = parameters.validate(require_registered=False)
        if errors:
            raise V5PipelineError("; ".join(errors))
        if parameters.structure_id != TRANSPORT_V2_STRUCTURE_ID:
            raise V5PipelineError("new transport fitting is restricted to transport v2")
        if generation_backend is None:
            if parameters.generation != GenerationSupervisionSpec.legacy():
                raise V5PipelineError("target-logit generation backend is required")
            generation_backend = TraceConstantGenerationBackend()
        if generation_backend.supervision_id != parameters.generation.supervision_id:
            raise V5PipelineError("generation backend differs from the training contract")
        if (
            parameters.generation.supervision_id == FULL_PREFIX_SUPERVISION_ID
            and not isinstance(generation_backend, FullPrefixGenerationBackend)
        ):
            raise V5PipelineError("full-prefix training requires its registered backend")
        if isinstance(generation_backend, FullPrefixGenerationBackend):
            import torch

            if torch.device(generation_backend.target_device) != torch.device(device):
                raise V5PipelineError("full-prefix backend and trainer target devices differ")
        self.workspace = workspace
        self.trace = trace
        self.parameters = parameters
        self.device = device
        self.checkpoint_every_steps = checkpoint_every_steps
        self.progress = progress
        self.loader = _TraceLoader(workspace, trace)
        self.generation_backend = generation_backend

    def fit(
        self,
        work: Path,
        *,
        stage_input_sha256: str,
    ) -> tuple[list[_FittedCandidate], str, str]:
        work.mkdir(parents=True, exist_ok=True)
        normalizers, normalizer_sha256 = self._load_or_fit_normalizers(
            work,
            stage_input_sha256=stage_input_sha256,
        )
        initializer, initializer_sha256 = self._load_or_fit_ridge_initializer(
            work,
            normalizers=normalizers,
            normalizer_sha256=normalizer_sha256,
            stage_input_sha256=stage_input_sha256,
        )
        contexts = self._build_contexts(normalizers, initializer)
        checkpoint_payload: dict[str, Any] = {
            "stage_input_sha256": stage_input_sha256,
            "trace_manifest_sha256": self.trace.content_sha256(),
            "training": self.parameters.to_dict(),
            "normalizer_sha256": normalizer_sha256,
            "training_initializer_sha256": initializer_sha256,
        }
        if self.parameters.generation.supervision_id == FULL_PREFIX_SUPERVISION_ID:
            checkpoint_payload["epoch_order_sha256"] = [
                _full_prefix_order_sha256(self.trace, epoch)
                for epoch in range(self.parameters.epochs)
            ]
        checkpoint_binding = _sha256_bytes(_canonical_json_bytes(checkpoint_payload))
        progress = _load_checkpoint_set(
            work / "checkpoint_set.json",
            contexts,
            binding_sha256=checkpoint_binding,
            record_count=len(self.trace.records),
            epochs=self.parameters.epochs,
            gradient_accumulation=self.parameters.gradient_accumulation,
        )
        if progress is None:
            progress = {
                "epoch": 0,
                "position": 0,
                "optimizer_steps": 0,
                "samples_seen": 0,
            }
        for context in contexts:
            context.optimizer.zero_grad(set_to_none=True)
            context.module.train()
        total_samples = len(self.trace.records) * self.parameters.epochs
        with self.generation_backend:
            if isinstance(self.generation_backend, FullPrefixGenerationBackend):
                self._train_full_prefix_contexts(
                    contexts,
                    backend=self.generation_backend,
                    progress=progress,
                    total_samples=total_samples,
                    work=work,
                    checkpoint_binding=checkpoint_binding,
                )
            else:
                self._train_contexts(
                    contexts,
                    progress=progress,
                    total_samples=total_samples,
                    work=work,
                    checkpoint_binding=checkpoint_binding,
                )
        if int(progress["samples_seen"]) != total_samples:
            raise V5PipelineError("transport training checkpoint sample count is inconsistent")
        _validate_progress(
            progress,
            record_count=len(self.trace.records),
            epochs=self.parameters.epochs,
            gradient_accumulation=self.parameters.gradient_accumulation,
        )
        fitted: list[_FittedCandidate] = []
        weights_dir = work / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        for context in contexts:
            runtime_state = context.module.runtime_state()
            runtime = HeadAwareKVTransport(
                self.trace.source,
                self.trace.target,
                context.spec,
                runtime_state,
                device="cpu",
            )
            path = weights_dir / f"{context.candidate_id}.safetensors"
            _save_runtime_weights(
                path,
                runtime_state,
                metadata=transport_artifact_metadata(
                    direction=self.trace.direction,
                    source=self.trace.source,
                    target=self.trace.target,
                    spec=context.spec,
                ),
            )
            samples = int(progress["samples_seen"])
            metrics = CandidateTrainingMetrics(
                samples=samples,
                optimizer_steps=int(progress["optimizer_steps"]),
                **{name: context.metric_sums[name] / samples for name in _TERM_NAMES},
            )
            expected_steps = (
                math.ceil(len(self.trace.records) / self.parameters.gradient_accumulation)
                * self.parameters.epochs
            )
            metric_errors = metrics.validate(total_samples, expected_steps)
            if metric_errors:
                raise V5PipelineError("; ".join(metric_errors))
            fitted.append(
                _FittedCandidate(
                    candidate_id=context.candidate_id,
                    rank=context.rank,
                    seed=context.seed,
                    path=path,
                    parameter_count=runtime.parameter_count(),
                    metrics=metrics,
                )
            )
        return fitted, normalizer_sha256, initializer_sha256

    def _train_contexts(
        self,
        contexts: Sequence[_CandidateContext],
        *,
        progress: dict[str, Any],
        total_samples: int,
        work: Path,
        checkpoint_binding: str,
    ) -> None:
        import torch
        import torch.nn.functional as functional

        for epoch in range(int(progress["epoch"]), self.parameters.epochs):
            order = list(range(len(self.trace.records)))
            random.Random(10_000 * epoch + DEPLOYMENT_SEED).shuffle(order)
            start_position = int(progress["position"]) if epoch == int(progress["epoch"]) else 0
            for group_start in range(
                start_position,
                len(order),
                self.parameters.gradient_accumulation,
            ):
                group = order[group_start : group_start + self.parameters.gradient_accumulation]
                for index in group:
                    record = self.trace.records[index]
                    tensors = self.loader.load(record)
                    source_kv = tensors["source_kv"].to(self.device)
                    target_kv = tensors["target_kv"].to(self.device)
                    positions = tensors["key_positions"].to(self.device)
                    query = tensors["target_query"].to(self.device)
                    mask = tensors["causal_mask"].to(self.device)
                    native_output = tensors["native_attention_output"].to(self.device)
                    transformed_values = []
                    attention_values = []
                    for context in contexts:
                        transformed = context.module(source_kv, positions)
                        logit_kl, output_mse = attention_distillation_terms(
                            query,
                            target_kv[0],
                            target_kv[1],
                            transformed[0],
                            transformed[1],
                            attention_mask=mask,
                            native_attention_output=native_output,
                        )
                        anchor = functional.mse_loss(transformed.float(), target_kv.float())
                        transformed_values.append(transformed)
                        attention_values.append((logit_kl, output_mse, anchor))
                    transformed_batch = torch.stack(transformed_values)
                    generation, distillation = self.generation_backend.losses(
                        record,
                        tensors,
                        transformed_batch,
                    )
                    if (
                        generation.shape != (len(contexts),)
                        or distillation.shape != (len(contexts),)
                        or not bool(torch.isfinite(generation).all())
                        or not bool(torch.isfinite(distillation).all())
                    ):
                        raise V5PipelineError("generation backend returned invalid losses")
                    terms_by_candidate = []
                    for candidate_index, context in enumerate(contexts):
                        logit_kl, output_mse, anchor = attention_values[candidate_index]
                        terms = attention_preserving_loss(
                            native_generation_loss=generation[candidate_index],
                            prompt_tail_distillation_loss=distillation[candidate_index],
                            attention_logit_kl=logit_kl,
                            attention_output_mse=output_mse,
                            transformed_kv_anchor_loss=anchor,
                            contract=self.parameters.loss,
                        )
                        if any(
                            not bool(torch.isfinite(getattr(terms, name)).all())
                            for name in _TERM_NAMES
                        ):
                            raise V5PipelineError(
                                f"transport candidate {context.candidate_id} produced "
                                "a non-finite loss"
                            )
                        terms_by_candidate.append(terms)
                    (
                        torch.stack([item.total for item in terms_by_candidate]).sum()
                        / len(group)
                    ).backward()
                    for context, terms in zip(contexts, terms_by_candidate, strict=True):
                        for name in _TERM_NAMES:
                            context.metric_sums[name] += float(
                                getattr(terms, name).detach().float().item()
                            )
                    progress["samples_seen"] = int(progress["samples_seen"]) + 1
                    if self.progress is not None:
                        self.progress(
                            int(progress["samples_seen"]),
                            total_samples,
                            epoch,
                            record.sample_id,
                        )
                for context in contexts:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        context.module.parameters(),
                        self.parameters.max_grad_norm,
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise V5PipelineError(
                            f"transport candidate {context.candidate_id} produced "
                            "non-finite gradients"
                        )
                    context.optimizer.step()
                    context.optimizer.zero_grad(set_to_none=True)
                progress["optimizer_steps"] = int(progress["optimizer_steps"]) + 1
                progress["position"] = group_start + len(group)
                if int(progress["position"]) == len(order):
                    progress["epoch"] = epoch + 1
                    progress["position"] = 0
                if (
                    int(progress["optimizer_steps"]) % self.checkpoint_every_steps == 0
                    or int(progress["position"]) == 0
                ):
                    _save_checkpoint_set(
                        work,
                        contexts,
                        progress=progress,
                        binding_sha256=checkpoint_binding,
                    )

    def _train_full_prefix_contexts(
        self,
        contexts: Sequence[_CandidateContext],
        *,
        backend: FullPrefixGenerationBackend,
        progress: dict[str, Any],
        total_samples: int,
        work: Path,
        checkpoint_binding: str,
    ) -> None:
        import torch

        for epoch in range(int(progress["epoch"]), self.parameters.epochs):
            order = _full_prefix_epoch_order(self.trace, epoch)
            start_position = int(progress["position"]) if epoch == int(progress["epoch"]) else 0
            backend.clear_prefix_asset()
            for group_start in range(
                start_position,
                len(order),
                self.parameters.gradient_accumulation,
            ):
                group = order[
                    group_start : group_start + self.parameters.gradient_accumulation
                ]
                for segment in _prefix_segments(self.trace, group):
                    records = tuple(self.trace.records[index] for index in segment)
                    self._train_full_prefix_segment(
                        contexts,
                        backend=backend,
                        records=records,
                        denominator=len(group),
                    )
                    for record in records:
                        progress["samples_seen"] = int(progress["samples_seen"]) + 1
                        if self.progress is not None:
                            self.progress(
                                int(progress["samples_seen"]),
                                total_samples,
                                epoch,
                                record.sample_id,
                            )
                for context in contexts:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        context.module.parameters(),
                        self.parameters.max_grad_norm,
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise V5PipelineError(
                            f"transport candidate {context.candidate_id} produced "
                            "non-finite gradients"
                        )
                    context.optimizer.step()
                    context.optimizer.zero_grad(set_to_none=True)
                progress["optimizer_steps"] = int(progress["optimizer_steps"]) + 1
                progress["position"] = group_start + len(group)
                if int(progress["position"]) == len(order):
                    progress["epoch"] = epoch + 1
                    progress["position"] = 0
                    backend.clear_prefix_asset()
                if (
                    int(progress["optimizer_steps"]) % self.checkpoint_every_steps == 0
                    or int(progress["position"]) == 0
                ):
                    _save_checkpoint_set(
                        work,
                        contexts,
                        progress=progress,
                        binding_sha256=checkpoint_binding,
                    )

    def _train_full_prefix_segment(
        self,
        contexts: Sequence[_CandidateContext],
        *,
        backend: FullPrefixGenerationBackend,
        records: Sequence[TraceRecord],
        denominator: int,
    ) -> None:
        import torch
        import torch.nn.functional as functional
        from torch.utils.checkpoint import checkpoint

        if not records or denominator <= 0 or len(records) > denominator:
            raise V5PipelineError("full-prefix segment has an invalid accumulation weight")
        representative = records[0]
        if any(
            record.prefix_group_id != representative.prefix_group_id
            or record.shard != representative.shard
            for record in records
        ):
            raise V5PipelineError("full-prefix segment spans multiple trace objects")
        tensors = self.loader.load(representative)
        asset = backend.prefix_asset(representative)
        teachers = tuple(backend.teacher(record, asset) for record in records)
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        full_positions = torch.arange(asset.token_count, device=self.device)
        key_positions = tensors["key_positions"].to(self.device)
        target_kv = tensors["target_kv"].to(self.device)
        query = tensors["target_query"].to(self.device)
        mask = tensors["causal_mask"].to(self.device)
        native_output = tensors["native_attention_output"].to(self.device)
        model_dtype = next(backend.model.parameters()).dtype
        contract = self.parameters.loss
        microbatch = self.parameters.full_prefix_candidate_microbatch
        segment_weight = len(records) / denominator
        for start in range(0, len(contexts), microbatch):
            batch_contexts = contexts[start : start + microbatch]
            transformed_values = []
            for context in batch_contexts:
                transformed_values.append(
                    checkpoint(
                        lambda source, positions, module=context.module: module(
                            source,
                            positions,
                            compute_dtype=model_dtype,
                        ),
                        asset.source_kv,
                        full_positions,
                        use_reentrant=False,
                    )
                )
            transformed = torch.stack(transformed_values)
            del transformed_values
            proxy = transformed.detach().requires_grad_(True)
            generation_sums = [0.0] * len(batch_contexts)
            distillation_sums = [0.0] * len(batch_contexts)
            for teacher in teachers:
                generation, distillation = backend.student_losses(proxy, teacher)
                if (
                    generation.shape != (len(batch_contexts),)
                    or distillation.shape != (len(batch_contexts),)
                    or not bool(torch.isfinite(generation).all())
                    or not bool(torch.isfinite(distillation).all())
                ):
                    raise V5PipelineError("full-prefix generation losses are invalid")
                for index in range(len(batch_contexts)):
                    generation_sums[index] += float(generation[index].detach().float().item())
                    distillation_sums[index] += float(
                        distillation[index].detach().float().item()
                    )
                row_loss = (
                    generation + contract.prompt_tail_distillation * distillation
                ).sum() / denominator
                row_loss.backward()
            if proxy.grad is None or not bool(torch.isfinite(proxy.grad).all()):
                raise V5PipelineError("full-prefix cache proxy gradient is invalid")
            sampled = transformed.index_select(4, key_positions)
            alignment_losses = []
            alignment_values = []
            for index in range(len(batch_contexts)):
                logit_kl, output_mse = attention_distillation_terms(
                    query,
                    target_kv[0],
                    target_kv[1],
                    sampled[index, 0],
                    sampled[index, 1],
                    attention_mask=mask,
                    native_attention_output=native_output,
                )
                anchor = functional.mse_loss(sampled[index].float(), target_kv.float())
                alignment = (
                    contract.attention_logit_kl * logit_kl
                    + contract.attention_output_mse * output_mse
                    + contract.transformed_kv_anchor * anchor
                )
                alignment_losses.append(alignment)
                alignment_values.append(
                    (
                        float(logit_kl.detach().float().item()),
                        float(output_mse.detach().float().item()),
                        float(anchor.detach().float().item()),
                    )
                )
            alignment_loss = torch.stack(alignment_losses).sum() * segment_weight
            torch.autograd.backward(
                (transformed, alignment_loss),
                (proxy.grad, torch.ones_like(alignment_loss)),
            )
            for index, context in enumerate(batch_contexts):
                logit_kl_value, output_mse_value, anchor_value = alignment_values[index]
                context.metric_sums["native_generation"] += generation_sums[index]
                context.metric_sums["prompt_tail_distillation"] += distillation_sums[index]
                context.metric_sums["attention_logit_kl"] += logit_kl_value * len(records)
                context.metric_sums["attention_output_mse"] += output_mse_value * len(records)
                context.metric_sums["transformed_kv_anchor"] += anchor_value * len(records)
                context.metric_sums["total"] += (
                    generation_sums[index]
                    + contract.prompt_tail_distillation * distillation_sums[index]
                    + len(records)
                    * (
                        contract.attention_logit_kl * logit_kl_value
                        + contract.attention_output_mse * output_mse_value
                        + contract.transformed_kv_anchor * anchor_value
                    )
                )
            del transformed, proxy, sampled, alignment_loss
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()

    def _build_contexts(
        self,
        normalizers: Mapping[str, Any],
        initializer: Mapping[str, Any],
    ) -> list[_CandidateContext]:
        import torch

        contexts = []
        for rank in self.parameters.ranks:
            for seed in self.parameters.seeds:
                spec = self.parameters.transport_spec(
                    weights_uri="candidate.safetensors",
                    weights_sha256="0" * 64,
                    rank=rank,
                )
                module = build_trainable_head_aware_transport(
                    self.trace.source,
                    self.trace.target,
                    spec,
                    device=self.device,
                    seed=seed,
                )
                with torch.no_grad():
                    for name, value in normalizers.items():
                        getattr(module, name).copy_(value.to(self.device))
                initialize_trainable_from_ridge(module, initializer, seed=seed)
                optimizer = torch.optim.AdamW(
                    module.parameters(),
                    lr=self.parameters.learning_rate,
                    weight_decay=self.parameters.weight_decay,
                )
                contexts.append(
                    _CandidateContext(
                        candidate_id=_candidate_id(
                            self.trace,
                            self.parameters,
                            rank=rank,
                            seed=seed,
                        ),
                        rank=rank,
                        seed=seed,
                        spec=spec,
                        module=module,
                        optimizer=optimizer,
                        metric_sums={name: 0.0 for name in _TERM_NAMES},
                    )
                )
        return contexts

    def _load_or_fit_normalizers(
        self,
        work: Path,
        *,
        stage_input_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        from safetensors import safe_open
        from safetensors.torch import save_file

        pointer_path = work / "normalizers.json"
        binding = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "stage_input_sha256": stage_input_sha256,
                    "trace_manifest_sha256": self.trace.content_sha256(),
                    "source_window": self.parameters.source_window,
                }
            )
        )
        if pointer_path.is_file():
            if pointer_path.is_symlink():
                raise V5PipelineError("normalizer checkpoint pointer cannot be a symbolic link")
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if pointer.get("binding_sha256") != binding:
                raise V5PipelineError("normalizer checkpoint input binding mismatch")
            digest = pointer.get("sha256")
            if not _is_sha256(digest):
                raise V5PipelineError("normalizer checkpoint checksum is invalid")
            expected_relative = Path("normalizer_objects") / f"{digest}.safetensors"
            if str(pointer.get("path")) != expected_relative.as_posix():
                raise V5PipelineError("normalizer checkpoint path is invalid")
            path = _resolve_read_only_work_file(work, expected_relative)
            before = _file_signature(path)
            if sha256_file(path) != digest:
                raise V5PipelineError("normalizer checkpoint checksum mismatch")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if (handle.metadata() or {}) != {
                    "schema_version": V5_NORMALIZER_CHECKPOINT_SCHEMA,
                    "binding_sha256": binding,
                }:
                    raise V5PipelineError("normalizer checkpoint metadata is invalid")
                values = {
                    name: handle.get_tensor(name)
                    for name in handle.keys()  # noqa: SIM118
                }
            if _file_signature(path) != before or sha256_file(path) != digest:
                raise V5PipelineError("normalizer checkpoint changed while loading")
            _validate_normalizers(values, self.trace.target)
            return values, digest
        spec = self.parameters.transport_spec(
            weights_uri="normalizer.safetensors",
            weights_sha256="0" * 64,
            rank=min(self.parameters.ranks),
        )
        module = build_trainable_head_aware_transport(
            self.trace.source,
            self.trace.target,
            spec,
            device=self.device,
            seed=min(self.parameters.seeds),
        )

        def batches() -> Iterator[tuple[Any, Any, int]]:
            for record, weight in _row_weighted_unique_records(self.trace):
                tensors = self.loader.load(record)
                yield (
                    tensors["source_kv"].to(self.device),
                    tensors["key_positions"].to(self.device),
                    weight,
                )

        fit_head_aware_normalizers(module, batches())
        values = {
            name: getattr(module, name).detach().cpu()
            for name in (
                "key_normalizer_mean",
                "key_normalizer_scale",
                "value_normalizer_mean",
                "value_normalizer_scale",
            )
        }
        _validate_normalizers(values, self.trace.target)
        directory = work / "normalizer_objects"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        save_file(
            values,
            temporary,
            metadata={
                "schema_version": V5_NORMALIZER_CHECKPOINT_SCHEMA,
                "binding_sha256": binding,
            },
        )
        canonicalize_safetensors_header(temporary)
        digest = sha256_file(temporary)
        path = directory / f"{digest}.safetensors"
        temporary.replace(path)
        os.chmod(path, 0o444)
        _write_json_replace(
            pointer_path,
            {
                "binding_sha256": binding,
                "path": path.relative_to(work).as_posix(),
                "sha256": digest,
            },
        )
        return values, digest

    def _load_or_fit_ridge_initializer(
        self,
        work: Path,
        *,
        normalizers: Mapping[str, Any],
        normalizer_sha256: str,
        stage_input_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        from safetensors import safe_open
        from safetensors.torch import save_file

        pointer_path = work / "ridge_initializer.json"
        binding = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "stage_input_sha256": stage_input_sha256,
                    "trace_manifest_sha256": self.trace.content_sha256(),
                    "normalizer_sha256": normalizer_sha256,
                    "structure_id": self.parameters.structure_id,
                    "initializer": self.parameters.initializer,
                    "ridge_ratio": self.parameters.ridge_ratio,
                    "source_window": self.parameters.source_window,
                }
            )
        )
        metadata = {
            "schema_version": V5_RIDGE_INITIALIZER_CHECKPOINT_SCHEMA,
            "binding_sha256": binding,
            "trace_manifest_sha256": self.trace.content_sha256(),
            "normalizer_sha256": normalizer_sha256,
            "ridge_ratio": str(self.parameters.ridge_ratio),
            "row_weighting": "frozen_trace_rows",
        }
        if pointer_path.is_file():
            if pointer_path.is_symlink():
                raise V5PipelineError("ridge initializer pointer cannot be a symbolic link")
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise V5PipelineError("ridge initializer pointer is malformed") from exc
            if pointer.get("binding_sha256") != binding:
                raise V5PipelineError("ridge initializer input binding mismatch")
            digest = pointer.get("sha256")
            if not _is_sha256(digest):
                raise V5PipelineError("ridge initializer checksum is invalid")
            expected_relative = Path("ridge_initializer_objects") / f"{digest}.safetensors"
            if str(pointer.get("path")) != expected_relative.as_posix():
                raise V5PipelineError("ridge initializer path is invalid")
            path = _resolve_read_only_work_file(work, expected_relative)
            before = _file_signature(path)
            if sha256_file(path) != digest:
                raise V5PipelineError("ridge initializer checksum mismatch")
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if (handle.metadata() or {}) != metadata:
                        raise V5PipelineError("ridge initializer metadata is invalid")
                    values = {
                        name: handle.get_tensor(name)
                        for name in handle.keys()  # noqa: SIM118
                    }
            except V5PipelineError:
                raise
            except Exception as exc:
                raise V5PipelineError("ridge initializer payload is invalid") from exc
            if _file_signature(path) != before or sha256_file(path) != digest:
                raise V5PipelineError("ridge initializer changed while loading")
            _validate_ridge_initializer_checkpoint(values, self.trace.target)
            return values, digest
        spec = self.parameters.transport_spec(
            weights_uri="ridge-initializer.safetensors",
            weights_sha256="0" * 64,
            rank=min(self.parameters.ranks),
        )
        module = build_trainable_head_aware_transport(
            self.trace.source,
            self.trace.target,
            spec,
            device=self.device,
            seed=self.parameters.deployment_seed,
        )
        import torch

        with torch.no_grad():
            for name, value in normalizers.items():
                getattr(module, name).copy_(value.to(self.device))

        def batches() -> Iterator[tuple[Any, Any, Any, int]]:
            for record, weight in _row_weighted_unique_records(self.trace):
                tensors = self.loader.load(record)
                yield (
                    tensors["source_kv"].to(self.device),
                    tensors["target_kv"].to(self.device),
                    tensors["key_positions"].to(self.device),
                    weight,
                )

        try:
            values = fit_head_aware_ridge_initializer(
                module,
                batches(),
                ridge_ratio=self.parameters.ridge_ratio,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise V5PipelineError("transport ridge initialization failed") from exc
        _validate_ridge_initializer_checkpoint(values, self.trace.target)
        directory = work / "ridge_initializer_objects"
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".{uuid.uuid4().hex}.tmp"
        save_file(values, temporary, metadata=metadata)
        canonicalize_safetensors_header(temporary)
        digest = sha256_file(temporary)
        path = directory / f"{digest}.safetensors"
        temporary.replace(path)
        os.chmod(path, 0o444)
        _write_json_replace(
            pointer_path,
            {
                "binding_sha256": binding,
                "path": path.relative_to(work).as_posix(),
                "sha256": digest,
            },
        )
        return values, digest


def run_fit_transport_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    identity_cache_path: str | Path | None,
    source_device: str = "cuda:0",
    device: str = "cuda:1",
    resume: bool = False,
    checkpoint_every_steps: int = 256,
    progress: Callable[[int, int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Fit all structure-screening candidates from transport-train traces."""

    if direction != SCREENING_DIRECTION:
        raise V5PipelineError(
            "non-screening directions require a frozen 4B-to-8B method-dev structure receipt"
        )
    training = TransportTrainingParameters()
    errors = training.validate(require_registered=True)
    if errors:
        raise V5PipelineError("; ".join(errors))
    benchmark = load_bound_benchmark(workspace)
    trace = load_completed_trace_manifest(workspace, direction, "transport_train", benchmark)
    store_path = Path(sample_store_path)
    try:
        store_before = _file_signature(store_path)
        store_sha256 = sha256_file(store_path)
    except OSError as exc:
        raise V5PipelineError("transport-train raw sample store is unavailable") from exc
    if store_sha256 != trace.raw_sample_store_sha256:
        raise V5PipelineError("transport fitting raw store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="transport_train")
    sample_by_id = {record.sample_id: sample for record, sample in samples}
    direction_config = workspace.config.direction(direction)
    stage_parameters = {
        "trace_manifest_sha256": trace.content_sha256(),
        "raw_sample_store_sha256": store_sha256,
        "training": training.to_dict(),
    }
    generation_backend = FullPrefixGenerationBackend(
        source_path=direction_config.source_model_path,
        target_path=direction_config.target_model_path,
        source=trace.source,
        target=trace.target,
        samples=sample_by_id,
        source_device=source_device,
        target_device=device,
        identity_cache_path=identity_cache_path,
        spec=training.generation,
    )
    stage_parameters["generation_backend"] = generation_backend.parameters()
    lease = workspace.begin_stage(
        direction,
        "fit_transport",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/fit_transport"]
    work = workspace.control / "work" / direction / "fit_transport"
    try:
        trainer = SynchronousTransportTrainer(
            workspace=workspace,
            trace=trace,
            parameters=training,
            device=device,
            generation_backend=generation_backend,
            checkpoint_every_steps=checkpoint_every_steps,
            progress=progress,
        )
        fitted, normalizer_sha256, initializer_sha256 = trainer.fit(
            work,
            stage_input_sha256=lease.input_sha256,
        )
        if _file_signature(store_path) != store_before or sha256_file(store_path) != store_sha256:
            raise V5PipelineError("transport-train raw sample store changed during fitting")
        candidates = []
        for item in fitted:
            artifact = workspace.publish_file(
                item.path,
                logical_name=f"transport_r{item.rank}_s{item.seed}",
            )
            candidates.append(
                TransportCandidateArtifact(
                    candidate_id=item.candidate_id,
                    rank=item.rank,
                    seed=item.seed,
                    deployment_seed=item.seed == training.deployment_seed,
                    weights=TraceObjectRef.from_artifact(artifact),
                    parameter_count=item.parameter_count,
                    metrics=item.metrics,
                )
            )
        manifest = V5TransportFitManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            transport_train_split_sha256=workspace.config.split_sha256["transport_train"],
            trace_manifest_sha256=trace.content_sha256(),
            normalizer_sha256=normalizer_sha256,
            source=trace.source,
            target=trace.target,
            training=training,
            candidates=tuple(candidates),
            training_initializer_sha256=initializer_sha256,
            generation_sample_store_sha256=store_sha256,
        )
        manifest_errors = manifest.validate(workspace=workspace, trace=trace)
        if manifest_errors:
            raise V5PipelineError("; ".join(manifest_errors))
        manifest_path = work / "transport_fit_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={"transport_fit_manifest": manifest_path},
            metadata={
                "candidate_count": len(candidates),
                "trace_manifest_sha256": trace.content_sha256(),
                "training_initializer_sha256": initializer_sha256,
                "generation_sample_store_sha256": store_sha256,
                "transport_fit_manifest_sha256": manifest.content_sha256(),
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def _candidate_id(
    trace: V5TraceManifest,
    parameters: TransportTrainingParameters,
    *,
    rank: int,
    seed: int,
) -> str:
    digest = _sha256_bytes(
        _canonical_json_bytes(
            {
                "trace_manifest_sha256": trace.content_sha256(),
                "training": parameters.to_dict(),
                "rank": rank,
                "seed": seed,
            }
        )
    )
    return f"transport-r{rank}-s{seed}-{digest[:12]}"


def _validate_normalizers(values: Mapping[str, Any], target: CachedKVModelSpec) -> None:
    import torch

    expected_names = {
        "key_normalizer_mean",
        "key_normalizer_scale",
        "value_normalizer_mean",
        "value_normalizer_scale",
    }
    if set(values) != expected_names:
        raise V5PipelineError("transport normalizer tensor set is invalid")
    expected_shape = (target.num_layers, target.num_key_value_heads, target.head_dim)
    for name, value in values.items():
        if tuple(value.shape) != expected_shape or value.dtype != torch.float32:
            raise V5PipelineError(f"transport normalizer {name} shape or dtype is invalid")
        if not bool(torch.isfinite(value).all()):
            raise V5PipelineError(f"transport normalizer {name} is non-finite")
        if name.endswith("_scale") and bool((value <= 0).any()):
            raise V5PipelineError(f"transport normalizer {name} is not positive")


def _validate_ridge_initializer_checkpoint(
    values: Mapping[str, Any],
    target: CachedKVModelSpec,
) -> None:
    import torch

    if set(values) != RIDGE_INITIALIZER_TENSORS:
        raise V5PipelineError("ridge initializer tensor set is invalid")
    shapes = {
        "left_singular_vectors": (
            target.num_layers,
            target.num_key_value_heads,
            target.head_dim,
            target.head_dim,
        ),
        "singular_values": (
            target.num_layers,
            target.num_key_value_heads,
            target.head_dim,
        ),
        "right_singular_vectors": (
            target.num_layers,
            target.num_key_value_heads,
            target.head_dim,
            target.head_dim,
        ),
        "bias": (
            target.num_layers,
            target.num_key_value_heads,
            target.head_dim,
        ),
    }
    for prefix in ("key", "value"):
        for suffix, shape in shapes.items():
            value = values[f"{prefix}_{suffix}"]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise V5PipelineError(
                    f"ridge initializer {prefix}_{suffix} shape or dtype is invalid"
                )
            if not bool(torch.isfinite(value).all()):
                raise V5PipelineError(f"ridge initializer {prefix}_{suffix} is non-finite")
        singular = values[f"{prefix}_singular_values"]
        if bool((singular < 0).any()) or bool((singular[..., 1:] > singular[..., :-1]).any()):
            raise V5PipelineError(f"ridge initializer {prefix} singular values are invalid")


def _save_checkpoint_set(
    work: Path,
    contexts: Sequence[_CandidateContext],
    *,
    progress: Mapping[str, Any],
    binding_sha256: str,
) -> None:
    _validate_progress(progress)
    for context in contexts:
        _validate_candidate_state(context, progress=progress)
    generation = uuid.uuid4().hex
    directory = work / "checkpoints" / generation
    directory.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    for context in contexts:
        path = directory / f"{context.candidate_id}.safetensors"
        _save_candidate_checkpoint(
            path,
            context,
            progress=progress,
            binding_sha256=binding_sha256,
        )
        files[context.candidate_id] = {
            "path": path.relative_to(work).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _write_json_replace(
        work / "checkpoint_set.json",
        {
            "schema_version": V5_TRANSPORT_CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "generation": generation,
            "progress": dict(progress),
            "files": files,
        },
    )


def _load_checkpoint_set(
    pointer_path: Path,
    contexts: Sequence[_CandidateContext],
    *,
    binding_sha256: str,
    record_count: int,
    epochs: int,
    gradient_accumulation: int,
) -> dict[str, Any] | None:
    if not pointer_path.is_file():
        return None
    try:
        if pointer_path.is_symlink():
            raise V5PipelineError("transport checkpoint pointer cannot be a symbolic link")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if pointer.get("schema_version") != V5_TRANSPORT_CHECKPOINT_SCHEMA:
            raise V5PipelineError("transport checkpoint set schema mismatch")
        if pointer.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("transport checkpoint set input binding mismatch")
        generation = pointer.get("generation")
        if (
            not isinstance(generation, str)
            or len(generation) != 32
            or any(character not in "0123456789abcdef" for character in generation)
        ):
            raise V5PipelineError("transport checkpoint generation is invalid")
        files = pointer["files"]
        if set(files) != {context.candidate_id for context in contexts}:
            raise V5PipelineError("transport checkpoint candidate set is incomplete")
        expected_progress = dict(pointer["progress"])
        _validate_progress(
            expected_progress,
            record_count=record_count,
            epochs=epochs,
            gradient_accumulation=gradient_accumulation,
        )
        for context in contexts:
            item = files[context.candidate_id]
            expected_relative = (
                Path("checkpoints") / generation / f"{context.candidate_id}.safetensors"
            )
            if str(item.get("path")) != expected_relative.as_posix():
                raise V5PipelineError("transport checkpoint path is invalid")
            path = _resolve_read_only_work_file(pointer_path.parent, expected_relative)
            if type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0:
                raise V5PipelineError("transport checkpoint size is invalid")
            before = _file_signature(path)
            if before[2] != item["size_bytes"]:
                raise V5PipelineError("transport checkpoint size mismatch")
            if not _is_sha256(item.get("sha256")) or sha256_file(path) != item["sha256"]:
                raise V5PipelineError("transport checkpoint checksum mismatch")
            progress = _load_candidate_checkpoint(
                path,
                context,
                binding_sha256=binding_sha256,
            )
            if _file_signature(path) != before or sha256_file(path) != item["sha256"]:
                raise V5PipelineError("transport checkpoint changed while loading")
            if progress != expected_progress:
                raise V5PipelineError("transport candidate checkpoint progress differs")
        return expected_progress
    except V5PipelineError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("transport checkpoint set is malformed") from exc


def _save_candidate_checkpoint(
    path: Path,
    context: _CandidateContext,
    *,
    progress: Mapping[str, Any],
    binding_sha256: str,
) -> None:
    import torch
    from safetensors.torch import save_file

    _validate_candidate_state(context, progress=progress)
    tensors = {
        f"model.{name}": value.detach().cpu().contiguous()
        for name, value in context.module.state_dict().items()
    }
    optimizer_state = context.optimizer.state_dict()
    scalar_state: dict[str, dict[str, Any]] = {}
    for parameter_id, values in optimizer_state["state"].items():
        scalars: dict[str, Any] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                tensors[f"optimizer.{parameter_id}.{name}"] = value.detach().cpu().contiguous()
            else:
                scalars[name] = value
        if scalars:
            scalar_state[str(parameter_id)] = scalars
    metadata = {
        "schema_version": V5_TRANSPORT_CHECKPOINT_SCHEMA,
        "binding_sha256": binding_sha256,
        "candidate_id": context.candidate_id,
        "progress": _canonical_json_text(dict(progress)),
        "metric_sums": _canonical_json_text(context.metric_sums),
        "optimizer_param_groups": _canonical_json_text(optimizer_state["param_groups"]),
        "optimizer_scalar_state": _canonical_json_text(scalar_state),
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(tensors, temporary, metadata=metadata)
        canonicalize_safetensors_header(temporary)
        os.chmod(temporary, 0o444)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_candidate_checkpoint(
    path: Path,
    context: _CandidateContext,
    *,
    binding_sha256: str,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    expected_param_groups = json.loads(
        _canonical_json_text(context.optimizer.state_dict()["param_groups"])
    )
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("schema_version") != V5_TRANSPORT_CHECKPOINT_SCHEMA:
            raise V5PipelineError("transport candidate checkpoint schema mismatch")
        if metadata.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("transport candidate checkpoint binding mismatch")
        if metadata.get("candidate_id") != context.candidate_id:
            raise V5PipelineError("transport candidate checkpoint id mismatch")
        model_state: dict[str, Any] = {}
        optimizer_tensors: dict[int, dict[str, Any]] = {}
        for name in handle.keys():  # noqa: SIM118
            if name.startswith("model."):
                model_state[name.removeprefix("model.")] = handle.get_tensor(name)
            elif name.startswith("optimizer."):
                _, parameter_id, state_name = name.split(".", 2)
                optimizer_tensors.setdefault(int(parameter_id), {})[state_name] = handle.get_tensor(
                    name
                )
            else:
                raise V5PipelineError("transport checkpoint contains an unknown tensor")
        if any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in (
                *model_state.values(),
                *(tensor for values in optimizer_tensors.values() for tensor in values.values()),
            )
        ):
            raise V5PipelineError("transport candidate checkpoint contains non-finite tensors")
    context.module.load_state_dict(model_state, strict=True)
    scalar_state = json.loads(metadata["optimizer_scalar_state"])
    if not isinstance(scalar_state, dict):
        raise V5PipelineError("transport optimizer scalar state is invalid")
    optimizer_state: dict[int, dict[str, Any]] = optimizer_tensors
    for parameter_id, values in scalar_state.items():
        if not isinstance(values, dict):
            raise V5PipelineError("transport optimizer scalar state is invalid")
        optimizer_state.setdefault(int(parameter_id), {}).update(values)
    param_groups = json.loads(metadata["optimizer_param_groups"])
    if not isinstance(param_groups, list) or len(param_groups) != len(expected_param_groups):
        raise V5PipelineError("transport optimizer parameter groups are invalid")
    critical_group_fields = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
    }
    for observed, expected in zip(param_groups, expected_param_groups, strict=True):
        if not isinstance(observed, dict) or any(
            observed.get(name) != expected.get(name) for name in critical_group_fields
        ):
            raise V5PipelineError("transport optimizer hyperparameters changed")
    context.optimizer.load_state_dict(
        {
            "state": optimizer_state,
            "param_groups": param_groups,
        }
    )
    metric_sums = json.loads(metadata["metric_sums"])
    if not isinstance(metric_sums, dict) or set(metric_sums) != set(_TERM_NAMES):
        raise V5PipelineError("transport checkpoint metric set is invalid")
    restored_metrics = {name: float(metric_sums[name]) for name in _TERM_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in restored_metrics.values()):
        raise V5PipelineError("transport checkpoint metrics are invalid")
    context.metric_sums = restored_metrics
    progress = json.loads(metadata["progress"])
    _validate_progress(progress)
    _validate_candidate_state(context, progress=progress)
    return dict(progress)


def _validate_progress(
    progress: Mapping[str, Any],
    *,
    record_count: int | None = None,
    epochs: int | None = None,
    gradient_accumulation: int | None = None,
) -> None:
    if set(progress) != {"epoch", "position", "optimizer_steps", "samples_seen"}:
        raise V5PipelineError("transport checkpoint progress fields are invalid")
    if any(type(progress[name]) is not int or progress[name] < 0 for name in progress):
        raise V5PipelineError("transport checkpoint progress values are invalid")
    if record_count is None and epochs is None and gradient_accumulation is None:
        return
    if (
        type(record_count) is not int
        or record_count <= 0
        or type(epochs) is not int
        or epochs <= 0
        or type(gradient_accumulation) is not int
        or gradient_accumulation <= 0
    ):
        raise V5PipelineError("transport checkpoint validation contract is invalid")
    epoch = progress["epoch"]
    position = progress["position"]
    if epoch > epochs or (epoch == epochs and position != 0):
        raise V5PipelineError("transport checkpoint progress exceeds training bounds")
    if position >= record_count or (position and position % gradient_accumulation != 0):
        raise V5PipelineError("transport checkpoint position is not an optimizer boundary")
    expected_samples = epoch * record_count + position
    steps_per_epoch = math.ceil(record_count / gradient_accumulation)
    expected_steps = epoch * steps_per_epoch + position // gradient_accumulation
    if progress["samples_seen"] != expected_samples:
        raise V5PipelineError("transport checkpoint sample count is inconsistent")
    if progress["optimizer_steps"] != expected_steps:
        raise V5PipelineError("transport checkpoint optimizer step count is inconsistent")


def _validate_candidate_state(
    context: _CandidateContext,
    *,
    progress: Mapping[str, Any],
) -> None:
    import torch

    _validate_progress(progress)
    for value in context.module.state_dict().values():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise V5PipelineError("transport candidate model state is non-finite")
    if set(context.metric_sums) != set(_TERM_NAMES) or any(
        not math.isfinite(value) or value < 0 for value in context.metric_sums.values()
    ):
        raise V5PipelineError("transport candidate metric state is invalid")
    optimizer_steps = progress["optimizer_steps"]
    state_by_parameter = context.optimizer.state
    parameters = [
        parameter for group in context.optimizer.param_groups for parameter in group["params"]
    ]
    if optimizer_steps == 0:
        if state_by_parameter:
            raise V5PipelineError("unstepped transport optimizer unexpectedly has state")
        return
    if set(state_by_parameter) != set(parameters):
        raise V5PipelineError("transport optimizer state is incomplete")
    for parameter in parameters:
        state = state_by_parameter[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise V5PipelineError("transport optimizer tensor set is invalid")
        step = state["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all())
            or float(step.item()) != optimizer_steps
        ):
            raise V5PipelineError("transport optimizer step state is inconsistent")
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not bool(torch.isfinite(value).all())
            ):
                raise V5PipelineError(f"transport optimizer {name} state is invalid")


def _save_runtime_weights(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            temporary,
            metadata=dict(metadata),
        )
        canonicalize_safetensors_header(temporary)
        os.chmod(temporary, 0o444)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(dict(value), indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_read_only_work_file(work: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise V5PipelineError("transport checkpoint path escapes its work directory")
    root = work.resolve()
    lexical = root.joinpath(*relative.parts)
    try:
        resolved = lexical.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise V5PipelineError("transport checkpoint file is unavailable") from exc
    if resolved != lexical or not resolved.is_relative_to(root):
        raise V5PipelineError("transport checkpoint path uses a symbolic-link escape")
    if not resolved.is_file() or stat.st_mode & 0o222:
        raise V5PipelineError("transport checkpoint file must be immutable")
    return resolved


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _verify_workspace_object(
    workspace: V5PipelineWorkspace,
    reference: TraceObjectRef,
) -> Path:
    relative = Path(reference.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise V5PipelineError("transport candidate path escapes the pipeline workspace")
    lexical = workspace.root.joinpath(*relative.parts)
    try:
        path = lexical.resolve(strict=True)
        before = _file_signature(path)
    except OSError as exc:
        raise V5PipelineError("transport candidate object is unavailable") from exc
    if path != lexical or not path.is_relative_to(workspace.root):
        raise V5PipelineError("transport candidate path uses a symbolic-link escape")
    if before[2] != reference.size_bytes or path.stat().st_mode & 0o222:
        raise V5PipelineError("transport candidate size or read-only mode changed")
    if sha256_file(path) != reference.sha256:
        raise V5PipelineError("transport candidate checksum mismatch")
    if _file_signature(path) != before or sha256_file(path) != reference.sha256:
        raise V5PipelineError("transport candidate changed while hashing")
    return path


def _canonical_json_text(value: Any) -> str:
    return _canonical_json_bytes(value).decode("utf-8").rstrip("\n")


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
        raise V5PipelineError("transport metadata is not finite canonical JSON") from exc


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


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def stderr_fit_progress(every: int = 16) -> Callable[[int, int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("transport progress interval must be positive")

    def report(index: int, total: int, epoch: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(
                f"transport fit {index}/{total} epoch={epoch + 1} sample={sample_id}",
                file=sys.stderr,
                flush=True,
            )

    return report
