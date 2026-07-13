"""Method-dev evaluation and one-time transport structure freezing for v5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from goldenexperience.benchmarks.publication import SPLIT_COUNTS
from goldenexperience.size_variant.cached_kv_manifest import sha256_file
from goldenexperience.size_variant.head_aware_transport import (
    TransportScreeningCandidate,
    select_transport_candidate,
)
from goldenexperience.size_variant.risk_gate import unsafe_label
from goldenexperience.size_variant.selective_manifest import TransportQualityEvidence
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
)
from goldenexperience.size_variant.v5_fit import (
    DEPLOYMENT_SEED,
    REGISTERED_RANKS,
    SCREENING_DIRECTION,
    TransportCandidateArtifact,
    V5TransportFitManifest,
    load_completed_transport_fit,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)

V5_METHOD_DEV_REPORT_SCHEMA = "goldenexperience.v5_method_dev_report.v1"
V5_METHOD_DEV_CHECKPOINT_SCHEMA = "goldenexperience.v5_method_dev_checkpoint.v1"
V5_FROZEN_STRUCTURE_SCHEMA = "goldenexperience.v5_frozen_transport_structure.v1"
METHOD_DEV_GENERATION_TOKENS = 16
METHOD_DEV_SELECTION_RULE = (
    "lexicographic_mean_task_preservation_then_oracle_safe_coverage_then_"
    "greedy_agreement_then_negative_mean_p95_transform_ms"
)
METHOD_DEV_SEED_AGGREGATION = "arithmetic_mean_and_population_standard_deviation"


@dataclass(frozen=True)
class MethodDevMeasurement:
    sample_id: str
    candidate_id: str
    rank: int
    seed: int
    native_task_score: float
    bridge_task_score: float
    task_pass_threshold: float
    greedy_matches: int
    greedy_tokens: int
    native_nll: float
    bridge_nll: float
    teacher_tokens: int
    transform_ms: float
    native_prediction_sha256: str
    bridge_prediction_sha256: str
    native_tokens_sha256: str
    bridge_tokens_sha256: str

    @property
    def task_preservation(self) -> float:
        return 1.0 - max(0.0, self.native_task_score - self.bridge_task_score)

    @property
    def greedy_agreement(self) -> float:
        return self.greedy_matches / self.greedy_tokens

    @property
    def perplexity_drift_pct(self) -> float:
        if self.teacher_tokens <= 0:
            return math.inf
        log_ratio = (self.bridge_nll - self.native_nll) / self.teacher_tokens
        try:
            return abs(math.expm1(log_ratio)) * 100
        except OverflowError:
            return math.inf

    @property
    def oracle_safe(self) -> bool:
        return not unsafe_label(
            native_task_passed=self.native_task_score >= self.task_pass_threshold,
            bridge_task_passed=self.bridge_task_score >= self.task_pass_threshold,
            greedy_agreement=self.greedy_agreement,
            perplexity_drift_pct=self.perplexity_drift_pct,
        )

    def validate(
        self,
        *,
        record: TraceRecord | None = None,
        candidate: TransportCandidateArtifact | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if not self.sample_id or not self.candidate_id:
            errors.append("method-dev measurement identifiers are required")
        for name in (
            "native_prediction_sha256",
            "bridge_prediction_sha256",
            "native_tokens_sha256",
            "bridge_tokens_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"method-dev {name} is invalid")
        if record is not None and self.sample_id != record.sample_id:
            errors.append("method-dev measurement belongs to another sample")
        if candidate is not None and (
            self.candidate_id != candidate.candidate_id
            or self.rank != candidate.rank
            or self.seed != candidate.seed
        ):
            errors.append("method-dev measurement belongs to another candidate")
        if (
            type(self.rank) is not int
            or type(self.seed) is not int
            or self.rank not in REGISTERED_RANKS
            or self.seed not in {17, 29, 43}
        ):
            errors.append("method-dev measurement rank or seed is invalid")
        for name in ("native_task_score", "bridge_task_score", "task_pass_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"method-dev {name} must be between zero and one")
        if (
            type(self.greedy_matches) is not int
            or type(self.greedy_tokens) is not int
            or self.greedy_tokens <= 0
            or not 0 <= self.greedy_matches <= self.greedy_tokens
        ):
            errors.append("method-dev greedy counts are invalid")
        if type(self.teacher_tokens) is not int or self.teacher_tokens <= 0:
            errors.append("method-dev teacher token count is invalid")
        for name in ("native_nll", "bridge_nll", "transform_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"method-dev {name} must be finite and non-negative")
        if not math.isfinite(self.perplexity_drift_pct):
            errors.append("method-dev perplexity drift is non-finite")
        return errors


@dataclass(frozen=True)
class CandidateMethodDevMetrics:
    candidate_id: str
    rank: int
    seed: int
    prompt_count: int
    safe_count: int
    greedy_matches: int
    greedy_tokens: int
    teacher_tokens: int
    native_task_score: float
    bridge_task_score: float
    task_score: float
    oracle_safe_coverage: float
    greedy_agreement: float
    perplexity_drift_pct: float
    p50_transform_ms: float
    p95_transform_ms: float

    @classmethod
    def aggregate(
        cls,
        candidate: TransportCandidateArtifact,
        measurements: Sequence[MethodDevMeasurement],
    ) -> CandidateMethodDevMetrics:
        if not measurements:
            raise V5PipelineError("method-dev candidate has no measurements")
        errors = [
            error
            for measurement in measurements
            for error in measurement.validate(candidate=candidate)
        ]
        if errors:
            raise V5PipelineError("; ".join(errors))
        prompt_count = len(measurements)
        safe_count = sum(measurement.oracle_safe for measurement in measurements)
        greedy_matches = sum(item.greedy_matches for item in measurements)
        greedy_tokens = sum(item.greedy_tokens for item in measurements)
        teacher_tokens = sum(item.teacher_tokens for item in measurements)
        native_nll = sum(item.native_nll for item in measurements)
        bridge_nll = sum(item.bridge_nll for item in measurements)
        transform_times = sorted(item.transform_ms for item in measurements)
        log_ratio = (bridge_nll - native_nll) / teacher_tokens
        try:
            perplexity_drift = abs(math.expm1(log_ratio)) * 100
        except OverflowError:
            perplexity_drift = math.inf
        return cls(
            candidate_id=candidate.candidate_id,
            rank=candidate.rank,
            seed=candidate.seed,
            prompt_count=prompt_count,
            safe_count=safe_count,
            greedy_matches=greedy_matches,
            greedy_tokens=greedy_tokens,
            teacher_tokens=teacher_tokens,
            native_task_score=sum(item.native_task_score for item in measurements) / prompt_count,
            bridge_task_score=sum(item.bridge_task_score for item in measurements) / prompt_count,
            task_score=sum(item.task_preservation for item in measurements) / prompt_count,
            oracle_safe_coverage=safe_count / prompt_count,
            greedy_agreement=greedy_matches / greedy_tokens,
            perplexity_drift_pct=perplexity_drift,
            p50_transform_ms=_percentile(transform_times, 0.50),
            p95_transform_ms=_percentile(transform_times, 0.95),
        )

    def validate(self, *, expected_prompts: int) -> list[str]:
        errors: list[str] = []
        if not self.candidate_id or type(self.rank) is not int or self.rank not in REGISTERED_RANKS:
            errors.append("method-dev candidate identity is invalid")
        if type(self.seed) is not int or self.seed not in {17, 29, 43}:
            errors.append("method-dev candidate seed is invalid")
        if type(self.prompt_count) is not int or self.prompt_count != expected_prompts:
            errors.append("method-dev candidate prompt count is inconsistent")
        if (
            type(self.safe_count) is not int
            or type(self.prompt_count) is not int
            or not 0 <= self.safe_count <= self.prompt_count
        ):
            errors.append("method-dev candidate safe count is invalid")
        if (
            type(self.greedy_matches) is not int
            or type(self.greedy_tokens) is not int
            or self.greedy_tokens <= 0
            or not 0 <= self.greedy_matches <= self.greedy_tokens
        ):
            errors.append("method-dev candidate greedy counts are invalid")
        if type(self.teacher_tokens) is not int or self.teacher_tokens <= 0:
            errors.append("method-dev candidate teacher token count is invalid")
        for name in (
            "native_task_score",
            "bridge_task_score",
            "task_score",
            "oracle_safe_coverage",
            "greedy_agreement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"method-dev candidate {name} is invalid")
        if (
            type(self.prompt_count) is int
            and self.prompt_count > 0
            and abs(self.oracle_safe_coverage - self.safe_count / self.prompt_count) > 1e-12
        ):
            errors.append("method-dev candidate coverage is inconsistent")
        if (
            type(self.greedy_tokens) is int
            and self.greedy_tokens > 0
            and abs(self.greedy_agreement - self.greedy_matches / self.greedy_tokens) > 1e-12
        ):
            errors.append("method-dev candidate greedy agreement is inconsistent")
        for name in ("perplexity_drift_pct", "p50_transform_ms", "p95_transform_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"method-dev candidate {name} is invalid")
        if self.p95_transform_ms < self.p50_transform_ms:
            errors.append("method-dev transform percentiles are inconsistent")
        return errors


@dataclass(frozen=True)
class RankMethodDevAggregate:
    rank: int
    seed_count: int
    mean_task_score: float
    std_task_score: float
    mean_oracle_safe_coverage: float
    std_oracle_safe_coverage: float
    mean_greedy_agreement: float
    std_greedy_agreement: float
    mean_p95_transform_ms: float
    std_p95_transform_ms: float

    @classmethod
    def aggregate(
        cls,
        rank: int,
        candidates: Sequence[CandidateMethodDevMetrics],
    ) -> RankMethodDevAggregate:
        if {item.seed for item in candidates} != {17, 29, 43} or any(
            item.rank != rank for item in candidates
        ):
            raise V5PipelineError("method-dev rank aggregate lacks the registered seed set")

        def moments(name: str) -> tuple[float, float]:
            values = [float(getattr(item, name)) for item in candidates]
            return statistics.fmean(values), statistics.pstdev(values)

        task_mean, task_std = moments("task_score")
        coverage_mean, coverage_std = moments("oracle_safe_coverage")
        greedy_mean, greedy_std = moments("greedy_agreement")
        transform_mean, transform_std = moments("p95_transform_ms")
        return cls(
            rank=rank,
            seed_count=len(candidates),
            mean_task_score=task_mean,
            std_task_score=task_std,
            mean_oracle_safe_coverage=coverage_mean,
            std_oracle_safe_coverage=coverage_std,
            mean_greedy_agreement=greedy_mean,
            std_greedy_agreement=greedy_std,
            mean_p95_transform_ms=transform_mean,
            std_p95_transform_ms=transform_std,
        )

    def screening_candidate(self, *, attention_loss_weight: float) -> TransportScreeningCandidate:
        return TransportScreeningCandidate(
            candidate_id=f"rank-{self.rank}-three-seed-aggregate",
            direction=SCREENING_DIRECTION,
            rank=self.rank,
            attention_loss_weight=attention_loss_weight,
            task_score=self.mean_task_score,
            oracle_safe_coverage=self.mean_oracle_safe_coverage,
            greedy_agreement=self.mean_greedy_agreement,
            transform_cost_ms=self.mean_p95_transform_ms,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if (
            type(self.rank) is not int
            or type(self.seed_count) is not int
            or self.rank not in REGISTERED_RANKS
            or self.seed_count != 3
        ):
            errors.append("method-dev rank aggregate identity is invalid")
        for name in (
            "mean_task_score",
            "mean_oracle_safe_coverage",
            "mean_greedy_agreement",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                errors.append(f"method-dev rank {name} is invalid")
        for name in (
            "std_task_score",
            "std_oracle_safe_coverage",
            "std_greedy_agreement",
            "mean_p95_transform_ms",
            "std_p95_transform_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                errors.append(f"method-dev rank {name} is invalid")
        return errors


@dataclass(frozen=True)
class FrozenTransportStructure:
    pipeline_id: str
    direction: str
    code_sha256: str
    benchmark_manifest_sha256: str
    method_dev_split_sha256: str
    method_dev_trace_manifest_sha256: str
    method_dev_raw_store_sha256: str
    transport_fit_manifest_sha256: str
    method_dev_report_sha256: str
    generation_tokens: int
    selection_rule: str
    seed_aggregation: str
    selected_rank: int
    source_window: int
    deployment_seed: int
    deployment_candidate_id: str
    deployment_weights: TraceObjectRef
    candidates: tuple[CandidateMethodDevMetrics, ...]
    rank_aggregates: tuple[RankMethodDevAggregate, ...]
    deployment_quality: TransportQualityEvidence
    schema_version: str = V5_FROZEN_STRUCTURE_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        fit: V5TransportFitManifest,
        method_trace: V5TraceManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_FROZEN_STRUCTURE_SCHEMA:
            errors.append("unsupported frozen transport structure schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("frozen structure belongs to another pipeline")
        if self.direction != SCREENING_DIRECTION or fit.direction != self.direction:
            errors.append("transport structure was not screened on Qwen3 4B-to-8B")
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("frozen structure code hash mismatch")
        if self.benchmark_manifest_sha256 != workspace.config.benchmark_manifest_sha256:
            errors.append("frozen structure benchmark hash mismatch")
        if self.method_dev_split_sha256 != workspace.config.split_sha256["method_dev"]:
            errors.append("frozen structure method-dev split hash mismatch")
        if self.method_dev_trace_manifest_sha256 != method_trace.content_sha256():
            errors.append("frozen structure trace manifest hash mismatch")
        if self.method_dev_raw_store_sha256 != method_trace.raw_sample_store_sha256:
            errors.append("frozen structure raw sample hash mismatch")
        if self.transport_fit_manifest_sha256 != fit.content_sha256():
            errors.append("frozen structure fit manifest hash mismatch")
        for name in ("method_dev_report_sha256",):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"frozen structure {name} is invalid")
        if (
            type(self.generation_tokens) is not int
            or self.generation_tokens != METHOD_DEV_GENERATION_TOKENS
        ):
            errors.append("frozen structure generation length changed")
        if self.selection_rule != METHOD_DEV_SELECTION_RULE:
            errors.append("frozen structure selection rule changed")
        if self.seed_aggregation != METHOD_DEV_SEED_AGGREGATION:
            errors.append("frozen structure seed aggregation changed")
        if (
            type(self.source_window) is not int
            or self.source_window != fit.training.source_window
            or self.source_window != 3
        ):
            errors.append("frozen structure source window changed")
        if type(self.deployment_seed) is not int or self.deployment_seed != DEPLOYMENT_SEED:
            errors.append("frozen structure deployment seed changed")
        if type(self.selected_rank) is not int or self.selected_rank not in REGISTERED_RANKS:
            errors.append("frozen structure selected rank is invalid")
        expected_pairs = {(rank, seed) for rank in REGISTERED_RANKS for seed in (17, 29, 43)}
        observed_pairs = {(item.rank, item.seed) for item in self.candidates}
        if observed_pairs != expected_pairs or len(self.candidates) != len(expected_pairs):
            errors.append("frozen structure candidate matrix is incomplete")
        expected_prompts = SPLIT_COUNTS["method_dev"]
        fit_by_pair = {(item.rank, item.seed): item for item in fit.candidates}
        for candidate in self.candidates:
            errors.extend(candidate.validate(expected_prompts=expected_prompts))
            fit_candidate = fit_by_pair.get((candidate.rank, candidate.seed))
            if fit_candidate is None or candidate.candidate_id != fit_candidate.candidate_id:
                errors.append("frozen structure candidate identity differs from fitted weights")
        if {item.rank for item in self.rank_aggregates} != set(REGISTERED_RANKS) or len(
            self.rank_aggregates
        ) != len(REGISTERED_RANKS):
            errors.append("frozen structure rank aggregates are incomplete")
        for aggregate in self.rank_aggregates:
            errors.extend(aggregate.validate())
            try:
                expected_aggregate = RankMethodDevAggregate.aggregate(
                    aggregate.rank,
                    [item for item in self.candidates if item.rank == aggregate.rank],
                )
            except V5PipelineError as exc:
                errors.append(str(exc))
            else:
                if aggregate != expected_aggregate:
                    errors.append("frozen structure rank aggregate is inconsistent")
        if len(self.rank_aggregates) == len(REGISTERED_RANKS):
            selected = select_transport_candidate(
                tuple(
                    item.screening_candidate(
                        attention_loss_weight=fit.training.loss.attention_logit_kl
                    )
                    for item in self.rank_aggregates
                )
            )
            if selected.rank != self.selected_rank:
                errors.append("frozen structure selected rank differs from registered ordering")
        fit_deployment = [
            item
            for item in fit.candidates
            if item.rank == self.selected_rank and item.seed == DEPLOYMENT_SEED
        ]
        metric_deployment = [
            item
            for item in self.candidates
            if item.rank == self.selected_rank and item.seed == DEPLOYMENT_SEED
        ]
        if len(fit_deployment) != 1 or len(metric_deployment) != 1:
            errors.append("frozen structure deployment candidate is missing")
        else:
            fit_candidate = fit_deployment[0]
            metrics = metric_deployment[0]
            if (
                self.deployment_candidate_id != fit_candidate.candidate_id
                or self.deployment_weights != fit_candidate.weights
            ):
                errors.append("frozen structure deployment artifact changed")
            expected_quality = TransportQualityEvidence(
                evaluation_dataset_sha256=self.method_dev_split_sha256,
                prompt_count=metrics.prompt_count,
                task_score=metrics.task_score,
                oracle_safe_coverage=metrics.oracle_safe_coverage,
                greedy_agreement=metrics.greedy_agreement,
            )
            if self.deployment_quality != expected_quality:
                errors.append("frozen structure deployment quality is inconsistent")
        errors.extend(self.deployment_quality.gate_errors(self.method_dev_split_sha256))
        if type(self.deployment_quality.prompt_count) is not int:
            errors.append("frozen structure quality prompt count must be an integer")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenTransportStructure:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            method_dev_split_sha256=str(payload["method_dev_split_sha256"]),
            method_dev_trace_manifest_sha256=str(payload["method_dev_trace_manifest_sha256"]),
            method_dev_raw_store_sha256=str(payload["method_dev_raw_store_sha256"]),
            transport_fit_manifest_sha256=str(payload["transport_fit_manifest_sha256"]),
            method_dev_report_sha256=str(payload["method_dev_report_sha256"]),
            generation_tokens=payload["generation_tokens"],
            selection_rule=str(payload["selection_rule"]),
            seed_aggregation=str(payload["seed_aggregation"]),
            selected_rank=payload["selected_rank"],
            source_window=payload["source_window"],
            deployment_seed=payload["deployment_seed"],
            deployment_candidate_id=str(payload["deployment_candidate_id"]),
            deployment_weights=TraceObjectRef(**payload["deployment_weights"]),
            candidates=tuple(
                CandidateMethodDevMetrics(**item) for item in payload.get("candidates", ())
            ),
            rank_aggregates=tuple(
                RankMethodDevAggregate(**item) for item in payload.get("rank_aggregates", ())
            ),
            deployment_quality=TransportQualityEvidence(**payload["deployment_quality"]),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        fit: V5TransportFitManifest,
        method_trace: V5TraceManifest,
    ) -> FrozenTransportStructure:
        try:
            value = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = value.validate(workspace=workspace, fit=fit, method_trace=method_trace)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("frozen structure receipt is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        return value


class MethodDevEvaluator(Protocol):
    def __enter__(self) -> MethodDevEvaluator: ...

    def __exit__(self, *_args: object) -> None: ...

    def evaluate(
        self,
        record: TraceRecord,
        sample: RawBenchmarkSample,
    ) -> Sequence[MethodDevMeasurement]: ...


def freeze_transport_structure(
    *,
    workspace: V5PipelineWorkspace,
    fit: V5TransportFitManifest,
    method_trace: V5TraceManifest,
    measurements: Sequence[MethodDevMeasurement],
    report_sha256: str,
) -> FrozenTransportStructure:
    if len(measurements) != len(method_trace.records) * len(fit.candidates):
        raise V5PipelineError("method-dev detailed measurement matrix is incomplete")
    measurements_by_sample: dict[str, list[MethodDevMeasurement]] = {}
    for measurement in measurements:
        measurements_by_sample.setdefault(measurement.sample_id, []).append(measurement)
    if set(measurements_by_sample) != {record.sample_id for record in method_trace.records}:
        raise V5PipelineError("method-dev detailed measurement sample set is incomplete")
    for record in method_trace.records:
        _validate_measurement_set(
            measurements_by_sample[record.sample_id],
            record,
            fit.candidates,
        )
    candidates = []
    for artifact in sorted(fit.candidates, key=lambda item: (item.rank, item.seed)):
        rows = [item for item in measurements if item.candidate_id == artifact.candidate_id]
        candidates.append(CandidateMethodDevMetrics.aggregate(artifact, rows))
    aggregates = tuple(
        RankMethodDevAggregate.aggregate(
            rank,
            [item for item in candidates if item.rank == rank],
        )
        for rank in REGISTERED_RANKS
    )
    selected = select_transport_candidate(
        tuple(
            item.screening_candidate(attention_loss_weight=fit.training.loss.attention_logit_kl)
            for item in aggregates
        )
    )
    deployment_artifact = next(
        item
        for item in fit.candidates
        if item.rank == selected.rank and item.seed == DEPLOYMENT_SEED
    )
    deployment_metrics = next(
        item for item in candidates if item.rank == selected.rank and item.seed == DEPLOYMENT_SEED
    )
    quality = TransportQualityEvidence(
        evaluation_dataset_sha256=workspace.config.split_sha256["method_dev"],
        prompt_count=deployment_metrics.prompt_count,
        task_score=deployment_metrics.task_score,
        oracle_safe_coverage=deployment_metrics.oracle_safe_coverage,
        greedy_agreement=deployment_metrics.greedy_agreement,
    )
    receipt = FrozenTransportStructure(
        pipeline_id=workspace.config.pipeline_id,
        direction=fit.direction,
        code_sha256=workspace.config.code_sha256,
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        method_dev_split_sha256=workspace.config.split_sha256["method_dev"],
        method_dev_trace_manifest_sha256=method_trace.content_sha256(),
        method_dev_raw_store_sha256=method_trace.raw_sample_store_sha256,
        transport_fit_manifest_sha256=fit.content_sha256(),
        method_dev_report_sha256=report_sha256,
        generation_tokens=METHOD_DEV_GENERATION_TOKENS,
        selection_rule=METHOD_DEV_SELECTION_RULE,
        seed_aggregation=METHOD_DEV_SEED_AGGREGATION,
        selected_rank=selected.rank,
        source_window=fit.training.source_window,
        deployment_seed=DEPLOYMENT_SEED,
        deployment_candidate_id=deployment_artifact.candidate_id,
        deployment_weights=deployment_artifact.weights,
        candidates=tuple(candidates),
        rank_aggregates=aggregates,
        deployment_quality=quality,
    )
    errors = receipt.validate(workspace=workspace, fit=fit, method_trace=method_trace)
    if errors:
        raise V5PipelineError("; ".join(errors))
    return receipt


def run_method_dev_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], MethodDevEvaluator],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Evaluate all fitted seeds and atomically freeze the selected rank."""

    if direction != SCREENING_DIRECTION:
        raise V5PipelineError("transport structure screening is restricted to Qwen3 4B-to-8B")
    benchmark = load_bound_benchmark(workspace)
    fit, _ = load_completed_transport_fit(workspace, direction, benchmark)
    method_trace = load_completed_trace_manifest(workspace, direction, "method_dev", benchmark)
    store_path = Path(sample_store_path)
    before = _file_signature(store_path)
    store_sha256 = sha256_file(store_path)
    if store_sha256 != method_trace.raw_sample_store_sha256:
        raise V5PipelineError("method-dev raw sample store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="method_dev")
    trace_by_id = {record.sample_id: record for record in method_trace.records}
    if set(trace_by_id) != {record.sample_id for record, _ in samples}:
        raise V5PipelineError("method-dev samples differ from trace manifest")
    stage_parameters = {
        "transport_fit_manifest_sha256": fit.content_sha256(),
        "method_dev_trace_manifest_sha256": method_trace.content_sha256(),
        "raw_sample_store_sha256": store_sha256,
        "generation_tokens": METHOD_DEV_GENERATION_TOKENS,
        "selection_rule": METHOD_DEV_SELECTION_RULE,
        "seed_aggregation": METHOD_DEV_SEED_AGGREGATION,
        "evaluator": dict(evaluator_parameters),
    }
    lease = workspace.begin_stage(
        direction,
        "evaluate_method_dev",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/evaluate_method_dev"]
    work = workspace.control / "work" / direction / "evaluate_method_dev"
    checkpoint_dir = work / "samples"
    try:
        measurements: list[MethodDevMeasurement] = []
        with evaluator_factory() as evaluator:
            for index, (benchmark_record, sample) in enumerate(samples, start=1):
                trace_record = trace_by_id[benchmark_record.sample_id]
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                restored = _load_measurement_checkpoint(
                    checkpoint,
                    binding_sha256=lease.input_sha256,
                    record=trace_record,
                    candidates=fit.candidates,
                )
                if restored is None:
                    restored = tuple(evaluator.evaluate(trace_record, sample))
                    _validate_measurement_set(restored, trace_record, fit.candidates)
                    _write_measurement_checkpoint(
                        checkpoint,
                        binding_sha256=lease.input_sha256,
                        record=trace_record,
                        measurements=restored,
                    )
                measurements.extend(restored)
                if progress is not None:
                    progress(index, len(samples), sample.sample_id)
        if _file_signature(store_path) != before or sha256_file(store_path) != store_sha256:
            raise V5PipelineError("method-dev raw sample store changed during evaluation")
        report = {
            "schema_version": V5_METHOD_DEV_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": direction,
            "code_sha256": workspace.config.code_sha256,
            "method_dev_split_sha256": workspace.config.split_sha256["method_dev"],
            "method_dev_trace_manifest_sha256": method_trace.content_sha256(),
            "transport_fit_manifest_sha256": fit.content_sha256(),
            "raw_sample_store_sha256": store_sha256,
            "evaluator": dict(evaluator_parameters),
            "generation_tokens": METHOD_DEV_GENERATION_TOKENS,
            "measurements": [asdict(item) for item in measurements],
        }
        report_path = work / "method_dev_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        receipt = freeze_transport_structure(
            workspace=workspace,
            fit=fit,
            method_trace=method_trace,
            measurements=measurements,
            report_sha256=report_sha256,
        )
        receipt_path = work / "frozen_transport_structure.json"
        _write_json_replace(receipt_path, receipt.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "method_dev_report": report_path,
                "frozen_transport_structure": receipt_path,
            },
            metadata={
                "selected_rank": receipt.selected_rank,
                "deployment_seed": receipt.deployment_seed,
                "method_dev_report_sha256": report_sha256,
                "frozen_structure_sha256": receipt.content_sha256(),
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def load_frozen_transport_structure(
    workspace: V5PipelineWorkspace,
) -> tuple[FrozenTransportStructure, V5TransportFitManifest, V5TraceManifest]:
    """Load the globally frozen 4B-to-8B structure receipt for downstream directions."""

    benchmark = load_bound_benchmark(workspace)
    fit, _ = load_completed_transport_fit(workspace, SCREENING_DIRECTION, benchmark)
    method_trace = load_completed_trace_manifest(
        workspace,
        SCREENING_DIRECTION,
        "method_dev",
        benchmark,
    )
    state = workspace.state()
    stage = state.stages.get(f"{SCREENING_DIRECTION}/evaluate_method_dev")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("downstream transport fitting requires frozen method-dev structure")
    artifact = stage.outputs.get("frozen_transport_structure")
    report_artifact = stage.outputs.get("method_dev_report")
    if artifact is None:
        raise V5PipelineError("method-dev stage lacks its frozen structure receipt")
    if report_artifact is None:
        raise V5PipelineError("method-dev stage lacks its detailed report")
    path = workspace.artifact_path(artifact, verify_hash=True)
    workspace.artifact_path(report_artifact, verify_hash=True)
    structure = FrozenTransportStructure.load(
        path,
        workspace=workspace,
        fit=fit,
        method_trace=method_trace,
    )
    if structure.method_dev_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("frozen structure refers to another method-dev report")
    return structure, fit, method_trace


def _validate_measurement_set(
    measurements: Sequence[MethodDevMeasurement],
    record: TraceRecord,
    candidates: Sequence[TransportCandidateArtifact],
) -> None:
    by_id = {item.candidate_id: item for item in candidates}
    if len(by_id) != len(candidates):
        raise V5PipelineError("fit manifest candidate ids are not unique")
    observed = [item.candidate_id for item in measurements]
    if len(observed) != len(set(observed)) or set(observed) != set(by_id):
        raise V5PipelineError("method-dev sample candidate measurements are incomplete")
    native_signatures = {
        (
            item.native_task_score,
            item.task_pass_threshold,
            item.native_nll,
            item.native_prediction_sha256,
            item.native_tokens_sha256,
        )
        for item in measurements
    }
    if len(native_signatures) != 1:
        raise V5PipelineError("method-dev candidates do not share one native baseline")
    if any(
        item.greedy_tokens != METHOD_DEV_GENERATION_TOKENS
        or item.teacher_tokens != METHOD_DEV_GENERATION_TOKENS
        for item in measurements
    ):
        raise V5PipelineError("method-dev continuation length differs from the frozen contract")
    errors = [
        error
        for item in measurements
        for error in item.validate(record=record, candidate=by_id.get(item.candidate_id))
    ]
    if errors:
        raise V5PipelineError("; ".join(errors))


def _write_measurement_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    record: TraceRecord,
    measurements: Sequence[MethodDevMeasurement],
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_METHOD_DEV_CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "sample_id": record.sample_id,
            "content_sha256": record.content_sha256,
            "measurements": [asdict(item) for item in measurements],
        },
    )


def _load_measurement_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    record: TraceRecord,
    candidates: Sequence[TransportCandidateArtifact],
) -> tuple[MethodDevMeasurement, ...] | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("method-dev checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_METHOD_DEV_CHECKPOINT_SCHEMA:
            raise V5PipelineError("method-dev checkpoint schema mismatch")
        if payload.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("method-dev checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != record.sample_id
            or payload.get("content_sha256") != record.content_sha256
        ):
            raise V5PipelineError("method-dev checkpoint sample binding mismatch")
        values = tuple(MethodDevMeasurement(**item) for item in payload["measurements"])
        _validate_measurement_set(values, record, candidates)
        return values
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("method-dev checkpoint is malformed") from exc


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values or not 0 <= quantile <= 1:
        raise V5PipelineError("method-dev percentile input is invalid")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


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
        raise V5PipelineError("method-dev metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("method-dev sample store is unavailable") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256_text(value: str) -> str:
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


def stderr_method_dev_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("method-dev progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"method-dev {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
