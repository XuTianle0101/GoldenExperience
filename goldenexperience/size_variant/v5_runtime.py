"""Publication runtime audit and final v5 artifact approval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from goldenexperience.benchmarks.publication import (
    SPLIT_COUNTS,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.benchmarks.selective_runtime import (
    SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
    SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
    SELECTIVE_RUNTIME_REQUEST_ORDER,
    build_selective_runtime_report,
    runtime_cost_evidence_from_report,
)
from goldenexperience.runtime.lmcache_retrieve_transform import (
    RuntimeSourceIdentity,
    RuntimeStackIdentity,
    probe_runtime_stack,
    verify_runtime_stack_identity,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file
from goldenexperience.size_variant.risk_gate import RISK_FEATURE_DIM, RiskPredictor, unsafe_label
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    DirectInjectionEvidence,
    RuntimeCostEvidence,
    SelectiveKVBridgeManifest,
)
from goldenexperience.size_variant.v5_calibration import (
    RISK_CALIBRATION_PREDICTOR_DEVICE,
    V5RiskCalibrationManifest,
    load_completed_risk_calibration,
)
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceRecord,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
)
from goldenexperience.size_variant.v5_directional_fit import (
    V5DirectionalTransportFitManifest,
)
from goldenexperience.size_variant.v5_fit import (
    TransportCandidateArtifact,
    V5TransportFitManifest,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskHistory,
    V5RiskFitManifest,
    load_risk_predictor,
)
from goldenexperience.size_variant.v5_semantic import (
    V5SemanticManifest,
    load_completed_semantic,
)
from goldenexperience.size_variant.v5_validation import VALIDATION_DECISIONS, _admission_decision

V5_RUNTIME_CHECKPOINT_SCHEMA = "goldenexperience.v5_runtime_checkpoint.v2"
V5_RUNTIME_REPORT_SCHEMA = "goldenexperience.v5_runtime_report.v2"
V5_RUNTIME_MANIFEST_SCHEMA = "goldenexperience.v5_runtime_manifest.v2"
RUNTIME_AUDIT_WARMUP_ITERATIONS = 20
RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH = 100
RUNTIME_AUDIT_PREDICTOR_DEVICE = RISK_CALIBRATION_PREDICTOR_DEVICE
RUNTIME_AUDIT_SHADOW_POLICY = "reference_free_greedy_agreement_lt_0.98_or_perplexity_drift_gt_2pct"


@dataclass(frozen=True)
class RuntimeRiskObservation:
    """Reference-free shadow outcome used only for causal runtime history."""

    sample_id: str
    prefix_group_id: str
    features: tuple[float, ...]
    shadow_failure: bool
    greedy_matches: int
    greedy_tokens: int
    native_nll: float
    bridge_nll: float
    teacher_tokens: int
    history_samples: int
    history_failures: int
    history_greedy_agreement: float
    sidecar_sha256: str
    native_tokens_sha256: str
    bridge_tokens_sha256: str

    @property
    def greedy_agreement(self) -> float:
        if type(self.greedy_matches) is not int or type(self.greedy_tokens) is not int:
            return math.nan
        if self.greedy_tokens <= 0:
            return math.nan
        return self.greedy_matches / self.greedy_tokens

    @property
    def perplexity_drift_pct(self) -> float:
        if (
            type(self.teacher_tokens) is not int
            or self.teacher_tokens <= 0
            or not _finite_nonnegative(self.native_nll)
            or not _finite_nonnegative(self.bridge_nll)
        ):
            return math.inf
        try:
            drift = abs(math.expm1((self.bridge_nll - self.native_nll) / self.teacher_tokens)) * 100
        except OverflowError:
            return sys.float_info.max
        if not math.isfinite(drift):
            return math.inf if math.isnan(drift) else sys.float_info.max
        return min(sys.float_info.max, drift)

    def history(self) -> RiskHistory:
        agreement_sum = (
            self.history_greedy_agreement * self.history_samples
            if type(self.history_samples) is int
            and _finite_probability(self.history_greedy_agreement)
            else math.nan
        )
        return RiskHistory(
            samples=self.history_samples,
            failures=self.history_failures,
            greedy_agreement_sum=agreement_sum,
        )

    def update_history(self, history: RiskHistory) -> RiskHistory:
        return RiskHistory(
            samples=history.samples + 1,
            failures=history.failures + int(self.shadow_failure),
            greedy_agreement_sum=history.greedy_agreement_sum + self.greedy_agreement,
        )

    def validate(
        self,
        *,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        expected_history: RiskHistory,
    ) -> list[str]:
        errors: list[str] = []
        if (
            self.sample_id != benchmark_record.sample_id
            or self.sample_id != trace_record.sample_id
            or self.prefix_group_id != benchmark_record.prefix_group_id
        ):
            errors.append("runtime shadow observation binding changed")
        if (
            not isinstance(self.features, tuple)
            or len(self.features) != RISK_FEATURE_DIM
            or any(not _finite_number(item) for item in self.features)
        ):
            errors.append("runtime shadow feature vector is invalid")
        if not _strict_bool(self.shadow_failure):
            errors.append("runtime shadow failure marker must be boolean")
        if (
            type(self.greedy_matches) is not int
            or type(self.greedy_tokens) is not int
            or self.greedy_tokens != RISK_LABEL_GENERATION_TOKENS
            or not 0 <= self.greedy_matches <= self.greedy_tokens
        ):
            errors.append("runtime shadow greedy counts are invalid")
        if (
            type(self.teacher_tokens) is not int
            or self.teacher_tokens != RISK_LABEL_GENERATION_TOKENS
            or not _finite_nonnegative(self.native_nll)
            or not _finite_nonnegative(self.bridge_nll)
            or not math.isfinite(self.perplexity_drift_pct)
        ):
            errors.append("runtime shadow NLL evidence is invalid")
        observed_history = self.history()
        if (
            observed_history.validate()
            or expected_history.validate()
            or (
                observed_history.samples != expected_history.samples
                or observed_history.failures != expected_history.failures
                or abs(observed_history.greedy_agreement - expected_history.greedy_agreement)
                > 1e-12
            )
        ):
            errors.append("runtime shadow observation uses non-causal history")
        for name in ("sidecar_sha256", "native_tokens_sha256", "bridge_tokens_sha256"):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"runtime shadow {name} is invalid")
        if _strict_bool(self.shadow_failure) and self.shadow_failure != unsafe_label(
            native_task_passed=False,
            bridge_task_passed=False,
            greedy_agreement=self.greedy_agreement,
            perplexity_drift_pct=self.perplexity_drift_pct,
        ):
            errors.append("runtime reference-free shadow failure is inconsistent")
        return errors


@dataclass(frozen=True)
class RuntimeExecutionMeasurement:
    native_prefill_ms: float
    native_ttft_ms: float
    observed_ttft_ms: float
    materialization_ms: float | None
    retrieve_transform_success: bool
    load_complete_published: bool
    source_read_attempted: bool
    source_chunks_read: int
    tokens_scattered: int
    fallback_reason: str
    target_mooncake_puts: int
    backing_files_remaining: int

    def validate(
        self,
        *,
        accepted: bool,
        decision: str,
        expected_tokens: int,
    ) -> list[str]:
        errors: list[str] = []
        for name in ("native_prefill_ms", "native_ttft_ms", "observed_ttft_ms"):
            if not _finite_positive(getattr(self, name)):
                errors.append(f"runtime {name} must be finite and positive")
        if not _strict_bool(self.retrieve_transform_success):
            errors.append("runtime retrieve success marker must be boolean")
        if not _strict_bool(self.load_complete_published):
            errors.append("runtime load-complete marker must be boolean")
        if not _strict_bool(self.source_read_attempted):
            errors.append("runtime source-read marker must be boolean")
        for name in (
            "source_chunks_read",
            "tokens_scattered",
            "target_mooncake_puts",
            "backing_files_remaining",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                errors.append(f"runtime {name} must be a non-negative integer")
        if not isinstance(self.fallback_reason, str) or not self.fallback_reason:
            errors.append("runtime fallback reason is required")
        if accepted:
            if not _finite_positive(self.materialization_ms):
                errors.append("accepted runtime materialization latency is invalid")
            if not self.retrieve_transform_success or not self.load_complete_published:
                errors.append("accepted runtime request did not complete direct injection")
            if not self.source_read_attempted or self.source_chunks_read <= 0:
                errors.append("accepted runtime request did not read source chunks")
            if self.tokens_scattered != expected_tokens:
                errors.append("accepted runtime request scattered another token count")
            if self.fallback_reason != "none":
                errors.append("accepted runtime request carries a fallback reason")
        else:
            if self.materialization_ms is not None:
                errors.append("rejected runtime request cannot carry materialization latency")
            if self.retrieve_transform_success or self.load_complete_published:
                errors.append("rejected runtime request entered direct injection")
            if self.source_read_attempted or self.source_chunks_read != 0:
                errors.append("rejected runtime request read source KV")
            if self.tokens_scattered != 0:
                errors.append("rejected runtime request scattered target KV")
            if self.fallback_reason != decision:
                errors.append("rejected runtime fallback differs from the gate decision")
        if self.target_mooncake_puts != 0:
            errors.append("runtime request wrote translated target Mooncake objects")
        if self.backing_files_remaining != 0:
            errors.append("runtime request left filesystem backing artifacts")
        return errors


@dataclass(frozen=True)
class RuntimeAuditMeasurement:
    observation: RuntimeRiskObservation
    unsafe_probability: float
    accepted: bool
    decision: str
    execution: RuntimeExecutionMeasurement

    def validate(
        self,
        *,
        predictor: RiskPredictor,
        risk_gate: Any,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        expected_history: RiskHistory,
    ) -> list[str]:
        errors = self.observation.validate(
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=expected_history,
        )
        if not _finite_probability(self.unsafe_probability):
            errors.append("runtime predictor probability is invalid")
        else:
            try:
                expected_probability = predictor.unsafe_probability(self.observation.features)
            except (RuntimeError, TypeError, ValueError) as exc:
                errors.append(f"runtime predictor failed: {type(exc).__name__}")
            else:
                if abs(self.unsafe_probability - expected_probability) > 1e-12:
                    errors.append("runtime probability differs from the frozen predictor")
        if not _strict_bool(self.accepted):
            errors.append("runtime admission marker must be boolean")
        if self.decision not in VALIDATION_DECISIONS:
            errors.append("runtime admission decision is invalid")
        else:
            try:
                expected_accepted, expected_decision = _admission_decision(
                    self.observation,
                    self.unsafe_probability,
                    risk_gate,
                )
            except (TypeError, ValueError, V5PipelineError) as exc:
                errors.append(f"runtime admission contract is malformed: {type(exc).__name__}")
            else:
                if self.accepted != expected_accepted or self.decision != expected_decision:
                    errors.append("runtime admission differs from the calibrated gate")
        errors.extend(
            self.execution.validate(
                accepted=self.accepted,
                decision=self.decision,
                expected_tokens=trace_record.token_count,
            )
        )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeAuditMeasurement:
        observation = dict(payload["observation"])
        observation["features"] = tuple(observation["features"])
        return cls(
            observation=RuntimeRiskObservation(**observation),
            unsafe_probability=payload["unsafe_probability"],
            accepted=payload["accepted"],
            decision=str(payload["decision"]),
            execution=RuntimeExecutionMeasurement(**payload["execution"]),
        )


@dataclass(frozen=True)
class RuntimeFailureAudit:
    probe_id: str
    paged_slot_mapping_verified: bool
    load_complete_after_all_layers: bool
    partial_failure_invalidates_blocks: bool
    native_prefill_overwrites_invalid_blocks: bool
    injected_failure_count: int
    invalidated_block_count: int
    recomputed_token_count: int
    accepted_target_mooncake_puts: int
    backing_files_remaining: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not _is_sha256(self.probe_id):
            errors.append("runtime failure probe identity is invalid")
        for name in (
            "paged_slot_mapping_verified",
            "load_complete_after_all_layers",
            "partial_failure_invalidates_blocks",
            "native_prefill_overwrites_invalid_blocks",
        ):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                errors.append(f"runtime failure audit {name} is not verified")
        for name in (
            "injected_failure_count",
            "invalidated_block_count",
            "recomputed_token_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                errors.append(f"runtime failure audit {name} must be positive")
        if type(self.accepted_target_mooncake_puts) is not int or (
            self.accepted_target_mooncake_puts != 0
        ):
            errors.append("runtime failure audit observed target Mooncake puts")
        if type(self.backing_files_remaining) is not int or self.backing_files_remaining != 0:
            errors.append("runtime failure audit left backing files")
        return errors

    def direct_injection_evidence(self, *, report_sha256: str) -> DirectInjectionEvidence:
        return DirectInjectionEvidence(
            report_sha256=report_sha256,
            paged_slot_mapping_verified=self.paged_slot_mapping_verified,
            load_complete_after_all_layers=self.load_complete_after_all_layers,
            partial_failure_invalidates_blocks=self.partial_failure_invalidates_blocks,
            native_prefill_overwrites_invalid_blocks=(
                self.native_prefill_overwrites_invalid_blocks
            ),
            accepted_target_mooncake_puts=self.accepted_target_mooncake_puts,
            backing_files_remaining=self.backing_files_remaining,
            runtime_audit_passed=True,
        )


class RuntimeAuditEvaluator(Protocol):
    def __enter__(self) -> RuntimeAuditEvaluator: ...

    def __exit__(self, *_args: object) -> None: ...

    def warmup(self, iterations: int) -> None: ...

    def build_observation(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RuntimeRiskObservation: ...

    def measure(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord,
        sample: RawBenchmarkSample,
        observation: RuntimeRiskObservation,
        *,
        accepted: bool,
        decision: str,
    ) -> RuntimeExecutionMeasurement: ...

    def audit_failure_recovery(self) -> RuntimeFailureAudit: ...


@dataclass(frozen=True)
class V5RuntimeManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    benchmark_manifest_sha256: str
    runtime_audit_split_sha256: str
    runtime_trace_manifest_sha256: str
    runtime_raw_store_sha256: str
    semantic_manifest_sha256: str
    semantic_report_sha256: str
    semantic_selective_artifact_id: str
    risk_fit_manifest_sha256: str
    risk_calibration_manifest_sha256: str
    transport_weights_sha256: str
    predictor_sha256: str
    threshold: float
    runtime_stack: RuntimeStackIdentity
    runtime_report_sha256: str
    evaluator_sha256: str
    approved_artifact_id: str
    approved_manifest_file_sha256: str
    runtime_cost: RuntimeCostEvidence
    direct_injection: DirectInjectionEvidence
    failure_audit: RuntimeFailureAudit
    sample_count: int
    accepted_count: int
    rejected_count: int
    warmup_iterations: int
    measurement_protocol: str
    request_order: str
    arrival_timestamps_replayed: bool
    shadow_policy: str
    passed: bool = True
    schema_version: str = V5_RUNTIME_MANIFEST_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        semantic: V5SemanticManifest,
        semantic_selective: SelectiveKVBridgeManifest,
        risk_fit: V5RiskFitManifest,
        calibration: V5RiskCalibrationManifest,
        candidate: TransportCandidateArtifact,
        approved: SelectiveKVBridgeManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_RUNTIME_MANIFEST_SCHEMA:
            errors.append("unsupported v5 runtime manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("runtime manifest belongs to another pipeline")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        expected = {
            "code_sha256": workspace.config.code_sha256,
            "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
            "runtime_audit_split_sha256": workspace.config.split_sha256["runtime_audit"],
            "runtime_trace_manifest_sha256": trace.content_sha256(),
            "runtime_raw_store_sha256": trace.raw_sample_store_sha256,
            "semantic_manifest_sha256": semantic.content_sha256(),
            "semantic_report_sha256": semantic.semantic_report_sha256,
            "semantic_selective_artifact_id": semantic_selective.artifact_id,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "risk_calibration_manifest_sha256": calibration.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "predictor_sha256": risk_fit.predictor.sha256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"runtime manifest {name} changed")
        for name in (
            "code_sha256",
            "benchmark_manifest_sha256",
            "runtime_audit_split_sha256",
            "runtime_trace_manifest_sha256",
            "runtime_raw_store_sha256",
            "semantic_manifest_sha256",
            "semantic_report_sha256",
            "risk_fit_manifest_sha256",
            "risk_calibration_manifest_sha256",
            "transport_weights_sha256",
            "predictor_sha256",
            "runtime_report_sha256",
            "evaluator_sha256",
            "approved_manifest_file_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"runtime manifest {name} is invalid")
        if self.direction != trace.direction or self.direction != semantic.direction:
            errors.append("runtime manifest direction is inconsistent")
        if calibration.risk_gate.threshold is None or self.threshold != (
            calibration.risk_gate.threshold
        ):
            errors.append("runtime calibrated threshold changed")
        errors.extend(self.runtime_stack.validate())
        try:
            errors.extend(
                self.runtime_cost.validate(workspace.config.split_sha256["runtime_audit"])
            )
            errors.extend(self.direct_injection.validate())
            errors.extend(self.failure_audit.validate())
        except (TypeError, ValueError) as exc:
            errors.append(f"runtime evidence is malformed: {type(exc).__name__}")
        if self.runtime_cost.report_sha256 != self.runtime_report_sha256:
            errors.append("runtime cost evidence refers to another report")
        if self.direct_injection.report_sha256 != self.runtime_report_sha256:
            errors.append("direct injection evidence refers to another report")
        if self.direct_injection != self.failure_audit.direct_injection_evidence(
            report_sha256=self.runtime_report_sha256
        ):
            errors.append("direct injection evidence differs from the failure audit")
        if (
            type(self.sample_count) is not int
            or self.sample_count != SPLIT_COUNTS["runtime_audit"]
            or type(self.accepted_count) is not int
            or type(self.rejected_count) is not int
            or self.accepted_count + self.rejected_count != self.sample_count
        ):
            errors.append("runtime request counts are inconsistent")
        if (
            self.accepted_count < RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH
            or self.rejected_count < RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH
        ):
            errors.append("runtime audit requires 100 accepted and rejected measurements")
        if self.warmup_iterations != RUNTIME_AUDIT_WARMUP_ITERATIONS:
            errors.append("runtime audit warmup count changed")
        if self.measurement_protocol != SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL:
            errors.append("runtime audit measurement protocol changed")
        if self.request_order != SELECTIVE_RUNTIME_REQUEST_ORDER:
            errors.append("runtime audit request order changed")
        if (
            type(self.arrival_timestamps_replayed) is not bool
            or self.arrival_timestamps_replayed is not SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED
        ):
            errors.append("isolated runtime audit cannot claim arrival-timestamp replay")
        if self.shadow_policy != RUNTIME_AUDIT_SHADOW_POLICY:
            errors.append("runtime reference-free shadow policy changed")
        if type(self.passed) is not bool or not self.passed:
            errors.append("runtime manifest does not carry a passing result")
        if semantic_selective.state is not ArtifactState.SEMANTIC_APPROVED:
            errors.append("runtime input is not semantic_approved")
        if self.approved_artifact_id != approved.artifact_id:
            errors.append("approved artifact id changed")
        try:
            expected_approved = build_approved_candidate(
                semantic_selective=semantic_selective,
                runtime_cost=self.runtime_cost,
                direct_injection=self.direct_injection,
            )
        except (TypeError, ValueError, V5PipelineError) as exc:
            errors.append(f"approved artifact cannot be rebuilt: {type(exc).__name__}")
        else:
            if approved != expected_approved:
                errors.append("approved artifact differs from runtime evidence")
        try:
            errors.extend(approved.validate())
        except (TypeError, ValueError) as exc:
            errors.append(f"approved artifact is malformed: {type(exc).__name__}")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5RuntimeManifest:
        stack = dict(payload["runtime_stack"])
        stack["sources"] = tuple(
            _runtime_source_from_dict(item) for item in stack.get("sources", ())
        )
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            runtime_audit_split_sha256=str(payload["runtime_audit_split_sha256"]),
            runtime_trace_manifest_sha256=str(payload["runtime_trace_manifest_sha256"]),
            runtime_raw_store_sha256=str(payload["runtime_raw_store_sha256"]),
            semantic_manifest_sha256=str(payload["semantic_manifest_sha256"]),
            semantic_report_sha256=str(payload["semantic_report_sha256"]),
            semantic_selective_artifact_id=str(payload["semantic_selective_artifact_id"]),
            risk_fit_manifest_sha256=str(payload["risk_fit_manifest_sha256"]),
            risk_calibration_manifest_sha256=str(payload["risk_calibration_manifest_sha256"]),
            transport_weights_sha256=str(payload["transport_weights_sha256"]),
            predictor_sha256=str(payload["predictor_sha256"]),
            threshold=payload["threshold"],
            runtime_stack=RuntimeStackIdentity(**stack),
            runtime_report_sha256=str(payload["runtime_report_sha256"]),
            evaluator_sha256=str(payload["evaluator_sha256"]),
            approved_artifact_id=str(payload["approved_artifact_id"]),
            approved_manifest_file_sha256=str(payload["approved_manifest_file_sha256"]),
            runtime_cost=RuntimeCostEvidence(**payload["runtime_cost"]),
            direct_injection=DirectInjectionEvidence(**payload["direct_injection"]),
            failure_audit=RuntimeFailureAudit(**payload["failure_audit"]),
            sample_count=payload["sample_count"],
            accepted_count=payload["accepted_count"],
            rejected_count=payload["rejected_count"],
            warmup_iterations=payload["warmup_iterations"],
            measurement_protocol=str(payload["measurement_protocol"]),
            request_order=str(payload["request_order"]),
            arrival_timestamps_replayed=payload["arrival_timestamps_replayed"],
            shadow_policy=str(payload["shadow_policy"]),
            passed=payload.get("passed", False),
            schema_version=str(payload.get("schema_version", "")),
        )


def run_runtime_audit_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], RuntimeAuditEvaluator],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Measure one semantic-approved direction and grant final authority on success."""

    benchmark = load_bound_benchmark(workspace)
    semantic, semantic_selective = load_completed_semantic(workspace, direction)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    trace = load_completed_trace_manifest(workspace, direction, "runtime_audit", benchmark)
    store_path = Path(sample_store_path)
    store_signature = _file_signature(store_path)
    store_sha256 = sha256_file(store_path)
    if store_sha256 != trace.raw_sample_store_sha256:
        raise V5PipelineError("runtime raw store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="runtime_audit")
    traces = {item.sample_id: item for item in trace.records}
    if set(traces) != {record.sample_id for record, _ in samples}:
        raise V5PipelineError("runtime samples differ from trace manifest")
    stack = probe_runtime_stack()
    evaluator_payload = dict(evaluator_parameters)
    evaluator_sha256 = _sha256_bytes(_canonical_json_bytes(evaluator_payload))
    stage_parameters = {
        "runtime_trace_manifest_sha256": trace.content_sha256(),
        "runtime_raw_store_sha256": store_sha256,
        "semantic_manifest_sha256": semantic.content_sha256(),
        "semantic_report_sha256": semantic.semantic_report_sha256,
        "semantic_selective_artifact_id": semantic_selective.artifact_id,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "risk_calibration_manifest_sha256": calibration.content_sha256(),
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "predictor_sha256": risk_fit.predictor.sha256,
        "threshold": calibration.risk_gate.threshold,
        "predictor_device": RUNTIME_AUDIT_PREDICTOR_DEVICE,
        "warmup_iterations": RUNTIME_AUDIT_WARMUP_ITERATIONS,
        "minimum_measurements_per_path": RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH,
        "measurement_protocol": SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
        "request_order": SELECTIVE_RUNTIME_REQUEST_ORDER,
        "arrival_timestamps_replayed": SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
        "shadow_policy": RUNTIME_AUDIT_SHADOW_POLICY,
        "runtime_stack_sha256": stack.content_sha256(),
        "evaluator": evaluator_payload,
    }
    lease = workspace.begin_stage(
        direction,
        "runtime_audit",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/runtime_audit"]
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=RUNTIME_AUDIT_PREDICTOR_DEVICE,
    )
    work = workspace.control / "work" / direction / "runtime_audit"
    checkpoint_dir = work / "examples"
    histories: dict[str, RiskHistory] = {}
    measurements: list[RuntimeAuditMeasurement] = []
    try:
        with evaluator_factory() as evaluator:
            evaluator.warmup(RUNTIME_AUDIT_WARMUP_ITERATIONS)
            for index, (record, sample) in enumerate(samples, start=1):
                trace_record = traces[record.sample_id]
                history = histories.get(record.prefix_group_id, RiskHistory())
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                measurement = _load_runtime_checkpoint(
                    checkpoint,
                    stage_binding_sha256=lease.input_sha256,
                    benchmark_record=record,
                    trace_record=trace_record,
                    expected_history=history,
                    predictor=predictor,
                    risk_gate=calibration.risk_gate,
                )
                if measurement is None:
                    observation = evaluator.build_observation(
                        record,
                        trace_record,
                        sample,
                        history,
                    )
                    probability = predictor.unsafe_probability(observation.features)
                    accepted, decision = _admission_decision(
                        observation,
                        probability,
                        calibration.risk_gate,
                    )
                    execution = evaluator.measure(
                        record,
                        trace_record,
                        sample,
                        observation,
                        accepted=accepted,
                        decision=decision,
                    )
                    measurement = RuntimeAuditMeasurement(
                        observation=observation,
                        unsafe_probability=probability,
                        accepted=accepted,
                        decision=decision,
                        execution=execution,
                    )
                    errors = measurement.validate(
                        predictor=predictor,
                        risk_gate=calibration.risk_gate,
                        benchmark_record=record,
                        trace_record=trace_record,
                        expected_history=history,
                    )
                    if errors:
                        raise V5PipelineError("; ".join(errors))
                    _write_runtime_checkpoint(
                        checkpoint,
                        stage_binding_sha256=lease.input_sha256,
                        benchmark_record=record,
                        measurement=measurement,
                    )
                histories[record.prefix_group_id] = measurement.observation.update_history(history)
                measurements.append(measurement)
                if progress is not None:
                    progress(index, len(samples), sample.sample_id)
            failure_audit = evaluator.audit_failure_recovery()
        if (
            _file_signature(store_path) != store_signature
            or sha256_file(store_path) != store_sha256
        ):
            raise V5PipelineError("runtime raw store changed during measurement")
        verify_runtime_stack_identity(stack)
        failure_errors = failure_audit.validate()
        if failure_errors:
            raise V5PipelineError("; ".join(failure_errors))
        summary = build_runtime_summary(
            direction=direction,
            dataset_sha256=workspace.config.split_sha256["runtime_audit"],
            measurements=measurements,
            failure_audit=failure_audit,
        )
        if not summary["eligible_for_approval"]:
            raise V5PipelineError("runtime latency or direct-injection gates failed")
        report = {
            **_report_header(
                workspace=workspace,
                direction=direction,
                trace=trace,
                semantic=semantic,
                semantic_selective=semantic_selective,
                risk_fit=risk_fit,
                calibration=calibration,
                transport_manifest=transport_manifest,
                candidate=candidate,
                stack=stack,
            ),
            "evaluator": evaluator_payload,
            "measurements": [item.to_dict() for item in measurements],
            "failure_audit": asdict(failure_audit),
            "runtime_summary": summary,
        }
        report_path = work / "runtime_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        runtime_cost = runtime_cost_evidence_from_report(
            summary,
            report_sha256=report_sha256,
        )
        direct_injection = failure_audit.direct_injection_evidence(report_sha256=report_sha256)
        approved = build_approved_candidate(
            semantic_selective=semantic_selective,
            runtime_cost=runtime_cost,
            direct_injection=direct_injection,
        )
        approved_errors = approved.validate()
        if approved_errors:
            raise V5PipelineError("; ".join(approved_errors))
        approved_path = work / "approved_selective_manifest.json"
        _write_json_replace(approved_path, approved.to_dict())
        approved_file_sha256 = sha256_file(approved_path)
        accepted_count = sum(int(item.accepted) for item in measurements)
        manifest = V5RuntimeManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
            runtime_audit_split_sha256=workspace.config.split_sha256["runtime_audit"],
            runtime_trace_manifest_sha256=trace.content_sha256(),
            runtime_raw_store_sha256=trace.raw_sample_store_sha256,
            semantic_manifest_sha256=semantic.content_sha256(),
            semantic_report_sha256=semantic.semantic_report_sha256,
            semantic_selective_artifact_id=semantic_selective.artifact_id,
            risk_fit_manifest_sha256=risk_fit.content_sha256(),
            risk_calibration_manifest_sha256=calibration.content_sha256(),
            transport_weights_sha256=candidate.weights.sha256,
            predictor_sha256=risk_fit.predictor.sha256,
            threshold=_required_threshold(calibration),
            runtime_stack=stack,
            runtime_report_sha256=report_sha256,
            evaluator_sha256=evaluator_sha256,
            approved_artifact_id=approved.artifact_id,
            approved_manifest_file_sha256=approved_file_sha256,
            runtime_cost=runtime_cost,
            direct_injection=direct_injection,
            failure_audit=failure_audit,
            sample_count=len(measurements),
            accepted_count=accepted_count,
            rejected_count=len(measurements) - accepted_count,
            warmup_iterations=RUNTIME_AUDIT_WARMUP_ITERATIONS,
            measurement_protocol=SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
            request_order=SELECTIVE_RUNTIME_REQUEST_ORDER,
            arrival_timestamps_replayed=SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
            shadow_policy=RUNTIME_AUDIT_SHADOW_POLICY,
        )
        errors = manifest.validate(
            workspace=workspace,
            trace=trace,
            semantic=semantic,
            semantic_selective=semantic_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            candidate=candidate,
            approved=approved,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        manifest_path = work / "runtime_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "runtime_report": report_path,
                "runtime_manifest": manifest_path,
                "approved_selective_manifest": approved_path,
            },
            metadata={
                "sample_count": len(measurements),
                "accepted_count": accepted_count,
                "rejected_count": len(measurements) - accepted_count,
                "runtime_stack_sha256": stack.content_sha256(),
                "runtime_manifest_sha256": manifest.content_sha256(),
                "approved_artifact_id": approved.artifact_id,
                "passed": True,
                "authority": ArtifactState.APPROVED.value,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def build_runtime_summary(
    *,
    direction: str,
    dataset_sha256: str,
    measurements: Sequence[RuntimeAuditMeasurement],
    failure_audit: RuntimeFailureAudit,
) -> dict[str, Any]:
    if len(measurements) != SPLIT_COUNTS["runtime_audit"]:
        raise V5PipelineError("runtime summary requires the complete registered split")
    accepted = [item for item in measurements if item.accepted]
    rejected = [item for item in measurements if not item.accepted]
    if (
        len(accepted) < RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH
        or len(rejected) < RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH
    ):
        raise V5PipelineError("runtime summary requires 100 accepted and rejected requests")
    failure_errors = failure_audit.validate()
    if failure_errors:
        raise V5PipelineError("; ".join(failure_errors))
    return build_selective_runtime_report(
        direction=direction,
        runtime_audit_dataset_sha256=dataset_sha256,
        audit_requests=len(measurements),
        warmup_iterations=RUNTIME_AUDIT_WARMUP_ITERATIONS,
        materialization_ms=[_required_materialization(item) for item in accepted],
        native_prefill_ms=[item.execution.native_prefill_ms for item in accepted],
        accepted_native_ttft_ms=[item.execution.native_ttft_ms for item in accepted],
        accepted_reuse_ttft_ms=[item.execution.observed_ttft_ms for item in accepted],
        rejected_native_ttft_ms=[item.execution.native_ttft_ms for item in rejected],
        rejected_fallback_ttft_ms=[item.execution.observed_ttft_ms for item in rejected],
        accepted_target_mooncake_puts=sum(item.execution.target_mooncake_puts for item in accepted),
        backing_files_remaining=max(
            [failure_audit.backing_files_remaining]
            + [item.execution.backing_files_remaining for item in measurements]
        ),
    )


def build_approved_candidate(
    *,
    semantic_selective: SelectiveKVBridgeManifest,
    runtime_cost: RuntimeCostEvidence,
    direct_injection: DirectInjectionEvidence,
) -> SelectiveKVBridgeManifest:
    if semantic_selective.state is not ArtifactState.SEMANTIC_APPROVED:
        raise V5PipelineError("runtime audit requires a semantic_approved artifact")
    if (
        semantic_selective.runtime_cost is not None
        or semantic_selective.direct_injection is not None
    ):
        raise V5PipelineError("semantic artifact carries premature runtime evidence")
    return replace(
        semantic_selective,
        artifact_id="",
        state=ArtifactState.APPROVED,
        runtime_cost=runtime_cost,
        direct_injection=direct_injection,
    ).with_content_id()


def load_completed_runtime_audit(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[V5RuntimeManifest, SelectiveKVBridgeManifest]:
    benchmark = load_bound_benchmark(workspace)
    semantic, semantic_selective = load_completed_semantic(workspace, direction)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    trace = load_completed_trace_manifest(workspace, direction, "runtime_audit", benchmark)
    state = workspace.state()
    stage = state.stages.get(f"{direction}/runtime_audit")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed passing runtime audit")
    report_artifact = stage.outputs.get("runtime_report")
    manifest_artifact = stage.outputs.get("runtime_manifest")
    approved_artifact = stage.outputs.get("approved_selective_manifest")
    if report_artifact is None or manifest_artifact is None or approved_artifact is None:
        raise V5PipelineError("runtime stage lacks required evidence artifacts")
    report_path = workspace.artifact_path(report_artifact, verify_hash=True)
    manifest_path = workspace.artifact_path(manifest_artifact, verify_hash=True)
    approved_path = workspace.artifact_path(approved_artifact, verify_hash=True)
    try:
        manifest = V5RuntimeManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        approved = SelectiveKVBridgeManifest.from_dict(
            json.loads(approved_path.read_text(encoding="utf-8"))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("runtime manifest is unreadable or malformed") from exc
    errors = manifest.validate(
        workspace=workspace,
        trace=trace,
        semantic=semantic,
        semantic_selective=semantic_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        candidate=candidate,
        approved=approved,
    )
    if errors:
        raise V5PipelineError("; ".join(errors))
    if manifest.runtime_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("runtime manifest refers to another report")
    if manifest.approved_manifest_file_sha256 != approved_artifact.sha256:
        raise V5PipelineError("runtime manifest refers to another approved artifact")
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=RUNTIME_AUDIT_PREDICTOR_DEVICE,
    )
    _load_and_validate_runtime_report(
        report_path,
        benchmark=benchmark,
        workspace=workspace,
        trace=trace,
        semantic=semantic,
        semantic_selective=semantic_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        transport_manifest=transport_manifest,
        candidate=candidate,
        manifest=manifest,
        predictor=predictor,
    )
    return manifest, approved


def _load_and_validate_runtime_report(
    path: Path,
    *,
    benchmark: PublicationBenchmarkManifest,
    workspace: V5PipelineWorkspace,
    trace: V5TraceManifest,
    semantic: V5SemanticManifest,
    semantic_selective: SelectiveKVBridgeManifest,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    manifest: V5RuntimeManifest,
    predictor: RiskPredictor,
) -> tuple[RuntimeAuditMeasurement, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_header = _report_header(
            workspace=workspace,
            direction=manifest.direction,
            trace=trace,
            semantic=semantic,
            semantic_selective=semantic_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport_manifest,
            candidate=candidate,
            stack=manifest.runtime_stack,
        )
        if any(payload.get(name) != value for name, value in expected_header.items()):
            raise V5PipelineError("runtime report header binding changed")
        evaluator = payload["evaluator"]
        if not isinstance(evaluator, dict) or (
            _sha256_bytes(_canonical_json_bytes(evaluator)) != manifest.evaluator_sha256
        ):
            raise V5PipelineError("runtime evaluator metadata changed")
        measurements = tuple(
            RuntimeAuditMeasurement.from_dict(item) for item in payload["measurements"]
        )
        failure_audit = RuntimeFailureAudit(**payload["failure_audit"])
        reported_summary = dict(payload["runtime_summary"])
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("runtime report is unreadable or malformed") from exc
    expected_records = sorted(
        (item for item in benchmark.records if item.split == "runtime_audit"),
        key=lambda item: item.sample_id,
    )
    trace_by_id = {item.sample_id: item for item in trace.records}
    if len(measurements) != SPLIT_COUNTS["runtime_audit"] or len(measurements) != len(
        expected_records
    ):
        raise V5PipelineError("runtime report measurement count is inconsistent")
    accepted_count = sum(int(item.accepted) for item in measurements)
    if (
        manifest.sample_count != len(measurements)
        or manifest.accepted_count != accepted_count
        or manifest.rejected_count != len(measurements) - accepted_count
    ):
        raise V5PipelineError("runtime manifest counts differ from the detailed report")
    histories: dict[str, RiskHistory] = {}
    for measurement, record in zip(measurements, expected_records, strict=True):
        trace_record = trace_by_id.get(record.sample_id)
        if trace_record is None:
            raise V5PipelineError("runtime report refers to a missing trace")
        history = histories.get(record.prefix_group_id, RiskHistory())
        errors = measurement.validate(
            predictor=predictor,
            risk_gate=calibration.risk_gate,
            benchmark_record=record,
            trace_record=trace_record,
            expected_history=history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        histories[record.prefix_group_id] = measurement.observation.update_history(history)
    expected_summary = build_runtime_summary(
        direction=manifest.direction,
        dataset_sha256=workspace.config.split_sha256["runtime_audit"],
        measurements=measurements,
        failure_audit=failure_audit,
    )
    expected_cost = runtime_cost_evidence_from_report(
        expected_summary,
        report_sha256=manifest.runtime_report_sha256,
    )
    expected_direct = failure_audit.direct_injection_evidence(
        report_sha256=manifest.runtime_report_sha256
    )
    if (
        reported_summary != expected_summary
        or expected_cost != manifest.runtime_cost
        or expected_direct != manifest.direct_injection
        or failure_audit != manifest.failure_audit
    ):
        raise V5PipelineError("runtime aggregates differ from the detailed report")
    verify_runtime_stack_identity(manifest.runtime_stack)
    return measurements


def _report_header(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    trace: V5TraceManifest,
    semantic: V5SemanticManifest,
    semantic_selective: SelectiveKVBridgeManifest,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    stack: RuntimeStackIdentity,
) -> dict[str, Any]:
    return {
        "schema_version": V5_RUNTIME_REPORT_SCHEMA,
        "pipeline_id": workspace.config.pipeline_id,
        "direction": direction,
        "code_sha256": workspace.config.code_sha256,
        "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
        "runtime_audit_split_sha256": workspace.config.split_sha256["runtime_audit"],
        "runtime_trace_manifest_sha256": trace.content_sha256(),
        "runtime_raw_store_sha256": trace.raw_sample_store_sha256,
        "semantic_manifest_sha256": semantic.content_sha256(),
        "semantic_report_sha256": semantic.semantic_report_sha256,
        "semantic_selective_artifact_id": semantic_selective.artifact_id,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "risk_calibration_manifest_sha256": calibration.content_sha256(),
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "predictor_sha256": risk_fit.predictor.sha256,
        "threshold": calibration.risk_gate.threshold,
        "predictor_device": RUNTIME_AUDIT_PREDICTOR_DEVICE,
        "warmup_iterations": RUNTIME_AUDIT_WARMUP_ITERATIONS,
        "minimum_measurements_per_path": RUNTIME_AUDIT_MIN_MEASUREMENTS_PER_PATH,
        "measurement_protocol": SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
        "request_order": SELECTIVE_RUNTIME_REQUEST_ORDER,
        "arrival_timestamps_replayed": SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
        "shadow_policy": RUNTIME_AUDIT_SHADOW_POLICY,
        "runtime_stack": {
            **asdict(stack),
            "sources": [asdict(item) for item in stack.sources],
        },
        "runtime_stack_sha256": stack.content_sha256(),
        "passed": True,
        "authority": ArtifactState.APPROVED.value,
    }


def _write_runtime_checkpoint(
    path: Path,
    *,
    stage_binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    measurement: RuntimeAuditMeasurement,
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_RUNTIME_CHECKPOINT_SCHEMA,
            "stage_binding_sha256": stage_binding_sha256,
            "sample_id": benchmark_record.sample_id,
            "content_sha256": benchmark_record.content_sha256,
            "measurement": measurement.to_dict(),
        },
    )


def _load_runtime_checkpoint(
    path: Path,
    *,
    stage_binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    trace_record: TraceRecord,
    expected_history: RiskHistory,
    predictor: RiskPredictor,
    risk_gate: Any,
) -> RuntimeAuditMeasurement | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("runtime checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_RUNTIME_CHECKPOINT_SCHEMA:
            raise V5PipelineError("runtime checkpoint schema mismatch")
        if payload.get("stage_binding_sha256") != stage_binding_sha256:
            raise V5PipelineError("runtime checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != benchmark_record.sample_id
            or payload.get("content_sha256") != benchmark_record.content_sha256
        ):
            raise V5PipelineError("runtime checkpoint sample binding mismatch")
        measurement = RuntimeAuditMeasurement.from_dict(payload["measurement"])
        errors = measurement.validate(
            predictor=predictor,
            risk_gate=risk_gate,
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=expected_history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        return measurement
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("runtime checkpoint is malformed") from exc


def _required_materialization(measurement: RuntimeAuditMeasurement) -> float:
    value = measurement.execution.materialization_ms
    if not _finite_positive(value):
        raise V5PipelineError("accepted runtime measurement lacks materialization latency")
    assert value is not None
    return float(value)


def _required_threshold(calibration: V5RiskCalibrationManifest) -> float:
    threshold = calibration.risk_gate.threshold
    if not _finite_probability(threshold):
        raise V5PipelineError("runtime audit requires a calibrated threshold")
    assert threshold is not None
    return float(threshold)


def _runtime_source_from_dict(payload: Mapping[str, Any]) -> RuntimeSourceIdentity:
    return RuntimeSourceIdentity(**payload)


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
        raise V5PipelineError("runtime metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("runtime input file is unavailable") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def _finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def _strict_bool(value: Any) -> bool:
    return type(value) is bool


def stderr_runtime_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("runtime progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"runtime {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
