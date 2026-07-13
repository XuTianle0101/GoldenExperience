"""Four-direction validation for calibrated selective KV v5 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    SPLIT_COUNTS,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.size_variant.cached_kv_manifest import sha256_file
from goldenexperience.size_variant.risk_gate import (
    RISK_FEATURE_DIM,
    RISK_FEATURE_OOD_INDEX,
    SELECTOR_COSINE_THRESHOLD,
    RiskPredictor,
    SelectorEvaluation,
    SelectorEvaluationExample,
    clopper_pearson_upper_bound,
    evaluate_selector_baselines,
)
from goldenexperience.size_variant.selective_manifest import (
    AcceptedSubsetQualityEvidence,
    ArtifactState,
    SelectiveKVBridgeManifest,
    SelectiveQualityThresholds,
    TransportSpec,
)
from goldenexperience.size_variant.v5_calibration import (
    RISK_CALIBRATION_PREDICTOR_DEVICE,
    RiskCalibrationMeasurement,
    V5RiskCalibrationManifest,
    load_completed_risk_calibration,
)
from goldenexperience.size_variant.v5_collect import (
    TraceRecord,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
)
from goldenexperience.size_variant.v5_directional_fit import V5DirectionalTransportFitManifest
from goldenexperience.size_variant.v5_fit import (
    TransportCandidateArtifact,
    V5TransportFitManifest,
)
from goldenexperience.size_variant.v5_method_dev import (
    FrozenTransportStructure,
    load_frozen_transport_structure,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskExampleEvaluator,
    RiskHistory,
    RiskPrefixTokenBinding,
    RiskTrainingExample,
    V5RiskFitManifest,
    load_risk_predictor,
)

V5_VALIDATION_CHECKPOINT_SCHEMA = "goldenexperience.v5_validation_checkpoint.v1"
V5_VALIDATION_REPORT_SCHEMA = "goldenexperience.v5_validation_report.v1"
V5_VALIDATION_MANIFEST_SCHEMA = "goldenexperience.v5_validation_manifest.v1"
VALIDATION_PREDICTOR_DEVICE = RISK_CALIBRATION_PREDICTOR_DEVICE
VALIDATION_DECISIONS = frozenset(
    {
        "accepted",
        "unseen_or_insufficient_shadow_history",
        "out_of_distribution",
        "predicted_unsafe",
    }
)


@dataclass(frozen=True)
class RiskValidationMeasurement:
    example: RiskTrainingExample
    unsafe_probability: float
    accepted: bool
    decision: str

    @property
    def runtime_eligible(self) -> bool:
        return self.decision not in {
            "unseen_or_insufficient_shadow_history",
            "out_of_distribution",
        }

    def validate(
        self,
        *,
        predictor: RiskPredictor,
        risk_gate: Any,
        benchmark_record: GroupedPrefixRecord | None = None,
        trace_record: TraceRecord | RiskPrefixTokenBinding | None = None,
        expected_history: RiskHistory | None = None,
    ) -> list[str]:
        calibration = RiskCalibrationMeasurement(self.example, self.unsafe_probability)
        errors = calibration.validate(
            predictor=predictor,
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=expected_history,
        )
        if not _strict_bool(self.accepted):
            errors.append("validation admission marker must be boolean")
        if self.decision not in VALIDATION_DECISIONS:
            errors.append("validation admission decision is invalid")
            return errors
        try:
            expected_accepted, expected_decision = _admission_decision(
                self.example,
                self.unsafe_probability,
                risk_gate,
            )
        except (TypeError, ValueError, V5PipelineError) as exc:
            errors.append(f"validation admission contract is malformed: {type(exc).__name__}")
        else:
            if self.accepted != expected_accepted or self.decision != expected_decision:
                errors.append("validation admission differs from the calibrated runtime gate")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskValidationMeasurement:
        raw_example = dict(payload["example"])
        raw_example["features"] = tuple(raw_example["features"])
        return cls(
            example=RiskTrainingExample(**raw_example),
            unsafe_probability=payload["unsafe_probability"],
            accepted=payload["accepted"],
            decision=str(payload["decision"]),
        )


@dataclass(frozen=True)
class V5ValidationManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    validation_split_sha256: str
    validation_trace_manifest_sha256: str
    validation_raw_store_sha256: str
    frozen_structure_sha256: str
    transport_fit_manifest_sha256: str
    transport_weights_sha256: str
    risk_fit_manifest_sha256: str
    risk_calibration_manifest_sha256: str
    predictor_sha256: str
    threshold: float
    validation_report_sha256: str
    evaluator_sha256: str
    selective_artifact_id: str
    selective_manifest_file_sha256: str
    thresholds: SelectiveQualityThresholds
    accepted_quality: AcceptedSubsetQualityEvidence
    baselines: tuple[SelectorEvaluation, ...]
    sample_count: int
    passed: bool = True
    schema_version: str = V5_VALIDATION_MANIFEST_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        structure: FrozenTransportStructure,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
        risk_fit: V5RiskFitManifest,
        calibration: V5RiskCalibrationManifest,
        selective: SelectiveKVBridgeManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_VALIDATION_MANIFEST_SCHEMA:
            errors.append("unsupported v5 validation manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("validation manifest belongs to another pipeline")
        if self.direction != trace.direction or self.direction != calibration.direction:
            errors.append("validation direction is inconsistent")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("validation code hash mismatch")
        if self.validation_split_sha256 != workspace.config.split_sha256["validation"]:
            errors.append("validation split hash mismatch")
        if self.validation_trace_manifest_sha256 != trace.content_sha256():
            errors.append("validation trace hash mismatch")
        if self.validation_raw_store_sha256 != trace.raw_sample_store_sha256:
            errors.append("validation raw store hash mismatch")
        if self.frozen_structure_sha256 != structure.content_sha256():
            errors.append("validation frozen structure hash mismatch")
        if self.transport_fit_manifest_sha256 != transport_manifest.content_sha256():
            errors.append("validation transport manifest hash mismatch")
        if self.transport_weights_sha256 != candidate.weights.sha256:
            errors.append("validation transport weights changed")
        if self.risk_fit_manifest_sha256 != risk_fit.content_sha256():
            errors.append("validation risk fit manifest changed")
        if self.risk_calibration_manifest_sha256 != calibration.content_sha256():
            errors.append("validation risk calibration manifest changed")
        if self.predictor_sha256 != risk_fit.predictor.sha256:
            errors.append("validation predictor changed")
        if calibration.risk_gate.threshold is None or self.threshold != (
            calibration.risk_gate.threshold
        ):
            errors.append("validation threshold changed")
        for name in (
            "validation_report_sha256",
            "evaluator_sha256",
            "selective_manifest_file_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"validation {name} is invalid")
        if self.thresholds != SelectiveQualityThresholds():
            errors.append("validation quality thresholds differ from the frozen contract")
        errors.extend(_accepted_quality_type_errors(self.accepted_quality))
        if self.accepted_quality.evaluation_dataset_sha256 != self.validation_split_sha256:
            errors.append("accepted quality refers to another validation split")
        try:
            errors.extend(self.accepted_quality.gate_errors(self.thresholds))
        except (TypeError, ValueError) as exc:
            errors.append(f"accepted validation quality is malformed: {type(exc).__name__}")
        errors.extend(_baseline_errors(self.baselines, self.accepted_quality, self.sample_count))
        if type(self.sample_count) is not int or self.sample_count != SPLIT_COUNTS["validation"]:
            errors.append("validation sample count is inconsistent")
        if not _strict_bool(self.passed) or not self.passed:
            errors.append("validation manifest does not carry a passing result")
        if self.selective_artifact_id != selective.artifact_id:
            errors.append("validation selective artifact id changed")
        if selective.direction != self.direction:
            errors.append("validation selective manifest direction changed")
        if selective.transport.weights_sha256 != candidate.weights.sha256:
            errors.append("validation selective transport changed")
        if selective.risk_gate != calibration.risk_gate:
            errors.append("validation selective risk gate changed")
        if selective.thresholds != self.thresholds:
            errors.append("validation selective thresholds changed")
        if selective.accepted_quality != self.accepted_quality:
            errors.append("validation selective accepted quality changed")
        if selective.transport_quality != structure.deployment_quality:
            errors.append("validation selective method-dev quality changed")
        try:
            errors.extend(selective.artifact_errors())
            errors.extend(selective.validation_errors())
        except (TypeError, ValueError) as exc:
            errors.append(f"validation selective manifest is malformed: {type(exc).__name__}")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5ValidationManifest:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            validation_split_sha256=str(payload["validation_split_sha256"]),
            validation_trace_manifest_sha256=str(payload["validation_trace_manifest_sha256"]),
            validation_raw_store_sha256=str(payload["validation_raw_store_sha256"]),
            frozen_structure_sha256=str(payload["frozen_structure_sha256"]),
            transport_fit_manifest_sha256=str(payload["transport_fit_manifest_sha256"]),
            transport_weights_sha256=str(payload["transport_weights_sha256"]),
            risk_fit_manifest_sha256=str(payload["risk_fit_manifest_sha256"]),
            risk_calibration_manifest_sha256=str(payload["risk_calibration_manifest_sha256"]),
            predictor_sha256=str(payload["predictor_sha256"]),
            threshold=payload["threshold"],
            validation_report_sha256=str(payload["validation_report_sha256"]),
            evaluator_sha256=str(payload["evaluator_sha256"]),
            selective_artifact_id=str(payload["selective_artifact_id"]),
            selective_manifest_file_sha256=str(payload["selective_manifest_file_sha256"]),
            thresholds=SelectiveQualityThresholds(**payload["thresholds"]),
            accepted_quality=AcceptedSubsetQualityEvidence(**payload["accepted_quality"]),
            baselines=tuple(SelectorEvaluation(**item) for item in payload["baselines"]),
            sample_count=payload["sample_count"],
            passed=payload.get("passed", False),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        structure: FrozenTransportStructure,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
        risk_fit: V5RiskFitManifest,
        calibration: V5RiskCalibrationManifest,
        selective: SelectiveKVBridgeManifest,
    ) -> V5ValidationManifest:
        try:
            value = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = value.validate(
                workspace=workspace,
                trace=trace,
                structure=structure,
                transport_manifest=transport_manifest,
                candidate=candidate,
                risk_fit=risk_fit,
                calibration=calibration,
                selective=selective,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("validation manifest is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        return value


def run_validate_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], RiskExampleEvaluator],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Validate one calibrated direction and emit a non-authoritative candidate."""

    benchmark = load_bound_benchmark(workspace)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    structure, _, _ = load_frozen_transport_structure(workspace)
    trace = load_completed_trace_manifest(workspace, direction, "validation", benchmark)
    store_path = Path(sample_store_path)
    before = _file_signature(store_path)
    store_sha256 = sha256_file(store_path)
    if store_sha256 != trace.raw_sample_store_sha256:
        raise V5PipelineError("validation raw store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="validation")
    traces = {item.sample_id: item for item in trace.records}
    if set(traces) != {record.sample_id for record, _ in samples}:
        raise V5PipelineError("validation samples differ from trace manifest")
    evaluator_payload = dict(evaluator_parameters)
    evaluator_sha256 = _sha256_bytes(_canonical_json_bytes(evaluator_payload))
    thresholds = SelectiveQualityThresholds()
    stage_parameters = {
        "validation_trace_manifest_sha256": trace.content_sha256(),
        "raw_sample_store_sha256": store_sha256,
        "frozen_structure_sha256": structure.content_sha256(),
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "risk_calibration_manifest_sha256": calibration.content_sha256(),
        "predictor_sha256": risk_fit.predictor.sha256,
        "threshold": calibration.risk_gate.threshold,
        "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
        "predictor_device": VALIDATION_PREDICTOR_DEVICE,
        "selector_cosine_threshold": SELECTOR_COSINE_THRESHOLD,
        "quality_thresholds": asdict(thresholds),
        "evaluator": evaluator_payload,
    }
    lease = workspace.begin_stage(
        direction,
        "validate",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/validate"]
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=VALIDATION_PREDICTOR_DEVICE,
    )
    work = workspace.control / "work" / direction / "validate"
    checkpoint_dir = work / "examples"
    histories: dict[str, RiskHistory] = {}
    measurements: list[RiskValidationMeasurement] = []
    try:
        with evaluator_factory() as evaluator:
            for index, (benchmark_record, sample) in enumerate(samples, start=1):
                trace_record = traces[benchmark_record.sample_id]
                history = histories.get(benchmark_record.prefix_group_id, RiskHistory())
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                measurement = _load_validation_checkpoint(
                    checkpoint,
                    binding_sha256=lease.input_sha256,
                    benchmark_record=benchmark_record,
                    trace_record=trace_record,
                    expected_history=history,
                    predictor=predictor,
                    risk_gate=calibration.risk_gate,
                )
                if measurement is None:
                    example = evaluator.evaluate(
                        benchmark_record,
                        trace_record,
                        sample,
                        history,
                    )
                    probability = predictor.unsafe_probability(example.features)
                    accepted, decision = _admission_decision(
                        example,
                        probability,
                        calibration.risk_gate,
                    )
                    measurement = RiskValidationMeasurement(
                        example=example,
                        unsafe_probability=probability,
                        accepted=accepted,
                        decision=decision,
                    )
                    errors = measurement.validate(
                        predictor=predictor,
                        risk_gate=calibration.risk_gate,
                        benchmark_record=benchmark_record,
                        trace_record=trace_record,
                        expected_history=history,
                    )
                    if errors:
                        raise V5PipelineError("; ".join(errors))
                    _write_validation_checkpoint(
                        checkpoint,
                        binding_sha256=lease.input_sha256,
                        benchmark_record=benchmark_record,
                        measurement=measurement,
                    )
                histories[benchmark_record.prefix_group_id] = history.update(measurement.example)
                measurements.append(measurement)
                if progress is not None:
                    progress(index, len(samples), sample.sample_id)
        if _file_signature(store_path) != before or sha256_file(store_path) != store_sha256:
            raise V5PipelineError("validation raw store changed during evaluation")
        quality = aggregate_accepted_quality(
            measurements,
            dataset_sha256=workspace.config.split_sha256["validation"],
        )
        baselines = validation_selector_baselines(
            measurements,
            calibrated_threshold=calibration.risk_gate.threshold,
        )
        gate_errors = quality.gate_errors(thresholds)
        if gate_errors:
            raise V5PipelineError("validation quality gates failed: " + "; ".join(gate_errors))
        selective = build_validation_candidate(
            workspace=workspace,
            transport_manifest=transport_manifest,
            candidate=candidate,
            structure=structure,
            calibration=calibration,
            quality=quality,
            thresholds=thresholds,
        )
        selective_errors = selective.artifact_errors() + selective.validation_errors()
        if selective_errors:
            raise V5PipelineError("; ".join(selective_errors))
        report = {
            "schema_version": V5_VALIDATION_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": direction,
            "code_sha256": workspace.config.code_sha256,
            "validation_split_sha256": workspace.config.split_sha256["validation"],
            "validation_trace_manifest_sha256": trace.content_sha256(),
            "validation_raw_store_sha256": store_sha256,
            "frozen_structure_sha256": structure.content_sha256(),
            "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "risk_calibration_manifest_sha256": calibration.content_sha256(),
            "predictor_sha256": risk_fit.predictor.sha256,
            "threshold": calibration.risk_gate.threshold,
            "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "predictor_device": VALIDATION_PREDICTOR_DEVICE,
            "selector_cosine_threshold": SELECTOR_COSINE_THRESHOLD,
            "evaluator": evaluator_payload,
            "quality_thresholds": asdict(thresholds),
            "measurements": [item.to_dict() for item in measurements],
            "accepted_quality": asdict(quality),
            "baselines": [asdict(item) for item in baselines],
            "selective_artifact_id": selective.artifact_id,
            "passed": True,
        }
        report_path = work / "validation_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        selective_path = work / "selective_manifest.json"
        _write_json_replace(selective_path, selective.to_dict())
        selective_file_sha256 = sha256_file(selective_path)
        manifest = V5ValidationManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            validation_split_sha256=workspace.config.split_sha256["validation"],
            validation_trace_manifest_sha256=trace.content_sha256(),
            validation_raw_store_sha256=store_sha256,
            frozen_structure_sha256=structure.content_sha256(),
            transport_fit_manifest_sha256=transport_manifest.content_sha256(),
            transport_weights_sha256=candidate.weights.sha256,
            risk_fit_manifest_sha256=risk_fit.content_sha256(),
            risk_calibration_manifest_sha256=calibration.content_sha256(),
            predictor_sha256=risk_fit.predictor.sha256,
            threshold=_required_threshold(calibration),
            validation_report_sha256=report_sha256,
            evaluator_sha256=evaluator_sha256,
            selective_artifact_id=selective.artifact_id,
            selective_manifest_file_sha256=selective_file_sha256,
            thresholds=thresholds,
            accepted_quality=quality,
            baselines=baselines,
            sample_count=len(measurements),
        )
        errors = manifest.validate(
            workspace=workspace,
            trace=trace,
            structure=structure,
            transport_manifest=transport_manifest,
            candidate=candidate,
            risk_fit=risk_fit,
            calibration=calibration,
            selective=selective,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        manifest_path = work / "validation_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "validation_report": report_path,
                "validation_manifest": manifest_path,
                "selective_manifest": selective_path,
            },
            metadata={
                "sample_count": len(measurements),
                "accepted_count": quality.accepted_count,
                "unsafe_count": quality.unsafe_count,
                "coverage": quality.coverage,
                "regression_risk_upper_bound": quality.regression_risk_upper_bound,
                "selective_artifact_id": selective.artifact_id,
                "validation_manifest_sha256": manifest.content_sha256(),
                "passed": True,
                "authority": ArtifactState.VALIDATION_CANDIDATE.value,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def aggregate_accepted_quality(
    measurements: Sequence[RiskValidationMeasurement],
    *,
    dataset_sha256: str,
    expected_count: int | None = None,
) -> AcceptedSubsetQualityEvidence:
    required_count = SPLIT_COUNTS["validation"] if expected_count is None else expected_count
    if type(required_count) is not int or required_count <= 0:
        raise V5PipelineError("quality aggregate expected count is invalid")
    if len(measurements) != required_count:
        raise V5PipelineError("quality aggregate requires the complete registered split")
    accepted = [item for item in measurements if item.accepted]
    if not accepted:
        raise V5PipelineError("validation calibrated selector accepted no samples")
    native = _finite_mean([item.example.native_task_score for item in accepted])
    bridge = _finite_mean([item.example.bridge_task_score for item in accepted])
    task_drop = max(0.0, (native - bridge) / native * 100) if native > 0 else 0.0
    unsafe_count = sum(int(item.example.unsafe) for item in accepted)
    return AcceptedSubsetQualityEvidence(
        evaluation_dataset_sha256=dataset_sha256,
        total_count=len(measurements),
        accepted_count=len(accepted),
        unsafe_count=unsafe_count,
        coverage=len(accepted) / len(measurements),
        native_task_score=native,
        bridge_task_score=bridge,
        task_score_drop_pct=task_drop,
        greedy_agreement=_finite_mean([item.example.greedy_agreement for item in accepted]),
        perplexity_drift_pct=_finite_mean([item.example.perplexity_drift_pct for item in accepted]),
        regression_risk_upper_bound=clopper_pearson_upper_bound(
            unsafe_count,
            len(accepted),
        ),
        key_cosine=_finite_mean([item.example.key_cosine for item in accepted]),
        value_cosine=None,
    )


def validation_selector_baselines(
    measurements: Sequence[RiskValidationMeasurement],
    *,
    calibrated_threshold: float | None,
) -> tuple[SelectorEvaluation, ...]:
    if calibrated_threshold is None:
        raise V5PipelineError("validation requires a calibrated threshold")
    examples = tuple(
        SelectorEvaluationExample(
            unsafe=item.example.unsafe,
            predictor_probability=item.unsafe_probability,
            cosine=item.example.key_cosine,
            runtime_eligible=item.runtime_eligible,
        )
        for item in measurements
    )
    try:
        return evaluate_selector_baselines(
            examples,
            calibrated_threshold=calibrated_threshold,
            cosine_threshold=SELECTOR_COSINE_THRESHOLD,
        )
    except (TypeError, ValueError) as exc:
        raise V5PipelineError("validation selector baselines are malformed") from exc


def build_validation_candidate(
    *,
    workspace: V5PipelineWorkspace,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    structure: FrozenTransportStructure,
    calibration: V5RiskCalibrationManifest,
    quality: AcceptedSubsetQualityEvidence,
    thresholds: SelectiveQualityThresholds,
) -> SelectiveKVBridgeManifest:
    transport = TransportSpec(
        weights_uri=candidate.weights.path,
        weights_sha256=candidate.weights.sha256,
        rank=candidate.rank,
        source_window=transport_manifest.training.source_window,
        loss=transport_manifest.training.loss,
    )
    manifest = SelectiveKVBridgeManifest(
        artifact_id="",
        direction=transport_manifest.direction,
        source=transport_manifest.source,
        target=transport_manifest.target,
        transport=transport,
        risk_gate=calibration.risk_gate,
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        transport_train_dataset_sha256=workspace.config.split_sha256["transport_train"],
        selector_train_dataset_sha256=workspace.config.split_sha256["selector_train"],
        method_dev_dataset_sha256=workspace.config.split_sha256["method_dev"],
        risk_calibration_dataset_sha256=workspace.config.split_sha256["risk_calibration"],
        validation_dataset_sha256=workspace.config.split_sha256["validation"],
        semantic_sealed_dataset_sha256=workspace.config.split_sha256["semantic_sealed_test"],
        runtime_audit_dataset_sha256=workspace.config.split_sha256["runtime_audit"],
        transport_quality=structure.deployment_quality,
        accepted_quality=quality,
        thresholds=thresholds,
        state=ArtifactState.VALIDATION_CANDIDATE,
    )
    return manifest.with_content_id()


def load_completed_validation(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[V5ValidationManifest, SelectiveKVBridgeManifest, V5TraceManifest]:
    benchmark = load_bound_benchmark(workspace)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    structure, _, _ = load_frozen_transport_structure(workspace)
    trace = load_completed_trace_manifest(workspace, direction, "validation", benchmark)
    state = workspace.state()
    stage = state.stages.get(f"{direction}/validate")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed passing validation")
    report_artifact = stage.outputs.get("validation_report")
    manifest_artifact = stage.outputs.get("validation_manifest")
    selective_artifact = stage.outputs.get("selective_manifest")
    if report_artifact is None or manifest_artifact is None or selective_artifact is None:
        raise V5PipelineError("validation stage lacks required evidence artifacts")
    report_path = workspace.artifact_path(report_artifact, verify_hash=True)
    manifest_path = workspace.artifact_path(manifest_artifact, verify_hash=True)
    selective_path = workspace.artifact_path(selective_artifact, verify_hash=True)
    try:
        selective = SelectiveKVBridgeManifest.from_dict(
            json.loads(selective_path.read_text(encoding="utf-8"))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("validation selective manifest is unreadable or malformed") from exc
    manifest = V5ValidationManifest.load(
        manifest_path,
        workspace=workspace,
        trace=trace,
        structure=structure,
        transport_manifest=transport_manifest,
        candidate=candidate,
        risk_fit=risk_fit,
        calibration=calibration,
        selective=selective,
    )
    if manifest.validation_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("validation manifest refers to another report")
    if manifest.selective_manifest_file_sha256 != selective_artifact.sha256:
        raise V5PipelineError("validation manifest refers to another selective manifest")
    predictor = load_risk_predictor(workspace, risk_fit, device=VALIDATION_PREDICTOR_DEVICE)
    _load_and_validate_report(
        report_path,
        benchmark=benchmark,
        workspace=workspace,
        trace=trace,
        structure=structure,
        transport_manifest=transport_manifest,
        candidate=candidate,
        risk_fit=risk_fit,
        calibration=calibration,
        manifest=manifest,
        predictor=predictor,
    )
    return manifest, selective, trace


def _load_and_validate_report(
    path: Path,
    *,
    benchmark: PublicationBenchmarkManifest,
    workspace: V5PipelineWorkspace,
    trace: V5TraceManifest,
    structure: FrozenTransportStructure,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
    manifest: V5ValidationManifest,
    predictor: RiskPredictor,
) -> tuple[RiskValidationMeasurement, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_header = {
            "schema_version": V5_VALIDATION_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": trace.direction,
            "code_sha256": workspace.config.code_sha256,
            "validation_split_sha256": workspace.config.split_sha256["validation"],
            "validation_trace_manifest_sha256": trace.content_sha256(),
            "validation_raw_store_sha256": trace.raw_sample_store_sha256,
            "frozen_structure_sha256": structure.content_sha256(),
            "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "risk_calibration_manifest_sha256": calibration.content_sha256(),
            "predictor_sha256": risk_fit.predictor.sha256,
            "threshold": calibration.risk_gate.threshold,
            "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "predictor_device": VALIDATION_PREDICTOR_DEVICE,
            "selector_cosine_threshold": SELECTOR_COSINE_THRESHOLD,
            "selective_artifact_id": manifest.selective_artifact_id,
            "passed": True,
        }
        if any(payload.get(name) != value for name, value in expected_header.items()):
            raise V5PipelineError("validation report header binding changed")
        evaluator = payload["evaluator"]
        if not isinstance(evaluator, dict) or (
            _sha256_bytes(_canonical_json_bytes(evaluator)) != manifest.evaluator_sha256
        ):
            raise V5PipelineError("validation evaluator metadata changed")
        if payload["quality_thresholds"] != asdict(manifest.thresholds):
            raise V5PipelineError("validation report quality thresholds changed")
        measurements = tuple(
            RiskValidationMeasurement.from_dict(item) for item in payload["measurements"]
        )
        reported_quality = AcceptedSubsetQualityEvidence(**payload["accepted_quality"])
        reported_baselines = tuple(SelectorEvaluation(**item) for item in payload["baselines"])
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("validation report is unreadable or malformed") from exc
    quality_type_errors = _accepted_quality_type_errors(reported_quality)
    if quality_type_errors:
        raise V5PipelineError("; ".join(quality_type_errors))
    expected_records = sorted(
        (item for item in benchmark.records if item.split == "validation"),
        key=lambda item: item.sample_id,
    )
    trace_by_id = {item.sample_id: item for item in trace.records}
    if len(measurements) != SPLIT_COUNTS["validation"] or len(measurements) != len(
        expected_records
    ):
        raise V5PipelineError("validation report measurement count is inconsistent")
    histories: dict[str, RiskHistory] = {}
    for measurement, benchmark_record in zip(measurements, expected_records, strict=True):
        if measurement.example.sample_id != benchmark_record.sample_id:
            raise V5PipelineError("validation report order or sample identity changed")
        trace_record = trace_by_id.get(benchmark_record.sample_id)
        if trace_record is None:
            raise V5PipelineError("validation report refers to a missing trace")
        history = histories.get(benchmark_record.prefix_group_id, RiskHistory())
        errors = measurement.validate(
            predictor=predictor,
            risk_gate=calibration.risk_gate,
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        histories[benchmark_record.prefix_group_id] = history.update(measurement.example)
    expected_quality = aggregate_accepted_quality(
        measurements,
        dataset_sha256=workspace.config.split_sha256["validation"],
    )
    expected_baselines = validation_selector_baselines(
        measurements,
        calibrated_threshold=calibration.risk_gate.threshold,
    )
    if (
        reported_quality != expected_quality
        or reported_quality != manifest.accepted_quality
        or reported_baselines != expected_baselines
        or reported_baselines != manifest.baselines
    ):
        raise V5PipelineError("validation aggregates differ from the detailed report")
    return measurements


def _admission_decision(
    example: RiskTrainingExample,
    unsafe_probability: float,
    risk_gate: Any,
) -> tuple[bool, str]:
    threshold = risk_gate.threshold
    if threshold is None or not _finite_probability(threshold):
        raise V5PipelineError("validation risk threshold is invalid")
    if (
        type(example.history_samples) is not int
        or type(risk_gate.min_shadow_samples) is not int
        or risk_gate.min_shadow_samples < 1
    ):
        raise V5PipelineError("validation shadow-history contract is invalid")
    if example.history_samples < risk_gate.min_shadow_samples:
        return False, "unseen_or_insufficient_shadow_history"
    if len(example.features) != RISK_FEATURE_DIM:
        raise V5PipelineError("validation risk feature width is invalid")
    ood_distance = example.features[RISK_FEATURE_OOD_INDEX]
    if not _finite_nonnegative(ood_distance):
        raise V5PipelineError("validation source OOD feature is invalid")
    if not _finite_nonnegative(risk_gate.ood_threshold):
        raise V5PipelineError("validation OOD threshold is invalid")
    if ood_distance > risk_gate.ood_threshold:
        return False, "out_of_distribution"
    if not _finite_probability(unsafe_probability):
        raise V5PipelineError("validation predictor probability is invalid")
    if unsafe_probability > threshold:
        return False, "predicted_unsafe"
    return True, "accepted"


def _baseline_errors(
    baselines: Sequence[SelectorEvaluation],
    quality: AcceptedSubsetQualityEvidence,
    sample_count: Any,
) -> list[str]:
    errors: list[str] = []
    expected_names = (
        "no_selector",
        "cosine_threshold",
        "uncalibrated_mlp",
        "calibrated_selector",
        "oracle_selector",
    )
    if tuple(item.name for item in baselines) != expected_names:
        errors.append("validation selector baseline set is incomplete")
        return errors
    for item in baselines:
        if (
            type(item.accepted_count) is not int
            or type(item.total_count) is not int
            or type(item.error_count) is not int
            or item.total_count != sample_count
            or not 0 <= item.accepted_count <= item.total_count
            or not 0 <= item.error_count <= item.accepted_count
        ):
            errors.append(f"validation selector baseline {item.name} counts are invalid")
            continue
        expected_coverage = item.accepted_count / item.total_count if item.total_count else 0.0
        if not _finite_probability(item.coverage) or abs(item.coverage - expected_coverage) > 1e-9:
            errors.append(f"validation selector baseline {item.name} coverage is invalid")
        expected_upper = (
            clopper_pearson_upper_bound(item.error_count, item.accepted_count)
            if item.accepted_count
            else 1.0
        )
        if (
            not _finite_probability(item.regression_risk_upper_bound)
            or abs(item.regression_risk_upper_bound - expected_upper) > 1e-8
        ):
            errors.append(f"validation selector baseline {item.name} risk bound is invalid")
    calibrated = baselines[3]
    if (
        calibrated.accepted_count != quality.accepted_count
        or calibrated.error_count != quality.unsafe_count
        or calibrated.coverage != quality.coverage
        or calibrated.regression_risk_upper_bound != quality.regression_risk_upper_bound
    ):
        errors.append("calibrated selector baseline differs from accepted quality")
    return errors


def _accepted_quality_type_errors(quality: AcceptedSubsetQualityEvidence) -> list[str]:
    errors: list[str] = []
    for name in ("total_count", "accepted_count", "unsafe_count"):
        if type(getattr(quality, name)) is not int:
            errors.append(f"accepted validation quality {name} must be an integer")
    for name in (
        "coverage",
        "native_task_score",
        "bridge_task_score",
        "task_score_drop_pct",
        "greedy_agreement",
        "perplexity_drift_pct",
        "regression_risk_upper_bound",
    ):
        if not _finite_number(getattr(quality, name)):
            errors.append(f"accepted validation quality {name} must be finite")
    for name in ("key_cosine", "value_cosine"):
        value = getattr(quality, name)
        if value is not None and not _finite_number(value):
            errors.append(f"accepted validation quality {name} must be finite when present")
    return errors


def _required_threshold(calibration: V5RiskCalibrationManifest) -> float:
    threshold = calibration.risk_gate.threshold
    if threshold is None or not _finite_probability(threshold):
        raise V5PipelineError("validation requires a finite calibrated threshold")
    return threshold


def _write_validation_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    measurement: RiskValidationMeasurement,
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_VALIDATION_CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "sample_id": benchmark_record.sample_id,
            "content_sha256": benchmark_record.content_sha256,
            "measurement": measurement.to_dict(),
        },
    )


def _load_validation_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    trace_record: TraceRecord,
    expected_history: RiskHistory,
    predictor: RiskPredictor,
    risk_gate: Any,
) -> RiskValidationMeasurement | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("validation checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_VALIDATION_CHECKPOINT_SCHEMA:
            raise V5PipelineError("validation checkpoint schema mismatch")
        if payload.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("validation checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != benchmark_record.sample_id
            or payload.get("content_sha256") != benchmark_record.content_sha256
        ):
            raise V5PipelineError("validation checkpoint sample binding mismatch")
        measurement = RiskValidationMeasurement.from_dict(payload["measurement"])
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
        raise V5PipelineError("validation checkpoint is malformed") from exc


def _finite_mean(values: Sequence[float]) -> float:
    if not values or any(not _finite_number(value) for value in values):
        raise V5PipelineError("validation aggregate contains invalid values")
    try:
        result = math.fsum(values) / len(values)
    except OverflowError as exc:
        raise V5PipelineError("validation aggregate overflowed") from exc
    if not math.isfinite(result):
        raise V5PipelineError("validation aggregate is non-finite")
    return result


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
        raise V5PipelineError("validation metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("validation input file is unavailable") from exc
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


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _strict_bool(value: Any) -> bool:
    return type(value) is bool


def _finite_probability(value: Any) -> bool:
    return _finite_number(value) and 0 <= value <= 1


def _finite_nonnegative(value: Any) -> bool:
    return _finite_number(value) and value >= 0


def stderr_validation_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("validation progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"validation {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
