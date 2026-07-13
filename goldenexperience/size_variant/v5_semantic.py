"""Resumable four-direction evaluation of the one-shot v5 semantic snapshot."""

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
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.cached_kv_manifest import (
    sha256_file,
    tokenizer_semantic_sha256,
)
from goldenexperience.size_variant.risk_gate import RiskPredictor, SelectorEvaluation
from goldenexperience.size_variant.selective_manifest import (
    AcceptedSubsetQualityEvidence,
    ArtifactState,
    SelectiveKVBridgeManifest,
    SelectiveQualityThresholds,
    SemanticSealedEvidence,
)
from goldenexperience.size_variant.v5_calibration import (
    RISK_CALIBRATION_PREDICTOR_DEVICE,
    V5RiskCalibrationManifest,
    load_completed_risk_calibration,
)
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    load_bound_benchmark,
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
    RiskPrefixTokenBinding,
    RiskTrainingExample,
    V5RiskFitManifest,
    load_risk_predictor,
)
from goldenexperience.size_variant.v5_sealed import (
    V5SemanticOpenReceipt,
    load_semantic_open_receipt,
    load_semantic_snapshot_bytes,
)
from goldenexperience.size_variant.v5_validation import (
    VALIDATION_PREDICTOR_DEVICE,
    RiskValidationMeasurement,
    V5ValidationManifest,
    _accepted_quality_type_errors,
    _admission_decision,
    _baseline_errors,
    aggregate_accepted_quality,
    load_completed_validation,
    validation_selector_baselines,
)

V5_SEMANTIC_CHECKPOINT_SCHEMA = "goldenexperience.v5_semantic_checkpoint.v1"
V5_SEMANTIC_REPORT_SCHEMA = "goldenexperience.v5_semantic_report.v1"
V5_SEMANTIC_MANIFEST_SCHEMA = "goldenexperience.v5_semantic_manifest.v1"
SEMANTIC_PREDICTOR_DEVICE = RISK_CALIBRATION_PREDICTOR_DEVICE

if SEMANTIC_PREDICTOR_DEVICE != VALIDATION_PREDICTOR_DEVICE:
    raise RuntimeError("semantic and validation predictor devices must remain identical")


class SemanticRiskExampleEvaluator(Protocol):
    def __enter__(self) -> SemanticRiskExampleEvaluator: ...

    def __exit__(self, *_args: object) -> None: ...

    def bind_semantic_prefix(
        self,
        benchmark_record: GroupedPrefixRecord,
        sample: RawBenchmarkSample,
    ) -> RiskPrefixTokenBinding: ...

    def evaluate(
        self,
        benchmark_record: GroupedPrefixRecord,
        prefix_binding: RiskPrefixTokenBinding,
        sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample: ...


@dataclass(frozen=True)
class SemanticRiskMeasurement:
    prefix_binding: RiskPrefixTokenBinding
    validation: RiskValidationMeasurement

    def validate(
        self,
        *,
        predictor: RiskPredictor,
        risk_gate: Any,
        benchmark_record: GroupedPrefixRecord,
        expected_history: RiskHistory,
    ) -> list[str]:
        errors = self.prefix_binding.validate(benchmark_record)
        errors.extend(
            self.validation.validate(
                predictor=predictor,
                risk_gate=risk_gate,
                benchmark_record=benchmark_record,
                trace_record=self.prefix_binding,
                expected_history=expected_history,
            )
        )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_binding": asdict(self.prefix_binding),
            "validation": self.validation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SemanticRiskMeasurement:
        return cls(
            prefix_binding=RiskPrefixTokenBinding(**payload["prefix_binding"]),
            validation=RiskValidationMeasurement.from_dict(payload["validation"]),
        )


@dataclass(frozen=True)
class V5SemanticManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    benchmark_manifest_sha256: str
    semantic_split_sha256: str
    sealed_payload_sha256: str
    semantic_open_receipt_sha256: str
    validation_manifest_sha256: str
    validation_report_sha256: str
    validation_selective_artifact_id: str
    risk_fit_manifest_sha256: str
    risk_calibration_manifest_sha256: str
    transport_weights_sha256: str
    predictor_sha256: str
    threshold: float
    semantic_report_sha256: str
    evaluator_sha256: str
    semantic_selective_artifact_id: str
    semantic_selective_manifest_file_sha256: str
    thresholds: SelectiveQualityThresholds
    accepted_quality: AcceptedSubsetQualityEvidence
    baselines: tuple[SelectorEvaluation, ...]
    sample_count: int
    passed: bool = True
    schema_version: str = V5_SEMANTIC_MANIFEST_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        receipt: V5SemanticOpenReceipt,
        validation: V5ValidationManifest,
        validation_selective: SelectiveKVBridgeManifest,
        risk_fit: V5RiskFitManifest,
        calibration: V5RiskCalibrationManifest,
        candidate: TransportCandidateArtifact,
        semantic_selective: SelectiveKVBridgeManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_SEMANTIC_MANIFEST_SCHEMA:
            errors.append("unsupported v5 semantic manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("semantic manifest belongs to another pipeline")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        expected = {
            "code_sha256": workspace.config.code_sha256,
            "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
            "semantic_split_sha256": workspace.config.split_sha256["semantic_sealed_test"],
            "sealed_payload_sha256": workspace.config.sealed_payload_sha256,
            "semantic_open_receipt_sha256": receipt.content_sha256(),
            "validation_manifest_sha256": validation.content_sha256(),
            "validation_report_sha256": validation.validation_report_sha256,
            "validation_selective_artifact_id": validation_selective.artifact_id,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "risk_calibration_manifest_sha256": calibration.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "predictor_sha256": risk_fit.predictor.sha256,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                errors.append(f"semantic manifest {name} changed")
        for name in (
            "code_sha256",
            "benchmark_manifest_sha256",
            "semantic_split_sha256",
            "sealed_payload_sha256",
            "semantic_open_receipt_sha256",
            "validation_manifest_sha256",
            "validation_report_sha256",
            "risk_fit_manifest_sha256",
            "risk_calibration_manifest_sha256",
            "transport_weights_sha256",
            "predictor_sha256",
            "semantic_report_sha256",
            "evaluator_sha256",
            "semantic_selective_manifest_file_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"semantic manifest {name} is invalid")
        if self.direction != validation.direction or self.direction != calibration.direction:
            errors.append("semantic manifest direction is inconsistent")
        binding = next(
            (item for item in receipt.directions if item.direction == self.direction),
            None,
        )
        if binding is None:
            errors.append("semantic open receipt lacks the evaluated direction")
        elif (
            binding.validation_manifest_sha256 != self.validation_manifest_sha256
            or binding.validation_report_sha256 != self.validation_report_sha256
            or binding.risk_calibration_manifest_sha256 != self.risk_calibration_manifest_sha256
            or binding.selective_artifact_id != self.validation_selective_artifact_id
            or binding.transport_weights_sha256 != self.transport_weights_sha256
            or binding.predictor_sha256 != self.predictor_sha256
            or binding.threshold != self.threshold
            or not binding.passed
        ):
            errors.append("semantic direction differs from the one-shot open receipt")
        if calibration.risk_gate.threshold is None or self.threshold != (
            calibration.risk_gate.threshold
        ):
            errors.append("semantic calibrated threshold changed")
        if self.thresholds != validation.thresholds:
            errors.append("semantic quality thresholds changed after validation")
        errors.extend(_accepted_quality_type_errors(self.accepted_quality))
        if self.accepted_quality.evaluation_dataset_sha256 != self.semantic_split_sha256:
            errors.append("semantic quality refers to another dataset")
        try:
            errors.extend(self.accepted_quality.gate_errors(self.thresholds))
        except (TypeError, ValueError) as exc:
            errors.append(f"semantic quality is malformed: {type(exc).__name__}")
        errors.extend(_baseline_errors(self.baselines, self.accepted_quality, self.sample_count))
        if (
            type(self.sample_count) is not int
            or self.sample_count != SPLIT_COUNTS["semantic_sealed_test"]
        ):
            errors.append("semantic sample count is inconsistent")
        if type(self.passed) is not bool or not self.passed:
            errors.append("semantic manifest does not carry a passing result")
        if validation_selective.state is not ArtifactState.VALIDATION_CANDIDATE:
            errors.append("semantic input is not a validation candidate")
        if self.semantic_selective_artifact_id != semantic_selective.artifact_id:
            errors.append("semantic approved artifact id changed")
        try:
            expected_selective = build_semantic_approved_candidate(
                validation_selective=validation_selective,
                report_sha256=self.semantic_report_sha256,
                code_sha256=self.code_sha256,
                quality=self.accepted_quality,
                sample_count=self.sample_count,
            )
        except (TypeError, ValueError, V5PipelineError) as exc:
            errors.append(f"semantic approved artifact cannot be rebuilt: {type(exc).__name__}")
        else:
            if semantic_selective != expected_selective:
                errors.append("semantic approved artifact differs from sealed evidence")
        try:
            errors.extend(semantic_selective.validate())
        except (TypeError, ValueError) as exc:
            errors.append(f"semantic approved artifact is malformed: {type(exc).__name__}")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5SemanticManifest:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            benchmark_manifest_sha256=str(payload["benchmark_manifest_sha256"]),
            semantic_split_sha256=str(payload["semantic_split_sha256"]),
            sealed_payload_sha256=str(payload["sealed_payload_sha256"]),
            semantic_open_receipt_sha256=str(payload["semantic_open_receipt_sha256"]),
            validation_manifest_sha256=str(payload["validation_manifest_sha256"]),
            validation_report_sha256=str(payload["validation_report_sha256"]),
            validation_selective_artifact_id=str(payload["validation_selective_artifact_id"]),
            risk_fit_manifest_sha256=str(payload["risk_fit_manifest_sha256"]),
            risk_calibration_manifest_sha256=str(payload["risk_calibration_manifest_sha256"]),
            transport_weights_sha256=str(payload["transport_weights_sha256"]),
            predictor_sha256=str(payload["predictor_sha256"]),
            threshold=payload["threshold"],
            semantic_report_sha256=str(payload["semantic_report_sha256"]),
            evaluator_sha256=str(payload["evaluator_sha256"]),
            semantic_selective_artifact_id=str(payload["semantic_selective_artifact_id"]),
            semantic_selective_manifest_file_sha256=str(
                payload["semantic_selective_manifest_file_sha256"]
            ),
            thresholds=SelectiveQualityThresholds(**payload["thresholds"]),
            accepted_quality=AcceptedSubsetQualityEvidence(**payload["accepted_quality"]),
            baselines=tuple(SelectorEvaluation(**item) for item in payload["baselines"]),
            sample_count=payload["sample_count"],
            passed=payload.get("passed", False),
            schema_version=str(payload.get("schema_version", "")),
        )


def run_semantic_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], SemanticRiskExampleEvaluator],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Evaluate one direction using only the guarded immutable semantic snapshot."""

    benchmark = load_bound_benchmark(workspace)
    receipt, snapshot_path = load_semantic_open_receipt(workspace)
    validation, validation_selective, _ = load_completed_validation(workspace, direction)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    _require_direction_open_binding(
        receipt,
        validation=validation,
        validation_selective=validation_selective,
        calibration=calibration,
        risk_fit=risk_fit,
        candidate=candidate,
    )
    samples, snapshot_signature = _load_snapshot(workspace, snapshot_path, benchmark)
    evaluator_payload = dict(evaluator_parameters)
    evaluator_sha256 = _sha256_bytes(_canonical_json_bytes(evaluator_payload))
    thresholds = validation.thresholds
    stage_parameters = {
        "sealed_payload_sha256": receipt.sealed_payload_sha256,
        "validation_manifest_sha256": validation.content_sha256(),
        "validation_report_sha256": validation.validation_report_sha256,
        "validation_selective_artifact_id": validation_selective.artifact_id,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "risk_calibration_manifest_sha256": calibration.content_sha256(),
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "predictor_sha256": risk_fit.predictor.sha256,
        "threshold": calibration.risk_gate.threshold,
        "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
        "predictor_device": SEMANTIC_PREDICTOR_DEVICE,
        "quality_thresholds": asdict(thresholds),
        "evaluator": evaluator_payload,
    }
    lease = workspace.begin_semantic_stage(
        direction,
        parameters=stage_parameters,
        open_receipt_sha256=receipt.content_sha256(),
        snapshot_sha256=receipt.sealed_payload_sha256,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/semantic_sealed"]
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=SEMANTIC_PREDICTOR_DEVICE,
    )
    work = workspace.control / "work" / direction / "semantic_sealed"
    checkpoint_dir = work / "examples"
    histories: dict[str, RiskHistory] = {}
    measurements: list[SemanticRiskMeasurement] = []
    try:
        with evaluator_factory() as evaluator:
            for index, (record, sample) in enumerate(samples, start=1):
                history = histories.get(record.prefix_group_id, RiskHistory())
                binding = evaluator.bind_semantic_prefix(record, sample)
                binding_errors = binding.validate(record)
                if binding_errors:
                    raise V5PipelineError("; ".join(binding_errors))
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                measurement = _load_semantic_checkpoint(
                    checkpoint,
                    stage_binding_sha256=lease.input_sha256,
                    benchmark_record=record,
                    prefix_binding=binding,
                    expected_history=history,
                    predictor=predictor,
                    risk_gate=calibration.risk_gate,
                )
                if measurement is None:
                    example = evaluator.evaluate(record, binding, sample, history)
                    probability = predictor.unsafe_probability(example.features)
                    accepted, decision = _admission_decision(
                        example,
                        probability,
                        calibration.risk_gate,
                    )
                    measurement = SemanticRiskMeasurement(
                        prefix_binding=binding,
                        validation=RiskValidationMeasurement(
                            example=example,
                            unsafe_probability=probability,
                            accepted=accepted,
                            decision=decision,
                        ),
                    )
                    errors = measurement.validate(
                        predictor=predictor,
                        risk_gate=calibration.risk_gate,
                        benchmark_record=record,
                        expected_history=history,
                    )
                    if errors:
                        raise V5PipelineError("; ".join(errors))
                    _write_semantic_checkpoint(
                        checkpoint,
                        stage_binding_sha256=lease.input_sha256,
                        benchmark_record=record,
                        measurement=measurement,
                    )
                histories[record.prefix_group_id] = history.update(measurement.validation.example)
                measurements.append(measurement)
                if progress is not None:
                    progress(index, len(samples), sample.sample_id)
        _verify_snapshot_signature(
            snapshot_path,
            expected_signature=snapshot_signature,
            expected_sha256=receipt.sealed_payload_sha256,
        )
        validation_measurements = tuple(item.validation for item in measurements)
        quality = aggregate_accepted_quality(
            validation_measurements,
            dataset_sha256=workspace.config.split_sha256["semantic_sealed_test"],
            expected_count=SPLIT_COUNTS["semantic_sealed_test"],
        )
        baselines = validation_selector_baselines(
            validation_measurements,
            calibrated_threshold=calibration.risk_gate.threshold,
        )
        gate_errors = quality.gate_errors(thresholds)
        if gate_errors:
            raise V5PipelineError("semantic quality gates failed: " + "; ".join(gate_errors))
        report = {
            **_report_header(
                workspace=workspace,
                direction=direction,
                receipt=receipt,
                validation=validation,
                validation_selective=validation_selective,
                risk_fit=risk_fit,
                calibration=calibration,
                transport_manifest=transport_manifest,
                candidate=candidate,
            ),
            "evaluator": evaluator_payload,
            "quality_thresholds": asdict(thresholds),
            "measurements": [item.to_dict() for item in measurements],
            "accepted_quality": asdict(quality),
            "baselines": [asdict(item) for item in baselines],
        }
        report_path = work / "semantic_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        semantic_selective = build_semantic_approved_candidate(
            validation_selective=validation_selective,
            report_sha256=report_sha256,
            code_sha256=workspace.config.code_sha256,
            quality=quality,
            sample_count=len(measurements),
        )
        selective_errors = semantic_selective.validate()
        if selective_errors:
            raise V5PipelineError("; ".join(selective_errors))
        selective_path = work / "semantic_selective_manifest.json"
        _write_json_replace(selective_path, semantic_selective.to_dict())
        selective_file_sha256 = sha256_file(selective_path)
        manifest = V5SemanticManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
            semantic_split_sha256=workspace.config.split_sha256["semantic_sealed_test"],
            sealed_payload_sha256=receipt.sealed_payload_sha256,
            semantic_open_receipt_sha256=receipt.content_sha256(),
            validation_manifest_sha256=validation.content_sha256(),
            validation_report_sha256=validation.validation_report_sha256,
            validation_selective_artifact_id=validation_selective.artifact_id,
            risk_fit_manifest_sha256=risk_fit.content_sha256(),
            risk_calibration_manifest_sha256=calibration.content_sha256(),
            transport_weights_sha256=candidate.weights.sha256,
            predictor_sha256=risk_fit.predictor.sha256,
            threshold=_required_threshold(calibration),
            semantic_report_sha256=report_sha256,
            evaluator_sha256=evaluator_sha256,
            semantic_selective_artifact_id=semantic_selective.artifact_id,
            semantic_selective_manifest_file_sha256=selective_file_sha256,
            thresholds=thresholds,
            accepted_quality=quality,
            baselines=baselines,
            sample_count=len(measurements),
        )
        errors = manifest.validate(
            workspace=workspace,
            receipt=receipt,
            validation=validation,
            validation_selective=validation_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            candidate=candidate,
            semantic_selective=semantic_selective,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        manifest_path = work / "semantic_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "semantic_report": report_path,
                "semantic_manifest": manifest_path,
                "semantic_selective_manifest": selective_path,
            },
            metadata={
                "sample_count": len(measurements),
                "accepted_count": quality.accepted_count,
                "unsafe_count": quality.unsafe_count,
                "coverage": quality.coverage,
                "regression_risk_upper_bound": quality.regression_risk_upper_bound,
                "semantic_selective_artifact_id": semantic_selective.artifact_id,
                "semantic_manifest_sha256": manifest.content_sha256(),
                "passed": True,
                "authority": ArtifactState.SEMANTIC_APPROVED.value,
                "runtime_authority": False,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def build_semantic_approved_candidate(
    *,
    validation_selective: SelectiveKVBridgeManifest,
    report_sha256: str,
    code_sha256: str,
    quality: AcceptedSubsetQualityEvidence,
    sample_count: int,
) -> SelectiveKVBridgeManifest:
    if validation_selective.state is not ArtifactState.VALIDATION_CANDIDATE:
        raise V5PipelineError("semantic evaluation requires a validation candidate")
    if (
        validation_selective.semantic_sealed is not None
        or validation_selective.runtime_cost is not None
        or validation_selective.direct_injection is not None
    ):
        raise V5PipelineError("validation candidate carries premature authority evidence")
    evidence = SemanticSealedEvidence(
        dataset_sha256=validation_selective.semantic_sealed_dataset_sha256,
        report_sha256=report_sha256,
        sample_count=sample_count,
        code_sha256=code_sha256,
        transport_weights_sha256=validation_selective.transport.weights_sha256,
        predictor_sha256=validation_selective.risk_gate.predictor_sha256,
        threshold=_gate_threshold(validation_selective),
        quality=quality,
    )
    return replace(
        validation_selective,
        artifact_id="",
        semantic_sealed=evidence,
        state=ArtifactState.SEMANTIC_APPROVED,
    ).with_content_id()


def load_completed_semantic(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[V5SemanticManifest, SelectiveKVBridgeManifest]:
    """Load semantic evidence and replay every source-only decision and aggregate."""

    benchmark = load_bound_benchmark(workspace)
    receipt, snapshot_path = load_semantic_open_receipt(workspace)
    validation, validation_selective, _ = load_completed_validation(workspace, direction)
    calibration, risk_fit, _, transport_manifest, candidate = load_completed_risk_calibration(
        workspace, direction
    )
    _require_direction_open_binding(
        receipt,
        validation=validation,
        validation_selective=validation_selective,
        calibration=calibration,
        risk_fit=risk_fit,
        candidate=candidate,
    )
    state = workspace.state()
    stage = state.stages.get(f"{direction}/semantic_sealed")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed passing semantic evaluation")
    report_artifact = stage.outputs.get("semantic_report")
    manifest_artifact = stage.outputs.get("semantic_manifest")
    selective_artifact = stage.outputs.get("semantic_selective_manifest")
    if report_artifact is None or manifest_artifact is None or selective_artifact is None:
        raise V5PipelineError("semantic stage lacks required evidence artifacts")
    report_path = workspace.artifact_path(report_artifact, verify_hash=True)
    manifest_path = workspace.artifact_path(manifest_artifact, verify_hash=True)
    selective_path = workspace.artifact_path(selective_artifact, verify_hash=True)
    try:
        semantic_selective = SelectiveKVBridgeManifest.from_dict(
            json.loads(selective_path.read_text(encoding="utf-8"))
        )
        manifest = V5SemanticManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("semantic manifest is unreadable or malformed") from exc
    errors = manifest.validate(
        workspace=workspace,
        receipt=receipt,
        validation=validation,
        validation_selective=validation_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        candidate=candidate,
        semantic_selective=semantic_selective,
    )
    if errors:
        raise V5PipelineError("; ".join(errors))
    if manifest.semantic_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("semantic manifest refers to another report")
    if manifest.semantic_selective_manifest_file_sha256 != selective_artifact.sha256:
        raise V5PipelineError("semantic manifest refers to another selective manifest")
    samples, _ = _load_snapshot(workspace, snapshot_path, benchmark)
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=SEMANTIC_PREDICTOR_DEVICE,
    )
    _load_and_validate_semantic_report(
        report_path,
        benchmark=benchmark,
        samples=samples,
        workspace=workspace,
        receipt=receipt,
        validation=validation,
        validation_selective=validation_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        transport_manifest=transport_manifest,
        candidate=candidate,
        manifest=manifest,
        predictor=predictor,
    )
    return manifest, semantic_selective


def _load_and_validate_semantic_report(
    path: Path,
    *,
    benchmark: PublicationBenchmarkManifest,
    samples: Sequence[tuple[GroupedPrefixRecord, RawBenchmarkSample]],
    workspace: V5PipelineWorkspace,
    receipt: V5SemanticOpenReceipt,
    validation: V5ValidationManifest,
    validation_selective: SelectiveKVBridgeManifest,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    manifest: V5SemanticManifest,
    predictor: RiskPredictor,
) -> tuple[SemanticRiskMeasurement, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_header = _report_header(
            workspace=workspace,
            direction=manifest.direction,
            receipt=receipt,
            validation=validation,
            validation_selective=validation_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport_manifest,
            candidate=candidate,
        )
        if any(payload.get(name) != value for name, value in expected_header.items()):
            raise V5PipelineError("semantic report header binding changed")
        evaluator = payload["evaluator"]
        if not isinstance(evaluator, dict) or (
            _sha256_bytes(_canonical_json_bytes(evaluator)) != manifest.evaluator_sha256
        ):
            raise V5PipelineError("semantic evaluator metadata changed")
        if payload["quality_thresholds"] != asdict(manifest.thresholds):
            raise V5PipelineError("semantic report quality thresholds changed")
        measurements = tuple(
            SemanticRiskMeasurement.from_dict(item) for item in payload["measurements"]
        )
        reported_quality = AcceptedSubsetQualityEvidence(**payload["accepted_quality"])
        reported_baselines = tuple(SelectorEvaluation(**item) for item in payload["baselines"])
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("semantic report is unreadable or malformed") from exc
    quality_type_errors = _accepted_quality_type_errors(reported_quality)
    if quality_type_errors:
        raise V5PipelineError("; ".join(quality_type_errors))
    expected_records = sorted(
        (item for item in benchmark.records if item.split == "semantic_sealed_test"),
        key=lambda item: item.sample_id,
    )
    if (
        len(measurements) != SPLIT_COUNTS["semantic_sealed_test"]
        or len(measurements) != len(expected_records)
        or len(measurements) != len(samples)
    ):
        raise V5PipelineError("semantic report measurement count is inconsistent")
    expected_bindings = _recompute_prefix_bindings(workspace, manifest.direction, samples)
    histories: dict[str, RiskHistory] = {}
    for measurement, record in zip(measurements, expected_records, strict=True):
        if measurement.prefix_binding != expected_bindings[record.sample_id]:
            raise V5PipelineError("semantic report prefix token identity changed")
        history = histories.get(record.prefix_group_id, RiskHistory())
        errors = measurement.validate(
            predictor=predictor,
            risk_gate=calibration.risk_gate,
            benchmark_record=record,
            expected_history=history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        histories[record.prefix_group_id] = history.update(measurement.validation.example)
    validation_measurements = tuple(item.validation for item in measurements)
    expected_quality = aggregate_accepted_quality(
        validation_measurements,
        dataset_sha256=workspace.config.split_sha256["semantic_sealed_test"],
        expected_count=SPLIT_COUNTS["semantic_sealed_test"],
    )
    expected_baselines = validation_selector_baselines(
        validation_measurements,
        calibrated_threshold=calibration.risk_gate.threshold,
    )
    if (
        reported_quality != expected_quality
        or reported_quality != manifest.accepted_quality
        or reported_baselines != expected_baselines
        or reported_baselines != manifest.baselines
    ):
        raise V5PipelineError("semantic aggregates differ from the detailed report")
    return measurements


def _require_direction_open_binding(
    receipt: V5SemanticOpenReceipt,
    *,
    validation: V5ValidationManifest,
    validation_selective: SelectiveKVBridgeManifest,
    calibration: V5RiskCalibrationManifest,
    risk_fit: V5RiskFitManifest,
    candidate: TransportCandidateArtifact,
) -> None:
    binding = next(
        (item for item in receipt.directions if item.direction == validation.direction),
        None,
    )
    if binding is None or (
        binding.validation_manifest_sha256 != validation.content_sha256()
        or binding.validation_report_sha256 != validation.validation_report_sha256
        or binding.risk_calibration_manifest_sha256 != calibration.content_sha256()
        or binding.selective_artifact_id != validation_selective.artifact_id
        or binding.transport_weights_sha256 != candidate.weights.sha256
        or binding.predictor_sha256 != risk_fit.predictor.sha256
        or binding.threshold != calibration.risk_gate.threshold
        or not binding.passed
    ):
        raise V5PipelineError("semantic direction no longer matches its one-shot binding")


def _load_snapshot(
    workspace: V5PipelineWorkspace,
    path: Path,
    benchmark: PublicationBenchmarkManifest,
) -> tuple[
    tuple[tuple[GroupedPrefixRecord, RawBenchmarkSample], ...],
    tuple[int, int, int, int, int],
]:
    before = _file_signature(path)
    try:
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise V5PipelineError("semantic snapshot must remain immutable")
        payload = path.read_bytes()
    except V5PipelineError:
        raise
    except OSError as exc:
        raise V5PipelineError("semantic snapshot is unavailable") from exc
    if _sha256_bytes(payload) != workspace.config.sealed_payload_sha256:
        raise V5PipelineError("semantic snapshot checksum changed")
    if _file_signature(path) != before:
        raise V5PipelineError("semantic snapshot changed while reading")
    return load_semantic_snapshot_bytes(payload, benchmark), before


def _verify_snapshot_signature(
    path: Path,
    *,
    expected_signature: tuple[int, int, int, int, int],
    expected_sha256: str,
) -> None:
    if _file_signature(path) != expected_signature or sha256_file(path) != expected_sha256:
        raise V5PipelineError("semantic snapshot changed during evaluation")
    if path.is_symlink() or path.stat().st_mode & 0o222:
        raise V5PipelineError("semantic snapshot lost its immutable identity")


def _recompute_prefix_bindings(
    workspace: V5PipelineWorkspace,
    direction: str,
    samples: Sequence[tuple[GroupedPrefixRecord, RawBenchmarkSample]],
) -> dict[str, RiskPrefixTokenBinding]:
    from transformers import AutoTokenizer

    target_path = Path(workspace.config.direction(direction).target_model_path)
    try:
        tokenizer_hash = tokenizer_semantic_sha256(target_path)
    except (OSError, TypeError, ValueError) as exc:
        raise V5PipelineError("semantic tokenizer identity is unavailable") from exc
    if tokenizer_hash != workspace.config.tokenizer_sha256:
        raise V5PipelineError("semantic tokenizer identity changed")
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    result: dict[str, RiskPrefixTokenBinding] = {}
    for record, sample in samples:
        encoded = tokenizer(
            sample.prefix_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if encoded.numel() < record.token_bucket:
            raise V5PipelineError("semantic prefix has fewer tokens than registered")
        tokens = encoded[: record.token_bucket].long().tolist()
        binding = RiskPrefixTokenBinding(
            sample_id=record.sample_id,
            token_count=record.token_bucket,
            token_ids_sha256=token_ids_sha256(tokens),
        )
        errors = binding.validate(record)
        if errors:
            raise V5PipelineError("; ".join(errors))
        result[record.sample_id] = binding
    try:
        final_tokenizer_hash = tokenizer_semantic_sha256(target_path)
    except (OSError, TypeError, ValueError) as exc:
        raise V5PipelineError("semantic tokenizer identity changed during replay") from exc
    if final_tokenizer_hash != tokenizer_hash:
        raise V5PipelineError("semantic tokenizer identity changed during replay")
    return result


def _report_header(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    receipt: V5SemanticOpenReceipt,
    validation: V5ValidationManifest,
    validation_selective: SelectiveKVBridgeManifest,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
) -> dict[str, Any]:
    return {
        "schema_version": V5_SEMANTIC_REPORT_SCHEMA,
        "pipeline_id": workspace.config.pipeline_id,
        "direction": direction,
        "code_sha256": workspace.config.code_sha256,
        "benchmark_manifest_sha256": workspace.config.benchmark_manifest_sha256,
        "semantic_split_sha256": workspace.config.split_sha256["semantic_sealed_test"],
        "sealed_payload_sha256": receipt.sealed_payload_sha256,
        "semantic_open_receipt_sha256": receipt.content_sha256(),
        "validation_manifest_sha256": validation.content_sha256(),
        "validation_report_sha256": validation.validation_report_sha256,
        "validation_selective_artifact_id": validation_selective.artifact_id,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "risk_calibration_manifest_sha256": calibration.content_sha256(),
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "predictor_sha256": risk_fit.predictor.sha256,
        "threshold": calibration.risk_gate.threshold,
        "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
        "predictor_device": SEMANTIC_PREDICTOR_DEVICE,
        "opened_once": receipt.opened_once,
        "passed": True,
        "runtime_authority": False,
    }


def _write_semantic_checkpoint(
    path: Path,
    *,
    stage_binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    measurement: SemanticRiskMeasurement,
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_SEMANTIC_CHECKPOINT_SCHEMA,
            "stage_binding_sha256": stage_binding_sha256,
            "sample_id": benchmark_record.sample_id,
            "content_sha256": benchmark_record.content_sha256,
            "measurement": measurement.to_dict(),
        },
    )


def _load_semantic_checkpoint(
    path: Path,
    *,
    stage_binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    prefix_binding: RiskPrefixTokenBinding,
    expected_history: RiskHistory,
    predictor: RiskPredictor,
    risk_gate: Any,
) -> SemanticRiskMeasurement | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("semantic checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_SEMANTIC_CHECKPOINT_SCHEMA:
            raise V5PipelineError("semantic checkpoint schema mismatch")
        if payload.get("stage_binding_sha256") != stage_binding_sha256:
            raise V5PipelineError("semantic checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != benchmark_record.sample_id
            or payload.get("content_sha256") != benchmark_record.content_sha256
        ):
            raise V5PipelineError("semantic checkpoint sample binding mismatch")
        measurement = SemanticRiskMeasurement.from_dict(payload["measurement"])
        if measurement.prefix_binding != prefix_binding:
            raise V5PipelineError("semantic checkpoint prefix token identity changed")
        errors = measurement.validate(
            predictor=predictor,
            risk_gate=risk_gate,
            benchmark_record=benchmark_record,
            expected_history=expected_history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        return measurement
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("semantic checkpoint is malformed") from exc


def _gate_threshold(manifest: SelectiveKVBridgeManifest) -> float:
    threshold = manifest.risk_gate.threshold
    if threshold is None or not _finite_probability(threshold):
        raise V5PipelineError("semantic candidate requires a calibrated threshold")
    return threshold


def _required_threshold(calibration: V5RiskCalibrationManifest) -> float:
    threshold = calibration.risk_gate.threshold
    if threshold is None or not _finite_probability(threshold):
        raise V5PipelineError("semantic evaluation requires a calibrated threshold")
    return threshold


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
        raise V5PipelineError("semantic metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("semantic input file is unavailable") from exc
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


def stderr_semantic_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("semantic progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"semantic {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
