"""Independent risk-calibration collection and threshold freezing for v5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from collections.abc import Callable, Mapping
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
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_SCHEMA_VERSION,
    RiskCalibrationExample,
    RiskCalibrationResult,
    RiskGateError,
    RiskPredictor,
    select_calibrated_threshold,
)
from goldenexperience.size_variant.selective_manifest import RiskGateSpec
from goldenexperience.size_variant.v5_collect import (
    TraceObjectRef,
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
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskExampleEvaluator,
    RiskHistory,
    RiskTrainingExample,
    V5RiskFitManifest,
    load_completed_risk_fit,
    load_risk_predictor,
)

V5_RISK_CALIBRATION_CHECKPOINT_SCHEMA = "goldenexperience.v5_risk_calibration_checkpoint.v1"
V5_RISK_CALIBRATION_REPORT_SCHEMA = "goldenexperience.v5_risk_calibration_report.v1"
V5_RISK_CALIBRATION_MANIFEST_SCHEMA = "goldenexperience.v5_risk_calibration_manifest.v1"
RISK_CALIBRATION_MIN_ACCEPTED = 300
RISK_CALIBRATION_MAX_RISK_UPPER_BOUND = 0.01
RISK_CALIBRATION_CONFIDENCE = 0.95
RISK_CALIBRATION_PREDICTOR_DEVICE = "cpu"


@dataclass(frozen=True)
class RiskCalibrationMeasurement:
    example: RiskTrainingExample
    unsafe_probability: float

    def validate(
        self,
        *,
        predictor: RiskPredictor,
        benchmark_record: GroupedPrefixRecord | None = None,
        trace_record: TraceRecord | None = None,
        expected_history: RiskHistory | None = None,
    ) -> list[str]:
        errors = self.example.validate(
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=expected_history,
        )
        if not _finite_number(self.unsafe_probability) or not 0 <= self.unsafe_probability <= 1:
            errors.append("risk calibration probability is invalid")
            return errors
        try:
            expected_probability = predictor.unsafe_probability(self.example.features)
        except (RiskGateError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"risk calibration predictor failed: {type(exc).__name__}")
        else:
            if abs(self.unsafe_probability - expected_probability) > 1e-12:
                errors.append("risk calibration probability differs from the frozen predictor")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskCalibrationMeasurement:
        raw_example = dict(payload["example"])
        raw_example["features"] = tuple(raw_example["features"])
        return cls(
            example=RiskTrainingExample(**raw_example),
            unsafe_probability=payload["unsafe_probability"],
        )


@dataclass(frozen=True)
class V5RiskCalibrationManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    risk_calibration_split_sha256: str
    calibration_trace_manifest_sha256: str
    calibration_raw_store_sha256: str
    transport_fit_manifest_sha256: str
    transport_weights_sha256: str
    risk_fit_manifest_sha256: str
    calibration_report_sha256: str
    predictor: TraceObjectRef
    evaluator_sha256: str
    risk_gate: RiskGateSpec
    calibrated: bool = True
    schema_version: str = V5_RISK_CALIBRATION_MANIFEST_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
        risk_fit: V5RiskFitManifest,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_RISK_CALIBRATION_MANIFEST_SCHEMA:
            errors.append("unsupported v5 risk calibration manifest schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("risk calibration belongs to another pipeline")
        if self.direction != trace.direction or self.direction != risk_fit.direction:
            errors.append("risk calibration direction is inconsistent")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("risk calibration code hash mismatch")
        expected_split = workspace.config.split_sha256["risk_calibration"]
        if self.risk_calibration_split_sha256 != expected_split:
            errors.append("risk calibration split hash mismatch")
        if self.calibration_trace_manifest_sha256 != trace.content_sha256():
            errors.append("risk calibration trace hash mismatch")
        if self.calibration_raw_store_sha256 != trace.raw_sample_store_sha256:
            errors.append("risk calibration raw store hash mismatch")
        if self.transport_fit_manifest_sha256 != transport_manifest.content_sha256():
            errors.append("risk calibration transport manifest hash mismatch")
        if self.transport_weights_sha256 != candidate.weights.sha256:
            errors.append("risk calibration transport weights changed")
        if self.risk_fit_manifest_sha256 != risk_fit.content_sha256():
            errors.append("risk calibration uses another fitted predictor manifest")
        if not _is_sha256(self.calibration_report_sha256):
            errors.append("risk calibration report hash is invalid")
        if not _is_sha256(self.evaluator_sha256):
            errors.append("risk calibration evaluator hash is invalid")
        errors.extend(self.predictor.validate())
        if self.predictor != risk_fit.predictor:
            errors.append("risk calibration predictor reference changed")
        expected_gate_identity = (
            type(self.risk_gate.hidden_size) is int
            and type(self.risk_gate.min_shadow_samples) is int
            and self.risk_gate.predictor_uri == risk_fit.predictor.path
            and self.risk_gate.predictor_sha256 == risk_fit.predictor.sha256
            and self.risk_gate.calibration_dataset_sha256 == expected_split
            and self.risk_gate.feature_schema_version == RISK_FEATURE_SCHEMA_VERSION
            and self.risk_gate.hidden_size == risk_fit.training.hidden_size
            and self.risk_gate.ood_threshold == 6.0
            and self.risk_gate.min_shadow_samples == 1
        )
        if not expected_gate_identity:
            errors.append("risk calibration gate identity is inconsistent")
        try:
            errors.extend(
                self.risk_gate.calibration_errors(min_accepted=RISK_CALIBRATION_MIN_ACCEPTED)
            )
        except (TypeError, ValueError, RiskGateError) as exc:
            errors.append(f"risk calibration gate is malformed: {type(exc).__name__}")
        count_fields = (
            self.risk_gate.candidate_threshold_count,
            self.risk_gate.accepted_count,
            self.risk_gate.total_count,
            self.risk_gate.error_count,
        )
        if any(type(value) is not int for value in count_fields):
            errors.append("risk calibration gate counts must be integers")
        if (
            type(self.risk_gate.total_count) is not int
            or self.risk_gate.total_count != SPLIT_COUNTS["risk_calibration"]
        ):
            errors.append("risk calibration sample count is inconsistent")
        if not _finite_number(self.risk_gate.regression_risk_upper_bound) or (
            self.risk_gate.regression_risk_upper_bound > RISK_CALIBRATION_MAX_RISK_UPPER_BOUND
        ):
            errors.append("risk calibration upper bound exceeds the frozen contract")
        if (
            not _finite_number(self.risk_gate.confidence_level)
            or self.risk_gate.confidence_level != RISK_CALIBRATION_CONFIDENCE
        ):
            errors.append("risk calibration confidence differs from the frozen contract")
        if type(self.calibrated) is not bool or not self.calibrated:
            errors.append("risk calibration manifest must freeze a calibrated threshold")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5RiskCalibrationManifest:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            risk_calibration_split_sha256=str(payload["risk_calibration_split_sha256"]),
            calibration_trace_manifest_sha256=str(payload["calibration_trace_manifest_sha256"]),
            calibration_raw_store_sha256=str(payload["calibration_raw_store_sha256"]),
            transport_fit_manifest_sha256=str(payload["transport_fit_manifest_sha256"]),
            transport_weights_sha256=str(payload["transport_weights_sha256"]),
            risk_fit_manifest_sha256=str(payload["risk_fit_manifest_sha256"]),
            calibration_report_sha256=str(payload["calibration_report_sha256"]),
            predictor=TraceObjectRef(**payload["predictor"]),
            evaluator_sha256=str(payload["evaluator_sha256"]),
            risk_gate=RiskGateSpec(**payload["risk_gate"]),
            calibrated=payload.get("calibrated", False),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
        risk_fit: V5RiskFitManifest,
    ) -> V5RiskCalibrationManifest:
        try:
            value = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = value.validate(
                workspace=workspace,
                trace=trace,
                transport_manifest=transport_manifest,
                candidate=candidate,
                risk_fit=risk_fit,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("risk calibration manifest is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        return value


def run_calibrate_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], RiskExampleEvaluator],
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Freeze one threshold using only the independent risk-calibration split."""

    benchmark = load_bound_benchmark(workspace)
    risk_fit, _, transport_manifest, candidate = load_completed_risk_fit(workspace, direction)
    trace = load_completed_trace_manifest(workspace, direction, "risk_calibration", benchmark)
    store_path = Path(sample_store_path)
    before = _file_signature(store_path)
    store_sha256 = sha256_file(store_path)
    if store_sha256 != trace.raw_sample_store_sha256:
        raise V5PipelineError("risk-calibration raw store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="risk_calibration")
    traces = {item.sample_id: item for item in trace.records}
    if set(traces) != {record.sample_id for record, _ in samples}:
        raise V5PipelineError("risk-calibration samples differ from trace manifest")
    evaluator_payload = dict(evaluator_parameters)
    evaluator_sha256 = _sha256_bytes(_canonical_json_bytes(evaluator_payload))
    stage_parameters = {
        "calibration_trace_manifest_sha256": trace.content_sha256(),
        "raw_sample_store_sha256": store_sha256,
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "risk_fit_manifest_sha256": risk_fit.content_sha256(),
        "predictor_sha256": risk_fit.predictor.sha256,
        "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
        "calibration_method": RISK_CALIBRATION_METHOD,
        "min_accepted": RISK_CALIBRATION_MIN_ACCEPTED,
        "max_risk_upper_bound": RISK_CALIBRATION_MAX_RISK_UPPER_BOUND,
        "confidence": RISK_CALIBRATION_CONFIDENCE,
        "predictor_device": RISK_CALIBRATION_PREDICTOR_DEVICE,
        "evaluator": evaluator_payload,
    }
    lease = workspace.begin_stage(
        direction,
        "calibrate",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/calibrate"]
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=RISK_CALIBRATION_PREDICTOR_DEVICE,
    )
    work = workspace.control / "work" / direction / "calibrate"
    checkpoint_dir = work / "examples"
    histories: dict[str, RiskHistory] = {}
    measurements: list[RiskCalibrationMeasurement] = []
    try:
        with evaluator_factory() as evaluator:
            for index, (benchmark_record, sample) in enumerate(samples, start=1):
                trace_record = traces[benchmark_record.sample_id]
                history = histories.get(benchmark_record.prefix_group_id, RiskHistory())
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                measurement = _load_calibration_checkpoint(
                    checkpoint,
                    binding_sha256=lease.input_sha256,
                    benchmark_record=benchmark_record,
                    trace_record=trace_record,
                    expected_history=history,
                    predictor=predictor,
                )
                if measurement is None:
                    example = evaluator.evaluate(
                        benchmark_record,
                        trace_record,
                        sample,
                        history,
                    )
                    measurement = RiskCalibrationMeasurement(
                        example=example,
                        unsafe_probability=predictor.unsafe_probability(example.features),
                    )
                    errors = measurement.validate(
                        predictor=predictor,
                        benchmark_record=benchmark_record,
                        trace_record=trace_record,
                        expected_history=history,
                    )
                    if errors:
                        raise V5PipelineError("; ".join(errors))
                    _write_calibration_checkpoint(
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
            raise V5PipelineError("risk-calibration raw store changed during calibration")
        result = select_calibrated_threshold(
            (
                RiskCalibrationExample(item.unsafe_probability, item.example.unsafe)
                for item in measurements
            ),
            min_accepted=RISK_CALIBRATION_MIN_ACCEPTED,
            max_risk_upper_bound=RISK_CALIBRATION_MAX_RISK_UPPER_BOUND,
            confidence=RISK_CALIBRATION_CONFIDENCE,
        )
        report = {
            "schema_version": V5_RISK_CALIBRATION_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": direction,
            "code_sha256": workspace.config.code_sha256,
            "risk_calibration_split_sha256": workspace.config.split_sha256["risk_calibration"],
            "calibration_trace_manifest_sha256": trace.content_sha256(),
            "calibration_raw_store_sha256": store_sha256,
            "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "predictor_sha256": risk_fit.predictor.sha256,
            "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "predictor_device": RISK_CALIBRATION_PREDICTOR_DEVICE,
            "evaluator": evaluator_payload,
            "measurements": [item.to_dict() for item in measurements],
            "result": asdict(result),
        }
        report_path = work / "risk_calibration_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        risk_gate = RiskGateSpec(
            predictor_uri=risk_fit.predictor.path,
            predictor_sha256=risk_fit.predictor.sha256,
            threshold=result.threshold,
            calibration_dataset_sha256=workspace.config.split_sha256["risk_calibration"],
            calibration_method=result.calibration_method,
            candidate_threshold_count=result.candidate_threshold_count,
            accepted_count=result.accepted_count,
            total_count=result.total_count,
            error_count=result.error_count,
            coverage=result.coverage,
            regression_risk_upper_bound=result.regression_risk_upper_bound,
            confidence_level=result.confidence_level,
        )
        manifest = V5RiskCalibrationManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            risk_calibration_split_sha256=workspace.config.split_sha256["risk_calibration"],
            calibration_trace_manifest_sha256=trace.content_sha256(),
            calibration_raw_store_sha256=store_sha256,
            transport_fit_manifest_sha256=transport_manifest.content_sha256(),
            transport_weights_sha256=candidate.weights.sha256,
            risk_fit_manifest_sha256=risk_fit.content_sha256(),
            calibration_report_sha256=report_sha256,
            predictor=risk_fit.predictor,
            evaluator_sha256=evaluator_sha256,
            risk_gate=risk_gate,
        )
        errors = manifest.validate(
            workspace=workspace,
            trace=trace,
            transport_manifest=transport_manifest,
            candidate=candidate,
            risk_fit=risk_fit,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        manifest_path = work / "risk_calibration_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "risk_calibration_report": report_path,
                "risk_calibration_manifest": manifest_path,
            },
            metadata={
                "threshold": result.threshold,
                "accepted_count": result.accepted_count,
                "total_count": result.total_count,
                "error_count": result.error_count,
                "coverage": result.coverage,
                "regression_risk_upper_bound": result.regression_risk_upper_bound,
                "candidate_threshold_count": result.candidate_threshold_count,
                "risk_calibration_manifest_sha256": manifest.content_sha256(),
                "calibrated": True,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def load_completed_risk_calibration(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[
    V5RiskCalibrationManifest,
    V5RiskFitManifest,
    V5TraceManifest,
    V5TransportFitManifest | V5DirectionalTransportFitManifest,
    TransportCandidateArtifact,
]:
    benchmark = load_bound_benchmark(workspace)
    risk_fit, _, transport_manifest, candidate = load_completed_risk_fit(workspace, direction)
    trace = load_completed_trace_manifest(workspace, direction, "risk_calibration", benchmark)
    state = workspace.state()
    stage = state.stages.get(f"{direction}/calibrate")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed independent risk calibration")
    manifest_artifact = stage.outputs.get("risk_calibration_manifest")
    report_artifact = stage.outputs.get("risk_calibration_report")
    if manifest_artifact is None or report_artifact is None:
        raise V5PipelineError("risk calibration stage lacks its manifest or report")
    manifest_path = workspace.artifact_path(manifest_artifact, verify_hash=True)
    report_path = workspace.artifact_path(report_artifact, verify_hash=True)
    manifest = V5RiskCalibrationManifest.load(
        manifest_path,
        workspace=workspace,
        trace=trace,
        transport_manifest=transport_manifest,
        candidate=candidate,
        risk_fit=risk_fit,
    )
    if manifest.calibration_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("risk calibration manifest refers to another report")
    predictor = load_risk_predictor(
        workspace,
        risk_fit,
        device=RISK_CALIBRATION_PREDICTOR_DEVICE,
    )
    _load_and_validate_calibration_report(
        report_path,
        benchmark=benchmark,
        workspace=workspace,
        trace=trace,
        transport_manifest=transport_manifest,
        candidate=candidate,
        risk_fit=risk_fit,
        manifest=manifest,
        predictor=predictor,
    )
    return manifest, risk_fit, trace, transport_manifest, candidate


def _load_and_validate_calibration_report(
    path: Path,
    *,
    benchmark: PublicationBenchmarkManifest,
    workspace: V5PipelineWorkspace,
    trace: V5TraceManifest,
    transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
    candidate: TransportCandidateArtifact,
    risk_fit: V5RiskFitManifest,
    manifest: V5RiskCalibrationManifest,
    predictor: RiskPredictor,
) -> tuple[tuple[RiskCalibrationMeasurement, ...], RiskCalibrationResult]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_header = {
            "schema_version": V5_RISK_CALIBRATION_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": trace.direction,
            "code_sha256": workspace.config.code_sha256,
            "risk_calibration_split_sha256": workspace.config.split_sha256["risk_calibration"],
            "calibration_trace_manifest_sha256": trace.content_sha256(),
            "calibration_raw_store_sha256": trace.raw_sample_store_sha256,
            "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "risk_fit_manifest_sha256": risk_fit.content_sha256(),
            "predictor_sha256": risk_fit.predictor.sha256,
            "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
            "predictor_device": RISK_CALIBRATION_PREDICTOR_DEVICE,
        }
        if any(payload.get(name) != value for name, value in expected_header.items()):
            raise V5PipelineError("risk calibration report header binding changed")
        evaluator = payload["evaluator"]
        if not isinstance(evaluator, dict):
            raise V5PipelineError("risk calibration evaluator metadata is malformed")
        if _sha256_bytes(_canonical_json_bytes(evaluator)) != manifest.evaluator_sha256:
            raise V5PipelineError("risk calibration evaluator metadata changed")
        measurements = tuple(
            RiskCalibrationMeasurement.from_dict(item) for item in payload["measurements"]
        )
        result = RiskCalibrationResult(**payload["result"])
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("risk calibration report is unreadable or malformed") from exc
    result_errors = _calibration_result_errors(result)
    if result_errors:
        raise V5PipelineError("; ".join(result_errors))
    expected_records = sorted(
        (item for item in benchmark.records if item.split == "risk_calibration"),
        key=lambda item: item.sample_id,
    )
    trace_by_id = {item.sample_id: item for item in trace.records}
    if len(measurements) != SPLIT_COUNTS["risk_calibration"] or len(measurements) != len(
        expected_records
    ):
        raise V5PipelineError("risk calibration report measurement count is inconsistent")
    histories: dict[str, RiskHistory] = {}
    for measurement, benchmark_record in zip(measurements, expected_records, strict=True):
        if measurement.example.sample_id != benchmark_record.sample_id:
            raise V5PipelineError("risk calibration report order or sample identity changed")
        trace_record = trace_by_id.get(benchmark_record.sample_id)
        if trace_record is None:
            raise V5PipelineError("risk calibration report refers to a missing trace")
        history = histories.get(benchmark_record.prefix_group_id, RiskHistory())
        errors = measurement.validate(
            predictor=predictor,
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        histories[benchmark_record.prefix_group_id] = history.update(measurement.example)
    expected_result = select_calibrated_threshold(
        (
            RiskCalibrationExample(item.unsafe_probability, item.example.unsafe)
            for item in measurements
        ),
        min_accepted=RISK_CALIBRATION_MIN_ACCEPTED,
        max_risk_upper_bound=RISK_CALIBRATION_MAX_RISK_UPPER_BOUND,
        confidence=RISK_CALIBRATION_CONFIDENCE,
    )
    if result != expected_result or asdict(result) != {
        "threshold": manifest.risk_gate.threshold,
        "accepted_count": manifest.risk_gate.accepted_count,
        "total_count": manifest.risk_gate.total_count,
        "error_count": manifest.risk_gate.error_count,
        "coverage": manifest.risk_gate.coverage,
        "regression_risk_upper_bound": manifest.risk_gate.regression_risk_upper_bound,
        "confidence_level": manifest.risk_gate.confidence_level,
        "calibration_method": manifest.risk_gate.calibration_method,
        "candidate_threshold_count": manifest.risk_gate.candidate_threshold_count,
    }:
        raise V5PipelineError("risk calibration result differs from the detailed report")
    return measurements, result


def _write_calibration_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    measurement: RiskCalibrationMeasurement,
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_RISK_CALIBRATION_CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "sample_id": benchmark_record.sample_id,
            "content_sha256": benchmark_record.content_sha256,
            "measurement": measurement.to_dict(),
        },
    )


def _load_calibration_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    trace_record: TraceRecord,
    expected_history: RiskHistory,
    predictor: RiskPredictor,
) -> RiskCalibrationMeasurement | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("risk calibration checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_RISK_CALIBRATION_CHECKPOINT_SCHEMA:
            raise V5PipelineError("risk calibration checkpoint schema mismatch")
        if payload.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("risk calibration checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != benchmark_record.sample_id
            or payload.get("content_sha256") != benchmark_record.content_sha256
        ):
            raise V5PipelineError("risk calibration checkpoint sample binding mismatch")
        measurement = RiskCalibrationMeasurement.from_dict(payload["measurement"])
        errors = measurement.validate(
            predictor=predictor,
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
        raise V5PipelineError("risk calibration checkpoint is malformed") from exc


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
        raise V5PipelineError("risk calibration metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("risk calibration input file is unavailable") from exc
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


def _calibration_result_errors(result: RiskCalibrationResult) -> list[str]:
    errors: list[str] = []
    for name in ("accepted_count", "total_count", "error_count", "candidate_threshold_count"):
        if type(getattr(result, name)) is not int:
            errors.append(f"risk calibration result {name} must be an integer")
    for name in (
        "threshold",
        "coverage",
        "regression_risk_upper_bound",
        "confidence_level",
    ):
        if not _finite_number(getattr(result, name)):
            errors.append(f"risk calibration result {name} must be finite")
    if result.calibration_method != RISK_CALIBRATION_METHOD:
        errors.append("risk calibration result method changed")
    return errors


def stderr_calibration_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("risk calibration progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(
                f"risk calibration {index}/{total}: {sample_id}",
                file=sys.stderr,
                flush=True,
            )

    return report
