"""Selector-train-only risk example collection and predictor fitting for v5."""

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
from typing import Any, Protocol

from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.size_variant.cached_kv_manifest import (
    canonicalize_safetensors_header,
    sha256_file,
)
from goldenexperience.size_variant.risk_gate import (
    RISK_FEATURE_DIM,
    RISK_FEATURE_SCHEMA_VERSION,
    RiskPredictor,
    fit_risk_predictor,
    unsafe_label,
)
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    load_bound_benchmark,
    load_completed_trace_manifest,
    load_raw_sample_store,
)
from goldenexperience.size_variant.v5_directional_fit import (
    V5DirectionalTransportFitManifest,
    load_completed_directional_fit,
)
from goldenexperience.size_variant.v5_fit import (
    SCREENING_DIRECTION,
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

V5_RISK_EXAMPLE_CHECKPOINT_SCHEMA = "goldenexperience.v5_risk_example_checkpoint.v1"
V5_RISK_TRAINING_REPORT_SCHEMA = "goldenexperience.v5_risk_training_report.v1"
V5_RISK_FIT_SCHEMA = "goldenexperience.v5_risk_fit.v1"
RISK_LABEL_GENERATION_TOKENS = 16


@dataclass(frozen=True)
class RiskPrefixTokenBinding:
    """Minimal prefix identity required by risk evaluation without a trace shard."""

    sample_id: str
    token_count: int
    token_ids_sha256: str

    def validate(self, record: GroupedPrefixRecord) -> list[str]:
        errors: list[str] = []
        if self.sample_id != record.sample_id:
            errors.append("risk prefix binding sample identity changed")
        if type(self.token_count) is not int or self.token_count != record.token_bucket:
            errors.append("risk prefix binding token count changed")
        if not _is_sha256(self.token_ids_sha256):
            errors.append("risk prefix binding token hash is invalid")
        return errors


@dataclass(frozen=True)
class RiskTrainingParameters:
    seed: int = 17
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 64
    feature_schema_version: str = RISK_FEATURE_SCHEMA_VERSION
    feature_dim: int = RISK_FEATURE_DIM

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self != RiskTrainingParameters():
            errors.append("risk predictor training differs from the frozen contract")
        if type(self.seed) is not int or type(self.epochs) is not int or self.epochs <= 0:
            errors.append("risk predictor seed or epoch count is invalid")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not _finite_number(value) or value < 0:
                errors.append(f"risk predictor {name} is invalid")
        if self.learning_rate == 0:
            errors.append("risk predictor learning rate must be positive")
        if self.hidden_size != 64 or self.feature_dim != RISK_FEATURE_DIM:
            errors.append("risk predictor dimensions changed")
        if self.feature_schema_version != RISK_FEATURE_SCHEMA_VERSION:
            errors.append("risk predictor feature schema changed")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskHistory:
    samples: int = 0
    failures: int = 0
    greedy_agreement_sum: float = 0.0

    @property
    def greedy_agreement(self) -> float:
        return self.greedy_agreement_sum / self.samples if self.samples else 1.0

    def update(self, example: RiskTrainingExample) -> RiskHistory:
        return RiskHistory(
            samples=self.samples + 1,
            failures=self.failures + int(example.unsafe),
            greedy_agreement_sum=self.greedy_agreement_sum + example.greedy_agreement,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        counts_valid = type(self.samples) is int and type(self.failures) is int
        if not counts_valid or self.samples < 0 or not 0 <= self.failures <= self.samples:
            errors.append("risk history counts are invalid")
        agreement_valid = _finite_number(self.greedy_agreement_sum)
        if not agreement_valid or (
            counts_valid and not 0 <= self.greedy_agreement_sum <= self.samples
        ):
            errors.append("risk history agreement sum is invalid")
        return errors


@dataclass(frozen=True)
class RiskTrainingExample:
    sample_id: str
    prefix_group_id: str
    features: tuple[float, ...]
    unsafe: bool
    native_task_score: float
    bridge_task_score: float
    task_pass_threshold: float
    greedy_matches: int
    greedy_tokens: int
    native_nll: float
    bridge_nll: float
    teacher_tokens: int
    key_cosine: float
    history_samples: int
    history_failures: int
    history_greedy_agreement: float
    sidecar_sha256: str
    native_prediction_sha256: str
    bridge_prediction_sha256: str
    native_tokens_sha256: str
    bridge_tokens_sha256: str

    @property
    def greedy_agreement(self) -> float:
        if (
            type(self.greedy_matches) is not int
            or type(self.greedy_tokens) is not int
            or self.greedy_tokens <= 0
        ):
            return math.nan
        return self.greedy_matches / self.greedy_tokens

    @property
    def perplexity_drift_pct(self) -> float:
        if (
            type(self.teacher_tokens) is not int
            or self.teacher_tokens <= 0
            or not _finite_number(self.native_nll)
            or not _finite_number(self.bridge_nll)
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
            if type(self.history_samples) is int and _finite_number(self.history_greedy_agreement)
            else math.nan
        )
        return RiskHistory(
            samples=self.history_samples,
            failures=self.history_failures,
            greedy_agreement_sum=agreement_sum,
        )

    def validate(
        self,
        *,
        benchmark_record: GroupedPrefixRecord | None = None,
        trace_record: TraceRecord | RiskPrefixTokenBinding | None = None,
        expected_history: RiskHistory | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or not isinstance(self.prefix_group_id, str)
            or not self.prefix_group_id
        ):
            errors.append("risk example identifiers are required")
        if benchmark_record is not None and (
            self.sample_id != benchmark_record.sample_id
            or self.prefix_group_id != benchmark_record.prefix_group_id
        ):
            errors.append("risk example benchmark binding changed")
        if trace_record is not None and self.sample_id != trace_record.sample_id:
            errors.append("risk example trace binding changed")
        features_valid = (
            isinstance(self.features, tuple)
            and len(self.features) == RISK_FEATURE_DIM
            and all(_finite_number(value) for value in self.features)
        )
        if not features_valid:
            errors.append("risk example feature vector is invalid")
        if not _strict_bool(self.unsafe):
            errors.append("risk example label must be boolean")
        score_valid: dict[str, bool] = {}
        for name in ("native_task_score", "bridge_task_score", "task_pass_threshold"):
            value = getattr(self, name)
            score_valid[name] = _finite_number(value) and 0 <= value <= 1
            if not score_valid[name]:
                errors.append(f"risk example {name} is invalid")
        greedy_counts_valid = (
            type(self.greedy_matches) is int
            and type(self.greedy_tokens) is int
            and self.greedy_tokens == RISK_LABEL_GENERATION_TOKENS
            and 0 <= self.greedy_matches <= self.greedy_tokens
        )
        if not greedy_counts_valid:
            errors.append("risk example greedy counts are invalid")
        teacher_count_valid = (
            type(self.teacher_tokens) is int and self.teacher_tokens == RISK_LABEL_GENERATION_TOKENS
        )
        if not teacher_count_valid:
            errors.append("risk example teacher token count is invalid")
        nll_valid: dict[str, bool] = {}
        for name in ("native_nll", "bridge_nll"):
            value = getattr(self, name)
            nll_valid[name] = _finite_number(value) and value >= 0
            if not nll_valid[name]:
                errors.append(f"risk example {name} is invalid")
        drift_valid = teacher_count_valid and all(nll_valid.values())
        if drift_valid and not math.isfinite(self.perplexity_drift_pct):
            errors.append("risk example perplexity drift is non-finite")
            drift_valid = False
        if not _finite_number(self.key_cosine) or not -1 <= self.key_cosine <= 1:
            errors.append("risk example key cosine is invalid")
        history_fields_valid = (
            type(self.history_samples) is int
            and type(self.history_failures) is int
            and self.history_samples >= 0
            and 0 <= self.history_failures <= self.history_samples
            and _finite_number(self.history_greedy_agreement)
            and 0 <= self.history_greedy_agreement <= 1
            and (self.history_samples > 0 or self.history_greedy_agreement == 1.0)
        )
        history = self.history()
        errors.extend(history.validate())
        if not history_fields_valid and not history.validate():
            errors.append("risk example history is invalid")
        expected_history_valid = expected_history is not None and not expected_history.validate()
        if expected_history is not None and (
            not expected_history_valid
            or not history_fields_valid
            or self.history_samples != expected_history.samples
            or self.history_failures != expected_history.failures
            or abs(self.history_greedy_agreement - expected_history.greedy_agreement) > 1e-12
        ):
            errors.append("risk example uses non-causal history")
        for name in (
            "sidecar_sha256",
            "native_prediction_sha256",
            "bridge_prediction_sha256",
            "native_tokens_sha256",
            "bridge_tokens_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                errors.append(f"risk example {name} is invalid")
        label_inputs_valid = (
            _strict_bool(self.unsafe)
            and all(score_valid.values())
            and greedy_counts_valid
            and teacher_count_valid
            and all(nll_valid.values())
            and drift_valid
        )
        if label_inputs_valid:
            expected_unsafe = unsafe_label(
                native_task_passed=self.native_task_score >= self.task_pass_threshold,
                bridge_task_passed=self.bridge_task_score >= self.task_pass_threshold,
                greedy_agreement=self.greedy_agreement,
                perplexity_drift_pct=self.perplexity_drift_pct,
            )
            if self.unsafe != expected_unsafe:
                errors.append("risk example unsafe label is inconsistent")
        return errors


class RiskExampleEvaluator(Protocol):
    def __enter__(self) -> RiskExampleEvaluator: ...

    def __exit__(self, *_args: object) -> None: ...

    def evaluate(
        self,
        benchmark_record: GroupedPrefixRecord,
        trace_record: TraceRecord | RiskPrefixTokenBinding,
        sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample: ...


@dataclass(frozen=True)
class RiskTrainingMetrics:
    training_objective: float
    log_loss: float
    accuracy_at_half: float
    roc_auc: float

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in ("training_objective", "log_loss"):
            value = getattr(self, name)
            if not _finite_number(value) or value < 0:
                errors.append(f"risk predictor {name} is invalid")
        for name in ("accuracy_at_half", "roc_auc"):
            value = getattr(self, name)
            if not _finite_number(value) or not 0 <= value <= 1:
                errors.append(f"risk predictor {name} is invalid")
        return errors


@dataclass(frozen=True)
class V5RiskFitManifest:
    pipeline_id: str
    direction: str
    code_sha256: str
    selector_train_split_sha256: str
    selector_trace_manifest_sha256: str
    selector_raw_store_sha256: str
    transport_fit_manifest_sha256: str
    transport_weights_sha256: str
    risk_training_report_sha256: str
    predictor: TraceObjectRef
    training: RiskTrainingParameters
    metrics: RiskTrainingMetrics
    sample_count: int
    unsafe_count: int
    safe_count: int
    calibrated: bool = False
    schema_version: str = V5_RISK_FIT_SCHEMA

    def validate(
        self,
        *,
        workspace: V5PipelineWorkspace,
        trace: V5TraceManifest,
        transport_manifest: V5TransportFitManifest | V5DirectionalTransportFitManifest,
        candidate: TransportCandidateArtifact,
    ) -> list[str]:
        errors: list[str] = []
        if self.schema_version != V5_RISK_FIT_SCHEMA:
            errors.append("unsupported v5 risk fit schema")
        if self.pipeline_id != workspace.config.pipeline_id:
            errors.append("risk fit belongs to another pipeline")
        if self.direction != trace.direction:
            errors.append("risk fit direction differs from selector traces")
        try:
            workspace.config.direction(self.direction)
        except V5PipelineError as exc:
            errors.append(str(exc))
        if self.code_sha256 != workspace.config.code_sha256:
            errors.append("risk fit code hash mismatch")
        if self.selector_train_split_sha256 != workspace.config.split_sha256["selector_train"]:
            errors.append("risk fit selector split hash mismatch")
        if self.selector_trace_manifest_sha256 != trace.content_sha256():
            errors.append("risk fit selector trace hash mismatch")
        if self.selector_raw_store_sha256 != trace.raw_sample_store_sha256:
            errors.append("risk fit raw sample hash mismatch")
        if self.transport_fit_manifest_sha256 != transport_manifest.content_sha256():
            errors.append("risk fit transport manifest hash mismatch")
        if self.transport_weights_sha256 != candidate.weights.sha256:
            errors.append("risk fit transport weights changed")
        if not _is_sha256(self.risk_training_report_sha256):
            errors.append("risk fit report hash is invalid")
        errors.extend(self.predictor.validate())
        if Path(self.predictor.path).suffix != ".safetensors":
            errors.append("risk predictor must use safetensors")
        errors.extend(self.training.validate())
        errors.extend(self.metrics.validate())
        expected_samples = SPLIT_COUNTS["selector_train"]
        if type(self.sample_count) is not int or self.sample_count != expected_samples:
            errors.append("risk fit sample count is inconsistent")
        if (
            type(self.unsafe_count) is not int
            or type(self.safe_count) is not int
            or self.unsafe_count <= 0
            or self.safe_count <= 0
            or self.unsafe_count + self.safe_count != self.sample_count
        ):
            errors.append("risk fit requires both safe and unsafe selector examples")
        if self.calibrated:
            errors.append("selector-train risk fit cannot carry a calibrated threshold")
        return errors

    def content_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> V5RiskFitManifest:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            direction=str(payload["direction"]),
            code_sha256=str(payload["code_sha256"]),
            selector_train_split_sha256=str(payload["selector_train_split_sha256"]),
            selector_trace_manifest_sha256=str(payload["selector_trace_manifest_sha256"]),
            selector_raw_store_sha256=str(payload["selector_raw_store_sha256"]),
            transport_fit_manifest_sha256=str(payload["transport_fit_manifest_sha256"]),
            transport_weights_sha256=str(payload["transport_weights_sha256"]),
            risk_training_report_sha256=str(payload["risk_training_report_sha256"]),
            predictor=TraceObjectRef(**payload["predictor"]),
            training=RiskTrainingParameters(**payload["training"]),
            metrics=RiskTrainingMetrics(**payload["metrics"]),
            sample_count=payload["sample_count"],
            unsafe_count=payload["unsafe_count"],
            safe_count=payload["safe_count"],
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
    ) -> V5RiskFitManifest:
        try:
            value = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            errors = value.validate(
                workspace=workspace,
                trace=trace,
                transport_manifest=transport_manifest,
                candidate=candidate,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V5PipelineError("risk fit manifest is unreadable or malformed") from exc
        if errors:
            raise V5PipelineError("; ".join(errors))
        predictor_path = _verify_workspace_ref(workspace, value.predictor)
        try:
            RiskPredictor.from_artifact(
                predictor_path,
                expected_sha256=value.predictor.sha256,
            )
        except Exception as exc:
            raise V5PipelineError("risk predictor artifact contract is invalid") from exc
        return value


def load_deployment_transport_binding(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[
    V5TransportFitManifest | V5DirectionalTransportFitManifest,
    TransportCandidateArtifact,
    FrozenTransportStructure,
]:
    benchmark = load_bound_benchmark(workspace)
    if direction == SCREENING_DIRECTION:
        structure, fit, _ = load_frozen_transport_structure(workspace)
        candidates = [
            item
            for item in fit.candidates
            if item.rank == structure.selected_rank and item.seed == structure.deployment_seed
        ]
        if len(candidates) != 1:
            raise V5PipelineError("screening fit lacks its frozen deployment candidate")
        return fit, candidates[0], structure
    manifest, _, structure = load_completed_directional_fit(workspace, direction, benchmark)
    if len(manifest.candidates) != 1:
        raise V5PipelineError("directional fit lacks one deployment candidate")
    return manifest, manifest.candidates[0], structure


def run_fit_risk_stage(
    *,
    workspace: V5PipelineWorkspace,
    direction: str,
    sample_store_path: str | Path,
    evaluator_parameters: Mapping[str, Any],
    evaluator_factory: Callable[[], RiskExampleEvaluator],
    predictor_device: str = "cpu",
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> PipelineStageRecord:
    """Collect causal source-side examples and fit an uncalibrated risk ranker."""

    benchmark = load_bound_benchmark(workspace)
    transport_manifest, candidate, _ = load_deployment_transport_binding(workspace, direction)
    trace = load_completed_trace_manifest(workspace, direction, "selector_train", benchmark)
    store_path = Path(sample_store_path)
    before = _file_signature(store_path)
    store_sha256 = sha256_file(store_path)
    if store_sha256 != trace.raw_sample_store_sha256:
        raise V5PipelineError("selector-train raw store differs from collected traces")
    samples = load_raw_sample_store(store_path, benchmark, split="selector_train")
    traces = {item.sample_id: item for item in trace.records}
    if set(traces) != {record.sample_id for record, _ in samples}:
        raise V5PipelineError("selector-train samples differ from trace manifest")
    training = RiskTrainingParameters()
    stage_parameters = {
        "selector_trace_manifest_sha256": trace.content_sha256(),
        "raw_sample_store_sha256": store_sha256,
        "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
        "transport_weights_sha256": candidate.weights.sha256,
        "label_generation_tokens": RISK_LABEL_GENERATION_TOKENS,
        "training": training.to_dict(),
        "metrics_device": "cpu",
        "evaluator": dict(evaluator_parameters),
    }
    lease = workspace.begin_stage(
        direction,
        "fit_risk",
        parameters=stage_parameters,
        resume=resume,
    )
    if lease.reused:
        return workspace.state().stages[f"{direction}/fit_risk"]
    work = workspace.control / "work" / direction / "fit_risk"
    checkpoint_dir = work / "examples"
    histories: dict[str, RiskHistory] = {}
    examples: list[RiskTrainingExample] = []
    try:
        with evaluator_factory() as evaluator:
            for index, (benchmark_record, sample) in enumerate(samples, start=1):
                trace_record = traces[benchmark_record.sample_id]
                history = histories.get(benchmark_record.prefix_group_id, RiskHistory())
                checkpoint = checkpoint_dir / f"{_sha256_text(sample.sample_id)}.json"
                example = _load_risk_checkpoint(
                    checkpoint,
                    binding_sha256=lease.input_sha256,
                    benchmark_record=benchmark_record,
                    trace_record=trace_record,
                    expected_history=history,
                )
                if example is None:
                    example = evaluator.evaluate(
                        benchmark_record,
                        trace_record,
                        sample,
                        history,
                    )
                    errors = example.validate(
                        benchmark_record=benchmark_record,
                        trace_record=trace_record,
                        expected_history=history,
                    )
                    if errors:
                        raise V5PipelineError("; ".join(errors))
                    _write_risk_checkpoint(
                        checkpoint,
                        binding_sha256=lease.input_sha256,
                        benchmark_record=benchmark_record,
                        example=example,
                    )
                histories[benchmark_record.prefix_group_id] = history.update(example)
                examples.append(example)
                if progress is not None:
                    progress(index, len(samples), sample.sample_id)
        if _file_signature(store_path) != before or sha256_file(store_path) != store_sha256:
            raise V5PipelineError("selector-train raw store changed during risk fitting")
        unsafe_count = sum(item.unsafe for item in examples)
        safe_count = len(examples) - unsafe_count
        if unsafe_count <= 0 or safe_count <= 0:
            raise V5PipelineError("risk fitting requires both safe and unsafe selector examples")
        feature_rows = [item.features for item in examples]
        labels = [int(item.unsafe) for item in examples]
        state = fit_risk_predictor(
            feature_rows,
            labels,
            seed=training.seed,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            device=predictor_device,
        )
        predictor = RiskPredictor(state)
        probabilities = [predictor.unsafe_probability(row) for row in feature_rows]
        metrics = _risk_training_metrics(probabilities, labels)
        report = {
            "schema_version": V5_RISK_TRAINING_REPORT_SCHEMA,
            "pipeline_id": workspace.config.pipeline_id,
            "direction": direction,
            "code_sha256": workspace.config.code_sha256,
            "selector_train_split_sha256": workspace.config.split_sha256["selector_train"],
            "selector_trace_manifest_sha256": trace.content_sha256(),
            "selector_raw_store_sha256": store_sha256,
            "transport_fit_manifest_sha256": transport_manifest.content_sha256(),
            "transport_weights_sha256": candidate.weights.sha256,
            "training": training.to_dict(),
            "metrics_device": "cpu",
            "evaluator": dict(evaluator_parameters),
            "examples": [asdict(item) for item in examples],
            "metrics": asdict(metrics),
        }
        report_path = work / "risk_training_report.json"
        _write_json_replace(report_path, report)
        report_sha256 = sha256_file(report_path)
        predictor_path = work / "risk_predictor.safetensors"
        _save_predictor(predictor_path, state)
        predictor_artifact = workspace.publish_file(
            predictor_path,
            logical_name="risk_predictor",
        )
        manifest = V5RiskFitManifest(
            pipeline_id=workspace.config.pipeline_id,
            direction=direction,
            code_sha256=workspace.config.code_sha256,
            selector_train_split_sha256=workspace.config.split_sha256["selector_train"],
            selector_trace_manifest_sha256=trace.content_sha256(),
            selector_raw_store_sha256=store_sha256,
            transport_fit_manifest_sha256=transport_manifest.content_sha256(),
            transport_weights_sha256=candidate.weights.sha256,
            risk_training_report_sha256=report_sha256,
            predictor=TraceObjectRef.from_artifact(predictor_artifact),
            training=training,
            metrics=metrics,
            sample_count=len(examples),
            unsafe_count=unsafe_count,
            safe_count=safe_count,
        )
        manifest_errors = manifest.validate(
            workspace=workspace,
            trace=trace,
            transport_manifest=transport_manifest,
            candidate=candidate,
        )
        if manifest_errors:
            raise V5PipelineError("; ".join(manifest_errors))
        manifest_path = work / "risk_fit_manifest.json"
        _write_json_replace(manifest_path, manifest.to_dict())
        return workspace.complete_stage(
            lease,
            outputs={
                "risk_training_report": report_path,
                "risk_fit_manifest": manifest_path,
            },
            metadata={
                "sample_count": len(examples),
                "unsafe_count": unsafe_count,
                "safe_count": safe_count,
                "predictor_sha256": predictor_artifact.sha256,
                "risk_fit_manifest_sha256": manifest.content_sha256(),
                "calibrated": False,
            },
        )
    except Exception as exc:
        with suppress(V5PipelineError):
            workspace.fail_stage(lease, exc)
        raise


def load_completed_risk_fit(
    workspace: V5PipelineWorkspace,
    direction: str,
) -> tuple[
    V5RiskFitManifest,
    V5TraceManifest,
    V5TransportFitManifest | V5DirectionalTransportFitManifest,
    TransportCandidateArtifact,
]:
    benchmark = load_bound_benchmark(workspace)
    transport_manifest, candidate, _ = load_deployment_transport_binding(workspace, direction)
    trace = load_completed_trace_manifest(workspace, direction, "selector_train", benchmark)
    state = workspace.state()
    stage = state.stages.get(f"{direction}/fit_risk")
    if stage is None or stage.status != "completed" or stage.outputs is None:
        raise V5PipelineError("stage requires completed selector-train risk fitting")
    manifest_artifact = stage.outputs.get("risk_fit_manifest")
    report_artifact = stage.outputs.get("risk_training_report")
    if manifest_artifact is None or report_artifact is None:
        raise V5PipelineError("risk fit stage lacks its manifest or report")
    path = workspace.artifact_path(manifest_artifact, verify_hash=True)
    workspace.artifact_path(report_artifact, verify_hash=True)
    manifest = V5RiskFitManifest.load(
        path,
        workspace=workspace,
        trace=trace,
        transport_manifest=transport_manifest,
        candidate=candidate,
    )
    if manifest.risk_training_report_sha256 != report_artifact.sha256:
        raise V5PipelineError("risk fit manifest refers to another training report")
    return manifest, trace, transport_manifest, candidate


def load_risk_predictor(
    workspace: V5PipelineWorkspace,
    manifest: V5RiskFitManifest,
    *,
    device: str = "cpu",
) -> RiskPredictor:
    """Load the immutable predictor referenced by a validated risk-fit manifest."""

    path = _verify_workspace_ref(workspace, manifest.predictor)
    try:
        return RiskPredictor.from_artifact(
            path,
            expected_sha256=manifest.predictor.sha256,
            device=device,
        )
    except Exception as exc:
        raise V5PipelineError("risk predictor artifact contract is invalid") from exc


def _risk_training_metrics(
    probabilities: Sequence[float],
    labels: Sequence[int],
) -> RiskTrainingMetrics:
    if len(probabilities) != len(labels) or not probabilities:
        raise V5PipelineError("risk metric rows are inconsistent")
    epsilon = 1e-7
    metric_rows: list[tuple[float, int]] = []
    correct = 0
    positives: list[float] = []
    negatives: list[float] = []
    for probability, label in zip(probabilities, labels, strict=True):
        if (
            not _finite_number(probability)
            or not 0 <= probability <= 1
            or type(label) is not int
            or label not in {0, 1}
        ):
            raise V5PipelineError("risk metric row is invalid")
        clipped = min(1 - epsilon, max(epsilon, probability))
        metric_rows.append((clipped, label))
        correct += int((probability >= 0.5) == bool(label))
        (positives if label else negatives).append(probability)
    if not positives or not negatives:
        raise V5PipelineError("risk ROC-AUC requires both classes")
    positive_weight = len(negatives) / len(positives)
    log_losses = [
        -(label * math.log(probability) + (1 - label) * math.log1p(-probability))
        for probability, label in metric_rows
    ]
    objective_losses = [
        -(positive_weight * label * math.log(probability) + (1 - label) * math.log1p(-probability))
        for probability, label in metric_rows
    ]
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    metrics = RiskTrainingMetrics(
        training_objective=sum(objective_losses) / len(objective_losses),
        log_loss=sum(log_losses) / len(log_losses),
        accuracy_at_half=correct / len(labels),
        roc_auc=wins / (len(positives) * len(negatives)),
    )
    errors = metrics.validate()
    if errors:
        raise V5PipelineError("; ".join(errors))
    return metrics


def _save_predictor(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            temporary,
            metadata={
                "feature_schema_version": RISK_FEATURE_SCHEMA_VERSION,
                "hidden_size": "64",
            },
        )
        canonicalize_safetensors_header(temporary)
        os.chmod(temporary, 0o444)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_risk_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    example: RiskTrainingExample,
) -> None:
    _write_json_replace(
        path,
        {
            "schema_version": V5_RISK_EXAMPLE_CHECKPOINT_SCHEMA,
            "binding_sha256": binding_sha256,
            "sample_id": benchmark_record.sample_id,
            "content_sha256": benchmark_record.content_sha256,
            "example": asdict(example),
        },
    )


def _load_risk_checkpoint(
    path: Path,
    *,
    binding_sha256: str,
    benchmark_record: GroupedPrefixRecord,
    trace_record: TraceRecord,
    expected_history: RiskHistory,
) -> RiskTrainingExample | None:
    if not path.is_file():
        return None
    try:
        if path.is_symlink():
            raise V5PipelineError("risk example checkpoint cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != V5_RISK_EXAMPLE_CHECKPOINT_SCHEMA:
            raise V5PipelineError("risk example checkpoint schema mismatch")
        if payload.get("binding_sha256") != binding_sha256:
            raise V5PipelineError("risk example checkpoint input binding mismatch")
        if (
            payload.get("sample_id") != benchmark_record.sample_id
            or payload.get("content_sha256") != benchmark_record.content_sha256
        ):
            raise V5PipelineError("risk example checkpoint sample binding mismatch")
        raw = dict(payload["example"])
        raw["features"] = tuple(raw["features"])
        example = RiskTrainingExample(**raw)
        errors = example.validate(
            benchmark_record=benchmark_record,
            trace_record=trace_record,
            expected_history=expected_history,
        )
        if errors:
            raise V5PipelineError("; ".join(errors))
        return example
    except V5PipelineError:
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise V5PipelineError("risk example checkpoint is malformed") from exc


def _verify_workspace_ref(workspace: V5PipelineWorkspace, reference: TraceObjectRef) -> Path:
    relative = Path(reference.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise V5PipelineError("risk predictor path escapes the workspace")
    lexical = workspace.root.joinpath(*relative.parts)
    try:
        path = lexical.resolve(strict=True)
        before = _file_signature(path)
    except (OSError, V5PipelineError) as exc:
        raise V5PipelineError("risk predictor object is unavailable") from exc
    if path != lexical or not path.is_relative_to(workspace.root):
        raise V5PipelineError("risk predictor path uses a symbolic-link escape")
    if before[2] != reference.size_bytes or path.stat().st_mode & 0o222:
        raise V5PipelineError("risk predictor size or read-only mode changed")
    if sha256_file(path) != reference.sha256:
        raise V5PipelineError("risk predictor checksum mismatch")
    if _file_signature(path) != before or sha256_file(path) != reference.sha256:
        raise V5PipelineError("risk predictor changed while hashing")
    return path


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
        raise V5PipelineError("risk fit metadata is not finite canonical JSON") from exc


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise V5PipelineError("risk fit input file is unavailable") from exc
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


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _strict_bool(value: Any) -> bool:
    return type(value) is bool


def stderr_risk_progress(every: int = 1) -> Callable[[int, int, str], None]:
    if every <= 0:
        raise V5PipelineError("risk progress interval must be positive")

    def report(index: int, total: int, sample_id: str) -> None:
        if index == total or index % every == 0:
            print(f"risk examples {index}/{total}: {sample_id}", file=sys.stderr, flush=True)

    return report
