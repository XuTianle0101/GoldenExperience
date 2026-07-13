from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import goldenexperience.size_variant.v5_validation as validation_module
from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.cli.v5_pipeline import DIRECTION_SIZES, build_parser
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.risk_gate import (
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_DIM,
    RISK_FEATURE_OOD_INDEX,
    RiskPredictor,
    clopper_pearson_upper_bound,
)
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    RiskGateSpec,
    TransportLossContract,
    TransportQualityEvidence,
)
from goldenexperience.size_variant.v5_calibration import V5RiskCalibrationManifest
from goldenexperience.size_variant.v5_collect import RawBenchmarkSample, TraceObjectRef
from goldenexperience.size_variant.v5_fit import (
    CandidateTrainingMetrics,
    TransportCandidateArtifact,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineStageRecord,
    StageLease,
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskHistory,
    RiskTrainingExample,
    RiskTrainingMetrics,
    RiskTrainingParameters,
    V5RiskFitManifest,
)
from goldenexperience.size_variant.v5_validation import (
    RiskValidationMeasurement,
    aggregate_accepted_quality,
    run_validate_stage,
    validation_selector_baselines,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model(model_id: str, size: float, layers: int) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=size,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest(f"{model_id}-config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest(f"{model_id}-weights"),
        num_layers=layers,
        num_key_value_heads=2,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=40960,
        chat_template_sha256=_digest("chat"),
    )


def _candidate() -> TransportCandidateArtifact:
    digest = _digest("transport")
    return TransportCandidateArtifact(
        candidate_id="transport-r32-s17",
        rank=32,
        seed=17,
        deployment_seed=True,
        weights=TraceObjectRef(
            digest,
            f"objects/{digest[:2]}/{digest}.safetensors",
            10,
        ),
        parameter_count=1,
        metrics=CandidateTrainingMetrics(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _risk_fit() -> V5RiskFitManifest:
    predictor_sha = _digest("predictor")
    return V5RiskFitManifest(
        pipeline_id="pipeline",
        direction="qwen3_4b_to_8b",
        code_sha256=_digest("code"),
        selector_train_split_sha256=_digest("selector_train"),
        selector_trace_manifest_sha256=_digest("selector-trace"),
        selector_raw_store_sha256=_digest("selector-raw"),
        transport_fit_manifest_sha256=_digest("fit"),
        transport_weights_sha256=_candidate().weights.sha256,
        risk_training_report_sha256=_digest("risk-report"),
        predictor=TraceObjectRef(
            predictor_sha,
            f"objects/{predictor_sha[:2]}/{predictor_sha}.safetensors",
            10,
        ),
        training=RiskTrainingParameters(),
        metrics=RiskTrainingMetrics(0.1, 0.1, 1.0, 1.0),
        sample_count=2048,
        unsafe_count=1024,
        safe_count=1024,
    )


def _calibration(risk_fit: V5RiskFitManifest) -> V5RiskCalibrationManifest:
    accepted = 300
    gate = RiskGateSpec(
        predictor_uri=risk_fit.predictor.path,
        predictor_sha256=risk_fit.predictor.sha256,
        threshold=0.2,
        calibration_dataset_sha256=_digest("risk_calibration"),
        calibration_method=RISK_CALIBRATION_METHOD,
        candidate_threshold_count=1,
        accepted_count=accepted,
        total_count=accepted,
        error_count=0,
        coverage=1.0,
        regression_risk_upper_bound=clopper_pearson_upper_bound(0, accepted),
    )
    return V5RiskCalibrationManifest(
        pipeline_id="pipeline",
        direction="qwen3_4b_to_8b",
        code_sha256=_digest("code"),
        risk_calibration_split_sha256=_digest("risk_calibration"),
        calibration_trace_manifest_sha256=_digest("calibration-trace"),
        calibration_raw_store_sha256=_digest("calibration-raw"),
        transport_fit_manifest_sha256=_digest("fit"),
        transport_weights_sha256=_candidate().weights.sha256,
        risk_fit_manifest_sha256=risk_fit.content_sha256(),
        calibration_report_sha256=_digest("calibration-report"),
        predictor=risk_fit.predictor,
        evaluator_sha256=_digest("calibration-evaluator"),
        risk_gate=gate,
    )


class _FakePredictor:
    def unsafe_probability(self, _features: Any) -> float:
        return 0.1


def _example(
    sample_id: str = "sample-0000",
    *,
    history: RiskHistory | None = None,
    unsafe: bool = False,
    ood_distance: float = 0.0,
) -> RiskTrainingExample:
    history = history or RiskHistory()
    features = [0.0] * RISK_FEATURE_DIM
    features[RISK_FEATURE_OOD_INDEX] = ood_distance
    return RiskTrainingExample(
        sample_id=sample_id,
        prefix_group_id="shared-group",
        features=tuple(features),
        unsafe=unsafe,
        native_task_score=1.0,
        bridge_task_score=1.0,
        task_pass_threshold=1.0,
        greedy_matches=15 if unsafe else RISK_LABEL_GENERATION_TOKENS,
        greedy_tokens=RISK_LABEL_GENERATION_TOKENS,
        native_nll=1.0,
        bridge_nll=1.0,
        teacher_tokens=RISK_LABEL_GENERATION_TOKENS,
        key_cosine=1.0,
        history_samples=history.samples,
        history_failures=history.failures,
        history_greedy_agreement=history.greedy_agreement,
        sidecar_sha256=_digest(f"sidecar-{sample_id}"),
        native_prediction_sha256=_digest(f"native-{sample_id}"),
        bridge_prediction_sha256=_digest(f"bridge-{sample_id}"),
        native_tokens_sha256=_digest(f"native-tokens-{sample_id}"),
        bridge_tokens_sha256=_digest(f"bridge-tokens-{sample_id}"),
    )


def test_validation_measurement_matches_full_runtime_gate() -> None:
    risk_fit = _risk_fit()
    gate = _calibration(risk_fit).risk_gate
    predictor = cast(RiskPredictor, _FakePredictor())
    unseen = RiskValidationMeasurement(
        _example(), 0.1, False, "unseen_or_insufficient_shadow_history"
    )
    assert unseen.validate(predictor=predictor, risk_gate=gate) == []

    history = RiskHistory(samples=1, failures=0, greedy_agreement_sum=1.0)
    ood = RiskValidationMeasurement(
        _example(history=history, ood_distance=7.0),
        0.1,
        False,
        "out_of_distribution",
    )
    assert ood.validate(predictor=predictor, risk_gate=gate) == []
    predicted = RiskValidationMeasurement(
        _example(history=history),
        0.9,
        False,
        "predicted_unsafe",
    )
    wrong_predictor = cast(
        RiskPredictor,
        SimpleNamespace(unsafe_probability=lambda _features: 0.9),
    )
    assert predicted.validate(predictor=wrong_predictor, risk_gate=gate) == []
    accepted = RiskValidationMeasurement(_example(history=history), 0.1, True, "accepted")
    assert accepted.validate(predictor=predictor, risk_gate=gate) == []
    assert "validation admission differs from the calibrated runtime gate" in replace(
        accepted, accepted=False, decision="predicted_unsafe"
    ).validate(predictor=predictor, risk_gate=gate)
    assert "validation admission marker must be boolean" in replace(
        accepted, accepted=cast(Any, 1)
    ).validate(predictor=predictor, risk_gate=gate)
    assert "validation admission decision is invalid" in replace(
        accepted, decision="unknown"
    ).validate(predictor=predictor, risk_gate=gate)


def test_validation_quality_and_five_baselines_use_actual_gate(monkeypatch) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "validation", 301)
    history = RiskHistory()
    measurements = []
    for index in range(301):
        example = _example(f"sample-{index:04d}", history=history)
        accepted = index > 0
        measurements.append(
            RiskValidationMeasurement(
                example,
                0.1,
                accepted,
                "accepted" if accepted else "unseen_or_insufficient_shadow_history",
            )
        )
        history = history.update(example)

    quality = aggregate_accepted_quality(
        measurements,
        dataset_sha256=_digest("validation"),
    )
    baselines = validation_selector_baselines(measurements, calibrated_threshold=0.2)

    assert quality.accepted_count == 300
    assert quality.unsafe_count == 0
    assert quality.bridge_task_score == 1.0
    assert quality.gate_errors(validation_module.SelectiveQualityThresholds()) == []
    assert [item.name for item in baselines] == [
        "no_selector",
        "cosine_threshold",
        "uncalibrated_mlp",
        "calibrated_selector",
        "oracle_selector",
    ]
    assert baselines[0].accepted_count == 301
    assert baselines[3].accepted_count == 300
    assert baselines[4].error_count == 0

    with pytest.raises(V5PipelineError, match="accepted no samples"):
        aggregate_accepted_quality(
            [replace(item, accepted=False) for item in measurements],
            dataset_sha256=_digest("validation"),
        )
    with pytest.raises(V5PipelineError, match="calibrated threshold"):
        validation_selector_baselines(measurements, calibrated_threshold=None)


class _FakeValidationWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        split_sha256 = {name: _digest(name) for name in SPLIT_COUNTS}
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            code_sha256=_digest("code"),
            benchmark_manifest_sha256=_digest("benchmark"),
            split_sha256=split_sha256,
            direction=lambda _direction: object(),
        )
        self.completed_outputs: dict[str, Path] | None = None
        self.failures = 0

    def begin_stage(
        self,
        direction: str,
        stage: str,
        *,
        parameters: dict[str, Any],
        resume: bool = False,
    ) -> StageLease:
        assert direction == "qwen3_4b_to_8b"
        assert stage == "validate"
        assert parameters["threshold"] == 0.2
        assert parameters["predictor_device"] == "cpu"
        assert parameters["selector_cosine_threshold"] == 0.95
        if self.failures:
            assert resume
        return StageLease(direction, stage, _digest("stage-input"), "attempt")

    def complete_stage(
        self,
        lease: StageLease,
        *,
        outputs: dict[str, Path],
        metadata: dict[str, Any],
    ) -> PipelineStageRecord:
        assert metadata["sample_count"] == 301
        assert metadata["accepted_count"] == 300
        assert metadata["unsafe_count"] == 0
        assert metadata["passed"] is True
        assert metadata["authority"] == "validation_candidate"
        self.completed_outputs = outputs
        return PipelineStageRecord(
            direction=lease.direction,
            stage=lease.stage,
            status="completed",
            input_sha256=lease.input_sha256,
            attempt_id=lease.attempt_id,
            attempt_count=self.failures + 1,
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T00:00:01+00:00",
            receipt_sha256=_digest("receipt"),
            receipt_path="receipts/receipt.json",
            outputs={},
        )

    def fail_stage(self, _lease: StageLease, _error: Exception) -> None:
        self.failures += 1


class _FakeValidationEvaluator:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self) -> _FakeValidationEvaluator:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def evaluate(
        self,
        record: GroupedPrefixRecord,
        _trace_record: Any,
        _sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic validation interruption")
        return _example(record.sample_id, history=history)


def _validation_rows(count: int) -> tuple[list[GroupedPrefixRecord], list[RawBenchmarkSample]]:
    records = []
    samples = []
    for index in range(count):
        sample_id = f"sample-{index:04d}"
        records.append(
            GroupedPrefixRecord(
                sample_id=sample_id,
                split="validation",
                dataset_id="gsm8k",
                prefix_group_id="shared-group",
                prefix_sha256=_digest(f"prefix-{index}"),
                suffix_query_sha256=_digest(f"suffix-{index}"),
                content_sha256=_digest(f"content-{index}"),
                token_bucket=128,
                task="qa",
            )
        )
        samples.append(
            RawBenchmarkSample(
                sample_id=sample_id,
                prefix_text=f"prefix-{index}",
                suffix_query=f"suffix-{index}",
                reference="answer",
                evaluation={"metric": "exact_match"},
                provenance={"row": index},
            )
        )
    return records, samples


def test_validation_stage_resumes_and_recomputes_detailed_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    count = 301
    monkeypatch.setitem(SPLIT_COUNTS, "validation", count)
    monkeypatch.setitem(SPLIT_COUNTS, "risk_calibration", 300)
    records, samples = _validation_rows(count)
    benchmark = SimpleNamespace(records=tuple(records))
    store = tmp_path / "validation.jsonl"
    store.write_text("synthetic validation store\n", encoding="utf-8")
    store_sha = hashlib.sha256(store.read_bytes()).hexdigest()
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=store_sha,
        records=tuple(SimpleNamespace(sample_id=item.sample_id) for item in records),
        content_sha256=lambda: _digest("validation-trace"),
    )
    source = _model("Qwen/Qwen3-4B", 4.0, 3)
    target = _model("Qwen/Qwen3-8B", 8.0, 4)
    transport = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        source=source,
        target=target,
        training=SimpleNamespace(source_window=3, loss=TransportLossContract()),
        content_sha256=lambda: _digest("fit"),
    )
    candidate = _candidate()
    risk_fit = _risk_fit()
    calibration = _calibration(risk_fit)
    structure = SimpleNamespace(
        content_sha256=lambda: _digest("structure"),
        deployment_quality=TransportQualityEvidence(
            evaluation_dataset_sha256=_digest("method_dev"),
            prompt_count=1024,
            task_score=0.99,
            oracle_safe_coverage=0.9,
            greedy_agreement=0.99,
        ),
    )
    predictor = cast(RiskPredictor, _FakePredictor())
    monkeypatch.setattr(validation_module, "load_bound_benchmark", lambda _workspace: benchmark)
    monkeypatch.setattr(
        validation_module,
        "load_completed_risk_calibration",
        lambda _workspace, _direction: (
            calibration,
            risk_fit,
            object(),
            transport,
            candidate,
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "load_frozen_transport_structure",
        lambda _workspace: (structure, object(), object()),
    )
    monkeypatch.setattr(
        validation_module,
        "load_completed_trace_manifest",
        lambda _workspace, _direction, split, _benchmark: (
            trace if split == "validation" else pytest.fail("validation accessed another split")
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "load_raw_sample_store",
        lambda _path, _benchmark, *, split: (
            tuple(zip(records, samples, strict=True))
            if split == "validation"
            else pytest.fail("validation loaded another split")
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "load_risk_predictor",
        lambda _workspace, _manifest, *, device: (
            predictor
            if device == "cpu"
            else pytest.fail("validation predictor was not loaded on CPU")
        ),
    )

    workspace = _FakeValidationWorkspace(tmp_path / "workspace")
    interrupted = _FakeValidationEvaluator(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_validate_stage(
            workspace=cast(V5PipelineWorkspace, workspace),
            direction="qwen3_4b_to_8b",
            sample_store_path=store,
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
        )
    resumed = _FakeValidationEvaluator()
    run_validate_stage(
        workspace=cast(V5PipelineWorkspace, workspace),
        direction="qwen3_4b_to_8b",
        sample_store_path=store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        resume=True,
    )

    assert interrupted.calls == 3
    assert resumed.calls == 299
    assert workspace.completed_outputs is not None
    report_path = workspace.completed_outputs["validation_report"]
    manifest_path = workspace.completed_outputs["validation_manifest"]
    selective_path = workspace.completed_outputs["selective_manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted_quality"]["accepted_count"] == 300
    assert report["baselines"][3]["accepted_count"] == 300
    assert report["measurements"][0]["decision"] == ("unseen_or_insufficient_shadow_history")
    manifest = validation_module.V5ValidationManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    selective = validation_module.SelectiveKVBridgeManifest.from_dict(
        json.loads(selective_path.read_text(encoding="utf-8"))
    )
    assert selective.state is ArtifactState.VALIDATION_CANDIDATE
    assert selective.validation_errors() == []
    assert manifest.selective_artifact_id == selective.artifact_id
    manifest_kwargs = {
        "workspace": cast(V5PipelineWorkspace, workspace),
        "trace": trace,
        "structure": cast(Any, structure),
        "transport_manifest": transport,
        "candidate": candidate,
        "risk_fit": risk_fit,
        "calibration": calibration,
        "selective": selective,
    }
    assert manifest.validate(**manifest_kwargs) == []
    assert "validation manifest does not carry a passing result" in replace(
        manifest, passed=cast(Any, 1)
    ).validate(**manifest_kwargs)
    malformed_quality = replace(manifest.accepted_quality, unsafe_count=cast(Any, False))
    assert "accepted validation quality unsafe_count must be an integer" in replace(
        manifest, accepted_quality=malformed_quality
    ).validate(**manifest_kwargs)
    measurements = validation_module._load_and_validate_report(
        report_path,
        benchmark=cast(Any, benchmark),
        workspace=cast(V5PipelineWorkspace, workspace),
        trace=trace,
        structure=cast(Any, structure),
        transport_manifest=transport,
        candidate=candidate,
        risk_fit=risk_fit,
        calibration=calibration,
        manifest=manifest,
        predictor=predictor,
    )
    assert len(measurements) == count

    report["measurements"][0]["accepted"] = True
    tampered = tmp_path / "tampered-validation.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5PipelineError, match="runtime gate"):
        validation_module._load_and_validate_report(
            tampered,
            benchmark=cast(Any, benchmark),
            workspace=cast(V5PipelineWorkspace, workspace),
            trace=trace,
            structure=cast(Any, structure),
            transport_manifest=transport,
            candidate=candidate,
            risk_fit=risk_fit,
            calibration=calibration,
            manifest=manifest,
            predictor=predictor,
        )


def test_validation_production_surface_has_no_quality_override() -> None:
    parameters = inspect.signature(run_validate_stage).parameters
    for forbidden in ("threshold", "quality", "coverage", "risk_bound", "predictor_device"):
        assert forbidden not in parameters

    parser = build_parser()
    validate = next(
        action.choices["validate"]
        for action in parser._actions
        if getattr(action, "choices", None) and "validate" in action.choices
    )
    option_strings = {
        option for action in validate._actions for option in getattr(action, "option_strings", ())
    }
    assert not any(
        word in option
        for option in option_strings
        for word in ("threshold", "quality", "coverage", "risk", "predictor")
    )
    for direction in DIRECTION_SIZES:
        args = parser.parse_args(
            [
                "validate",
                "--workspace",
                "workspace",
                "--direction",
                direction,
                "--samples",
                "validation.jsonl",
            ]
        )
        assert args.direction == direction
