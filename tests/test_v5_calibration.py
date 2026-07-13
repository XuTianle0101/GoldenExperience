from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import goldenexperience.size_variant.v5_calibration as calibration_module
from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.cli.v5_pipeline import DIRECTION_SIZES, build_parser
from goldenexperience.size_variant.risk_gate import (
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_DIM,
    RiskCalibrationExample,
    RiskGateError,
    RiskPredictor,
    select_calibrated_threshold,
)
from goldenexperience.size_variant.selective_manifest import RiskGateSpec
from goldenexperience.size_variant.v5_calibration import (
    RISK_CALIBRATION_CONFIDENCE,
    RISK_CALIBRATION_MAX_RISK_UPPER_BOUND,
    RISK_CALIBRATION_MIN_ACCEPTED,
    RiskCalibrationMeasurement,
    V5RiskCalibrationManifest,
    run_calibrate_stage,
)
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FakePredictor:
    def unsafe_probability(self, features: Any) -> float:
        return 0.9 if float(features[0]) > 0.5 else 0.1


def _example(
    sample_id: str = "sample-0000",
    *,
    unsafe: bool = False,
    history: RiskHistory | None = None,
) -> RiskTrainingExample:
    history = history or RiskHistory()
    features = (1.0 if unsafe else 0.0,) + (0.0,) * (RISK_FEATURE_DIM - 1)
    return RiskTrainingExample(
        sample_id=sample_id,
        prefix_group_id="shared-group",
        features=features,
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


def _candidate() -> TransportCandidateArtifact:
    digest = _digest("transport")
    return TransportCandidateArtifact(
        candidate_id="transport-r64-s17",
        rank=64,
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
        selector_train_split_sha256=_digest("selector-split"),
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


def _manifest_parts(total: int = 380) -> tuple[Any, Any, TransportCandidateArtifact, Any]:
    candidate = _candidate()
    risk_fit = _risk_fit()
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=_digest("calibration-raw"),
        content_sha256=lambda: _digest("calibration-trace"),
    )
    transport = SimpleNamespace(content_sha256=lambda: _digest("fit"))
    result = select_calibrated_threshold(
        [RiskCalibrationExample(0.1, False) for _ in range(total - 10)]
        + [RiskCalibrationExample(0.9, True) for _ in range(10)]
    )
    return trace, transport, candidate, (risk_fit, result)


def test_calibration_measurement_recomputes_frozen_predictor_probability() -> None:
    predictor = cast(RiskPredictor, _FakePredictor())
    history = RiskHistory()
    measurement = RiskCalibrationMeasurement(_example(history=history), 0.1)

    assert measurement.validate(predictor=predictor, expected_history=history) == []
    assert "risk calibration probability differs from the frozen predictor" in replace(
        measurement, unsafe_probability=0.2
    ).validate(predictor=predictor, expected_history=history)
    assert "risk calibration probability is invalid" in replace(
        measurement, unsafe_probability=math.nan
    ).validate(predictor=predictor)


def test_threshold_search_validates_before_sorting_and_keeps_ties_together() -> None:
    rows = [RiskCalibrationExample(0.1, False) for _ in range(370)]
    rows.extend(RiskCalibrationExample(0.9, True) for _ in range(10))
    result = select_calibrated_threshold(rows)

    assert result.threshold == 0.1
    assert result.accepted_count == 370
    assert result.error_count == 0
    assert result.candidate_threshold_count == 2
    assert result.regression_risk_upper_bound <= 0.01
    with pytest.raises(RiskGateError, match="probabilities"):
        select_calibrated_threshold(
            [
                RiskCalibrationExample(cast(Any, "bad"), False),
                RiskCalibrationExample(0.1, False),
            ]
        )
    with pytest.raises(RiskGateError, match="labels"):
        select_calibrated_threshold([RiskCalibrationExample(0.1, cast(Any, 1)) for _ in range(300)])


def test_calibration_manifest_freezes_gate_and_all_provenance(monkeypatch) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "risk_calibration", 380)
    trace, transport, candidate, packed = _manifest_parts()
    risk_fit, result = packed
    config = SimpleNamespace(
        pipeline_id="pipeline",
        code_sha256=_digest("code"),
        split_sha256={"risk_calibration": _digest("calibration-split")},
        direction=lambda _direction: object(),
    )
    workspace = cast(V5PipelineWorkspace, SimpleNamespace(config=config))
    gate = RiskGateSpec(
        predictor_uri=risk_fit.predictor.path,
        predictor_sha256=risk_fit.predictor.sha256,
        threshold=result.threshold,
        calibration_dataset_sha256=_digest("calibration-split"),
        calibration_method=RISK_CALIBRATION_METHOD,
        candidate_threshold_count=result.candidate_threshold_count,
        accepted_count=result.accepted_count,
        total_count=result.total_count,
        error_count=result.error_count,
        coverage=result.coverage,
        regression_risk_upper_bound=result.regression_risk_upper_bound,
        confidence_level=result.confidence_level,
    )
    manifest = V5RiskCalibrationManifest(
        pipeline_id="pipeline",
        direction=trace.direction,
        code_sha256=_digest("code"),
        risk_calibration_split_sha256=_digest("calibration-split"),
        calibration_trace_manifest_sha256=_digest("calibration-trace"),
        calibration_raw_store_sha256=_digest("calibration-raw"),
        transport_fit_manifest_sha256=_digest("fit"),
        transport_weights_sha256=candidate.weights.sha256,
        risk_fit_manifest_sha256=risk_fit.content_sha256(),
        calibration_report_sha256=_digest("calibration-report"),
        predictor=risk_fit.predictor,
        evaluator_sha256=_digest("evaluator"),
        risk_gate=gate,
    )
    kwargs = {
        "workspace": workspace,
        "trace": trace,
        "transport_manifest": transport,
        "candidate": candidate,
        "risk_fit": risk_fit,
    }

    assert manifest.validate(**kwargs) == []
    assert "risk calibration gate identity is inconsistent" in replace(
        manifest,
        risk_gate=replace(gate, ood_threshold=7.0),
    ).validate(**kwargs)
    assert "risk calibration predictor reference changed" in replace(
        manifest,
        predictor=replace(risk_fit.predictor, sha256=_digest("changed")),
    ).validate(**kwargs)
    assert "risk calibration manifest must freeze a calibrated threshold" in replace(
        manifest, calibrated=cast(Any, 1)
    ).validate(**kwargs)


class _FakeCalibrationWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            code_sha256=_digest("code"),
            split_sha256={"risk_calibration": _digest("calibration-split")},
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
        assert stage == "calibrate"
        assert parameters["calibration_method"] == RISK_CALIBRATION_METHOD
        assert parameters["min_accepted"] == RISK_CALIBRATION_MIN_ACCEPTED
        assert parameters["max_risk_upper_bound"] == RISK_CALIBRATION_MAX_RISK_UPPER_BOUND
        assert parameters["confidence"] == RISK_CALIBRATION_CONFIDENCE
        assert parameters["predictor_device"] == "cpu"
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
        assert metadata["accepted_count"] == 300
        assert metadata["total_count"] == 300
        assert metadata["error_count"] == 0
        assert metadata["candidate_threshold_count"] == 1
        assert metadata["calibrated"] is True
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


class _FakeCalibrationEvaluator:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self) -> _FakeCalibrationEvaluator:
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
            raise RuntimeError("synthetic calibration interruption")
        return _example(record.sample_id, history=history)


def _calibration_rows(count: int) -> tuple[list[GroupedPrefixRecord], list[RawBenchmarkSample]]:
    records = []
    samples = []
    for index in range(count):
        sample_id = f"sample-{index:04d}"
        records.append(
            GroupedPrefixRecord(
                sample_id=sample_id,
                split="risk_calibration",
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


def test_calibration_stage_resumes_and_report_recomputes_every_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    count = 300
    monkeypatch.setitem(SPLIT_COUNTS, "risk_calibration", count)
    records, samples = _calibration_rows(count)
    benchmark = SimpleNamespace(records=tuple(records))
    store = tmp_path / "risk-calibration.jsonl"
    store.write_text("synthetic calibration store\n", encoding="utf-8")
    store_sha = hashlib.sha256(store.read_bytes()).hexdigest()
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=store_sha,
        records=tuple(SimpleNamespace(sample_id=item.sample_id) for item in records),
        content_sha256=lambda: _digest("calibration-trace"),
    )
    transport = SimpleNamespace(content_sha256=lambda: _digest("fit"))
    candidate = _candidate()
    risk_fit = _risk_fit()
    predictor = cast(RiskPredictor, _FakePredictor())
    monkeypatch.setattr(calibration_module, "load_bound_benchmark", lambda _workspace: benchmark)
    monkeypatch.setattr(
        calibration_module,
        "load_completed_risk_fit",
        lambda _workspace, _direction: (risk_fit, object(), transport, candidate),
    )
    monkeypatch.setattr(
        calibration_module,
        "load_completed_trace_manifest",
        lambda _workspace, _direction, split, _benchmark: (
            trace
            if split == "risk_calibration"
            else pytest.fail("calibration accessed another split")
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "load_raw_sample_store",
        lambda _path, _benchmark, *, split: (
            tuple(zip(records, samples, strict=True))
            if split == "risk_calibration"
            else pytest.fail("calibration loaded another split")
        ),
    )
    monkeypatch.setattr(
        calibration_module,
        "load_risk_predictor",
        lambda _workspace, _manifest, *, device: (
            predictor
            if device == "cpu"
            else pytest.fail("calibration predictor was not loaded on CPU")
        ),
    )

    workspace = _FakeCalibrationWorkspace(tmp_path / "workspace")
    interrupted = _FakeCalibrationEvaluator(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_calibrate_stage(
            workspace=cast(V5PipelineWorkspace, workspace),
            direction="qwen3_4b_to_8b",
            sample_store_path=store,
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
        )
    resumed = _FakeCalibrationEvaluator()
    run_calibrate_stage(
        workspace=cast(V5PipelineWorkspace, workspace),
        direction="qwen3_4b_to_8b",
        sample_store_path=store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        resume=True,
    )

    assert interrupted.calls == 3
    assert resumed.calls == 298
    assert workspace.completed_outputs is not None
    report_path = workspace.completed_outputs["risk_calibration_report"]
    manifest_path = workspace.completed_outputs["risk_calibration_manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [item["example"]["history_samples"] for item in report["measurements"]] == list(
        range(count)
    )
    assert report["result"]["accepted_count"] == count
    assert report["result"]["regression_risk_upper_bound"] < 0.01
    manifest = V5RiskCalibrationManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    measurements, result = calibration_module._load_and_validate_calibration_report(
        report_path,
        benchmark=cast(Any, benchmark),
        workspace=cast(V5PipelineWorkspace, workspace),
        trace=trace,
        transport_manifest=transport,
        candidate=candidate,
        risk_fit=risk_fit,
        manifest=manifest,
        predictor=predictor,
    )
    assert len(measurements) == count
    assert result.threshold == 0.1

    report["result"]["threshold"] = 0.2
    tampered = tmp_path / "tampered-report.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5PipelineError, match="differs from the detailed report"):
        calibration_module._load_and_validate_calibration_report(
            tampered,
            benchmark=cast(Any, benchmark),
            workspace=cast(V5PipelineWorkspace, workspace),
            trace=trace,
            transport_manifest=transport,
            candidate=candidate,
            risk_fit=risk_fit,
            manifest=manifest,
            predictor=predictor,
        )


def test_calibration_production_surface_has_no_threshold_override() -> None:
    parameters = inspect.signature(run_calibrate_stage).parameters
    assert "threshold" not in parameters
    assert "confidence" not in parameters
    assert "min_accepted" not in parameters
    assert "max_risk" not in parameters

    parser = build_parser()
    calibrate = next(
        action.choices["calibrate"]
        for action in parser._actions
        if getattr(action, "choices", None) and "calibrate" in action.choices
    )
    option_strings = {
        option for action in calibrate._actions for option in getattr(action, "option_strings", ())
    }
    assert not any(
        word in option
        for option in option_strings
        for word in ("threshold", "confidence", "accepted", "risk-bound")
    )
    for direction in DIRECTION_SIZES:
        args = parser.parse_args(
            [
                "calibrate",
                "--workspace",
                "workspace",
                "--direction",
                direction,
                "--samples",
                "risk-calibration.jsonl",
            ]
        )
        assert args.direction == direction
