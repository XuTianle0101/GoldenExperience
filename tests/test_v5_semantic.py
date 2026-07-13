from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import goldenexperience.size_variant.v5_semantic as semantic_module
from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.cli.v5_pipeline import DIRECTION_SIZES, build_parser
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.risk_gate import (
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_DIM,
    RiskPredictor,
    clopper_pearson_upper_bound,
)
from goldenexperience.size_variant.selective_manifest import (
    ArtifactState,
    RiskGateSpec,
    SelectiveKVBridgeManifest,
    SelectiveQualityThresholds,
    TransportLossContract,
    TransportQualityEvidence,
)
from goldenexperience.size_variant.v5_calibration import V5RiskCalibrationManifest
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceObjectRef,
    publication_sample_content_sha256,
)
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
    RiskPrefixTokenBinding,
    RiskTrainingExample,
    RiskTrainingMetrics,
    RiskTrainingParameters,
    V5RiskFitManifest,
)
from goldenexperience.size_variant.v5_sealed import (
    V5SealedDirectionBinding,
    V5SemanticOpenReceipt,
)
from goldenexperience.size_variant.v5_validation import (
    RiskValidationMeasurement,
    aggregate_accepted_quality,
    build_validation_candidate,
)


def _digest(value: str | bytes) -> str:
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _example(record: GroupedPrefixRecord, history: RiskHistory) -> RiskTrainingExample:
    return RiskTrainingExample(
        sample_id=record.sample_id,
        prefix_group_id=record.prefix_group_id,
        features=(0.0,) * RISK_FEATURE_DIM,
        unsafe=False,
        native_task_score=1.0,
        bridge_task_score=1.0,
        task_pass_threshold=1.0,
        greedy_matches=RISK_LABEL_GENERATION_TOKENS,
        greedy_tokens=RISK_LABEL_GENERATION_TOKENS,
        native_nll=1.0,
        bridge_nll=1.0,
        teacher_tokens=RISK_LABEL_GENERATION_TOKENS,
        key_cosine=1.0,
        history_samples=history.samples,
        history_failures=history.failures,
        history_greedy_agreement=history.greedy_agreement,
        sidecar_sha256=_digest(f"sidecar-{record.sample_id}"),
        native_prediction_sha256=_digest(f"native-{record.sample_id}"),
        bridge_prediction_sha256=_digest(f"bridge-{record.sample_id}"),
        native_tokens_sha256=_digest(f"native-tokens-{record.sample_id}"),
        bridge_tokens_sha256=_digest(f"bridge-tokens-{record.sample_id}"),
    )


def _semantic_rows(
    count: int,
) -> tuple[list[GroupedPrefixRecord], list[RawBenchmarkSample], bytes]:
    records = []
    samples = []
    rows = []
    for index in range(count):
        sample_id = f"semantic-{index:04d}"
        prefix = f"prefix {index}"
        suffix = f"query {index}"
        reference = f"answer {index}"
        evaluation = {"metric": "exact_match"}
        record = GroupedPrefixRecord(
            sample_id=sample_id,
            split="semantic_sealed_test",
            dataset_id="gsm8k",
            prefix_group_id="shared-group",
            prefix_sha256=_digest(prefix),
            suffix_query_sha256=_digest(suffix),
            content_sha256=publication_sample_content_sha256(
                prefix_text=prefix,
                suffix_query=suffix,
                reference=reference,
                evaluation=evaluation,
                task="qa",
            ),
            token_bucket=128,
            task="qa",
        )
        sample = RawBenchmarkSample(
            sample_id=sample_id,
            prefix_text=prefix,
            suffix_query=suffix,
            reference=reference,
            evaluation=evaluation,
            provenance={"row": index},
        )
        records.append(record)
        samples.append(sample)
        rows.append(sample.to_dict())
    payload = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    return records, samples, payload


class _FakePredictor:
    def unsafe_probability(self, _features: Any) -> float:
        return 0.1


class _FakeSemanticEvaluator:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self) -> _FakeSemanticEvaluator:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def bind_semantic_prefix(
        self,
        record: GroupedPrefixRecord,
        _sample: RawBenchmarkSample,
    ) -> RiskPrefixTokenBinding:
        return RiskPrefixTokenBinding(
            record.sample_id,
            record.token_bucket,
            _digest(record.sample_id),
        )

    def evaluate(
        self,
        record: GroupedPrefixRecord,
        _binding: RiskPrefixTokenBinding,
        _sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic semantic interruption")
        return _example(record, history)


class _FakeSemanticWorkspace:
    def __init__(self, root: Path, payload_sha256: str) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            code_sha256=_digest("code"),
            benchmark_manifest_sha256=_digest("benchmark"),
            sealed_payload_sha256=payload_sha256,
            tokenizer_sha256=_digest("tokenizer"),
            split_sha256={name: _digest(name) for name in SPLIT_COUNTS},
            direction=lambda _direction: object(),
        )
        self.completed_outputs: dict[str, Path] | None = None
        self.failures = 0
        self.open_receipt_sha256: str | None = None

    def begin_semantic_stage(
        self,
        direction: str,
        *,
        parameters: dict[str, Any],
        open_receipt_sha256: str,
        snapshot_sha256: str,
        resume: bool = False,
    ) -> StageLease:
        assert direction == "qwen3_4b_to_8b"
        assert parameters["predictor_device"] == "cpu"
        assert parameters["threshold"] == 0.2
        assert open_receipt_sha256 == self.open_receipt_sha256
        assert snapshot_sha256 == self.config.sealed_payload_sha256
        if self.failures:
            assert resume
        return StageLease(direction, "semantic_sealed", _digest("stage-input"), "attempt")

    def complete_stage(
        self,
        lease: StageLease,
        *,
        outputs: dict[str, Path],
        metadata: dict[str, Any],
    ) -> PipelineStageRecord:
        assert metadata["sample_count"] == 301
        assert metadata["accepted_count"] == 300
        assert metadata["passed"] is True
        assert metadata["authority"] == "semantic_approved"
        assert metadata["runtime_authority"] is False
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


def _open_receipt(
    workspace: _FakeSemanticWorkspace,
    *,
    validation: Any,
    validation_selective: SelectiveKVBridgeManifest,
    calibration: V5RiskCalibrationManifest,
    risk_fit: V5RiskFitManifest,
    candidate: TransportCandidateArtifact,
) -> V5SemanticOpenReceipt:
    bindings = []
    for direction in DIRECTION_SIZES:
        current = direction == "qwen3_4b_to_8b"
        bindings.append(
            V5SealedDirectionBinding(
                direction=direction,
                validation_manifest_sha256=(
                    validation.content_sha256() if current else _digest(f"{direction}-validation")
                ),
                validation_report_sha256=(
                    validation.validation_report_sha256
                    if current
                    else _digest(f"{direction}-report")
                ),
                risk_calibration_manifest_sha256=(
                    calibration.content_sha256() if current else _digest(f"{direction}-calibration")
                ),
                selective_artifact_id=(
                    validation_selective.artifact_id
                    if current
                    else f"selective-kv-{_digest(direction)[:24]}"
                ),
                code_sha256=workspace.config.code_sha256,
                transport_weights_sha256=(
                    candidate.weights.sha256 if current else _digest(f"{direction}-transport")
                ),
                predictor_sha256=(
                    risk_fit.predictor.sha256 if current else _digest(f"{direction}-predictor")
                ),
                threshold=0.2,
                threshold_sha256=_digest(f"{direction}-threshold"),
                passed=True,
            )
        )
    receipt = V5SemanticOpenReceipt(
        pipeline_id=workspace.config.pipeline_id,
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        validation_split_sha256=workspace.config.split_sha256["validation"],
        semantic_split_sha256=workspace.config.split_sha256["semantic_sealed_test"],
        sealed_payload_sha256=workspace.config.sealed_payload_sha256,
        code_sha256=workspace.config.code_sha256,
        validation_gate_receipt_sha256=_digest("gate"),
        snapshot_path=f".pipeline/semantic_sealed/{workspace.config.sealed_payload_sha256}.jsonl",
        directions=tuple(bindings),
    )
    workspace.open_receipt_sha256 = receipt.content_sha256()
    return receipt


def test_semantic_stage_resumes_replays_and_grants_no_runtime_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    count = 301
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", count)
    monkeypatch.setitem(SPLIT_COUNTS, "validation", count)
    monkeypatch.setitem(SPLIT_COUNTS, "risk_calibration", 300)
    records, samples, payload = _semantic_rows(count)
    benchmark = SimpleNamespace(records=tuple(records))
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_bytes(payload)
    snapshot.chmod(0o444)
    workspace = _FakeSemanticWorkspace(tmp_path / "workspace", _digest(payload))
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
    history = RiskHistory()
    validation_measurements = []
    for index, record in enumerate(records):
        example = _example(record, history)
        validation_measurements.append(
            RiskValidationMeasurement(
                example,
                0.1,
                index > 0,
                "accepted" if index > 0 else "unseen_or_insufficient_shadow_history",
            )
        )
        history = history.update(example)
    validation_quality = aggregate_accepted_quality(
        validation_measurements,
        dataset_sha256=workspace.config.split_sha256["validation"],
        expected_count=count,
    )
    structure = SimpleNamespace(
        deployment_quality=TransportQualityEvidence(
            evaluation_dataset_sha256=_digest("method_dev"),
            prompt_count=1024,
            task_score=0.99,
            oracle_safe_coverage=0.9,
            greedy_agreement=0.99,
        )
    )
    validation_selective = build_validation_candidate(
        workspace=cast(V5PipelineWorkspace, workspace),
        transport_manifest=transport,
        candidate=candidate,
        structure=structure,
        calibration=calibration,
        quality=validation_quality,
        thresholds=SelectiveQualityThresholds(),
    )
    validation = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        threshold=0.2,
        thresholds=SelectiveQualityThresholds(),
        validation_report_sha256=_digest("validation-report"),
        content_sha256=lambda: _digest("validation-manifest"),
    )
    receipt = _open_receipt(
        workspace,
        validation=validation,
        validation_selective=validation_selective,
        calibration=calibration,
        risk_fit=risk_fit,
        candidate=candidate,
    )
    predictor = cast(RiskPredictor, _FakePredictor())
    monkeypatch.setattr(semantic_module, "load_bound_benchmark", lambda _workspace: benchmark)
    monkeypatch.setattr(
        semantic_module,
        "load_semantic_open_receipt",
        lambda _workspace: (receipt, snapshot),
    )
    monkeypatch.setattr(
        semantic_module,
        "load_completed_validation",
        lambda _workspace, _direction: (validation, validation_selective, object()),
    )
    monkeypatch.setattr(
        semantic_module,
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
        semantic_module,
        "load_risk_predictor",
        lambda _workspace, _risk_fit, *, device: (
            predictor if device == "cpu" else pytest.fail("semantic predictor was not on CPU")
        ),
    )
    interrupted = _FakeSemanticEvaluator(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        semantic_module.run_semantic_stage(
            workspace=cast(V5PipelineWorkspace, workspace),
            direction="qwen3_4b_to_8b",
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
        )
    resumed = _FakeSemanticEvaluator()
    semantic_module.run_semantic_stage(
        workspace=cast(V5PipelineWorkspace, workspace),
        direction="qwen3_4b_to_8b",
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        resume=True,
    )

    assert interrupted.calls == 3
    assert resumed.calls == 299
    assert workspace.completed_outputs is not None
    report_path = workspace.completed_outputs["semantic_report"]
    manifest_path = workspace.completed_outputs["semantic_manifest"]
    selective_path = workspace.completed_outputs["semantic_selective_manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["accepted_quality"]["accepted_count"] == 300
    assert report["baselines"][3]["accepted_count"] == 300
    assert report["runtime_authority"] is False
    manifest = semantic_module.V5SemanticManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    semantic_selective = SelectiveKVBridgeManifest.from_dict(
        json.loads(selective_path.read_text(encoding="utf-8"))
    )
    assert semantic_selective.state is ArtifactState.SEMANTIC_APPROVED
    assert semantic_selective.semantic_approved
    assert not semantic_selective.approved
    assert semantic_selective.runtime_cost is None
    assert semantic_selective.direct_injection is None
    kwargs = {
        "workspace": cast(V5PipelineWorkspace, workspace),
        "receipt": receipt,
        "validation": validation,
        "validation_selective": validation_selective,
        "risk_fit": risk_fit,
        "calibration": calibration,
        "candidate": candidate,
        "semantic_selective": semantic_selective,
    }
    assert manifest.validate(**kwargs) == []
    assert "semantic manifest does not carry a passing result" in replace(
        manifest, passed=cast(Any, 1)
    ).validate(**kwargs)

    expected_bindings = {
        record.sample_id: RiskPrefixTokenBinding(
            record.sample_id,
            record.token_bucket,
            _digest(record.sample_id),
        )
        for record in records
    }
    monkeypatch.setattr(
        semantic_module,
        "_recompute_prefix_bindings",
        lambda _workspace, _direction, _samples: expected_bindings,
    )
    semantic_module._load_and_validate_semantic_report(
        report_path,
        benchmark=cast(Any, benchmark),
        samples=tuple(zip(records, samples, strict=True)),
        workspace=cast(V5PipelineWorkspace, workspace),
        receipt=receipt,
        validation=validation,
        validation_selective=validation_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        transport_manifest=transport,
        candidate=candidate,
        manifest=manifest,
        predictor=predictor,
    )
    report["measurements"][0]["validation"]["accepted"] = True
    tampered = tmp_path / "tampered-semantic.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5PipelineError, match="runtime gate"):
        semantic_module._load_and_validate_semantic_report(
            tampered,
            benchmark=cast(Any, benchmark),
            samples=tuple(zip(records, samples, strict=True)),
            workspace=cast(V5PipelineWorkspace, workspace),
            receipt=receipt,
            validation=validation,
            validation_selective=validation_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport,
            candidate=candidate,
            manifest=manifest,
            predictor=predictor,
        )


def test_semantic_cli_has_no_snapshot_or_quality_override() -> None:
    parser = build_parser()
    semantic = next(
        action.choices["evaluate-semantic"]
        for action in parser._actions
        if getattr(action, "choices", None) and "evaluate-semantic" in action.choices
    )
    option_strings = {
        option for action in semantic._actions for option in getattr(action, "option_strings", ())
    }
    assert "--resume" in option_strings
    assert not any(
        word in option
        for option in option_strings
        for word in ("snapshot", "payload", "threshold", "quality", "coverage", "risk")
    )
    args = parser.parse_args(
        [
            "evaluate-semantic",
            "--workspace",
            "workspace",
            "--direction",
            "qwen3_8b_to_14b",
        ]
    )
    assert args.direction == "qwen3_8b_to_14b"
    assert not hasattr(args, "samples")
    assert not hasattr(args, "force")


def test_semantic_prefix_binding_is_minimal_and_strict() -> None:
    record = GroupedPrefixRecord(
        sample_id="semantic-0000",
        split="semantic_sealed_test",
        dataset_id="gsm8k",
        prefix_group_id="group",
        prefix_sha256=_digest("prefix"),
        suffix_query_sha256=_digest("suffix"),
        content_sha256=_digest("content"),
        token_bucket=128,
        task="qa",
    )
    binding = RiskPrefixTokenBinding(record.sample_id, 128, _digest("tokens"))
    assert binding.validate(record) == []
    assert "risk prefix binding token count changed" in replace(binding, token_count=127).validate(
        record
    )
    assert set(binding.__dict__) == {"sample_id", "token_count", "token_ids_sha256"}
    assert "shard" not in binding.__dict__
