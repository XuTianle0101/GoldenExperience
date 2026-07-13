from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import goldenexperience.runtime.lmcache_retrieve_transform as bridge_module
import goldenexperience.size_variant.v5_runtime as runtime_module
from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.runtime.lmcache_retrieve_transform import (
    RuntimeSourceIdentity,
    RuntimeStackIdentity,
)
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.risk_gate import (
    RISK_CALIBRATION_METHOD,
    RISK_FEATURE_DIM,
    RiskPredictor,
    clopper_pearson_upper_bound,
)
from goldenexperience.size_variant.selective_manifest import (
    AcceptedSubsetQualityEvidence,
    ArtifactState,
    RiskGateSpec,
    SelectiveKVBridgeManifest,
    SemanticSealedEvidence,
    TransportLossContract,
    TransportQualityEvidence,
    TransportSpec,
)
from goldenexperience.size_variant.v5_calibration import V5RiskCalibrationManifest
from goldenexperience.size_variant.v5_collect import RawBenchmarkSample, TraceObjectRef
from goldenexperience.size_variant.v5_fit import (
    CandidateTrainingMetrics,
    TransportCandidateArtifact,
)
from goldenexperience.size_variant.v5_pipeline import (
    PipelineArtifact,
    PipelineStageRecord,
    StageLease,
    V5PipelineError,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskHistory,
    RiskTrainingMetrics,
    RiskTrainingParameters,
    V5RiskFitManifest,
)
from goldenexperience.size_variant.v5_runtime import (
    RuntimeExecutionMeasurement,
    RuntimeFailureAudit,
    RuntimeRiskObservation,
    V5RuntimeManifest,
    build_runtime_summary,
    load_completed_runtime_audit,
    run_runtime_audit_stage,
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
    predictor = _digest("predictor")
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
            predictor,
            f"objects/{predictor[:2]}/{predictor}.safetensors",
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


def _quality(dataset: str, total: int, accepted: int) -> AcceptedSubsetQualityEvidence:
    return AcceptedSubsetQualityEvidence(
        evaluation_dataset_sha256=dataset,
        total_count=total,
        accepted_count=accepted,
        unsafe_count=0,
        coverage=accepted / total,
        native_task_score=1.0,
        bridge_task_score=1.0,
        task_score_drop_pct=0.0,
        greedy_agreement=1.0,
        perplexity_drift_pct=0.0,
        regression_risk_upper_bound=clopper_pearson_upper_bound(0, accepted),
        key_cosine=1.0,
    )


def _semantic_selective(
    workspace: Any,
    risk_fit: V5RiskFitManifest,
    calibration: V5RiskCalibrationManifest,
) -> SelectiveKVBridgeManifest:
    source = _model("Qwen/Qwen3-4B", 4.0, 3)
    target = _model("Qwen/Qwen3-8B", 8.0, 4)
    candidate = _candidate()
    validation_quality = _quality(workspace.config.split_sha256["validation"], 301, 300)
    semantic_quality = _quality(workspace.config.split_sha256["semantic_sealed_test"], 301, 300)
    base = SelectiveKVBridgeManifest(
        artifact_id="",
        direction="qwen3_4b_to_8b",
        source=source,
        target=target,
        transport=TransportSpec(
            weights_uri=candidate.weights.path,
            weights_sha256=candidate.weights.sha256,
            rank=32,
            source_window=3,
            loss=TransportLossContract(),
        ),
        risk_gate=calibration.risk_gate,
        benchmark_manifest_sha256=workspace.config.benchmark_manifest_sha256,
        transport_train_dataset_sha256=workspace.config.split_sha256["transport_train"],
        selector_train_dataset_sha256=workspace.config.split_sha256["selector_train"],
        method_dev_dataset_sha256=workspace.config.split_sha256["method_dev"],
        risk_calibration_dataset_sha256=workspace.config.split_sha256["risk_calibration"],
        validation_dataset_sha256=workspace.config.split_sha256["validation"],
        semantic_sealed_dataset_sha256=workspace.config.split_sha256["semantic_sealed_test"],
        runtime_audit_dataset_sha256=workspace.config.split_sha256["runtime_audit"],
        transport_quality=TransportQualityEvidence(
            evaluation_dataset_sha256=workspace.config.split_sha256["method_dev"],
            prompt_count=1024,
            task_score=0.99,
            oracle_safe_coverage=0.9,
            greedy_agreement=0.99,
        ),
        accepted_quality=validation_quality,
    ).with_content_id()
    sealed = SemanticSealedEvidence(
        dataset_sha256=base.semantic_sealed_dataset_sha256,
        report_sha256=_digest("semantic-report"),
        sample_count=301,
        code_sha256=workspace.config.code_sha256,
        transport_weights_sha256=base.transport.weights_sha256,
        predictor_sha256=risk_fit.predictor.sha256,
        threshold=calibration.risk_gate.threshold,
        quality=semantic_quality,
    )
    return replace(
        base,
        artifact_id="",
        state=ArtifactState.SEMANTIC_APPROVED,
        semantic_sealed=sealed,
    ).with_content_id()


def _rows(count: int) -> tuple[list[GroupedPrefixRecord], list[RawBenchmarkSample]]:
    records = []
    samples = []
    for index in range(count):
        sample_id = f"runtime-{index:04d}"
        records.append(
            GroupedPrefixRecord(
                sample_id=sample_id,
                split="runtime_audit",
                dataset_id="sharegpt",
                prefix_group_id="runtime-group",
                prefix_sha256=_digest(f"prefix-{index}"),
                suffix_query_sha256=_digest(f"suffix-{index}"),
                content_sha256=_digest(f"content-{index}"),
                token_bucket=128,
                task="trace",
            )
        )
        samples.append(
            RawBenchmarkSample(
                sample_id,
                f"prefix-{index}",
                f"suffix-{index}",
                None,
                {},
                {"timestamp": float(index)},
            )
        )
    return records, samples


def _stack() -> RuntimeStackIdentity:
    return RuntimeStackIdentity(
        lmcache_version="0.4.6",
        vllm_version="0.24.0",
        torch_version="2.11.0",
        cuda_version="13.0",
        sources=tuple(
            RuntimeSourceIdentity(
                module=name,
                distribution_relative_path=name.replace(".", "/") + ".py",
                sha256=_digest(name),
            )
            for name in bridge_module._PINNED_RUNTIME_MODULES
        ),
    )


class _Predictor:
    def unsafe_probability(self, features: Any) -> float:
        return float(features[0])


class _Evaluator:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0
        self.warmups: list[int] = []

    def __enter__(self) -> _Evaluator:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def warmup(self, iterations: int) -> None:
        self.warmups.append(iterations)

    def build_observation(
        self,
        record: GroupedPrefixRecord,
        _trace: Any,
        _sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RuntimeRiskObservation:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic runtime interruption")
        index = int(record.sample_id.rsplit("-", 1)[1])
        probability = 0.1 if index <= 101 else 0.9
        features = [0.0] * RISK_FEATURE_DIM
        features[0] = probability
        return RuntimeRiskObservation(
            sample_id=record.sample_id,
            prefix_group_id=record.prefix_group_id,
            features=tuple(features),
            shadow_failure=False,
            greedy_matches=RISK_LABEL_GENERATION_TOKENS,
            greedy_tokens=RISK_LABEL_GENERATION_TOKENS,
            native_nll=1.0,
            bridge_nll=1.0,
            teacher_tokens=RISK_LABEL_GENERATION_TOKENS,
            history_samples=history.samples,
            history_failures=history.failures,
            history_greedy_agreement=history.greedy_agreement,
            sidecar_sha256=_digest(f"sidecar-{record.sample_id}"),
            native_tokens_sha256=_digest(f"native-tokens-{record.sample_id}"),
            bridge_tokens_sha256=_digest(f"bridge-tokens-{record.sample_id}"),
        )

    def measure(
        self,
        _record: GroupedPrefixRecord,
        trace: Any,
        _sample: RawBenchmarkSample,
        _observation: RuntimeRiskObservation,
        *,
        accepted: bool,
        decision: str,
    ) -> RuntimeExecutionMeasurement:
        return RuntimeExecutionMeasurement(
            native_prefill_ms=100.0,
            native_ttft_ms=200.0 if accepted else 100.0,
            observed_ttft_ms=130.0 if accepted else 104.0,
            materialization_ms=50.0 if accepted else None,
            retrieve_transform_success=accepted,
            load_complete_published=accepted,
            source_read_attempted=accepted,
            source_chunks_read=1 if accepted else 0,
            tokens_scattered=trace.token_count if accepted else 0,
            fallback_reason="none" if accepted else decision,
            target_mooncake_puts=0,
            backing_files_remaining=0,
        )

    def audit_failure_recovery(self) -> RuntimeFailureAudit:
        return RuntimeFailureAudit(
            probe_id=_digest("failure-probe"),
            paged_slot_mapping_verified=True,
            load_complete_after_all_layers=True,
            partial_failure_invalidates_blocks=True,
            native_prefill_overwrites_invalid_blocks=True,
            injected_failure_count=1,
            invalidated_block_count=1,
            recomputed_token_count=128,
            accepted_target_mooncake_puts=0,
            backing_files_remaining=0,
        )


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            code_sha256=_digest("code"),
            benchmark_manifest_sha256=_digest("benchmark"),
            split_sha256={name: _digest(name) for name in SPLIT_COUNTS},
            direction=lambda _direction: object(),
        )
        self.failures = 0
        self.completed_outputs: dict[str, Path] | None = None
        self.completed_stage: PipelineStageRecord | None = None
        self.artifact_paths: dict[str, Path] = {}

    def begin_stage(
        self,
        direction: str,
        stage: str,
        *,
        parameters: dict[str, Any],
        resume: bool = False,
    ) -> StageLease:
        assert direction == "qwen3_4b_to_8b"
        assert stage == "runtime_audit"
        assert parameters["warmup_iterations"] == 20
        assert parameters["minimum_measurements_per_path"] == 100
        assert parameters["measurement_protocol"] == "isolated_paired_request_latency_v1"
        assert parameters["request_order"] == "lexicographic_sample_id"
        assert parameters["arrival_timestamps_replayed"] is False
        assert parameters["shadow_policy"] == (
            "reference_free_greedy_agreement_lt_0.98_or_perplexity_drift_gt_2pct"
        )
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
        assert metadata["sample_count"] == 202
        assert metadata["accepted_count"] == 101
        assert metadata["rejected_count"] == 101
        assert metadata["authority"] == "approved"
        self.completed_outputs = outputs
        artifacts: dict[str, PipelineArtifact] = {}
        for name, path in outputs.items():
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            artifact = PipelineArtifact(
                sha256=sha256,
                path=f"objects/{sha256[:2]}/{sha256}",
                size_bytes=stat.st_size,
                device=stat.st_dev,
                inode=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )
            artifacts[name] = artifact
            self.artifact_paths[artifact.path] = path
        self.completed_stage = PipelineStageRecord(
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
            outputs=artifacts,
        )
        return self.completed_stage

    def fail_stage(self, _lease: StageLease, _error: Exception) -> None:
        self.failures += 1

    def state(self) -> Any:
        stages = {}
        if self.completed_stage is not None:
            stages["qwen3_4b_to_8b/runtime_audit"] = self.completed_stage
        return SimpleNamespace(stages=stages)

    def artifact_path(self, artifact: PipelineArtifact, *, verify_hash: bool) -> Path:
        assert verify_hash
        path = self.artifact_paths[artifact.path]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
        return path


def test_runtime_stage_resumes_recomputes_and_grants_final_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "runtime_audit", 202)
    monkeypatch.setitem(SPLIT_COUNTS, "risk_calibration", 300)
    monkeypatch.setitem(SPLIT_COUNTS, "validation", 301)
    monkeypatch.setitem(SPLIT_COUNTS, "semantic_sealed_test", 301)
    workspace = _Workspace(tmp_path / "workspace")
    risk_fit = _risk_fit()
    calibration = _calibration(risk_fit)
    candidate = _candidate()
    semantic_selective = _semantic_selective(workspace, risk_fit, calibration)
    assert semantic_selective.validate() == []
    semantic = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        semantic_report_sha256=_digest("semantic-report"),
        content_sha256=lambda: _digest("semantic-manifest"),
    )
    transport = SimpleNamespace(content_sha256=lambda: _digest("fit"))
    records, samples = _rows(202)
    benchmark = SimpleNamespace(records=tuple(records))
    store = tmp_path / "runtime.jsonl"
    store.write_text("runtime audit store\n", encoding="utf-8")
    store_sha = hashlib.sha256(store.read_bytes()).hexdigest()
    traces = tuple(
        SimpleNamespace(sample_id=record.sample_id, token_count=record.token_bucket)
        for record in records
    )
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=store_sha,
        records=traces,
        content_sha256=lambda: _digest("runtime-trace"),
    )
    stack = _stack()
    predictor = cast(RiskPredictor, _Predictor())
    monkeypatch.setattr(runtime_module, "load_bound_benchmark", lambda _workspace: benchmark)
    monkeypatch.setattr(
        runtime_module,
        "load_completed_semantic",
        lambda _workspace, _direction: (semantic, semantic_selective),
    )
    monkeypatch.setattr(
        runtime_module,
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
        runtime_module,
        "load_completed_trace_manifest",
        lambda _workspace, _direction, split, _benchmark: (
            trace if split == "runtime_audit" else pytest.fail("wrong runtime split")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_raw_sample_store",
        lambda _path, _benchmark, *, split: (
            tuple(zip(records, samples, strict=True))
            if split == "runtime_audit"
            else pytest.fail("wrong raw runtime split")
        ),
    )
    monkeypatch.setattr(runtime_module, "probe_runtime_stack", lambda: stack)
    monkeypatch.setattr(runtime_module, "verify_runtime_stack_identity", lambda _stack: None)
    monkeypatch.setattr(
        runtime_module,
        "load_risk_predictor",
        lambda _workspace, _risk_fit, *, device: (
            predictor if device == "cpu" else pytest.fail("runtime predictor was not on CPU")
        ),
    )
    interrupted = _Evaluator(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_runtime_audit_stage(
            workspace=cast(V5PipelineWorkspace, workspace),
            direction="qwen3_4b_to_8b",
            sample_store_path=store,
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
        )
    resumed = _Evaluator()
    run_runtime_audit_stage(
        workspace=cast(V5PipelineWorkspace, workspace),
        direction="qwen3_4b_to_8b",
        sample_store_path=store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        resume=True,
    )

    assert interrupted.calls == 3
    assert resumed.calls == 200
    assert interrupted.warmups == [20]
    assert resumed.warmups == [20]
    assert workspace.completed_outputs is not None
    report_path = workspace.completed_outputs["runtime_report"]
    manifest_path = workspace.completed_outputs["runtime_manifest"]
    approved_path = workspace.completed_outputs["approved_selective_manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["runtime_summary"]["eligible_for_approval"] is True
    assert report["runtime_summary"]["accepted_p95_ttft_reduction_pct"] == 35.0
    assert report["runtime_summary"]["rejected_p95_fallback_overhead_pct"] == 4.0
    assert report["measurement_protocol"] == "isolated_paired_request_latency_v1"
    assert report["request_order"] == "lexicographic_sample_id"
    assert report["arrival_timestamps_replayed"] is False
    assert report["shadow_policy"] == (
        "reference_free_greedy_agreement_lt_0.98_or_perplexity_drift_gt_2pct"
    )
    assert "native_task_score" not in report["measurements"][0]["observation"]
    assert "bridge_task_score" not in report["measurements"][0]["observation"]
    manifest = V5RuntimeManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    approved = SelectiveKVBridgeManifest.from_dict(
        json.loads(approved_path.read_text(encoding="utf-8"))
    )
    assert approved.state is ArtifactState.APPROVED
    assert approved.approved
    kwargs = {
        "workspace": cast(V5PipelineWorkspace, workspace),
        "trace": trace,
        "semantic": semantic,
        "semantic_selective": semantic_selective,
        "risk_fit": risk_fit,
        "calibration": calibration,
        "candidate": candidate,
        "approved": approved,
    }
    assert manifest.validate(**kwargs) == []
    assert "runtime manifest does not carry a passing result" in replace(
        manifest, passed=cast(Any, 1)
    ).validate(**kwargs)

    loaded_manifest, loaded_approved = load_completed_runtime_audit(
        cast(V5PipelineWorkspace, workspace),
        "qwen3_4b_to_8b",
    )
    assert loaded_manifest == manifest
    assert loaded_approved == approved

    runtime_module._load_and_validate_runtime_report(
        report_path,
        benchmark=cast(Any, benchmark),
        workspace=cast(V5PipelineWorkspace, workspace),
        trace=trace,
        semantic=semantic,
        semantic_selective=semantic_selective,
        risk_fit=risk_fit,
        calibration=calibration,
        transport_manifest=transport,
        candidate=candidate,
        manifest=manifest,
        predictor=predictor,
    )
    with pytest.raises(V5PipelineError, match="manifest counts differ"):
        runtime_module._load_and_validate_runtime_report(
            report_path,
            benchmark=cast(Any, benchmark),
            workspace=cast(V5PipelineWorkspace, workspace),
            trace=trace,
            semantic=semantic,
            semantic_selective=semantic_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport,
            candidate=candidate,
            manifest=replace(
                manifest,
                accepted_count=manifest.accepted_count + 1,
                rejected_count=manifest.rejected_count - 1,
            ),
            predictor=predictor,
        )
    report["measurements"][0]["observation"]["shadow_failure"] = True
    bad_shadow = tmp_path / "tampered-runtime-shadow.json"
    bad_shadow.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5PipelineError, match="reference-free shadow failure"):
        runtime_module._load_and_validate_runtime_report(
            bad_shadow,
            benchmark=cast(Any, benchmark),
            workspace=cast(V5PipelineWorkspace, workspace),
            trace=trace,
            semantic=semantic,
            semantic_selective=semantic_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport,
            candidate=candidate,
            manifest=manifest,
            predictor=predictor,
        )
    report["measurements"][0]["observation"]["shadow_failure"] = False
    report["measurements"][0]["execution"]["source_read_attempted"] = True
    tampered = tmp_path / "tampered-runtime.json"
    tampered.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5PipelineError, match="read source KV"):
        runtime_module._load_and_validate_runtime_report(
            tampered,
            benchmark=cast(Any, benchmark),
            workspace=cast(V5PipelineWorkspace, workspace),
            trace=trace,
            semantic=semantic,
            semantic_selective=semantic_selective,
            risk_fit=risk_fit,
            calibration=calibration,
            transport_manifest=transport,
            candidate=candidate,
            manifest=manifest,
            predictor=predictor,
        )


def test_runtime_summary_requires_both_measured_paths(monkeypatch) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "runtime_audit", 100)
    example = SimpleNamespace()
    execution = RuntimeExecutionMeasurement(
        100.0,
        200.0,
        130.0,
        50.0,
        True,
        True,
        True,
        1,
        128,
        "none",
        0,
        0,
    )
    measurements = [
        SimpleNamespace(accepted=True, execution=execution, example=example) for _ in range(100)
    ]
    with pytest.raises(V5PipelineError, match="accepted and rejected"):
        build_runtime_summary(
            direction="qwen3_4b_to_8b",
            dataset_sha256=_digest("runtime"),
            measurements=cast(Any, measurements),
            failure_audit=_Evaluator().audit_failure_recovery(),
        )


def test_runtime_execution_rejects_source_reads_before_gate_acceptance() -> None:
    execution = RuntimeExecutionMeasurement(
        native_prefill_ms=100.0,
        native_ttft_ms=100.0,
        observed_ttft_ms=104.0,
        materialization_ms=None,
        retrieve_transform_success=False,
        load_complete_published=False,
        source_read_attempted=True,
        source_chunks_read=1,
        tokens_scattered=0,
        fallback_reason="predicted_unsafe",
        target_mooncake_puts=0,
        backing_files_remaining=0,
    )

    errors = execution.validate(
        accepted=False,
        decision="predicted_unsafe",
        expected_tokens=128,
    )

    assert "rejected runtime request read source KV" in errors
