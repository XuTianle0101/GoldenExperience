from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors import safe_open

import goldenexperience.size_variant.v5_real_risk as real_risk_module
import goldenexperience.size_variant.v5_risk as risk_module
from goldenexperience.benchmarks.publication import SPLIT_COUNTS, GroupedPrefixRecord
from goldenexperience.cli.v5_pipeline import DIRECTION_SIZES, build_parser
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.risk_gate import (
    RISK_FEATURE_DIM,
    RISK_FEATURE_SCHEMA_VERSION,
    RiskGateError,
    RiskPredictor,
    fit_risk_predictor,
)
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    TraceObjectRef,
)
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
from goldenexperience.size_variant.v5_real_risk import RealQwenRiskExampleEvaluator
from goldenexperience.size_variant.v5_risk import (
    RISK_LABEL_GENERATION_TOKENS,
    RiskHistory,
    RiskTrainingExample,
    RiskTrainingMetrics,
    RiskTrainingParameters,
    V5RiskFitManifest,
    run_fit_risk_stage,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _example(
    sample_id: str = "sample-0",
    prefix_group_id: str = "group",
    *,
    unsafe: bool = False,
    history: RiskHistory | None = None,
) -> RiskTrainingExample:
    history = history or RiskHistory()
    return RiskTrainingExample(
        sample_id=sample_id,
        prefix_group_id=prefix_group_id,
        features=(0.0,) * RISK_FEATURE_DIM,
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


def test_risk_example_and_history_are_causal_and_fail_closed() -> None:
    history = RiskHistory()
    safe = _example(history=history)
    assert safe.validate(expected_history=history) == []

    history = history.update(safe)
    unsafe = _example("sample-1", unsafe=True, history=history)
    assert unsafe.validate(expected_history=history) == []
    assert unsafe.greedy_agreement == 15 / 16

    history = history.update(unsafe)
    assert history.samples == 2
    assert history.failures == 1
    assert history.greedy_agreement == pytest.approx(31 / 32)
    assert history.validate() == []
    assert RiskHistory(samples=cast(Any, "bad")).validate()
    assert RiskHistory(greedy_agreement_sum=cast(Any, None)).validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"features": (math.nan,) * RISK_FEATURE_DIM}, "feature vector"),
        ({"features": cast(Any, [0.0] * RISK_FEATURE_DIM)}, "feature vector"),
        ({"unsafe": cast(Any, 1)}, "label must be boolean"),
        ({"native_task_score": cast(Any, "one")}, "native_task_score"),
        ({"greedy_tokens": 0}, "greedy counts"),
        ({"greedy_tokens": cast(Any, "16")}, "greedy counts"),
        ({"teacher_tokens": cast(Any, "16")}, "teacher token count"),
        ({"native_nll": cast(Any, None)}, "native_nll"),
        ({"bridge_nll": math.nan}, "bridge_nll"),
        ({"key_cosine": cast(Any, [])}, "key cosine"),
        ({"history_samples": cast(Any, "zero")}, "risk history counts"),
        ({"history_greedy_agreement": 0.0}, "history is invalid"),
        ({"sidecar_sha256": "bad"}, "sidecar_sha256"),
    ],
)
def test_risk_example_validation_handles_malformed_values_without_exceptions(
    changes: dict[str, Any],
    message: str,
) -> None:
    errors = replace(_example(), **changes).validate()
    assert any(message in error for error in errors)


def test_risk_example_rejects_label_and_history_leakage_and_clamps_overflow() -> None:
    assert (
        "risk example unsafe label is inconsistent" in replace(_example(), unsafe=True).validate()
    )
    assert "risk example uses non-causal history" in replace(
        _example(),
        history_samples=1,
        history_failures=0,
        history_greedy_agreement=1.0,
    ).validate(expected_history=RiskHistory())

    overflow = replace(
        _example(),
        unsafe=True,
        bridge_nll=sys.float_info.max,
    )
    assert overflow.perplexity_drift_pct == sys.float_info.max
    assert overflow.validate() == []


def test_risk_predictor_training_artifact_and_metrics_are_deterministic(tmp_path: Path) -> None:
    features = torch.zeros(4, RISK_FEATURE_DIM)
    features[1::2, 0] = 1.0
    labels = torch.tensor([0, 1, 0, 1])
    state = fit_risk_predictor(features, labels, epochs=3)
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    risk_module._save_predictor(first, state)
    risk_module._save_predictor(second, state)

    assert first.read_bytes() == second.read_bytes()
    expected = hashlib.sha256(first.read_bytes()).hexdigest()
    predictor = RiskPredictor.from_artifact(first, expected_sha256=expected)
    assert 0 <= predictor.unsafe_probability(features[0].tolist()) <= 1
    with safe_open(first, framework="pt", device="cpu") as handle:
        assert handle.metadata() == {
            "feature_schema_version": RISK_FEATURE_SCHEMA_VERSION,
            "hidden_size": "64",
        }

    metrics = risk_module._risk_training_metrics((0.5, 0.5), (0, 1))
    assert metrics.roc_auc == 0.5
    assert metrics.accuracy_at_half == 0.5
    assert metrics.training_objective == metrics.log_loss
    with pytest.raises(V5PipelineError, match="metric row"):
        risk_module._risk_training_metrics((0.5, 0.5), (cast(Any, False), 1))
    with pytest.raises(RiskGateError, match="both safe and unsafe"):
        fit_risk_predictor(features, torch.zeros(4), epochs=1)


def test_risk_manifest_binds_all_selector_training_inputs(monkeypatch) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "selector_train", 4)
    config = SimpleNamespace(
        pipeline_id="pipeline",
        code_sha256=_digest("code"),
        split_sha256={"selector_train": _digest("selector-split")},
        direction=lambda _direction: object(),
    )
    workspace = cast(V5PipelineWorkspace, SimpleNamespace(config=config))
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=_digest("raw"),
        content_sha256=lambda: _digest("trace"),
    )
    transport = SimpleNamespace(content_sha256=lambda: _digest("fit"))
    candidate = _candidate()
    predictor_sha = _digest("predictor")
    manifest = V5RiskFitManifest(
        pipeline_id="pipeline",
        direction=trace.direction,
        code_sha256=_digest("code"),
        selector_train_split_sha256=_digest("selector-split"),
        selector_trace_manifest_sha256=_digest("trace"),
        selector_raw_store_sha256=_digest("raw"),
        transport_fit_manifest_sha256=_digest("fit"),
        transport_weights_sha256=candidate.weights.sha256,
        risk_training_report_sha256=_digest("report"),
        predictor=TraceObjectRef(
            predictor_sha,
            f"objects/{predictor_sha[:2]}/{predictor_sha}.safetensors",
            10,
        ),
        training=RiskTrainingParameters(),
        metrics=RiskTrainingMetrics(0.1, 0.1, 0.75, 0.8),
        sample_count=4,
        unsafe_count=2,
        safe_count=2,
    )

    kwargs = {
        "workspace": workspace,
        "trace": trace,
        "transport_manifest": transport,
        "candidate": candidate,
    }
    assert manifest.validate(**kwargs) == []
    assert "selector-train risk fit cannot carry a calibrated threshold" in replace(
        manifest, calibrated=True
    ).validate(**kwargs)
    assert "risk fit selector split hash mismatch" in replace(
        manifest, selector_train_split_sha256=_digest("changed")
    ).validate(**kwargs)
    assert "risk fit requires both safe and unsafe selector examples" in replace(
        manifest, unsafe_count=0, safe_count=4
    ).validate(**kwargs)


class _FakeRiskWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.control = root / ".pipeline"
        self.control.mkdir(parents=True)
        self.config = SimpleNamespace(
            pipeline_id="pipeline",
            code_sha256=_digest("code"),
            split_sha256={"selector_train": _digest("selector-split")},
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
        assert stage == "fit_risk"
        assert parameters["label_generation_tokens"] == 16
        assert parameters["training"] == RiskTrainingParameters().to_dict()
        assert parameters["metrics_device"] == "cpu"
        assert parameters["evaluator"]["evaluator_id"] == "synthetic"
        if self.failures:
            assert resume
        return StageLease(direction, stage, _digest("stage-input"), "attempt")

    def publish_file(self, source_path: str | Path, *, logical_name: str) -> PipelineArtifact:
        assert logical_name == "risk_predictor"
        path = Path(source_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        return PipelineArtifact(
            sha256=digest,
            path=f"objects/{digest[:2]}/{digest}.safetensors",
            size_bytes=stat.st_size,
            device=stat.st_dev,
            inode=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )

    def complete_stage(
        self,
        lease: StageLease,
        *,
        outputs: dict[str, Path],
        metadata: dict[str, Any],
    ) -> PipelineStageRecord:
        assert metadata["sample_count"] == 4
        assert metadata["unsafe_count"] == 2
        assert metadata["safe_count"] == 2
        assert metadata["calibrated"] is False
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


class _FakeRiskEvaluator:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def __enter__(self) -> _FakeRiskEvaluator:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def evaluate(
        self,
        benchmark_record: GroupedPrefixRecord,
        _trace_record: Any,
        _sample: RawBenchmarkSample,
        history: RiskHistory,
    ) -> RiskTrainingExample:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic risk interruption")
        index = int(benchmark_record.sample_id.rsplit("-", maxsplit=1)[1])
        return _example(
            benchmark_record.sample_id,
            benchmark_record.prefix_group_id,
            unsafe=bool(index % 2),
            history=history,
        )


def _risk_rows() -> tuple[list[GroupedPrefixRecord], list[RawBenchmarkSample]]:
    records = []
    samples = []
    for index in range(4):
        sample_id = f"sample-{index}"
        records.append(
            GroupedPrefixRecord(
                sample_id=sample_id,
                split="selector_train",
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


def test_fit_risk_stage_resume_matches_uninterrupted_causal_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(SPLIT_COUNTS, "selector_train", 4)
    records, samples = _risk_rows()
    store = tmp_path / "selector.jsonl"
    store.write_text("synthetic selector store\n", encoding="utf-8")
    store_sha = hashlib.sha256(store.read_bytes()).hexdigest()
    trace = SimpleNamespace(
        direction="qwen3_4b_to_8b",
        raw_sample_store_sha256=store_sha,
        records=tuple(SimpleNamespace(sample_id=item.sample_id) for item in records),
        content_sha256=lambda: _digest("trace"),
    )
    transport = SimpleNamespace(content_sha256=lambda: _digest("fit"))
    candidate = _candidate()
    monkeypatch.setattr(risk_module, "load_bound_benchmark", lambda _workspace: object())
    monkeypatch.setattr(
        risk_module,
        "load_deployment_transport_binding",
        lambda _workspace, _direction: (transport, candidate, object()),
    )
    monkeypatch.setattr(
        risk_module,
        "load_completed_trace_manifest",
        lambda _workspace, _direction, split, _benchmark: (
            trace if split == "selector_train" else pytest.fail("fit-risk accessed another split")
        ),
    )
    monkeypatch.setattr(
        risk_module,
        "load_raw_sample_store",
        lambda _path, _benchmark, *, split: (
            tuple(zip(records, samples, strict=True))
            if split == "selector_train"
            else pytest.fail("fit-risk loaded another split")
        ),
    )

    resumed_workspace = _FakeRiskWorkspace(tmp_path / "resumed")
    interrupted = _FakeRiskEvaluator(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_fit_risk_stage(
            workspace=cast(V5PipelineWorkspace, resumed_workspace),
            direction="qwen3_4b_to_8b",
            sample_store_path=store,
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
            predictor_device="cpu",
        )
    resumed = _FakeRiskEvaluator()
    run_fit_risk_stage(
        workspace=cast(V5PipelineWorkspace, resumed_workspace),
        direction="qwen3_4b_to_8b",
        sample_store_path=store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        predictor_device="cpu",
        resume=True,
    )
    assert interrupted.calls == 3
    assert resumed.calls == 2

    continuous_workspace = _FakeRiskWorkspace(tmp_path / "continuous")
    continuous = _FakeRiskEvaluator()
    run_fit_risk_stage(
        workspace=cast(V5PipelineWorkspace, continuous_workspace),
        direction="qwen3_4b_to_8b",
        sample_store_path=store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: continuous,
        predictor_device="cpu",
    )
    assert continuous.calls == 4

    assert resumed_workspace.completed_outputs is not None
    assert continuous_workspace.completed_outputs is not None
    resumed_report = resumed_workspace.completed_outputs["risk_training_report"]
    continuous_report = continuous_workspace.completed_outputs["risk_training_report"]
    assert resumed_report.read_bytes() == continuous_report.read_bytes()
    report = json.loads(resumed_report.read_text(encoding="utf-8"))
    assert [item["history_samples"] for item in report["examples"]] == [0, 1, 2, 3]
    assert [item["history_failures"] for item in report["examples"]] == [0, 0, 1, 1]
    resumed_predictor = (
        resumed_workspace.control / "work/qwen3_4b_to_8b/fit_risk/risk_predictor.safetensors"
    )
    continuous_predictor = (
        continuous_workspace.control / "work/qwen3_4b_to_8b/fit_risk/risk_predictor.safetensors"
    )
    assert resumed_predictor.read_bytes() == continuous_predictor.read_bytes()


def test_fit_risk_production_surface_has_no_threshold_or_training_override() -> None:
    parameters = inspect.signature(run_fit_risk_stage).parameters
    assert "training" not in parameters
    assert "threshold" not in parameters
    assert "calibration" not in parameters

    parser = build_parser()
    fit_risk = next(
        action.choices["fit-risk"]
        for action in parser._actions
        if getattr(action, "choices", None) and "fit-risk" in action.choices
    )
    option_strings = {
        option for action in fit_risk._actions for option in getattr(action, "option_strings", ())
    }
    assert not any("threshold" in option or "calibr" in option for option in option_strings)
    for direction in DIRECTION_SIZES:
        args = parser.parse_args(
            [
                "fit-risk",
                "--workspace",
                "workspace",
                "--direction",
                direction,
                "--samples",
                "selector.jsonl",
            ]
        )
        assert args.direction == direction


def test_real_risk_evaluator_uses_round_tripped_source_only_features(monkeypatch) -> None:
    source = SimpleNamespace(dtype="bfloat16", max_position_embeddings=128)
    target = SimpleNamespace(dtype="bfloat16", max_position_embeddings=128)
    transport_manifest = SimpleNamespace(
        source=source,
        target=target,
        direction="qwen3_4b_to_8b",
    )
    evaluator = RealQwenRiskExampleEvaluator(
        workspace=cast(V5PipelineWorkspace, object()),
        transport_manifest=cast(Any, transport_manifest),
        candidate=_candidate(),
        source_path="/tmp/source",
        target_path="/tmp/target",
        source_device="cpu",
        target_device="cpu",
        identity_cache_path=None,
    )

    class FakeTokenizer:
        def __call__(self, text: str, **_kwargs: Any) -> Any:
            values = [1, 2] if text == "prefix" else [3]
            return SimpleNamespace(input_ids=torch.tensor([values]))

    class FakeModel:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def __call__(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(past_key_values=self.marker)

    class FakeTransport:
        def transform(self, source_kv: torch.Tensor, *, position_ids: torch.Tensor) -> torch.Tensor:
            assert position_ids.tolist() == [0, 1]
            return source_kv.clone()

    class FakeRuntimeSidecar:
        def risk_features(self) -> tuple[float, ...]:
            return (0.25,) * RISK_FEATURE_DIM

    class FakeSourceKVSidecar:
        @classmethod
        def from_bytes(cls, payload: bytes) -> FakeRuntimeSidecar:
            assert payload == b"quantized-sidecar"
            return FakeRuntimeSidecar()

    class FakeSidecar:
        def to_bytes(self) -> bytes:
            return b"quantized-sidecar"

    observed_history: list[tuple[int, int, float]] = []

    def fake_sidecar(_source_kv: Any, _transport: Any, **kwargs: Any) -> FakeSidecar:
        observed_history.append(
            (
                kwargs["history_samples"],
                kwargs["history_failures"],
                kwargs["history_greedy_agreement"],
            )
        )
        return FakeSidecar()

    source_kv = torch.ones(2, 1, 1, 2, 2)
    target_kv = torch.ones(2, 1, 1, 2, 2)
    monkeypatch.setattr(
        real_risk_module,
        "dynamic_cache_to_head_object",
        lambda marker: source_kv.clone() if marker == "source" else target_kv.clone(),
    )
    monkeypatch.setattr(real_risk_module, "build_transport_source_sidecar", fake_sidecar)
    monkeypatch.setattr(real_risk_module, "SourceKVSidecar", FakeSourceKVSidecar)
    decodes = iter(
        (
            ([1] * 16, "answer", 1.0),
            ([1] * 15 + [2], "wrong", 1.0),
        )
    )
    monkeypatch.setattr(real_risk_module, "greedy_decode", lambda *_args, **_kwargs: next(decodes))
    monkeypatch.setattr(real_risk_module, "teacher_nll", lambda *_args, **_kwargs: 1.0)
    evaluator.tokenizer = FakeTokenizer()
    evaluator.source_model = FakeModel("source")
    evaluator.target_model = FakeModel("target")
    evaluator.transport = cast(Any, FakeTransport())
    history = RiskHistory(samples=2, failures=1, greedy_agreement_sum=1.75)
    benchmark = GroupedPrefixRecord(
        sample_id="sample",
        split="selector_train",
        dataset_id="gsm8k",
        prefix_group_id="group",
        prefix_sha256=_digest("prefix"),
        suffix_query_sha256=_digest("suffix"),
        content_sha256=_digest("content"),
        token_bucket=128,
        task="qa",
    )
    trace = SimpleNamespace(
        sample_id="sample",
        token_count=2,
        token_ids_sha256=token_ids_sha256([1, 2]),
    )
    sample = RawBenchmarkSample(
        sample_id="sample",
        prefix_text="prefix",
        suffix_query="suffix",
        reference="answer",
        evaluation={"metric": "exact_match"},
        provenance={},
    )

    example = evaluator.evaluate(benchmark, trace, sample, history)

    assert example.features == (0.25,) * RISK_FEATURE_DIM
    assert example.unsafe is True
    assert example.greedy_matches == 15
    assert observed_history == [(2, 1, 0.875)]
    assert example.validate(expected_history=history) == []
    parameters = evaluator.parameters()
    assert parameters["sidecar_round_trip_before_features"] is True
    assert parameters["source_device_type"] == "cpu"


def test_real_risk_evaluator_fails_before_use_when_unloaded() -> None:
    evaluator = RealQwenRiskExampleEvaluator(
        workspace=cast(V5PipelineWorkspace, object()),
        transport_manifest=cast(
            Any,
            SimpleNamespace(
                source=SimpleNamespace(dtype="bfloat16", max_position_embeddings=128),
                target=SimpleNamespace(dtype="bfloat16", max_position_embeddings=128),
                direction="qwen3_4b_to_8b",
            ),
        ),
        candidate=_candidate(),
        source_path="/tmp/source",
        target_path="/tmp/target",
        source_device="cpu",
        target_device="cpu",
        identity_cache_path=None,
    )
    record = SimpleNamespace(sample_id="sample")
    sample = RawBenchmarkSample("sample", "prefix", "suffix", "answer", {}, {})
    with pytest.raises(V5PipelineError, match="not loaded"):
        evaluator.evaluate(cast(Any, record), cast(Any, record), sample, RiskHistory())
    with pytest.raises(V5PipelineError, match="tokenizer is not loaded"):
        evaluator.bind_semantic_prefix(cast(Any, record), sample)


def test_real_risk_evaluator_builds_minimal_semantic_prefix_binding() -> None:
    evaluator = RealQwenRiskExampleEvaluator(
        workspace=cast(V5PipelineWorkspace, object()),
        transport_manifest=cast(Any, SimpleNamespace()),
        candidate=_candidate(),
        source_path="/tmp/source",
        target_path="/tmp/target",
        source_device="cpu",
        target_device="cpu",
        identity_cache_path=None,
    )

    class FakeTokenizer:
        def __call__(self, _text: str, **_kwargs: Any) -> Any:
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    evaluator.tokenizer = FakeTokenizer()
    record = GroupedPrefixRecord(
        sample_id="semantic-sample",
        split="semantic_sealed_test",
        dataset_id="gsm8k",
        prefix_group_id="group",
        prefix_sha256=_digest("prefix"),
        suffix_query_sha256=_digest("suffix"),
        content_sha256=_digest("content"),
        token_bucket=2,
        task="qa",
    )
    sample = RawBenchmarkSample("semantic-sample", "prefix", "suffix", "answer", {}, {})

    binding = evaluator.bind_semantic_prefix(record, sample)

    assert binding.sample_id == record.sample_id
    assert binding.token_count == 2
    assert binding.token_ids_sha256 == token_ids_sha256([1, 2])
    assert binding.validate(record) == []
