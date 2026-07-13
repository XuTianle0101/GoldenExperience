import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import goldenexperience.benchmarks.publication as publication_module
import goldenexperience.size_variant.v5_real_method_dev as real_method_dev_module
from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    SPLIT_COUNTS,
    DatasetSource,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec, sha256_file
from goldenexperience.size_variant.v5_collect import (
    RAW_BENCHMARK_SAMPLE_SCHEMA,
    RawBenchmarkSample,
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    publication_sample_content_sha256,
)
from goldenexperience.size_variant.v5_fit import (
    CandidateTrainingMetrics,
    TransportCandidateArtifact,
    TransportTrainingParameters,
    V5TransportFitManifest,
)
from goldenexperience.size_variant.v5_method_dev import (
    METHOD_DEV_GENERATION_TOKENS,
    CandidateMethodDevMetrics,
    MethodDevMeasurement,
    freeze_transport_structure,
    load_frozen_transport_structure,
    run_method_dev_stage,
)
from goldenexperience.size_variant.v5_pipeline import (
    V5DirectionConfig,
    V5PipelineConfig,
    V5PipelineWorkspace,
)
from goldenexperience.size_variant.v5_real_method_dev import RealQwenMethodDevEvaluator


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def method_dev_contract(monkeypatch):
    for split in tuple(SPLIT_COUNTS):
        monkeypatch.setitem(SPLIT_COUNTS, split, 4)
    monkeypatch.setitem(SPLIT_COUNTS, "method_dev", 1024)
    monkeypatch.setattr(publication_module, "REQUIRED_DATASETS", frozenset({"gsm8k"}))


def _model(model_id: str, size: float) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=size,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest(f"{model_id}-config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest(f"{model_id}-weights"),
        num_layers=3,
        num_key_value_heads=1,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=40960,
        chat_template_sha256=_digest("chat"),
    )


def _raw_payload(sample_id: str) -> dict:
    return {
        "schema_version": RAW_BENCHMARK_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "prefix_text": f"prefix {sample_id}",
        "suffix_query": f"query {sample_id}",
        "reference": "answer",
        "evaluation": {"metric": "exact_match"},
        "provenance": {"row": sample_id},
    }


def _benchmark(tmp_path: Path) -> tuple[PublicationBenchmarkManifest, Path]:
    records = []
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            sample_id = f"{split}-{index}"
            raw = _raw_payload(sample_id)
            records.append(
                GroupedPrefixRecord(
                    sample_id=sample_id,
                    split=split,
                    dataset_id="gsm8k",
                    prefix_group_id=f"{split}-group-{index}",
                    prefix_sha256=_digest(raw["prefix_text"]),
                    suffix_query_sha256=_digest(raw["suffix_query"]),
                    content_sha256=publication_sample_content_sha256(
                        prefix_text=raw["prefix_text"],
                        suffix_query=raw["suffix_query"],
                        reference=raw["reference"],
                        evaluation=raw["evaluation"],
                        task="qa",
                    ),
                    token_bucket=PREFIX_BUCKETS[index % len(PREFIX_BUCKETS)],
                    task="qa",
                )
            )
    source = DatasetSource(
        dataset_id="gsm8k",
        revision="test",
        content_sha256=_digest("gsm8k"),
        license_id="MIT",
        license_uri="https://example.invalid/license",
        source_uri="https://example.invalid/gsm8k",
    )
    provisional = PublicationBenchmarkManifest(
        sources=(source,),
        records=tuple(records),
        split_sha256={},
        tokenizer_sha256=_digest("tokenizer"),
        chat_template_sha256=_digest("chat"),
        sealed_payload_sha256=_digest("sealed"),
    )
    manifest = replace(
        provisional,
        split_sha256={split: provisional.compute_split_sha256(split) for split in SPLIT_COUNTS},
    )
    assert manifest.validate() == []
    path = tmp_path / "benchmark.json"
    manifest.save(path)
    return manifest, path


def _workspace(
    tmp_path: Path,
    manifest: PublicationBenchmarkManifest,
    manifest_path: Path,
) -> V5PipelineWorkspace:
    models = {
        "4b": _model("Qwen/Qwen3-4B", 4.0),
        "8b": _model("Qwen/Qwen3-8B", 8.0),
        "14b": _model("Qwen/Qwen3-14B", 14.0),
    }
    pairs = {
        "qwen3_4b_to_8b": ("4b", "8b"),
        "qwen3_8b_to_4b": ("8b", "4b"),
        "qwen3_8b_to_14b": ("8b", "14b"),
        "qwen3_14b_to_8b": ("14b", "8b"),
    }
    directions = tuple(
        V5DirectionConfig(
            direction=direction,
            source_model_path=str(tmp_path / source_size),
            target_model_path=str(tmp_path / target_size),
            source=models[source_size],
            target=models[target_size],
        )
        for direction, (source_size, target_size) in pairs.items()
    )
    config = V5PipelineConfig.from_benchmark(
        manifest,
        manifest_path=manifest_path,
        code_sha256=_digest("code"),
        directions=directions,
    )
    return V5PipelineWorkspace.create(tmp_path / "workspace", config)


def _publish_trace(
    tmp_path: Path,
    workspace: V5PipelineWorkspace,
    manifest: PublicationBenchmarkManifest,
    *,
    split: str,
    raw_store_sha256: str,
    shard: TraceObjectRef,
) -> V5TraceManifest:
    direction = workspace.config.direction("qwen3_4b_to_8b")

    def group_shard(group_id: str) -> TraceObjectRef:
        digest = _digest(f"{shard.sha256}:{group_id}")
        return TraceObjectRef(
            sha256=digest,
            path=f"objects/{digest[:2]}/{digest}.safetensors",
            size_bytes=shard.size_bytes,
        )

    records = tuple(
        TraceRecord(
            sample_id=item.sample_id,
            prefix_group_id=item.prefix_group_id,
            dataset_id=item.dataset_id,
            task=item.task,
            token_bucket=item.token_bucket,
            content_sha256=item.content_sha256,
            prefix_sha256=item.prefix_sha256,
            suffix_query_sha256=item.suffix_query_sha256,
            token_ids_sha256=_digest(f"tokens-{item.sample_id}"),
            token_count=item.token_bucket,
            query_sample_count=1,
            key_sample_count=1,
            shard=group_shard(item.prefix_group_id),
        )
        for item in manifest.records
        if item.split == split
    )
    trace = V5TraceManifest(
        pipeline_id=workspace.config.pipeline_id,
        direction=direction.direction,
        split=split,
        split_sha256=manifest.split_sha256[split],
        benchmark_manifest_sha256=manifest.content_sha256(),
        code_sha256=workspace.config.code_sha256,
        raw_sample_store_sha256=raw_store_sha256,
        source=direction.source,
        target=direction.target,
        collector={"collector_id": "synthetic"},
        records=records,
    )
    assert trace.validate(workspace=workspace, benchmark=manifest) == []
    path = tmp_path / f"{split}-trace.json"
    path.write_text(json.dumps(trace.to_dict(), sort_keys=True))
    lease = workspace.begin_stage(
        direction.direction,
        f"collect_{split}",
        parameters={"collector": "synthetic", "split": split},
    )
    workspace.complete_stage(
        lease,
        outputs={"trace_manifest": path},
        metadata={"record_count": len(records)},
    )
    return trace


def _publish_fit(
    tmp_path: Path,
    workspace: V5PipelineWorkspace,
    trace: V5TraceManifest,
    weights: TraceObjectRef,
) -> V5TransportFitManifest:
    training = TransportTrainingParameters()
    expected_samples = len(trace.records) * training.epochs
    expected_steps = training.epochs
    metrics = CandidateTrainingMetrics(
        samples=expected_samples,
        optimizer_steps=expected_steps,
        native_generation=1.0,
        prompt_tail_distillation=0.1,
        attention_logit_kl=0.2,
        attention_output_mse=0.3,
        transformed_kv_anchor=0.4,
        total=1.5,
    )
    candidates = tuple(
        TransportCandidateArtifact(
            candidate_id=f"transport-r{rank}-s{seed}",
            rank=rank,
            seed=seed,
            deployment_seed=seed == 17,
            weights=weights,
            parameter_count=1,
            metrics=metrics,
        )
        for rank in training.ranks
        for seed in training.seeds
    )
    fit = V5TransportFitManifest(
        pipeline_id=workspace.config.pipeline_id,
        direction=trace.direction,
        code_sha256=workspace.config.code_sha256,
        transport_train_split_sha256=trace.split_sha256,
        trace_manifest_sha256=trace.content_sha256(),
        normalizer_sha256=_digest("normalizer"),
        source=trace.source,
        target=trace.target,
        training=training,
        candidates=candidates,
    )
    assert fit.validate(workspace=workspace, trace=trace) == []
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(fit.to_dict(), sort_keys=True))
    lease = workspace.begin_stage(
        trace.direction,
        "fit_transport",
        parameters={"fit": "synthetic"},
    )
    workspace.complete_stage(
        lease,
        outputs={"transport_fit_manifest": path},
        metadata={"candidate_count": 9},
    )
    return fit


class _FakeEvaluator:
    def __init__(self, fit: V5TransportFitManifest, *, fail_after: int | None = None) -> None:
        self.fit = fit
        self.fail_after = fail_after
        self.calls = 0
        self.entered = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *_args):
        return None

    def evaluate(self, record, sample):
        del sample
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic method-dev interruption")
        rank_score = {32: 0.96, 64: 0.99, 128: 0.98}
        return tuple(
            MethodDevMeasurement(
                sample_id=record.sample_id,
                candidate_id=candidate.candidate_id,
                rank=candidate.rank,
                seed=candidate.seed,
                native_task_score=1.0,
                bridge_task_score=rank_score[candidate.rank],
                task_pass_threshold=0.95,
                greedy_matches=METHOD_DEV_GENERATION_TOKENS,
                greedy_tokens=METHOD_DEV_GENERATION_TOKENS,
                native_nll=1.0,
                bridge_nll=1.0,
                teacher_tokens=METHOD_DEV_GENERATION_TOKENS,
                transform_ms=float(candidate.rank) / 32 + candidate.seed / 1000,
                native_prediction_sha256=_digest(f"native-{record.sample_id}"),
                bridge_prediction_sha256=_digest(
                    f"bridge-{record.sample_id}-{candidate.candidate_id}"
                ),
                native_tokens_sha256=_digest(f"native-tokens-{record.sample_id}"),
                bridge_tokens_sha256=_digest(
                    f"bridge-tokens-{record.sample_id}-{candidate.candidate_id}"
                ),
            )
            for candidate in self.fit.candidates
        )


def _prepared_pipeline(tmp_path: Path):
    manifest, manifest_path = _benchmark(tmp_path)
    workspace = _workspace(tmp_path, manifest, manifest_path)
    dummy = tmp_path / "dummy.safetensors"
    dummy.write_bytes(b"synthetic-object")
    object_ref = TraceObjectRef.from_artifact(
        workspace.publish_file(dummy, logical_name="synthetic_object")
    )
    train_trace = _publish_trace(
        tmp_path,
        workspace,
        manifest,
        split="transport_train",
        raw_store_sha256=_digest("train-raw"),
        shard=object_ref,
    )
    fit = _publish_fit(tmp_path, workspace, train_trace, object_ref)
    sample_store = tmp_path / "method-dev.jsonl"
    sample_store.write_text(
        "\n".join(
            json.dumps(_raw_payload(item.sample_id), sort_keys=True)
            for item in manifest.records
            if item.split == "method_dev"
        )
        + "\n"
    )
    method_trace = _publish_trace(
        tmp_path,
        workspace,
        manifest,
        split="method_dev",
        raw_store_sha256=sha256_file(sample_store),
        shard=object_ref,
    )
    return workspace, fit, method_trace, sample_store


def test_method_dev_stage_resumes_and_freezes_three_seed_rank(
    tmp_path: Path,
    method_dev_contract,
) -> None:
    workspace, fit, _trace, sample_store = _prepared_pipeline(tmp_path)
    interrupted = _FakeEvaluator(fit, fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_method_dev_stage(
            workspace=workspace,
            direction=fit.direction,
            sample_store_path=sample_store,
            evaluator_parameters={"evaluator_id": "synthetic"},
            evaluator_factory=lambda: interrupted,
        )

    resumed = _FakeEvaluator(fit)
    stage = run_method_dev_stage(
        workspace=workspace,
        direction=fit.direction,
        sample_store_path=sample_store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: resumed,
        resume=True,
    )

    assert stage.status == "completed"
    assert resumed.calls == 1022
    structure, loaded_fit, method_trace = load_frozen_transport_structure(workspace)
    assert loaded_fit.content_sha256() == fit.content_sha256()
    assert method_trace.split == "method_dev"
    assert structure.selected_rank == 64
    assert structure.deployment_seed == 17
    assert structure.deployment_candidate_id == "transport-r64-s17"
    assert structure.deployment_quality.task_score == pytest.approx(0.99)
    assert structure.deployment_quality.oracle_safe_coverage == 1.0

    unused = _FakeEvaluator(fit)
    reused = run_method_dev_stage(
        workspace=workspace,
        direction=fit.direction,
        sample_store_path=sample_store,
        evaluator_parameters={"evaluator_id": "synthetic"},
        evaluator_factory=lambda: unused,
    )
    assert reused.receipt_sha256 == stage.receipt_sha256
    assert unused.entered == 0


def test_method_dev_metrics_fail_closed_on_nonfinite_or_inconsistent_values() -> None:
    candidate = TransportCandidateArtifact(
        candidate_id="candidate",
        rank=32,
        seed=17,
        deployment_seed=True,
        weights=TraceObjectRef(
            _digest("weights"), f"objects/00/{_digest('weights')}.safetensors", 1
        ),
        parameter_count=1,
        metrics=CandidateTrainingMetrics(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    measurement = MethodDevMeasurement(
        sample_id="sample",
        candidate_id="candidate",
        rank=32,
        seed=17,
        native_task_score=1.0,
        bridge_task_score=1.0,
        task_pass_threshold=1.0,
        greedy_matches=16,
        greedy_tokens=16,
        native_nll=1.0,
        bridge_nll=1.0,
        teacher_tokens=16,
        transform_ms=1.0,
        native_prediction_sha256=_digest("native"),
        bridge_prediction_sha256=_digest("bridge"),
        native_tokens_sha256=_digest("native-tokens"),
        bridge_tokens_sha256=_digest("bridge-tokens"),
    )
    metrics = CandidateMethodDevMetrics.aggregate(candidate, (measurement,))

    assert metrics.task_score == 1.0
    assert metrics.oracle_safe_coverage == 1.0
    assert replace(measurement, bridge_nll=float("nan")).validate()
    assert replace(metrics, safe_count=0).validate(expected_prompts=1)


def test_frozen_structure_recomputes_rank_aggregates(
    tmp_path: Path,
    method_dev_contract,
) -> None:
    workspace, fit, method_trace, _sample_store = _prepared_pipeline(tmp_path)
    evaluator = _FakeEvaluator(fit)
    measurements = []
    for record in method_trace.records:
        measurements.extend(evaluator.evaluate(record, _raw_payload(record.sample_id)))
    structure = freeze_transport_structure(
        workspace=workspace,
        fit=fit,
        method_trace=method_trace,
        measurements=measurements,
        report_sha256=_digest("report"),
    )
    tampered = replace(
        structure,
        rank_aggregates=(
            replace(structure.rank_aggregates[0], mean_task_score=1.0),
            *structure.rank_aggregates[1:],
        ),
    )

    assert "frozen structure rank aggregate is inconsistent" in tampered.validate(
        workspace=workspace,
        fit=fit,
        method_trace=method_trace,
    )


def test_real_method_dev_reuses_prefill_but_times_each_candidate(monkeypatch) -> None:
    source = SimpleNamespace(max_position_embeddings=128)
    target = SimpleNamespace(max_position_embeddings=128)
    candidates = (
        SimpleNamespace(candidate_id="candidate-32", rank=32, seed=17),
        SimpleNamespace(candidate_id="candidate-64", rank=64, seed=17),
    )
    fit = SimpleNamespace(source=source, target=target, candidates=candidates)
    evaluator = RealQwenMethodDevEvaluator(
        workspace=cast(V5PipelineWorkspace, object()),
        fit=cast(Any, fit),
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

        def decode(self, _tokens: list[int], **_kwargs: Any) -> str:
            return "answer"

    class FakeModel:
        def __init__(self, marker: str) -> None:
            self.marker = marker
            self.calls = 0

        def __call__(self, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(past_key_values=self.marker)

    class FakeTransport:
        def __init__(self) -> None:
            self.calls = 0

        def transform(self, source_kv: torch.Tensor, *, position_ids: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert position_ids.tolist() == [0, 1]
            return source_kv.clone()

    source_kv = torch.ones(2, 1, 1, 2, 2)
    target_kv = torch.ones(2, 1, 1, 2, 2)
    monkeypatch.setattr(
        real_method_dev_module,
        "dynamic_cache_to_head_object",
        lambda marker: source_kv.clone() if marker == "source" else target_kv.clone(),
    )
    monkeypatch.setattr(
        real_method_dev_module,
        "greedy_decode",
        lambda *_args, **_kwargs: ([1] * METHOD_DEV_GENERATION_TOKENS, "answer", 1.0),
    )
    monkeypatch.setattr(real_method_dev_module, "teacher_nll", lambda *_args, **_kwargs: 1.0)
    source_model = FakeModel("source")
    target_model = FakeModel("target")
    transports = {candidate.candidate_id: FakeTransport() for candidate in candidates}
    evaluator.tokenizer = FakeTokenizer()
    evaluator.source_model = source_model
    evaluator.target_model = target_model
    evaluator.transports = cast(Any, transports)
    record = SimpleNamespace(
        sample_id="sample-1",
        prefix_group_id="shared-group",
        token_count=2,
        token_ids_sha256=token_ids_sha256([1, 2]),
    )
    sample = RawBenchmarkSample(
        sample_id="sample-1",
        prefix_text="prefix",
        suffix_query="suffix-1",
        reference="answer",
        evaluation={"metric": "exact_match"},
        provenance={},
    )

    first = evaluator.evaluate(cast(Any, record), sample)
    second = evaluator.evaluate(
        cast(Any, SimpleNamespace(**{**vars(record), "sample_id": "sample-2"})),
        replace(sample, sample_id="sample-2", suffix_query="suffix-2"),
    )

    assert len(first) == len(second) == len(candidates)
    assert source_model.calls == 1
    assert target_model.calls == 1
    assert {name: transport.calls for name, transport in transports.items()} == {
        "candidate-32": 2,
        "candidate-64": 2,
    }
    assert all(measurement.transform_ms >= 0 for measurement in (*first, *second))
