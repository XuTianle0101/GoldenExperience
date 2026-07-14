import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import goldenexperience.size_variant.v5_directional_fit as directional_module
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec
from goldenexperience.size_variant.selective_manifest import TransportQualityEvidence
from goldenexperience.size_variant.v5_collect import TraceObjectRef, TraceRecord, V5TraceManifest
from goldenexperience.size_variant.v5_directional_fit import (
    V5_RIDGE_DIRECTIONAL_FIT_SCHEMA,
    V5_TARGET_LOGIT_DIRECTIONAL_FIT_SCHEMA,
    V5DirectionalTransportFitManifest,
    frozen_direction_training_parameters,
    run_frozen_direction_fit_stage,
)
from goldenexperience.size_variant.v5_fit import (
    CandidateTrainingMetrics,
    TransportCandidateArtifact,
)
from goldenexperience.size_variant.v5_generation import GenerationSupervisionSpec
from goldenexperience.size_variant.v5_method_dev import FrozenTransportStructure
from goldenexperience.size_variant.v5_pipeline import (
    PipelineArtifact,
    PipelineStageRecord,
    StageLease,
    V5PipelineWorkspace,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _structure() -> FrozenTransportStructure:
    return FrozenTransportStructure(
        pipeline_id="pipeline",
        direction="qwen3_4b_to_8b",
        code_sha256=_digest("code"),
        benchmark_manifest_sha256=_digest("benchmark"),
        method_dev_split_sha256=_digest("method-dev"),
        method_dev_trace_manifest_sha256=_digest("method-trace"),
        method_dev_raw_store_sha256=_digest("method-raw"),
        transport_fit_manifest_sha256=_digest("fit"),
        method_dev_report_sha256=_digest("report"),
        generation_tokens=16,
        selection_rule=(
            "lexicographic_mean_task_preservation_then_oracle_safe_coverage_then_"
            "greedy_agreement_then_negative_mean_p95_transform_ms"
        ),
        seed_aggregation="arithmetic_mean_and_population_standard_deviation",
        selected_rank=64,
        source_window=3,
        deployment_seed=17,
        deployment_candidate_id="screening-r64-s17",
        deployment_weights=TraceObjectRef(
            _digest("screening-weights"),
            f"objects/00/{_digest('screening-weights')}.safetensors",
            1,
        ),
        candidates=(),
        rank_aggregates=(),
        deployment_quality=TransportQualityEvidence(
            evaluation_dataset_sha256=_digest("method-dev"),
            prompt_count=1024,
            task_score=0.99,
            oracle_safe_coverage=0.5,
            greedy_agreement=0.99,
        ),
    )


def _trace(source: CachedKVModelSpec, target: CachedKVModelSpec) -> V5TraceManifest:
    shard = TraceObjectRef(
        _digest("shard"),
        f"objects/00/{_digest('shard')}.safetensors",
        1,
    )
    records = tuple(
        TraceRecord(
            sample_id=f"sample-{index}",
            prefix_group_id=f"group-{index}",
            dataset_id="synthetic",
            task="qa",
            token_bucket=128,
            content_sha256=_digest(f"content-{index}"),
            prefix_sha256=_digest(f"prefix-{index}"),
            suffix_query_sha256=_digest(f"suffix-{index}"),
            token_ids_sha256=_digest(f"tokens-{index}"),
            token_count=128,
            query_sample_count=1,
            key_sample_count=1,
            shard=shard,
        )
        for index in range(4)
    )
    return V5TraceManifest(
        pipeline_id="pipeline",
        direction="qwen3_8b_to_4b",
        split="transport_train",
        split_sha256=_digest("transport-train"),
        benchmark_manifest_sha256=_digest("benchmark"),
        code_sha256=_digest("code"),
        raw_sample_store_sha256=_digest("raw"),
        source=source,
        target=target,
        collector={"collector_id": "synthetic"},
        records=records,
    )


def test_directional_manifest_is_locked_to_selected_rank_and_seed() -> None:
    source = _model("Qwen/Qwen3-8B", 8.0)
    target = _model("Qwen/Qwen3-4B", 4.0)
    trace = _trace(source, target)
    structure = _structure()
    training = frozen_direction_training_parameters(structure)
    weights = TraceObjectRef(
        _digest("weights"),
        f"objects/00/{_digest('weights')}.safetensors",
        1,
    )
    candidate = TransportCandidateArtifact(
        candidate_id="direction-r64-s17",
        rank=64,
        seed=17,
        deployment_seed=True,
        weights=weights,
        parameter_count=1,
        metrics=CandidateTrainingMetrics(
            samples=12,
            optimizer_steps=3,
            native_generation=1.0,
            prompt_tail_distillation=0.1,
            attention_logit_kl=0.2,
            attention_output_mse=0.3,
            transformed_kv_anchor=0.4,
            total=1.5,
        ),
    )
    config = SimpleNamespace(
        pipeline_id="pipeline",
        code_sha256=_digest("code"),
        split_sha256={"transport_train": _digest("transport-train")},
        direction=lambda _name: object(),
    )
    workspace = cast(V5PipelineWorkspace, SimpleNamespace(config=config))
    manifest = V5DirectionalTransportFitManifest(
        pipeline_id="pipeline",
        direction=trace.direction,
        code_sha256=_digest("code"),
        transport_train_split_sha256=trace.split_sha256,
        trace_manifest_sha256=trace.content_sha256(),
        frozen_structure_sha256=structure.content_sha256(),
        normalizer_sha256=_digest("normalizer"),
        source=source,
        target=target,
        training=training,
        candidates=(candidate,),
        training_initializer_sha256=_digest("initializer"),
        generation_sample_store_sha256=trace.raw_sample_store_sha256,
    )

    assert manifest.validate(workspace=workspace, trace=trace, structure=structure) == []
    assert training.ranks == (64,)
    assert training.seeds == (17,)
    wrong_rank = replace(manifest, training=replace(training, ranks=(32,)))
    assert "directional transport training differs from the frozen contract" in wrong_rank.validate(
        workspace=workspace,
        trace=trace,
        structure=structure,
    )
    wrong_seed = replace(manifest, candidates=(replace(candidate, seed=29),))
    assert "directional transport candidate differs from frozen structure" in wrong_seed.validate(
        workspace=workspace,
        trace=trace,
        structure=structure,
    )
    v2_payload = manifest.to_dict()
    v2_payload["schema_version"] = V5_RIDGE_DIRECTIONAL_FIT_SCHEMA
    v2_payload["training"].pop("generation")
    v2_payload["training"].pop("full_prefix")
    v2_payload.pop("generation_sample_store_sha256")

    loaded_v2 = V5DirectionalTransportFitManifest.from_dict(v2_payload)

    assert loaded_v2.to_dict() == v2_payload
    assert loaded_v2.validate(workspace=workspace, trace=trace, structure=structure) == []

    v3_payload = manifest.to_dict()
    v3_payload["schema_version"] = V5_TARGET_LOGIT_DIRECTIONAL_FIT_SCHEMA
    v3_payload["training"]["generation"] = GenerationSupervisionSpec().to_dict()
    v3_payload["training"].pop("full_prefix")
    loaded_v3 = V5DirectionalTransportFitManifest.from_dict(v3_payload)

    assert loaded_v3.to_dict() == v3_payload
    assert loaded_v3.validate(workspace=workspace, trace=trace, structure=structure) == []


def test_directional_production_runner_has_no_training_override() -> None:
    parameters = inspect.signature(run_frozen_direction_fit_stage).parameters

    assert "training" not in parameters
    assert "rank" not in parameters
    assert "seed" not in parameters
    assert parameters["source_device"].default == "cuda:0"


def test_directional_runner_emits_one_frozen_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _model("Qwen/Qwen3-8B", 8.0)
    target = _model("Qwen/Qwen3-4B", 4.0)
    trace = _trace(source, target)
    structure = _structure()
    sample_store = tmp_path / "transport_train.jsonl"
    sample_store.write_text("synthetic transport train\n", encoding="utf-8")
    trace = replace(
        trace,
        raw_sample_store_sha256=hashlib.sha256(sample_store.read_bytes()).hexdigest(),
    )

    class FakeWorkspace:
        def __init__(self) -> None:
            self.control = tmp_path / ".pipeline"
            self.config = SimpleNamespace(
                pipeline_id="pipeline",
                code_sha256=_digest("code"),
                split_sha256={"transport_train": _digest("transport-train")},
                direction=lambda _name: SimpleNamespace(
                    source_model_path=tmp_path / "source",
                    target_model_path=tmp_path / "target",
                ),
            )
            self.completed_outputs = None

        def begin_stage(self, direction, stage, *, parameters, resume=False):
            assert direction == trace.direction
            assert stage == "fit_transport"
            assert parameters["frozen_structure_sha256"] == structure.content_sha256()
            assert resume is False
            return StageLease(direction, stage, _digest("stage-input"), "attempt")

        def publish_file(self, source_path, *, logical_name):
            path = Path(source_path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stat = path.stat()
            assert logical_name == "transport_r64_s17"
            return PipelineArtifact(
                sha256=digest,
                path=f"objects/{digest[:2]}/{digest}.safetensors",
                size_bytes=stat.st_size,
                device=stat.st_dev,
                inode=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )

        def complete_stage(self, lease, *, outputs, metadata):
            self.completed_outputs = outputs
            assert metadata["selected_rank"] == 64
            return PipelineStageRecord(
                direction=lease.direction,
                stage=lease.stage,
                status="completed",
                input_sha256=lease.input_sha256,
                attempt_id=lease.attempt_id,
                attempt_count=1,
                started_at="2026-01-01T00:00:00+00:00",
                completed_at="2026-01-01T00:00:01+00:00",
                receipt_sha256=_digest("receipt"),
                receipt_path="receipts/receipt.json",
                outputs={},
            )

        def fail_stage(self, _lease, _error):
            raise AssertionError("directional runner unexpectedly failed")

    class FakeTrainer:
        def __init__(self, **kwargs) -> None:
            assert kwargs["parameters"].ranks == (64,)
            assert kwargs["parameters"].seeds == (17,)

        def fit(self, work, *, stage_input_sha256):
            assert stage_input_sha256 == _digest("stage-input")
            path = Path(work) / "weights" / "direction-r64-s17.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"directional-weights")
            return (
                [
                    SimpleNamespace(
                        candidate_id="direction-r64-s17",
                        rank=64,
                        seed=17,
                        path=path,
                        parameter_count=1,
                        metrics=CandidateTrainingMetrics(
                            samples=12,
                            optimizer_steps=3,
                            native_generation=1.0,
                            prompt_tail_distillation=0.1,
                            attention_logit_kl=0.2,
                            attention_output_mse=0.3,
                            transformed_kv_anchor=0.4,
                            total=1.5,
                        ),
                    )
                ],
                _digest("normalizer"),
                _digest("initializer"),
            )

    workspace = FakeWorkspace()
    monkeypatch.setattr(directional_module, "load_bound_benchmark", lambda _workspace: object())
    monkeypatch.setattr(
        directional_module,
        "load_frozen_transport_structure",
        lambda _workspace: (structure, object(), object()),
    )
    monkeypatch.setattr(
        directional_module,
        "load_completed_trace_manifest",
        lambda *_args: trace,
    )
    monkeypatch.setattr(
        directional_module,
        "load_raw_sample_store",
        lambda *_args, **_kwargs: tuple(
            (SimpleNamespace(sample_id=item.sample_id), SimpleNamespace())
            for item in trace.records
        ),
    )
    monkeypatch.setattr(directional_module, "SynchronousTransportTrainer", FakeTrainer)

    stage = run_frozen_direction_fit_stage(
        workspace=cast(V5PipelineWorkspace, workspace),
        direction=trace.direction,
        sample_store_path=sample_store,
        identity_cache_path=None,
        source_device="cpu",
        device="cpu",
    )

    assert stage.status == "completed"
    manifest_path = workspace.completed_outputs["transport_fit_manifest"]
    manifest = V5DirectionalTransportFitManifest.from_dict(
        json.loads(Path(manifest_path).read_text())
    )
    assert len(manifest.candidates) == 1
    assert manifest.candidates[0].rank == 64
    assert manifest.candidates[0].seed == 17
