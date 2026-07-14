import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import goldenexperience.size_variant.v5_fit as v5_fit_module
from goldenexperience.size_variant.attention_collection import causal_sample_mask
from goldenexperience.size_variant.cached_kv_manifest import CachedKVModelSpec, sha256_file
from goldenexperience.size_variant.head_aware_transport import (
    HeadAwareKVTransport,
    _apply_rope_heads,
    build_trainable_head_aware_transport,
    fit_head_aware_normalizers,
    fit_head_aware_ridge_initializer,
    initialize_trainable_from_ridge,
    transport_artifact_metadata,
)
from goldenexperience.size_variant.v5_collect import (
    TraceObjectRef,
    TraceRecord,
    V5TraceManifest,
    _sampled_attention_output,
    load_trace_shard,
    trace_shard_metadata,
)
from goldenexperience.size_variant.v5_fit import (
    DEPLOYMENT_SEED,
    REGISTERED_RANKS,
    CandidateTrainingMetrics,
    SynchronousTransportTrainer,
    TransportCandidateArtifact,
    TransportTrainingParameters,
    V5TransportFitManifest,
    _row_weighted_unique_records,
    _TraceLoader,
    _validate_progress,
    run_fit_transport_stage,
)
from goldenexperience.size_variant.v5_pipeline import V5PipelineError, V5PipelineWorkspace


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _model(model_id: str) -> CachedKVModelSpec:
    return CachedKVModelSpec(
        model_id=model_id,
        parameter_count_b=1.0 if model_id == "source" else 2.0,
        revision="test",
        architecture="qwen3",
        config_sha256=_digest(f"{model_id}-config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest(f"{model_id}-weights"),
        num_layers=3,
        num_key_value_heads=1,
        head_dim=32,
        dtype="bfloat16",
        rope_theta=1_000_000,
        max_position_embeddings=4096,
        chat_template_sha256=_digest("chat"),
    )


def _synthetic_trace(
    tmp_path: Path,
    *,
    record_count: int = 2,
) -> tuple[V5PipelineWorkspace, V5TraceManifest]:
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = cast(V5PipelineWorkspace, SimpleNamespace(root=root))
    source = _model("source")
    target = _model("target")
    records = []
    for index in range(record_count):
        sample_id = f"sample-{index}"
        provisional = TraceRecord(
            sample_id=sample_id,
            prefix_group_id=f"group-{index}",
            dataset_id="synthetic",
            task="qa",
            token_bucket=128,
            content_sha256=_digest(f"content-{index}"),
            prefix_sha256=_digest(f"prefix-{index}"),
            suffix_query_sha256=_digest(f"suffix-{index}"),
            token_ids_sha256=_digest(f"tokens-{index}"),
            token_count=4,
            query_sample_count=2,
            key_sample_count=4,
            shard=TraceObjectRef("0" * 64, "objects/00/placeholder.safetensors", 1),
        )
        generator = torch.Generator().manual_seed(100 + index)
        source_kv = torch.randn(2, 3, 1, 4, 32, generator=generator).to(torch.bfloat16)
        target_kv = torch.randn(2, 3, 1, 4, 32, generator=generator).to(torch.bfloat16)
        target_query = torch.randn(3, 1, 2, 32, generator=generator).to(torch.bfloat16)
        query_positions = torch.tensor([1, 3], dtype=torch.int64)
        key_positions = torch.arange(4, dtype=torch.int64)
        mask = causal_sample_mask(query_positions, key_positions)
        native_output = _sampled_attention_output(
            target_query,
            target_kv[0],
            target_kv[1],
            mask,
        )
        tensors = {
            "source_kv": source_kv,
            "target_kv": target_kv,
            "target_query": target_query,
            "native_attention_output": native_output,
            "full_native_attention_output": native_output.to(torch.bfloat16),
            "query_positions": query_positions,
            "key_positions": key_positions,
            "causal_mask": mask,
            "constant_losses": torch.tensor([1.0, 0.1], dtype=torch.float32),
        }
        temporary = root / f"{sample_id}.safetensors"
        save_file(tensors, temporary, metadata=trace_shard_metadata(provisional, source, target))
        digest = sha256_file(temporary)
        destination = root / "objects" / digest[:2] / f"{digest}.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        destination.chmod(0o444)
        records.append(
            replace(
                provisional,
                shard=TraceObjectRef(
                    digest,
                    destination.relative_to(root).as_posix(),
                    destination.stat().st_size,
                ),
            )
        )
    trace = V5TraceManifest(
        pipeline_id="synthetic-pipeline",
        direction="qwen3_4b_to_8b",
        split="transport_train",
        split_sha256=_digest("transport-train"),
        benchmark_manifest_sha256=_digest("benchmark"),
        code_sha256=_digest("code"),
        raw_sample_store_sha256=_digest("raw"),
        source=source,
        target=target,
        collector={"collector_id": "synthetic"},
        records=tuple(records),
    )
    return workspace, trace


def _parameters(*, epochs: int = 1) -> TransportTrainingParameters:
    return TransportTrainingParameters(
        ranks=(32,),
        seeds=(17,),
        deployment_seed=17,
        source_window=3,
        epochs=epochs,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_accumulation=1,
        max_grad_norm=1.0,
    )


def _trainer(
    workspace: V5PipelineWorkspace,
    trace: V5TraceManifest,
    *,
    epochs: int = 1,
    progress=None,
) -> SynchronousTransportTrainer:
    return SynchronousTransportTrainer(
        workspace=workspace,
        trace=trace,
        parameters=_parameters(epochs=epochs),
        device="cpu",
        checkpoint_every_steps=1,
        progress=progress,
    )


def _runtime_state(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        tensors = {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118
        }
    return tensors, metadata


def test_synchronous_transport_fit_emits_runtime_loadable_weights(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)
    fitted, normalizer_sha256, initializer_sha256 = _trainer(workspace, trace).fit(
        tmp_path / "work",
        stage_input_sha256=_digest("stage-input"),
    )

    assert len(fitted) == 1
    candidate = fitted[0]
    assert candidate.metrics.samples == 2
    assert candidate.metrics.optimizer_steps == 2
    assert len(normalizer_sha256) == 64
    assert len(initializer_sha256) == 64
    state, metadata = _runtime_state(candidate.path)
    assert metadata == transport_artifact_metadata(
        direction=trace.direction,
        source=trace.source,
        target=trace.target,
        spec=_transport_spec(_parameters()),
    )
    runtime = HeadAwareKVTransport(
        trace.source,
        trace.target,
        _transport_spec(_parameters()),
        state,
    )
    first = trace.records[0]
    source_kv = load_trace_shard(
        workspace.root / first.shard.path,
        first,
        source=trace.source,
        target=trace.target,
    )["source_kv"]
    transformed = runtime.transform(source_kv, position_ids=torch.arange(4))
    assert transformed.shape == (2, 3, 1, 4, 32)
    assert bool(torch.isfinite(transformed).all())


def _known_affine_batch() -> tuple[
    CachedKVModelSpec,
    CachedKVModelSpec,
    Any,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    source = _model("source")
    target = _model("target")
    spec = _transport_spec(_parameters())
    module = build_trainable_head_aware_transport(source, target, spec)
    generator = torch.Generator().manual_seed(911)
    source_kv = torch.randn(2, 3, 1, 96, 32, generator=generator).to(torch.bfloat16)
    positions = torch.arange(96)
    source_value = source_kv.float()
    source_key = _apply_rope_heads(
        source_value[0],
        positions,
        theta=source.rope_theta,
        inverse=True,
    )
    mixed = torch.stack((module._mix(source_key), module._mix(source_value[1])))
    weight = torch.eye(32).view(1, 1, 1, 32, 32).expand(2, 3, 1, -1, -1).clone()
    weight += 0.01 * torch.randn(weight.shape, generator=generator)
    bias = 0.05 * torch.randn(2, 3, 1, 32, generator=generator)
    target_unrotated = torch.einsum("alhti,alhid->alhtd", mixed, weight) + bias.unsqueeze(3)
    target_key = _apply_rope_heads(
        target_unrotated[0],
        positions,
        theta=target.rope_theta,
        inverse=False,
    )
    target_kv = torch.stack((target_key, target_unrotated[1])).to(torch.bfloat16)
    return source, target, spec, source_kv, target_kv, positions


def test_full_rank_ridge_initializer_recovers_a_known_affine_map() -> None:
    source, target, spec, source_kv, target_kv, positions = _known_affine_batch()
    module = build_trainable_head_aware_transport(source, target, spec)
    fit_head_aware_normalizers(module, ((source_kv, positions),))
    initializer = fit_head_aware_ridge_initializer(
        module,
        ((source_kv, target_kv, positions),),
        ridge_ratio=1e-6,
    )
    initialize_trainable_from_ridge(module, initializer, seed=17)
    runtime = HeadAwareKVTransport(
        source,
        target,
        spec,
        module.runtime_state(),
        compute_dtype=torch.float32,
    )

    transformed = runtime.transform(source_kv, position_ids=positions)

    assert torch.nn.functional.mse_loss(transformed.float(), target_kv.float()) < 2e-3
    assert (
        torch.nn.functional.cosine_similarity(
            transformed.float().reshape(-1, 32),
            target_kv.float().reshape(-1, 32),
            dim=-1,
        ).mean()
        > 0.999
    )


def test_ridge_rank_truncation_is_seeded_reproducible_and_function_preserving() -> None:
    source, target, spec, source_kv, target_kv, positions = _known_affine_batch()
    fitted = build_trainable_head_aware_transport(source, target, spec)
    fit_head_aware_normalizers(fitted, ((source_kv, positions),))
    initializer = fit_head_aware_ridge_initializer(
        fitted,
        ((source_kv, target_kv, positions),),
        ridge_ratio=1e-6,
    )

    def initialized(seed: int):
        module = build_trainable_head_aware_transport(source, target, spec)
        with torch.no_grad():
            for prefix in ("key", "value"):
                for suffix in ("normalizer_mean", "normalizer_scale"):
                    name = f"{prefix}_{suffix}"
                    getattr(module, name).copy_(getattr(fitted, name))
        initialize_trainable_from_ridge(module, initializer, seed=seed)
        return module

    deployment = initialized(17)
    alternate = initialized(29)
    repeated = initialized(29)

    torch.testing.assert_close(alternate.key_down, repeated.key_down, atol=0, rtol=0)
    assert not torch.equal(deployment.key_down, alternate.key_down)
    torch.testing.assert_close(
        torch.matmul(deployment.key_down, deployment.key_up),
        torch.matmul(alternate.key_down, alternate.key_up),
        atol=2e-5,
        rtol=2e-5,
    )


def test_weighted_ridge_matches_expanded_frozen_rows() -> None:
    source, target, spec, source_kv, target_kv, positions = _known_affine_batch()
    second_source = torch.flip(source_kv, dims=(3,))
    second_target = torch.flip(target_kv, dims=(3,))

    weighted = build_trainable_head_aware_transport(source, target, spec)
    fit_head_aware_normalizers(
        weighted,
        ((source_kv, positions, 3), (second_source, positions, 1)),
    )
    weighted_state = fit_head_aware_ridge_initializer(
        weighted,
        (
            (source_kv, target_kv, positions, 3),
            (second_source, second_target, positions, 1),
        ),
        ridge_ratio=1e-6,
    )

    expanded = build_trainable_head_aware_transport(source, target, spec)
    expanded_rows = ((source_kv, positions),) * 3 + ((second_source, positions),)
    fit_head_aware_normalizers(expanded, expanded_rows)
    expanded_state = fit_head_aware_ridge_initializer(
        expanded,
        ((source_kv, target_kv, positions),) * 3 + ((second_source, second_target, positions),),
        ridge_ratio=1e-6,
    )

    for name in weighted_state:
        torch.testing.assert_close(weighted_state[name], expanded_state[name], atol=1e-5, rtol=1e-5)


def test_trace_loader_reuses_only_compatibly_bound_shared_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace, trace = _synthetic_trace(tmp_path, record_count=1)
    real_load = v5_fit_module.load_trace_shard
    calls = 0

    def counted_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(v5_fit_module, "load_trace_shard", counted_load)
    loader = _TraceLoader(workspace, trace)
    first = trace.records[0]
    compatible = replace(
        first,
        sample_id="sample-shared",
        content_sha256=_digest("shared-content"),
        suffix_query_sha256=_digest("shared-suffix"),
    )

    loaded = loader.load(first)

    assert loader.load(compatible) is loaded
    assert calls == 1
    with pytest.raises(V5PipelineError, match="inconsistent identity bindings"):
        loader.load(replace(compatible, prefix_group_id="different-group"))
    assert calls == 1


def test_ridge_initializer_groups_shared_shards_by_frozen_row_count(tmp_path: Path) -> None:
    _, trace = _synthetic_trace(tmp_path)
    first, second = trace.records
    shared = replace(
        second,
        prefix_group_id=first.prefix_group_id,
        prefix_sha256=first.prefix_sha256,
        token_ids_sha256=first.token_ids_sha256,
        token_count=first.token_count,
        query_sample_count=first.query_sample_count,
        key_sample_count=first.key_sample_count,
        shard=first.shard,
    )

    grouped = _row_weighted_unique_records(replace(trace, records=(first, shared)))

    assert grouped == ((first, 2),)


def test_transport_trainer_rejects_non_training_trace(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)

    with pytest.raises(V5PipelineError, match="transport-train"):
        _trainer(workspace, replace(trace, split="method_dev"))


def _transport_spec(parameters: TransportTrainingParameters):
    return parameters.transport_spec(
        weights_uri="candidate.safetensors",
        weights_sha256="0" * 64,
        rank=parameters.ranks[0],
    )


def test_transport_checkpoint_resume_matches_uninterrupted_fit(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)

    def interrupt(index: int, _total: int, _epoch: int, _sample_id: str) -> None:
        if index == 2:
            raise RuntimeError("synthetic interruption")

    resumed_work = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _trainer(workspace, trace, epochs=2, progress=interrupt).fit(
            resumed_work,
            stage_input_sha256=_digest("stage-input"),
        )
    resumed, resumed_normalizer, resumed_initializer = _trainer(workspace, trace, epochs=2).fit(
        resumed_work,
        stage_input_sha256=_digest("stage-input"),
    )
    uninterrupted, uninterrupted_normalizer, uninterrupted_initializer = _trainer(
        workspace, trace, epochs=2
    ).fit(tmp_path / "uninterrupted", stage_input_sha256=_digest("stage-input"))

    assert resumed_normalizer == uninterrupted_normalizer
    assert resumed_initializer == uninterrupted_initializer
    assert resumed[0].metrics == uninterrupted[0].metrics
    assert sha256_file(resumed[0].path) == sha256_file(uninterrupted[0].path)


def test_transport_checkpoint_rejects_corruption_and_wrong_binding(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)
    work = tmp_path / "work"

    def interrupt(index: int, _total: int, _epoch: int, _sample_id: str) -> None:
        if index == 2:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _trainer(workspace, trace, progress=interrupt).fit(
            work,
            stage_input_sha256=_digest("stage-input"),
        )
    pointer = json.loads((work / "checkpoint_set.json").read_text())
    item = next(iter(pointer["files"].values()))
    checkpoint = work / item["path"]
    checkpoint.chmod(0o644)
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    checkpoint.chmod(0o444)
    with pytest.raises(V5PipelineError, match="checksum"):
        _trainer(workspace, trace).fit(work, stage_input_sha256=_digest("stage-input"))

    clean_work = tmp_path / "clean"
    _trainer(workspace, trace).fit(clean_work, stage_input_sha256=_digest("stage-input"))
    with pytest.raises(V5PipelineError, match="normalizer checkpoint input binding"):
        _trainer(workspace, trace).fit(
            clean_work,
            stage_input_sha256=_digest("different-input"),
        )


def test_transport_checkpoint_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)
    work = tmp_path / "work"
    _trainer(workspace, trace).fit(work, stage_input_sha256=_digest("stage-input"))
    pointer_path = work / "checkpoint_set.json"
    pointer = json.loads(pointer_path.read_text())
    item = next(iter(pointer["files"].values()))
    checkpoint = work / item["path"]
    tensors, metadata = _runtime_state(checkpoint)
    metric_sums = json.loads(metadata["metric_sums"])
    metric_sums["total"] = float("nan")
    metadata["metric_sums"] = json.dumps(metric_sums)
    replacement = checkpoint.with_suffix(".replacement")
    save_file(tensors, replacement, metadata=metadata)
    replacement.chmod(0o444)
    replacement.replace(checkpoint)
    item["sha256"] = sha256_file(checkpoint)
    item["size_bytes"] = checkpoint.stat().st_size
    pointer_path.write_text(json.dumps(pointer))

    with pytest.raises(V5PipelineError, match="metrics are invalid"):
        _trainer(workspace, trace).fit(work, stage_input_sha256=_digest("stage-input"))


def test_transport_progress_requires_exact_integer_optimizer_boundaries() -> None:
    with pytest.raises(V5PipelineError, match="values are invalid"):
        _validate_progress({"epoch": 0, "position": 1.5, "optimizer_steps": 0, "samples_seen": 0})
    with pytest.raises(V5PipelineError, match="optimizer boundary"):
        _validate_progress(
            {"epoch": 0, "position": 1, "optimizer_steps": 1, "samples_seen": 1},
            record_count=4,
            epochs=2,
            gradient_accumulation=2,
        )


def test_publication_transport_runner_has_no_matrix_override() -> None:
    parameters = inspect.signature(run_fit_transport_stage).parameters

    assert "parameters" not in parameters
    assert "require_registered" not in parameters
    assert TransportTrainingParameters().ranks == REGISTERED_RANKS
    assert TransportTrainingParameters().deployment_seed == DEPLOYMENT_SEED


def test_fit_manifest_requires_the_complete_registered_matrix(tmp_path: Path) -> None:
    workspace, trace = _synthetic_trace(tmp_path)
    config = SimpleNamespace(
        pipeline_id=trace.pipeline_id,
        code_sha256=trace.code_sha256,
        split_sha256={"transport_train": trace.split_sha256},
    )
    workspace.config = config  # type: ignore[attr-defined]
    training = TransportTrainingParameters()
    expected_samples = len(trace.records) * training.epochs
    expected_steps = training.epochs
    reference = TraceObjectRef(
        _digest("candidate"),
        f"objects/00/{_digest('candidate')}.safetensors",
        1,
    )
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
            candidate_id=f"r{rank}-s{seed}",
            rank=rank,
            seed=seed,
            deployment_seed=seed == DEPLOYMENT_SEED,
            weights=reference,
            parameter_count=1,
            metrics=metrics,
        )
        for rank in training.ranks
        for seed in training.seeds
    )
    manifest = V5TransportFitManifest(
        pipeline_id=trace.pipeline_id,
        direction=trace.direction,
        code_sha256=trace.code_sha256,
        transport_train_split_sha256=trace.split_sha256,
        trace_manifest_sha256=trace.content_sha256(),
        normalizer_sha256=_digest("normalizer"),
        source=trace.source,
        target=trace.target,
        training=training,
        candidates=candidates,
        training_initializer_sha256=_digest("initializer"),
    )

    assert manifest.validate(workspace=workspace, trace=trace) == []
    incomplete = replace(manifest, candidates=manifest.candidates[:-1])
    assert "transport fit candidate rank/seed matrix is incomplete" in incomplete.validate(
        workspace=workspace,
        trace=trace,
    )
