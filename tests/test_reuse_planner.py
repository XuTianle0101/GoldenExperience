from dataclasses import replace

from goldenexperience.lmcache_patch import CrossModelCacheKey
from goldenexperience.reuse import (
    CrossModelReusePlanner,
    KVShape,
    ModelRef,
    PlanStatus,
    ReuseRequest,
    ReuseScenario,
    ReuseStrategy,
)
from goldenexperience.size_variant import (
    CachedKVBridgeManifest,
    CachedKVModelSpec,
    CachedKVQualityEvidence,
    QualityGateResult,
    artifact_id_for,
    build_calibration_manifest,
)


def make_model(
    model_id: str,
    size_b: float,
    layers: int = 32,
    head_dim: int = 128,
    family: str = "qwen",
    architecture: str = "qwen3",
    tokenizer_id: str = "qwen3",
) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family=family,
        architecture=architecture,
        tokenizer_id=tokenizer_id,
        parameter_count_b=size_b,
        kv_shape=KVShape(
            num_layers=layers,
            hidden_size=4096 if size_b <= 8 else 5120,
            num_attention_heads=32 if size_b <= 8 else 40,
            num_key_value_heads=8,
            head_dim=head_dim,
            dtype="bfloat16",
            rope_theta=1_000_000.0,
            model_config_hash=f"{model_id}-hash",
            tokenizer_hash="qwen3-tokenizer-hash",
        ),
    )


def passing_quality() -> QualityGateResult:
    return QualityGateResult.from_metrics(
        hidden_cosine=0.99,
        min_hidden_cosine=0.97,
        kv_cosine=0.99,
        attention_proxy_cosine=0.99,
        perplexity_drift_pct=0.0,
        task_score_drop_pct=0.0,
    )


def cached_kv_artifact(tmp_path, *, include_cost: bool = True):
    tokenizer_hash = "a" * 64
    source_config_hash = "b" * 64
    target_config_hash = "c" * 64
    source = make_model("Qwen/Qwen3-8B", 8, layers=36)
    target = make_model("Qwen/Qwen3-14B", 14, layers=40)
    source = replace(
        source,
        kv_shape=replace(
            source.kv_shape,
            model_config_hash=source_config_hash,
            tokenizer_hash=tokenizer_hash,
        ),
    )
    target = replace(
        target,
        kv_shape=replace(
            target.kv_shape,
            model_config_hash=target_config_hash,
            tokenizer_hash=tokenizer_hash,
        ),
    )
    source_spec = CachedKVModelSpec(
        model_id=source.model_id,
        parameter_count_b=8,
        revision="content-addressed",
        architecture="qwen3",
        config_sha256=source_config_hash,
        tokenizer_sha256=tokenizer_hash,
        weights_sha256="d" * 64,
        num_layers=36,
        num_key_value_heads=8,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000.0,
        max_position_embeddings=40960,
    )
    target_spec = replace(
        source_spec,
        model_id=target.model_id,
        parameter_count_b=14,
        config_sha256=target_config_hash,
        weights_sha256="e" * 64,
        num_layers=40,
    )
    quality = CachedKVQualityEvidence(
        evaluation_dataset_sha256="3" * 64,
        held_out_prompts=64,
        evaluated_tokens=4096,
        token_buckets=(32, 128, 512, 2048),
        key_cosine=0.99,
        value_cosine=0.99,
        next_token_top1_agreement=0.99,
        perplexity_drift_pct=1.0,
        task_prompts=64,
        native_task_score=1.0,
        bridge_task_score=0.99,
        task_score_drop_pct=1.0,
        greedy_continuation_match_rate=0.99,
        cost_report_sha256="4" * 64 if include_cost else None,
        cost_candidate_manifest_sha256="5" * 64 if include_cost else None,
        p95_source_read_transform_put_ms=40.0 if include_cost else None,
        p95_target_prefill_ms=100.0 if include_cost else None,
    )
    manifest = CachedKVBridgeManifest(
        bridge_id="pending",
        direction="8b_to_14b",
        source=source_spec,
        target=target_spec,
        weights_uri="bridge.safetensors",
        weights_sha256="f" * 64,
        rank=64,
        source_window=3,
        train_dataset_sha256="1" * 64,
        validation_dataset_sha256="2" * 64,
        test_dataset_sha256="3" * 64,
        quality=quality,
    )
    manifest = replace(manifest, bridge_id=artifact_id_for(manifest))
    path = tmp_path / "cached-kv.json"
    manifest.save(path)
    return source, target, manifest, path


def test_lora_pair_requires_measured_quality_evidence() -> None:
    base = make_model("qwen3-8b", 8)
    lora = ModelRef(
        model_id="qwen3-8b-lora-math",
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=8,
        base_model_id="qwen3-8b",
        lora_adapter_id="math-adapter",
        kv_shape=base.kv_shape,
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(source=base, target=lora, prefix_hash="shared")
    )

    assert plan.scenario == ReuseScenario.LORA_ADAPTER
    assert plan.strategy == ReuseStrategy.ADAPTER_DELTA_GATED_ALIAS
    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert not plan.executable
    assert "lora_delta_quality_gate" in plan.required_gates
    assert (
        CrossModelCacheKey.from_plan(plan).to_sidecar_fields()["ge_scenario"] == plan.scenario.value
    )


def test_lora_pair_is_ready_with_quality_evidence() -> None:
    base = make_model("qwen3-8b", 8)
    lora = ModelRef(
        model_id="qwen3-8b-lora-math",
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=8,
        base_model_id="qwen3-8b",
        lora_adapter_id="math-adapter",
        kv_shape=base.kv_shape,
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=base,
            target=lora,
            prefix_hash="shared",
            calibration_id="lora-probe-v1",
            lora_quality_score=0.98,
        )
    )

    assert plan.status == PlanStatus.READY
    assert plan.executable
    assert plan.confidence == 0.98


def test_size_variant_requires_calibration_when_shape_differs() -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(source=small, target=large, prefix_hash="shared")
    )

    assert plan.scenario == ReuseScenario.SAME_MODEL_SIZE_VARIANT
    assert plan.strategy == ReuseStrategy.HIDDEN_STATE_BRIDGE
    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert not plan.executable
    assert "hidden_bridge_calibration" in plan.required_gates


def test_size_variant_with_calibration_id_still_needs_artifact() -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="shared",
            calibration_id="qwen3_8b_to_14b_hidden_bridge_v0",
        )
    )

    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert not plan.executable


def test_size_variant_uses_approved_cached_kv_artifact(tmp_path) -> None:
    source, target, manifest, path = cached_kv_artifact(tmp_path)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=source,
            target=target,
            prefix_hash="shared",
            calibration_id=manifest.bridge_id,
            artifact_uri=str(path),
        )
    )

    assert plan.status == PlanStatus.READY
    assert plan.strategy == ReuseStrategy.CACHED_KV_TRANSLATION
    assert plan.cached_kv_bridge_id == manifest.bridge_id
    assert plan.transform_id == manifest.bridge_id
    assert plan.estimated_materialization_ms == 40.0
    assert plan.estimated_prefill_saved_ms == 60.0
    assert plan.as_metadata()["ge_cached_kv_bridge_id"] == manifest.bridge_id


def test_size_variant_rejects_cached_kv_artifact_without_cost(tmp_path) -> None:
    source, target, manifest, path = cached_kv_artifact(tmp_path, include_cost=False)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=source,
            target=target,
            prefix_hash="shared",
            calibration_id=manifest.bridge_id,
            artifact_uri=str(path),
        )
    )

    assert plan.status == PlanStatus.BLOCKED
    assert plan.strategy == ReuseStrategy.CACHED_KV_TRANSLATION
    assert plan.fallback_reason == "quality_gate_failed"
    assert "p95_source_read_transform_put_ms is required" in " ".join(plan.notes)


def test_size_variant_with_hidden_bridge_artifact_is_executable(tmp_path) -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)
    manifest = build_calibration_manifest(
        small,
        large,
        calibration_id="qwen3_8b_to_14b_hidden_bridge_v0",
        prompts_count=3,
        quality=passing_quality(),
        bridge_method="identity_pad_truncate",
        scope="global",
        evaluation_dataset_hash="c" * 64,
        held_out_prompts_count=3,
    )
    path = tmp_path / "qwen3_8b_to_14b_hidden_bridge_v0.json"
    manifest.save(path)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="shared",
            calibration_id=manifest.calibration_id,
            artifact_uri=str(path),
            estimated_target_prefill_ms=100.0,
            estimated_materialization_ms=30.0,
        )
    )

    assert plan.status == PlanStatus.READY
    assert plan.strategy == ReuseStrategy.HIDDEN_STATE_BRIDGE
    assert plan.hidden_bridge_id == manifest.hidden_bridge_id
    assert plan.restore_id == manifest.restore_id
    assert plan.executable


def test_size_variant_rejects_unknown_cost(tmp_path) -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)
    manifest = build_calibration_manifest(
        small,
        large,
        calibration_id="qwen3_bridge_v0",
        prompts_count=3,
        quality=passing_quality(),
        bridge_method="identity_pad_truncate",
        scope="global",
        evaluation_dataset_hash="c" * 64,
        held_out_prompts_count=3,
    )
    path = tmp_path / "bridge.json"
    manifest.save(path)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="shared",
            calibration_id=manifest.calibration_id,
            artifact_uri=str(path),
        )
    )

    assert plan.status == PlanStatus.WARM_START_RECOMPUTE
    assert plan.fallback_reason == "cost_gate_failed"


def test_size_variant_enforces_quality_floor_and_prefix_scope(tmp_path) -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)
    manifest = build_calibration_manifest(
        small,
        large,
        calibration_id="prefix_bridge_v0",
        prompts_count=3,
        quality=passing_quality(),
        bridge_method="identity_pad_truncate",
        scope="prefix_allowlist",
        prefix_hash_allowlist=("allowed-prefix",),
        evaluation_dataset_hash="c" * 64,
        held_out_prompts_count=3,
    )
    path = tmp_path / "prefix-bridge.json"
    manifest.save(path)

    wrong_prefix = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="different-prefix",
            calibration_id=manifest.calibration_id,
            artifact_uri=str(path),
            estimated_target_prefill_ms=100.0,
            estimated_materialization_ms=30.0,
        )
    )
    low_confidence = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="allowed-prefix",
            calibration_id=manifest.calibration_id,
            artifact_uri=str(path),
            estimated_target_prefill_ms=100.0,
            estimated_materialization_ms=30.0,
            quality_floor=0.995,
        )
    )

    assert wrong_prefix.status == PlanStatus.BLOCKED
    assert "outside artifact allowlist" in " ".join(wrong_prefix.notes)
    assert low_confidence.status == PlanStatus.BLOCKED
    assert low_confidence.fallback_reason == "quality_gate_failed"


def test_cross_base_is_conservative_without_opt_in_and_calibration() -> None:
    qwen = make_model("qwen3-8b", 8)
    llama = make_model(
        "llama-3.1-8b",
        8,
        family="llama",
        architecture="llama3",
        tokenizer_id="llama-3.1",
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(source=qwen, target=llama, prefix_hash="shared")
    )

    assert plan.scenario == ReuseScenario.CROSS_BASE_MODEL
    assert plan.strategy == ReuseStrategy.LEARNED_CROSS_BASE_TRANSLATOR
    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert plan.confidence == 0.0
    assert not plan.executable


def test_cross_base_with_only_calibration_id_remains_blocked() -> None:
    qwen = make_model("qwen3-8b", 8)
    llama = make_model(
        "llama-3.1-8b",
        8,
        family="llama",
        architecture="llama3",
        tokenizer_id="llama-3.1",
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=qwen,
            target=llama,
            prefix_hash="shared",
            allow_cross_base=True,
            calibration_id="cross_base_v0",
        )
    )

    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert not plan.executable
