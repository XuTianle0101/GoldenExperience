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
        kv_shape=KVShape(num_layers=layers, num_key_value_heads=8, head_dim=head_dim),
    )


def test_lora_pair_is_ready_when_layout_matches() -> None:
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
    assert plan.status == PlanStatus.READY
    assert plan.executable
    assert "lora_delta_quality_gate" in plan.required_gates
    assert CrossModelCacheKey.from_plan(plan).to_sidecar_fields()["ge_scenario"] == plan.scenario.value


def test_size_variant_requires_calibration_when_shape_differs() -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(source=small, target=large, prefix_hash="shared")
    )

    assert plan.scenario == ReuseScenario.SAME_MODEL_SIZE_VARIANT
    assert plan.strategy == ReuseStrategy.LAYERWISE_PROJECTION
    assert plan.status == PlanStatus.NEEDS_CALIBRATION
    assert not plan.executable
    assert "projection_calibration" in plan.required_gates


def test_size_variant_with_calibration_is_executable() -> None:
    small = make_model("qwen3-8b", 8, layers=36)
    large = make_model("qwen3-14b", 14, layers=40)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=small,
            target=large,
            prefix_hash="shared",
            calibration_id="qwen3_projection_v0",
        )
    )

    assert plan.status == PlanStatus.READY
    assert plan.executable


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


def test_cross_base_with_calibration_and_opt_in_is_ready() -> None:
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

    assert plan.status == PlanStatus.READY
    assert plan.executable
