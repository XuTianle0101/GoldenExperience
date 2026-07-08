from goldenexperience.cache_core import CacheBlock, DeviceTier, KVPayload
from goldenexperience.cross_model_mapper import LinearProjectionKVMapper, ReusePolicy
from goldenexperience.engine_adapter import ArchitectureSignature, CompatibilityLevel


def signature(model_id: str, head_dim: int) -> ArchitectureSignature:
    return ArchitectureSignature(
        model_id=model_id,
        family="qwen",
        architecture="qwen3",
        num_layers=2,
        hidden_size=head_dim * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=head_dim,
    )


def test_linear_mapper_projects_last_dimension() -> None:
    source = signature("qwen3-8b", 2)
    target = signature("qwen3-14b", 3)
    block = CacheBlock.from_payload(
        payload=KVPayload(key=[[[1.0, 2.0]]], value=[[[3.0, 4.0]]]),
        model_id=source.model_id,
        layer_id=0,
        head_id=0,
        token_range=(0, 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        quality_score=0.99,
    )

    result = LinearProjectionKVMapper().fit(source, target).transform(block)

    assert result.compatibility == CompatibilityLevel.SHAPE_MISMATCH
    assert result.block.metadata.model_id == target.model_id
    assert result.block.metadata.shape == ((1, 1, 3), (1, 1, 3))
    assert result.block.payload.key == [[[1.0, 2.0, 0.0]]]


def test_reuse_policy_falls_back_for_low_quality() -> None:
    source = signature("qwen3-8b", 2)
    target = signature("qwen3-14b", 3)
    block = CacheBlock.from_payload(
        payload=KVPayload(key=[[[1.0, 2.0]]], value=[[[3.0, 4.0]]]),
        model_id=source.model_id,
        layer_id=0,
        head_id=0,
        token_range=(0, 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        quality_score=0.50,
    )

    decision = ReusePolicy().decide(
        source_metadata=block.metadata,
        source_signature=source,
        target_signature=target,
        mapper_confidence=0.50,
        prefix_similarity=0.99,
        expected_latency_savings_ms=10.0,
    )

    assert decision.action.value == "fallback_recompute"

