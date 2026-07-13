import pytest

from goldenexperience.cache_core import CacheBlock, DeviceTier, KVPayload
from goldenexperience.cross_model_mapper import (
    CalibrationPair,
    IdentityKVMapper,
    LinearProjectionKVMapper,
    ReusePolicy,
)
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
        rope_theta=1_000_000.0,
        tokenizer_id="qwen3-tokenizer",
    )


def block_for(source: ArchitectureSignature, quality_score: float = 0.99) -> CacheBlock:
    return CacheBlock.from_payload(
        payload=KVPayload(key=[[[1.0, 2.0]]], value=[[[3.0, 4.0]]]),
        model_id=source.model_id,
        layer_id=0,
        head_id=0,
        token_range=(0, 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        quality_score=quality_score,
    )


def test_linear_mapper_projects_last_dimension() -> None:
    source = signature("qwen3-8b", 2)
    target = signature("qwen3-14b", 3)
    block = block_for(source)
    target_block = CacheBlock.from_payload(
        payload=KVPayload(key=[[[1.0, 2.0, 0.0]]], value=[[[3.0, 4.0, 0.0]]]),
        model_id=target.model_id,
        layer_id=0,
        head_id=0,
        token_range=(0, 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
    )

    mapper = LinearProjectionKVMapper().fit(
        source,
        target,
        calibration_data=[CalibrationPair(source=block, target=target_block)],
    )
    result = mapper.transform(block)

    assert result.compatibility == CompatibilityLevel.SHAPE_MISMATCH
    assert result.block.metadata.model_id == target.model_id
    assert result.block.metadata.shape == ((1, 1, 3), (1, 1, 3))
    assert result.block.payload.key == [[[1.0, 2.0, 0.0]]]
    assert mapper.calibrated
    assert mapper.confidence == pytest.approx(0.90)


def test_linear_mapper_rejects_missing_calibration() -> None:
    source = signature("qwen3-8b", 2)
    target = signature("qwen3-14b", 3)
    mapper = LinearProjectionKVMapper().fit(source, target)

    assert mapper.confidence == 0.0
    with pytest.raises(ValueError, match="calibration pairs"):
        mapper.transform(block_for(source))


def test_reuse_policy_falls_back_for_low_quality() -> None:
    source = signature("qwen3-8b", 2)
    target = signature("qwen3-14b", 3)
    block = block_for(source, quality_score=0.50)

    decision = ReusePolicy().decide(
        source_metadata=block.metadata,
        source_signature=source,
        target_signature=target,
        mapper_confidence=0.50,
        prefix_similarity=0.99,
        expected_latency_savings_ms=10.0,
    )

    assert decision.action.value == "fallback_recompute"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tokenizer_id", "different-tokenizer"),
        ("rope_theta", 10_000.0),
        ("dtype", "bfloat16"),
    ],
)
def test_runtime_contract_mismatch_is_incompatible(field: str, value: object) -> None:
    source = signature("qwen3-8b", 2)
    values = {
        "model_id": "qwen3-14b",
        "family": source.family,
        "architecture": source.architecture,
        "num_layers": source.num_layers,
        "hidden_size": source.hidden_size,
        "num_attention_heads": source.num_attention_heads,
        "num_key_value_heads": source.num_key_value_heads,
        "head_dim": source.head_dim,
        "rope_theta": source.rope_theta,
        "tokenizer_id": source.tokenizer_id,
        "dtype": source.dtype,
    }
    values[field] = value
    target = ArchitectureSignature(**values)

    assert source.compatibility_with(target) == CompatibilityLevel.INCOMPATIBLE


def test_identity_mapper_rejects_shape_only_compatibility() -> None:
    source = signature("qwen3-a", 2)
    target = signature("qwen3-b", 2)
    mapper = IdentityKVMapper().fit(source, target)

    assert mapper.compatibility == CompatibilityLevel.SHAPE_COMPATIBLE
    assert mapper.confidence == 0.0
    with pytest.raises(ValueError, match="shape_compatible"):
        mapper.transform(block_for(source))


def test_verified_weight_alias_is_exact() -> None:
    common = {
        "family": "qwen",
        "architecture": "qwen3",
        "num_layers": 2,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rope_theta": 1_000_000.0,
        "tokenizer_id": "qwen3-tokenizer",
        "model_config_hash": "a" * 64,
        "weights_hash": "b" * 64,
    }
    source = ArchitectureSignature(model_id="canonical", **common)
    target = ArchitectureSignature(model_id="alias", **common)

    assert source.compatibility_with(target) == CompatibilityLevel.EXACT
    assert IdentityKVMapper().fit(source, target).confidence == 1.0


def test_reuse_policy_requires_calibrated_non_exact_mapper() -> None:
    source = signature("qwen3-a", 2)
    target = signature("qwen3-b", 2)

    decision = ReusePolicy().decide(
        source_metadata=block_for(source).metadata,
        source_signature=source,
        target_signature=target,
        mapper_confidence=0.99,
        prefix_similarity=1.0,
        expected_latency_savings_ms=10.0,
    )

    assert decision.action.value == "fallback_recompute"
    assert "verified calibration" in decision.reason
