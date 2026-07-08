from pathlib import Path

from goldenexperience.lmcache_patch import CrossModelCacheKey
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, PlanStatus, ReuseRequest
from goldenexperience.size_variant import (
    KVChunk,
    SizeVariantDirection,
    SizeVariantMaterializer,
    build_calibration_manifest,
    qwen3_model_pair,
)


def model_ref(model_id: str, size_b: float, layers: int, kv_heads: int, head_dim: int) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen-tokenizer",
        parameter_count_b=size_b,
        kv_shape=KVShape(
            num_layers=layers,
            hidden_size=kv_heads * head_dim * 4,
            num_attention_heads=kv_heads * 2,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            dtype="float16",
            rope_theta=1_000_000.0,
            model_config_hash=f"{model_id}-hash",
            tokenizer_hash="qwen-tokenizer-hash",
        ),
    )


def test_calibration_manifest_round_trip_and_materializer(tmp_path: Path) -> None:
    source = model_ref("qwen-small", 7, layers=2, kv_heads=1, head_dim=2)
    target = model_ref("qwen-large", 14, layers=3, kv_heads=1, head_dim=3)
    manifest = build_calibration_manifest(
        source=source,
        target=target,
        calibration_id="small_to_large_v0",
        prompts_count=3,
        artifact_root=str(tmp_path),
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = manifest.load(path)

    assert loaded.direction == SizeVariantDirection.SMALL_TO_LARGE
    assert loaded.validate() == []
    assert loaded.layer_map.target_num_layers == 3
    assert loaded.projection.source_width == 2
    assert loaded.projection.target_width == 3

    source_chunks = {
        0: KVChunk(layer_id=0, key=[[[1.0, 2.0]]], value=[[[3.0, 4.0]]], token_end=1),
        1: KVChunk(layer_id=1, key=[[[5.0, 6.0]]], value=[[[7.0, 8.0]]], token_end=1),
    }
    result = SizeVariantMaterializer(loaded).materialize(source_chunks)

    assert result.success
    assert [chunk.layer_id for chunk in result.chunks] == [0, 1, 2]
    assert result.chunks[0].key == [[[1.0, 2.0, 0.0]]]
    assert result.chunks[-1].source_layer_ids == (1,)


def test_planner_uses_size_variant_artifact_metadata(tmp_path: Path) -> None:
    source = model_ref("qwen-small", 7, layers=2, kv_heads=1, head_dim=2)
    target = model_ref("qwen-large", 14, layers=3, kv_heads=1, head_dim=3)
    manifest = build_calibration_manifest(source, target, calibration_id="small_to_large_v0")
    path = tmp_path / "small_to_large_v0.json"
    manifest.save(path)

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=source,
            target=target,
            prefix_hash="shared",
            calibration_id="small_to_large_v0",
            artifact_uri=str(path),
            estimated_target_prefill_ms=100.0,
            estimated_materialization_ms=30.0,
        )
    )

    assert plan.status == PlanStatus.READY
    assert plan.direction == "small_to_large"
    assert plan.layer_map_id == manifest.layer_map_id
    assert plan.projection_id == manifest.projection_id
    assert plan.estimated_prefill_saved_ms == 70.0
    fields = CrossModelCacheKey.from_plan(plan).to_sidecar_fields()
    assert fields["ge_pair_id"] == manifest.pair_id
    assert fields["ge_projection_id"] == manifest.projection_id
    assert fields["ge_source_config_hash"] == source.kv_shape.model_config_hash


def test_planner_blocks_artifact_hash_mismatch(tmp_path: Path) -> None:
    source = model_ref("qwen-small", 7, layers=2, kv_heads=1, head_dim=2)
    target = model_ref("qwen-large", 14, layers=3, kv_heads=1, head_dim=3)
    manifest = build_calibration_manifest(source, target, calibration_id="small_to_large_v0")
    path = tmp_path / "small_to_large_v0.json"
    manifest.save(path)
    changed_source = ModelRef(
        model_id=source.model_id,
        family=source.family,
        architecture=source.architecture,
        tokenizer_id=source.tokenizer_id,
        parameter_count_b=source.parameter_count_b,
        kv_shape=KVShape(
            num_layers=source.kv_shape.num_layers,
            hidden_size=source.kv_shape.hidden_size,
            num_attention_heads=source.kv_shape.num_attention_heads,
            num_key_value_heads=source.kv_shape.num_key_value_heads,
            head_dim=source.kv_shape.head_dim,
            dtype=source.kv_shape.dtype,
            rope_theta=source.kv_shape.rope_theta,
            model_config_hash="changed-hash",
            tokenizer_hash=source.kv_shape.tokenizer_hash,
        ),
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(
            source=changed_source,
            target=target,
            prefix_hash="shared",
            calibration_id="small_to_large_v0",
            artifact_uri=str(path),
        )
    )

    assert plan.status == PlanStatus.BLOCKED
    assert plan.fallback_reason == "artifact_hash_mismatch"


def test_qwen3_bidirectional_presets_are_valid() -> None:
    for direction in ("8b_to_14b", "14b_to_8b"):
        source, target = qwen3_model_pair(direction)
        manifest = build_calibration_manifest(source, target, calibration_id=f"qwen3_{direction}_projection_v0")
        assert manifest.validate() == []
        assert manifest.projection.source_width == source.kv_shape.num_key_value_heads * source.kv_shape.head_dim
        assert manifest.projection.target_width == target.kv_shape.num_key_value_heads * target.kv_shape.head_dim
