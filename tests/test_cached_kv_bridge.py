import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import torch as safetensors_torch

from goldenexperience.size_variant.cached_kv_bridge import (
    CachedKVBridgeError,
    Qwen3CachedKVBridge,
    ResidentQwen3CachedKVBridgeCache,
    _apply_rope_flat,
    safetensors_metadata,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVBridgeManifest,
    CachedKVQualityEvidence,
    artifact_id_for,
    model_spec_from_path,
    sha256_file,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_fake_model(
    root: Path,
    *,
    layers: int,
    hidden_size: int,
    parameter_count_b: float,
    model_id: str,
):
    root.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "num_hidden_layers": layers,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "hidden_size": hidden_size,
        "torch_dtype": "bfloat16",
        "rope_theta": 1_000_000,
        "max_position_embeddings": 40960,
        "rope_scaling": None,
        "sliding_window": None,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text('{"shared":true}', encoding="utf-8")
    safetensors_torch.save_file(
        {"model.weight": torch.arange(4, dtype=torch.float32)},
        root / "model.safetensors",
    )
    return model_spec_from_path(
        root,
        model_id=model_id,
        parameter_count_b=parameter_count_b,
        revision="local-test",
    )


def _quality(test_hash: str) -> CachedKVQualityEvidence:
    return CachedKVQualityEvidence(
        evaluation_dataset_sha256=test_hash,
        held_out_prompts=64,
        evaluated_tokens=8192,
        token_buckets=(32, 128, 512, 2048),
        key_cosine=0.99,
        value_cosine=0.99,
        next_token_top1_agreement=0.99,
        perplexity_drift_pct=1.0,
        task_prompts=64,
        native_task_score=0.99,
        bridge_task_score=0.99,
        task_score_drop_pct=0.5,
        greedy_continuation_match_rate=0.99,
        p95_source_read_transform_put_ms=10.0,
        p95_target_prefill_ms=20.0,
    )


def _state(*, source_layers: int, target_layers: int, source_window: int, rank: int):
    del source_layers
    width = 4
    feature_width = source_window * width * 2
    layer_ids = torch.arange(target_layers, dtype=torch.int64).remainder(2).unsqueeze(1)
    if source_window > 1:
        layer_ids = torch.cat(
            [layer_ids, (layer_ids + 1).remainder(2)],
            dim=1,
        )
    weights = torch.full((target_layers, source_window), 1 / source_window)
    return {
        "source_layer_ids": layer_ids,
        "source_layer_weights": weights,
        "feature_mean": torch.zeros(target_layers, feature_width),
        "key_base_scale": torch.ones(target_layers, width),
        "key_down": torch.zeros(target_layers, feature_width, rank),
        "key_up": torch.zeros(target_layers, rank, width),
        "key_bias": torch.zeros(target_layers, width),
        "value_base_scale": torch.ones(target_layers, width),
        "value_down": torch.zeros(target_layers, feature_width, rank),
        "value_up": torch.zeros(target_layers, rank, width),
        "value_bias": torch.zeros(target_layers, width),
    }


def _artifact(tmp_path: Path, direction: str = "8b_to_14b"):
    if direction == "8b_to_14b":
        source_args = (2, 8.0, "Qwen/Qwen3-8B")
        target_args = (3, 14.0, "Qwen/Qwen3-14B")
    else:
        source_args = (3, 14.0, "Qwen/Qwen3-14B")
        target_args = (2, 8.0, "Qwen/Qwen3-8B")
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source = _write_fake_model(
        source_dir,
        layers=source_args[0],
        hidden_size=8,
        parameter_count_b=source_args[1],
        model_id=source_args[2],
    )
    target = _write_fake_model(
        target_dir,
        layers=target_args[0],
        hidden_size=12,
        parameter_count_b=target_args[1],
        model_id=target_args[2],
    )
    test_hash = _digest("test")
    provisional = CachedKVBridgeManifest(
        bridge_id="pending",
        direction=direction,
        source=source,
        target=target,
        weights_uri="bridge.safetensors",
        weights_sha256="0" * 64,
        rank=1,
        source_window=1,
        train_dataset_sha256=_digest("train"),
        validation_dataset_sha256=_digest("validation"),
        test_dataset_sha256=test_hash,
        quality=_quality(test_hash),
    )
    state = _state(
        source_layers=source.num_layers,
        target_layers=target.num_layers,
        source_window=1,
        rank=1,
    )
    weights_path = tmp_path / "bridge.safetensors"
    safetensors_torch.save_file(
        state,
        weights_path,
        metadata=safetensors_metadata(provisional),
    )
    manifest = replace(provisional, weights_sha256=sha256_file(weights_path))
    manifest = replace(manifest, bridge_id=artifact_id_for(manifest))
    manifest_path = tmp_path / "bridge.json"
    manifest.save(manifest_path)
    return manifest_path, source_dir, target_dir, state


@pytest.mark.parametrize("direction", ["8b_to_14b", "14b_to_8b"])
def test_cached_kv_bridge_loads_and_transforms_both_directions(
    tmp_path: Path,
    direction: str,
) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path, direction)
    bridge = Qwen3CachedKVBridge.from_artifact(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )
    source = torch.randn(
        2,
        bridge.manifest.source.num_layers,
        5,
        bridge.manifest.source.kv_width,
        dtype=torch.bfloat16,
    )

    target = bridge.transform(source, position_start=13)

    assert tuple(target.shape) == (
        2,
        bridge.manifest.target.num_layers,
        5,
        bridge.manifest.target.kv_width,
    )
    selected = bridge._tensors["source_layer_ids"].cpu().squeeze(1)
    torch.testing.assert_close(target[0], source[0][selected], atol=0.03, rtol=0.03)
    torch.testing.assert_close(target[1], source[1][selected], atol=0, rtol=0)


def test_cached_kv_bridge_chunk_positions_match_full_transform(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    bridge = Qwen3CachedKVBridge.from_artifact(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )
    source = torch.randn(2, 2, 6, 4, dtype=torch.bfloat16)

    full = bridge.transform(source, position_start=17)
    chunked = torch.cat(
        (
            bridge.transform(source[:, :, :2], position_start=17),
            bridge.transform(source[:, :, 2:], position_start=19),
        ),
        dim=2,
    )

    torch.testing.assert_close(chunked, full, atol=0, rtol=0)


def test_resident_bridge_cache_reuses_and_invalidates_verified_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    cache = ResidentQwen3CachedKVBridgeCache()

    first, first_hit = cache.load(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )
    second, second_hit = cache.load(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )

    assert first_hit is False
    assert second_hit is True
    assert second is first

    config_path = source_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["hidden_size"] += 1
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CachedKVBridgeError, match="source model: config_sha256"):
        cache.load(
            manifest_path,
            source_model_path=source_dir,
            target_model_path=target_dir,
        )


def test_validation_candidate_loader_cannot_bypass_production_approval(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    manifest = CachedKVBridgeManifest.load(manifest_path)
    candidate = replace(
        manifest,
        quality=replace(
            manifest.quality,
            evaluation_dataset_sha256=manifest.validation_dataset_sha256,
            p95_source_read_transform_put_ms=None,
            p95_target_prefill_ms=None,
        ),
    )
    candidate = replace(candidate, bridge_id=artifact_id_for(candidate))
    candidate.save(manifest_path)

    with pytest.raises(CachedKVBridgeError, match="quality evidence must refer"):
        Qwen3CachedKVBridge.from_artifact(
            manifest_path,
            source_model_path=source_dir,
            target_model_path=target_dir,
        )

    bridge = Qwen3CachedKVBridge.from_validation_candidate_for_benchmark(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )

    assert bridge.manifest.approved is False
    assert bridge.manifest.artifact_errors() == []


def test_qwen_rope_inverse_round_trip_uses_absolute_positions() -> None:
    value = torch.randn(2, 7, 4, dtype=torch.float32)
    positions = torch.tensor([0, 1, 15, 16, 511, 4095, 40959])

    rotated = _apply_rope_flat(
        value,
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=False,
    )
    restored = _apply_rope_flat(
        rotated,
        positions,
        num_heads=1,
        head_dim=4,
        theta=1_000_000,
        inverse=True,
    )

    torch.testing.assert_close(restored, value, atol=1e-5, rtol=1e-5)


def test_artifact_loader_rejects_weight_tampering(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    with (tmp_path / "bridge.safetensors").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(CachedKVBridgeError, match="checksum mismatch"):
        Qwen3CachedKVBridge.from_artifact(
            manifest_path,
            source_model_path=source_dir,
            target_model_path=target_dir,
        )


def test_artifact_loader_rejects_model_weight_mismatch(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    safetensors_torch.save_file(
        {"model.weight": torch.arange(5, dtype=torch.float32)},
        source_dir / "model.safetensors",
    )

    with pytest.raises(CachedKVBridgeError, match="source model: weights_sha256"):
        Qwen3CachedKVBridge.from_artifact(
            manifest_path,
            source_model_path=source_dir,
            target_model_path=target_dir,
        )


def test_manifest_recomputes_quality_gate_and_rejects_nan(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _artifact(tmp_path)
    manifest = CachedKVBridgeManifest.load(manifest_path)
    invalid = replace(
        manifest,
        quality=replace(manifest.quality, next_token_top1_agreement=float("nan")),
    )
    invalid = replace(invalid, bridge_id=artifact_id_for(invalid))

    assert invalid.approved is False
    assert any("finite" in error for error in invalid.validate())


def test_bridge_rejects_wrong_source_layout_dtype_and_nonfinite(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    bridge = Qwen3CachedKVBridge.from_artifact(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )

    with pytest.raises(CachedKVBridgeError, match="shape"):
        bridge.transform(torch.zeros(2, 3, 4, 4, dtype=torch.bfloat16))
    with pytest.raises(CachedKVBridgeError, match="dtype"):
        bridge.transform(torch.zeros(2, 2, 4, 4, dtype=torch.float32))
    invalid = torch.zeros(2, 2, 4, 4, dtype=torch.bfloat16)
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(CachedKVBridgeError, match="non-finite"):
        bridge.transform(invalid)


def test_bridge_output_depends_on_current_prompt_source_object(tmp_path: Path) -> None:
    manifest_path, source_dir, target_dir, _ = _artifact(tmp_path)
    bridge = Qwen3CachedKVBridge.from_artifact(
        manifest_path,
        source_model_path=source_dir,
        target_model_path=target_dir,
    )
    prompt_a = torch.zeros(2, 2, 4, 4, dtype=torch.bfloat16)
    prompt_b = torch.ones(2, 2, 4, 4, dtype=torch.bfloat16)

    output_a = bridge.transform(prompt_a)
    output_b = bridge.transform(prompt_b)

    assert not torch.equal(output_a, output_b)
