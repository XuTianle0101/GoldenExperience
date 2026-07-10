"""Offline calibration artifact builders for same-model size variants."""

from __future__ import annotations

import json
from pathlib import Path

from goldenexperience.reuse.models import KVShape, ModelRef
from goldenexperience.size_variant.layer_mapping import build_linear_layer_map
from goldenexperience.size_variant.models import (
    CalibrationManifest,
    QualityGateResult,
    infer_direction,
    kv_width,
    pair_id_for,
    stable_artifact_id,
)
from goldenexperience.size_variant.projection import (
    build_hidden_bridge_spec,
    build_kv_restore_spec,
    build_projection_spec,
)


QWEN3_8B = ModelRef(
    model_id="Qwen/Qwen3-8B",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="Qwen/Qwen3",
    parameter_count_b=8,
    kv_shape=KVShape(
        num_layers=36,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000.0,
        model_config_hash="qwen3-8b-default",
        tokenizer_hash="qwen3-tokenizer-default",
    ),
)

QWEN3_14B = ModelRef(
    model_id="Qwen/Qwen3-14B",
    family="qwen",
    architecture="qwen3",
    tokenizer_id="Qwen/Qwen3",
    parameter_count_b=14,
    kv_shape=KVShape(
        num_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        dtype="bfloat16",
        rope_theta=1_000_000.0,
        model_config_hash="qwen3-14b-default",
        tokenizer_hash="qwen3-tokenizer-default",
    ),
)


def qwen3_model_pair(direction: str = "8b_to_14b") -> tuple[ModelRef, ModelRef]:
    if direction in {"8b_to_14b", "small_to_large"}:
        return QWEN3_8B, QWEN3_14B
    if direction in {"14b_to_8b", "large_to_small"}:
        return QWEN3_14B, QWEN3_8B
    raise ValueError(f"Unknown Qwen3 direction: {direction}")


def build_calibration_manifest(
    source: ModelRef,
    target: ModelRef,
    calibration_id: str | None = None,
    prompts_count: int = 0,
    quality: QualityGateResult | None = None,
    artifact_root: str = "artifacts/golden_scale",
    method: str = "hidden_bridge",
    bridge_rank: int | None = 256,
    bridge_method: str = "low_rank_linear",
    bridge_weight_uri: str | None = None,
    bridge_weight_sha256: str | None = None,
) -> CalibrationManifest:
    direction = infer_direction(source, target)
    pair_id = pair_id_for(source, target)
    calibration_id = calibration_id or stable_artifact_id("calibration", pair_id, direction.value)
    layer_map = build_linear_layer_map(
        pair_id=pair_id,
        direction=direction,
        source_num_layers=source.kv_shape.num_layers,
        target_num_layers=target.kv_shape.num_layers,
    )
    projection = build_projection_spec(
        pair_id=pair_id,
        direction=direction,
        source_kv_heads=source.kv_shape.num_key_value_heads,
        target_kv_heads=target.kv_shape.num_key_value_heads,
        source_head_dim=source.kv_shape.head_dim,
        target_head_dim=target.kv_shape.head_dim,
    )
    hidden_bridge = None
    kv_restore = None
    if method == "hidden_bridge":
        if source.kv_shape.hidden_size is None or target.kv_shape.hidden_size is None:
            raise ValueError("hidden_bridge calibration requires source and target hidden_size")
        hidden_bridge = build_hidden_bridge_spec(
            pair_id=pair_id,
            direction=direction,
            source_hidden_size=source.kv_shape.hidden_size,
            target_hidden_size=target.kv_shape.hidden_size,
            source_num_layers=source.kv_shape.num_layers,
            target_num_layers=target.kv_shape.num_layers,
            method=bridge_method,
            rank=bridge_rank,
            weight_uri=bridge_weight_uri,
            weight_sha256=bridge_weight_sha256,
        )
        kv_restore = build_kv_restore_spec(
            pair_id=pair_id,
            direction=direction,
            target_model_id=target.model_id,
            target_hidden_size=target.kv_shape.hidden_size,
            target_kv_heads=target.kv_shape.num_key_value_heads,
            target_head_dim=target.kv_shape.head_dim,
        )
    elif method != "kv_projection":
        raise ValueError(f"Unknown calibration method: {method}")
    quality = quality or QualityGateResult.uncalibrated()
    return CalibrationManifest(
        calibration_id=calibration_id,
        pair_id=pair_id,
        direction=direction,
        source=source,
        target=target,
        layer_map=layer_map,
        projection=projection,
        quality=quality,
        hidden_bridge=hidden_bridge,
        kv_restore=kv_restore,
        artifact_root=artifact_root,
        prompts_count=prompts_count,
        references=(
            "vllm_lmcache_mp_connector",
            "vllm_paged_attention",
            "mooncake_store_l2",
            "pagedattention_block_kv",
            "cka_layer_alignment",
            "hcache_hidden_state_recovery",
        ),
        metadata={
            "state_kind": "hidden" if hidden_bridge is not None else "kv",
            "hidden_contract": "pre_kv_hidden" if hidden_bridge is not None else "",
            "source_kv_width": kv_width(source.kv_shape),
            "target_kv_width": kv_width(target.kv_shape),
        },
    )


def save_prompt_manifest(path: str | Path, prompts: list[str], source: ModelRef, target: ModelRef) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_model_id": source.model_id,
        "target_model_id": target.model_id,
        "prompt_count": len(prompts),
        "prompts": prompts,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_prompt_count(path: str | Path | None) -> int:
    if path is None:
        return 0
    input_path = Path(path)
    if not input_path.exists():
        return 0
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return int(payload.get("prompt_count", len(payload.get("prompts", []))))
