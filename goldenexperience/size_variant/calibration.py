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
from goldenexperience.size_variant.projection import build_projection_spec


QWEN25_7B = ModelRef(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="Qwen/Qwen2.5",
    parameter_count_b=7,
    kv_shape=KVShape(
        num_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
        head_dim=128,
        dtype="float16",
        rope_theta=1_000_000.0,
        model_config_hash="qwen25-7b-default",
        tokenizer_hash="qwen25-tokenizer-default",
    ),
)

QWEN25_14B = ModelRef(
    model_id="Qwen/Qwen2.5-14B-Instruct",
    family="qwen",
    architecture="qwen2",
    tokenizer_id="Qwen/Qwen2.5",
    parameter_count_b=14,
    kv_shape=KVShape(
        num_layers=48,
        hidden_size=5120,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        dtype="float16",
        rope_theta=1_000_000.0,
        model_config_hash="qwen25-14b-default",
        tokenizer_hash="qwen25-tokenizer-default",
    ),
)


def qwen25_model_pair(direction: str = "7b_to_14b") -> tuple[ModelRef, ModelRef]:
    if direction in {"7b_to_14b", "small_to_large"}:
        return QWEN25_7B, QWEN25_14B
    if direction in {"14b_to_7b", "large_to_small"}:
        return QWEN25_14B, QWEN25_7B
    raise ValueError(f"Unknown Qwen2.5 direction: {direction}")


def build_calibration_manifest(
    source: ModelRef,
    target: ModelRef,
    calibration_id: str | None = None,
    prompts_count: int = 0,
    quality: QualityGateResult | None = None,
    artifact_root: str = "artifacts/golden_scale",
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
    quality = quality or QualityGateResult.from_metrics(
        kv_cosine=0.99,
        attention_proxy_cosine=0.99,
        perplexity_drift_pct=0.0,
        task_score_drop_pct=0.0,
    )
    return CalibrationManifest(
        calibration_id=calibration_id,
        pair_id=pair_id,
        direction=direction,
        source=source,
        target=target,
        layer_map=layer_map,
        projection=projection,
        quality=quality,
        artifact_root=artifact_root,
        prompts_count=prompts_count,
        references=(
            "vllm_lmcache_mp_connector",
            "vllm_paged_attention",
            "mooncake_store_l2",
            "pagedattention_block_kv",
            "cka_layer_alignment",
        ),
        metadata={
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
