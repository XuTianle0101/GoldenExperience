"""Same-model different-parameter-size KV reuse support."""

from goldenexperience.size_variant.calibration import (
    QWEN25_14B,
    QWEN25_7B,
    build_calibration_manifest,
    load_prompt_count,
    qwen25_model_pair,
    save_prompt_manifest,
)
from goldenexperience.size_variant.layer_mapping import build_linear_layer_map
from goldenexperience.size_variant.models import (
    CalibrationManifest,
    FallbackReason,
    LayerMap,
    LayerMapEntry,
    ProjectionSpec,
    QualityGateResult,
    SizeVariantDirection,
    infer_direction,
    kv_width,
    pair_id_for,
)
from goldenexperience.size_variant.projection import (
    KVChunk,
    MaterializationResult,
    MaterializedKVChunk,
    SizeVariantMaterializer,
    build_projection_spec,
    validate_projection_cost,
)

__all__ = [
    "CalibrationManifest",
    "FallbackReason",
    "KVChunk",
    "LayerMap",
    "LayerMapEntry",
    "MaterializationResult",
    "MaterializedKVChunk",
    "ProjectionSpec",
    "QWEN25_14B",
    "QWEN25_7B",
    "QualityGateResult",
    "SizeVariantDirection",
    "SizeVariantMaterializer",
    "build_calibration_manifest",
    "build_linear_layer_map",
    "build_projection_spec",
    "infer_direction",
    "kv_width",
    "load_prompt_count",
    "pair_id_for",
    "qwen25_model_pair",
    "save_prompt_manifest",
    "validate_projection_cost",
]
