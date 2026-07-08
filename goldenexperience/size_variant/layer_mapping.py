"""Layer mapping helpers for model-size-variant reuse."""

from __future__ import annotations

from goldenexperience.size_variant.models import (
    LayerMap,
    LayerMapEntry,
    SizeVariantDirection,
    stable_artifact_id,
)


def build_linear_layer_map(
    pair_id: str,
    direction: SizeVariantDirection,
    source_num_layers: int,
    target_num_layers: int,
    method: str = "linear_interpolation",
) -> LayerMap:
    """Map each target layer to one or two source layers by depth interpolation."""

    if source_num_layers <= 0 or target_num_layers <= 0:
        raise ValueError("source_num_layers and target_num_layers must be positive")
    entries = []
    if target_num_layers == 1:
        entries.append(LayerMapEntry(target_layer_id=0, source_layer_ids=(0,), weights=(1.0,)))
    else:
        for target_layer_id in range(target_num_layers):
            ratio = target_layer_id / max(1, target_num_layers - 1)
            source_position = ratio * max(0, source_num_layers - 1)
            lower = int(source_position)
            upper = min(source_num_layers - 1, lower + 1)
            if lower == upper:
                source_layer_ids = (lower,)
                weights = (1.0,)
            else:
                upper_weight = source_position - lower
                lower_weight = 1.0 - upper_weight
                source_layer_ids = (lower, upper)
                weights = (lower_weight, upper_weight)
            entries.append(
                LayerMapEntry(
                    target_layer_id=target_layer_id,
                    source_layer_ids=source_layer_ids,
                    weights=weights,
                )
            )
    layer_map_id = stable_artifact_id(
        "layer-map",
        pair_id,
        direction.value,
        source_num_layers,
        target_num_layers,
        method,
    )
    return LayerMap(
        layer_map_id=layer_map_id,
        pair_id=pair_id,
        direction=direction,
        source_num_layers=source_num_layers,
        target_num_layers=target_num_layers,
        entries=tuple(entries),
        method=method,
        score=1.0,
    )
