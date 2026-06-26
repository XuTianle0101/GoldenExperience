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
    """Map every target layer to the closest source layer by depth ratio."""

    if source_num_layers <= 0 or target_num_layers <= 0:
        raise ValueError("source_num_layers and target_num_layers must be positive")
    entries = []
    if target_num_layers == 1:
        entries.append(LayerMapEntry(target_layer_id=0, source_layer_ids=(0,), weights=(1.0,)))
    else:
        for target_layer_id in range(target_num_layers):
            ratio = target_layer_id / max(1, target_num_layers - 1)
            source_layer_id = round(ratio * max(0, source_num_layers - 1))
            entries.append(
                LayerMapEntry(
                    target_layer_id=target_layer_id,
                    source_layer_ids=(int(source_layer_id),),
                    weights=(1.0,),
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
