"""Offline fitting helpers for direction-specific cached-KV bridges."""

from __future__ import annotations

import math
from typing import Any

from goldenexperience.size_variant.cached_kv_bridge import _apply_rope_flat


def build_source_layer_plan(
    source_layers: int,
    target_layers: int,
    source_window: int,
) -> tuple[Any, Any]:
    """Build target-layer source windows and interpolation residual baselines."""

    import torch

    if source_layers <= 0 or target_layers <= 0:
        raise ValueError("source and target layer counts must be positive")
    if source_window <= 0 or source_window > source_layers:
        raise ValueError("source_window is outside source depth")
    layer_ids = torch.empty(target_layers, source_window, dtype=torch.int64)
    layer_weights = torch.zeros(target_layers, source_window, dtype=torch.float32)
    for target_layer in range(target_layers):
        position = (
            0.0 if target_layers == 1 else target_layer * (source_layers - 1) / (target_layers - 1)
        )
        if source_window == 1:
            layer_ids[target_layer, 0] = round(position)
            layer_weights[target_layer, 0] = 1.0
            continue
        low = math.floor(position)
        high = min(low + 1, source_layers - 1)
        start = max(0, min(source_layers - source_window, low - (source_window - 2) // 2))
        ids = list(range(start, start + source_window))
        layer_ids[target_layer] = torch.tensor(ids, dtype=torch.int64)
        if low == high:
            layer_weights[target_layer, ids.index(low)] = 1.0
        else:
            high_weight = position - low
            layer_weights[target_layer, ids.index(low)] = 1.0 - high_weight
            layer_weights[target_layer, ids.index(high)] = high_weight
    return layer_ids, layer_weights


def cache_to_object(past_key_values: Any) -> Any:
    """Convert a Transformers DynamicCache to `[2, layer, token, width]`."""

    import torch

    keys: list[Any] = []
    values: list[Any] = []
    layers = getattr(past_key_values, "layers", None)
    if not layers:
        raise ValueError("past_key_values does not contain cache layers")
    for layer in layers:
        key = layer.keys
        value = layer.values
        if key is None or value is None or key.ndim != 4 or value.ndim != 4:
            raise ValueError("cache layer must contain rank-4 key and value tensors")
        if key.shape[0] != 1 or value.shape[0] != 1 or key.shape != value.shape:
            raise ValueError("cached-KV training currently requires batch size one")
        token_count = int(key.shape[2])
        keys.append(key[0].transpose(0, 1).contiguous().reshape(token_count, -1))
        values.append(value[0].transpose(0, 1).contiguous().reshape(token_count, -1))
    return torch.stack((torch.stack(keys), torch.stack(values)))


def object_to_dynamic_cache(kv_object: Any, config: Any) -> Any:
    """Build a Transformers cache from `[2, layer, token, width]`."""

    from transformers.cache_utils import DynamicCache

    if kv_object.ndim != 4 or kv_object.shape[0] != 2:
        raise ValueError("KV object must have [2, layer, token, width] layout")
    heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    if int(kv_object.shape[-1]) != heads * head_dim:
        raise ValueError("KV object width does not match target config")
    cache = DynamicCache(config=config)
    for layer_id in range(int(kv_object.shape[1])):
        key = (
            kv_object[0, layer_id]
            .reshape(kv_object.shape[2], heads, head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        value = (
            kv_object[1, layer_id]
            .reshape(kv_object.shape[2], heads, head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        cache.update(key, value, layer_id)
    return cache


def build_training_matrices(
    source_kv: Any,
    target_kv: Any,
    position_ids: Any,
    source_layer_ids: Any,
    source_layer_weights: Any,
    *,
    source_heads: int,
    source_head_dim: int,
    source_rope_theta: float,
    target_heads: int,
    target_head_dim: int,
    target_rope_theta: float,
) -> tuple[Any, Any, Any]:
    """Return joint source features and target K/V residuals for fitting."""

    import torch

    _validate_object_pair(source_kv, target_kv, position_ids, source_layer_ids)
    source_key = _apply_rope_flat(
        source_kv[0].float(),
        position_ids,
        num_heads=source_heads,
        head_dim=source_head_dim,
        theta=source_rope_theta,
        inverse=True,
    )
    target_key = _apply_rope_flat(
        target_kv[0].float(),
        position_ids,
        num_heads=target_heads,
        head_dim=target_head_dim,
        theta=target_rope_theta,
        inverse=True,
    )
    source_value = source_kv[1].float()
    target_value = target_kv[1].float()
    selected_key = source_key[source_layer_ids.long()]
    selected_value = source_value[source_layer_ids.long()]
    target_layers, source_window, token_count, width = selected_key.shape
    key_features = selected_key.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1)
    value_features = selected_value.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1)
    features = torch.cat((key_features, value_features), dim=-1)
    weights = source_layer_weights.float()
    if tuple(weights.shape) != (target_layers, source_window):
        raise ValueError("source layer weight shape mismatch")
    base_key = torch.einsum("ls,lstw->ltw", weights, selected_key)
    base_value = torch.einsum("ls,lstw->ltw", weights, selected_value)
    return features, target_key - base_key, target_value.float() - base_value


def fit_low_rank_state(
    features: Any,
    key_residual: Any,
    value_residual: Any,
    source_layer_ids: Any,
    source_layer_weights: Any,
    *,
    rank: int,
    ridge_lambda: float,
    device: str,
) -> dict[str, Any]:
    """Fit supervised low-rank residual maps for every target layer."""

    import torch

    if features.ndim != 3 or key_residual.ndim != 3 or value_residual.ndim != 3:
        raise ValueError("training tensors must have [layer, sample, width] layout")
    if features.shape[:2] != key_residual.shape[:2] or key_residual.shape != value_residual.shape:
        raise ValueError("training tensor layer/sample dimensions differ")
    if rank <= 0 or ridge_lambda < 0 or not math.isfinite(ridge_lambda):
        raise ValueError("rank and ridge_lambda must be valid")
    target_layers = int(features.shape[0])
    feature_means: list[Any] = []
    key_down: list[Any] = []
    key_up: list[Any] = []
    key_bias: list[Any] = []
    value_down: list[Any] = []
    value_up: list[Any] = []
    value_bias: list[Any] = []
    for layer_id in range(target_layers):
        x = features[layer_id].to(device=device, dtype=torch.float32)
        key_y = key_residual[layer_id].to(device=device, dtype=torch.float32)
        value_y = value_residual[layer_id].to(device=device, dtype=torch.float32)
        mean_x = x.mean(dim=0)
        x_centered = x - mean_x
        key_mean, key_basis, key_projection = _fit_supervised_projection(
            x_centered,
            key_y,
            rank=rank,
            ridge_lambda=ridge_lambda,
        )
        value_mean, value_basis, value_projection = _fit_supervised_projection(
            x_centered,
            value_y,
            rank=rank,
            ridge_lambda=ridge_lambda,
        )
        feature_means.append(mean_x.cpu())
        key_down.append(key_basis.cpu())
        key_up.append(key_projection.cpu())
        key_bias.append(key_mean.cpu())
        value_down.append(value_basis.cpu())
        value_up.append(value_projection.cpu())
        value_bias.append(value_mean.cpu())
        del x, key_y, value_y, x_centered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "source_layer_ids": source_layer_ids.to(torch.int64).cpu(),
        "source_layer_weights": source_layer_weights.to(torch.float32).cpu(),
        "feature_mean": torch.stack(feature_means),
        "key_down": torch.stack(key_down),
        "key_up": torch.stack(key_up),
        "key_bias": torch.stack(key_bias),
        "value_down": torch.stack(value_down),
        "value_up": torch.stack(value_up),
        "value_bias": torch.stack(value_bias),
    }


def transform_with_state(
    source_kv: Any,
    position_ids: Any,
    state: dict[str, Any],
    *,
    source_heads: int,
    source_head_dim: int,
    source_rope_theta: float,
    target_heads: int,
    target_head_dim: int,
    target_rope_theta: float,
    device: str,
) -> Any:
    """Apply an in-memory training state before an artifact can be approved."""

    import torch

    parsed_device = torch.device(device)
    compute_dtype = torch.bfloat16 if parsed_device.type == "cuda" else torch.float32
    source = source_kv.to(device=parsed_device, dtype=compute_dtype)
    positions = position_ids.to(device=device, dtype=torch.long)
    layer_ids = state["source_layer_ids"].to(device=device, dtype=torch.long)
    weights = state["source_layer_weights"].to(device=device, dtype=compute_dtype)
    selected_key = source[0][layer_ids]
    selected_value = source[1][layer_ids]
    unrotated_key = _apply_rope_flat(
        selected_key,
        positions,
        num_heads=source_heads,
        head_dim=source_head_dim,
        theta=source_rope_theta,
        inverse=True,
    )
    target_layers, _, token_count, _ = selected_key.shape
    features = torch.cat(
        (
            unrotated_key.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1),
            selected_value.permute(0, 2, 1, 3).reshape(target_layers, token_count, -1),
        ),
        dim=-1,
    )
    centered = features - state["feature_mean"].to(device=device, dtype=source.dtype).unsqueeze(1)
    base_key = torch.einsum("ls,lstw->ltw", weights, unrotated_key)
    base_value = torch.einsum("ls,lstw->ltw", weights, selected_value)
    key = (
        base_key
        + torch.bmm(
            torch.bmm(
                centered,
                state["key_down"].to(device=device, dtype=source.dtype),
            ),
            state["key_up"].to(device=device, dtype=source.dtype),
        )
        + state["key_bias"].to(device=device, dtype=source.dtype).unsqueeze(1)
    )
    value = (
        base_value
        + torch.bmm(
            torch.bmm(
                centered,
                state["value_down"].to(device=device, dtype=source.dtype),
            ),
            state["value_up"].to(device=device, dtype=source.dtype),
        )
        + state["value_bias"].to(device=device, dtype=source.dtype).unsqueeze(1)
    )
    key = _apply_rope_flat(
        key,
        positions,
        num_heads=target_heads,
        head_dim=target_head_dim,
        theta=target_rope_theta,
        inverse=False,
    )
    return torch.stack((key, value)).to(dtype=source_kv.dtype).contiguous()


def cosine_mean(left: Any, right: Any) -> float:
    import torch.nn.functional as functional

    left_flat = left.float().reshape(-1, left.shape[-1])
    right_flat = right.float().reshape(-1, right.shape[-1])
    return float(functional.cosine_similarity(left_flat, right_flat, dim=-1).mean().item())


def _fit_supervised_projection(
    x_centered: Any,
    y: Any,
    *,
    rank: int,
    ridge_lambda: float,
) -> tuple[Any, Any, Any]:
    import torch

    mean_y = y.mean(dim=0)
    y_centered = y - mean_y
    effective_rank = min(
        rank,
        int(x_centered.shape[0]) - 1,
        int(x_centered.shape[1]),
        int(y_centered.shape[1]),
    )
    if effective_rank <= 0:
        raise ValueError("not enough samples to fit a low-rank projection")
    sample_count, feature_width = x_centered.shape
    if feature_width <= sample_count:
        gram = x_centered.T @ x_centered
        gram.diagonal().add_(ridge_lambda)
        full_projection = torch.linalg.solve(gram, x_centered.T @ y_centered)
    else:
        gram = x_centered @ x_centered.T
        gram.diagonal().add_(ridge_lambda)
        dual = torch.linalg.solve(gram, y_centered)
        full_projection = x_centered.T @ dual
    source_basis, singular_values, target_basis = torch.svd_lowrank(
        full_projection,
        q=effective_rank,
        niter=4,
    )
    source_factor = source_basis * singular_values.unsqueeze(0)
    target_factor = target_basis.T
    return mean_y, source_factor, target_factor


def _validate_object_pair(
    source_kv: Any,
    target_kv: Any,
    position_ids: Any,
    source_layer_ids: Any,
) -> None:
    if source_kv.ndim != 4 or target_kv.ndim != 4:
        raise ValueError("source and target KV objects must be rank four")
    if source_kv.shape[0] != 2 or target_kv.shape[0] != 2:
        raise ValueError("source and target KV objects must start with K/V axis")
    if source_kv.shape[2] != target_kv.shape[2] or source_kv.shape[2] != position_ids.numel():
        raise ValueError("source, target, and position token counts differ")
    if source_kv.shape[3] != target_kv.shape[3]:
        raise ValueError("cached-KV bridge v1 requires equal source and target widths")
    if source_layer_ids.ndim != 2 or source_layer_ids.shape[0] != target_kv.shape[1]:
        raise ValueError("source layer plan does not cover target depth")
