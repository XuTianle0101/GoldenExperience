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


def build_cka_source_layer_plan(
    source_kv: Any,
    target_kv: Any,
    position_ids: Any,
    source_window: int,
    *,
    source_heads: int,
    source_head_dim: int,
    source_rope_theta: float,
    target_heads: int,
    target_head_dim: int,
    target_rope_theta: float,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    """Learn a monotonic source-layer plan using train-only linear CKA."""

    import torch

    if source_kv.ndim != 4 or target_kv.ndim != 4:
        raise ValueError("layer alignment KV objects must be rank four")
    if source_kv.shape[0] != 2 or target_kv.shape[0] != 2:
        raise ValueError("layer alignment KV objects must start with a K/V axis")
    if source_kv.shape[2] != target_kv.shape[2] or source_kv.shape[2] != position_ids.numel():
        raise ValueError("layer alignment sample counts differ")
    if source_kv.shape[2] < 2:
        raise ValueError("layer alignment requires at least two samples")
    source_layers = int(source_kv.shape[1])
    target_layers = int(target_kv.shape[1])
    if source_window <= 0 or source_window > source_layers:
        raise ValueError("source_window is outside source depth")

    source = source_kv.to(device=device, dtype=torch.float32)
    target = target_kv.to(device=device, dtype=torch.float32)
    positions = position_ids.to(device=device, dtype=torch.long)
    source_key = _apply_rope_flat(
        source[0],
        positions,
        num_heads=source_heads,
        head_dim=source_head_dim,
        theta=source_rope_theta,
        inverse=True,
    )
    target_key = _apply_rope_flat(
        target[0],
        positions,
        num_heads=target_heads,
        head_dim=target_head_dim,
        theta=target_rope_theta,
        inverse=True,
    )
    key_scores = _linear_cka_scores(source_key, target_key)
    value_scores = _linear_cka_scores(source[1], target[1])
    combined_scores = (key_scores + value_scores) / 2
    matched = _monotonic_alignment_path(combined_scores)

    layer_ids = torch.empty(target_layers, source_window, dtype=torch.int64)
    layer_weights = torch.zeros(target_layers, source_window, dtype=torch.float32)
    for target_layer, source_layer in enumerate(matched.tolist()):
        start = max(
            0,
            min(source_layers - source_window, source_layer - source_window // 2),
        )
        ids = torch.arange(start, start + source_window, dtype=torch.int64)
        layer_ids[target_layer] = ids
        layer_weights[target_layer, int((ids == source_layer).nonzero().item())] = 1.0

    score_device = combined_scores.device
    target_indices = torch.arange(target_layers, device=score_device)
    matched_device = matched.to(score_device)
    depth_ids, depth_weights = build_source_layer_plan(
        source_layers,
        target_layers,
        source_window,
    )
    depth_ids = depth_ids.to(score_device)
    depth_weights = depth_weights.to(score_device)
    depth_score = (combined_scores.gather(1, depth_ids) * depth_weights).sum(dim=1).mean()
    evidence = {
        "method": "monotonic_linear_cka",
        "sample_count": int(source_kv.shape[2]),
        "matched_source_layers": matched.tolist(),
        "combined_score": float(combined_scores[target_indices, matched_device].mean().item()),
        "key_score": float(key_scores[target_indices, matched_device].mean().item()),
        "value_score": float(value_scores[target_indices, matched_device].mean().item()),
        "depth_baseline_score": float(depth_score.item()),
    }
    return layer_ids, layer_weights, evidence


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
    nonlinear_ridge_lambda: float | None = None,
    device: str,
) -> dict[str, Any]:
    """Fit supervised low-rank residual maps for every target layer."""

    import torch

    if features.ndim != 3 or key_residual.ndim != 3 or value_residual.ndim != 3:
        raise ValueError("training tensors must have [layer, sample, width] layout")
    if features.shape[:2] != key_residual.shape[:2] or key_residual.shape != value_residual.shape:
        raise ValueError("training tensor layer/sample dimensions differ")
    if rank <= 0 or ridge_lambda <= 0 or not math.isfinite(ridge_lambda):
        raise ValueError("rank and ridge_lambda must be valid")
    nonlinear_ridge = ridge_lambda if nonlinear_ridge_lambda is None else nonlinear_ridge_lambda
    if nonlinear_ridge <= 0 or not math.isfinite(nonlinear_ridge):
        raise ValueError("nonlinear_ridge_lambda must be finite and positive")
    target_layers = int(features.shape[0])
    feature_means: list[Any] = []
    key_down: list[Any] = []
    key_nonlinear_mean: list[Any] = []
    key_nonlinear_scale: list[Any] = []
    key_nonlinear_up: list[Any] = []
    key_up: list[Any] = []
    key_bias: list[Any] = []
    key_base_scale: list[Any] = []
    value_down: list[Any] = []
    value_nonlinear_mean: list[Any] = []
    value_nonlinear_scale: list[Any] = []
    value_nonlinear_up: list[Any] = []
    value_up: list[Any] = []
    value_bias: list[Any] = []
    value_base_scale: list[Any] = []
    source_window = int(source_layer_ids.shape[1])
    output_width = int(key_residual.shape[-1])
    expected_feature_width = source_window * output_width * 2
    if int(features.shape[-1]) != expected_feature_width:
        raise ValueError("training feature width does not match source layer plan")
    for layer_id in range(target_layers):
        x = features[layer_id].to(device=device, dtype=torch.float32)
        key_delta = key_residual[layer_id].to(device=device, dtype=torch.float32)
        value_delta = value_residual[layer_id].to(device=device, dtype=torch.float32)
        weights = source_layer_weights[layer_id].to(device=device, dtype=torch.float32)
        key_sources = x[:, : source_window * output_width].reshape(
            x.shape[0], source_window, output_width
        )
        value_sources = x[:, source_window * output_width :].reshape(
            x.shape[0], source_window, output_width
        )
        key_base = torch.einsum("s,tsw->tw", weights, key_sources)
        value_base = torch.einsum("s,tsw->tw", weights, value_sources)
        mean_x = x.mean(dim=0)
        x_centered = x - mean_x
        key_scale, key_mean, key_basis, key_projection = _fit_scaled_projection(
            x_centered,
            key_base,
            key_delta,
            rank=rank,
            ridge_lambda=ridge_lambda,
        )
        value_scale, value_mean, value_basis, value_projection = _fit_scaled_projection(
            x_centered,
            value_base,
            value_delta,
            rank=rank,
            ridge_lambda=ridge_lambda,
        )
        key_activation_scale, key_activation_mean, key_activation_up = _fit_nonlinear_correction(
            x_centered,
            key_base,
            key_delta,
            key_scale,
            key_mean,
            key_basis,
            key_projection,
            ridge_lambda=nonlinear_ridge,
        )
        value_activation_scale, value_activation_mean, value_activation_up = (
            _fit_nonlinear_correction(
                x_centered,
                value_base,
                value_delta,
                value_scale,
                value_mean,
                value_basis,
                value_projection,
                ridge_lambda=nonlinear_ridge,
            )
        )
        feature_means.append(mean_x.cpu())
        key_base_scale.append(key_scale.cpu())
        key_down.append(key_basis.cpu())
        key_nonlinear_mean.append(key_activation_mean.cpu())
        key_nonlinear_scale.append(key_activation_scale.cpu())
        key_nonlinear_up.append(key_activation_up.cpu())
        key_up.append(key_projection.cpu())
        key_bias.append(key_mean.cpu())
        value_base_scale.append(value_scale.cpu())
        value_down.append(value_basis.cpu())
        value_nonlinear_mean.append(value_activation_mean.cpu())
        value_nonlinear_scale.append(value_activation_scale.cpu())
        value_nonlinear_up.append(value_activation_up.cpu())
        value_up.append(value_projection.cpu())
        value_bias.append(value_mean.cpu())
        del x, key_delta, value_delta, x_centered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "source_layer_ids": source_layer_ids.to(torch.int64).cpu(),
        "source_layer_weights": source_layer_weights.to(torch.float32).cpu(),
        "feature_mean": torch.stack(feature_means),
        "key_base_scale": torch.stack(key_base_scale),
        "key_down": torch.stack(key_down),
        "key_nonlinear_mean": torch.stack(key_nonlinear_mean),
        "key_nonlinear_scale": torch.stack(key_nonlinear_scale),
        "key_nonlinear_up": torch.stack(key_nonlinear_up),
        "key_up": torch.stack(key_up),
        "key_bias": torch.stack(key_bias),
        "value_base_scale": torch.stack(value_base_scale),
        "value_down": torch.stack(value_down),
        "value_nonlinear_mean": torch.stack(value_nonlinear_mean),
        "value_nonlinear_scale": torch.stack(value_nonlinear_scale),
        "value_nonlinear_up": torch.stack(value_nonlinear_up),
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
    import torch.nn.functional as functional

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
    base_key = base_key * state["key_base_scale"].to(device=device, dtype=source.dtype).unsqueeze(1)
    base_value = torch.einsum("ls,lstw->ltw", weights, selected_value)
    base_value = base_value * state["value_base_scale"].to(
        device=device, dtype=source.dtype
    ).unsqueeze(1)
    key_latent = torch.bmm(
        centered,
        state["key_down"].to(device=device, dtype=source.dtype),
    )
    value_latent = torch.bmm(
        centered,
        state["value_down"].to(device=device, dtype=source.dtype),
    )
    key_nonlinear = functional.silu(
        key_latent / state["key_nonlinear_scale"].to(device=device, dtype=source.dtype).unsqueeze(1)
    ) - state["key_nonlinear_mean"].to(device=device, dtype=source.dtype).unsqueeze(1)
    value_nonlinear = functional.silu(
        value_latent
        / state["value_nonlinear_scale"].to(device=device, dtype=source.dtype).unsqueeze(1)
    ) - state["value_nonlinear_mean"].to(device=device, dtype=source.dtype).unsqueeze(1)
    key = base_key + torch.bmm(
        key_latent,
        state["key_up"].to(device=device, dtype=source.dtype),
    )
    key = (
        key
        + torch.bmm(
            key_nonlinear,
            state["key_nonlinear_up"].to(device=device, dtype=source.dtype),
        )
        + state["key_bias"].to(device=device, dtype=source.dtype).unsqueeze(1)
    )
    value = base_value + torch.bmm(
        value_latent,
        state["value_up"].to(device=device, dtype=source.dtype),
    )
    value = (
        value
        + torch.bmm(
            value_nonlinear,
            state["value_nonlinear_up"].to(device=device, dtype=source.dtype),
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


def logit_distillation_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    temperature: float,
    label_weight: float,
) -> tuple[Any, Any, Any]:
    """Combine target-logit distillation with a bounded teacher-forced label loss."""

    import torch.nn.functional as functional

    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must share [batch, token, vocab] shape")
    if tuple(labels.shape) != tuple(student_logits.shape[:2]):
        raise ValueError("distillation labels must cover every batch and token")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("distillation temperature must be finite and positive")
    if not math.isfinite(label_weight) or label_weight < 0:
        raise ValueError("distillation label_weight must be finite and non-negative")
    teacher = teacher_logits.detach().float() / temperature
    student = student_logits.float() / temperature
    teacher_probabilities = functional.softmax(teacher, dim=-1)
    teacher_log_probabilities = functional.log_softmax(teacher, dim=-1)
    student_log_probabilities = functional.log_softmax(student, dim=-1)
    distillation = (
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
    ).sum(dim=-1).mean() * (temperature * temperature)
    label_loss = functional.cross_entropy(
        student_logits.float().reshape(-1, student_logits.shape[-1]),
        labels.reshape(-1),
    )
    return distillation + label_weight * label_loss, distillation, label_loss


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


def _linear_cka_scores(source: Any, target: Any) -> Any:
    import torch

    source_centered = source - source.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    source_grams = torch.bmm(source_centered, source_centered.transpose(1, 2))
    target_grams = torch.bmm(target_centered, target_centered.transpose(1, 2))
    source_norms = source_grams.flatten(1).norm(dim=1)
    target_norms = target_grams.flatten(1).norm(dim=1)
    if bool((source_norms <= 0).any()) or bool((target_norms <= 0).any()):
        raise ValueError("layer alignment tensors must contain per-layer variation")
    source_grams = source_grams / source_norms[:, None, None]
    target_grams = target_grams / target_norms[:, None, None]
    return torch.einsum("sij,tij->ts", source_grams, target_grams)


def _monotonic_alignment_path(scores: Any) -> Any:
    import torch

    if scores.ndim != 2 or scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError("layer alignment scores must be a non-empty matrix")
    target_layers, source_layers = scores.shape
    cumulative = scores[0].clone()
    parents = torch.zeros(
        (target_layers, source_layers),
        dtype=torch.long,
        device=scores.device,
    )
    for target_layer in range(1, target_layers):
        prefix_scores, prefix_indices = torch.cummax(cumulative, dim=0)
        cumulative = prefix_scores + scores[target_layer]
        parents[target_layer] = prefix_indices
    source_layer = int(cumulative.argmax().item())
    path = [source_layer]
    for target_layer in range(target_layers - 1, 0, -1):
        source_layer = int(parents[target_layer, source_layer].item())
        path.append(source_layer)
    return torch.tensor(list(reversed(path)), dtype=torch.long, device=scores.device)


def _fit_diagonal_baseline(base: Any, target: Any, *, ridge_lambda: float) -> Any:
    base_centered = base - base.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    numerator = (base_centered * target_centered).sum(dim=0)
    denominator = base_centered.square().sum(dim=0) + ridge_lambda
    return numerator / denominator


def _fit_scaled_projection(
    x_centered: Any,
    base: Any,
    delta: Any,
    *,
    rank: int,
    ridge_lambda: float,
    alternating_steps: int = 4,
) -> tuple[Any, Any, Any, Any]:
    import torch

    scale = torch.ones(base.shape[-1], dtype=base.dtype, device=base.device)
    for _ in range(alternating_steps):
        mean, basis, projection = _fit_supervised_projection(
            x_centered,
            delta - base * (scale - 1),
            rank=rank,
            ridge_lambda=ridge_lambda,
        )
        prediction = (x_centered @ basis) @ projection + mean
        scale = 1 + _fit_diagonal_baseline(
            base,
            delta - prediction,
            ridge_lambda=ridge_lambda,
        )
    mean, basis, projection = _fit_supervised_projection(
        x_centered,
        delta - base * (scale - 1),
        rank=rank,
        ridge_lambda=ridge_lambda,
    )
    return scale, mean, basis, projection


def _fit_nonlinear_correction(
    x_centered: Any,
    base: Any,
    delta: Any,
    base_scale: Any,
    mean: Any,
    basis: Any,
    projection: Any,
    *,
    ridge_lambda: float,
) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    latent = x_centered @ basis
    latent_scale = latent.square().mean(dim=0).sqrt().clamp_min(1e-6)
    activation = functional.silu(latent / latent_scale)
    activation_mean = activation.mean(dim=0)
    activation_centered = activation - activation_mean
    target = delta - base * (base_scale - 1)
    residual = target - ((latent @ projection) + mean)
    gram = activation_centered.T @ activation_centered
    gram.diagonal().add_(ridge_lambda)
    nonlinear_up = torch.linalg.solve(gram, activation_centered.T @ residual)
    return latent_scale, activation_mean, nonlinear_up


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
