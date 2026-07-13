"""Version-tolerant accessors for Hugging Face model configuration fields."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Any


class ModelConfigError(ValueError):
    """Raised when a required runtime field cannot be resolved safely."""


def config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a raw JSON mapping or a config object."""

    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def resolve_head_dim(config: Any) -> int:
    """Resolve an explicit head dimension or infer it from hidden size."""

    configured = config_value(config, "head_dim")
    if configured is not None:
        return _positive_int(configured, "head_dim")
    hidden_size = _positive_int(config_value(config, "hidden_size"), "hidden_size")
    attention_heads = _positive_int(
        config_value(config, "num_attention_heads"),
        "num_attention_heads",
    )
    if hidden_size % attention_heads:
        raise ModelConfigError("hidden_size must be divisible by num_attention_heads")
    return hidden_size // attention_heads


def resolve_rope_theta(config: Any) -> float:
    """Resolve RoPE theta across legacy and Transformers 5.x layouts."""

    value = optional_rope_theta(config)
    if value is None:
        raise ModelConfigError("model config does not expose rope_theta")
    return value


def optional_rope_theta(config: Any) -> float | None:
    """Return RoPE theta when the architecture defines it."""

    candidates: list[tuple[str, Any]] = []
    direct = config_value(config, "rope_theta")
    if direct is not None:
        candidates.append(("rope_theta", direct))
    for name in ("rope_parameters", "rope_scaling"):
        nested = config_value(config, name)
        value = config_value(nested, "rope_theta") if nested is not None else None
        if value is not None:
            candidates.append((f"{name}.rope_theta", value))
    if not candidates:
        return None
    resolved = [_positive_float(value, name) for name, value in candidates]
    if any(value != resolved[0] for value in resolved[1:]):
        raise ModelConfigError("model config exposes conflicting rope_theta values")
    return resolved[0]


def resolve_dtype(config: Any, *, default: str | None = None) -> str:
    """Return a stable dtype name while preferring the Transformers 5.x field."""

    value = config_value(config, "dtype")
    if value is None:
        value = config_value(config, "torch_dtype")
    if value is None:
        value = default
    if value is None:
        raise ModelConfigError("model config does not expose dtype")
    normalized = str(value).lower().removeprefix("torch.")
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
    }
    normalized = aliases.get(normalized, normalized)
    if not normalized:
        raise ModelConfigError("model config dtype is empty")
    return normalized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ModelConfigError(f"{name} must be a positive integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise ModelConfigError(f"{name} must be a positive integer") from exc
    if resolved <= 0:
        raise ModelConfigError(f"{name} must be a positive integer")
    return resolved


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ModelConfigError(f"{name} must be finite and positive")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelConfigError(f"{name} must be finite and positive") from exc
    if not math.isfinite(resolved) or resolved <= 0:
        raise ModelConfigError(f"{name} must be finite and positive")
    return resolved
