"""Tensor helpers with optional PyTorch and NumPy support."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import is_dataclass
from typing import Any

try:  # pragma: no cover - depends on optional environment packages
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on optional environment packages
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def infer_shape(value: Any) -> tuple[int, ...]:
    if torch is not None and hasattr(value, "shape") and hasattr(value, "detach"):
        return tuple(int(dim) for dim in value.shape)
    if np is not None and hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value), *infer_shape(value[0]))
    return ()


def tensor_nbytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if torch is not None and hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if is_dataclass(value):
        return sum(tensor_nbytes(getattr(value, field.name)) for field in value.__dataclass_fields__.values())
    if isinstance(value, dict):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        return sum(tensor_nbytes(item) for item in value)
    if isinstance(value, (float, int, bool)):
        return 8
    if value is None:
        return 0
    return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def stable_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def _update_digest(hasher: Any, value: Any) -> None:
    if torch is not None and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        if np is not None:
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(pickle.dumps(tensor.tolist(), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if np is not None and hasattr(value, "tobytes"):
        hasher.update(str(tuple(value.shape)).encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(value.tobytes())
        return
    if is_dataclass(value):
        for field in value.__dataclass_fields__.values():
            hasher.update(field.name.encode())
            _update_digest(hasher, getattr(value, field.name))
        return
    hasher.update(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def move_to_tier(value: Any, tier_name: str, pin_cpu: bool = False) -> Any:
    """Best-effort movement for optional torch tensors; no-op for other payloads."""

    if torch is None or not hasattr(value, "to"):
        return value
    if tier_name == "hbm" and torch.cuda.is_available():
        return value.to("cuda", non_blocking=True)
    if tier_name in {"cpu", "nvme"}:
        moved = value.to("cpu", non_blocking=True)
        if pin_cpu and hasattr(moved, "pin_memory"):
            try:
                return moved.pin_memory()
            except RuntimeError:
                return moved
        return moved
    return value


def move_payload_to_tier(payload: Any, tier_name: str, pin_cpu: bool = False) -> Any:
    if is_dataclass(payload):
        for field in payload.__dataclass_fields__.values():
            setattr(payload, field.name, move_payload_to_tier(getattr(payload, field.name), tier_name, pin_cpu))
        return payload
    if isinstance(payload, dict):
        return {key: move_payload_to_tier(item, tier_name, pin_cpu) for key, item in payload.items()}
    if isinstance(payload, tuple):
        return tuple(move_payload_to_tier(item, tier_name, pin_cpu) for item in payload)
    if isinstance(payload, list):
        return [move_payload_to_tier(item, tier_name, pin_cpu) for item in payload]
    return move_to_tier(payload, tier_name, pin_cpu)


def identity_projection(source_dim: int, target_dim: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for source_idx in range(source_dim):
        row = []
        for target_idx in range(target_dim):
            row.append(1.0 if source_idx == target_idx else 0.0)
        matrix.append(row)
    return matrix


def project_last_dim(value: Any, weight: Any, bias: Any | None = None) -> Any:
    """Project the final dimension of a tensor-like object.

    Weight follows the shape [source_dim, target_dim]. For nested lists this function
    recurses until it reaches a vector leaf.
    """

    if torch is not None and hasattr(value, "matmul"):
        tensor_weight = weight
        if not hasattr(tensor_weight, "to"):
            tensor_weight = torch.tensor(weight, dtype=value.dtype, device=value.device)
        result = value.matmul(tensor_weight)
        if bias is not None:
            tensor_bias = bias if hasattr(bias, "to") else torch.tensor(bias, dtype=value.dtype, device=value.device)
            result = result + tensor_bias
        return result
    if np is not None and hasattr(value, "dot"):
        result = value.dot(np.asarray(weight))
        if bias is not None:
            result = result + np.asarray(bias)
        return result
    if isinstance(value, tuple):
        return tuple(project_last_dim(item, weight, bias) for item in value)
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, (int, float)) for item in value):
            return _project_vector([float(item) for item in value], weight, bias)
        return [project_last_dim(item, weight, bias) for item in value]
    return value


def _project_vector(vector: list[float], weight: list[list[float]], bias: list[float] | None) -> list[float]:
    if not weight:
        return []
    target_dim = len(weight[0])
    result = []
    for target_idx in range(target_dim):
        value = 0.0
        for source_idx, source_value in enumerate(vector):
            if source_idx < len(weight):
                value += source_value * float(weight[source_idx][target_idx])
        if bias is not None:
            value += float(bias[target_idx])
        result.append(value)
    return result

