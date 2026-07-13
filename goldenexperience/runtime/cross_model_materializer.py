"""Qwen3 cached-KV materialization and legacy experiment utilities."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from goldenexperience.runtime.mooncake_objects import (
    ExactMooncakeStore,
    MooncakeObjectError,
    publish_external_index,
)
from goldenexperience.size_variant.cached_kv_bridge import (
    CachedKVBridgeError,
    ResidentQwen3CachedKVBridgeCache,
)

_RESIDENT_BRIDGE_CACHE = ResidentQwen3CachedKVBridgeCache()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _gate(summary: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, dict[str, Any]]:
    metrics = summary.get("learned_low_rank_hidden_bridge", {})
    values = {
        "hidden_cosine_mean": float(metrics.get("hidden_cosine_mean", 0.0)),
        "key_cosine_mean": float(metrics.get("key_cosine_mean", 0.0)),
        "value_cosine_mean": float(metrics.get("value_cosine_mean", 0.0)),
        "decode_logit_cosine_mean": float(metrics.get("decode_logit_cosine_mean", 0.0)),
        "decode_top1_match_rate": float(metrics.get("decode_top1_match_rate", 0.0)),
    }
    checks = {
        "hidden_cosine": values["hidden_cosine_mean"] >= thresholds.get("hidden_cosine", 0.90),
        "key_cosine": values["key_cosine_mean"] >= thresholds.get("key_cosine", 0.85),
        "value_cosine": values["value_cosine_mean"] >= thresholds.get("value_cosine", 0.85),
        "decode_logit_cosine": values["decode_logit_cosine_mean"]
        >= thresholds.get("decode_logit_cosine", 0.90),
    }
    return all(checks.values()), {"values": values, "checks": checks, "thresholds": thresholds}


class _PreKVHiddenCapture:
    def __init__(self, model: Any) -> None:
        self.hidden_by_layer: dict[int, Any] = {}
        self._handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(
                    self._make_hook(layer_idx),
                    with_kwargs=True,
                )
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.hidden_by_layer[layer_idx] = hidden.detach()

        return hook

    def clear(self) -> None:
        self.hidden_by_layer.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _rotate_half(x):
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _qwen3_key_rope(rotary_emb: Any, key: Any, position_ids: Any) -> Any:
    cos, sin = rotary_emb(key.transpose(1, 2), position_ids)
    return (key * cos.unsqueeze(1)) + (_rotate_half(key) * sin.unsqueeze(1))


def _cosine_mean(lhs: Any, rhs: Any) -> float:
    import torch.nn.functional as F

    lhs_flat = lhs.float().reshape(-1, lhs.shape[-1])
    rhs_flat = rhs.float().reshape(-1, rhs.shape[-1])
    return float(F.cosine_similarity(lhs_flat, rhs_flat, dim=-1).mean().item())


def _make_hidden_projector(state: dict[str, Any], device: str):
    import torch

    mean_x = state["mean_x"].to(device=device, dtype=torch.float32)
    mean_y = state["mean_y"].to(device=device, dtype=torch.float32)
    source_basis = state["source_basis"].to(device=device, dtype=torch.float32)
    target_projection = state.get("target_projection")
    if target_projection is None:
        target_projection = state["target_basis"].T
    target_projection = target_projection.to(device=device, dtype=torch.float32)
    bridge = state["bridge"].to(device=device, dtype=torch.float32)

    def projector(hidden):
        original_dtype = hidden.dtype
        original_shape = hidden.shape
        flat = hidden.to(dtype=torch.float32).reshape(-1, original_shape[-1])
        out = (((flat - mean_x) @ source_basis) @ bridge) @ target_projection + mean_y
        return out.to(dtype=original_dtype).reshape(*original_shape[:-1], -1)

    return projector


def _build_layer_map(
    source_layers: int, target_layers: int
) -> list[tuple[int, tuple[int, ...], tuple[float, ...]]]:
    if target_layers == 1:
        return [(0, (0,), (1.0,))]
    entries: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
    for target_layer in range(target_layers):
        pos = target_layer * (source_layers - 1) / (target_layers - 1)
        low = int(pos)
        high = min(low + 1, source_layers - 1)
        high_w = pos - low
        low_w = 1.0 - high_w
        if low == high or high_w < 1e-9:
            entries.append((target_layer, (low,), (1.0,)))
        else:
            entries.append((target_layer, (low, high), (low_w, high_w)))
    return entries


def _project_target_kv(
    large_model: Any, hidden: Any, layer_id: int, position_ids: Any
) -> tuple[Any, Any]:
    attn = large_model.model.layers[layer_id].self_attn
    batch, seq, _ = hidden.shape
    view_shape = (batch, seq, attn.config.num_key_value_heads, attn.head_dim)
    key = attn.k_proj(hidden).view(view_shape)
    key = attn.k_norm(key).transpose(1, 2)
    value = attn.v_proj(hidden).view(view_shape).transpose(1, 2)
    key = _qwen3_key_rope(large_model.model.rotary_emb, key, position_ids)
    return key, value


def _flatten_layer_kv(key: Any, value: Any) -> tuple[Any, Any]:
    # [batch, kv_heads, seq, head_dim] -> [seq, kv_heads * head_dim]
    key_flat = key[0].transpose(0, 1).contiguous().reshape(key.shape[2], -1)
    value_flat = value[0].transpose(0, 1).contiguous().reshape(value.shape[2], -1)
    return key_flat, value_flat


def materialize_cached_qwen3(
    request: dict[str, Any],
    *,
    store_factory: Callable[[], Any] | None = None,
    bridge_loader: Callable[..., Any] | None = None,
    key_builder: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Read source KV objects, translate them, and publish target objects."""

    import torch

    from goldenexperience.runtime.cross_model_reuse import (
        default_kv_rank,
        mooncake_setup_config,
        object_key_string,
    )

    started = time.perf_counter()
    try:
        if bool(request.get("allow_unsafe", False)):
            raise ValueError("unsafe cached KV materialization is not supported")
        if int(request.get("world_size", 1)) != 1:
            raise ValueError("cached KV materialization currently requires world_size=1")
        chunk_size = int(request["chunk_size"])
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        all_chunk_hashes = _chunk_hashes(request.get("chunk_hashes"))
        max_chunks = int(request.get("max_chunks", 0))
        if max_chunks < 0:
            raise ValueError("max_chunks must be non-negative")
        chunk_hashes = all_chunk_hashes[:max_chunks] if max_chunks else all_chunk_hashes
        native_prefill_ms = float(request["native_target_prefill_ms"])
        if not math.isfinite(native_prefill_ms) or native_prefill_ms <= 0:
            raise ValueError("native_target_prefill_ms must be finite and positive")
        source_model_name = _required_string(request, "source_model_name")
        target_model_name = _required_string(request, "target_model_name")
        if source_model_name == target_model_name:
            raise ValueError("source and target Mooncake namespaces must differ")
        lookup_record = request.get("source_lookup_record")
        if not isinstance(lookup_record, dict):
            raise ValueError("source_lookup_record is required")
        prompt_binding = request.get("prompt_binding")
        if not isinstance(prompt_binding, dict):
            raise ValueError("prompt_binding is required")
        hash_algorithm = _required_string(request, "hash_algorithm")
        if hash_algorithm != "blake3":
            raise ValueError("cached KV materialization requires deterministic blake3 hashes")

        load_started = time.perf_counter()
        bridge_cache_hit = False
        bridge_device = request.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        if bridge_loader is None:
            bridge, bridge_cache_hit = _RESIDENT_BRIDGE_CACHE.load(
                request["bridge_manifest_path"],
                source_model_path=request["source_model_path"],
                target_model_path=request["target_model_path"],
                device=bridge_device,
            )
        else:
            bridge = bridge_loader(
                request["bridge_manifest_path"],
                source_model_path=request["source_model_path"],
                target_model_path=request["target_model_path"],
                device=bridge_device,
            )
        load_ms = (time.perf_counter() - load_started) * 1000
        if request.get("direction") not in {None, bridge.manifest.direction}:
            raise ValueError("requested direction does not match bridge artifact")
        _validate_lookup_record(
            lookup_record,
            source_model_name=source_model_name,
            chunk_hashes=all_chunk_hashes,
            chunk_size=chunk_size,
            source_layers=bridge.manifest.source.num_layers,
            source_width=bridge.manifest.source.kv_width,
            source_dtype=bridge.manifest.source.dtype,
        )
        _validate_prompt_binding(
            prompt_binding,
            lookup_record,
            chunk_hashes=all_chunk_hashes,
            hash_algorithm=hash_algorithm,
        )
        requested_ratio = float(
            request.get(
                "max_materialization_ratio",
                bridge.manifest.thresholds.max_materialization_to_prefill_ratio,
            )
        )
        artifact_ratio = bridge.manifest.thresholds.max_materialization_to_prefill_ratio
        if (
            not math.isfinite(requested_ratio)
            or requested_ratio <= 0
            or requested_ratio > artifact_ratio
        ):
            raise ValueError("max_materialization_ratio may only tighten the artifact gate")

        rank_value = request.get("kv_rank")
        if rank_value is None:
            rank = 0 if key_builder is not None else default_kv_rank()
        else:
            rank = int(rank_value)
            if rank < 0:
                raise ValueError("kv_rank must be non-negative")
        cache_salt_value = request.get("cache_salt", "")
        if not isinstance(cache_salt_value, str):
            raise ValueError("cache_salt must be a string")
        cache_salt = cache_salt_value
        build_key = key_builder or object_key_string
        source_keys = [
            build_key(
                model_name=source_model_name,
                chunk_hash=chunk_hash,
                kv_rank=rank,
                cache_salt=cache_salt,
            )
            for chunk_hash in chunk_hashes
        ]
        target_keys = [
            build_key(
                model_name=target_model_name,
                chunk_hash=chunk_hash,
                kv_rank=rank,
                cache_salt=cache_salt,
            )
            for chunk_hash in chunk_hashes
        ]
        source_shape = (
            2,
            bridge.manifest.source.num_layers,
            chunk_size,
            bridge.manifest.source.kv_width,
        )
        target_shape = (
            2,
            bridge.manifest.target.num_layers,
            chunk_size,
            bridge.manifest.target.kv_width,
        )
        source_dtype = _torch_dtype(bridge.manifest.source.dtype)
        target_dtype = _torch_dtype(bridge.manifest.target.dtype)
        source_bytes = math.prod(source_shape) * torch.empty((), dtype=source_dtype).element_size()
        target_bytes = math.prod(target_shape) * torch.empty((), dtype=target_dtype).element_size()
        setup_config = mooncake_setup_config(dict(request["mooncake_setup_config"]))

        operation_started = time.perf_counter()
        with ExactMooncakeStore(setup_config, store_factory=store_factory) as store:
            read_started = time.perf_counter()
            reads = store.read_many_exact(source_keys, [source_bytes] * len(source_keys))
            read_ms = (time.perf_counter() - read_started) * 1000

            transform_started = time.perf_counter()
            target_payloads: list[bytearray] = []
            for chunk_index, read in enumerate(reads):
                source_object = (
                    torch.frombuffer(read.data, dtype=source_dtype).reshape(source_shape).clone()
                )
                target_object = bridge.transform(
                    source_object,
                    position_start=chunk_index * chunk_size,
                )
                if (
                    tuple(target_object.shape) != target_shape
                    or target_object.dtype != target_dtype
                ):
                    raise CachedKVBridgeError("bridge output does not match target object layout")
                target_cpu = target_object.detach().to("cpu").contiguous()
                payload = bytearray(target_cpu.view(torch.uint8).numpy().tobytes())
                if len(payload) != target_bytes:
                    raise CachedKVBridgeError("serialized target object size mismatch")
                target_payloads.append(payload)
            if torch.cuda.is_available() and bridge.device.type == "cuda":
                torch.cuda.synchronize(bridge.device)
            transform_ms = (time.perf_counter() - transform_started) * 1000

            write_started = time.perf_counter()
            writes = store.write_many_exact(target_keys, target_payloads)
            write_ms = (time.perf_counter() - write_started) * 1000
            operation_ms = (time.perf_counter() - operation_started) * 1000
            prepublish_overall_ms = (time.perf_counter() - started) * 1000
            actual_ratio = prepublish_overall_ms / native_prefill_ms
            if actual_ratio > requested_ratio:
                rollback = store.rollback(target_keys)
                return _cached_fallback(
                    started,
                    "cost_gate_failed",
                    error=(
                        f"materialization ratio {actual_ratio:.6f} exceeds "
                        f"{requested_ratio:.6f}; rollback={rollback}"
                    ),
                    timings={
                        "artifact_load_ms": load_ms,
                        "source_read_ms": read_ms,
                        "transform_ms": transform_ms,
                        "target_write_ms": write_ms,
                        "operation_ms": operation_ms,
                        "prepublish_overall_ms": prepublish_overall_ms,
                    },
                )

            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            index_records = [
                {
                    "schema_version": "goldenexperience.mooncake_external_index.v2",
                    "key": write.key,
                    "bytes": write.bytes,
                    "chunk_hash": chunk_hash,
                    "chunk_index": chunk_index,
                    "model_name": target_model_name,
                    "kv_rank": rank,
                    "cache_salt": cache_salt,
                    "put_rc": write.put_rc,
                    "created_at_utc": created_at,
                    "provenance": {
                        "source_key": source_keys[chunk_index],
                        "bridge_id": bridge.manifest.bridge_id,
                        "direction": bridge.manifest.direction,
                        "hash_algorithm": hash_algorithm,
                        "source_token_ids_sha256": prompt_binding["token_ids_sha256"],
                        "target_token_ids_sha256": prompt_binding["target_token_ids_sha256"],
                        "shared_prefix_chunk_count": len(all_chunk_hashes),
                    },
                }
                for chunk_index, (chunk_hash, write) in enumerate(
                    zip(chunk_hashes, writes, strict=True)
                )
            ]
            index_started = time.perf_counter()
            try:
                publish_external_index(request["external_index_path"], index_records)
            except Exception:
                store.rollback(target_keys)
                raise
            index_ms = (time.perf_counter() - index_started) * 1000

        overall_ms = (time.perf_counter() - started) * 1000
        return {
            "success": True,
            "fallback_reason": "none",
            "fallback_safe": False,
            "materialized": True,
            "injected": True,
            "allow_unsafe": False,
            "bridge_id": bridge.manifest.bridge_id,
            "direction": bridge.manifest.direction,
            "artifact_cache": {
                "hit": bridge_cache_hit,
                "resident_loader": bridge_loader is None,
            },
            "offline_quality_gate": {
                "checks": {
                    "artifact_approved": bridge.manifest.approved,
                    "global_scope": bridge.manifest.scope == "global",
                    "model_identity_verified": True,
                }
            },
            "runtime_quality_gate": {
                "checks": {
                    "source_exact_read": all(
                        read.read_bytes == read.expected_bytes == read.remote_bytes
                        for read in reads
                    ),
                    "target_exact_write": all(
                        write.put_rc == 0 and write.bytes == write.remote_bytes for write in writes
                    ),
                    "prompt_prefix_bound": True,
                    "cost_gate": actual_ratio <= requested_ratio,
                }
            },
            "injection": {
                "success": True,
                "injected_count": len(writes),
                "keys": target_keys,
                "bytes": sum(write.bytes for write in writes),
                "shared_prefix_tokens": len(writes) * chunk_size,
                "external_index_path": str(request["external_index_path"]),
            },
            "prompt_binding": prompt_binding,
            "source_keys": source_keys,
            "object_shape": list(target_shape),
            "object_dtype": f"torch.{bridge.manifest.target.dtype}",
            "cost": {
                "native_target_prefill_ms": native_prefill_ms,
                "actual_materialization_ratio": actual_ratio,
                "max_materialization_ratio": requested_ratio,
            },
            "timings": {
                "artifact_load_ms": load_ms,
                "source_read_ms": read_ms,
                "transform_ms": transform_ms,
                "target_write_ms": write_ms,
                "external_index_publish_ms": index_ms,
                "operation_ms": operation_ms,
                "prepublish_overall_ms": prepublish_overall_ms,
                "overall_ms": overall_ms,
            },
            "elapsed_ms": overall_ms,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return _cached_fallback(started, "invalid_materializer_request", error=str(exc))
    except CachedKVBridgeError as exc:
        return _cached_fallback(started, "bridge_artifact_invalid", error=str(exc))
    except MooncakeObjectError as exc:
        return _cached_fallback(started, "mooncake_exact_io_failed", error=str(exc))
    except (OSError, RuntimeError) as exc:
        return _cached_fallback(started, "materializer_runtime_failed", error=repr(exc))


def preload_cached_qwen3_bridge(request: dict[str, Any]) -> dict[str, Any]:
    """Warm an approved bridge in a resident worker without touching Mooncake."""

    import torch

    started = time.perf_counter()
    try:
        device = request.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        bridge, cache_hit = _RESIDENT_BRIDGE_CACHE.load(
            _required_string(request, "bridge_manifest_path"),
            source_model_path=_required_string(request, "source_model_path"),
            target_model_path=_required_string(request, "target_model_path"),
            device=device,
        )
        if request.get("direction") not in {None, bridge.manifest.direction}:
            raise ValueError("requested direction does not match bridge artifact")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "success": True,
            "fallback_reason": "none",
            "fallback_safe": False,
            "materialized": False,
            "injected": False,
            "bridge_id": bridge.manifest.bridge_id,
            "direction": bridge.manifest.direction,
            "artifact_cache": {"hit": cache_hit, "resident_loader": True},
            "timings": {"artifact_load_ms": elapsed_ms, "overall_ms": elapsed_ms},
            "elapsed_ms": elapsed_ms,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return _cached_fallback(started, "invalid_preload_request", error=str(exc))
    except CachedKVBridgeError as exc:
        return _cached_fallback(started, "bridge_artifact_invalid", error=str(exc))
    except (OSError, RuntimeError) as exc:
        return _cached_fallback(started, "preload_runtime_failed", error=repr(exc))


def _cached_fallback(
    started: float,
    reason: str,
    *,
    error: str,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "success": False,
        "fallback_reason": reason,
        "fallback_safe": True,
        "materialized": False,
        "injected": False,
        "allow_unsafe": False,
        "error": error,
        "timings": {**(timings or {}), "overall_ms": elapsed_ms},
        "elapsed_ms": elapsed_ms,
    }


def _validate_lookup_record(
    record: dict[str, Any],
    *,
    source_model_name: str,
    chunk_hashes: list[str],
    chunk_size: int,
    source_layers: int,
    source_width: int,
    source_dtype: str,
) -> None:
    record_hashes = _chunk_hashes(record.get("chunk_hashes"))
    if record.get("model_name") != source_model_name:
        raise ValueError("source lookup model namespace mismatch")
    if record_hashes[: len(chunk_hashes)] != chunk_hashes:
        raise ValueError("source lookup chunk hashes do not match the shared prompt prefix")
    if int(record.get("chunk_size", -1)) != chunk_size:
        raise ValueError("source lookup chunk_size mismatch")
    seq_len = int(record.get("seq_len", -1))
    if seq_len < 0 or len(record_hashes) != seq_len // chunk_size:
        raise ValueError("source lookup chunk count does not match seq_len")
    expected_shape = [[2, source_layers, chunk_size, source_width]]
    if record.get("shapes") != expected_shape:
        raise ValueError(
            f"source lookup shape mismatch: expected={expected_shape}, got={record.get('shapes')}"
        )
    if record.get("dtypes") != [f"torch.{source_dtype}"]:
        raise ValueError("source lookup dtype mismatch")


def _validate_prompt_binding(
    binding: dict[str, Any],
    record: dict[str, Any],
    *,
    chunk_hashes: list[str],
    hash_algorithm: str,
) -> None:
    source_response_id = binding.get("source_response_id")
    lookup_request_id = record.get("request_id")
    if not isinstance(source_response_id, str) or not source_response_id:
        raise ValueError("prompt binding source_response_id is required")
    if not isinstance(lookup_request_id, str) or not (
        lookup_request_id == source_response_id
        or lookup_request_id.startswith(source_response_id + "-")
    ):
        raise ValueError("source lookup request_id does not match prompt binding")
    if binding.get("lookup_request_id") != lookup_request_id:
        raise ValueError("prompt binding lookup_request_id mismatch")
    if int(binding.get("token_count", -1)) != int(record.get("seq_len", -2)):
        raise ValueError("prompt binding token count mismatch")
    if int(binding.get("chunk_size", -1)) != int(record.get("chunk_size", -2)):
        raise ValueError("prompt binding chunk_size mismatch")
    target_token_count = int(binding.get("target_token_count", -1))
    chunk_size = int(binding["chunk_size"])
    if target_token_count < len(chunk_hashes) * chunk_size:
        raise ValueError("prompt binding target token count is shorter than the shared prefix")
    if int(binding.get("shared_prefix_chunk_count", -1)) != len(chunk_hashes):
        raise ValueError("prompt binding shared prefix chunk count mismatch")
    if int(binding.get("shared_prefix_token_count", -1)) != len(chunk_hashes) * chunk_size:
        raise ValueError("prompt binding shared prefix token count mismatch")
    if binding.get("hash_algorithm") != hash_algorithm:
        raise ValueError("prompt binding hash algorithm mismatch")
    for name in ("token_ids_sha256", "target_token_ids_sha256"):
        token_digest = binding.get(name)
        if not isinstance(token_digest, str) or len(token_digest) != 64:
            raise ValueError(f"prompt binding {name} is required")
        try:
            int(token_digest, 16)
        except ValueError as exc:
            raise ValueError(f"prompt binding {name} is invalid") from exc


def _chunk_hashes(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("chunk_hashes must be a non-empty list")
    hashes = [str(item).lower() for item in value]
    if len(set(hashes)) != len(hashes):
        raise ValueError("chunk_hashes must be unique")
    for item in hashes:
        text = item[2:] if item.startswith("0x") else item
        if not text:
            raise ValueError("chunk hash is empty")
        int(text, 16)
    return [item if item.startswith("0x") else "0x" + item for item in hashes]


def _required_string(request: dict[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported cached KV dtype: {name}") from exc


def materialize_qwen3_8b_to_14b(request: dict[str, Any]) -> dict[str, Any]:
    """Run the historical two-prefill experiment without production injection."""

    if request.get("inject_to_mooncake") or request.get("allow_unsafe"):
        return {
            "success": False,
            "fallback_reason": "legacy_materializer_injection_disabled",
            "fallback_safe": True,
            "materialized": False,
            "injected": False,
            "allow_unsafe": False,
            "elapsed_ms": 0.0,
        }

    import torch
    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    token_ids = [int(item) for item in request["token_ids"]]
    chunk_hashes = [str(item) for item in request["chunk_hashes"]]
    chunk_size = int(request.get("chunk_size", 16))
    requested_max_chunks = int(request.get("max_chunks", len(chunk_hashes)))
    max_chunks = (
        len(chunk_hashes)
        if requested_max_chunks <= 0
        else min(requested_max_chunks, len(chunk_hashes))
    )
    prefix_tokens = max_chunks * chunk_size
    token_ids = token_ids[:prefix_tokens]
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_summary_path = Path(request["bridge_summary_path"])
    bridge_weights_path = Path(request["bridge_weights_path"])
    thresholds = {
        "hidden_cosine": float(request.get("min_hidden_cosine", 0.90)),
        "key_cosine": float(request.get("min_key_cosine", 0.85)),
        "value_cosine": float(request.get("min_value_cosine", 0.85)),
        "decode_logit_cosine": float(request.get("min_decode_logit_cosine", 0.90)),
    }
    bridge_summary = json.loads(bridge_summary_path.read_text(encoding="utf-8"))
    offline_gate_passed, offline_gate = _gate(bridge_summary, thresholds)
    allow_unsafe = bool(request.get("allow_unsafe", False))
    if not offline_gate_passed and not allow_unsafe:
        return {
            "success": False,
            "fallback_reason": "quality_gate_failed",
            "offline_quality_gate": offline_gate,
            "materialized": False,
            "injected": False,
            "allow_unsafe": allow_unsafe,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    device = request.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    small_path = request["source_model_path"]
    large_path = request["target_model_path"]
    dtype = torch.bfloat16

    weights = torch.load(bridge_weights_path, map_location="cpu", weights_only=True)[
        "learned_states"
    ]
    projectors = {
        int(layer): _make_hidden_projector(state, device=device) for layer, state in weights.items()
    }
    source_layer_overrides = {
        int(layer): tuple(int(item) for item in state.get("source_layer_ids", ()))
        for layer, state in weights.items()
        if state.get("source_layer_ids") is not None
    }

    small_model = AutoModelForCausalLM.from_pretrained(
        small_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()
    large_model = AutoModelForCausalLM.from_pretrained(
        large_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},
    ).eval()
    small_capture = _PreKVHiddenCapture(small_model)
    large_capture = _PreKVHiddenCapture(large_model)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    small_capture.clear()
    large_capture.clear()
    with torch.inference_mode():
        small_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        large_out = large_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    layer_map = _build_layer_map(len(small_model.model.layers), len(large_model.model.layers))
    target_layer_kv: dict[int, tuple[Any, Any]] = {}
    hidden_cosines: list[float] = []
    key_cosines: list[float] = []
    value_cosines: list[float] = []
    for target_layer, source_layers, source_weights in layer_map:
        override_layers = source_layer_overrides.get(target_layer)
        if override_layers:
            projector_input = torch.cat(
                [
                    small_capture.hidden_by_layer[source_layer].to(device)
                    for source_layer in override_layers
                ],
                dim=-1,
            )
        else:
            blended = None
            for source_layer, source_weight in zip(source_layers, source_weights, strict=True):
                src = small_capture.hidden_by_layer[source_layer].to(device)
                blended = src * source_weight if blended is None else blended + src * source_weight
            assert blended is not None
            projector_input = blended
        bridged = projectors[target_layer](projector_input)
        native_hidden = large_capture.hidden_by_layer[target_layer].to(device)
        hidden_cosines.append(_cosine_mean(bridged, native_hidden))
        key, value = _project_target_kv(large_model, bridged, target_layer, position_ids)
        target_layer_kv[target_layer] = _flatten_layer_kv(key, value)
        native_layer = large_out.past_key_values.layers[target_layer]
        key_cosines.append(_cosine_mean(key, native_layer.keys))
        value_cosines.append(_cosine_mean(value, native_layer.values))

    runtime_quality = {
        "hidden_cosine_mean": _mean(hidden_cosines),
        "key_cosine_mean": _mean(key_cosines),
        "value_cosine_mean": _mean(value_cosines),
    }
    runtime_checks = {
        "hidden_cosine": runtime_quality["hidden_cosine_mean"] >= thresholds["hidden_cosine"],
        "key_cosine": runtime_quality["key_cosine_mean"] >= thresholds["key_cosine"],
        "value_cosine": runtime_quality["value_cosine_mean"] >= thresholds["value_cosine"],
    }
    runtime_gate_passed = all(runtime_checks.values())
    if not runtime_gate_passed and not allow_unsafe:
        small_capture.close()
        large_capture.close()
        del small_model, large_model, small_capture, large_capture, target_layer_kv, large_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "success": False,
            "fallback_reason": "runtime_quality_gate_failed",
            "offline_quality_gate": offline_gate,
            "runtime_quality_gate": {
                "values": runtime_quality,
                "checks": runtime_checks,
                "thresholds": thresholds,
            },
            "materialized": False,
            "injected": False,
            "allow_unsafe": allow_unsafe,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    chunk_files = []
    shape = [
        2,
        len(large_model.model.layers),
        chunk_size,
        large_model.config.num_key_value_heads * large_model.config.head_dim,
    ]
    for chunk_index, chunk_hash in enumerate(chunk_hashes[:max_chunks]):
        start = chunk_index * chunk_size
        end = start + chunk_size
        obj = torch.zeros(shape, dtype=dtype, device="cpu")
        for layer_id in range(shape[1]):
            key_flat, value_flat = target_layer_kv[layer_id]
            available = max(0, min(end, key_flat.shape[0]) - start)
            if available:
                obj[0, layer_id, :available].copy_(
                    key_flat[start : start + available].to("cpu"),
                    non_blocking=False,
                )
                obj[1, layer_id, :available].copy_(
                    value_flat[start : start + available].to("cpu"),
                    non_blocking=False,
                )
        path = output_dir / f"chunk_{chunk_index:05d}_{chunk_hash.replace('0x', '')}.bin"
        path.write_bytes(obj.contiguous().view(torch.uint8).numpy().tobytes())
        chunk_files.append(
            {
                "chunk_index": chunk_index,
                "chunk_hash": chunk_hash,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        )

    small_capture.close()
    large_capture.close()
    del small_model, large_model, small_capture, large_capture, target_layer_kv, large_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    injection = {"success": False, "reason": "disabled"}
    if request.get("inject_to_mooncake", False):
        from goldenexperience.runtime.cross_model_reuse import inject_chunks_to_mooncake

        injection = inject_chunks_to_mooncake(
            setup_config=dict(request["mooncake_setup_config"]),
            target_model_name=str(request.get("target_model_name", large_path)),
            chunk_files=chunk_files,
            external_index_path=Path(request["external_index_path"]),
            kv_rank=request.get("kv_rank"),
            cache_salt=str(request.get("cache_salt", "")),
            provenance={
                "source_model_path": small_path,
                "target_model_path": large_path,
                "bridge_summary_path": str(bridge_summary_path),
                "bridge_weights_path": str(bridge_weights_path),
                "allow_unsafe": allow_unsafe,
            },
        )

    return {
        "success": True,
        "fallback_reason": "none",
        "offline_quality_gate": offline_gate,
        "runtime_quality_gate": {
            "values": runtime_quality,
            "checks": runtime_checks,
            "thresholds": thresholds,
        },
        "allow_unsafe": allow_unsafe,
        "materialized": True,
        "injected": bool(injection.get("success")),
        "injection": injection,
        "chunk_files": chunk_files,
        "object_shape": shape,
        "object_dtype": "torch.bfloat16",
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def _dispatch_materializer_request(request: dict[str, Any]) -> dict[str, Any]:
    try:
        mode = request.get("mode", "cached_kv")
        if mode == "cached_kv":
            return materialize_cached_qwen3(request)
        if mode == "preload_cached_kv_bridge":
            return preload_cached_qwen3_bridge(request)
        if mode == "legacy_hidden_bridge_experiment":
            return materialize_qwen3_8b_to_14b(request)
        return {
            "success": False,
            "fallback_reason": "invalid_materializer_mode",
            "fallback_safe": True,
            "materialized": False,
            "injected": False,
        }
    except Exception as exc:  # noqa: BLE001 - request boundary.
        return {
            "success": False,
            "fallback_reason": "materializer_error",
            "fallback_safe": True,
            "materialized": False,
            "injected": False,
            "error": repr(exc),
        }


def serve_materializer_jsonl(input_stream: Any, output_stream: Any) -> int:
    """Serve one request and one compact response per line until EOF."""

    for line_number, line in enumerate(input_stream, start=1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            result = _dispatch_materializer_request(request)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {
                "success": False,
                "fallback_reason": "invalid_jsonl_request",
                "fallback_safe": True,
                "materialized": False,
                "injected": False,
                "line_number": line_number,
                "error": str(exc),
            }
        output_stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        output_stream.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize cross-size Qwen3 cached KV objects.")
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    if args.serve_jsonl:
        if args.request is not None or args.output is not None:
            parser.error("--serve-jsonl cannot be combined with --request or --output")
        return serve_materializer_jsonl(sys.stdin, sys.stdout)
    if args.request is None or args.output is None:
        parser.error("--request and --output are required outside --serve-jsonl mode")
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        parser.error("--request must contain a JSON object")
    result = _dispatch_materializer_request(request)
    _write_json(Path(args.output), result)
    return 0 if result.get("success") or result.get("fallback_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
