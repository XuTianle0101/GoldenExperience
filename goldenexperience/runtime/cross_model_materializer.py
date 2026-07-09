"""Qwen3 cross-size hidden-bridge KV materialization utilities."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


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
        "decode_logit_cosine": values["decode_logit_cosine_mean"] >= thresholds.get("decode_logit_cosine", 0.90),
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


def _build_layer_map(source_layers: int, target_layers: int) -> list[tuple[int, tuple[int, ...], tuple[float, ...]]]:
    if target_layers == 1:
        return [(0, (0,), (1.0,))]
    entries = []
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


def _project_target_kv(large_model: Any, hidden: Any, layer_id: int, position_ids: Any) -> tuple[Any, Any]:
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


def materialize_qwen3_8b_to_14b(request: dict[str, Any]) -> dict[str, Any]:
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

    weights = torch.load(bridge_weights_path, map_location="cpu", weights_only=False)["learned_states"]
    projectors = {int(layer): _make_hidden_projector(state, device=device) for layer, state in weights.items()}
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
                [small_capture.hidden_by_layer[source_layer].to(device) for source_layer in override_layers],
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
            "runtime_quality_gate": {"values": runtime_quality, "checks": runtime_checks, "thresholds": thresholds},
            "materialized": False,
            "injected": False,
            "allow_unsafe": allow_unsafe,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    chunk_files = []
    shape = [2, len(large_model.model.layers), chunk_size, large_model.config.num_key_value_heads * large_model.config.head_dim]
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
        chunk_files.append({"chunk_index": chunk_index, "chunk_hash": chunk_hash, "path": str(path), "bytes": path.stat().st_size})

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
        "runtime_quality_gate": {"values": runtime_quality, "checks": runtime_checks, "thresholds": thresholds},
        "allow_unsafe": allow_unsafe,
        "materialized": True,
        "injected": bool(injection.get("success")),
        "injection": injection,
        "chunk_files": chunk_files,
        "object_shape": shape,
        "object_dtype": "torch.bfloat16",
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Qwen3 8B->14B KV chunks from hidden bridge weights.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        result = materialize_qwen3_8b_to_14b(request)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        result = {"success": False, "fallback_reason": "materializer_error", "error": repr(exc)}
    _write_json(Path(args.output), result)
    return 0 if result.get("success") or result.get("fallback_reason") in {"quality_gate_failed", "runtime_quality_gate_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
