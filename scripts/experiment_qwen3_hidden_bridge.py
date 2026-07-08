#!/usr/bin/env python3
"""Offline Qwen3 hidden-state bridge experiment for cross-model KV reuse."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from goldenexperience.size_variant import (
    HiddenBridgeMaterializer,
    HiddenStateChunk,
    TargetKVRestorer,
    build_calibration_manifest,
)
from goldenexperience.size_variant.calibration import QWEN3_14B, QWEN3_8B


DEFAULT_SMALL_MODEL = "/workspace/volume/softdata/models/Qwen3-8B"
DEFAULT_LARGE_MODEL = "/workspace/volume/softdata/models/Qwen3-14B"

CALIBRATION_TEXTS = [
    (
        "A product team is planning a staged rollout for a document search feature. "
        "They need to define success metrics, decide how to instrument latency, and "
        "document what happens when the index is stale. Write a short design note "
        "that explains rollout steps, failure handling, and the tradeoff between "
        "freshness and cost."
    ),
    (
        "A city wants to reduce energy usage in public buildings. Describe a plan "
        "that combines sensor data, operator training, and maintenance scheduling. "
        "Include one concrete example about how delayed maintenance can distort "
        "observed savings and how to detect that issue early."
    ),
    (
        "Explain to a junior engineer how a distributed cache can improve latency "
        "while still creating hard consistency problems. Compare cache invalidation, "
        "background refresh, and write-through updates, and end with a recommendation "
        "for a service that reads much more often than it writes."
    ),
    (
        "You are reviewing a proposal for a code execution sandbox used by data "
        "scientists. Summarize the main security boundaries, the logging that should "
        "be kept for incident response, and the most likely operational bottlenecks "
        "once hundreds of notebook sessions run at the same time."
    ),
    (
        "A teaching assistant is preparing study material about probability. Provide "
        "an intuitive explanation of conditional probability, Bayes' rule, and why "
        "base-rate neglect leads people to make poor decisions in medical screening "
        "or fraud detection."
    ),
    (
        "Describe a migration from a monolithic application to service boundaries. "
        "The answer should discuss schema versioning, request tracing, backward "
        "compatibility, and how to reduce the blast radius when one new service "
        "introduces a latent performance regression."
    ),
]

EVAL_TEXTS = [
    (
        "A company is building an internal assistant that reads engineering tickets. "
        "The assistant must summarize incidents, extract action items, and avoid "
        "inventing root causes. Explain how retrieval, caching, and evaluation should "
        "work when the source data changes every hour."
    ),
    (
        "Write a compact explanation of why runtime profiling and offline benchmarks "
        "often disagree. Include discussion of cold start effects, queueing, cache "
        "state, and the difference between average latency and tail latency."
    ),
    (
        "A research group is comparing two language models for code review. Summarize "
        "an evaluation protocol that checks correctness, hallucination rate, and "
        "runtime cost, and note why replaying the same prompt with different cache "
        "state can change the observed result."
    ),
]


def build_synthetic_calibration_texts() -> list[str]:
    subjects = [
        "inventory forecasting",
        "incident response",
        "schema migration",
        "GPU scheduling",
        "policy evaluation",
        "warehouse routing",
        "fraud review",
        "A/B rollout",
        "distributed tracing",
        "backup verification",
        "pricing analytics",
        "customer support",
    ]
    actions = [
        (
            "Explain the rollout plan, define the main metrics, and identify one "
            "operational failure mode that should trigger a fallback."
        ),
        (
            "Describe how to instrument latency, measure accuracy drift, and "
            "communicate tradeoffs to a team that has limited on-call capacity."
        ),
    ]
    prompts = []
    for subject in subjects:
        for action in actions:
            prompts.append(
                (
                    f"A staff engineer is writing a design note about {subject}. "
                    f"{action} Include one concrete example and one risk that would "
                    "only appear after several weeks in production."
                )
            )
    return prompts


class PreKVHiddenCapture:
    """Capture the hidden state passed into each attention module."""

    def __init__(self, model: Any) -> None:
        self.hidden_by_layer: dict[int, torch.Tensor] = {}
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


def resolve_model_path(preferred: str, fallback: str) -> str:
    if Path(preferred).exists():
        return preferred
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError(f"Model path not found: {preferred} and fallback {fallback}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_key_rope(rotary_emb, key: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    cos, sin = rotary_emb(key.transpose(1, 2), position_ids)
    return (key * cos.unsqueeze(1)) + (rotate_half(key) * sin.unsqueeze(1))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def cosine_mean(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_flat = lhs.float().reshape(-1, lhs.shape[-1])
    rhs_flat = rhs.float().reshape(-1, rhs.shape[-1])
    return float(F.cosine_similarity(lhs_flat, rhs_flat, dim=-1).mean().item())


def format_prompt_text(item: dict[str, Any]) -> str:
    if "messages" not in item:
        return str(item)
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in item["messages"])


def load_prompt_file(path: str | None) -> list[str]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = payload.get("prompts", [])
    return [format_prompt_text(item) for item in prompts]


def tokenize_text(
    tokenizer,
    text: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )


def collect_states_from_ids(
    small_model,
    large_model,
    small_capture: PreKVHiddenCapture,
    large_capture: PreKVHiddenCapture,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)

    small_capture.clear()
    large_capture.clear()
    with torch.inference_mode():
        small_out = small_model(
            input_ids=input_ids.to("cuda:0"),
            attention_mask=attention_mask.to("cuda:0"),
            use_cache=True,
        )
        large_out = large_model(
            input_ids=input_ids.to("cuda:1"),
            attention_mask=attention_mask.to("cuda:1"),
            use_cache=True,
        )

    small_hidden = {layer: hidden for layer, hidden in small_capture.hidden_by_layer.items()}
    large_hidden = {layer: hidden for layer, hidden in large_capture.hidden_by_layer.items()}
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "small_hidden": small_hidden,
        "large_hidden": large_hidden,
        "small_out": small_out,
        "large_out": large_out,
    }


def build_projector_state(
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    rank: int,
    ridge_lambda: float,
    device: str,
    target_metric_cholesky: torch.Tensor | None = None,
    target_metric_inverse: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    x = x_samples.to(device=device, dtype=torch.float32)
    y = y_samples.to(device=device, dtype=torch.float32)
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    x_centered = x - mean_x
    y_centered = y - mean_y
    effective_rank = max(1, min(rank, x_centered.shape[0], x_centered.shape[1], y_centered.shape[0], y_centered.shape[1]))
    _, _, source_basis = torch.pca_lowrank(x_centered, q=effective_rank, center=False)
    if target_metric_cholesky is not None and target_metric_inverse is not None:
        y_metric = y_centered @ target_metric_cholesky
    else:
        y_metric = y_centered
    _, _, target_basis = torch.pca_lowrank(y_metric, q=effective_rank, center=False)
    x_reduced = x_centered @ source_basis
    y_reduced = y_metric @ target_basis
    gram = x_reduced.T @ x_reduced
    gram.diagonal().add_(ridge_lambda)
    bridge = torch.linalg.solve(gram, x_reduced.T @ y_reduced)
    target_projection = target_basis.T
    if target_metric_inverse is not None:
        target_projection = target_projection @ target_metric_inverse
    state = {
        "mean_x": mean_x.cpu(),
        "mean_y": mean_y.cpu(),
        "source_basis": source_basis.cpu(),
        "target_projection": target_projection.cpu(),
        "bridge": bridge.cpu(),
    }
    del x, y, x_centered, y_centered, y_metric, source_basis, target_basis, x_reduced, y_reduced, gram, bridge, target_projection
    torch.cuda.empty_cache()
    return state


def make_hidden_projector(state: dict[str, torch.Tensor], device: str):
    mean_x = state["mean_x"].to(device=device, dtype=torch.float32)
    mean_y = state["mean_y"].to(device=device, dtype=torch.float32)
    source_basis = state["source_basis"].to(device=device, dtype=torch.float32)
    target_projection = state.get("target_projection")
    if target_projection is None:
        target_projection = state["target_basis"].T
    target_projection = target_projection.to(device=device, dtype=torch.float32)
    bridge = state["bridge"].to(device=device, dtype=torch.float32)

    def projector(hidden: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden.dtype
        original_shape = hidden.shape
        flat = hidden.to(dtype=torch.float32).reshape(-1, original_shape[-1])
        out = (((flat - mean_x) @ source_basis) @ bridge) @ target_projection + mean_y
        return out.to(dtype=original_dtype).reshape(*original_shape[:-1], -1)

    return projector


def build_target_output_metric(
    attn,
    target_hidden_size: int,
    value_loss_weight: float,
    key_loss_weight: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if value_loss_weight <= 0.0 and key_loss_weight <= 0.0:
        return None, None
    metric = torch.eye(target_hidden_size, device=device, dtype=torch.float32)
    if value_loss_weight > 0.0:
        value_matrix = attn.v_proj.weight.detach().to(device=device, dtype=torch.float32).T
        metric = metric + value_loss_weight * (target_hidden_size / value_matrix.shape[1]) * (value_matrix @ value_matrix.T)
    if key_loss_weight > 0.0:
        key_matrix = attn.k_proj.weight.detach().to(device=device, dtype=torch.float32).T
        metric = metric + key_loss_weight * (target_hidden_size / key_matrix.shape[1]) * (key_matrix @ key_matrix.T)
    chol = torch.linalg.cholesky(metric)
    chol_inv = torch.linalg.solve(chol, torch.eye(target_hidden_size, device=device, dtype=torch.float32))
    del metric
    torch.cuda.empty_cache()
    return chol, chol_inv


def make_qwen3_key_projector(attn):
    def projector(hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        view_shape = (batch, seq, attn.config.num_key_value_heads, attn.head_dim)
        key = attn.k_proj(hidden).view(view_shape)
        key = attn.k_norm(key).transpose(1, 2)
        return key

    return projector


def make_qwen3_value_projector(attn):
    def projector(hidden: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        view_shape = (batch, seq, attn.config.num_key_value_heads, attn.head_dim)
        value = attn.v_proj(hidden).view(view_shape).transpose(1, 2)
        return value

    return projector


def make_qwen3_rope_fn(rotary_emb):
    def rope_fn(key: torch.Tensor, value: torch.Tensor, position_ids: tuple[int, ...] | None):
        if position_ids is None:
            raise ValueError("position_ids are required for Qwen3 RoPE restore")
        position_tensor = torch.tensor([list(position_ids)], device=key.device, dtype=torch.long)
        return qwen3_key_rope(rotary_emb, key, position_tensor), value

    return rope_fn


def fit_hidden_bridge(
    manifest,
    tokenizer,
    small_model,
    large_model,
    small_capture: PreKVHiddenCapture,
    large_capture: PreKVHiddenCapture,
    calibration_texts: list[str],
    max_length: int,
    max_samples_per_layer: int,
    rank: int,
    ridge_lambda: float,
    value_loss_weight: float,
    key_loss_weight: float,
) -> dict[int, dict[str, torch.Tensor]]:
    layer_samples: dict[int, dict[str, list[torch.Tensor] | int]] = {
        entry.target_layer_id: {"x": [], "y": [], "count": 0}
        for entry in manifest.layer_map.entries
    }

    for text in calibration_texts:
        encoded = tokenize_text(tokenizer, text, max_length)
        states = collect_states_from_ids(
            small_model,
            large_model,
            small_capture,
            large_capture,
            encoded["input_ids"],
            encoded["attention_mask"],
        )
        for entry in manifest.layer_map.entries:
            layer_id = entry.target_layer_id
            sample_info = layer_samples[layer_id]
            remaining = max_samples_per_layer - int(sample_info["count"])
            if remaining <= 0:
                continue
            blended = None
            for source_layer_id, weight in zip(entry.source_layer_ids, entry.weights):
                source_hidden = states["small_hidden"][source_layer_id].float().to("cpu")
                blended = source_hidden * weight if blended is None else blended + source_hidden * weight
            target_hidden = states["large_hidden"][layer_id].float().to("cpu")
            x = blended.reshape(-1, blended.shape[-1])
            y = target_hidden.reshape(-1, target_hidden.shape[-1])
            take = min(remaining, x.shape[0])
            sample_info["x"].append(x[:take].contiguous())
            sample_info["y"].append(y[:take].contiguous())
            sample_info["count"] = int(sample_info["count"]) + take
        del states
        gc.collect()
        torch.cuda.empty_cache()

    fitted: dict[int, dict[str, torch.Tensor]] = {}
    for entry in manifest.layer_map.entries:
        layer_id = entry.target_layer_id
        x = torch.cat(layer_samples[layer_id]["x"], dim=0)
        y = torch.cat(layer_samples[layer_id]["y"], dim=0)
        attn = large_model.model.layers[layer_id].self_attn
        metric_chol, metric_chol_inv = build_target_output_metric(
            attn=attn,
            target_hidden_size=y.shape[-1],
            value_loss_weight=value_loss_weight,
            key_loss_weight=key_loss_weight,
            device="cuda:1",
        )
        fitted[layer_id] = build_projector_state(
            x,
            y,
            rank=rank,
            ridge_lambda=ridge_lambda,
            device="cuda:1",
            target_metric_cholesky=metric_chol,
            target_metric_inverse=metric_chol_inv,
        )
        del metric_chol, metric_chol_inv
        del x, y
        gc.collect()
    return fitted


def evaluate_bridge(
    name: str,
    manifest,
    tokenizer,
    small_model,
    large_model,
    small_capture: PreKVHiddenCapture,
    large_capture: PreKVHiddenCapture,
    eval_texts: list[str],
    max_length: int,
    prefix_tokens: int,
    layer_projectors: dict[int, Any] | None,
) -> dict[str, Any]:
    rotary_emb = large_model.model.rotary_emb
    key_projectors = {
        idx: make_qwen3_key_projector(layer.self_attn)
        for idx, layer in enumerate(large_model.model.layers)
    }
    value_projectors = {
        idx: make_qwen3_value_projector(layer.self_attn)
        for idx, layer in enumerate(large_model.model.layers)
    }
    rope_fns = {idx: make_qwen3_rope_fn(rotary_emb) for idx, _ in enumerate(large_model.model.layers)}
    restorer = TargetKVRestorer(
        manifest,
        layer_key_projectors=key_projectors,
        layer_value_projectors=value_projectors,
        layer_rope_fns=rope_fns,
    )

    results = []
    for text in eval_texts:
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length + 4,
        )
        full_ids = tokenized["input_ids"][0]
        if full_ids.shape[0] <= 4:
            continue
        prompt_len = min(prefix_tokens, full_ids.shape[0] - 2)
        prefix_ids = full_ids[:prompt_len].unsqueeze(0)
        next_input = full_ids[prompt_len : prompt_len + 1].unsqueeze(0).to("cuda:1")
        attention_mask = torch.ones_like(prefix_ids)
        states = collect_states_from_ids(
            small_model,
            large_model,
            small_capture,
            large_capture,
            prefix_ids,
            attention_mask,
        )

        source_chunks = {
            layer_id: HiddenStateChunk(
                layer_id=layer_id,
                hidden=hidden.to("cuda:1"),
                token_start=0,
                token_end=hidden.shape[1],
                position_ids=tuple(range(hidden.shape[1])),
                dtype=str(hidden.dtype).replace("torch.", ""),
            )
            for layer_id, hidden in states["small_hidden"].items()
        }
        hidden_result = HiddenBridgeMaterializer(
            manifest,
            layer_projectors=layer_projectors,
        ).materialize(source_chunks)
        if not hidden_result.success:
            results.append(
                {
                    "name": name,
                    "prompt_len": int(prompt_len),
                    "error": hidden_result.error,
                    "success": False,
                }
            )
            continue

        reconstructed_cache = DynamicCache(config=large_model.config)
        hidden_cosines = []
        key_cosines = []
        value_cosines = []
        for hidden_chunk in hidden_result.chunks:
            target_hidden = states["large_hidden"][hidden_chunk.layer_id].to("cuda:1")
            hidden_cosines.append(cosine_mean(hidden_chunk.hidden, target_hidden))
            kv_result = restorer.restore_chunk(hidden_chunk)
            if not kv_result.success:
                raise RuntimeError(kv_result.error or "KV restore failed")
            kv_chunk = kv_result.chunks[0]
            reconstructed_cache.update(kv_chunk.key, kv_chunk.value, kv_chunk.layer_id)
            native_layer = states["large_out"].past_key_values.layers[kv_chunk.layer_id]
            key_cosines.append(cosine_mean(kv_chunk.key, native_layer.keys))
            value_cosines.append(cosine_mean(kv_chunk.value, native_layer.values))

        with torch.inference_mode():
            native_decode = large_model(input_ids=next_input, past_key_values=states["large_out"].past_key_values, use_cache=True)
            bridge_decode = large_model(input_ids=next_input, past_key_values=reconstructed_cache, use_cache=True)
        native_logits = native_decode.logits[:, -1, :]
        bridge_logits = bridge_decode.logits[:, -1, :]
        logit_cosine = cosine_mean(native_logits, bridge_logits)
        native_top1 = int(native_logits.argmax(dim=-1).item())
        bridge_top1 = int(bridge_logits.argmax(dim=-1).item())

        results.append(
            {
                "name": name,
                "success": True,
                "prompt_len": int(prompt_len),
                "hidden_cosine_mean": mean(hidden_cosines),
                "key_cosine_mean": mean(key_cosines),
                "value_cosine_mean": mean(value_cosines),
                "decode_logit_cosine": logit_cosine,
                "decode_top1_match": native_top1 == bridge_top1,
                "native_top1_token_id": native_top1,
                "bridge_top1_token_id": bridge_top1,
            }
        )
        del states, hidden_result, reconstructed_cache, native_decode, bridge_decode
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "name": name,
        "num_prompts": len(results),
        "prompt_results": results,
        "success_rate": mean([1.0 if item.get("success") else 0.0 for item in results]),
        "hidden_cosine_mean": mean([item["hidden_cosine_mean"] for item in results if item.get("success")]),
        "key_cosine_mean": mean([item["key_cosine_mean"] for item in results if item.get("success")]),
        "value_cosine_mean": mean([item["value_cosine_mean"] for item in results if item.get("success")]),
        "decode_logit_cosine_mean": mean([item["decode_logit_cosine"] for item in results if item.get("success")]),
        "decode_top1_match_rate": mean([1.0 if item.get("decode_top1_match") else 0.0 for item in results if item.get("success")]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline Qwen3 hidden-state bridge experiment.")
    parser.add_argument("--small-model-path", default="/data/models/Qwen3-8B")
    parser.add_argument("--large-model-path", default="/data/models/Qwen3-14B")
    parser.add_argument("--prompt-file", default="configs/kv_baseline_prompts.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=48)
    parser.add_argument("--rank", type=int, default=192)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--max-samples-per-layer", type=int, default=2048)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--key-loss-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hidden_bridge/qwen3_hidden_bridge_experiment.json"))
    args = parser.parse_args()

    small_model_path = resolve_model_path(args.small_model_path, DEFAULT_SMALL_MODEL)
    large_model_path = resolve_model_path(args.large_model_path, DEFAULT_LARGE_MODEL)
    prompt_file_texts = load_prompt_file(args.prompt_file)
    calibration_texts = CALIBRATION_TEXTS + build_synthetic_calibration_texts() + prompt_file_texts
    eval_texts = EVAL_TEXTS

    tokenizer = AutoTokenizer.from_pretrained(large_model_path, trust_remote_code=True)
    small_model = AutoModelForCausalLM.from_pretrained(
        small_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
    ).eval()
    large_model = AutoModelForCausalLM.from_pretrained(
        large_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda:1"},
    ).eval()

    small_capture = PreKVHiddenCapture(small_model)
    large_capture = PreKVHiddenCapture(large_model)
    manifest = build_calibration_manifest(
        source=QWEN3_8B,
        target=QWEN3_14B,
        calibration_id="qwen3_8b_to_14b_hidden_bridge_experiment",
        prompts_count=len(calibration_texts),
    )

    learned_states = fit_hidden_bridge(
        manifest=manifest,
        tokenizer=tokenizer,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        calibration_texts=calibration_texts,
        max_length=args.max_length,
        max_samples_per_layer=args.max_samples_per_layer,
        rank=args.rank,
        ridge_lambda=args.ridge_lambda,
        value_loss_weight=args.value_loss_weight,
        key_loss_weight=args.key_loss_weight,
    )
    learned_projectors = {
        layer_id: make_hidden_projector(state, device="cuda:1")
        for layer_id, state in learned_states.items()
    }

    identity_metrics = evaluate_bridge(
        name="identity_bridge_baseline",
        manifest=manifest,
        tokenizer=tokenizer,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        eval_texts=eval_texts,
        max_length=args.max_length,
        prefix_tokens=args.prefix_tokens,
        layer_projectors=None,
    )
    learned_metrics = evaluate_bridge(
        name="learned_low_rank_hidden_bridge",
        manifest=manifest,
        tokenizer=tokenizer,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        eval_texts=eval_texts,
        max_length=args.max_length,
        prefix_tokens=args.prefix_tokens,
        layer_projectors=learned_projectors,
    )

    summary = {
        "small_model_path": small_model_path,
        "large_model_path": large_model_path,
        "rank": args.rank,
        "ridge_lambda": args.ridge_lambda,
        "max_length": args.max_length,
        "prefix_tokens": args.prefix_tokens,
        "max_samples_per_layer": args.max_samples_per_layer,
        "value_loss_weight": args.value_loss_weight,
        "key_loss_weight": args.key_loss_weight,
        "calibration_prompts": len(calibration_texts),
        "evaluation_prompts": len(eval_texts),
        "identity_bridge_baseline": identity_metrics,
        "learned_low_rank_hidden_bridge": learned_metrics,
        "improves_kv_cosine": learned_metrics["key_cosine_mean"] > identity_metrics["key_cosine_mean"],
        "improves_decode_cosine": learned_metrics["decode_logit_cosine_mean"] > identity_metrics["decode_logit_cosine_mean"],
        "learned_bridge_quality_gate": {
            "hidden_cosine_ge_0_90": learned_metrics["hidden_cosine_mean"] >= 0.90,
            "kv_cosine_ge_0_85": learned_metrics["key_cosine_mean"] >= 0.85 and learned_metrics["value_cosine_mean"] >= 0.85,
            "decode_logit_cosine_ge_0_90": learned_metrics["decode_logit_cosine_mean"] >= 0.90,
        },
        "learned_weights_path": str(args.output.with_suffix(".pt")),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    torch.save(
        {
            "manifest_calibration_id": manifest.calibration_id,
            "learned_states": learned_states,
        },
        args.output.with_suffix(".pt"),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    small_capture.close()
    large_capture.close()


if __name__ == "__main__":
    main()
