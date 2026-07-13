#!/usr/bin/env python3
"""Train and evaluate a bidirectional Qwen3 cached-KV bridge artifact."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.benchmarks.cached_kv_cost import load_cached_kv_cost_evidence
from goldenexperience.size_variant.cached_kv_bridge import safetensors_metadata
from goldenexperience.size_variant.cached_kv_dataset import (
    CachedKVPrompt,
    CachedKVPromptDataset,
    contains_expected_final_answer,
    render_to_token_bucket,
)
from goldenexperience.size_variant.cached_kv_manifest import (
    CachedKVBridgeManifest,
    CachedKVQualityEvidence,
    artifact_id_for,
    model_spec_from_path,
    sha256_file,
)
from goldenexperience.size_variant.cached_kv_training import (
    build_cka_source_layer_plan,
    build_source_layer_plan,
    build_training_matrices,
    cache_to_object,
    cosine_mean,
    fit_low_rank_state,
    logit_distillation_loss,
    object_to_dynamic_cache,
    transform_with_state,
)

DEFAULT_8B = "/workspace/volume/softdata/models/Qwen3-8B"
DEFAULT_14B = "/workspace/volume/softdata/models/Qwen3-14B"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _model_paths(args: argparse.Namespace) -> tuple[str, str, str, str, float, float]:
    if args.direction == "8b_to_14b":
        return args.model_8b, args.model_14b, "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", 8.0, 14.0
    return args.model_14b, args.model_8b, "Qwen/Qwen3-14B", "Qwen/Qwen3-8B", 14.0, 8.0


def _sync(device: str) -> None:
    import torch

    parsed = torch.device(device)
    if parsed.type == "cuda":
        torch.cuda.synchronize(parsed)


def _rope_theta(config: Any) -> float:
    direct = getattr(config, "rope_theta", None)
    if direct is not None:
        return float(direct)
    parameters = getattr(config, "rope_parameters", None)
    if isinstance(parameters, dict) and parameters.get("rope_theta") is not None:
        return float(parameters["rope_theta"])
    scaling = getattr(config, "rope_scaling", None)
    if isinstance(scaling, dict) and scaling.get("rope_theta") is not None:
        return float(scaling["rope_theta"])
    raise ValueError("Qwen3 config does not expose rope_theta")


def _run_prefill(model: Any, input_ids: Any, device: str) -> tuple[Any, float]:
    import torch

    tensor = input_ids.to(device)
    attention_mask = torch.ones_like(tensor)
    _sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=tensor, attention_mask=attention_mask, use_cache=True)
    _sync(device)
    return output, (time.perf_counter() - started) * 1000


def _sample_positions(token_count: int, count: int) -> Any:
    import torch

    take = min(token_count, count)
    if take <= 0:
        raise ValueError("training prompt produced no tokens")
    return torch.linspace(0, token_count - 1, steps=take).round().long().unique()


def collect_training_data(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    source_device: str,
    target_device: str,
    source_layer_ids: Any,
    source_layer_weights: Any,
    samples_per_prompt: int,
    max_samples: int,
    suffix_tokens: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    feature_parts: list[Any] = []
    key_parts: list[Any] = []
    value_parts: list[Any] = []
    prompt_records: list[dict[str, Any]] = []
    collected = 0
    source_config = source_model.config
    target_config = target_model.config
    for sample in samples:
        if collected >= max_samples:
            break
        _, token_ids = render_to_token_bucket(
            sample,
            tokenizer,
            suffix_tokens=suffix_tokens,
        )
        usable = len(token_ids)
        input_ids = torch.tensor([token_ids[:usable]], dtype=torch.long)
        source_out, source_ms = _run_prefill(source_model, input_ids, source_device)
        target_out, target_ms = _run_prefill(target_model, input_ids, target_device)
        positions = _sample_positions(usable, min(samples_per_prompt, max_samples - collected))
        source_object = cache_to_object(source_out.past_key_values)
        target_object = cache_to_object(target_out.past_key_values)
        source_selected = source_object[:, :, positions.to(source_object.device), :].cpu()
        target_selected = target_object[:, :, positions.to(target_object.device), :].cpu()
        features, key_residual, value_residual = build_training_matrices(
            source_selected,
            target_selected,
            positions,
            source_layer_ids,
            source_layer_weights,
            source_heads=int(source_config.num_key_value_heads),
            source_head_dim=int(source_config.head_dim),
            source_rope_theta=_rope_theta(source_config),
            target_heads=int(target_config.num_key_value_heads),
            target_head_dim=int(target_config.head_dim),
            target_rope_theta=_rope_theta(target_config),
        )
        feature_parts.append(features.to(torch.bfloat16))
        key_parts.append(key_residual.to(torch.bfloat16))
        value_parts.append(value_residual.to(torch.bfloat16))
        collected += int(positions.numel())
        prompt_records.append(
            {
                "prompt_id": sample.prompt_id,
                "token_bucket": sample.token_bucket,
                "rendered_tokens": usable,
                "sampled_positions": int(positions.numel()),
                "source_prefill_ms": source_ms,
                "target_prefill_ms": target_ms,
            }
        )
        del source_out, target_out, source_object, target_object
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if collected <= 0:
        raise ValueError("no training samples were collected")
    return (
        torch.cat(feature_parts, dim=1),
        torch.cat(key_parts, dim=1),
        torch.cat(value_parts, dim=1),
        {"sample_count": collected, "prompts": prompt_records},
    )


def collect_layer_alignment_data(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    source_device: str,
    target_device: str,
    max_prompts: int,
    samples_per_prompt: int,
    suffix_tokens: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    source_parts: list[Any] = []
    target_parts: list[Any] = []
    position_parts: list[Any] = []
    prompt_ids: list[str] = []
    for sample in samples[:max_prompts]:
        _, token_ids = render_to_token_bucket(sample, tokenizer, suffix_tokens=suffix_tokens)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        source_out, _ = _run_prefill(source_model, input_ids, source_device)
        target_out, _ = _run_prefill(target_model, input_ids, target_device)
        positions = _sample_positions(len(token_ids), samples_per_prompt)
        source_object = cache_to_object(source_out.past_key_values)
        target_object = cache_to_object(target_out.past_key_values)
        source_parts.append(
            source_object[:, :, positions.to(source_object.device), :].to("cpu")
        )
        target_parts.append(
            target_object[:, :, positions.to(target_object.device), :].to("cpu")
        )
        position_parts.append(positions)
        prompt_ids.append(sample.prompt_id)
        del source_out, target_out, source_object, target_object
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not source_parts:
        raise ValueError("no layer alignment samples were collected")
    positions = torch.cat(position_parts)
    return (
        torch.cat(source_parts, dim=2),
        torch.cat(target_parts, dim=2),
        positions,
        {
            "prompt_ids": prompt_ids,
            "prompt_count": len(prompt_ids),
            "sample_count": int(positions.numel()),
        },
    )


def _quantize_state(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    quantized: dict[str, Any] = {}
    for name, tensor in state.items():
        if name in {
            "key_down",
            "key_nonlinear_up",
            "key_up",
            "value_down",
            "value_nonlinear_up",
            "value_up",
        }:
            quantized[name] = tensor.to(torch.bfloat16).contiguous()
        else:
            quantized[name] = tensor.contiguous()
    return quantized


_LOGIT_REFINEMENT_PARAMETERS = (
    "key_up",
    "key_nonlinear_up",
    "key_bias",
    "value_up",
    "value_nonlinear_up",
    "value_bias",
)

_LOGIT_REFINEMENT_PARAMETER_GROUPS = {
    "all": _LOGIT_REFINEMENT_PARAMETERS,
    "bias-only": ("key_bias", "value_bias"),
    "nonlinear-up-only": ("key_nonlinear_up", "value_nonlinear_up"),
}

_LOGIT_REFINEMENT_OBJECTIVES = ("native-generation", "prompt-tail", "mixed")


def _bucket_balanced_samples(
    samples: tuple[CachedKVPrompt, ...],
    count: int,
) -> tuple[CachedKVPrompt, ...]:
    """Select prompts round-robin across token buckets while preserving bucket order."""

    if count <= 0:
        return ()
    buckets: dict[int, list[CachedKVPrompt]] = {}
    for sample in samples:
        buckets.setdefault(sample.token_bucket, []).append(sample)
    ordered_buckets = sorted(buckets)
    for index, bucket in enumerate(ordered_buckets):
        bucket_samples = buckets[bucket]
        rotation = index % len(bucket_samples)
        buckets[bucket] = bucket_samples[rotation:] + bucket_samples[:rotation]
    selected: list[CachedKVPrompt] = []
    offset = 0
    while len(selected) < count:
        added = False
        for bucket in ordered_buckets:
            bucket_samples = buckets[bucket]
            if offset < len(bucket_samples):
                selected.append(bucket_samples[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _relative_mse(left: Any, reference: Any) -> Any:
    scale = reference.detach().float().square().mean().clamp_min(1e-8)
    return (left.float() - reference.detach().float()).square().mean() / scale


def _parameter_anchor_loss(
    state: dict[str, Any],
    anchors: dict[str, Any],
    parameter_names: tuple[str, ...],
) -> Any:
    import torch

    return torch.stack(
        [_relative_mse(state[name], anchors[name]) for name in parameter_names]
    ).mean()


def _kv_anchor_losses(refined: Any, reference: Any) -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as functional

    relative_mse = torch.stack(
        [_relative_mse(refined[index], reference[index]) for index in range(2)]
    ).mean()
    cosine_loss = 1 - functional.cosine_similarity(
        refined.float().reshape(-1, refined.shape[-1]),
        reference.detach().float().reshape(-1, reference.shape[-1]),
        dim=-1,
    ).mean()
    return (relative_mse + cosine_loss) / 2, relative_mse, cosine_loss


def _native_generation_teacher(
    target_model: Any,
    token_ids: list[int],
    *,
    target_device: str,
    generation_tokens: int,
) -> tuple[Any, Any]:
    """Collect frozen native greedy logits and their generated token labels."""

    import torch

    if not token_ids or generation_tokens <= 0:
        raise ValueError("native-generation refinement requires prompt and generation tokens")
    teacher_logits: list[Any] = []
    generated_tokens: list[int] = []
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=target_device)
    with torch.inference_mode():
        output = target_model(input_ids=input_ids, use_cache=True)
        cache = output.past_key_values
        next_logits = output.logits[:, -1:]
        for step in range(generation_tokens):
            teacher_logits.append(next_logits.detach())
            next_token = int(next_logits[:, -1].argmax(dim=-1).item())
            generated_tokens.append(next_token)
            if step + 1 == generation_tokens:
                break
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=target_device)
            output = target_model(
                input_ids=next_input,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            next_logits = output.logits[:, -1:]
    return (
        torch.cat(teacher_logits, dim=1),
        torch.tensor([generated_tokens], dtype=torch.long, device=target_device),
    )


def _bridge_free_running_metrics(
    target_model: Any,
    tokenizer: Any,
    bridge_object: Any,
    *,
    seed_token: int,
    teacher_tokens: Any,
    target_device: str,
    expected_answer: str | None,
) -> dict[str, Any]:
    """Decode freely from a bridge cache and compare with native greedy labels."""

    import torch

    expected_tokens = [int(token) for token in teacher_tokens.reshape(-1).tolist()]
    if not expected_tokens:
        raise ValueError("free-running holdout requires native teacher tokens")
    cache = object_to_dynamic_cache(bridge_object, target_model.config)
    input_ids = torch.tensor([[seed_token]], dtype=torch.long, device=target_device)
    generated_tokens: list[int] = []
    with torch.inference_mode():
        for _ in expected_tokens:
            output = target_model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            next_token = int(output.logits[:, -1].argmax(dim=-1).item())
            generated_tokens.append(next_token)
            input_ids = torch.tensor([[next_token]], dtype=torch.long, device=target_device)
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "greedy_matches": sum(
            generated == expected
            for generated, expected in zip(generated_tokens, expected_tokens, strict=True)
        ),
        "greedy_tokens": len(expected_tokens),
        "task_passed": (
            None
            if expected_answer is None
            else contains_expected_final_answer(text, expected_answer)
        ),
        "text": text,
    }


def _holdout_is_better(
    current: dict[str, Any],
    best: dict[str, Any],
    *,
    min_delta: float,
) -> bool:
    """Rank checkpoints by free-running task, greedy match, then teacher objective."""

    for metric in ("free_running_task_score", "free_running_greedy_match_rate"):
        current_value = current.get(metric)
        best_value = best.get(metric)
        if current_value is None or best_value is None:
            continue
        if current_value > best_value + min_delta:
            return True
        if current_value < best_value - min_delta:
            return False
    return current["objective"] < best["objective"] - min_delta


def _refinement_prompt_forward(
    sample: CachedKVPrompt,
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    objective_mode: str,
    generation_tokens: int,
) -> tuple[Any, Any, Any, Any, Any, float]:
    import torch

    _, token_ids = render_to_token_bucket(sample, tokenizer, suffix_tokens=suffix_tokens)
    if objective_mode == "prompt-tail":
        prefix = sample.token_bucket
        continuation = token_ids[prefix : prefix + suffix_tokens + 1]
        if len(continuation) != suffix_tokens + 1:
            raise ValueError("logit refinement prompt has an incomplete continuation")
        source_token_ids = token_ids[:prefix]
        teacher_input = torch.tensor(
            [token_ids[: prefix + suffix_tokens]],
            dtype=torch.long,
            device=target_device,
        )
        with torch.inference_mode():
            teacher_out = target_model(input_ids=teacher_input, use_cache=False)
            teacher_logits = teacher_out.logits[:, prefix : prefix + suffix_tokens].detach()
        del teacher_out
        labels = torch.tensor(
            [continuation[1:]],
            dtype=torch.long,
            device=target_device,
        )
        student_token_ids = continuation[:-1]
    elif objective_mode == "native-generation":
        if len(token_ids) < 2:
            raise ValueError("native-generation refinement prompt is too short")
        source_token_ids = token_ids[:-1]
        teacher_logits, labels = _native_generation_teacher(
            target_model,
            token_ids,
            target_device=target_device,
            generation_tokens=generation_tokens,
        )
        student_token_ids = [token_ids[-1], *labels[0, :-1].tolist()]
    else:
        raise ValueError(f"unsupported logit refinement objective: {objective_mode}")

    source_input = torch.tensor([source_token_ids], dtype=torch.long)
    source_out, source_prefill_ms = _run_prefill(source_model, source_input, source_device)
    source_object = cache_to_object(source_out.past_key_values)
    del source_out

    source_config = source_model.config
    target_config = target_model.config
    positions = torch.arange(len(source_token_ids), device=target_device)
    bridge_object = transform_with_state(
        source_object,
        positions,
        state,
        source_heads=int(source_config.num_key_value_heads),
        source_head_dim=int(source_config.head_dim),
        source_rope_theta=_rope_theta(source_config),
        target_heads=int(target_config.num_key_value_heads),
        target_head_dim=int(target_config.head_dim),
        target_rope_theta=_rope_theta(target_config),
        device=target_device,
    )
    bridge_cache = object_to_dynamic_cache(bridge_object, target_config)
    student_input = torch.tensor(
        [student_token_ids],
        dtype=torch.long,
        device=target_device,
    )
    student_out = target_model(
        input_ids=student_input,
        past_key_values=bridge_cache,
        use_cache=False,
    )
    student_logits = student_out.logits
    del bridge_cache, student_out
    return (
        source_object,
        bridge_object,
        student_logits,
        teacher_logits,
        labels,
        source_prefill_ms,
    )


def _measure_refinement_holdout_mode(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    objective_mode: str,
    generation_tokens: int,
    temperature: float,
    label_weight: float,
) -> dict[str, Any]:
    import torch

    objectives: list[float] = []
    distillations: list[float] = []
    label_losses: list[float] = []
    agreements: list[float] = []
    free_running_matches = 0
    free_running_tokens = 0
    free_running_task_passes = 0
    free_running_task_prompts = 0
    free_running_prompt_results: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            (
                source_object,
                bridge_object,
                student_logits,
                teacher_logits,
                labels,
                _,
            ) = _refinement_prompt_forward(
                sample,
                tokenizer=tokenizer,
                source_model=source_model,
                target_model=target_model,
                state=state,
                source_device=source_device,
                target_device=target_device,
                suffix_tokens=suffix_tokens,
                objective_mode=objective_mode,
                generation_tokens=generation_tokens,
            )
            objective, distillation, label_loss = logit_distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                label_weight=label_weight,
            )
            agreement = (
                student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
            ).float().mean()
            objectives.append(float(objective.item()))
            distillations.append(float(distillation.item()))
            label_losses.append(float(label_loss.item()))
            agreements.append(float(agreement.item()))
            if objective_mode == "native-generation":
                _, token_ids = render_to_token_bucket(
                    sample,
                    tokenizer,
                    suffix_tokens=suffix_tokens,
                )
                free_running = _bridge_free_running_metrics(
                    target_model,
                    tokenizer,
                    bridge_object,
                    seed_token=token_ids[-1],
                    teacher_tokens=labels,
                    target_device=target_device,
                    expected_answer=sample.expected_answer,
                )
                free_running_matches += int(free_running["greedy_matches"])
                free_running_tokens += int(free_running["greedy_tokens"])
                if free_running["task_passed"] is not None:
                    free_running_task_prompts += 1
                    free_running_task_passes += int(free_running["task_passed"] is True)
                free_running_prompt_results.append(
                    {
                        "prompt_id": sample.prompt_id,
                        "greedy_match_rate": int(free_running["greedy_matches"])
                        / int(free_running["greedy_tokens"]),
                        "task_passed": free_running["task_passed"],
                        "text": free_running["text"],
                    }
                )
            del source_object, bridge_object, student_logits, teacher_logits, labels
    return {
        "objective": sum(objectives) / len(objectives),
        "distillation_loss": sum(distillations) / len(distillations),
        "label_loss": sum(label_losses) / len(label_losses),
        "top1_agreement": sum(agreements) / len(agreements),
        "free_running_greedy_match_rate": (
            free_running_matches / free_running_tokens if free_running_tokens else None
        ),
        "free_running_task_score": (
            free_running_task_passes / free_running_task_prompts
            if free_running_task_prompts
            else None
        ),
        "free_running_task_prompts": free_running_task_prompts,
        "free_running_prompt_results": free_running_prompt_results,
    }


def _measure_refinement_holdout(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    objective_mode: str,
    generation_tokens: int,
    temperature: float,
    label_weight: float,
    prompt_tail_weight: float,
) -> dict[str, Any]:
    modes = (
        ("native-generation", "prompt-tail")
        if objective_mode == "mixed"
        else (objective_mode,)
    )
    measurements = {
        mode: _measure_refinement_holdout_mode(
            samples,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=state,
            source_device=source_device,
            target_device=target_device,
            suffix_tokens=suffix_tokens,
            objective_mode=mode,
            generation_tokens=generation_tokens,
            temperature=temperature,
            label_weight=label_weight,
        )
        for mode in modes
    }
    if objective_mode != "mixed":
        return measurements[objective_mode]

    native = measurements["native-generation"]
    prompt_tail = measurements["prompt-tail"]
    denominator = 1 + prompt_tail_weight
    combined = {
        metric: (native[metric] + prompt_tail_weight * prompt_tail[metric]) / denominator
        for metric in ("objective", "distillation_loss", "label_loss", "top1_agreement")
    }
    combined.update(
        {
            "free_running_greedy_match_rate": native["free_running_greedy_match_rate"],
            "free_running_task_score": native["free_running_task_score"],
            "free_running_task_prompts": native["free_running_task_prompts"],
            "free_running_prompt_results": native["free_running_prompt_results"],
            "objective_components": {
                "native-generation": {
                    metric: native[metric]
                    for metric in (
                        "objective",
                        "distillation_loss",
                        "label_loss",
                        "top1_agreement",
                    )
                },
                "prompt-tail": {
                    metric: prompt_tail[metric]
                    for metric in (
                        "objective",
                        "distillation_loss",
                        "label_loss",
                        "top1_agreement",
                    )
                },
            },
        }
    )
    return combined


def _refinement_objective_modes(objective_mode: str) -> tuple[str, ...]:
    if objective_mode == "mixed":
        return ("native-generation", "prompt-tail")
    return (objective_mode,)


def _combine_refinement_metric(
    passes: dict[str, dict[str, Any]],
    metric: str,
    *,
    objective_mode: str,
    prompt_tail_weight: float,
) -> Any:
    if objective_mode != "mixed":
        return passes[objective_mode][metric]
    return (
        passes["native-generation"][metric]
        + prompt_tail_weight * passes["prompt-tail"][metric]
    ) / (1 + prompt_tail_weight)


def _run_refinement_training_passes(
    sample: CachedKVPrompt,
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    objective_mode: str,
    generation_tokens: int,
    temperature: float,
    label_weight: float,
) -> dict[str, dict[str, Any]]:
    passes: dict[str, dict[str, Any]] = {}
    for mode in _refinement_objective_modes(objective_mode):
        (
            source_object,
            bridge_object,
            student_logits,
            teacher_logits,
            labels,
            source_prefill_ms,
        ) = _refinement_prompt_forward(
            sample,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=state,
            source_device=source_device,
            target_device=target_device,
            suffix_tokens=suffix_tokens,
            objective_mode=mode,
            generation_tokens=generation_tokens,
        )
        objective, distillation, label_loss = logit_distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=temperature,
            label_weight=label_weight,
        )
        top1_agreement = (
            student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
        ).float().mean()
        passes[mode] = {
            "source_object": source_object,
            "bridge_object": bridge_object,
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "labels": labels,
            "source_prefill_ms": source_prefill_ms,
            "objective": objective,
            "distillation_loss": distillation,
            "label_loss": label_loss,
            "top1_agreement": top1_agreement,
        }
    return passes


def refine_state_with_target_logits(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    objective_mode: str,
    generation_tokens: int,
    steps: int,
    max_prompts: int,
    learning_rate: float,
    temperature: float,
    label_weight: float,
    prompt_tail_weight: float,
    anchor_weight: float,
    parameter_group: str,
    kv_anchor_weight: float,
    kv_anchor_max_positions: int,
    holdout_prompts: int,
    early_stopping_patience: int,
    holdout_min_delta: float,
    holdout_max_top1_drop: float,
    max_grad_norm: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine the existing v4 map against frozen target-model continuation logits."""

    import torch

    if steps <= 0 or max_prompts <= 0:
        raise ValueError("logit refinement steps and prompt count must be positive")
    if objective_mode not in _LOGIT_REFINEMENT_OBJECTIVES:
        raise ValueError(f"unsupported logit refinement objective: {objective_mode}")
    if generation_tokens <= 0:
        raise ValueError("logit refinement generation tokens must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("logit refinement learning rate must be finite and positive")
    if prompt_tail_weight < 0 or not math.isfinite(prompt_tail_weight):
        raise ValueError("logit refinement prompt-tail weight must be finite and non-negative")
    if objective_mode == "mixed" and prompt_tail_weight == 0:
        raise ValueError("mixed logit refinement requires a positive prompt-tail weight")
    if anchor_weight < 0 or not math.isfinite(anchor_weight):
        raise ValueError("logit refinement anchor weight must be finite and non-negative")
    if parameter_group not in _LOGIT_REFINEMENT_PARAMETER_GROUPS:
        raise ValueError(f"unsupported logit refinement parameter group: {parameter_group}")
    if kv_anchor_weight < 0 or not math.isfinite(kv_anchor_weight):
        raise ValueError("logit refinement KV anchor weight must be finite and non-negative")
    if kv_anchor_max_positions <= 0:
        raise ValueError("logit refinement KV anchor positions must be positive")
    if holdout_prompts < 0 or early_stopping_patience < 0:
        raise ValueError("logit refinement holdout settings must be non-negative")
    if early_stopping_patience and not holdout_prompts:
        raise ValueError("logit refinement early stopping requires holdout prompts")
    if holdout_min_delta < 0 or not math.isfinite(holdout_min_delta):
        raise ValueError("logit refinement holdout minimum delta must be non-negative")
    if not 0 <= holdout_max_top1_drop <= 1 or not math.isfinite(holdout_max_top1_drop):
        raise ValueError("logit refinement holdout top-1 drop must be between zero and one")
    if max_grad_norm <= 0 or not math.isfinite(max_grad_norm):
        raise ValueError("logit refinement max grad norm must be finite and positive")
    selected = _bucket_balanced_samples(samples, max_prompts)
    if not selected:
        raise ValueError("logit refinement received no training prompts")
    selected_ids = {sample.prompt_id for sample in selected}
    holdout = _bucket_balanced_samples(
        tuple(sample for sample in samples if sample.prompt_id not in selected_ids),
        holdout_prompts,
    )
    if len(holdout) != holdout_prompts:
        raise ValueError("logit refinement received too few disjoint holdout prompts")

    target_model.requires_grad_(False)
    state_on_target = {name: tensor.to(target_device) for name, tensor in state.items()}
    reference_state = {name: tensor.detach() for name, tensor in state_on_target.items()}
    parameter_names = _LOGIT_REFINEMENT_PARAMETER_GROUPS[parameter_group]
    parameters: list[Any] = []
    anchors: dict[str, Any] = {}
    for name in parameter_names:
        anchor = state_on_target[name].float().detach().clone()
        parameter = torch.nn.Parameter(anchor.clone())
        state_on_target[name] = parameter
        reference_state[name] = anchor
        parameters.append(parameter)
        anchors[name] = anchor
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    source_config = source_model.config
    target_config = target_model.config
    records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    best_step: int | None = None
    best_holdout_metrics: dict[str, Any] | None = None
    best_parameters: dict[str, Any] | None = None
    stale_steps = 0
    stopped_reason: str | None = None

    if holdout:
        initial_holdout = _measure_refinement_holdout(
            holdout,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=reference_state,
            source_device=source_device,
            target_device=target_device,
            suffix_tokens=suffix_tokens,
            objective_mode=objective_mode,
            generation_tokens=generation_tokens,
            temperature=temperature,
            label_weight=label_weight,
            prompt_tail_weight=prompt_tail_weight,
        )
        holdout_records.append({"step": 0, **initial_holdout})
        best_step = 0
        best_holdout_metrics = initial_holdout
        best_parameters = {name: anchors[name].clone() for name in parameter_names}

    for step in range(steps):
        sample = selected[step % len(selected)]
        passes = _run_refinement_training_passes(
            sample,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=state_on_target,
            source_device=source_device,
            target_device=target_device,
            suffix_tokens=suffix_tokens,
            objective_mode=objective_mode,
            generation_tokens=generation_tokens,
            temperature=temperature,
            label_weight=label_weight,
        )
        objective = _combine_refinement_metric(
            passes,
            "objective",
            objective_mode=objective_mode,
            prompt_tail_weight=prompt_tail_weight,
        )
        distillation = _combine_refinement_metric(
            passes,
            "distillation_loss",
            objective_mode=objective_mode,
            prompt_tail_weight=prompt_tail_weight,
        )
        label_loss = _combine_refinement_metric(
            passes,
            "label_loss",
            objective_mode=objective_mode,
            prompt_tail_weight=prompt_tail_weight,
        )
        top1_agreement = _combine_refinement_metric(
            passes,
            "top1_agreement",
            objective_mode=objective_mode,
            prompt_tail_weight=prompt_tail_weight,
        )
        anchor_mode = "native-generation" if "native-generation" in passes else objective_mode
        source_object = passes[anchor_mode]["source_object"]
        bridge_object = passes[anchor_mode]["bridge_object"]
        source_prefill_ms = sum(float(item["source_prefill_ms"]) for item in passes.values())
        parameter_anchor_loss = _parameter_anchor_loss(
            state_on_target,
            anchors,
            parameter_names,
        )
        kv_anchor_loss = objective.new_zeros(())
        kv_anchor_relative_mse = objective.new_zeros(())
        kv_anchor_cosine_loss = objective.new_zeros(())
        if kv_anchor_weight:
            anchor_positions = _sample_positions(
                int(source_object.shape[2]),
                kv_anchor_max_positions,
            )
            with torch.no_grad():
                reference_object = transform_with_state(
                    source_object[:, :, anchor_positions.to(source_object.device), :],
                    anchor_positions.to(target_device),
                    reference_state,
                    source_heads=int(source_config.num_key_value_heads),
                    source_head_dim=int(source_config.head_dim),
                    source_rope_theta=_rope_theta(source_config),
                    target_heads=int(target_config.num_key_value_heads),
                    target_head_dim=int(target_config.head_dim),
                    target_rope_theta=_rope_theta(target_config),
                    device=target_device,
                )
            refined_object = bridge_object[
                :, :, anchor_positions.to(bridge_object.device), :
            ]
            (
                kv_anchor_loss,
                kv_anchor_relative_mse,
                kv_anchor_cosine_loss,
            ) = _kv_anchor_losses(refined_object, reference_object)
        loss = (
            objective
            + anchor_weight * parameter_anchor_loss
            + kv_anchor_weight * kv_anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        record = {
            "step": step + 1,
            "prompt_id": sample.prompt_id,
            "token_bucket": sample.token_bucket,
            "source_prefill_ms": source_prefill_ms,
            "loss": float(loss.detach().item()),
            "distillation_loss": float(distillation.detach().item()),
            "label_loss": float(label_loss.detach().item()),
            "anchor_loss": float(parameter_anchor_loss.detach().item()),
            "parameter_anchor_loss": float(parameter_anchor_loss.detach().item()),
            "kv_anchor_loss": float(kv_anchor_loss.detach().item()),
            "kv_anchor_relative_mse": float(kv_anchor_relative_mse.detach().item()),
            "kv_anchor_cosine_loss": float(kv_anchor_cosine_loss.detach().item()),
            "top1_agreement": float(top1_agreement.detach().item()),
            "gradient_norm": float(grad_norm.detach().item()),
            "objective_components": {
                mode: {
                    metric: float(pass_result[metric].detach().item())
                    for metric in (
                        "objective",
                        "distillation_loss",
                        "label_loss",
                        "top1_agreement",
                    )
                }
                for mode, pass_result in passes.items()
            },
        }
        del (
            source_object,
            bridge_object,
            objective,
            distillation,
            label_loss,
            top1_agreement,
            parameter_anchor_loss,
            kv_anchor_loss,
            kv_anchor_relative_mse,
            kv_anchor_cosine_loss,
            loss,
            passes,
        )
        if kv_anchor_weight:
            del reference_object, refined_object
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if holdout:
            current_holdout = _measure_refinement_holdout(
                holdout,
                tokenizer=tokenizer,
                source_model=source_model,
                target_model=target_model,
                state=state_on_target,
                source_device=source_device,
                target_device=target_device,
                suffix_tokens=suffix_tokens,
                objective_mode=objective_mode,
                generation_tokens=generation_tokens,
                temperature=temperature,
                label_weight=label_weight,
                prompt_tail_weight=prompt_tail_weight,
            )
            holdout_records.append({"step": step + 1, **current_holdout})
            record["holdout_objective"] = current_holdout["objective"]
            record["holdout_top1_agreement"] = current_holdout["top1_agreement"]
            record["holdout_free_running_greedy_match_rate"] = current_holdout[
                "free_running_greedy_match_rate"
            ]
            record["holdout_free_running_task_score"] = current_holdout[
                "free_running_task_score"
            ]
            collapsed = current_holdout["top1_agreement"] < (
                holdout_records[0]["top1_agreement"] - holdout_max_top1_drop
            )
            improved = (
                not collapsed
                and best_holdout_metrics is not None
                and _holdout_is_better(
                    current_holdout,
                    best_holdout_metrics,
                    min_delta=holdout_min_delta,
                )
            )
            if improved:
                best_step = step + 1
                best_holdout_metrics = current_holdout
                best_parameters = {
                    name: state_on_target[name].detach().clone() for name in parameter_names
                }
                stale_steps = 0
            else:
                stale_steps += 1
            if collapsed:
                stopped_reason = "holdout_top1_collapse"
            elif early_stopping_patience and stale_steps >= early_stopping_patience:
                stopped_reason = "holdout_objective_patience"
        records.append(record)
        if stopped_reason is not None:
            break

    if best_parameters is not None:
        with torch.no_grad():
            for name in parameter_names:
                state_on_target[name].copy_(best_parameters[name])

    refined = {
        name: tensor.detach().cpu() if name in parameter_names else tensor.cpu()
        for name, tensor in state_on_target.items()
    }
    return _quantize_state(refined), {
        "enabled": True,
        "steps": len(records),
        "requested_steps": steps,
        "max_prompts": max_prompts,
        "prompt_ids": [sample.prompt_id for sample in selected],
        "learning_rate": learning_rate,
        "objective": objective_mode,
        "generation_tokens": generation_tokens,
        "temperature": temperature,
        "label_weight": label_weight,
        "prompt_tail_weight": prompt_tail_weight,
        "anchor_weight": anchor_weight,
        "parameter_anchor_kind": "relative_mse",
        "parameter_group": parameter_group,
        "kv_anchor_weight": kv_anchor_weight,
        "kv_anchor_kind": "mean_relative_mse_and_cosine",
        "kv_anchor_max_positions": kv_anchor_max_positions,
        "max_grad_norm": max_grad_norm,
        "parameter_names": list(parameter_names),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "holdout_prompt_ids": [sample.prompt_id for sample in holdout],
        "holdout_records": holdout_records,
        "holdout_min_delta": holdout_min_delta,
        "holdout_max_top1_drop": holdout_max_top1_drop,
        "holdout_checkpoint_metric_order": [
            "free_running_task_score",
            "free_running_greedy_match_rate",
            "objective",
        ],
        "early_stopping_patience": early_stopping_patience,
        "best_step": best_step,
        "best_holdout_metrics": best_holdout_metrics,
        "stopped_reason": stopped_reason,
        "records": records,
    }


def _teacher_forced_metrics(
    target_model: Any,
    native_object: Any,
    bridge_object: Any,
    continuation: list[int],
    target_device: str,
) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as functional

    if len(continuation) < 2:
        raise ValueError("teacher-forced continuation is too short")
    native_cache = object_to_dynamic_cache(native_object.to(target_device), target_model.config)
    bridge_cache = object_to_dynamic_cache(bridge_object.to(target_device), target_model.config)
    input_ids = torch.tensor([continuation[:-1]], dtype=torch.long, device=target_device)
    labels = torch.tensor(continuation[1:], dtype=torch.long, device=target_device)
    with torch.inference_mode():
        native_out = target_model(input_ids=input_ids, past_key_values=native_cache, use_cache=True)
        bridge_out = target_model(input_ids=input_ids, past_key_values=bridge_cache, use_cache=True)
    native_logits = native_out.logits[0]
    bridge_logits = bridge_out.logits[0]
    native_loss = functional.cross_entropy(native_logits.float(), labels, reduction="sum")
    bridge_loss = functional.cross_entropy(bridge_logits.float(), labels, reduction="sum")
    matches = int((native_logits.argmax(dim=-1) == bridge_logits.argmax(dim=-1)).sum().item())
    return {
        "tokens": int(labels.numel()),
        "top1_matches": matches,
        "native_nll": float(native_loss.item()),
        "bridge_nll": float(bridge_loss.item()),
    }


def _greedy_metrics(
    target_model: Any,
    tokenizer: Any,
    native_object: Any,
    bridge_object: Any,
    seed_token: int,
    target_device: str,
    greedy_tokens: int,
    expected_answer: str | None,
) -> dict[str, Any]:
    import torch

    native_cache = object_to_dynamic_cache(native_object.to(target_device), target_model.config)
    bridge_cache = object_to_dynamic_cache(bridge_object.to(target_device), target_model.config)
    native_input = torch.tensor([[seed_token]], dtype=torch.long, device=target_device)
    bridge_input = native_input.clone()
    native_tokens: list[int] = []
    bridge_tokens: list[int] = []
    for _ in range(greedy_tokens):
        with torch.inference_mode():
            native_out = target_model(
                input_ids=native_input,
                past_key_values=native_cache,
                use_cache=True,
            )
            bridge_out = target_model(
                input_ids=bridge_input,
                past_key_values=bridge_cache,
                use_cache=True,
            )
        native_cache = native_out.past_key_values
        bridge_cache = bridge_out.past_key_values
        native_token = int(native_out.logits[:, -1].argmax(dim=-1).item())
        bridge_token = int(bridge_out.logits[:, -1].argmax(dim=-1).item())
        native_tokens.append(native_token)
        bridge_tokens.append(bridge_token)
        native_input = torch.tensor([[native_token]], dtype=torch.long, device=target_device)
        bridge_input = torch.tensor([[bridge_token]], dtype=torch.long, device=target_device)
    native_text = tokenizer.decode(native_tokens, skip_special_tokens=True)
    bridge_text = tokenizer.decode(bridge_tokens, skip_special_tokens=True)
    return {
        "tokens": greedy_tokens,
        "matches": sum(
            left == right for left, right in zip(native_tokens, bridge_tokens, strict=True)
        ),
        "native_text": native_text,
        "bridge_text": bridge_text,
        "native_task_passed": (
            None
            if expected_answer is None
            else contains_expected_final_answer(native_text, expected_answer)
        ),
        "bridge_task_passed": (
            None
            if expected_answer is None
            else contains_expected_final_answer(bridge_text, expected_answer)
        ),
    }


def evaluate_split(
    samples: tuple[CachedKVPrompt, ...],
    *,
    tokenizer: Any,
    source_model: Any,
    target_model: Any,
    state: dict[str, Any],
    source_device: str,
    target_device: str,
    suffix_tokens: int,
    greedy_tokens: int,
) -> dict[str, Any]:
    import torch

    source_config = source_model.config
    target_config = target_model.config
    state_on_target = {name: tensor.to(target_device) for name, tensor in state.items()}
    prompt_results: list[dict[str, Any]] = []
    total_teacher_tokens = 0
    total_top1_matches = 0
    native_nll = 0.0
    bridge_nll = 0.0
    greedy_total = 0
    greedy_matches = 0
    native_task_passes = 0
    bridge_task_passes = 0
    task_count = 0
    key_cosines: list[float] = []
    value_cosines: list[float] = []
    transform_times: list[float] = []
    target_prefill_times: list[float] = []
    for sample in samples:
        _, token_ids = render_to_token_bucket(
            sample,
            tokenizer,
            suffix_tokens=suffix_tokens,
        )
        full_length = len(token_ids)
        input_ids = torch.tensor([token_ids[:full_length]], dtype=torch.long)
        source_out, source_prefill_ms = _run_prefill(source_model, input_ids, source_device)
        target_out, target_prefill_ms = _run_prefill(target_model, input_ids, target_device)
        source_object = cache_to_object(source_out.past_key_values)
        target_object = cache_to_object(target_out.past_key_values)
        positions = torch.arange(full_length, device=target_device)
        _sync(target_device)
        transform_started = time.perf_counter()
        bridge_object = transform_with_state(
            source_object,
            positions,
            state_on_target,
            source_heads=int(source_config.num_key_value_heads),
            source_head_dim=int(source_config.head_dim),
            source_rope_theta=_rope_theta(source_config),
            target_heads=int(target_config.num_key_value_heads),
            target_head_dim=int(target_config.head_dim),
            target_rope_theta=_rope_theta(target_config),
            device=target_device,
        )
        _sync(target_device)
        transform_ms = (time.perf_counter() - transform_started) * 1000
        target_on_bridge_device = target_object.to(target_device)
        key_cosine = cosine_mean(bridge_object[0], target_on_bridge_device[0])
        value_cosine = cosine_mean(bridge_object[1], target_on_bridge_device[1])

        prefix = sample.token_bucket
        continuation = token_ids[prefix : prefix + suffix_tokens + 1]
        teacher = _teacher_forced_metrics(
            target_model,
            target_object[:, :, :prefix],
            bridge_object[:, :, :prefix],
            continuation,
            target_device,
        )
        greedy = _greedy_metrics(
            target_model,
            tokenizer,
            target_object[:, :, : full_length - 1],
            bridge_object[:, :, : full_length - 1],
            token_ids[full_length - 1],
            target_device,
            greedy_tokens,
            sample.expected_answer,
        )
        total_teacher_tokens += int(teacher["tokens"])
        total_top1_matches += int(teacher["top1_matches"])
        native_nll += float(teacher["native_nll"])
        bridge_nll += float(teacher["bridge_nll"])
        greedy_total += int(greedy["tokens"])
        greedy_matches += int(greedy["matches"])
        if sample.expected_answer is not None:
            task_count += 1
            native_task_passes += int(greedy["native_task_passed"] is True)
            bridge_task_passes += int(greedy["bridge_task_passed"] is True)
        key_cosines.append(key_cosine)
        value_cosines.append(value_cosine)
        transform_times.append(transform_ms)
        target_prefill_times.append(target_prefill_ms)
        prompt_results.append(
            {
                "prompt_id": sample.prompt_id,
                "category": sample.category,
                "token_bucket": sample.token_bucket,
                "rendered_tokens": full_length,
                "source_prefill_ms": source_prefill_ms,
                "target_prefill_ms": target_prefill_ms,
                "transform_ms": transform_ms,
                "key_cosine": key_cosine,
                "value_cosine": value_cosine,
                "next_token_top1_agreement": int(teacher["top1_matches"]) / int(teacher["tokens"]),
                "greedy_match_rate": int(greedy["matches"]) / int(greedy["tokens"]),
                "native_task_passed": greedy["native_task_passed"],
                "bridge_task_passed": greedy["bridge_task_passed"],
                "native_text": greedy["native_text"],
                "bridge_text": greedy["bridge_text"],
            }
        )
        del source_out, target_out, source_object, target_object, bridge_object
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not prompt_results or total_teacher_tokens <= 0 or greedy_total <= 0:
        raise ValueError("evaluation produced no quality evidence")
    native_ppl = math.exp(native_nll / total_teacher_tokens)
    bridge_ppl = math.exp(bridge_nll / total_teacher_tokens)
    native_task_rate = native_task_passes / task_count if task_count else 0.0
    bridge_task_rate = bridge_task_passes / task_count if task_count else 0.0
    return {
        "prompt_count": len(prompt_results),
        "evaluated_tokens": total_teacher_tokens,
        "token_buckets": sorted({sample.token_bucket for sample in samples}),
        "key_cosine": sum(key_cosines) / len(key_cosines),
        "value_cosine": sum(value_cosines) / len(value_cosines),
        "next_token_top1_agreement": total_top1_matches / total_teacher_tokens,
        "native_perplexity": native_ppl,
        "bridge_perplexity": bridge_ppl,
        "perplexity_drift_pct": abs(bridge_ppl - native_ppl) / native_ppl * 100,
        "greedy_continuation_match_rate": greedy_matches / greedy_total,
        "task_prompt_count": task_count,
        "native_task_score": native_task_rate,
        "bridge_task_score": bridge_task_rate,
        "task_score_drop_pct": max(0.0, native_task_rate - bridge_task_rate) * 100,
        "p95_transform_ms": _percentile(transform_times, 0.95),
        "p95_target_transformers_prefill_ms": _percentile(target_prefill_times, 0.95),
        "prompt_results": prompt_results,
    }


def _quality_evidence(
    metrics: dict[str, Any],
    *,
    test_hash: str,
    cost_evidence: dict[str, Any] | None,
) -> CachedKVQualityEvidence:
    cost = cost_evidence or {}
    return CachedKVQualityEvidence(
        evaluation_dataset_sha256=test_hash,
        held_out_prompts=int(metrics["prompt_count"]),
        evaluated_tokens=int(metrics["evaluated_tokens"]),
        token_buckets=tuple(int(item) for item in metrics["token_buckets"]),
        key_cosine=float(metrics["key_cosine"]),
        value_cosine=float(metrics["value_cosine"]),
        next_token_top1_agreement=float(metrics["next_token_top1_agreement"]),
        perplexity_drift_pct=float(metrics["perplexity_drift_pct"]),
        task_prompts=int(metrics["task_prompt_count"]),
        native_task_score=float(metrics["native_task_score"]),
        bridge_task_score=float(metrics["bridge_task_score"]),
        task_score_drop_pct=(
            float(metrics["task_score_drop_pct"])
            if int(metrics["task_prompt_count"]) > 0
            else 100.0
        ),
        greedy_continuation_match_rate=float(metrics["greedy_continuation_match_rate"]),
        cost_report_sha256=cost.get("cost_report_sha256"),
        cost_candidate_manifest_sha256=cost.get("cost_candidate_manifest_sha256"),
        p95_source_read_transform_put_ms=cost.get("p95_source_read_transform_put_ms"),
        p95_target_prefill_ms=cost.get("p95_target_prefill_ms"),
    )


def _provisional_manifest(
    *,
    direction: str,
    source_spec: Any,
    target_spec: Any,
    weights_name: str,
    rank: int,
    source_window: int,
    dataset: CachedKVPromptDataset,
    quality: CachedKVQualityEvidence,
) -> CachedKVBridgeManifest:
    return CachedKVBridgeManifest(
        bridge_id="pending",
        direction=direction,
        source=source_spec,
        target=target_spec,
        weights_uri=weights_name,
        weights_sha256="0" * 64,
        rank=rank,
        source_window=source_window,
        train_dataset_sha256=dataset.split_sha256("train"),
        validation_dataset_sha256=dataset.split_sha256("validation"),
        test_dataset_sha256=dataset.split_sha256("test"),
        quality=quality,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("8b_to_14b", "14b_to_8b"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output manifest JSON path.")
    parser.add_argument("--model-8b", default=DEFAULT_8B)
    parser.add_argument("--model-14b", default=DEFAULT_14B)
    parser.add_argument(
        "--model-identity-cache",
        type=Path,
        default=Path("artifacts/cached_kv/.model_identity_cache.json"),
        help="Stat-guarded digest cache for validation sweeps; finalization always rehashes.",
    )
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    parser.add_argument("--fit-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--source-window", type=int, default=3)
    parser.add_argument("--layer-plan", choices=("depth", "cka"), default="depth")
    parser.add_argument("--layer-alignment-prompts", type=int, default=32)
    parser.add_argument("--layer-alignment-samples-per-prompt", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1000.0)
    parser.add_argument("--nonlinear-ridge-lambda", type=float, default=1000.0)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--max-training-samples", type=int, default=2048)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--greedy-tokens", type=int, default=16)
    parser.add_argument("--logit-refinement-steps", type=int, default=0)
    parser.add_argument("--logit-refinement-prompts", type=int, default=16)
    parser.add_argument(
        "--logit-refinement-objective",
        choices=_LOGIT_REFINEMENT_OBJECTIVES,
        default="native-generation",
    )
    parser.add_argument("--logit-refinement-learning-rate", type=float, default=1e-5)
    parser.add_argument("--logit-refinement-temperature", type=float, default=1.0)
    parser.add_argument("--logit-refinement-label-weight", type=float, default=0.1)
    parser.add_argument("--logit-refinement-prompt-tail-weight", type=float, default=0.25)
    parser.add_argument("--logit-refinement-anchor-weight", type=float, default=0.1)
    parser.add_argument(
        "--logit-refinement-parameter-group",
        choices=tuple(_LOGIT_REFINEMENT_PARAMETER_GROUPS),
        default="bias-only",
    )
    parser.add_argument("--logit-refinement-kv-anchor-weight", type=float, default=1.0)
    parser.add_argument("--logit-refinement-kv-anchor-max-positions", type=int, default=32)
    parser.add_argument("--logit-refinement-holdout-prompts", type=int, default=16)
    parser.add_argument("--logit-refinement-early-stopping-patience", type=int, default=2)
    parser.add_argument("--logit-refinement-holdout-min-delta", type=float, default=1e-4)
    parser.add_argument("--logit-refinement-holdout-max-top1-drop", type=float, default=0.05)
    parser.add_argument("--logit-refinement-max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--paired-refinement-validation",
        action="store_true",
        help="Evaluate validation before and after refinement on the same fitted state.",
    )
    parser.add_argument(
        "--smoke-max-validation-prompts",
        type=int,
        default=0,
        help="Limit validation only for implementation smoke tests; zero evaluates all.",
    )
    parser.add_argument("--cost-report", type=Path)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Evaluate sealed test and emit manifest.",
    )
    parser.add_argument(
        "--emit-validation-candidate",
        action="store_true",
        help="Write unapproved weights for non-publishing runtime cost benchmarks.",
    )
    parser.add_argument("--require-approved", action="store_true")
    return parser


def main() -> int:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_parser().parse_args()
    if args.output.suffix != ".json":
        raise ValueError("--output must be a JSON manifest path")
    if args.layer_alignment_prompts <= 0 or args.layer_alignment_samples_per_prompt <= 0:
        raise ValueError("layer alignment prompt and sample counts must be positive")
    if args.logit_refinement_steps < 0:
        raise ValueError("--logit-refinement-steps must be non-negative")
    if args.paired_refinement_validation and not args.logit_refinement_steps:
        raise ValueError("--paired-refinement-validation requires logit refinement")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = CachedKVPromptDataset.load(args.dataset)
    if args.finalize and args.emit_validation_candidate:
        raise ValueError("--finalize and --emit-validation-candidate are mutually exclusive")
    if args.require_approved and not args.finalize:
        raise ValueError("--require-approved requires --finalize")
    if args.cost_report is not None and not args.finalize:
        raise ValueError("--cost-report is only consumed by --finalize")
    approval_errors = dataset.approval_errors()
    if args.finalize and approval_errors:
        raise ValueError("; ".join(approval_errors))
    source_path, target_path, source_id, target_id, source_size, target_size = _model_paths(args)
    tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)

    source_spec = model_spec_from_path(
        source_path,
        model_id=source_id,
        parameter_count_b=source_size,
        revision="local-content-addressed",
        identity_cache_path=args.model_identity_cache,
        refresh_identity=args.finalize,
    )
    target_spec = model_spec_from_path(
        target_path,
        model_id=target_id,
        parameter_count_b=target_size,
        revision="local-content-addressed",
        identity_cache_path=args.model_identity_cache,
        refresh_identity=args.finalize,
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        source_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": args.source_device},
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": args.target_device},
    ).eval()
    if args.layer_plan == "cka":
        source_alignment, target_alignment, alignment_positions, alignment_collection = (
            collect_layer_alignment_data(
                dataset.split("train"),
                tokenizer=tokenizer,
                source_model=source_model,
                target_model=target_model,
                source_device=args.source_device,
                target_device=args.target_device,
                max_prompts=args.layer_alignment_prompts,
                samples_per_prompt=args.layer_alignment_samples_per_prompt,
                suffix_tokens=args.suffix_tokens,
            )
        )
        source_layer_ids, source_layer_weights, layer_alignment = build_cka_source_layer_plan(
            source_alignment,
            target_alignment,
            alignment_positions,
            args.source_window,
            source_heads=source_spec.num_key_value_heads,
            source_head_dim=source_spec.head_dim,
            source_rope_theta=source_spec.rope_theta,
            target_heads=target_spec.num_key_value_heads,
            target_head_dim=target_spec.head_dim,
            target_rope_theta=target_spec.rope_theta,
            device=args.fit_device,
        )
        layer_alignment.update(alignment_collection)
        del source_alignment, target_alignment, alignment_positions
        gc.collect()
        torch.cuda.empty_cache()
    else:
        source_layer_ids, source_layer_weights = build_source_layer_plan(
            source_spec.num_layers,
            target_spec.num_layers,
            args.source_window,
        )
        layer_alignment = {"method": "normalized_depth"}
    features, key_residual, value_residual, collection = collect_training_data(
        dataset.split("train"),
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        source_device=args.source_device,
        target_device=args.target_device,
        source_layer_ids=source_layer_ids,
        source_layer_weights=source_layer_weights,
        samples_per_prompt=args.samples_per_prompt,
        max_samples=args.max_training_samples,
        suffix_tokens=args.suffix_tokens,
    )
    effective_rank = min(args.rank, int(features.shape[1]) - 1, int(key_residual.shape[-1]))
    # Reset immediately before the randomized SVD so model loading cannot perturb the fit.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    state = fit_low_rank_state(
        features,
        key_residual,
        value_residual,
        source_layer_ids,
        source_layer_weights,
        rank=effective_rank,
        ridge_lambda=args.ridge_lambda,
        nonlinear_ridge_lambda=args.nonlinear_ridge_lambda,
        device=args.fit_device,
    )
    state = _quantize_state(state)
    del features, key_residual, value_residual
    gc.collect()
    torch.cuda.empty_cache()

    validation_samples = dataset.split("validation")
    if args.smoke_max_validation_prompts > 0:
        validation_samples = validation_samples[: args.smoke_max_validation_prompts]
    pre_refinement_validation: dict[str, Any] | None = None
    if args.paired_refinement_validation:
        pre_refinement_validation = evaluate_split(
            validation_samples,
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=state,
            source_device=args.source_device,
            target_device=args.target_device,
            suffix_tokens=args.suffix_tokens,
            greedy_tokens=args.greedy_tokens,
        )

    logit_refinement: dict[str, Any] = {"enabled": False, "steps": 0}
    if args.logit_refinement_steps:
        state, logit_refinement = refine_state_with_target_logits(
            dataset.split("train"),
            tokenizer=tokenizer,
            source_model=source_model,
            target_model=target_model,
            state=state,
            source_device=args.source_device,
            target_device=args.target_device,
            suffix_tokens=args.suffix_tokens,
            objective_mode=args.logit_refinement_objective,
            generation_tokens=args.greedy_tokens,
            steps=args.logit_refinement_steps,
            max_prompts=args.logit_refinement_prompts,
            learning_rate=args.logit_refinement_learning_rate,
            temperature=args.logit_refinement_temperature,
            label_weight=args.logit_refinement_label_weight,
            prompt_tail_weight=args.logit_refinement_prompt_tail_weight,
            anchor_weight=args.logit_refinement_anchor_weight,
            parameter_group=args.logit_refinement_parameter_group,
            kv_anchor_weight=args.logit_refinement_kv_anchor_weight,
            kv_anchor_max_positions=args.logit_refinement_kv_anchor_max_positions,
            holdout_prompts=args.logit_refinement_holdout_prompts,
            early_stopping_patience=args.logit_refinement_early_stopping_patience,
            holdout_min_delta=args.logit_refinement_holdout_min_delta,
            holdout_max_top1_drop=args.logit_refinement_holdout_max_top1_drop,
            max_grad_norm=args.logit_refinement_max_grad_norm,
        )

    validation = evaluate_split(
        validation_samples,
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        state=state,
        source_device=args.source_device,
        target_device=args.target_device,
        suffix_tokens=args.suffix_tokens,
        greedy_tokens=args.greedy_tokens,
    )
    result_path = args.output.with_suffix(".results.json")
    candidate_summary = {
        "schema_version": "goldenexperience.cached_kv_training.v1",
        "direction": args.direction,
        "finalized": args.finalize,
        "dataset": str(args.dataset),
        "dataset_split_hashes": {
            split: dataset.split_sha256(split) for split in ("train", "validation", "test")
        },
        "source_model": asdict(source_spec),
        "target_model": asdict(target_spec),
        "seed": args.seed,
        "rank": effective_rank,
        "source_window": args.source_window,
        "layer_alignment": layer_alignment,
        "ridge_lambda": args.ridge_lambda,
        "nonlinear_ridge_lambda": args.nonlinear_ridge_lambda,
        "training_collection": collection,
        "logit_refinement": logit_refinement,
        "paired_refinement_validation": args.paired_refinement_validation,
        "validation": validation,
    }
    if pre_refinement_validation is not None:
        candidate_summary["pre_refinement_validation"] = pre_refinement_validation
        comparison_metrics = (
            "key_cosine",
            "value_cosine",
            "next_token_top1_agreement",
            "greedy_continuation_match_rate",
            "bridge_perplexity",
            "perplexity_drift_pct",
            "bridge_task_score",
            "task_score_drop_pct",
            "p95_transform_ms",
        )
        candidate_summary["refinement_validation_delta"] = {
            name: float(validation[name]) - float(pre_refinement_validation[name])
            for name in comparison_metrics
        }
    if not args.finalize and not args.emit_validation_candidate:
        _write_json(result_path, candidate_summary)
        print(json.dumps(candidate_summary, indent=2, sort_keys=True))
        return 0

    if args.emit_validation_candidate:
        quality = _quality_evidence(
            validation,
            test_hash=dataset.split_sha256("validation"),
            cost_evidence=None,
        )
        weights_path = args.output.with_suffix(".safetensors")
        provisional = _provisional_manifest(
            direction=args.direction,
            source_spec=source_spec,
            target_spec=target_spec,
            weights_name=weights_path.name,
            rank=effective_rank,
            source_window=args.source_window,
            dataset=dataset,
            quality=quality,
        )
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, weights_path, metadata=safetensors_metadata(provisional))
        manifest = replace(provisional, weights_sha256=sha256_file(weights_path))
        manifest = replace(manifest, bridge_id=artifact_id_for(manifest))
        manifest.save(args.output)
        candidate_summary.update(
            {
                "validation_candidate": True,
                "manifest_path": str(args.output),
                "weights_path": str(weights_path),
                "bridge_id": manifest.bridge_id,
                "automatic_reuse_approved": manifest.approved,
                "approval_errors": manifest.validate(),
            }
        )
        _write_json(result_path, candidate_summary)
        print(json.dumps(candidate_summary, indent=2, sort_keys=True))
        return 0

    test_metrics = evaluate_split(
        dataset.split("test"),
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        state=state,
        source_device=args.source_device,
        target_device=args.target_device,
        suffix_tokens=args.suffix_tokens,
        greedy_tokens=args.greedy_tokens,
    )
    metadata_quality = _quality_evidence(
        test_metrics,
        test_hash=dataset.split_sha256("test"),
        cost_evidence=None,
    )
    weights_path = args.output.with_suffix(".safetensors")
    metadata_manifest = _provisional_manifest(
        direction=args.direction,
        source_spec=source_spec,
        target_spec=target_spec,
        weights_name=weights_path.name,
        rank=effective_rank,
        source_window=args.source_window,
        dataset=dataset,
        quality=metadata_quality,
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, weights_path, metadata=safetensors_metadata(metadata_manifest))
    weights_sha256 = sha256_file(weights_path)
    cost_evidence = None
    if args.cost_report is not None:
        cost_evidence = load_cached_kv_cost_evidence(
            args.cost_report,
            direction=args.direction,
            weights_sha256=weights_sha256,
            source_model_weights_sha256=source_spec.weights_sha256,
            target_model_weights_sha256=target_spec.weights_sha256,
            validation_dataset_sha256=dataset.split_sha256("validation"),
        )
    quality = _quality_evidence(
        test_metrics,
        test_hash=dataset.split_sha256("test"),
        cost_evidence=cost_evidence,
    )
    provisional = _provisional_manifest(
        direction=args.direction,
        source_spec=source_spec,
        target_spec=target_spec,
        weights_name=weights_path.name,
        rank=effective_rank,
        source_window=args.source_window,
        dataset=dataset,
        quality=quality,
    )
    manifest = replace(provisional, weights_sha256=weights_sha256)
    manifest = replace(manifest, bridge_id=artifact_id_for(manifest))
    manifest.save(args.output)
    candidate_summary.update(
        {
            "test": test_metrics,
            "manifest_path": str(args.output),
            "weights_path": str(weights_path),
            "bridge_id": manifest.bridge_id,
            "automatic_reuse_approved": manifest.approved,
            "approval_errors": manifest.validate(),
        }
    )
    _write_json(result_path, candidate_summary)
    print(json.dumps(candidate_summary, indent=2, sort_keys=True))
    if args.require_approved and not manifest.approved:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
