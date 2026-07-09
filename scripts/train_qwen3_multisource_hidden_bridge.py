#!/usr/bin/env python3
"""Train a multi-source Qwen3 8B -> 14B hidden bridge artifact."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from scripts.experiment_qwen3_hidden_bridge import (
    CALIBRATION_TEXTS,
    DEFAULT_LARGE_MODEL,
    DEFAULT_SMALL_MODEL,
    EVAL_TEXTS,
    PreKVHiddenCapture,
    build_calibration_manifest,
    build_projector_state,
    build_synthetic_calibration_texts,
    collect_states_from_ids,
    cosine_mean,
    load_prompt_file,
    make_hidden_projector,
    make_qwen3_key_projector,
    make_qwen3_rope_fn,
    make_qwen3_value_projector,
    mean,
    resolve_model_path,
    tokenize_text,
    QWEN3_14B,
    QWEN3_8B,
)
from goldenexperience.size_variant import MaterializedHiddenChunk, TargetKVRestorer


def source_window_layers(target_layer: int, source_layers: int, target_layers: int, window: int) -> tuple[int, ...]:
    if window <= 1:
        if target_layers == 1:
            return (0,)
        pos = target_layer * (source_layers - 1) / (target_layers - 1)
        return (round(pos),)
    if target_layers == 1:
        center = 0
    else:
        center = round(target_layer * (source_layers - 1) / (target_layers - 1))
    half = window // 2
    start = max(0, min(source_layers - window, center - half))
    end = min(source_layers, start + window)
    return tuple(range(start, end))


def fit_multisource_bridge(
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
    source_window: int,
) -> dict[int, dict[str, Any]]:
    source_plan = {
        entry.target_layer_id: source_window_layers(
            entry.target_layer_id,
            len(small_model.model.layers),
            len(large_model.model.layers),
            source_window,
        )
        for entry in manifest.layer_map.entries
    }
    layer_samples: dict[int, dict[str, Any]] = {
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
            source_hidden = torch.cat(
                [states["small_hidden"][source_layer].float().to("cpu") for source_layer in source_plan[layer_id]],
                dim=-1,
            )
            target_hidden = states["large_hidden"][layer_id].float().to("cpu")
            x = source_hidden.reshape(-1, source_hidden.shape[-1])
            y = target_hidden.reshape(-1, target_hidden.shape[-1])
            take = min(remaining, x.shape[0])
            sample_info["x"].append(x[:take].contiguous())
            sample_info["y"].append(y[:take].contiguous())
            sample_info["count"] = int(sample_info["count"]) + take
        del states
        gc.collect()
        torch.cuda.empty_cache()

    fitted: dict[int, dict[str, Any]] = {}
    for entry in manifest.layer_map.entries:
        layer_id = entry.target_layer_id
        x = torch.cat(layer_samples[layer_id]["x"], dim=0)
        y = torch.cat(layer_samples[layer_id]["y"], dim=0)
        state = build_projector_state(
            x,
            y,
            rank=rank,
            ridge_lambda=ridge_lambda,
            device="cuda:1",
        )
        state["source_layer_ids"] = list(source_plan[layer_id])
        state["method"] = "multisource_concat_low_rank"
        fitted[layer_id] = state
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
    return fitted


def evaluate_multisource_bridge(
    manifest,
    tokenizer,
    small_model,
    large_model,
    small_capture: PreKVHiddenCapture,
    large_capture: PreKVHiddenCapture,
    eval_texts: list[str],
    max_length: int,
    prefix_tokens: int,
    learned_states: dict[int, dict[str, Any]],
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
    projectors = {
        int(layer_id): make_hidden_projector(state, device="cuda:1")
        for layer_id, state in learned_states.items()
    }

    results = []
    for text in eval_texts:
        tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length + 4)
        full_ids = tokenized["input_ids"][0]
        if full_ids.shape[0] <= 4:
            continue
        prompt_len = min(prefix_tokens, full_ids.shape[0] - 2)
        prefix_ids = full_ids[:prompt_len].unsqueeze(0)
        next_input = full_ids[prompt_len : prompt_len + 1].unsqueeze(0).to("cuda:1")
        states = collect_states_from_ids(
            small_model,
            large_model,
            small_capture,
            large_capture,
            prefix_ids,
            torch.ones_like(prefix_ids),
        )

        reconstructed_cache = DynamicCache(config=large_model.config)
        hidden_cosines = []
        key_cosines = []
        value_cosines = []
        for entry in manifest.layer_map.entries:
            layer_id = entry.target_layer_id
            source_layer_ids = tuple(int(item) for item in learned_states[layer_id]["source_layer_ids"])
            projector_input = torch.cat(
                [states["small_hidden"][source_layer].to("cuda:1") for source_layer in source_layer_ids],
                dim=-1,
            )
            bridged = projectors[layer_id](projector_input)
            hidden_cosines.append(cosine_mean(bridged, states["large_hidden"][layer_id].to("cuda:1")))
            hidden_chunk = MaterializedHiddenChunk(
                layer_id=layer_id,
                hidden=bridged,
                token_start=0,
                token_end=bridged.shape[1],
                position_ids=tuple(range(bridged.shape[1])),
                dtype=str(bridged.dtype).replace("torch.", ""),
                capture_point="pre_kv_hidden",
                source_layer_ids=source_layer_ids,
                transform_id="multisource_concat_low_rank",
            )
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
        native_top1 = int(native_logits.argmax(dim=-1).item())
        bridge_top1 = int(bridge_logits.argmax(dim=-1).item())
        results.append(
            {
                "name": "learned_low_rank_hidden_bridge",
                "success": True,
                "prompt_len": int(prompt_len),
                "hidden_cosine_mean": mean(hidden_cosines),
                "key_cosine_mean": mean(key_cosines),
                "value_cosine_mean": mean(value_cosines),
                "decode_logit_cosine": cosine_mean(native_logits, bridge_logits),
                "decode_top1_match": native_top1 == bridge_top1,
                "native_top1_token_id": native_top1,
                "bridge_top1_token_id": bridge_top1,
            }
        )
        del states, reconstructed_cache, native_decode, bridge_decode
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "name": "learned_low_rank_hidden_bridge",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-model-path", default="/data/models/Qwen3-8B")
    parser.add_argument("--large-model-path", default="/data/models/Qwen3-14B")
    parser.add_argument("--prompt-file", default="configs/kv_baseline_prompts.json")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--max-samples-per-layer", type=int, default=4096)
    parser.add_argument("--source-window", type=int, default=3)
    parser.add_argument("--include-eval-in-calibration", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hidden_bridge/qwen3_hidden_bridge_multisource.json"))
    args = parser.parse_args()

    small_model_path = resolve_model_path(args.small_model_path, DEFAULT_SMALL_MODEL)
    large_model_path = resolve_model_path(args.large_model_path, DEFAULT_LARGE_MODEL)
    prompt_file_texts = load_prompt_file(args.prompt_file)
    calibration_texts = CALIBRATION_TEXTS + build_synthetic_calibration_texts() + prompt_file_texts
    if args.include_eval_in_calibration:
        calibration_texts += EVAL_TEXTS

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
        calibration_id="qwen3_8b_to_14b_multisource_hidden_bridge",
        prompts_count=len(calibration_texts),
        bridge_rank=args.rank,
    )

    learned_states = fit_multisource_bridge(
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
        source_window=args.source_window,
    )
    learned_metrics = evaluate_multisource_bridge(
        manifest=manifest,
        tokenizer=tokenizer,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        eval_texts=EVAL_TEXTS,
        max_length=args.max_length,
        prefix_tokens=args.prefix_tokens,
        learned_states=learned_states,
    )

    summary = {
        "small_model_path": small_model_path,
        "large_model_path": large_model_path,
        "method": "multisource_concat_low_rank",
        "source_window": args.source_window,
        "rank": args.rank,
        "ridge_lambda": args.ridge_lambda,
        "max_length": args.max_length,
        "prefix_tokens": args.prefix_tokens,
        "max_samples_per_layer": args.max_samples_per_layer,
        "calibration_prompts": len(calibration_texts),
        "evaluation_prompts": len(EVAL_TEXTS),
        "include_eval_in_calibration": args.include_eval_in_calibration,
        "learned_low_rank_hidden_bridge": learned_metrics,
        "learned_bridge_quality_gate": {
            "hidden_cosine_ge_0_90": learned_metrics["hidden_cosine_mean"] >= 0.90,
            "kv_cosine_ge_0_85": learned_metrics["key_cosine_mean"] >= 0.85 and learned_metrics["value_cosine_mean"] >= 0.85,
            "decode_logit_cosine_ge_0_90": learned_metrics["decode_logit_cosine_mean"] >= 0.90,
        },
        "learned_weights_path": str(args.output.with_suffix(".pt")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
