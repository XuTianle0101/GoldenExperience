#!/usr/bin/env python3
"""Train a prefix-specific Qwen3 8B -> 14B hidden bridge calibration artifact."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.experiment_qwen3_hidden_bridge import (
    DEFAULT_LARGE_MODEL,
    DEFAULT_SMALL_MODEL,
    EVAL_TEXTS,
    PreKVHiddenCapture,
    build_calibration_manifest,
    build_projector_state,
    collect_states_from_ids,
    evaluate_bridge,
    make_hidden_projector,
    mean,
    resolve_model_path,
    QWEN3_14B,
    QWEN3_8B,
)
from goldenexperience.runtime.cross_model_reuse import token_ids_from_prompt


def _load_prompt_ids(
    *,
    tokenizer_path: str,
    prompt_file: Path,
    prompt_id: str,
    max_tokens: int,
) -> torch.Tensor:
    ids = token_ids_from_prompt(
        tokenizer_path=tokenizer_path,
        prompt_file=prompt_file,
        prompt_id=prompt_id,
    )
    if max_tokens > 0:
        ids = ids[:max_tokens]
    return torch.tensor([ids], dtype=torch.long)


def _eval_text_ids(tokenizer, max_length: int) -> list[torch.Tensor]:
    sequences = []
    for text in EVAL_TEXTS:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        sequences.append(encoded["input_ids"])
    return sequences


def _blend_source(states: dict[str, Any], entry) -> torch.Tensor:
    blended = None
    for source_layer_id, weight in zip(entry.source_layer_ids, entry.weights, strict=True):
        source_hidden = states["small_hidden"][source_layer_id].float().to("cpu")
        blended = source_hidden * weight if blended is None else blended + source_hidden * weight
    assert blended is not None
    return blended


def fit_prefix_bridge(
    manifest,
    small_model,
    large_model,
    small_capture: PreKVHiddenCapture,
    large_capture: PreKVHiddenCapture,
    sequences: list[torch.Tensor],
    rank: int,
    ridge_lambda: float,
) -> dict[int, dict[str, torch.Tensor]]:
    layer_samples: dict[int, dict[str, list[torch.Tensor]]] = {
        entry.target_layer_id: {"x": [], "y": []}
        for entry in manifest.layer_map.entries
    }
    for input_ids in sequences:
        states = collect_states_from_ids(
            small_model,
            large_model,
            small_capture,
            large_capture,
            input_ids,
            torch.ones_like(input_ids),
        )
        for entry in manifest.layer_map.entries:
            source_hidden = _blend_source(states, entry)
            target_hidden = states["large_hidden"][entry.target_layer_id].float().to("cpu")
            layer_samples[entry.target_layer_id]["x"].append(
                source_hidden.reshape(-1, source_hidden.shape[-1]).contiguous()
            )
            layer_samples[entry.target_layer_id]["y"].append(
                target_hidden.reshape(-1, target_hidden.shape[-1]).contiguous()
            )
        del states
        gc.collect()
        torch.cuda.empty_cache()

    fitted: dict[int, dict[str, torch.Tensor]] = {}
    for entry in manifest.layer_map.entries:
        x = torch.cat(layer_samples[entry.target_layer_id]["x"], dim=0)
        y = torch.cat(layer_samples[entry.target_layer_id]["y"], dim=0)
        fitted[entry.target_layer_id] = build_projector_state(
            x,
            y,
            rank=rank,
            ridge_lambda=ridge_lambda,
            device="cuda:1",
        )
        fitted[entry.target_layer_id]["method"] = "prefix_specific_low_rank"
        del x, y
        gc.collect()
        torch.cuda.empty_cache()
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-model-path", default="/data/models/Qwen3-8B")
    parser.add_argument("--large-model-path", default="/data/models/Qwen3-14B")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-id", default="kv_disk_long_prefix")
    parser.add_argument("--runtime-prefix-tokens", type=int, default=1776)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hidden_bridge/qwen3_hidden_bridge_prefix_specific.json"))
    args = parser.parse_args()

    small_model_path = resolve_model_path(args.small_model_path, DEFAULT_SMALL_MODEL)
    large_model_path = resolve_model_path(args.large_model_path, DEFAULT_LARGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(large_model_path, trust_remote_code=True)
    sequences = _eval_text_ids(tokenizer, args.max_length)
    runtime_ids = _load_prompt_ids(
        tokenizer_path=large_model_path,
        prompt_file=args.prompt_file,
        prompt_id=args.prompt_id,
        max_tokens=args.runtime_prefix_tokens,
    )
    sequences.append(runtime_ids)
    total_tokens = sum(int(seq.shape[1]) for seq in sequences)

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
        calibration_id="qwen3_8b_to_14b_prefix_specific_bridge",
        prompts_count=len(sequences),
        bridge_rank=args.rank,
    )
    learned_states = fit_prefix_bridge(
        manifest=manifest,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        sequences=sequences,
        rank=args.rank,
        ridge_lambda=args.ridge_lambda,
    )
    learned_projectors = {
        layer_id: make_hidden_projector(state, device="cuda:1")
        for layer_id, state in learned_states.items()
    }
    learned_metrics = evaluate_bridge(
        name="learned_low_rank_hidden_bridge",
        manifest=manifest,
        tokenizer=tokenizer,
        small_model=small_model,
        large_model=large_model,
        small_capture=small_capture,
        large_capture=large_capture,
        eval_texts=EVAL_TEXTS,
        max_length=args.max_length,
        prefix_tokens=args.prefix_tokens,
        layer_projectors=learned_projectors,
    )
    summary = {
        "small_model_path": small_model_path,
        "large_model_path": large_model_path,
        "method": "prefix_specific_low_rank",
        "prompt_file": str(args.prompt_file),
        "prompt_id": args.prompt_id,
        "runtime_prefix_tokens": args.runtime_prefix_tokens,
        "calibration_sequences": len(sequences),
        "calibration_tokens": total_tokens,
        "rank": args.rank,
        "ridge_lambda": args.ridge_lambda,
        "max_length": args.max_length,
        "prefix_tokens": args.prefix_tokens,
        "learned_low_rank_hidden_bridge": learned_metrics,
        "learned_bridge_quality_gate": {
            "hidden_cosine_ge_0_90": learned_metrics["hidden_cosine_mean"] >= 0.90,
            "kv_cosine_ge_0_85": learned_metrics["key_cosine_mean"] >= 0.85 and learned_metrics["value_cosine_mean"] >= 0.85,
            "decode_logit_cosine_ge_0_90": learned_metrics["decode_logit_cosine_mean"] >= 0.90,
        },
        "learned_weights_path": str(args.output.with_suffix(".pt")),
        "notes": [
            "This is a prefix-specific calibration artifact; it is valid only for prefixes represented in the calibration sequences.",
        ],
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
