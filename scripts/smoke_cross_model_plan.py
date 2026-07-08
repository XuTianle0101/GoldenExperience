#!/usr/bin/env python3
"""Smoke-test GoldenExperience cross-model reuse planning."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest
from goldenexperience.runtime import RuntimeConfig, check_runtime
from goldenexperience.size_variant import build_calibration_manifest


def make_model(
    model_id: str,
    size_b: float,
    layers: int,
    head_dim: int,
    family: str = "qwen",
    architecture: str = "qwen3",
    tokenizer_id: str = "qwen3",
) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family=family,
        architecture=architecture,
        tokenizer_id=tokenizer_id,
        parameter_count_b=size_b,
        kv_shape=KVShape(
            num_layers=layers,
            hidden_size=4096 if size_b <= 8 else 5120,
            num_attention_heads=32 if size_b <= 8 else 40,
            num_key_value_heads=8,
            head_dim=head_dim,
            dtype="bfloat16" if family == "qwen" else "float16",
            rope_theta=1_000_000.0 if family == "qwen" else None,
            model_config_hash=f"{model_id}-hash",
            tokenizer_hash=f"{tokenizer_id}-hash",
        ),
    )


def build_plans() -> list[dict[str, object]]:
    planner = CrossModelReusePlanner()
    base = make_model("qwen3-8b", size_b=8, layers=36, head_dim=128)
    lora = ModelRef(
        model_id="qwen3-8b-lora-math",
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=8,
        base_model_id="qwen3-8b",
        lora_adapter_id="math-adapter",
        kv_shape=base.kv_shape,
    )
    large = make_model("qwen3-14b", size_b=14, layers=40, head_dim=128)
    llama = make_model(
        "llama-3.1-8b",
        size_b=8,
        layers=32,
        head_dim=128,
        family="llama",
        architecture="llama3",
        tokenizer_id="llama-3.1",
    )

    with tempfile.TemporaryDirectory(prefix="ge-smoke-") as temp_dir:
        manifest = build_calibration_manifest(base, large, calibration_id="qwen3_8b_to_14b_hidden_bridge_v0")
        artifact_path = Path(temp_dir) / "qwen3_8b_to_14b_hidden_bridge_v0.json"
        manifest.save(artifact_path)
        requests = [
            ReuseRequest(source=base, target=lora, prefix_hash="shared-system-prompt"),
            ReuseRequest(
                source=base,
                target=large,
                prefix_hash="shared-system-prompt",
                calibration_id=manifest.calibration_id,
                artifact_uri=str(artifact_path),
                estimated_target_prefill_ms=100.0,
                estimated_materialization_ms=30.0,
            ),
            ReuseRequest(source=base, target=llama, prefix_hash="shared-system-prompt"),
        ]
        plans = [planner.plan(request) for request in requests]
    return [
        {
            "source": plan.request.source.model_id,
            "target": plan.request.target.model_id,
            "scenario": plan.scenario.value,
            "strategy": plan.strategy.value,
            "status": plan.status.value,
            "confidence": plan.confidence,
            "executable": plan.executable,
            "transform_id": plan.transform_id,
            "direction": plan.direction,
            "pair_id": plan.pair_id,
            "layer_map_id": plan.layer_map_id,
            "projection_id": plan.projection_id,
            "hidden_bridge_id": plan.hidden_bridge_id,
            "restore_id": plan.restore_id,
            "state_kind": plan.state_kind,
            "estimated_prefill_saved_ms": plan.estimated_prefill_saved_ms,
            "estimated_materialization_ms": plan.estimated_materialization_ms,
            "fallback_reason": plan.fallback_reason,
            "required_gates": list(plan.required_gates),
            "notes": list(plan.notes),
        }
        for plan in plans
    ]


def runtime_imports() -> dict[str, object]:
    status = check_runtime(RuntimeConfig(model_id="smoke"))
    return {
        "ready": status.ready,
        "available": status.available,
        "goldenexperience": importlib.util.find_spec("goldenexperience") is not None,
        "missing_hints": list(status.missing_hints),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Also report vLLM/LMCache/Mooncake runtime availability.",
    )
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Exit non-zero when any runtime dependency is missing.",
    )
    args = parser.parse_args()
    if args.strict_runtime:
        args.check_runtime = True

    payload: dict[str, object] = {"plans": build_plans()}
    if args.check_runtime:
        payload["runtime"] = runtime_imports()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.strict_runtime and not payload["runtime"]["ready"]:  # type: ignore[index]
            return 1
        return 0

    for plan in payload["plans"]:  # type: ignore[index]
        line = (
            "{scenario}: {source} -> {target} | {strategy} | {status} | "
            "confidence={confidence} | direction={direction}"
        )
        print(line.format(**plan))
    if args.check_runtime:
        runtime = payload["runtime"]  # type: ignore[index]
        print("runtime ready:", runtime["ready"])
        print("runtime imports:", runtime["available"])
        for hint in runtime["missing_hints"]:
            print("missing:", hint)
        if args.strict_runtime and not runtime["ready"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
