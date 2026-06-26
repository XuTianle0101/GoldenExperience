#!/usr/bin/env python3
"""Smoke-test GoldenExperience cross-model reuse planning."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest
from goldenexperience.sglang_runtime import RuntimeConfig, check_runtime


def make_model(
    model_id: str,
    size_b: float,
    layers: int,
    head_dim: int,
    family: str = "qwen",
    architecture: str = "qwen2",
    tokenizer_id: str = "qwen2.5",
) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family=family,
        architecture=architecture,
        tokenizer_id=tokenizer_id,
        parameter_count_b=size_b,
        kv_shape=KVShape(num_layers=layers, num_key_value_heads=8, head_dim=head_dim),
    )


def build_plans() -> list[dict[str, object]]:
    planner = CrossModelReusePlanner()
    base = make_model("qwen2.5-7b", size_b=7, layers=32, head_dim=128)
    lora = ModelRef(
        model_id="qwen2.5-7b-lora-math",
        family="qwen",
        architecture="qwen2",
        tokenizer_id="qwen2.5",
        parameter_count_b=7,
        base_model_id="qwen2.5-7b",
        lora_adapter_id="math-adapter",
        kv_shape=base.kv_shape,
    )
    large = make_model("qwen2.5-14b", size_b=14, layers=48, head_dim=128)
    llama = make_model(
        "llama-3.1-8b",
        size_b=8,
        layers=32,
        head_dim=128,
        family="llama",
        architecture="llama3",
        tokenizer_id="llama-3.1",
    )

    requests = [
        ReuseRequest(source=base, target=lora, prefix_hash="shared-system-prompt"),
        ReuseRequest(
            source=base,
            target=large,
            prefix_hash="shared-system-prompt",
            calibration_id="qwen25_projection_v0",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--check-runtime", action="store_true", help="Also report SGLang/LMCache imports.")
    args = parser.parse_args()

    payload: dict[str, object] = {"plans": build_plans()}
    if args.check_runtime:
        payload["runtime"] = runtime_imports()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for plan in payload["plans"]:  # type: ignore[index]
        print(
            "{scenario}: {source} -> {target} | {strategy} | {status} | confidence={confidence}".format(
                **plan
            )
        )
    if args.check_runtime:
        runtime = payload["runtime"]  # type: ignore[index]
        print("runtime ready:", runtime["ready"])
        print("runtime imports:", runtime["available"])
        for hint in runtime["missing_hints"]:
            print("missing:", hint)


if __name__ == "__main__":
    main()
