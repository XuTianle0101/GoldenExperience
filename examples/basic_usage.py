"""Minimal GoldenExperience planning example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest


def main() -> None:
    shape = KVShape(num_layers=36, num_key_value_heads=8, head_dim=128)
    base = ModelRef(
        model_id="qwen3-8b",
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=8,
        kv_shape=shape,
    )
    lora = ModelRef(
        model_id="qwen3-8b-lora-math",
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=8,
        base_model_id="qwen3-8b",
        lora_adapter_id="math-adapter",
        kv_shape=shape,
    )

    plan = CrossModelReusePlanner().plan(
        ReuseRequest(source=base, target=lora, prefix_hash="shared-system-prompt")
    )
    print(plan.as_metadata())


if __name__ == "__main__":
    main()
