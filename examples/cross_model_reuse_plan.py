"""Plan cross-model KV reuse without launching vLLM or LMCache MP."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.lmcache_patch import CrossModelCacheKey, PatchManifest
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest


def model(model_id: str, size_b: float, layers: int, head_dim: int) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family="qwen",
        architecture="qwen2",
        tokenizer_id="qwen2.5",
        parameter_count_b=size_b,
        kv_shape=KVShape(num_layers=layers, num_key_value_heads=8, head_dim=head_dim),
    )


def main() -> None:
    planner = CrossModelReusePlanner()
    base = model("qwen2.5-7b", size_b=7, layers=32, head_dim=128)
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
    large = model("qwen2.5-14b", size_b=14, layers=48, head_dim=128)

    for target in (lora, large):
        plan = planner.plan(
            ReuseRequest(
                source=base,
                target=target,
                prefix_hash="shared-system-prompt",
                calibration_id="qwen25_projection_v0" if target is large else None,
            )
        )
        print(plan.scenario.value, plan.strategy.value, plan.status.value, plan.confidence)
        print(CrossModelCacheKey.from_plan(plan).to_sidecar_fields())

    print(PatchManifest.default().as_markdown())


if __name__ == "__main__":
    main()
