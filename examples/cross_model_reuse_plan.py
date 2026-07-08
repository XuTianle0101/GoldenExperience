"""Plan cross-model KV reuse without launching vLLM or LMCache MP."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.lmcache_patch import CrossModelCacheKey, PatchManifest
from goldenexperience.reuse import CrossModelReusePlanner, KVShape, ModelRef, ReuseRequest
from goldenexperience.size_variant import build_calibration_manifest


def model(model_id: str, size_b: float, layers: int, head_dim: int) -> ModelRef:
    return ModelRef(
        model_id=model_id,
        family="qwen",
        architecture="qwen3",
        tokenizer_id="qwen3",
        parameter_count_b=size_b,
        kv_shape=KVShape(
            num_layers=layers,
            hidden_size=4096 if size_b <= 8 else 5120,
            num_attention_heads=32 if size_b <= 8 else 40,
            num_key_value_heads=8,
            head_dim=head_dim,
            dtype="bfloat16",
            rope_theta=1_000_000.0,
            model_config_hash=f"{model_id}-hash",
            tokenizer_hash="qwen3-tokenizer-hash",
        ),
    )


def main() -> None:
    planner = CrossModelReusePlanner()
    base = model("qwen3-8b", size_b=8, layers=36, head_dim=128)
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
    large = model("qwen3-14b", size_b=14, layers=40, head_dim=128)

    with tempfile.TemporaryDirectory(prefix="ge-example-") as temp_dir:
        manifest = build_calibration_manifest(base, large, calibration_id="qwen3_8b_to_14b_hidden_bridge_v0")
        artifact_path = Path(temp_dir) / "qwen3_8b_to_14b_hidden_bridge_v0.json"
        manifest.save(artifact_path)
        for target in (lora, large):
            plan = planner.plan(
                ReuseRequest(
                    source=base,
                    target=target,
                    prefix_hash="shared-system-prompt",
                    calibration_id=manifest.calibration_id if target is large else None,
                    artifact_uri=str(artifact_path) if target is large else None,
                )
            )
            print(plan.scenario.value, plan.strategy.value, plan.status.value, plan.confidence)
            print(CrossModelCacheKey.from_plan(plan).to_sidecar_fields())

    print(PatchManifest.default().as_markdown())


if __name__ == "__main__":
    main()
