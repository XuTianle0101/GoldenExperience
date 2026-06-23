"""Layerwise offload and retrieval example."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.cache_core import CacheQuery, DeviceTier
from goldenexperience.engine_adapter import ArchitectureSignature, MockModelAdapter
from goldenexperience.tiered_store import LayerwiseOffloadPlan, TieredKVStore


def main() -> None:
    signature = ArchitectureSignature(
        model_id="qwen2.5-7b",
        family="qwen",
        architecture="qwen2",
        num_layers=4,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    adapter = MockModelAdapter(signature)
    blocks = adapter.extract_kv(
        engine_state=None,
        token_ids=[1, 2, 3, 4],
        prefix_hash="shared-system-prompt",
        session_id="demo",
    )
    blocks_by_layer = {block.metadata.layer_id: [block] for block in blocks}

    store = TieredKVStore(
        capacities={
            DeviceTier.HBM: 4096,
            DeviceTier.CPU: 16_384,
            DeviceTier.NVME: 1024 * 1024,
        },
        nvme_path="artifacts/cache/layerwise_example_nvme",
    )

    for result in store.put_layers(blocks_by_layer, target_tier=DeviceTier.CPU):
        print(f"stored layer={result.layer_id} bytes={result.bytes_moved} tier={result.target_tier.value}")

    plan = LayerwiseOffloadPlan(
        query=CacheQuery(model_id=signature.model_id, prefix_hash="shared-system-prompt"),
        target_tier=DeviceTier.NVME,
    )
    for result in store.offload_layers(plan):
        print(f"offloaded layer={result.layer_id} ok={result.success} tier={result.target_tier.value}")

    for result in store.retrieve_layers(
        CacheQuery(model_id=signature.model_id, prefix_hash="shared-system-prompt"),
        target_tier=DeviceTier.HBM,
    ):
        injected = adapter.inject_kv(result.blocks)
        print(f"retrieved layer={result.layer_id} hit={result.hit} blocks={len(injected['blocks'])}")

    store.shutdown()


if __name__ == "__main__":
    main()
