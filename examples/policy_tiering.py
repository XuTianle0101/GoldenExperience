"""Policy-driven eviction, offload, and prefetch example."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload
from goldenexperience.tiered_store import (
    CostAwareEvictionPolicy,
    DecodeWindowPrefetchPolicy,
    PrefetchContext,
    TieredKVStore,
    WatermarkOffloadPolicy,
)


def make_block(layer_id: int, tier: DeviceTier = DeviceTier.HBM) -> CacheBlock:
    values = [float(layer_id + offset) for offset in range(16)]
    return CacheBlock.from_payload(
        payload=KVPayload(key=[values], value=[values]),
        model_id="qwen2.5-7b",
        layer_id=layer_id,
        head_id=0,
        token_range=(0, 16),
        dtype="float32",
        device_tier=tier,
        prefix_hash="shared-system-prompt",
        session_id="demo",
    )


def main() -> None:
    store = TieredKVStore(
        capacities={
            DeviceTier.HBM: 1_000,
            DeviceTier.CPU: 10_000,
            DeviceTier.NVME: 100_000,
        },
        nvme_path="artifacts/cache/policy_example_nvme",
        eviction_policy=CostAwareEvictionPolicy(),
        offload_policy=WatermarkOffloadPolicy(high_watermark=0.50, low_watermark=0.25),
        prefetch_policy=DecodeWindowPrefetchPolicy(),
    )

    for layer_id in range(5):
        store.put(make_block(layer_id))

    offload_results = store.enforce_offload_policy()
    print(f"policy offloaded {sum(result.success for result in offload_results)} blocks")

    plan = store.build_prefetch_plan(
        PrefetchContext(
            query=CacheQuery(model_id="qwen2.5-7b", prefix_hash="shared-system-prompt"),
            target_tier=DeviceTier.HBM,
            current_layer_id=1,
            lookahead_layers=2,
            asynchronous=False,
        )
    )
    print(f"prefetch plan selected {len(plan.block_ids)} blocks")
    print(store.prefetch(plan))
    store.shutdown()


if __name__ == "__main__":
    main()

