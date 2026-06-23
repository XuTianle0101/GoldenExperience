from pathlib import Path

from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload
from goldenexperience.tiered_store import PrefetchPlan, TieredKVStore


def make_block(idx: int) -> CacheBlock:
    values = [float(idx + offset) for offset in range(16)]
    return CacheBlock.from_payload(
        payload=KVPayload(key=[values], value=[values]),
        model_id="llama-family",
        layer_id=idx,
        head_id=0,
        token_range=(idx, idx + 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        prefix_hash="shared",
        session_id="s",
    )


def test_tiered_store_demotes_and_promotes(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 300, DeviceTier.CPU: 600, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
    )
    first = make_block(0)
    second = make_block(1)
    store.put(first)
    store.put(second)

    states = store.tier_states()
    assert states[DeviceTier.HBM].used_bytes <= 300 or states[DeviceTier.CPU].used_bytes > 0

    recovered = store.get(CacheQuery(model_id="llama-family", prefix_hash="shared"), promote_to=DeviceTier.HBM)
    assert recovered is not None
    assert recovered.metadata.model_id == "llama-family"

    futures = store.prefetch(PrefetchPlan([first.metadata.block_id], target_tier=DeviceTier.HBM))
    assert all(future.result() for future in futures)
    store.shutdown()


def test_pin_release_prevents_regular_evict(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
    )
    block = make_block(0)
    store.put(block)
    assert store.pin(block.metadata.block_id)
    assert store.evict(CacheQuery(model_id="llama-family"), required_bytes=1) == []
    assert store.release(block.metadata.block_id)
    assert store.evict(CacheQuery(model_id="llama-family"), required_bytes=1) == [block.metadata.block_id]
    store.shutdown()

