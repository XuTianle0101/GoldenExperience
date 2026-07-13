from pathlib import Path

from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload
from goldenexperience.tiered_store import (
    CapacityExceededError,
    CostAwareEvictionPolicy,
    DecodeWindowPrefetchPolicy,
    LayerwiseOffloadPlan,
    LFUEvictionPolicy,
    PrefetchContext,
    PrefetchPlan,
    TieredKVStore,
    WatermarkOffloadPolicy,
)


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

    recovered = store.get(
        CacheQuery(model_id="llama-family", prefix_hash="shared"), promote_to=DeviceTier.HBM
    )
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
    assert store.evict(CacheQuery(model_id="llama-family"), required_bytes=1) == [
        block.metadata.block_id
    ]
    store.shutdown()


def test_put_layers_retrieve_layers_and_layer_groups(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
    )
    layers = {
        0: [make_block(0)],
        1: [make_block(1)],
        2: [make_block(2)],
    }

    put_results = list(store.put_layers(layers, target_tier=DeviceTier.CPU))

    assert [result.layer_id for result in put_results] == [0, 1, 2]
    assert all(result.success for result in put_results)
    assert store.layer_ids(CacheQuery(model_id="llama-family")) == [0, 1, 2]
    groups = store.layer_groups(CacheQuery(model_id="llama-family"))
    assert groups
    assert groups[0].layer_ids == [0, 1, 2]

    retrieved = list(
        store.retrieve_layers(
            CacheQuery(model_id="llama-family", prefix_hash="shared"),
            target_tier=DeviceTier.HBM,
        )
    )

    assert [result.layer_id for result in retrieved] == [0, 1, 2]
    assert all(result.hit for result in retrieved)
    assert all(
        block.metadata.device_tier == DeviceTier.HBM
        for result in retrieved
        for block in result.blocks
    )
    store.shutdown()


def test_layerwise_offload_moves_selected_layers(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
    )
    for layer_id in range(4):
        store.put(make_block(layer_id))

    results = store.offload_layers(
        LayerwiseOffloadPlan(
            query=CacheQuery(model_id="llama-family", prefix_hash="shared"),
            target_tier=DeviceTier.NVME,
            layer_ids=[1, 2],
        )
    )

    assert [result.layer_id for result in results] == [1, 2]
    assert all(result.success for result in results)
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 0)[0].metadata.device_tier
        == DeviceTier.HBM
    )
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 1)[0].metadata.device_tier
        == DeviceTier.NVME
    )
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 2)[0].metadata.device_tier
        == DeviceTier.NVME
    )
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 3)[0].metadata.device_tier
        == DeviceTier.HBM
    )
    store.shutdown()


def test_layerwise_offload_async_respects_pipeline_depth(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
        max_workers=2,
    )
    for layer_id in range(3):
        store.put(make_block(layer_id))

    futures = store.offload_layers(
        LayerwiseOffloadPlan(
            query=CacheQuery(model_id="llama-family"),
            target_tier=DeviceTier.CPU,
            asynchronous=True,
            pipeline_depth=2,
        )
    )
    results = [future.result() for future in futures]

    assert [result.layer_id for result in results] == [0, 1, 2]
    assert all(result.success for result in results)
    assert all(
        block.metadata.device_tier == DeviceTier.CPU
        for layer_id in range(3)
        for block in store.get_layer(CacheQuery(model_id="llama-family"), layer_id)
    )
    store.shutdown()


def test_capacity_error_when_pinned_blocks_prevent_demotion(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 300, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
    )
    first = make_block(0)
    second = make_block(1)
    store.put(first)
    store.pin(first.metadata.block_id)

    try:
        store.put(second)
        raised = False
    except CapacityExceededError:
        raised = True

    assert raised
    assert store.get_by_id(first.metadata.block_id) is not None
    store.shutdown()


def test_watermark_offload_policy_demotes_cold_blocks(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 1_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
        offload_policy=WatermarkOffloadPolicy(high_watermark=0.50, low_watermark=0.25),
    )
    for layer_id in range(5):
        store.put(make_block(layer_id))

    results = store.enforce_offload_policy()

    assert results
    assert any(result.success and result.target_tier == DeviceTier.CPU for result in results)
    assert store.tier_states()[DeviceTier.HBM].used_bytes <= 500
    store.shutdown()


def test_decode_window_prefetch_policy_selects_upcoming_layers(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
        prefetch_policy=DecodeWindowPrefetchPolicy(),
    )
    for layer_id in range(5):
        block = make_block(layer_id)
        block.metadata.device_tier = DeviceTier.CPU
        store.put(block)

    plan = store.build_prefetch_plan(
        PrefetchContext(
            query=CacheQuery(model_id="llama-family", prefix_hash="shared"),
            target_tier=DeviceTier.HBM,
            current_layer_id=1,
            lookahead_layers=2,
            max_blocks=10,
            asynchronous=False,
        )
    )

    planned_layers = [
        store.index.get(block_id).layer_id
        for block_id in plan.block_ids
        if store.index.get(block_id) is not None
    ]
    assert planned_layers == [2, 3]
    assert store.prefetch(plan) == [True, True]
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 2)[0].metadata.device_tier
        == DeviceTier.HBM
    )
    assert (
        store.get_layer(CacheQuery(model_id="llama-family"), 3)[0].metadata.device_tier
        == DeviceTier.HBM
    )
    store.shutdown()


def test_lfu_eviction_policy_removes_least_accessed_block(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
        eviction_policy=LFUEvictionPolicy(),
    )
    cold = make_block(0)
    hot = make_block(1)
    store.put(cold)
    store.put(hot)
    store.get_by_id(hot.metadata.block_id)
    store.get_by_id(hot.metadata.block_id)

    victims = store.evict(
        CacheQuery(model_id="llama-family"), required_bytes=cold.metadata.bytes_size
    )

    assert victims == [cold.metadata.block_id]
    assert store.get_by_id(cold.metadata.block_id) is None
    assert store.get_by_id(hot.metadata.block_id) is not None
    store.shutdown()


def test_cost_aware_eviction_policy_keeps_high_quality_hot_block(tmp_path: Path) -> None:
    store = TieredKVStore(
        capacities={DeviceTier.HBM: 10_000, DeviceTier.CPU: 10_000, DeviceTier.NVME: 10_000},
        nvme_path=tmp_path,
        eviction_policy=CostAwareEvictionPolicy(),
    )
    low_quality = make_block(0)
    low_quality.metadata.quality_score = 0.10
    high_quality = make_block(1)
    high_quality.metadata.quality_score = 0.99
    store.put(low_quality)
    store.put(high_quality)
    store.get_by_id(high_quality.metadata.block_id)

    victims = store.evict(
        CacheQuery(model_id="llama-family"), required_bytes=low_quality.metadata.bytes_size
    )

    assert victims == [low_quality.metadata.block_id]
    assert store.get_by_id(low_quality.metadata.block_id) is None
    assert store.get_by_id(high_quality.metadata.block_id) is not None
    store.shutdown()
