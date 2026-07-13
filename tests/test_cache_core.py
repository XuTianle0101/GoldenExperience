from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload


def test_cache_block_metadata_and_query_match() -> None:
    payload = KVPayload(key=[[[1.0, 2.0]]], value=[[[3.0, 4.0]]])
    block = CacheBlock.from_payload(
        payload=payload,
        model_id="qwen3-8b",
        layer_id=1,
        head_id=0,
        token_range=(10, 11),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        prefix_hash="prefix",
        session_id="session",
    )

    assert block.metadata.shape == ((1, 1, 2), (1, 1, 2))
    assert block.metadata.bytes_size == 32
    assert len(block.metadata.checksum) == 64
    assert CacheQuery(model_id="qwen3-8b", token_range=(10, 20)).matches(block.metadata)
    assert not CacheQuery(model_id="qwen3-14b").matches(block.metadata)
