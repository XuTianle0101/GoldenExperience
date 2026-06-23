"""Minimal GoldenExperience usage example."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload
from goldenexperience.cross_model_mapper import LinearProjectionKVMapper, ReusePolicy
from goldenexperience.engine_adapter import ArchitectureSignature
from goldenexperience.tiered_store import TieredKVStore


def main() -> None:
    store = TieredKVStore(
        capacities={
            DeviceTier.HBM: 1024,
            DeviceTier.CPU: 4096,
            DeviceTier.NVME: 1024 * 1024,
        },
        nvme_path="artifacts/cache/example_nvme",
    )
    source_sig = ArchitectureSignature(
        model_id="qwen2.5-7b",
        family="qwen",
        architecture="qwen2",
        num_layers=2,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
    )
    target_sig = ArchitectureSignature(
        model_id="qwen2.5-14b",
        family="qwen",
        architecture="qwen2",
        num_layers=2,
        hidden_size=6,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=3,
    )

    source_block = CacheBlock.from_payload(
        payload=KVPayload(key=[[[1.0, 2.0]]], value=[[[0.5, 1.5]]]),
        model_id=source_sig.model_id,
        layer_id=0,
        head_id=0,
        token_range=(0, 1),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        prefix_hash="shared-system-prompt",
        session_id="demo",
    )
    store.put(source_block)

    mapper = LinearProjectionKVMapper().fit(source_sig, target_sig)
    mapped = mapper.transform(source_block)
    decision = ReusePolicy().decide(
        source_metadata=source_block.metadata,
        source_signature=source_sig,
        target_signature=target_sig,
        mapper_confidence=mapped.confidence,
        prefix_similarity=0.99,
        expected_latency_savings_ms=12.5,
    )

    store.put(mapped.block)
    recovered = store.get(CacheQuery(model_id=target_sig.model_id, prefix_hash="shared-system-prompt"))
    print(decision)
    print(recovered.metadata if recovered else None)
    store.shutdown()


if __name__ == "__main__":
    main()
