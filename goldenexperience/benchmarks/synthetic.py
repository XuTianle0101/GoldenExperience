"""Synthetic benchmark harness for tiered cache artifact checks."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from goldenexperience.cache_core import CacheBlock, CacheQuery, DeviceTier, KVPayload
from goldenexperience.tiered_store import PrefetchPlan, TieredKVStore
from goldenexperience.benchmarks.metrics import BenchmarkRecord, summarize


def run_synthetic_benchmark(
    blocks: int,
    tokens_per_block: int,
    head_dim: int,
    hbm_capacity_bytes: int,
    cpu_capacity_bytes: int,
    nvme_path: str | Path | None = None,
) -> dict[str, object]:
    records: list[BenchmarkRecord] = []
    cache_dir = nvme_path or tempfile.mkdtemp(prefix="goldenexperience-nvme-")
    store = TieredKVStore(
        capacities={
            DeviceTier.HBM: hbm_capacity_bytes,
            DeviceTier.CPU: cpu_capacity_bytes,
            DeviceTier.NVME: 8 * 1024**3,
        },
        nvme_path=cache_dir,
    )

    block_ids: list[str] = []
    for idx in range(blocks):
        block = _make_block(idx, tokens_per_block, head_dim)
        start = time.perf_counter()
        store.put(block)
        latency_ms = (time.perf_counter() - start) * 1000.0
        block_ids.append(block.metadata.block_id)
        records.append(BenchmarkRecord("put", latency_ms, block.metadata.bytes_size))

    for block_id in block_ids:
        start = time.perf_counter()
        block = store.get_by_id(block_id, promote_to=DeviceTier.HBM)
        latency_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            BenchmarkRecord(
                "get_promote",
                latency_ms,
                block.metadata.bytes_size if block is not None else 0,
                cache_hit=block is not None,
            )
        )

    prefetch_ids = block_ids[: min(8, len(block_ids))]
    start = time.perf_counter()
    futures = store.prefetch(PrefetchPlan(prefetch_ids, target_tier=DeviceTier.HBM, asynchronous=True))
    for future in futures:
        future.result()
    records.append(BenchmarkRecord("prefetch_batch", (time.perf_counter() - start) * 1000.0))

    result = {
        "summaries": {
            name: asdict(summarize(name, [record for record in records if record.name == name]))
            for name in sorted({record.name for record in records})
        },
        "tier_states": {
            tier.value: asdict(state)
            for tier, state in store.tier_states().items()
        },
    }
    store.shutdown()
    return result


def _make_block(idx: int, tokens_per_block: int, head_dim: int) -> CacheBlock:
    row = [float((idx + offset) % 17) for offset in range(head_dim)]
    payload = KVPayload(
        key=[[row[:] for _ in range(tokens_per_block)]],
        value=[[[value * 0.5 for value in row] for _ in range(tokens_per_block)]],
    )
    return CacheBlock.from_payload(
        payload=payload,
        model_id="synthetic-qwen-family",
        layer_id=idx % 4,
        head_id=0,
        token_range=(idx * tokens_per_block, (idx + 1) * tokens_per_block),
        dtype="float32",
        device_tier=DeviceTier.HBM,
        prefix_hash=f"prefix-{idx // 4}",
        session_id="synthetic",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GoldenExperience synthetic cache benchmark.")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hbm-capacity-mb", type=int, default=16)
    parser.add_argument("--cpu-capacity-mb", type=int, default=128)
    parser.add_argument("--nvme-path", type=str, default=None)
    args = parser.parse_args()
    result = run_synthetic_benchmark(
        blocks=args.blocks,
        tokens_per_block=args.tokens_per_block,
        head_dim=args.head_dim,
        hbm_capacity_bytes=args.hbm_capacity_mb * 1024 * 1024,
        cpu_capacity_bytes=args.cpu_capacity_mb * 1024 * 1024,
        nvme_path=args.nvme_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

