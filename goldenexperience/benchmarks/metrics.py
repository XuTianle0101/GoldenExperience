"""Benchmark metrics and summaries."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass(slots=True)
class BenchmarkRecord:
    name: str
    latency_ms: float
    bytes_size: int = 0
    cache_hit: bool = False
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSummary:
    name: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_per_s: float
    bytes_total: int
    hit_rate: float


def summarize(name: str, records: list[BenchmarkRecord]) -> BenchmarkSummary:
    if not records:
        return BenchmarkSummary(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    latencies = sorted(record.latency_ms for record in records)
    total_latency_s = sum(latencies) / 1000.0
    return BenchmarkSummary(
        name=name,
        count=len(records),
        mean_ms=statistics.fmean(latencies),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        throughput_ops_per_s=len(records) / total_latency_s if total_latency_s > 0 else 0.0,
        bytes_total=sum(record.bytes_size for record in records),
        hit_rate=sum(1 for record in records if record.cache_hit) / len(records),
    )


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
