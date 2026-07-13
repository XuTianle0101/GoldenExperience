"""Publication-grade latency accounting for selective direct injection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import SPLIT_COUNTS
from goldenexperience.size_variant.selective_manifest import (
    SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
    SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
    SELECTIVE_RUNTIME_REQUEST_ORDER,
    RuntimeCostEvidence,
)

SELECTIVE_RUNTIME_REPORT_SCHEMA = "goldenexperience.selective_runtime_report.v2"


@dataclass(frozen=True)
class LatencyPercentiles:
    p50_ms: float
    p95_ms: float
    p99_ms: float


def latency_percentiles(samples_ms: Sequence[float]) -> LatencyPercentiles:
    if not samples_ms:
        raise ValueError("latency samples are empty")
    samples = sorted(float(value) for value in samples_ms)
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")

    def percentile(quantile: float) -> float:
        index = min(len(samples) - 1, max(0, math.ceil(quantile * len(samples)) - 1))
        return samples[index]

    return LatencyPercentiles(percentile(0.50), percentile(0.95), percentile(0.99))


def build_selective_runtime_report(
    *,
    direction: str,
    runtime_audit_dataset_sha256: str,
    audit_requests: int,
    warmup_iterations: int,
    materialization_ms: Sequence[float],
    native_prefill_ms: Sequence[float],
    accepted_native_ttft_ms: Sequence[float],
    accepted_reuse_ttft_ms: Sequence[float],
    rejected_native_ttft_ms: Sequence[float],
    rejected_fallback_ttft_ms: Sequence[float],
    accepted_target_mooncake_puts: int,
    backing_files_remaining: int,
) -> dict[str, Any]:
    series = {
        "materialization": materialization_ms,
        "native_prefill": native_prefill_ms,
        "accepted_native_ttft": accepted_native_ttft_ms,
        "accepted_reuse_ttft": accepted_reuse_ttft_ms,
        "rejected_native_ttft": rejected_native_ttft_ms,
        "rejected_fallback_ttft": rejected_fallback_ttft_ms,
    }
    if type(warmup_iterations) is not int or warmup_iterations < 20:
        raise ValueError("runtime report requires at least 20 warmup iterations")
    if type(audit_requests) is not int or audit_requests != SPLIT_COUNTS["runtime_audit"]:
        raise ValueError("runtime report requires exactly 512 audit requests")
    if (
        type(accepted_target_mooncake_puts) is not int
        or accepted_target_mooncake_puts < 0
        or type(backing_files_remaining) is not int
        or backing_files_remaining < 0
    ):
        raise ValueError("runtime storage counters must be non-negative integers")
    if any(len(values) < 100 for values in series.values()):
        raise ValueError("every runtime configuration requires at least 100 measurements")
    accepted_count = len(accepted_native_ttft_ms)
    rejected_count = len(rejected_native_ttft_ms)
    if (
        len(
            {
                len(materialization_ms),
                len(native_prefill_ms),
                accepted_count,
                len(accepted_reuse_ttft_ms),
            }
        )
        != 1
    ):
        raise ValueError("accepted runtime series must contain paired measurements")
    if rejected_count != len(rejected_fallback_ttft_ms):
        raise ValueError("rejected runtime series must contain paired measurements")
    if accepted_count + rejected_count != audit_requests:
        raise ValueError("accepted and rejected runtime measurements must cover the audit split")
    percentiles = {name: latency_percentiles(values) for name, values in series.items()}
    materialization_p95 = percentiles["materialization"].p95_ms
    prefill_p95 = percentiles["native_prefill"].p95_ms
    accepted_native_p95 = percentiles["accepted_native_ttft"].p95_ms
    accepted_reuse_p95 = percentiles["accepted_reuse_ttft"].p95_ms
    rejected_native_p95 = percentiles["rejected_native_ttft"].p95_ms
    rejected_fallback_p95 = percentiles["rejected_fallback_ttft"].p95_ms
    if prefill_p95 <= 0 or accepted_native_p95 <= 0 or rejected_native_p95 <= 0:
        raise ValueError("native P95 latency must be positive")
    rejected_overhead = max(
        0.0,
        (rejected_fallback_p95 - rejected_native_p95) / rejected_native_p95 * 100,
    )
    return {
        "schema_version": SELECTIVE_RUNTIME_REPORT_SCHEMA,
        "direction": direction,
        "runtime_audit_dataset_sha256": runtime_audit_dataset_sha256,
        "audit_requests": audit_requests,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": min(len(values) for values in series.values()),
        "accepted_requests": accepted_count,
        "rejected_requests": rejected_count,
        "measurement_protocol": SELECTIVE_RUNTIME_MEASUREMENT_PROTOCOL,
        "request_order": SELECTIVE_RUNTIME_REQUEST_ORDER,
        "arrival_timestamps_replayed": SELECTIVE_RUNTIME_ARRIVAL_TIMESTAMPS_REPLAYED,
        "percentiles_ms": {name: asdict(value) for name, value in percentiles.items()},
        "p95_materialization_to_prefill_ratio": materialization_p95 / prefill_p95,
        "accepted_p95_ttft_reduction_pct": (
            (accepted_native_p95 - accepted_reuse_p95) / accepted_native_p95 * 100
        ),
        "rejected_p95_fallback_overhead_pct": rejected_overhead,
        "accepted_target_mooncake_puts": accepted_target_mooncake_puts,
        "backing_files_remaining": backing_files_remaining,
        "eligible_for_approval": bool(
            materialization_p95 / prefill_p95 <= 0.70
            and (accepted_native_p95 - accepted_reuse_p95) / accepted_native_p95 >= 0.30
            and rejected_overhead <= 5.0
            and accepted_target_mooncake_puts == 0
            and backing_files_remaining == 0
        ),
    }


def write_runtime_report(path: str | Path, report: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def runtime_cost_evidence_from_report(
    report: dict[str, Any],
    *,
    report_sha256: str,
) -> RuntimeCostEvidence:
    if report.get("schema_version") != SELECTIVE_RUNTIME_REPORT_SCHEMA:
        raise ValueError("unsupported selective runtime report schema")
    if report.get("eligible_for_approval") is not True:
        raise ValueError("runtime report is not eligible for approval")
    percentiles = report["percentiles_ms"]
    evidence = RuntimeCostEvidence(
        report_sha256=report_sha256,
        runtime_audit_dataset_sha256=report["runtime_audit_dataset_sha256"],
        audit_requests=report["audit_requests"],
        warmup_iterations=report["warmup_iterations"],
        measured_iterations=report["measured_iterations"],
        p95_materialization_ms=percentiles["materialization"]["p95_ms"],
        p95_native_prefill_ms=percentiles["native_prefill"]["p95_ms"],
        p95_materialization_to_prefill_ratio=report["p95_materialization_to_prefill_ratio"],
        accepted_p95_ttft_reduction_pct=report["accepted_p95_ttft_reduction_pct"],
        rejected_p95_fallback_overhead_pct=report["rejected_p95_fallback_overhead_pct"],
        measurement_protocol=str(report["measurement_protocol"]),
        request_order=str(report["request_order"]),
        arrival_timestamps_replayed=report["arrival_timestamps_replayed"],
    )
    errors = evidence.validate(evidence.runtime_audit_dataset_sha256)
    if errors:
        raise ValueError("; ".join(errors))
    return evidence
