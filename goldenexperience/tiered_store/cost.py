"""Tier-aware cost model for placement and benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass

from goldenexperience.cache_core.enums import DeviceTier


@dataclass(slots=True)
class TierState:
    tier: DeviceTier
    capacity_bytes: int
    used_bytes: int
    bandwidth_gbps: float
    latency_us: float

    @property
    def free_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes)

    @property
    def utilization(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes


@dataclass(slots=True)
class TierCostModel:
    """Simple analytical model used by policy decisions and benchmark logs."""

    hbm_bandwidth_gbps: float = 1500.0
    cpu_bandwidth_gbps: float = 64.0
    nvme_bandwidth_gbps: float = 7.0
    hbm_latency_us: float = 1.0
    cpu_latency_us: float = 10.0
    nvme_latency_us: float = 80.0

    def transfer_time_ms(self, bytes_size: int, source: DeviceTier, target: DeviceTier) -> float:
        if source == target:
            return 0.0
        bandwidth = min(self.bandwidth_gbps(source), self.bandwidth_gbps(target))
        if bandwidth <= 0:
            return float("inf")
        return (bytes_size / (bandwidth * 1_000_000_000.0)) * 1000.0 + self.latency_us(target) / 1000.0

    def bandwidth_gbps(self, tier: DeviceTier) -> float:
        return {
            DeviceTier.HBM: self.hbm_bandwidth_gbps,
            DeviceTier.CPU: self.cpu_bandwidth_gbps,
            DeviceTier.NVME: self.nvme_bandwidth_gbps,
        }[tier]

    def latency_us(self, tier: DeviceTier) -> float:
        return {
            DeviceTier.HBM: self.hbm_latency_us,
            DeviceTier.CPU: self.cpu_latency_us,
            DeviceTier.NVME: self.nvme_latency_us,
        }[tier]

