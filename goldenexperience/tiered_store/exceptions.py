"""Tiered store exceptions."""

from __future__ import annotations

from goldenexperience.cache_core.enums import DeviceTier


class TieredStoreError(RuntimeError):
    """Base error for tiered KV store failures."""


class CapacityExceededError(TieredStoreError):
    """Raised when a tier cannot admit a block after policy-driven demotion."""

    def __init__(self, tier: DeviceTier, required_bytes: int, free_bytes: int) -> None:
        self.tier = tier
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes
        super().__init__(
            f"Tier {tier.value} cannot admit {required_bytes} bytes; "
            f"only {free_bytes} bytes are free."
        )
