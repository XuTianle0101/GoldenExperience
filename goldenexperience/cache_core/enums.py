"""Shared enums for cache placement and reuse decisions."""

from __future__ import annotations

from enum import Enum


class DeviceTier(str, Enum):
    """Logical tiers used by the engine-decoupled cache store."""

    HBM = "hbm"
    CPU = "cpu"
    NVME = "nvme"


class ReuseAction(str, Enum):
    """Actions that a reuse policy can choose."""

    DIRECT_REUSE = "direct_reuse"
    PARTIAL_REUSE = "partial_reuse"
    WARM_START_RECOMPUTE = "warm_start_recompute"
    FALLBACK_RECOMPUTE = "fallback_recompute"

