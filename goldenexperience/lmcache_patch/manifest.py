"""Patch manifest describing how GoldenExperience layers on top of LMCache."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PatchHook:
    """One narrow hook that can be implemented as an LMCache patch."""

    name: str
    target: str
    purpose: str
    order: int
    required: bool = True


@dataclass(frozen=True)
class PatchManifest:
    """A declarative contract for the LMCache patch surface.

    GoldenExperience is intentionally a control-plane layer. The manifest makes that
    boundary explicit so future implementation work patches lookup/materialization paths
    without replacing serving-engine inference or LMCache storage/offload internals.
    """

    runtime_engine: str = "vLLM"
    cache_backend: str = "LMCache"
    hooks: tuple[PatchHook, ...] = field(default_factory=tuple)
    invariants: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> "PatchManifest":
        return cls(
            hooks=(
                PatchHook(
                    name="serving_request_metadata",
                    target="Serving request/session metadata before LMCache lookup",
                    purpose="Attach source/target ModelRef, prefix hash, and experiment flags.",
                    order=10,
                ),
                PatchHook(
                    name="lmcache_cross_model_lookup",
                    target="LMCache lookup miss path or secondary index lookup",
                    purpose="Ask the GoldenExperience planner whether a compatible source model entry exists.",
                    order=20,
                ),
                PatchHook(
                    name="goldenexperience_materializer",
                    target="LMCache retrieve path before KV is handed back to the serving engine",
                    purpose="Alias, project, or translate retrieved KV according to a ReusePlan.",
                    order=30,
                ),
                PatchHook(
                    name="quality_gate_accounting",
                    target="LMCache store/metrics metadata",
                    purpose="Record confidence, fallback reason, and calibration provenance.",
                    order=40,
                ),
            ),
            invariants=(
                "Do not modify serving-engine scheduling, attention kernels, or token generation semantics.",
                "Do not replace LMCache storage, offload, eviction, or prefetch implementations.",
                "If a ReusePlan is not ready, fall back to the original serving-engine plus LMCache path.",
                "All cross-model reuse must carry scenario, transform_id, confidence, and calibration metadata.",
            ),
            notes=(
                "The patch should be small enough to carry as a delta on top of upstream LMCache.",
                "Source installs of vLLM and LMCache are supported for development and debugging.",
                "SGLang remains supported through explicit legacy control scripts.",
            ),
        )

    def ordered_hooks(self) -> tuple[PatchHook, ...]:
        return tuple(sorted(self.hooks, key=lambda hook: hook.order))

    def planned_patch_points(self, lmcache_root: str | Path | None = None) -> list[str]:
        """Return human-readable patch points without assuming upstream file names."""

        root = Path(lmcache_root) if lmcache_root is not None else Path("<LMCache>")
        return [
            str(root / "lookup path: add secondary cross-model lookup"),
            str(root / "key builder: include GoldenExperience metadata sidecar"),
            str(root / "retrieve path: call GoldenExperience materializer before return"),
            str(root / "metrics path: emit reuse scenario and fallback reason"),
        ]

    def as_markdown(self) -> str:
        lines = [
            f"# GoldenExperience Patch Manifest: {self.runtime_engine} + {self.cache_backend}",
            "",
            "## Hooks",
        ]
        for hook in self.ordered_hooks():
            required = "required" if hook.required else "optional"
            lines.append(f"- `{hook.name}` ({required}): {hook.target} -- {hook.purpose}")
        lines.extend(["", "## Invariants"])
        lines.extend(f"- {item}" for item in self.invariants)
        if self.notes:
            lines.extend(["", "## Notes"])
            lines.extend(f"- {item}" for item in self.notes)
        return "\n".join(lines) + "\n"
