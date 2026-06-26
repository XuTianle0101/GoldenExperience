"""Runtime dependency helpers for the SGLang + LMCache stack."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeDependency:
    """A runtime package that GoldenExperience expects to be installed externally."""

    name: str
    import_name: str
    repo_url: str
    install_hint: str


@dataclass(frozen=True)
class RuntimeConfig:
    """Local development configuration for the open-source runtime stack."""

    model_id: str
    lmcache_config_path: str | None = None
    sglang_repo_path: str | None = None
    lmcache_repo_path: str | None = None
    enable_patch: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[RuntimeDependency, RuntimeDependency]:
        return (
            RuntimeDependency(
                name="SGLang",
                import_name="sglang",
                repo_url="https://github.com/sgl-project/sglang.git",
                install_hint="Install SGLang from upstream or a local clone before launching inference.",
            ),
            RuntimeDependency(
                name="LMCache",
                import_name="lmcache",
                repo_url="https://github.com/LMCache/LMCache.git",
                install_hint="Install LMCache from upstream or a local clone before enabling cache reuse.",
            ),
        )


@dataclass(frozen=True)
class RuntimeStatus:
    """Import and path checks for the required runtime projects."""

    available: dict[str, bool]
    repo_paths: dict[str, str | None]
    missing_hints: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return all(self.available.values())


def check_runtime(config: RuntimeConfig) -> RuntimeStatus:
    available: dict[str, bool] = {}
    repo_paths = {
        "SGLang": config.sglang_repo_path,
        "LMCache": config.lmcache_repo_path,
    }
    missing_hints: list[str] = []
    for dependency in config.dependencies:
        found = importlib.util.find_spec(dependency.import_name) is not None
        available[dependency.name] = found
        if not found:
            missing_hints.append(f"{dependency.name}: {dependency.install_hint} ({dependency.repo_url})")
    return RuntimeStatus(
        available=available,
        repo_paths=repo_paths,
        missing_hints=tuple(missing_hints),
    )


def build_patch_environment(config: RuntimeConfig, manifest_path: str | Path | None = None) -> dict[str, str]:
    """Build GoldenExperience-only env vars for wrapper scripts.

    These variables are intentionally namespaced. SGLang and LMCache remain responsible
    for their own documented launch flags and cache/offload configuration.
    """

    env = dict(config.extra_env)
    env["GE_ENABLE_CROSS_MODEL_REUSE"] = "1" if config.enable_patch else "0"
    env["GE_SGLANG_MODEL_ID"] = config.model_id
    if config.lmcache_config_path is not None:
        env["GE_LMCACHE_CONFIG"] = config.lmcache_config_path
    if config.sglang_repo_path is not None:
        env["GE_SGLANG_REPO"] = config.sglang_repo_path
    if config.lmcache_repo_path is not None:
        env["GE_LMCACHE_REPO"] = config.lmcache_repo_path
    if manifest_path is not None:
        env["GE_PATCH_MANIFEST"] = str(Path(manifest_path))
    return env
