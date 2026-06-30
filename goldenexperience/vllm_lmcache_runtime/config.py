"""Runtime dependency helpers for the vLLM + LMCache stack."""

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
    vllm_repo_path: str | None = None
    lmcache_repo_path: str | None = None
    legacy_sglang_repo_path: str | None = None
    include_legacy_sglang: bool = False
    enable_patch: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[RuntimeDependency, ...]:
        dependencies = [
            RuntimeDependency(
                name="vLLM",
                import_name="vllm",
                repo_url="https://github.com/vllm-project/vllm.git",
                install_hint="Install vLLM before running the default LMCache MP baseline.",
            ),
            RuntimeDependency(
                name="LMCache",
                import_name="lmcache",
                repo_url="https://github.com/LMCache/LMCache.git",
                install_hint="Install LMCache before enabling KV cache offload/reuse.",
            ),
        ]
        if self.include_legacy_sglang or self.legacy_sglang_repo_path is not None:
            dependencies.append(self.legacy_sglang_dependency)
        return tuple(dependencies)

    @property
    def legacy_sglang_dependency(self) -> RuntimeDependency:
        return RuntimeDependency(
            name="SGLang",
            import_name="sglang",
            repo_url="https://github.com/sgl-project/sglang.git",
            install_hint="Install SGLang only for the legacy in-process control path.",
        )

    @property
    def legacy_dependencies(self) -> tuple[RuntimeDependency, ...]:
        return (self.legacy_sglang_dependency,)


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
        "vLLM": config.vllm_repo_path,
        "LMCache": config.lmcache_repo_path,
    }
    if config.include_legacy_sglang or config.legacy_sglang_repo_path is not None:
        repo_paths["SGLang"] = config.legacy_sglang_repo_path
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

    These variables are intentionally namespaced. The serving engine and LMCache remain
    responsible for their own documented launch flags and cache/offload configuration.
    """

    env = dict(config.extra_env)
    env["GE_ENABLE_CROSS_MODEL_REUSE"] = "1" if config.enable_patch else "0"
    env["GE_RUNTIME_MODEL_ID"] = config.model_id
    env["GE_SERVING_MODEL_ID"] = config.model_id
    if config.lmcache_config_path is not None:
        env["GE_LMCACHE_CONFIG"] = config.lmcache_config_path
    if config.vllm_repo_path is not None:
        env["GE_VLLM_REPO"] = config.vllm_repo_path
    if config.lmcache_repo_path is not None:
        env["GE_LMCACHE_REPO"] = config.lmcache_repo_path
    if config.legacy_sglang_repo_path is not None:
        env["GE_LEGACY_SGLANG_REPO"] = config.legacy_sglang_repo_path
    if manifest_path is not None:
        env["GE_PATCH_MANIFEST"] = str(manifest_path)
    return env
