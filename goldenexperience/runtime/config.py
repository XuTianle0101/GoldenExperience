"""Runtime dependency helpers for vLLM + LMCache MP + Mooncake Store."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeDependency:
    """A runtime package or executable expected to be installed externally."""

    name: str
    import_name: str | None
    repo_url: str
    install_hint: str
    command_name: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Local development configuration for the shared KV runtime stack."""

    model_id: str
    lmcache_config_path: str | None = None
    vllm_repo_path: str | None = None
    lmcache_repo_path: str | None = None
    mooncake_repo_path: str | None = None
    enable_patch: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def dependencies(self) -> tuple[RuntimeDependency, ...]:
        return (
            RuntimeDependency(
                name="vLLM",
                import_name="vllm",
                command_name="vllm",
                repo_url="https://github.com/vllm-project/vllm.git",
                install_hint="Install vLLM before launching the OpenAI-compatible engine.",
            ),
            RuntimeDependency(
                name="LMCache",
                import_name="lmcache",
                command_name="lmcache",
                repo_url="https://github.com/LMCache/LMCache.git",
                install_hint="Install LMCache with MP server support and Mooncake Store adapter.",
            ),
            RuntimeDependency(
                name="Mooncake",
                import_name=None,
                command_name="mooncake_master",
                repo_url="https://github.com/kvcache-ai/Mooncake.git",
                install_hint=(
                    "Install Mooncake and ensure mooncake_master is on PATH; LMCache must be "
                    "built with Mooncake support for the mooncake_store L2 adapter."
                ),
            ),
            RuntimeDependency(
                name="LMCache Mooncake extension",
                import_name="lmcache.lmcache_mooncake",
                command_name=None,
                repo_url="https://github.com/LMCache/LMCache.git",
                install_hint=(
                    "Reinstall LMCache from source with BUILD_MOONCAKE=1 and "
                    "MOONCAKE_INCLUDE_DIR pointing at Mooncake Store headers."
                ),
            ),
        )


@dataclass(frozen=True)
class RuntimeStatus:
    """Import, command, and optional source path checks for runtime projects."""

    available: dict[str, bool]
    repo_paths: dict[str, str | None]
    missing_hints: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return all(self.available.values())


def _command_exists(command_name: str) -> bool:
    import shutil

    return shutil.which(command_name) is not None


def _import_exists(import_name: str) -> bool:
    if import_name == "lmcache.lmcache_mooncake":
        lmcache_spec = importlib.util.find_spec("lmcache")
        if lmcache_spec is None or lmcache_spec.submodule_search_locations is None:
            return False
        return any(
            candidate.exists()
            for location in lmcache_spec.submodule_search_locations
            for candidate in Path(location).glob("lmcache_mooncake*")
        )
    return importlib.util.find_spec(import_name) is not None


def check_runtime(config: RuntimeConfig) -> RuntimeStatus:
    available: dict[str, bool] = {}
    repo_paths = {
        "vLLM": config.vllm_repo_path,
        "LMCache": config.lmcache_repo_path,
        "Mooncake": config.mooncake_repo_path,
    }
    missing_hints: list[str] = []
    for dependency in config.dependencies:
        import_found = (
            True if dependency.import_name is None else _import_exists(dependency.import_name)
        )
        command_found = (
            True if dependency.command_name is None else _command_exists(dependency.command_name)
        )
        found = import_found and command_found
        available[dependency.name] = found
        if not found:
            missing_hints.append(
                f"{dependency.name}: {dependency.install_hint} ({dependency.repo_url})"
            )
    return RuntimeStatus(
        available=available,
        repo_paths=repo_paths,
        missing_hints=tuple(missing_hints),
    )


def build_patch_environment(
    config: RuntimeConfig, manifest_path: str | Path | None = None
) -> dict[str, str]:
    """Build GoldenExperience-only env vars for wrapper scripts."""

    env = dict(config.extra_env)
    env["GE_ENABLE_CROSS_MODEL_REUSE"] = "1" if config.enable_patch else "0"
    env["GE_MODEL_ID"] = config.model_id
    env["GE_INFERENCE_ENGINE"] = "vllm"
    env["GE_ENGINE"] = "vllm"
    env["GE_KV_BACKEND"] = "mp"
    env["GE_L2_BACKEND"] = "mooncake_store"
    env["GE_LMCACHE_MP_L2_ADAPTER_TYPE"] = "mooncake_store"
    if config.lmcache_config_path is not None:
        env["GE_LMCACHE_CONFIG"] = config.lmcache_config_path
    if config.vllm_repo_path is not None:
        env["GE_VLLM_REPO"] = config.vllm_repo_path
    if config.lmcache_repo_path is not None:
        env["GE_LMCACHE_REPO"] = config.lmcache_repo_path
    if config.mooncake_repo_path is not None:
        env["GE_MOONCAKE_REPO"] = config.mooncake_repo_path
    if manifest_path is not None:
        env["GE_PATCH_MANIFEST"] = str(Path(manifest_path))
    return env
