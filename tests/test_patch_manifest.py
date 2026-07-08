from goldenexperience.lmcache_patch import PatchManifest
from goldenexperience.runtime import RuntimeConfig, build_patch_environment, check_runtime


def test_patch_manifest_documents_non_invasive_hooks() -> None:
    manifest = PatchManifest.default()

    assert [hook.name for hook in manifest.ordered_hooks()] == [
        "engine_request_metadata",
        "lmcache_cross_model_lookup",
        "goldenexperience_materializer",
        "quality_gate_accounting",
    ]
    assert any("Do not modify vLLM" in item for item in manifest.invariants)
    assert any("Do not replace LMCache MP" in item for item in manifest.invariants)
    assert "LMCache" in manifest.as_markdown()


def test_runtime_environment_is_namespaced() -> None:
    config = RuntimeConfig(
        model_id="qwen3-8b",
        lmcache_config_path="configs/lmcache.example.yaml",
        vllm_repo_path="third_party/vllm",
        lmcache_repo_path="third_party/LMCache",
        mooncake_repo_path="third_party/Mooncake",
    )

    env = build_patch_environment(config, manifest_path="docs/generated_patch_manifest.md")

    assert env["GE_ENABLE_CROSS_MODEL_REUSE"] == "1"
    assert env["GE_MODEL_ID"] == "qwen3-8b"
    assert env["GE_INFERENCE_ENGINE"] == "vllm"
    assert env["GE_ENGINE"] == "vllm"
    assert env["GE_KV_BACKEND"] == "mp"
    assert env["GE_L2_BACKEND"] == "mooncake_store"
    assert env["GE_LMCACHE_MP_L2_ADAPTER_TYPE"] == "mooncake_store"
    assert env["GE_LMCACHE_CONFIG"] == "configs/lmcache.example.yaml"
    assert env["GE_PATCH_MANIFEST"] == "docs/generated_patch_manifest.md"


def test_runtime_check_reports_dependency_names() -> None:
    status = check_runtime(RuntimeConfig(model_id="qwen3-8b"))

    assert set(status.available) == {
        "vLLM",
        "LMCache",
        "Mooncake",
        "LMCache Mooncake extension",
    }
    assert isinstance(status.ready, bool)
