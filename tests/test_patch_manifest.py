from goldenexperience.lmcache_patch import PatchManifest
from goldenexperience.sglang_runtime import RuntimeConfig, build_patch_environment, check_runtime


def test_patch_manifest_documents_non_invasive_hooks() -> None:
    manifest = PatchManifest.default()

    assert [hook.name for hook in manifest.ordered_hooks()] == [
        "sglang_request_metadata",
        "lmcache_cross_model_lookup",
        "goldenexperience_materializer",
        "quality_gate_accounting",
    ]
    assert any("Do not modify SGLang" in item for item in manifest.invariants)
    assert any("Do not replace LMCache" in item for item in manifest.invariants)
    assert "LMCache" in manifest.as_markdown()


def test_runtime_environment_is_namespaced() -> None:
    config = RuntimeConfig(
        model_id="qwen2.5-7b",
        lmcache_config_path="configs/lmcache.example.yaml",
        sglang_repo_path="third_party/sglang",
        lmcache_repo_path="third_party/LMCache",
    )

    env = build_patch_environment(config, manifest_path="docs/generated_patch_manifest.md")

    assert env["GE_ENABLE_CROSS_MODEL_REUSE"] == "1"
    assert env["GE_SGLANG_MODEL_ID"] == "qwen2.5-7b"
    assert env["GE_LMCACHE_CONFIG"] == "configs/lmcache.example.yaml"
    assert env["GE_PATCH_MANIFEST"] == "docs/generated_patch_manifest.md"


def test_runtime_check_reports_dependency_names() -> None:
    status = check_runtime(RuntimeConfig(model_id="qwen2.5-7b"))

    assert set(status.available) == {"SGLang", "LMCache"}
    assert isinstance(status.ready, bool)
