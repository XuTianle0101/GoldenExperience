from goldenexperience.lmcache_patch import PatchManifest
from goldenexperience.vllm_lmcache_runtime import RuntimeConfig, build_patch_environment, check_runtime


def test_patch_manifest_documents_non_invasive_hooks() -> None:
    manifest = PatchManifest.default()

    assert [hook.name for hook in manifest.ordered_hooks()] == [
        "serving_request_metadata",
        "lmcache_cross_model_lookup",
        "goldenexperience_materializer",
        "quality_gate_accounting",
    ]
    assert any("Do not modify serving-engine" in item for item in manifest.invariants)
    assert any("Do not replace LMCache" in item for item in manifest.invariants)
    assert "LMCache" in manifest.as_markdown()


def test_runtime_environment_is_namespaced() -> None:
    config = RuntimeConfig(
        model_id="qwen2.5-7b",
        lmcache_config_path="configs/lmcache.example.yaml",
        vllm_repo_path="third_party/vllm",
        lmcache_repo_path="third_party/LMCache",
        legacy_sglang_repo_path="third_party/sglang",
    )

    env = build_patch_environment(config, manifest_path="docs/generated_patch_manifest.md")

    assert env["GE_ENABLE_CROSS_MODEL_REUSE"] == "1"
    assert env["GE_RUNTIME_MODEL_ID"] == "qwen2.5-7b"
    assert env["GE_SERVING_MODEL_ID"] == "qwen2.5-7b"
    assert "GE_SGLANG_MODEL_ID" not in env
    assert env["GE_LMCACHE_CONFIG"] == "configs/lmcache.example.yaml"
    assert env["GE_VLLM_REPO"] == "third_party/vllm"
    assert env["GE_LEGACY_SGLANG_REPO"] == "third_party/sglang"
    assert env["GE_PATCH_MANIFEST"] == "docs/generated_patch_manifest.md"


def test_runtime_check_reports_dependency_names() -> None:
    status = check_runtime(RuntimeConfig(model_id="qwen2.5-7b"))

    assert set(status.available) == {"vLLM", "LMCache"}
    assert isinstance(status.ready, bool)


def test_runtime_check_can_include_legacy_sglang() -> None:
    status = check_runtime(RuntimeConfig(model_id="qwen2.5-7b", include_legacy_sglang=True))

    assert set(status.available) == {"vLLM", "LMCache", "SGLang"}
    assert isinstance(status.ready, bool)
