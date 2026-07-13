from argparse import Namespace
from pathlib import Path

from goldenexperience.cli.publication_benchmark import _parser, freeze_manifest
from goldenexperience.size_variant.cached_kv_manifest import (
    chat_template_sha256,
    tokenizer_semantic_sha256,
)


def _tokenizer_model(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    (root / "tokenizer.json").write_text('{"model":"test"}', encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        '{"chat_template":"{{ messages }}","eos_token":"<end>"}',
        encoding="utf-8",
    )
    return root


def test_freeze_binds_semantic_tokenizer_and_chat_identities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tokenizer_model = _tokenizer_model(tmp_path)
    sources = tmp_path / "sources.json"
    records = tmp_path / "records.jsonl"
    sealed = tmp_path / "sealed.bin"
    sources.write_text("[]\n", encoding="utf-8")
    records.write_text("", encoding="utf-8")
    sealed.write_bytes(b"sealed")
    monkeypatch.setattr(
        "goldenexperience.benchmarks.publication.PublicationBenchmarkManifest.save",
        lambda self, path: None,
    )

    manifest = freeze_manifest(
        Namespace(
            sources=sources,
            records=records,
            tokenizer_model=tokenizer_model,
            sealed_payload=sealed,
            deprecated_synthetic_sealed=None,
            output=tmp_path / "manifest.json",
        )
    )

    assert manifest.tokenizer_sha256 == tokenizer_semantic_sha256(tokenizer_model)
    assert manifest.chat_template_sha256 == chat_template_sha256(tokenizer_model)


def test_freeze_cli_requires_one_canonical_tokenizer_model() -> None:
    args = _parser().parse_args(
        [
            "freeze",
            "--sources",
            "sources.json",
            "--records",
            "records.jsonl",
            "--tokenizer-model",
            "Qwen3-8B",
            "--sealed-payload",
            "sealed.bin",
            "--output",
            "benchmark.json",
        ]
    )

    assert args.tokenizer_model == Path("Qwen3-8B")
    assert not hasattr(args, "tokenizer")
    assert not hasattr(args, "chat_template")
