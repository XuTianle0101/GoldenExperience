from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from paper.tools.check_related_work import _repository_path, validate_audit

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts/publication_v5/development/related_work_fulltext_audit.json"


def _mutated_audit(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    payload = copy.deepcopy(json.loads(AUDIT.read_text(encoding="utf-8")))
    mutate(payload)
    path = tmp_path / "related-work-audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_tracked_related_work_audit_passes() -> None:
    assert validate_audit(ROOT) == {
        "direct_tls_attempts": 2,
        "documents": 6,
        "sources": 16,
        "status": "passed",
    }


def test_related_work_audit_rejects_document_tampering(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["document_sha256"]["paper/paper.md"] = "0" * 64

    with pytest.raises(ValueError, match="document hash differs"):
        validate_audit(ROOT, _mutated_audit(tmp_path, mutate))


def test_related_work_audit_rejects_duplicate_sources(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["sources"].append(copy.deepcopy(payload["sources"][0]))
        payload["verification"]["source_count"] += 1

    with pytest.raises(ValueError, match="duplicate arXiv version"):
        validate_audit(ROOT, _mutated_audit(tmp_path, mutate))


def test_related_work_audit_rejects_executable_binding_change(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["executable_code_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="executable-code binding differs"):
        validate_audit(ROOT, _mutated_audit(tmp_path, mutate))


def test_related_work_audit_rejects_false_pmlr_index_claim(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["official_proceedings_checks"]["icml_2026_pmlr_repository"][
            "paper_index_present"
        ] = True

    with pytest.raises(ValueError, match="mislabeled as an index"):
        validate_audit(ROOT, _mutated_audit(tmp_path, mutate))


def test_related_work_audit_rejects_search_query_removal(tmp_path: Path) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["search_queries"]["arxiv_combinations"].pop()

    with pytest.raises(ValueError, match="search query set differs"):
        validate_audit(ROOT, _mutated_audit(tmp_path, mutate))


def test_related_work_audit_rejects_repository_escape() -> None:
    with pytest.raises(ValueError, match="escapes repository"):
        _repository_path(ROOT, "../outside.md")


def test_related_work_audit_rejects_sealed_path() -> None:
    with pytest.raises(ValueError, match="refusing sealed path"):
        _repository_path(ROOT, "artifacts/cache/semantic_sealed/payload.json")


def test_related_work_audit_rejects_symlink_into_sealed_path(tmp_path: Path) -> None:
    sealed = tmp_path / "semantic_sealed" / "payload.json"
    sealed.parent.mkdir()
    sealed.write_text("{}\n", encoding="ascii")
    visible = tmp_path / "visible.json"
    visible.symlink_to(sealed)

    with pytest.raises(ValueError, match="refusing resolved sealed path"):
        _repository_path(tmp_path, visible.name)
