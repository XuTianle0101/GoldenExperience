#!/usr/bin/env python3
"""Validate the tracked related-work audit without downloading source payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = Path("artifacts/publication_v5/development/related_work_fulltext_audit.json")
INITIALIZATION = Path("artifacts/publication_v5/initialization_v4.json")
SCHEMA_VERSION = "goldenexperience.publication_v5_related_work_audit.v3"
MIN_SOURCE_COUNT = 16

EXPECTED_DOCUMENTS = {
    "docs/paper_outline.md",
    "docs/related_work_matrix.md",
    "paper/paper.md",
    "paper/references.bib",
    "paper/tools/check_manuscript.py",
    "paper/tools/check_related_work.py",
}
EXPECTED_DOMAINS = {
    "proceedings.mlsys.org",
    "www.usenix.org",
    "www.sigops.org",
    "2026.eurosys.org",
    "proceedings.neurips.cc",
    "neurips.cc",
    "proceedings.mlr.press",
    "icml.cc",
    "openreview.net",
    "api.openreview.net",
    "dl.acm.org",
}
EXPECTED_QUERIES = {
    "cross-model AND KV cache",
    "cross-model AND cache reuse",
    "multi-model AND KV cache",
    "heterogeneous AND KV cache",
    "model-to-model AND cache",
    "cross-architecture AND KV",
    "latent communication AND KV cache",
    "KV cache AND translation",
}
EXPECTED_FINDINGS = {
    "closest_serving_predecessor",
    "co_designed_exact_sharing_prior_art",
    "cross_size_transport_prior_art",
    "guarantee_prior_art",
    "learned_communication_prior_art",
    "online_context_reuse_prior_art",
    "privacy_boundary",
    "proxykv_boundary",
    "runtime_boundary",
    "storage_placement_boundary",
    "unequal_head_correction",
}
EXPECTED_PMLR_TREE = [
    ".github",
    ".github/pull_request_template.md",
    "README.md",
]
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_GIT_SHA = re.compile(r"[0-9a-f]{40}")
ARXIV_VERSION = re.compile(r"[0-9]{4}\.[0-9]{5}v[1-9][0-9]*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_object(path: Path) -> dict[str, Any]:
    _require(
        not any("sealed" in part.lower() for part in path.parts),
        f"refusing sealed path: {path}",
    )
    payload = json.loads(path.read_bytes())
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and HEX_GIT_SHA.fullmatch(value) is not None


def _timestamp(value: object, label: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"invalid timestamp: {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {label}") from exc
    return parsed


def _repository_path(root: Path, value: object) -> Path:
    _require(isinstance(value, str) and value, "document path must be a non-empty string")
    relative = Path(value)
    _require(not relative.is_absolute(), f"document path is absolute: {value}")
    _require(
        not any("sealed" in part.lower() for part in relative.parts),
        f"refusing sealed path: {value}",
    )
    resolved = (root / relative).resolve()
    _require(resolved.is_relative_to(root), f"document path escapes repository: {value}")
    _require(
        not any("sealed" in part.lower() for part in resolved.parts),
        f"refusing resolved sealed path: {value}",
    )
    _require(resolved.is_file(), f"document is missing: {value}")
    return resolved


def _verify_documents(root: Path, audit: dict[str, Any]) -> None:
    documents = audit.get("document_sha256")
    _require(isinstance(documents, dict), "document_sha256 must be an object")
    _require(set(documents) == EXPECTED_DOCUMENTS, "audited document set differs")
    updated = audit.get("documents_updated")
    _require(isinstance(updated, list), "documents_updated must be an array")
    _require(
        set(updated) == EXPECTED_DOCUMENTS,
        "updated document set differs",
    )
    for relative, expected in documents.items():
        _require(_is_sha256(expected), f"invalid document hash: {relative}")
        path = _repository_path(root, relative)
        _require(_sha256(path) == expected, f"document hash differs: {relative}")


def _verify_sources(audit: dict[str, Any], matrix: str) -> None:
    sources = audit.get("sources")
    _require(isinstance(sources, list), "sources must be an array")
    verification = audit.get("verification")
    _require(isinstance(verification, dict), "verification must be an object")
    _require(len(sources) >= MIN_SOURCE_COUNT, "too few fixed-version full-text sources")
    _require(verification.get("source_count") == len(sources), "source count differs")

    arxiv_ids: set[str] = set()
    titles: set[str] = set()
    pdf_hashes: set[str] = set()
    for source in sources:
        _require(isinstance(source, dict), "source entry must be an object")
        arxiv_id = source.get("arxiv_id")
        title = source.get("title")
        pdf_hash = source.get("pdf_sha256")
        _require(
            isinstance(arxiv_id, str) and ARXIV_VERSION.fullmatch(arxiv_id), "invalid arXiv version"
        )
        _require(isinstance(title, str) and title.strip(), f"missing source title: {arxiv_id}")
        _require(_is_sha256(pdf_hash), f"invalid PDF hash: {arxiv_id}")
        _require(
            isinstance(source.get("pdf_size_bytes"), int) and source["pdf_size_bytes"] > 0,
            f"invalid PDF size: {arxiv_id}",
        )
        venue = source.get("venue")
        _require(
            venue is None or isinstance(venue, str) and venue.strip(), f"invalid venue: {arxiv_id}"
        )
        _require(arxiv_id not in arxiv_ids, f"duplicate arXiv version: {arxiv_id}")
        _require(title not in titles, f"duplicate source title: {title}")
        _require(pdf_hash not in pdf_hashes, f"duplicate PDF hash: {pdf_hash}")
        _require(
            arxiv_id.rsplit("v", 1)[0] in matrix, f"source absent from claim matrix: {arxiv_id}"
        )
        arxiv_ids.add(arxiv_id)
        titles.add(title)
        pdf_hashes.add(pdf_hash)

    for key in (
        "arxiv_current_versions_checked",
        "broad_arxiv_keyword_combinations_checked",
        "exact_version_pdfs_hashed",
        "full_text_claims_checked",
        "official_eurosys_2026_title_list_checked",
        "official_icml_2026_pmlr_repository_checked",
        "stale_vcache_ar5iv_text_rejected",
    ):
        _require(verification.get(key) is True, f"verification flag is not true: {key}")
    _require(
        verification.get("tracked_full_text_payloads") is False, "full-text payload policy differs"
    )


def _verify_search(audit: dict[str, Any]) -> None:
    search = audit.get("search_queries")
    _require(isinstance(search, dict), "search_queries must be an object")
    _require(
        set(search.get("arxiv_combinations", [])) == EXPECTED_QUERIES, "search query set differs"
    )
    for key in ("candidate_metadata_snapshot_sha256", "fixed_source_metadata_snapshot_sha256"):
        _require(_is_sha256(search.get(key)), f"invalid search snapshot hash: {key}")
    result_hashes = search.get("result_snapshots_sha256")
    _require(
        isinstance(result_hashes, dict) and len(result_hashes) >= 6,
        "result snapshots are incomplete",
    )
    _require(
        all(_is_sha256(value) for value in result_hashes.values()), "invalid result snapshot hash"
    )


def _verify_official_checks(audit: dict[str, Any], audited_at: datetime) -> None:
    expected_status = (
        "partial_eurosys_verified_icml_repository_placeholder_other_direct_tls_blocked"
    )
    _require(
        audit.get("official_proceedings_index_status") == expected_status,
        "official index status differs",
    )
    note = audit.get("official_proceedings_note")
    _require(
        isinstance(note, str) and "no unqualified first claim" in note,
        "priority-claim boundary is missing",
    )

    checks = audit.get("official_proceedings_checks")
    _require(isinstance(checks, dict), "official_proceedings_checks must be an object")
    direct = checks.get("direct_tls")
    _require(isinstance(direct, dict), "direct TLS check is missing")
    attempts = direct.get("attempted_at")
    _require(isinstance(attempts, list) and len(attempts) >= 2, "too few direct TLS attempts")
    attempt_times = [_timestamp(value, "direct_tls.attempted_at") for value in attempts]
    _require(
        attempt_times == sorted(set(attempt_times)), "direct TLS attempt times differ or repeat"
    )
    _require(direct.get("attempt_count") == len(attempts), "direct TLS attempt count differs")
    _require(set(direct.get("domains", [])) == EXPECTED_DOMAINS, "official domain set differs")
    _require(
        direct.get("latest_probe_endpoint_count") == len(EXPECTED_DOMAINS),
        "probe endpoint count differs",
    )
    _require(
        direct.get("latest_probe_http_response_count") == 0,
        "blocked probe recorded an HTTP response",
    )
    _require(direct.get("latest_probe_curl_exit_code") == 35, "blocked probe exit code differs")
    _require(
        _is_sha256(direct.get("latest_probe_snapshot_sha256")), "invalid direct probe snapshot hash"
    )
    _require(
        direct.get("latest_probe_snapshot_tracked") is False, "direct probe payload policy differs"
    )
    _require(
        direct.get("result")
        == "tls_connection_reset_by_network_policy_on_every_endpoint_and_attempt",
        "direct TLS result differs",
    )
    _require(audited_at >= attempt_times[-1], "audit predates the latest direct TLS attempt")

    eurosys = checks.get("eurosys_2026")
    _require(isinstance(eurosys, dict), "EuroSys check is missing")
    _require(
        eurosys.get("conference_repository")
        == "https://github.com/eurosys2026/eurosys2026.github.io",
        "EuroSys repository differs",
    )
    _require(_is_git_sha(eurosys.get("repository_commit")), "invalid EuroSys commit")
    _require(_is_sha256(eurosys.get("paper_list_sha256")), "invalid EuroSys paper-list hash")
    _require(
        isinstance(eurosys.get("paper_list_size_bytes"), int)
        and eurosys["paper_list_size_bytes"] > 0,
        "invalid EuroSys paper-list size",
    )
    matches = eurosys.get("title_level_kv_matches")
    _require(isinstance(matches, list) and matches, "EuroSys KV title matches are missing")
    _require(
        all(
            isinstance(item, dict) and item.get("title") and item.get("disposition")
            for item in matches
        ),
        "invalid EuroSys title match",
    )

    pmlr = checks.get("icml_2026_pmlr_repository")
    _require(isinstance(pmlr, dict), "ICML PMLR repository check is missing")
    _require(pmlr.get("publisher_organization") == "mlresearch", "PMLR organization differs")
    _require(
        pmlr.get("publisher_identity") == "Proceedings of Machine Learning Research",
        "PMLR identity differs",
    )
    _require(
        pmlr.get("repository") == "https://github.com/mlresearch/v306", "PMLR repository differs"
    )
    _require(_is_git_sha(pmlr.get("repository_commit")), "invalid PMLR commit")
    _require(_is_git_sha(pmlr.get("recursive_tree_sha")), "invalid PMLR tree hash")
    _require(pmlr.get("recursive_tree_entries") == EXPECTED_PMLR_TREE, "PMLR tree entries differ")
    _require(
        not any(Path(value).suffix.lower() in {".bib", ".pdf"} for value in EXPECTED_PMLR_TREE),
        "PMLR placeholder unexpectedly contains papers",
    )
    _require(pmlr.get("paper_index_present") is False, "PMLR placeholder is mislabeled as an index")
    _require(
        pmlr.get("disposition") == "publisher_owned_placeholder_not_an_accepted_paper_index",
        "PMLR placeholder disposition differs",
    )
    _require(_is_sha256(pmlr.get("readme_sha256")), "invalid PMLR README hash")
    _require(
        isinstance(pmlr.get("readme_size_bytes"), int) and pmlr["readme_size_bytes"] > 0,
        "invalid PMLR README size",
    )

    verification = audit["verification"]
    _require(
        verification.get("official_proceedings_direct_tls_attempts") == len(attempts),
        "verified TLS attempt count differs",
    )


def validate_audit(root: Path = ROOT, audit_path: Path | None = None) -> dict[str, int | str]:
    root = root.resolve()
    selected = audit_path or root / DEFAULT_AUDIT
    if not selected.is_absolute():
        selected = root / selected
    audit = _load_object(selected.resolve())
    _require(audit.get("schema_version") == SCHEMA_VERSION, "related-work schema version differs")
    _require(
        audit.get("authority") == "claim_scoping_evidence_not_empirical_result",
        "audit authority differs",
    )
    audited_at = _timestamp(audit.get("audited_at"), "audited_at")
    try:
        search_date = date.fromisoformat(str(audit.get("search_snapshot")))
    except ValueError as exc:
        raise ValueError("invalid search snapshot date") from exc
    _require(search_date <= audited_at.date(), "search snapshot postdates audit")

    initialization = _load_object(root / INITIALIZATION)
    _require(
        audit.get("benchmark_manifest_sha256") == initialization["benchmark"]["content_sha256"],
        "benchmark binding differs",
    )
    _require(
        audit.get("executable_code_sha256") == initialization["code"]["source_tree_sha256"],
        "executable-code binding differs",
    )
    findings = audit.get("findings")
    _require(isinstance(findings, dict), "findings must be an object")
    _require(set(findings) == EXPECTED_FINDINGS, "finding set differs")
    _require(
        all(isinstance(value, str) and value.strip() for value in findings.values()),
        "finding text is empty",
    )

    _verify_documents(root, audit)
    matrix = (root / "docs/related_work_matrix.md").read_text(encoding="ascii")
    paper = (root / "paper/paper.md").read_text(encoding="ascii")
    _verify_sources(audit, matrix)
    _verify_search(audit)
    _verify_official_checks(audit, audited_at)
    _require(
        'avoids unqualified "first" claims' in paper, "manuscript priority boundary is missing"
    )
    for phrase in (
        "we introduce the first",
        "we present the first",
        "we demonstrate the first",
        "this paper presents the first",
        "this work is the first",
    ):
        _require(phrase not in paper.lower(), f"unqualified priority claim found: {phrase}")

    return {
        "direct_tls_attempts": audit["official_proceedings_checks"]["direct_tls"]["attempt_count"],
        "documents": len(audit["document_sha256"]),
        "sources": len(audit["sources"]),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = validate_audit(args.root, args.audit)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
