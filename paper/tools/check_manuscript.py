#!/usr/bin/env python3
"""Audit the paper against tracked publication-v5 evidence without opening sealed data."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/paper.md"
BIBLIOGRAPHY = ROOT / "paper/references.bib"
EVIDENCE_DIR = ROOT / "artifacts/publication_v5/evidence"
FIGURE_DIR = ROOT / "artifacts/publication_v5/figures"
INITIALIZATION = ROOT / "artifacts/publication_v5/initialization_v4.json"

FIGURE_NAMES = {
    "fig01_candidate_coverage",
    "fig02_full_prefix_by_length",
    "fig03_task_heterogeneity",
    "fig04_failure_overlap",
    "fig05_method_progression",
    "fig06_pipeline_stop",
}


def _reject_sealed_path(path: Path) -> None:
    if any("sealed" in part.lower() for part in path.parts):
        raise ValueError(f"refusing to access sealed path: {path}")


def _read(path: Path) -> bytes:
    _reject_sealed_path(path)
    return path.read_bytes()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read(path)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload, raw


def _load_csv(path: Path) -> list[dict[str, str]]:
    raw = _read(path)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_manifest_artifacts(manifest: dict[str, Any]) -> None:
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        raw = _read(path)
        _require(_sha256(raw) == artifact["sha256"], f"artifact hash differs: {path}")
        _require(len(raw) == artifact["size_bytes"], f"artifact size differs: {path}")


def _expand_citations(value: str) -> set[int]:
    citations: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            _require(start <= end, f"reversed citation range: {item}")
            citations.update(range(start, end + 1))
        else:
            citations.add(int(item))
    return citations


def _check_references(paper: str, bibliography: str) -> None:
    body, rendered_references = paper.split("\n## References\n", 1)
    reference_numbers = [
        int(value) for value in re.findall(r"^\[(\d+)\] ", rendered_references, re.MULTILINE)
    ]
    _require(reference_numbers == list(range(1, 24)), "rendered references are not 1 through 23")

    citations: set[int] = set()
    for match in re.finditer(r"\[([0-9]+(?:-[0-9]+)?(?:, [0-9]+(?:-[0-9]+)?)*)\]", body):
        citations.update(_expand_citations(match.group(1)))
    _require(
        citations == set(reference_numbers),
        "not every rendered reference is cited exactly in range",
    )

    bib_keys = re.findall(r"^@[A-Za-z]+\{([^,]+),", bibliography, re.MULTILINE)
    _require(len(bib_keys) == 23 and len(set(bib_keys)) == 23, "BibTeX entry count differs")
    _require(bibliography.count("{") == bibliography.count("}"), "BibTeX braces are unbalanced")


def _check_tables(paper: str) -> None:
    current_columns: int | None = None
    for line_number, line in enumerate(paper.splitlines(), start=1):
        if not line.startswith("|"):
            current_columns = None
            continue
        columns = line.count("|") - 1
        _require(columns > 1, f"invalid table row at line {line_number}")
        if current_columns is None:
            current_columns = columns
        _require(columns == current_columns, f"table width changes at line {line_number}")


def _check_links(paper: str) -> None:
    image_links = re.findall(r"!\[([^\]]+)\]\(([^)]+)\)", paper)
    _require(len(image_links) == 6, "manuscript must contain six figures")
    linked_names: set[str] = set()
    for description, target in image_links:
        _require(bool(description.strip()), "figure lacks alternative text")
        path = (PAPER.parent / target).resolve()
        _require(path.is_relative_to(ROOT), f"figure escapes repository: {target}")
        _require(path.is_file(), f"missing figure: {target}")
        linked_names.add(path.stem)
    _require(linked_names == FIGURE_NAMES, "manuscript figure set differs")

    captions = [int(value) for value in re.findall(r"^\*\*Figure (\d+)\.\*\*", paper, re.MULTILINE)]
    _require(captions == list(range(1, 7)), "figure captions are not numbered 1 through 6")

    repository_paths = {
        value
        for value in re.findall(r"`([^`]+)`", paper)
        if value.startswith(("artifacts/", "configs/", "docs/", "paper/"))
    }
    for value in repository_paths:
        _require((ROOT / value).exists(), f"manuscript references a missing path: {value}")


def _check_candidate_table(paper: str, candidates: list[dict[str, str]]) -> None:
    for row in candidates:
        expected = (
            f"| {row['rank']} | {row['seed']} | {float(row['training_total_loss']):.6f} | "
            f"{float(row['task_preservation']):.6f} | "
            f"{float(row['greedy_agreement']):.6f} | "
            f"{float(row['perplexity_drift_pct']):.2f}% | {row['safe_count']} | "
            f"{float(row['oracle_safe_coverage']):.6f} |"
        )
        _require(expected in paper, f"candidate row differs: rank {row['rank']} seed {row['seed']}")


def _check_rank_table(paper: str, ranks: list[dict[str, str]]) -> None:
    for row in ranks:
        task = f"{float(row['mean_task_preservation']):.6f}"
        coverage = f"{float(row['mean_oracle_safe_coverage']):.6f}"
        if row["is_selected_rank"] == "True":
            task = f"**{task}**"
            coverage = f"**{coverage}**"
        expected = (
            f"| {row['rank']} | {task} | {coverage} | "
            f"{float(row['mean_greedy_agreement']):.6f} | "
            f"{float(row['mean_p95_transform_ms']):.3f} ms |"
        )
        _require(expected in paper, f"rank aggregate differs: {row['rank']}")


def _check_task_table(paper: str, tasks: list[dict[str, str]]) -> None:
    labels = {
        "function_calling": "Function calling",
        "competition_math": "Competition math",
        "grade_school_math": "Grade-school math",
        "long_context_qa": "Long-context QA",
        "python_code_generation": "Python code",
    }
    for row in tasks:
        expected = (
            f"| {labels[row['task']]} | {row['safe_count']} / {row['sample_count']} | "
            f"{float(row['oracle_safe_coverage']):.6f} | "
            f"{float(row['greedy_agreement']):.6f} | "
            f"{float(row['perplexity_drift_pct']):.2f}% |"
        )
        _require(expected in paper, f"task aggregate differs: {row['task']}")


def main() -> None:
    paper_raw = _read(PAPER)
    bibliography_raw = _read(BIBLIOGRAPHY)
    paper = paper_raw.decode("ascii")
    bibliography = bibliography_raw.decode("ascii")
    _require("TODO" not in paper and "TBD" not in paper, "manuscript contains unfinished markers")

    evidence_manifest, _ = _load_json(EVIDENCE_DIR / "method_dev_evidence_manifest.v4.json")
    figure_manifest, _ = _load_json(FIGURE_DIR / "figures_manifest.v4.json")
    initialization, _ = _load_json(INITIALIZATION)
    _verify_manifest_artifacts(evidence_manifest)
    _verify_manifest_artifacts(figure_manifest)

    _require(
        evidence_manifest["registered_deployment"]["gate_passed"] is False,
        "evidence no longer records a failed deployment gate",
    )
    _require(
        initialization["pipeline"]["semantic_sealed_state"] == "locked",
        "workspace receipt no longer records a locked semantic split",
    )
    for required in (
        evidence_manifest["pipeline_id"],
        evidence_manifest["code_sha256"],
        evidence_manifest["registered_deployment"]["candidate_id"],
        "142/1024 = 0.138671875",
        "377/1024 = 0.368164",
        "no approved cross-model",
        "semantic payload remains locked",
    ):
        _require(str(required).lower() in paper.lower(), f"missing claim boundary: {required}")

    forbidden = (
        "the runtime audit passed",
        "the semantic evaluation passed",
        "we demonstrate a cross-model ttft improvement",
        "all four directions passed",
    )
    _require(not any(value in paper.lower() for value in forbidden), "manuscript overclaims")

    _check_references(paper, bibliography)
    _check_tables(paper)
    _check_links(paper)
    _check_candidate_table(paper, _load_csv(EVIDENCE_DIR / "method_dev_candidates.v4.csv"))
    _check_rank_table(paper, _load_csv(EVIDENCE_DIR / "method_dev_ranks.v4.csv"))
    _check_task_table(paper, _load_csv(EVIDENCE_DIR / "method_dev_tasks.v4.csv"))

    word_count = len(re.findall(r"\b[\w'-]+\b", paper))
    _require(word_count >= 5_500, f"manuscript is unexpectedly short: {word_count} words")
    print(f"manuscript_sha256={_sha256(paper_raw)}")
    print(f"bibliography_sha256={_sha256(bibliography_raw)}")
    print(f"word_count={word_count}")
    print("figures=6 references=23 evidence=verified sealed_state=locked")


if __name__ == "__main__":
    main()
