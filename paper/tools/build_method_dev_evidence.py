#!/usr/bin/env python3
"""Build the deterministic publication bundle for the terminal v4 method-dev result."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/cache/publication_v5_workspace_v5/.pipeline/work/qwen3_4b_to_8b"
    / "evaluate_method_dev/method_dev_report.json"
)
FIT_RECEIPT = ROOT / "artifacts/publication_v5/stages/qwen3_4b_to_8b.fit_transport.v4.json"
FAILED_RECEIPT = (
    ROOT / "artifacts/publication_v5/stages" / "qwen3_4b_to_8b.evaluate_method_dev.v4.failed.json"
)
DIAGNOSTIC = ROOT / "artifacts/publication_v5/development/v4_method_dev_diagnostic.json"
OUTPUT_DIR = ROOT / "artifacts/publication_v5/evidence"

GREEDY_GATE = 0.98
DRIFT_GATE = 2.0
COVERAGE_GATE = 0.45
DEPLOYMENT_SEED = 17
EXPECTED_RANKS = (32, 64, 128)
EXPECTED_SEEDS = (17, 29, 43)

TASK_DATASETS = {
    "function_calling": ("bfcl",),
    "competition_math": ("math",),
    "grade_school_math": ("gsm8k",),
    "long_context_qa": (
        "longbench_hotpotqa",
        "longbench_multifieldqa",
        "longbench_qasper",
    ),
    "python_code_generation": ("humaneval", "mbpp"),
}
DATASET_TO_TASK = {
    dataset: task for task, datasets in TASK_DATASETS.items() for dataset in datasets
}

ARCHIVE_NAME = "method_dev_report.v4.json.gz"
CANDIDATE_TABLE_NAME = "method_dev_candidates.v4.csv"
RANK_TABLE_NAME = "method_dev_ranks.v4.csv"
TASK_TABLE_NAME = "method_dev_tasks.v4.csv"
BUCKET_TABLE_NAME = "method_dev_token_buckets.v4.csv"
FAILURE_TABLE_NAME = "method_dev_failure_overlap.v4.csv"
SAFE_SET_TABLE_NAME = "method_dev_safe_sets.v4.csv"
MANIFEST_NAME = "method_dev_evidence_manifest.v4.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to the uncompressed v4 method-dev report.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in memory and fail if tracked outputs differ.",
    )
    return parser.parse_args()


def _reject_sealed_path(path: Path) -> None:
    if any("sealed" in part.lower() for part in path.parts):
        raise ValueError(f"refusing to access sealed path: {path}")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    _reject_sealed_path(path)
    raw = path.read_bytes()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _gzip_bytes(raw: bytes) -> bytes:
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=stream,
        mtime=0,
    ) as handle:
        handle.write(raw)
    compressed = stream.getvalue()
    if gzip.decompress(compressed) != raw:
        raise ValueError("deterministic report archive failed its round trip")
    return compressed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: derived {actual}, receipt {expected}")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered) and 0 <= quantile <= 1, "invalid percentile input")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _sample_metadata(sample_id: str) -> tuple[int, str, str]:
    parts = sample_id.split(".")
    _require(len(parts) == 4 and parts[0] == "method_dev", f"invalid sample id {sample_id}")
    bucket = int(parts[1])
    dataset = parts[2]
    _require(bucket in {128, 512, 2048, 8192}, f"unexpected token bucket {bucket}")
    _require(dataset in DATASET_TO_TASK, f"unexpected dataset {dataset}")
    return bucket, dataset, DATASET_TO_TASK[dataset]


def _row_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    greedy_tokens = int(row["greedy_tokens"])
    teacher_tokens = int(row["teacher_tokens"])
    _require(greedy_tokens > 0 and teacher_tokens > 0, "measurement has no evaluation tokens")
    greedy = int(row["greedy_matches"]) / greedy_tokens
    drift = (
        abs(math.expm1((float(row["bridge_nll"]) - float(row["native_nll"])) / teacher_tokens))
        * 100
    )
    task_regression = bool(
        float(row["native_task_score"]) >= float(row["task_pass_threshold"])
        and float(row["bridge_task_score"]) < float(row["task_pass_threshold"])
    )
    greedy_failure = greedy < GREEDY_GATE
    drift_failure = drift > DRIFT_GATE
    bucket, dataset, task = _sample_metadata(str(row["sample_id"]))
    return {
        "bucket": bucket,
        "dataset": dataset,
        "task": task,
        "task_preservation": 1.0
        - max(0.0, float(row["native_task_score"]) - float(row["bridge_task_score"])),
        "greedy_agreement": greedy,
        "perplexity_drift_pct": drift,
        "task_regression": task_regression,
        "greedy_failure": greedy_failure,
        "drift_failure": drift_failure,
        "safe": not (task_regression or greedy_failure or drift_failure),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot aggregate an empty measurement group")
    metrics = [_row_metrics(row) for row in rows]
    teacher_tokens = sum(int(row["teacher_tokens"]) for row in rows)
    log_ratio = (
        sum(float(row["bridge_nll"]) for row in rows)
        - sum(float(row["native_nll"]) for row in rows)
    ) / teacher_tokens
    safe_count = sum(bool(item["safe"]) for item in metrics)
    greedy_matches = sum(int(row["greedy_matches"]) for row in rows)
    greedy_tokens = sum(int(row["greedy_tokens"]) for row in rows)
    return {
        "sample_count": len(rows),
        "safe_count": safe_count,
        "oracle_safe_coverage": safe_count / len(rows),
        "task_preservation": statistics.fmean(float(item["task_preservation"]) for item in metrics),
        "native_task_score": statistics.fmean(float(row["native_task_score"]) for row in rows),
        "bridge_task_score": statistics.fmean(float(row["bridge_task_score"]) for row in rows),
        "greedy_agreement": greedy_matches / greedy_tokens,
        "perplexity_drift_pct": abs(math.expm1(log_ratio)) * 100,
        "p50_transform_ms": _percentile([float(row["transform_ms"]) for row in rows], 0.50),
        "p95_transform_ms": _percentile([float(row["transform_ms"]) for row in rows], 0.95),
        "task_regression_count": sum(bool(item["task_regression"]) for item in metrics),
        "greedy_failure_count": sum(bool(item["greedy_failure"]) for item in metrics),
        "drift_failure_count": sum(bool(item["drift_failure"]) for item in metrics),
        "unsafe_count": len(rows) - safe_count,
    }


def _validate_inputs(
    report: Mapping[str, Any],
    report_raw: bytes,
    fit: Mapping[str, Any],
    failed: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> None:
    expected_report = failed["method_dev_report"]
    _require(_sha256(report_raw) == expected_report["file_sha256"], "report SHA-256 mismatch")
    _require(len(report_raw) == expected_report["size_bytes"], "report size mismatch")
    _require(
        report["schema_version"] == "goldenexperience.v5_method_dev_report.v1",
        "bad report schema",
    )
    _require(fit["authority"] == "verified_pipeline_stage_receipt", "fit receipt lacks authority")
    _require(
        failed["authority"] == "non_authoritative_failed_stage_diagnostic",
        "method-dev receipt is not a failed-stage diagnostic",
    )
    _require(
        diagnostic["authority"] == "diagnostic_only_not_publication_success_evidence",
        "mechanism diagnostic authority changed",
    )
    expected_disposition = (
        "selector_calibration_validation_semantic_sealed_and_runtime_stages_blocked"
    )
    _require(
        failed["protocol_disposition"] == expected_disposition,
        "downstream stages are not blocked",
    )
    bindings = {
        report["pipeline_id"],
        fit["pipeline_id"],
        failed["pipeline_id"],
    }
    _require(len(bindings) == 1, "pipeline identities differ")
    code_hashes = {report["code_sha256"], fit["code_sha256"], failed["code_sha256"]}
    _require(len(code_hashes) == 1, "executable source hashes differ")
    _require(
        report["transport_fit_manifest_sha256"] == failed["fit_manifest_content_sha256"],
        "fit binding differs",
    )
    _require(
        fit["fit_manifest"]["content_sha256"] == failed["fit_manifest_content_sha256"],
        "verified fit receipt binding differs",
    )
    _require(
        report["method_dev_trace_manifest_sha256"]
        == failed["method_dev_trace_manifest_content_sha256"],
        "trace binding differs",
    )
    _require(
        report["raw_sample_store_sha256"] == failed["raw_sample_store_sha256"],
        "method-dev sample-store binding differs",
    )
    _require(
        diagnostic["source_bindings"]["method_dev_report_file_sha256"]
        == expected_report["file_sha256"],
        "mechanism diagnostic report binding differs",
    )
    _require(
        diagnostic["source_bindings"]["raw_sample_store_sha256"]
        == report["raw_sample_store_sha256"],
        "mechanism diagnostic sample-store binding differs",
    )
    _require(
        len(report["measurements"]) == expected_report["measurement_count"],
        "receipt measurement count differs",
    )
    _require(report["generation_tokens"] == 16, "generation horizon differs")


def _candidate_rows(
    report: Mapping[str, Any],
    fit: Mapping[str, Any],
    failed: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[Mapping[str, Any]]],
    dict[str, set[str]],
    int,
    str,
]:
    measurements = report["measurements"]
    _require(isinstance(measurements, list) and len(measurements) == 9216, "matrix is incomplete")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in measurements:
        _require(isinstance(row, dict), "measurement is not an object")
        identity = (str(row["sample_id"]), str(row["candidate_id"]))
        _require(identity not in seen, f"duplicate measurement {identity}")
        seen.add(identity)
        grouped[str(row["candidate_id"])].append(row)

    fit_candidates = {item["candidate_id"]: item for item in fit["candidates"]}
    receipt_candidates = {item["candidate_id"]: item for item in failed["candidate_metrics"]}
    _require(
        set(grouped) == set(fit_candidates) == set(receipt_candidates),
        "candidate sets differ",
    )
    _require(len(grouped) == 9, "expected nine candidates")

    expected_pairs = {(rank, seed) for rank in EXPECTED_RANKS for seed in EXPECTED_SEEDS}
    observed_pairs = {(int(rows[0]["rank"]), int(rows[0]["seed"])) for rows in grouped.values()}
    _require(observed_pairs == expected_pairs, "rank/seed matrix differs")
    sample_sets = [{str(row["sample_id"]) for row in rows} for rows in grouped.values()]
    _require(all(len(rows) == 1024 for rows in grouped.values()), "candidate prompt count differs")
    _require(all(samples == sample_sets[0] for samples in sample_sets[1:]), "sample matrix differs")

    selected_rank = int(failed["diagnosis"]["selected_rank_under_registered_ordering"])
    deployment_id = str(failed["diagnosis"]["deployment_candidate_id_under_registered_ordering"])
    output: list[dict[str, Any]] = []
    safe_sets: dict[str, set[str]] = {}
    for candidate_id, rows in sorted(
        grouped.items(), key=lambda item: (int(item[1][0]["rank"]), int(item[1][0]["seed"]))
    ):
        rank = int(rows[0]["rank"])
        seed = int(rows[0]["seed"])
        _require(all(int(row["rank"]) == rank for row in rows), "candidate rank changed")
        _require(all(int(row["seed"]) == seed for row in rows), "candidate seed changed")
        aggregate = _aggregate(rows)
        receipt = receipt_candidates[candidate_id]
        comparisons = {
            "task_preservation": "task_score",
            "greedy_agreement": "greedy_agreement",
            "perplexity_drift_pct": "perplexity_drift_pct",
            "oracle_safe_coverage": "oracle_safe_coverage",
            "p50_transform_ms": "p50_transform_ms",
            "p95_transform_ms": "p95_transform_ms",
        }
        for derived_key, receipt_key in comparisons.items():
            _assert_close(
                float(aggregate[derived_key]),
                float(receipt[receipt_key]),
                f"{candidate_id} {derived_key}",
            )
        _require(aggregate["safe_count"] == receipt["safe_count"], "candidate safe count differs")
        training = fit_candidates[candidate_id]["metrics"]
        row = {
            "candidate_id": candidate_id,
            "rank": rank,
            "seed": seed,
            "is_deployment_seed": seed == DEPLOYMENT_SEED,
            "is_selected_rank": rank == selected_rank,
            "is_deployment_candidate": candidate_id == deployment_id,
            "training_total_loss": training["total"],
            "training_samples": training["samples"],
            "optimizer_steps": training["optimizer_steps"],
            "task_preservation": aggregate["task_preservation"],
            "greedy_agreement": aggregate["greedy_agreement"],
            "perplexity_drift_pct": aggregate["perplexity_drift_pct"],
            "safe_count": aggregate["safe_count"],
            "sample_count": aggregate["sample_count"],
            "oracle_safe_coverage": aggregate["oracle_safe_coverage"],
            "p50_transform_ms": aggregate["p50_transform_ms"],
            "p95_transform_ms": aggregate["p95_transform_ms"],
            "coverage_gate": COVERAGE_GATE,
            "gate_passed": aggregate["oracle_safe_coverage"] >= COVERAGE_GATE,
        }
        output.append(row)
        safe_sets[candidate_id] = {
            str(measurement["sample_id"])
            for measurement in rows
            if _row_metrics(measurement)["safe"]
        }
    _require(sum(bool(row["is_deployment_candidate"]) for row in output) == 1, "bad deployment id")
    return output, dict(grouped), safe_sets, selected_rank, deployment_id


def _rank_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    safe_sets: Mapping[str, set[str]],
    failed: Mapping[str, Any],
    selected_rank: int,
) -> list[dict[str, Any]]:
    receipt_ranks = {int(item["rank"]): item for item in failed["rank_aggregates"]}
    output: list[dict[str, Any]] = []
    for rank in EXPECTED_RANKS:
        candidates = [row for row in candidate_rows if int(row["rank"]) == rank]
        ids = [str(row["candidate_id"]) for row in candidates]
        _require(len(ids) == 3, f"rank {rank} does not contain three seeds")

        candidate_values = {
            field: [float(row[field]) for row in candidates]
            for field in (
                "task_preservation",
                "oracle_safe_coverage",
                "greedy_agreement",
                "p95_transform_ms",
            )
        }

        task_mean = statistics.fmean(candidate_values["task_preservation"])
        task_std = statistics.pstdev(candidate_values["task_preservation"])
        coverage_mean = statistics.fmean(candidate_values["oracle_safe_coverage"])
        coverage_std = statistics.pstdev(candidate_values["oracle_safe_coverage"])
        greedy_mean = statistics.fmean(candidate_values["greedy_agreement"])
        greedy_std = statistics.pstdev(candidate_values["greedy_agreement"])
        p95_mean = statistics.fmean(candidate_values["p95_transform_ms"])
        p95_std = statistics.pstdev(candidate_values["p95_transform_ms"])
        union = set().union(*(safe_sets[candidate_id] for candidate_id in ids))
        intersection = set.intersection(*(safe_sets[candidate_id] for candidate_id in ids))
        row = {
            "rank": rank,
            "seed_count": len(candidates),
            "mean_task_preservation": task_mean,
            "std_task_preservation": task_std,
            "mean_oracle_safe_coverage": coverage_mean,
            "std_oracle_safe_coverage": coverage_std,
            "mean_greedy_agreement": greedy_mean,
            "std_greedy_agreement": greedy_std,
            "mean_p95_transform_ms": p95_mean,
            "std_p95_transform_ms": p95_std,
            "any_seed_safe_count": len(union),
            "any_seed_safe_coverage": len(union) / 1024,
            "all_seeds_safe_count": len(intersection),
            "all_seeds_safe_coverage": len(intersection) / 1024,
            "is_selected_rank": rank == selected_rank,
        }
        receipt = receipt_ranks[rank]
        for field in (
            "mean_oracle_safe_coverage",
            "std_oracle_safe_coverage",
            "mean_greedy_agreement",
            "std_greedy_agreement",
            "mean_p95_transform_ms",
            "std_p95_transform_ms",
        ):
            _assert_close(float(row[field]), float(receipt[field]), f"rank {rank} {field}")
        _assert_close(task_mean, float(receipt["mean_task_score"]), f"rank {rank} mean task")
        _assert_close(task_std, float(receipt["std_task_score"]), f"rank {rank} std task")
        output.append(row)
    ranked = sorted(
        output,
        key=lambda row: (
            float(row["mean_task_preservation"]),
            float(row["mean_oracle_safe_coverage"]),
            float(row["mean_greedy_agreement"]),
            -float(row["mean_p95_transform_ms"]),
        ),
        reverse=True,
    )
    _require(int(ranked[0]["rank"]) == selected_rank, "registered rank ordering does not reproduce")
    return output


def _group_table_rows(
    deployment_rows: Sequence[Mapping[str, Any]],
    group_field: str,
    groups: Sequence[Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = [row for row in deployment_rows if _row_metrics(row)[group_field] == group]
        aggregate = _aggregate(selected)
        base = {
            "sample_count": aggregate["sample_count"],
            "safe_count": aggregate["safe_count"],
            "oracle_safe_coverage": aggregate["oracle_safe_coverage"],
            "task_preservation": aggregate["task_preservation"],
            "greedy_agreement": aggregate["greedy_agreement"],
            "perplexity_drift_pct": aggregate["perplexity_drift_pct"],
            "native_pass_to_bridge_fail_count": aggregate["task_regression_count"],
            "greedy_failure_count": aggregate["greedy_failure_count"],
            "drift_failure_count": aggregate["drift_failure_count"],
            "unsafe_count": aggregate["unsafe_count"],
        }
        if group_field == "task":
            base = {
                "task": group,
                "datasets": ";".join(TASK_DATASETS[str(group)]),
                **base,
            }
        else:
            base = {"token_bucket": group, **base}
        output.append(base)
    return output


def _failure_rows(deployment_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        (False, False, False): "safe",
        (False, False, True): "drift_only",
        (False, True, False): "greedy_only",
        (False, True, True): "greedy_and_drift",
        (True, False, False): "task_regression_only",
        (True, False, True): "task_regression_and_drift",
        (True, True, False): "task_regression_and_greedy",
        (True, True, True): "task_regression_and_greedy_and_drift",
    }
    counts: Counter[tuple[bool, bool, bool]] = Counter()
    for row in deployment_rows:
        metrics = _row_metrics(row)
        counts[
            (
                bool(metrics["task_regression"]),
                bool(metrics["greedy_failure"]),
                bool(metrics["drift_failure"]),
            )
        ] += 1
    return [
        {
            "reason_combination": label,
            "task_regression": key[0],
            "greedy_failure": key[1],
            "drift_failure": key[2],
            "is_unsafe": any(key),
            "count": counts[key],
            "fraction": counts[key] / len(deployment_rows),
        }
        for key, label in labels.items()
    ]


def _safe_set_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    safe_sets: Mapping[str, set[str]],
    deployment_id: str,
) -> list[dict[str, Any]]:
    output = [
        {
            "choice_set": "registered_deployment_candidate",
            "selection_mode": "fixed_source_only_identity",
            "candidate_count": 1,
            "safe_count": len(safe_sets[deployment_id]),
            "coverage": len(safe_sets[deployment_id]) / 1024,
            "deployable": True,
        }
    ]
    for row in rank_rows:
        rank = int(row["rank"])
        output.append(
            {
                "choice_set": f"rank_{rank}_any_seed",
                "selection_mode": "post_hoc_per_prompt_target_oracle",
                "candidate_count": 3,
                "safe_count": row["any_seed_safe_count"],
                "coverage": row["any_seed_safe_coverage"],
                "deployable": False,
            }
        )
    all_ids = [str(row["candidate_id"]) for row in candidate_rows]
    union = set().union(*(safe_sets[candidate_id] for candidate_id in all_ids))
    intersection = set.intersection(*(safe_sets[candidate_id] for candidate_id in all_ids))
    output.extend(
        [
            {
                "choice_set": "all_nine_any_candidate",
                "selection_mode": "post_hoc_per_prompt_target_oracle",
                "candidate_count": 9,
                "safe_count": len(union),
                "coverage": len(union) / 1024,
                "deployable": False,
            },
            {
                "choice_set": "all_nine_intersection",
                "selection_mode": "safe_under_every_candidate",
                "candidate_count": 9,
                "safe_count": len(intersection),
                "coverage": len(intersection) / 1024,
                "deployable": False,
            },
        ]
    )
    return output


def _crosscheck_diagnostic(
    diagnostic: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    bucket_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    safe_set_rows: Sequence[Mapping[str, Any]],
) -> None:
    deployment = next(row for row in candidate_rows if row["is_deployment_candidate"])
    recorded_deployment = diagnostic["deployment_candidate"]
    for derived, recorded in (
        (deployment["task_preservation"], recorded_deployment["task_preservation"]),
        (deployment["greedy_agreement"], recorded_deployment["greedy_agreement"]),
        (deployment["perplexity_drift_pct"], recorded_deployment["perplexity_drift_pct"]),
        (deployment["oracle_safe_coverage"], recorded_deployment["oracle_safe_coverage"]),
    ):
        _assert_close(float(derived), float(recorded), "deployment diagnostic")

    for row in rank_rows:
        recorded = diagnostic["candidate_safe_set_analysis"]["per_rank"][str(row["rank"])]
        _require(
            row["any_seed_safe_count"] == recorded["any_seed_safe_count"],
            "rank union differs",
        )
        _require(
            row["all_seeds_safe_count"] == recorded["all_three_seeds_safe_count"],
            "rank intersection differs",
        )

    for row in task_rows:
        recorded = diagnostic["deployment_breakdown"]["by_task"][str(row["task"])]
        _require(row["safe_count"] == recorded["safe_count"], "task safe count differs")
        _assert_close(
            float(row["oracle_safe_coverage"]),
            float(recorded["oracle_safe_coverage"]),
            f"task {row['task']} coverage",
        )

    for row in bucket_rows:
        recorded = diagnostic["deployment_breakdown"]["by_token_bucket"][str(row["token_bucket"])]
        _require(row["safe_count"] == recorded["safe_count"], "bucket safe count differs")
        for field, recorded_field in (
            ("oracle_safe_coverage", "oracle_safe_coverage"),
            ("greedy_agreement", "greedy_agreement"),
            ("perplexity_drift_pct", "aggregate_perplexity_drift_pct"),
        ):
            _assert_close(
                float(row[field]),
                float(recorded[recorded_field]),
                f"bucket {row['token_bucket']}",
            )

    nonzero_failures = {
        row["reason_combination"]: row["count"] for row in failure_rows if row["count"]
    }
    _require(
        nonzero_failures == diagnostic["failure_modes"]["failure_reason_combinations"],
        "failure overlap differs",
    )
    all_nine = next(row for row in safe_set_rows if row["choice_set"] == "all_nine_any_candidate")
    recorded_safe_sets = diagnostic["candidate_safe_set_analysis"]
    _require(all_nine["safe_count"] == recorded_safe_sets["any_nine_safe_count"], "union differs")


def _source_object(path: Path, raw: bytes, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }


def _build(report_path: Path) -> dict[str, bytes]:
    report_path = report_path.resolve()
    _reject_sealed_path(report_path)
    report, report_raw = _load_json(report_path)
    fit, fit_raw = _load_json(FIT_RECEIPT)
    failed, failed_raw = _load_json(FAILED_RECEIPT)
    diagnostic, diagnostic_raw = _load_json(DIAGNOSTIC)
    _validate_inputs(report, report_raw, fit, failed, diagnostic)

    candidate_rows, grouped, safe_sets, selected_rank, deployment_id = _candidate_rows(
        report, fit, failed
    )
    rank_rows = _rank_rows(candidate_rows, safe_sets, failed, selected_rank)
    deployment_rows = grouped[deployment_id]
    task_rows = _group_table_rows(deployment_rows, "task", tuple(TASK_DATASETS))
    bucket_rows = _group_table_rows(deployment_rows, "bucket", (128, 512, 2048, 8192))
    failure_rows = _failure_rows(deployment_rows)
    safe_set_rows = _safe_set_rows(candidate_rows, rank_rows, safe_sets, deployment_id)
    _crosscheck_diagnostic(
        diagnostic,
        candidate_rows,
        rank_rows,
        task_rows,
        bucket_rows,
        failure_rows,
        safe_set_rows,
    )

    artifacts = {
        ARCHIVE_NAME: _gzip_bytes(report_raw),
        CANDIDATE_TABLE_NAME: _csv_bytes(tuple(candidate_rows[0]), candidate_rows),
        RANK_TABLE_NAME: _csv_bytes(tuple(rank_rows[0]), rank_rows),
        TASK_TABLE_NAME: _csv_bytes(tuple(task_rows[0]), task_rows),
        BUCKET_TABLE_NAME: _csv_bytes(tuple(bucket_rows[0]), bucket_rows),
        FAILURE_TABLE_NAME: _csv_bytes(tuple(failure_rows[0]), failure_rows),
        SAFE_SET_TABLE_NAME: _csv_bytes(tuple(safe_set_rows[0]), safe_set_rows),
    }
    roles = {
        ARCHIVE_NAME: "canonical_compressed_method_dev_report",
        CANDIDATE_TABLE_NAME: "all_registered_candidate_metrics",
        RANK_TABLE_NAME: "registered_rank_aggregation_and_safe_sets",
        TASK_TABLE_NAME: "deployment_candidate_task_breakdown",
        BUCKET_TABLE_NAME: "deployment_candidate_token_bucket_breakdown",
        FAILURE_TABLE_NAME: "deployment_candidate_failure_intersections",
        SAFE_SET_TABLE_NAME: "fixed_and_post_hoc_candidate_choice_sets",
    }
    artifact_manifest = []
    for name, raw in artifacts.items():
        entry = {
            "role": roles[name],
            "path": (OUTPUT_DIR / name).relative_to(ROOT).as_posix(),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
            "media_type": "application/gzip" if name.endswith(".gz") else "text/csv",
        }
        if name == ARCHIVE_NAME:
            entry["uncompressed_sha256"] = _sha256(report_raw)
            entry["uncompressed_size_bytes"] = len(report_raw)
        artifact_manifest.append(entry)

    deployment = next(row for row in candidate_rows if row["is_deployment_candidate"])
    all_nine = next(row for row in safe_set_rows if row["choice_set"] == "all_nine_any_candidate")
    manifest = {
        "schema_version": "goldenexperience.publication_v5_method_dev_evidence.v1",
        "authority": "curated_terminal_negative_result_evidence_not_runtime_approval",
        "direction": report["direction"],
        "pipeline_id": report["pipeline_id"],
        "code_sha256": report["code_sha256"],
        "protocol_disposition": failed["protocol_disposition"],
        "counts": {
            "candidate_count": len(candidate_rows),
            "measurement_count": len(report["measurements"]),
            "sample_count": len({row["sample_id"] for row in report["measurements"]}),
        },
        "thresholds": {
            "min_greedy_agreement": GREEDY_GATE,
            "max_perplexity_drift_pct": DRIFT_GATE,
            "native_pass_to_bridge_fail_is_unsafe": True,
            "min_oracle_safe_coverage": COVERAGE_GATE,
        },
        "registered_deployment": {
            "candidate_id": deployment_id,
            "rank": selected_rank,
            "seed": DEPLOYMENT_SEED,
            "task_preservation": deployment["task_preservation"],
            "greedy_agreement": deployment["greedy_agreement"],
            "perplexity_drift_pct": deployment["perplexity_drift_pct"],
            "safe_count": deployment["safe_count"],
            "oracle_safe_coverage": deployment["oracle_safe_coverage"],
            "gate_passed": deployment["gate_passed"],
        },
        "post_hoc_all_candidate_oracle": {
            "safe_count": all_nine["safe_count"],
            "oracle_safe_coverage": all_nine["coverage"],
            "deployable": False,
        },
        "source_objects": [
            _source_object(report_path, report_raw, "uncompressed_method_dev_report"),
            _source_object(FIT_RECEIPT, fit_raw, "verified_fit_stage_receipt"),
            _source_object(FAILED_RECEIPT, failed_raw, "failed_method_dev_stage_receipt"),
            _source_object(DIAGNOSTIC, diagnostic_raw, "post_failure_mechanism_diagnostic"),
        ],
        "artifacts": artifact_manifest,
        "reproduction": {
            "command": "python3 paper/tools/build_method_dev_evidence.py",
            "check_command": "python3 paper/tools/build_method_dev_evidence.py --check",
            "gzip": "RFC 1952, DEFLATE level 9, mtime 0, empty original filename",
            "sealed_payload_access": False,
        },
    }
    artifacts[MANIFEST_NAME] = _canonical_json(manifest)
    return artifacts


def _write_or_check(artifacts: Mapping[str, bytes], *, check: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name, expected in artifacts.items():
        path = OUTPUT_DIR / name
        _reject_sealed_path(path)
        if check:
            if not path.is_file():
                failures.append(f"missing {path.relative_to(ROOT)}")
                continue
            actual = path.read_bytes()
            if actual != expected:
                failures.append(f"content differs for {path.relative_to(ROOT)}")
        else:
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_bytes(expected)
            os.replace(temporary, path)
        print(f"{_sha256(expected)}  {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> None:
    args = _parse_args()
    artifacts = _build(args.report)
    _write_or_check(artifacts, check=args.check)


if __name__ == "__main__":
    main()
