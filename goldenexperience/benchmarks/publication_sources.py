"""Audited source locks and adapters for the publication benchmark."""

from __future__ import annotations

import ast
import csv
import hashlib
import heapq
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    REQUIRED_DATASETS,
    DatasetSource,
)
from goldenexperience.benchmarks.publication_eval import validate_publication_evaluation
from goldenexperience.size_variant.cached_kv_manifest import sha256_file

PUBLICATION_SOURCE_LOCK_SCHEMA = "goldenexperience.publication_source_lock.v1"
SOURCE_MERKLE_SCHEMA = "goldenexperience.publication_source_merkle.v1"
SOURCE_ADAPTERS = {
    "bfcl": "bfcl_v4",
    "burstgpt": "burstgpt",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "longbench_hotpotqa": "longbench",
    "longbench_multifieldqa": "longbench",
    "longbench_qasper": "longbench",
    "math": "math",
    "mbpp": "mbpp",
    "sharegpt": "sharegpt",
}
EXPECTED_FILE_ROLES = {
    "bfcl": frozenset({"questions", "answers"}),
    "burstgpt": frozenset({"data"}),
    "gsm8k": frozenset({"train", "test"}),
    "humaneval": frozenset({"data"}),
    "longbench_hotpotqa": frozenset({"data"}),
    "longbench_multifieldqa": frozenset({"data"}),
    "longbench_qasper": frozenset({"data"}),
    "math": frozenset({"data"}),
    "mbpp": frozenset({"data"}),
    "sharegpt": frozenset({"data"}),
}
_LONG_DATASET_NAMES = {
    "longbench_hotpotqa": "hotpotqa",
    "longbench_multifieldqa": "multifieldqa_en",
    "longbench_qasper": "qasper",
}
_MATH_CATEGORIES = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
EXPECTED_FILE_ROLES["math"] = frozenset(
    f"{category}_{split}" for category in _MATH_CATEGORIES for split in ("train", "test")
)
_MATH_BOX = re.compile(r"\\(?:boxed|fbox)\{")


class PublicationSourceError(RuntimeError):
    """Raised when a source lock or source dataset cannot be reproduced."""


@dataclass(frozen=True)
class SourceFileLock:
    role: str
    path: str
    sha256: str
    size_bytes: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.role or not self.path:
            errors.append("source file role and path are required")
        if not _is_sha256(self.sha256):
            errors.append(f"source file {self.role!r} hash is invalid")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            errors.append(f"source file {self.role!r} size is invalid")
        return errors


@dataclass(frozen=True)
class LockedDatasetSource:
    source: DatasetSource
    adapter: str
    files: tuple[SourceFileLock, ...]

    def validate(self) -> list[str]:
        errors = self.source.validate()
        expected_adapter = SOURCE_ADAPTERS.get(self.source.dataset_id)
        if self.adapter != expected_adapter:
            errors.append(f"source {self.source.dataset_id!r} uses the wrong adapter")
        roles = [item.role for item in self.files]
        if len(roles) != len(set(roles)):
            errors.append(f"source {self.source.dataset_id!r} has duplicate file roles")
        expected_roles = EXPECTED_FILE_ROLES.get(self.source.dataset_id, frozenset())
        if set(roles) != expected_roles:
            errors.append(f"source {self.source.dataset_id!r} file roles are incomplete")
        for item in self.files:
            errors.extend(item.validate())
        if self.source.content_sha256 != source_file_merkle_sha256(self.files):
            errors.append(f"source {self.source.dataset_id!r} Merkle hash changed")
        return errors


@dataclass(frozen=True)
class PublicationSourceLock:
    sources: tuple[LockedDatasetSource, ...]
    tokenizer_sha256: str
    chat_template_sha256: str
    selection_seed: int = 17
    schema_version: str = PUBLICATION_SOURCE_LOCK_SCHEMA

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != PUBLICATION_SOURCE_LOCK_SCHEMA:
            errors.append("unsupported publication source lock schema")
        if type(self.selection_seed) is not int or self.selection_seed < 0:
            errors.append("publication source selection seed is invalid")
        if not _is_sha256(self.tokenizer_sha256) or not _is_sha256(self.chat_template_sha256):
            errors.append("publication source lock tokenizer identities are invalid")
        ids = [item.source.dataset_id for item in self.sources]
        if len(ids) != len(set(ids)):
            errors.append("publication source lock has duplicate datasets")
        if set(ids) != set(REQUIRED_DATASETS):
            errors.append("publication source lock must contain exactly the required datasets")
        for item in self.sources:
            errors.extend(item.validate())
        return errors

    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_seed": self.selection_seed,
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "sources": [
                {
                    "source": asdict(item.source),
                    "adapter": item.adapter,
                    "files": [asdict(file) for file in item.files],
                }
                for item in self.sources
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PublicationSourceLock:
        sources = []
        for raw_source in payload.get("sources", ()):
            raw_source = dict(raw_source)
            sources.append(
                LockedDatasetSource(
                    source=DatasetSource(**raw_source["source"]),
                    adapter=str(raw_source["adapter"]),
                    files=tuple(SourceFileLock(**item) for item in raw_source["files"]),
                )
            )
        return cls(
            sources=tuple(sources),
            tokenizer_sha256=str(payload.get("tokenizer_sha256", "")),
            chat_template_sha256=str(payload.get("chat_template_sha256", "")),
            selection_seed=payload.get("selection_seed", -1),
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> PublicationSourceLock:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise PublicationSourceError("publication source lock must be a JSON object")
            lock = cls.from_dict(raw)
        except PublicationSourceError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise PublicationSourceError("publication source lock is malformed") from exc
        errors = lock.validate()
        if errors:
            raise PublicationSourceError("; ".join(errors))
        return lock


@dataclass(frozen=True)
class CanonicalSourceExample:
    dataset_id: str
    row_id: str
    source_split: str
    query: str
    context: str
    reference: Any
    evaluation: Mapping[str, Any]
    task: str
    demonstration: str
    source_role: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.dataset_id not in REQUIRED_DATASETS:
            errors.append("canonical source example dataset is unknown")
        if not all((self.row_id, self.source_split, self.query, self.task, self.source_role)):
            errors.append(f"canonical source example {self.row_id!r} is incomplete")
        if self.dataset_id not in {"sharegpt", "burstgpt"} and self.reference is None:
            errors.append(f"semantic source example {self.row_id!r} lacks a reference")
        try:
            json.dumps(
                {
                    "reference": self.reference,
                    "evaluation": dict(self.evaluation),
                },
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"canonical source example {self.row_id!r} is not JSON-safe: {exc}")
        return errors


@dataclass(frozen=True)
class BurstWorkloadRow:
    row_id: str
    timestamp: float
    model: str
    request_tokens: int
    response_tokens: int
    log_type: str
    token_bucket: int


@dataclass(frozen=True)
class ResolvedSourceFile:
    dataset_id: str
    role: str
    logical_path: str
    path: Path
    sha256: str
    size_bytes: int
    signature: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class AuditedPublicationSources:
    lock: PublicationSourceLock
    files: tuple[ResolvedSourceFile, ...]

    def file(self, dataset_id: str, role: str) -> Path:
        for item in self.files:
            if item.dataset_id == dataset_id and item.role == role:
                return item.path
        raise PublicationSourceError(f"source file {dataset_id}:{role} is not registered")

    def verify_unchanged(self) -> None:
        for item in self.files:
            try:
                stat = item.path.stat()
            except OSError as exc:
                raise PublicationSourceError("publication source disappeared during build") from exc
            if item.path.is_symlink() or not item.path.is_file():
                raise PublicationSourceError("publication source became a non-regular file")
            if _signature(stat) != item.signature:
                raise PublicationSourceError(
                    "publication source stat identity changed during build"
                )
            if sha256_file(item.path) != item.sha256:
                raise PublicationSourceError("publication source content changed during build")

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "goldenexperience.publication_source_manifest.v1",
            "source_lock_sha256": self.lock.content_sha256(),
            "source_hash_algorithm": SOURCE_MERKLE_SCHEMA,
            "selection_seed": self.lock.selection_seed,
            "tokenizer_sha256": self.lock.tokenizer_sha256,
            "chat_template_sha256": self.lock.chat_template_sha256,
            "sources": [asdict(item.source) for item in self.lock.sources],
            "files": [
                {
                    "dataset_id": item.dataset_id,
                    "role": item.role,
                    "logical_path": item.logical_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
        }


@dataclass(frozen=True)
class LoadedPublicationSources:
    examples: Mapping[str, tuple[CanonicalSourceExample, ...]]
    burst_rows: tuple[BurstWorkloadRow, ...]

    def validate(self) -> list[str]:
        errors: list[str] = []
        if set(self.examples) != set(REQUIRED_DATASETS) - {"burstgpt"}:
            errors.append("loaded publication source set is incomplete")
        for dataset_id, rows in self.examples.items():
            if not rows:
                errors.append(f"loaded publication source {dataset_id!r} is empty")
            seen: set[str] = set()
            for row in rows:
                errors.extend(row.validate())
                if row.dataset_id != dataset_id:
                    errors.append(f"loaded publication source {dataset_id!r} is mixed")
                if row.row_id in seen:
                    errors.append(f"loaded publication source {dataset_id!r} has duplicate rows")
                seen.add(row.row_id)
        if not self.burst_rows:
            errors.append("loaded BurstGPT source is empty")
        return errors


def source_file_merkle_sha256(files: Sequence[SourceFileLock]) -> str:
    payload = {
        "schema_version": SOURCE_MERKLE_SCHEMA,
        "files": [
            {
                "role": item.role,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in sorted(files, key=lambda value: value.role)
        ],
    }
    return _canonical_sha256(payload)


def audit_publication_sources(
    lock: PublicationSourceLock,
    *,
    source_root: str | Path,
    path_overrides: Mapping[tuple[str, str], str | Path] | None = None,
) -> AuditedPublicationSources:
    """Verify every byte before any dataset parser is allowed to consume it."""

    root = Path(source_root).resolve()
    overrides = dict(path_overrides or {})
    known_keys = {
        (source.source.dataset_id, file.role) for source in lock.sources for file in source.files
    }
    unknown = set(overrides) - known_keys
    if unknown:
        raise PublicationSourceError(f"source path overrides are unknown: {sorted(unknown)}")
    resolved: list[ResolvedSourceFile] = []
    for dataset in sorted(lock.sources, key=lambda item: item.source.dataset_id):
        for file in sorted(dataset.files, key=lambda item: item.role):
            override = overrides.get((dataset.source.dataset_id, file.role))
            candidate = Path(override) if override is not None else root / file.path
            try:
                if candidate.is_symlink():
                    raise PublicationSourceError("publication source cannot be a symbolic link")
                path = candidate.resolve()
                if not path.is_file():
                    raise PublicationSourceError("publication source must be a regular file")
                before = path.stat()
                observed = sha256_file(path)
                after = path.stat()
            except PublicationSourceError:
                raise
            except OSError as exc:
                raise PublicationSourceError(
                    f"publication source {dataset.source.dataset_id}:{file.role} is unavailable"
                ) from exc
            if _signature(before) != _signature(after):
                raise PublicationSourceError("publication source changed while hashing")
            if after.st_size != file.size_bytes or observed != file.sha256:
                raise PublicationSourceError(
                    f"publication source {dataset.source.dataset_id}:{file.role} checksum mismatch"
                )
            resolved.append(
                ResolvedSourceFile(
                    dataset_id=dataset.source.dataset_id,
                    role=file.role,
                    logical_path=file.path,
                    path=path,
                    sha256=file.sha256,
                    size_bytes=file.size_bytes,
                    signature=_signature(after),
                )
            )
    return AuditedPublicationSources(lock=lock, files=tuple(resolved))


def load_publication_sources(audited: AuditedPublicationSources) -> LoadedPublicationSources:
    seed = audited.lock.selection_seed
    examples: dict[str, tuple[CanonicalSourceExample, ...]] = {}
    for dataset in sorted(audited.lock.sources, key=lambda item: item.source.dataset_id):
        dataset_id = dataset.source.dataset_id
        if dataset_id == "burstgpt":
            continue
        loader = {
            "bfcl_v4": _load_bfcl,
            "gsm8k": _load_gsm8k,
            "humaneval": _load_humaneval,
            "longbench": _load_longbench,
            "math": _load_math,
            "mbpp": _load_mbpp,
            "sharegpt": _load_sharegpt,
        }[dataset.adapter]
        rows = tuple(loader(audited, dataset_id, seed))
        examples[dataset_id] = rows
    burst_rows = tuple(_load_burstgpt(audited.file("burstgpt", "data"), seed))
    loaded = LoadedPublicationSources(examples=examples, burst_rows=burst_rows)
    errors = loaded.validate()
    if errors:
        raise PublicationSourceError("; ".join(errors))
    audited.verify_unchanged()
    return loaded


def _load_longbench(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    expected_name = _LONG_DATASET_NAMES[dataset_id]
    for line_number, row in _jsonl(audited.file(dataset_id, "data")):
        if row.get("dataset") != expected_name:
            raise PublicationSourceError(f"{dataset_id} row has the wrong dataset tag")
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers:
            raise PublicationSourceError(f"{dataset_id} row has no answers")
        yield CanonicalSourceExample(
            dataset_id=dataset_id,
            row_id=str(row.get("_id") or f"line-{line_number}"),
            source_split="test",
            query=_required_text(row, "input"),
            context=_required_text(row, "context"),
            reference=answers,
            evaluation={"metric": "token_f1", "pass_threshold": 0.5},
            task="long_context_qa",
            demonstration="",
            source_role="data",
        )


def _load_gsm8k(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    for split in ("train", "test"):
        for line_number, row in _jsonl(audited.file(dataset_id, split)):
            answer = _required_text(row, "answer")
            marker = answer.rfind("####")
            if marker < 0 or not answer[marker + 4 :].strip():
                raise PublicationSourceError("GSM8K answer lacks its canonical final marker")
            final = answer[marker + 4 :].strip()
            question = _required_text(row, "question")
            yield CanonicalSourceExample(
                dataset_id=dataset_id,
                row_id=f"{split}-{line_number:05d}",
                source_split=split,
                query=question,
                context="",
                reference=final,
                evaluation={
                    "metric": "numeric_exact",
                    "absolute_tolerance": 1e-9,
                    "relative_tolerance": 1e-9,
                },
                task="grade_school_math",
                demonstration=f"Problem: {question}\nSolution: {answer}\n",
                source_role=split,
            )


def _load_math(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PublicationSourceError(
            "MATH Parquet sources require the publication optional dependencies"
        ) from exc
    for category in _MATH_CATEGORIES:
        for split in ("train", "test"):
            role = f"{category}_{split}"
            try:
                rows = parquet.read_table(audited.file(dataset_id, role)).to_pylist()
            except Exception as exc:
                raise PublicationSourceError(f"MATH source role {role!r} is malformed") from exc
            for index, raw in enumerate(rows):
                if not isinstance(raw, dict):
                    raise PublicationSourceError("MATH row must be a JSON-compatible object")
                row_id = f"{category}/{split}/{index:05d}"
                problem = _required_text(raw, "problem")
                solution = _required_text(raw, "solution")
                answer = _last_math_box(solution)
                if answer is None:
                    # The official snapshot has a few rows without a deterministic final marker.
                    continue
                evaluation = {"metric": "math_exact"}
                if validate_publication_evaluation(answer, evaluation):
                    continue
                yield CanonicalSourceExample(
                    dataset_id=dataset_id,
                    row_id=row_id,
                    source_split=split,
                    query=problem,
                    context="",
                    reference=answer,
                    evaluation=evaluation,
                    task="competition_math",
                    demonstration=f"Problem: {problem}\nSolution: {solution}\n",
                    source_role=role,
                )


def _load_humaneval(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    for line_number, row in _jsonl(audited.file(dataset_id, "data")):
        prompt = _required_text(row, "prompt")
        entry_point = _required_text(row, "entry_point")
        test = _required_text(row, "test")
        solution = _required_text(row, "canonical_solution")
        yield CanonicalSourceExample(
            dataset_id=dataset_id,
            row_id=str(row.get("task_id") or f"line-{line_number}"),
            source_split="test",
            query=prompt,
            context="",
            reference={"entry_point": entry_point, "test_code": test, "test_mode": "check"},
            evaluation={"metric": "python_tests"},
            task="python_code_generation",
            demonstration=f"Task:\n{prompt}\nReference implementation:\n{prompt}{solution}\n",
            source_role="data",
        )


def _load_mbpp(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    for line_number, row in _jsonl(audited.file(dataset_id, "data")):
        prompt = _required_text(row, "text")
        code = _required_text(row, "code")
        entry_point = _first_function_name(code)
        raw_tests = row.get("test_list")
        if (
            not isinstance(raw_tests, list)
            or not raw_tests
            or not all(isinstance(item, str) and item.strip() for item in raw_tests)
        ):
            raise PublicationSourceError("MBPP tests are malformed")
        setup = row.get("test_setup_code", "")
        if not isinstance(setup, str):
            raise PublicationSourceError("MBPP test setup is malformed")
        test_code = "\n".join(item for item in (setup, *raw_tests) if item.strip())
        yield CanonicalSourceExample(
            dataset_id=dataset_id,
            row_id=str(row.get("task_id") or f"line-{line_number}"),
            source_split="benchmark",
            query=prompt,
            context="",
            reference={
                "entry_point": entry_point,
                "test_code": test_code,
                "test_mode": "exec",
            },
            evaluation={"metric": "python_tests"},
            task="python_code_generation",
            demonstration=f"Task: {prompt}\nReference implementation:\n{code}\n",
            source_role="data",
        )


def _load_bfcl(
    audited: AuditedPublicationSources,
    dataset_id: str,
    _seed: int,
) -> Iterable[CanonicalSourceExample]:
    answers = {str(row.get("id")): row for _, row in _jsonl(audited.file(dataset_id, "answers"))}
    question_count = 0
    for line_number, row in _jsonl(audited.file(dataset_id, "questions")):
        question_count += 1
        row_id = str(row.get("id") or f"line-{line_number}")
        answer = answers.get(row_id)
        if answer is None:
            raise PublicationSourceError(f"BFCL row {row_id!r} lacks a ground truth")
        query = _bfcl_query(row.get("question"))
        functions = row.get("function")
        if not isinstance(functions, list) or not functions:
            raise PublicationSourceError(f"BFCL row {row_id!r} has no function declarations")
        reference = _bfcl_reference(answer.get("ground_truth"))
        functions_json = json.dumps(
            functions, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        yield CanonicalSourceExample(
            dataset_id=dataset_id,
            row_id=row_id,
            source_split="test",
            query=query,
            context=functions_json,
            reference=reference,
            evaluation={"metric": "function_call"},
            task="function_calling",
            demonstration=f"Available functions: {functions_json}\n",
            source_role="questions+answers",
        )
    if len(answers) != question_count:
        raise PublicationSourceError("BFCL question and answer counts differ")


def _load_sharegpt(
    audited: AuditedPublicationSources,
    dataset_id: str,
    seed: int,
) -> Iterable[CanonicalSourceExample]:
    selected: list[tuple[int, int, CanonicalSourceExample]] = []
    for index, raw in enumerate(_json_array(audited.file(dataset_id, "data")), start=1):
        row_id = str(raw.get("id") or f"row-{index}")
        conversations = raw.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            continue
        turns: list[tuple[str, str]] = []
        for turn in conversations:
            if not isinstance(turn, dict):
                turns = []
                break
            role = turn.get("from")
            value = turn.get("value")
            if role not in {"human", "gpt"} or not isinstance(value, str) or not value.strip():
                turns = []
                break
            turns.append((role, value.strip()))
        human_turns = [value for role, value in turns if role == "human"]
        if not turns or not human_turns:
            continue
        demonstration = "\n".join(f"{role.title()}: {value}" for role, value in turns)
        example = CanonicalSourceExample(
            dataset_id=dataset_id,
            row_id=row_id,
            source_split="trace",
            query=human_turns[-1],
            context="",
            reference=None,
            evaluation={},
            task="multi_turn_chat_trace",
            demonstration=demonstration,
            source_role="data",
        )
        _keep_smallest(selected, 4096, _stable_int(seed, dataset_id, row_id), index, example)
    return tuple(item[2] for item in sorted(selected, key=lambda item: (-item[0], item[1])))


def _load_burstgpt(path: Path, seed: int) -> Iterable[BurstWorkloadRow]:
    selected: dict[int, list[tuple[int, int, BurstWorkloadRow]]] = {
        bucket: [] for bucket in PREFIX_BUCKETS
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {
                "Timestamp",
                "Model",
                "Request tokens",
                "Response tokens",
                "Total tokens",
                "Log Type",
            }
            if set(reader.fieldnames or ()) != expected:
                raise PublicationSourceError("BurstGPT columns differ from the locked schema")
            for line_number, row in enumerate(reader, start=2):
                try:
                    timestamp = float(row["Timestamp"])
                    request_tokens = int(row["Request tokens"])
                    response_tokens = int(row["Response tokens"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise PublicationSourceError(
                        f"BurstGPT line {line_number} is malformed"
                    ) from exc
                if (
                    not math.isfinite(timestamp)
                    or timestamp < 0
                    or request_tokens <= 0
                    or response_tokens <= 0
                ):
                    continue
                bucket = min(
                    PREFIX_BUCKETS,
                    key=lambda value: abs(math.log2(request_tokens) - math.log2(value)),
                )
                row_id = f"line-{line_number:07d}"
                item = BurstWorkloadRow(
                    row_id=row_id,
                    timestamp=timestamp,
                    model=str(row["Model"]),
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    log_type=str(row["Log Type"]),
                    token_bucket=bucket,
                )
                _keep_smallest(
                    selected[bucket],
                    256,
                    _stable_int(seed, "burstgpt", row_id),
                    line_number,
                    item,
                )
    except PublicationSourceError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PublicationSourceError("BurstGPT source is unreadable") from exc
    for bucket in PREFIX_BUCKETS:
        if len(selected[bucket]) < 64:
            raise PublicationSourceError(
                f"BurstGPT has fewer than 64 usable rows for bucket {bucket}"
            )
        yield from (
            item[2] for item in sorted(selected[bucket], key=lambda value: (-value[0], value[1]))
        )


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PublicationSourceError(f"{path.name}:{line_number} is not an object")
                yield line_number, value
    except PublicationSourceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationSourceError(f"source file {path.name!r} is malformed JSONL") from exc


def _json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            while True:
                chunk = handle.read(chunk_size)
                eof = not chunk
                buffer = buffer[position:] + chunk
                position = 0
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if not started:
                        if position >= len(buffer):
                            break
                        if buffer[position] != "[":
                            raise PublicationSourceError("streamed JSON source must be an array")
                        started = True
                        position += 1
                        continue
                    while position < len(buffer) and (
                        buffer[position].isspace() or buffer[position] == ","
                    ):
                        position += 1
                    if position >= len(buffer):
                        break
                    if buffer[position] == "]":
                        finished = True
                        position += 1
                        break
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError:
                        if eof:
                            raise PublicationSourceError(
                                "streamed JSON array is truncated"
                            ) from None
                        break
                    if not isinstance(value, dict):
                        raise PublicationSourceError("streamed JSON array item must be an object")
                    position = end
                    yield value
                if finished:
                    if buffer[position:].strip() or handle.read(1):
                        raise PublicationSourceError("streamed JSON array has trailing content")
                    return
                if eof:
                    break
    except PublicationSourceError:
        raise
    except (OSError, UnicodeError) as exc:
        raise PublicationSourceError("streamed JSON source is unreadable") from exc
    raise PublicationSourceError("streamed JSON array lacks a closing bracket")


def _bfcl_query(value: Any) -> str:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise PublicationSourceError("BFCL question turns are malformed")
    messages = value[0]
    text: list[str] = []
    for item in messages:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise PublicationSourceError("BFCL question has an invalid user message")
        text.append(content)
    if not text:
        raise PublicationSourceError("BFCL question lacks a user message")
    return "\n".join(text)


def _bfcl_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        raise PublicationSourceError("BFCL ground truth is malformed")
    sequences: list[list[dict[str, Any]]] = []
    for alternative in value:
        if not isinstance(alternative, dict) or not alternative:
            raise PublicationSourceError("BFCL ground-truth alternative is malformed")
        calls = []
        for name, arguments in alternative.items():
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise PublicationSourceError("BFCL ground-truth call is malformed")
            choices = {
                key: raw if isinstance(raw, list) else [raw]
                for key, raw in arguments.items()
                if isinstance(key, str)
            }
            if len(choices) != len(arguments) or any(not raw for raw in choices.values()):
                raise PublicationSourceError("BFCL ground-truth argument choices are malformed")
            calls.append({"name": name, "arguments": choices})
        sequences.append(calls)
    return {"accepted_call_sequences": sequences}


def _first_function_name(code: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError as exc:
        raise PublicationSourceError("MBPP reference code is invalid") from exc
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise PublicationSourceError("MBPP reference code has no function definition")


def _last_math_box(solution: str) -> str | None:
    matches = list(_MATH_BOX.finditer(solution))
    for match in reversed(matches):
        depth = 1
        start = match.end()
        for index in range(start, len(solution)):
            if solution[index] == "{":
                depth += 1
            elif solution[index] == "}":
                depth -= 1
                if depth == 0:
                    return solution[start:index]
    return None


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise PublicationSourceError(f"source field {name!r} must be non-empty text")
    return value.strip()


def _keep_smallest(
    heap: list[tuple[int, int, Any]],
    limit: int,
    key: int,
    ordinal: int,
    value: Any,
) -> None:
    item = (-key, ordinal, value)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _stable_int(seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(seed.to_bytes(16, "big", signed=False))
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
