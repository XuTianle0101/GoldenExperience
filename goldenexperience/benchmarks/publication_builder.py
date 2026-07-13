"""Deterministic, leak-resistant builder for the real publication benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from goldenexperience.benchmarks.publication import (
    PREFIX_BUCKETS,
    SPLIT_COUNTS,
    TRACE_ONLY_DATASETS,
    GroupedPrefixRecord,
    PublicationBenchmarkManifest,
)
from goldenexperience.benchmarks.publication_sources import (
    AuditedPublicationSources,
    BurstWorkloadRow,
    CanonicalSourceExample,
    LoadedPublicationSources,
    PublicationSourceError,
)
from goldenexperience.runtime.cross_model_reuse import token_ids_sha256
from goldenexperience.size_variant.cached_kv_manifest import (
    chat_template_sha256,
    sha256_file,
    tokenizer_semantic_sha256,
)
from goldenexperience.size_variant.v5_collect import (
    RawBenchmarkSample,
    publication_sample_content_sha256,
)

PUBLICATION_BUILDER_ID = "goldenexperience.grouped_prefix_builder.v1"
PUBLICATION_BUILD_REPORT_SCHEMA = "goldenexperience.publication_build_report.v1"
SEMANTIC_SPLITS = tuple(name for name in SPLIT_COUNTS if name != "runtime_audit")
_LONG_DATASETS = frozenset({"longbench_hotpotqa", "longbench_multifieldqa", "longbench_qasper"})
_SHARED_PREFIX_DATASETS = frozenset({"bfcl", "gsm8k", "humaneval", "math", "mbpp"})
_MINORITY_QUOTAS = {
    "longbench_hotpotqa": {128: 6, 512: 6, 2048: 7, 8192: 5},
    "longbench_multifieldqa": {bucket: 6 for bucket in PREFIX_BUCKETS},
    "longbench_qasper": {128: 8, 512: 8, 2048: 7, 8192: 1},
    "humaneval": {bucket: 6 for bucket in PREFIX_BUCKETS},
    "mbpp": {bucket: 12 for bucket in PREFIX_BUCKETS},
    "bfcl": {bucket: 12 for bucket in PREFIX_BUCKETS},
}
_DEMO_COUNTS = {"gsm8k": 128, "math": 128, "mbpp": 256, "sharegpt": 256}


class PublicationBuildError(RuntimeError):
    """Raised when a publication split cannot be built without weakening its contract."""


@dataclass(frozen=True)
class AllocationCell:
    split: str
    token_bucket: int
    dataset_id: str
    count: int


@dataclass(frozen=True)
class PrefixRecipe:
    prefix_group_id: str
    prefix_sha256: str
    token_ids_sha256: str
    token_bucket: int
    family: str
    dataset_id: str
    source_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class PublicationBuildResult:
    output_dir: Path
    sealed_payload: Path
    manifest_path: Path
    source_manifest_path: Path
    build_report_path: Path
    manifest: PublicationBenchmarkManifest


class PublicationTokenizer:
    """Minimal tokenizer boundary with exact decode/re-encode checks."""

    def __init__(self, tokenizer: Any, *, model_path: str | Path) -> None:
        self.tokenizer = tokenizer
        self.model_path = Path(model_path).resolve()
        self.semantic_sha256 = tokenizer_semantic_sha256(self.model_path)
        self.chat_template_sha256 = chat_template_sha256(self.model_path)

    @classmethod
    def from_model(cls, model_path: str | Path) -> PublicationTokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception as exc:
            raise PublicationBuildError(
                "canonical publication tokenizer could not be loaded"
            ) from exc
        return cls(tokenizer, model_path=model_path)

    def encode(self, text: str) -> list[int]:
        try:
            values = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise PublicationBuildError("publication text could not be tokenized") from exc
        if (
            not isinstance(values, list)
            or not values
            or not all(type(item) is int for item in values)
        ):
            raise PublicationBuildError("publication tokenizer returned invalid token ids")
        return values

    def exact_prefix(self, text: str, token_bucket: int) -> tuple[str, list[int]]:
        token_ids = self.encode(text)
        if len(token_ids) < token_bucket:
            raise PublicationBuildError(
                f"source prefix has {len(token_ids)} tokens, fewer than bucket {token_bucket}"
            )
        expected = token_ids[:token_bucket]
        try:
            prefix = self.tokenizer.decode(
                expected,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise PublicationBuildError("publication prefix tokens could not be decoded") from exc
        if not isinstance(prefix, str) or not prefix or "\ufffd" in prefix:
            raise PublicationBuildError("publication prefix decode is empty or lossy")
        observed = self.encode(prefix)
        if observed != expected:
            raise PublicationBuildError("publication prefix is not decode/re-encode stable")
        return prefix, expected


def publication_allocation() -> tuple[AllocationCell, ...]:
    cells: list[AllocationCell] = []
    for split in SEMANTIC_SPLITS:
        per_bucket = SPLIT_COUNTS[split] // len(PREFIX_BUCKETS)
        if per_bucket * len(PREFIX_BUCKETS) != SPLIT_COUNTS[split]:
            raise PublicationBuildError(f"split {split!r} cannot be balanced across token buckets")
        for bucket in PREFIX_BUCKETS:
            minority = 0
            for dataset_id, quotas in _MINORITY_QUOTAS.items():
                count = quotas[bucket]
                cells.append(AllocationCell(split, bucket, dataset_id, count))
                minority += count
            core = per_bucket - minority
            if core <= 0 or core % 2:
                raise PublicationBuildError(f"split {split!r} bucket {bucket} has invalid quotas")
            cells.append(AllocationCell(split, bucket, "gsm8k", core // 2))
            cells.append(AllocationCell(split, bucket, "math", core // 2))
    runtime_per_bucket = SPLIT_COUNTS["runtime_audit"] // len(PREFIX_BUCKETS)
    if runtime_per_bucket % 2:
        raise PublicationBuildError("runtime audit cannot be balanced across trace sources")
    for bucket in PREFIX_BUCKETS:
        cells.append(AllocationCell("runtime_audit", bucket, "sharegpt", runtime_per_bucket // 2))
        cells.append(AllocationCell("runtime_audit", bucket, "burstgpt", runtime_per_bucket // 2))
    _validate_allocation(cells)
    return tuple(cells)


class PublicationDatasetBuilder:
    def __init__(
        self,
        *,
        audited_sources: AuditedPublicationSources,
        loaded_sources: LoadedPublicationSources,
        tokenizer: PublicationTokenizer,
        allocation: Sequence[AllocationCell] | None = None,
    ) -> None:
        self.audited = audited_sources
        self.loaded = loaded_sources
        self.tokenizer = tokenizer
        self.allocation = tuple(allocation or publication_allocation())
        _validate_allocation(self.allocation)
        if (
            tokenizer.semantic_sha256 != audited_sources.lock.tokenizer_sha256
            or tokenizer.chat_template_sha256 != audited_sources.lock.chat_template_sha256
        ):
            raise PublicationBuildError("publication tokenizer differs from the frozen source lock")
        self.seed = audited_sources.lock.selection_seed
        self.used_rows: set[tuple[str, str]] = set()
        self.used_suffixes: set[str] = set()
        self.records: list[GroupedPrefixRecord] = []
        self.samples: dict[str, list[RawBenchmarkSample]] = defaultdict(list)
        self.recipes: dict[str, PrefixRecipe] = {}
        self.demo_rows = self._reserve_demonstrations()
        self.shared_prefixes: dict[tuple[str, int, str], tuple[str, str]] = {}
        self.long_prefix_cache: dict[tuple[str, int, str], tuple[str, str]] = {}
        self.share_query_rows = tuple(
            item
            for item in self._ordered_examples("sharegpt", purpose="runtime-query")
            if (item.dataset_id, item.row_id) not in self.used_rows
        )

    def build(
        self,
    ) -> tuple[
        PublicationBenchmarkManifest,
        dict[str, Any],
        dict[str, Any],
    ]:
        semantic_cells = [cell for cell in self.allocation if cell.split != "runtime_audit"]
        split_order = {name: index for index, name in enumerate(SEMANTIC_SPLITS)}
        semantic_cells.sort(
            key=lambda cell: (-cell.token_bucket, split_order[cell.split], cell.dataset_id)
        )
        for cell in semantic_cells:
            self._build_semantic_cell(cell)
        for cell in (item for item in self.allocation if item.split == "runtime_audit"):
            self._build_runtime_cell(cell)
        self._validate_built_rows()
        sealed_bytes = _jsonl_bytes(
            sample.to_dict()
            for sample in sorted(
                self.samples["semantic_sealed_test"], key=lambda item: item.sample_id
            )
        )
        sources = tuple(
            item.source
            for item in sorted(self.audited.lock.sources, key=lambda value: value.source.dataset_id)
        )
        provisional = PublicationBenchmarkManifest(
            sources=sources,
            records=tuple(sorted(self.records, key=lambda item: item.sample_id)),
            split_sha256={},
            tokenizer_sha256=self.tokenizer.semantic_sha256,
            chat_template_sha256=self.tokenizer.chat_template_sha256,
            sealed_payload_sha256=hashlib.sha256(sealed_bytes).hexdigest(),
        )
        manifest = replace(
            provisional,
            split_sha256={split: provisional.compute_split_sha256(split) for split in SPLIT_COUNTS},
        )
        errors = manifest.validate()
        if errors:
            raise PublicationBuildError("; ".join(errors))
        source_manifest = self.audited.public_manifest()
        report = self._build_report(manifest, source_manifest)
        self.audited.verify_unchanged()
        return manifest, source_manifest, {"report": report, "sealed_bytes": sealed_bytes}

    def _reserve_demonstrations(self) -> dict[str, tuple[CanonicalSourceExample, ...]]:
        reserved: dict[str, tuple[CanonicalSourceExample, ...]] = {}
        for dataset_id, count in _DEMO_COUNTS.items():
            candidates = [
                item
                for item in self._ordered_examples(dataset_id, purpose="prefix-demonstration")
                if dataset_id not in {"gsm8k", "math"} or item.source_split == "train"
            ]
            if len(candidates) < count:
                raise PublicationBuildError(
                    f"source {dataset_id!r} has fewer than {count} prefix demonstrations"
                )
            chosen = tuple(candidates[:count])
            reserved[dataset_id] = chosen
            self.used_rows.update((item.dataset_id, item.row_id) for item in chosen)
        return reserved

    def _build_semantic_cell(self, cell: AllocationCell) -> None:
        candidates = self._ordered_examples(
            cell.dataset_id,
            purpose=f"query:{cell.split}:{cell.token_bucket}",
        )
        selected = 0
        for example in candidates:
            if selected == cell.count:
                break
            if (example.dataset_id, example.row_id) in self.used_rows:
                continue
            if cell.dataset_id in {"gsm8k", "math"}:
                required_split = "test" if cell.split == "semantic_sealed_test" else "train"
                if example.source_split != required_split:
                    continue
            try:
                prefix_text, recipe_id = self._semantic_prefix(example, cell)
            except PublicationBuildError:
                if cell.dataset_id in _LONG_DATASETS:
                    continue
                raise
            suffix = _semantic_suffix(example)
            if not self._suffix_available(suffix):
                continue
            self._append_sample(
                dataset_id=cell.dataset_id,
                split=cell.split,
                token_bucket=cell.token_bucket,
                example=example,
                prefix_text=prefix_text,
                suffix=suffix,
                recipe_id=recipe_id,
                provenance={
                    "source_dataset_id": example.dataset_id,
                    "source_row_id": example.row_id,
                    "source_split": example.source_split,
                    "source_role": example.source_role,
                },
            )
            self.used_rows.add((example.dataset_id, example.row_id))
            selected += 1
        if selected != cell.count:
            raise PublicationBuildError(
                f"source {cell.dataset_id!r} cannot fill {cell.split}/{cell.token_bucket}: "
                f"needed {cell.count}, selected {selected}"
            )

    def _build_runtime_cell(self, cell: AllocationCell) -> None:
        prefix_text, recipe_id = self._shared_prefix(
            "sharegpt",
            cell.token_bucket,
            "runtime_audit",
        )
        selected = 0
        burst_rows = [
            item for item in self.loaded.burst_rows if item.token_bucket == cell.token_bucket
        ]
        for example in self.share_query_rows:
            if selected == cell.count:
                break
            if (example.dataset_id, example.row_id) in self.used_rows:
                continue
            suffix = _runtime_suffix(example.query)
            if not self._suffix_available(suffix):
                continue
            if cell.dataset_id == "sharegpt":
                provenance: dict[str, Any] = {
                    "source_dataset_id": "sharegpt",
                    "source_row_id": example.row_id,
                    "source_split": "trace",
                    "source_role": example.source_role,
                }
                row_id = example.row_id
            else:
                if selected >= len(burst_rows):
                    break
                burst = burst_rows[selected]
                provenance = _burst_provenance(burst, example.row_id)
                row_id = burst.row_id
            runtime_example = replace(
                example,
                dataset_id=cell.dataset_id,
                row_id=row_id,
                task="request_trace_replay",
                source_role="data",
            )
            self._append_sample(
                dataset_id=cell.dataset_id,
                split=cell.split,
                token_bucket=cell.token_bucket,
                example=runtime_example,
                prefix_text=prefix_text,
                suffix=suffix,
                recipe_id=recipe_id,
                provenance=provenance,
            )
            self.used_rows.add((example.dataset_id, example.row_id))
            selected += 1
        if selected != cell.count:
            raise PublicationBuildError(
                f"trace source {cell.dataset_id!r} cannot fill runtime bucket {cell.token_bucket}"
            )

    def _semantic_prefix(
        self,
        example: CanonicalSourceExample,
        cell: AllocationCell,
    ) -> tuple[str, str]:
        family = "transport_train" if cell.split == "transport_train" else "post_transport"
        if example.dataset_id in _LONG_DATASETS:
            key = (example.row_id, cell.token_bucket, family)
            cached = self.long_prefix_cache.get(key)
            if cached is not None:
                return cached
            raw = (
                f"<|im_start|>system\nGoldenExperience publication benchmark. "
                f"Prefix family: {family}. Use the following source context to answer the next "
                f"question faithfully.\n\n{example.context}"
            )
            prefix, token_ids = self.tokenizer.exact_prefix(raw, cell.token_bucket)
            recipe_id = self._register_recipe(
                prefix=prefix,
                token_ids=token_ids,
                token_bucket=cell.token_bucket,
                family=family,
                dataset_id=example.dataset_id,
                source_rows=(example.row_id,),
            )
            result = (prefix, recipe_id)
            self.long_prefix_cache[key] = result
            return result
        return self._shared_prefix(example.dataset_id, cell.token_bucket, family)

    def _shared_prefix(
        self,
        dataset_id: str,
        token_bucket: int,
        family: str,
    ) -> tuple[str, str]:
        key = (dataset_id, token_bucket, family)
        cached = self.shared_prefixes.get(key)
        if cached is not None:
            return cached
        demo_dataset = "mbpp" if dataset_id == "humaneval" else dataset_id
        if demo_dataset == "bfcl":
            rows = self._ordered_examples("bfcl", purpose="function-prefix")
        else:
            rows = self.demo_rows.get(demo_dataset, ())
        if not rows:
            raise PublicationBuildError(f"source {dataset_id!r} has no prefix corpus")
        header = _prefix_header(dataset_id, family)
        selected_rows: list[CanonicalSourceExample] = []
        demonstrations: list[str] = []
        raw = f"<|im_start|>system\n{header}"
        for row in rows:
            if not row.demonstration:
                continue
            selected_rows.append(row)
            demonstrations.append(row.demonstration)
            if len(selected_rows) % 8:
                continue
            raw = f"<|im_start|>system\n{header}\n\n" + "\n\n".join(demonstrations)
            if len(self.tokenizer.encode(raw)) >= token_bucket:
                break
        if not selected_rows:
            raise PublicationBuildError(f"source {dataset_id!r} prefix corpus is empty")
        raw = f"<|im_start|>system\n{header}\n\n" + "\n\n".join(demonstrations)
        prefix, token_ids = self.tokenizer.exact_prefix(raw, token_bucket)
        recipe_id = self._register_recipe(
            prefix=prefix,
            token_ids=token_ids,
            token_bucket=token_bucket,
            family=family,
            dataset_id=dataset_id,
            source_rows=tuple(item.row_id for item in selected_rows),
        )
        result = (prefix, recipe_id)
        self.shared_prefixes[key] = result
        return result

    def _register_recipe(
        self,
        *,
        prefix: str,
        token_ids: Sequence[int],
        token_bucket: int,
        family: str,
        dataset_id: str,
        source_rows: tuple[str, ...],
    ) -> str:
        prefix_sha = _text_sha256(prefix)
        recipe_id = "pg-" + prefix_sha
        recipe = PrefixRecipe(
            prefix_group_id=recipe_id,
            prefix_sha256=prefix_sha,
            token_ids_sha256=token_ids_sha256(list(token_ids)),
            token_bucket=token_bucket,
            family=family,
            dataset_id=dataset_id,
            source_row_ids=source_rows,
        )
        previous = self.recipes.get(recipe_id)
        if previous is not None and previous != recipe:
            raise PublicationBuildError("prefix recipe hash collision or provenance drift")
        self.recipes[recipe_id] = recipe
        return recipe_id

    def _suffix_available(self, suffix: str) -> bool:
        digest = _text_sha256(suffix)
        if digest in self.used_suffixes:
            return False
        self.used_suffixes.add(digest)
        return True

    def _append_sample(
        self,
        *,
        dataset_id: str,
        split: str,
        token_bucket: int,
        example: CanonicalSourceExample,
        prefix_text: str,
        suffix: str,
        recipe_id: str,
        provenance: Mapping[str, Any],
    ) -> None:
        prefix_sha = _text_sha256(prefix_text)
        if recipe_id != "pg-" + prefix_sha:
            raise PublicationBuildError("sample prefix differs from its registered recipe")
        sample_id = _sample_id(split, token_bucket, dataset_id, example.row_id)
        raw = RawBenchmarkSample(
            sample_id=sample_id,
            prefix_text=prefix_text,
            suffix_query=suffix,
            reference=example.reference,
            evaluation=example.evaluation,
            provenance={
                "builder_id": PUBLICATION_BUILDER_ID,
                "prefix_recipe_id": recipe_id,
                **dict(provenance),
            },
        )
        content_sha = publication_sample_content_sha256(
            prefix_text=prefix_text,
            suffix_query=suffix,
            reference=example.reference,
            evaluation=example.evaluation,
            task=example.task,
        )
        record = GroupedPrefixRecord(
            sample_id=sample_id,
            split=split,
            dataset_id=dataset_id,
            prefix_group_id=recipe_id,
            prefix_sha256=prefix_sha,
            suffix_query_sha256=_text_sha256(suffix),
            content_sha256=content_sha,
            token_bucket=token_bucket,
            task=example.task,
        )
        errors = raw.validate(record)
        if errors:
            raise PublicationBuildError("; ".join(errors))
        self.records.append(record)
        self.samples[split].append(raw)

    def _ordered_examples(
        self,
        dataset_id: str,
        *,
        purpose: str,
    ) -> tuple[CanonicalSourceExample, ...]:
        try:
            rows = self.loaded.examples[dataset_id]
        except KeyError as exc:
            raise PublicationBuildError(f"source {dataset_id!r} has no canonical examples") from exc
        return tuple(
            sorted(
                rows,
                key=lambda item: (
                    _selection_digest(self.seed, purpose, dataset_id, item.row_id),
                    item.row_id,
                ),
            )
        )

    def _validate_built_rows(self) -> None:
        if set(self.samples) != set(SPLIT_COUNTS):
            raise PublicationBuildError("built raw sample stores are incomplete")
        if len(self.records) != sum(SPLIT_COUNTS.values()):
            raise PublicationBuildError("built record count differs from the frozen contract")
        ids = [item.sample_id for item in self.records]
        contents = [item.content_sha256 for item in self.records]
        if len(ids) != len(set(ids)) or len(contents) != len(set(contents)):
            raise PublicationBuildError("built benchmark ids or contents are not unique")
        for split, expected in SPLIT_COUNTS.items():
            rows = [item for item in self.records if item.split == split]
            if len(rows) != expected:
                raise PublicationBuildError(f"built split {split!r} has the wrong size")
            bucket_counts = Counter(item.token_bucket for item in rows)
            target = expected // len(PREFIX_BUCKETS)
            if bucket_counts != Counter({bucket: target for bucket in PREFIX_BUCKETS}):
                raise PublicationBuildError(f"built split {split!r} is not bucket-balanced")
        protected: dict[str, str] = {}
        for record in self.records:
            if record.split not in {"risk_calibration", "validation", "semantic_sealed_test"}:
                continue
            previous = protected.setdefault(record.suffix_query_sha256, record.split)
            if previous != record.split:
                raise PublicationBuildError("protected query leaked across final benchmark splits")

    def _build_report(
        self,
        manifest: PublicationBenchmarkManifest,
        source_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        allocations: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for record in self.records:
            bucket = str(record.token_bucket)
            current = allocations[record.split][bucket].get(record.dataset_id, 0)
            allocations[record.split][bucket][record.dataset_id] = current + 1
        return {
            "schema_version": PUBLICATION_BUILD_REPORT_SCHEMA,
            "builder_id": PUBLICATION_BUILDER_ID,
            "selection_seed": self.seed,
            "benchmark_manifest_sha256": manifest.content_sha256(),
            "source_manifest_sha256": _canonical_sha256(source_manifest),
            "source_lock_sha256": self.audited.lock.content_sha256(),
            "tokenizer_sha256": self.tokenizer.semantic_sha256,
            "chat_template_sha256": self.tokenizer.chat_template_sha256,
            "record_count": len(self.records),
            "split_counts": dict(SPLIT_COUNTS),
            "allocations": allocations,
            "prefix_recipes": [
                asdict(item)
                for item in sorted(self.recipes.values(), key=lambda value: value.prefix_group_id)
            ],
            "isolation": {
                "query_rows_globally_unique": True,
                "transport_prefix_family_isolated": True,
                "protected_suffix_hashes_disjoint": True,
                "semantic_sealed_raw_rows_in_public_output": False,
                "trace_only_sources_restricted_to_runtime_audit": True,
            },
        }


def publish_publication_build(
    builder: PublicationDatasetBuilder,
    *,
    output_dir: str | Path,
    sealed_payload: str | Path,
) -> PublicationBuildResult:
    output = Path(output_dir).resolve()
    sealed = Path(sealed_payload).resolve()
    if output.exists() or sealed.exists():
        raise PublicationBuildError("publication output and sealed payload must not already exist")
    if output == sealed or output in sealed.parents or sealed in output.parents:
        raise PublicationBuildError("sealed payload must be outside the public output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.publication-build.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise PublicationBuildError("publication output is already being built") from exc
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    sealed_descriptor = -1
    temporary_sealed: Path | None = None
    sealed_published = False
    completed = False
    try:
        os.write(descriptor, (PUBLICATION_BUILDER_ID + "\n").encode("ascii"))
        os.fsync(descriptor)
        manifest, source_manifest, payload = builder.build()
        report = dict(payload["report"])
        sealed_bytes = payload["sealed_bytes"]
        if not isinstance(sealed_bytes, bytes):
            raise PublicationBuildError("builder returned an invalid sealed payload")
        raw_dir = temporary_output / "raw"
        raw_dir.mkdir()
        _write_json(temporary_output / "source_manifest.json", source_manifest)
        _write_jsonl(
            temporary_output / "records.jsonl",
            (asdict(item) for item in manifest.records),
        )
        for split in SPLIT_COUNTS:
            if split == "semantic_sealed_test":
                continue
            _write_jsonl(
                raw_dir / f"{split}.jsonl",
                (
                    item.to_dict()
                    for item in sorted(builder.samples[split], key=lambda value: value.sample_id)
                ),
            )
        manifest.save(temporary_output / "benchmark_manifest.json")
        output_hashes = {
            str(path.relative_to(temporary_output)): sha256_file(path)
            for path in sorted(temporary_output.rglob("*"))
            if path.is_file()
        }
        report["public_output_sha256"] = output_hashes
        report["sealed_payload_sha256"] = manifest.sealed_payload_sha256
        _write_json(temporary_output / "build_report.json", report)
        for path in temporary_output.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        sealed_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{sealed.name}.", suffix=".tmp", dir=sealed.parent
        )
        temporary_sealed = Path(temporary_name)
        _write_all(sealed_descriptor, sealed_bytes)
        os.fchmod(sealed_descriptor, 0o400)
        os.fsync(sealed_descriptor)
        if hashlib.sha256(sealed_bytes).hexdigest() != manifest.sealed_payload_sha256:
            raise PublicationBuildError("sealed payload changed before publication")
        os.close(sealed_descriptor)
        sealed_descriptor = -1
        os.link(temporary_sealed, sealed)
        sealed_published = True
        _fsync_directory(sealed.parent)
        os.rename(temporary_output, output)
        _fsync_directory(output.parent)
        manifest_path = output / "benchmark_manifest.json"
        completed = True
        return PublicationBuildResult(
            output_dir=output,
            sealed_payload=sealed,
            manifest_path=manifest_path,
            source_manifest_path=output / "source_manifest.json",
            build_report_path=output / "build_report.json",
            manifest=manifest,
        )
    except (PublicationBuildError, PublicationSourceError):
        raise
    except Exception as exc:
        raise PublicationBuildError(
            "publication dataset could not be published atomically"
        ) from exc
    finally:
        if sealed_descriptor >= 0:
            os.close(sealed_descriptor)
        if temporary_sealed is not None:
            temporary_sealed.unlink(missing_ok=True)
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        if sealed_published and not completed:
            sealed.unlink(missing_ok=True)
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _validate_allocation(cells: Sequence[AllocationCell]) -> None:
    counts: Counter[str] = Counter()
    buckets: dict[str, Counter[int]] = defaultdict(Counter)
    observed_cells: set[tuple[str, int, str]] = set()
    for cell in cells:
        cell_key = (cell.split, cell.token_bucket, cell.dataset_id)
        if (
            cell.split not in SPLIT_COUNTS
            or cell.token_bucket not in PREFIX_BUCKETS
            or cell.dataset_id not in _SHARED_PREFIX_DATASETS | _LONG_DATASETS | TRACE_ONLY_DATASETS
            or type(cell.count) is not int
            or cell.count <= 0
            or cell_key in observed_cells
        ):
            raise PublicationBuildError("publication allocation cell is invalid")
        observed_cells.add(cell_key)
        if cell.dataset_id in TRACE_ONLY_DATASETS and cell.split != "runtime_audit":
            raise PublicationBuildError("trace-only source appears in a semantic allocation")
        if cell.split == "runtime_audit" and cell.dataset_id not in TRACE_ONLY_DATASETS:
            raise PublicationBuildError("runtime audit allocation must use frozen trace sources")
        counts[cell.split] += cell.count
        buckets[cell.split][cell.token_bucket] += cell.count
    if counts != Counter(SPLIT_COUNTS):
        raise PublicationBuildError("publication allocation does not match frozen split sizes")
    for split, expected in SPLIT_COUNTS.items():
        target = expected // len(PREFIX_BUCKETS)
        if buckets[split] != Counter({bucket: target for bucket in PREFIX_BUCKETS}):
            raise PublicationBuildError(f"publication allocation for {split!r} is not balanced")


def _prefix_header(dataset_id: str, family: str) -> str:
    instructions = {
        "bfcl": "Use the registered function schemas and return only the requested JSON call.",
        "gsm8k": "Solve the next arithmetic word problem and return only its final number.",
        "humaneval": "Complete the next Python function and return only executable Python code.",
        "math": "Solve the next competition problem and return only the final mathematical answer.",
        "mbpp": "Implement the next Python task and return only executable Python code.",
        "sharegpt": "Continue the next user request naturally and concisely.",
    }
    try:
        instruction = instructions[dataset_id]
    except KeyError as exc:
        raise PublicationBuildError(
            f"source {dataset_id!r} has no shared-prefix instruction"
        ) from exc
    return (
        f"GoldenExperience publication benchmark. Prefix family: {family}. {instruction} "
        "The following frozen public examples form a shared serving prefix; the prefix may end "
        "at an exact cache boundary."
    )


def _semantic_suffix(example: CanonicalSourceExample) -> str:
    opening = "\n<|im_end|>\n<|im_start|>user\n"
    closing = "\n/no_think\n<|im_end|>\n<|im_start|>assistant\n"
    if example.dataset_id in _LONG_DATASETS:
        body = f"Question: {example.query}\nReturn a concise answer supported by the context."
    elif example.dataset_id == "gsm8k":
        body = f"Problem: {example.query}\nReturn only the final numeric answer."
    elif example.dataset_id == "math":
        body = f"Problem: {example.query}\nReturn only the final answer."
    elif example.dataset_id in {"humaneval", "mbpp"}:
        body = f"Python task:\n{example.query}\nReturn only the complete Python implementation."
    elif example.dataset_id == "bfcl":
        body = (
            f"Candidate functions (JSON): {example.context}\nUser request: {example.query}\n"
            'Return only JSON as {"name":...,"arguments":{...}}.'
        )
    else:
        raise PublicationBuildError(f"source {example.dataset_id!r} has no suffix formatter")
    return opening + body + closing


def _runtime_suffix(query: str) -> str:
    return (
        "\n<|im_end|>\n<|im_start|>user\n"
        + query
        + "\n/no_think\n<|im_end|>\n<|im_start|>assistant\n"
    )


def _burst_provenance(row: BurstWorkloadRow, sharegpt_row_id: str) -> dict[str, Any]:
    return {
        "source_dataset_id": "burstgpt",
        "source_row_id": row.row_id,
        "source_split": "trace",
        "source_role": "data",
        "paired_text_dataset_id": "sharegpt",
        "paired_text_row_id": sharegpt_row_id,
        "timestamp": row.timestamp,
        "model": row.model,
        "request_tokens": row.request_tokens,
        "response_tokens": row.response_tokens,
        "log_type": row.log_type,
    }


def _sample_id(split: str, bucket: int, dataset_id: str, row_id: str) -> str:
    row_digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:20]
    return f"{split}.{bucket:04d}.{dataset_id}.{row_digest}"


def _selection_digest(seed: int, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(seed.to_bytes(16, "big", signed=False))
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonl_bytes(values: Any) -> bytes:
    return b"".join(
        (json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for value in values
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Any) -> None:
    path.write_bytes(_jsonl_bytes(values))


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("failed to write publication payload")
        remaining = remaining[written:]


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
