"""Leak-resistant prompt dataset contracts for cached-KV bridge training."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CACHED_KV_DATASET_SCHEMA_VERSION = "goldenexperience.cached_kv_dataset.v1"
DATASET_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class CachedKVPrompt:
    prompt_id: str
    split: str
    category: str
    group_id: str
    token_bucket: int
    template: str
    context_seed: str
    expected_answer: str | None = None

    def normalized_content(self) -> str:
        # Only fields that affect rendered model input belong in the leakage identity.
        fields = (
            self.template,
            self.context_seed,
        )
        return "\n".join(_normalize_text(item) for item in fields)

    def content_sha256(self) -> str:
        return hashlib.sha256(self.normalized_content().encode("utf-8")).hexdigest()

    def render(self, *, context_items: int) -> str:
        if context_items < 0:
            raise ValueError("context_items must be non-negative")
        context = " ".join(
            f"{self.context_seed} fact-{index:05d} value-{(index * 7919 + 104729) % 999983}."
            for index in range(context_items)
        )
        if "{context}" in self.template:
            return self.template.replace("{context}", context)
        if context:
            return f"{self.template}\nContext: {context}"
        return self.template


@dataclass(frozen=True)
class CachedKVPromptDataset:
    samples: tuple[CachedKVPrompt, ...]
    schema_version: str = CACHED_KV_DATASET_SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.schema_version != CACHED_KV_DATASET_SCHEMA_VERSION:
            errors.append("unsupported cached-KV dataset schema_version")
        if not self.samples:
            errors.append("cached-KV dataset must contain prompts")
            return errors
        ids: set[str] = set()
        content_to_split: dict[str, str] = {}
        group_to_split: dict[str, str] = {}
        split_counts = {split: 0 for split in DATASET_SPLITS}
        for sample in self.samples:
            if not sample.prompt_id:
                errors.append("prompt_id is required")
            elif sample.prompt_id in ids:
                errors.append(f"duplicate prompt_id: {sample.prompt_id}")
            ids.add(sample.prompt_id)
            if sample.split not in DATASET_SPLITS:
                errors.append(f"prompt {sample.prompt_id} has an invalid split")
            else:
                split_counts[sample.split] += 1
            if not sample.category:
                errors.append(f"prompt {sample.prompt_id} category is required")
            if not sample.group_id:
                errors.append(f"prompt {sample.prompt_id} group_id is required")
            if sample.token_bucket <= 0:
                errors.append(f"prompt {sample.prompt_id} token_bucket must be positive")
            if not sample.template.strip():
                errors.append(f"prompt {sample.prompt_id} template is required")
            if not sample.context_seed.strip():
                errors.append(f"prompt {sample.prompt_id} context_seed is required")

            content_hash = sample.content_sha256()
            previous_content_split = content_to_split.get(content_hash)
            if previous_content_split is not None:
                errors.append(
                    f"normalized prompt content is duplicated across {previous_content_split} "
                    f"and {sample.split}: {sample.prompt_id}"
                )
            content_to_split[content_hash] = sample.split
            previous_group_split = group_to_split.get(sample.group_id)
            if previous_group_split is not None and previous_group_split != sample.split:
                errors.append(
                    f"prompt group {sample.group_id} crosses {previous_group_split} and "
                    f"{sample.split} splits"
                )
            group_to_split[sample.group_id] = sample.split
        for split, count in split_counts.items():
            if count <= 0:
                errors.append(f"dataset split {split} is empty")
        hashes = [self.split_sha256(split) for split in DATASET_SPLITS if split_counts[split]]
        if len(hashes) != len(set(hashes)):
            errors.append("dataset split identities must be distinct")
        return errors

    def approval_errors(
        self,
        *,
        min_test_prompts: int = 32,
        required_token_buckets: tuple[int, ...] = (32, 128, 512, 2048),
    ) -> list[str]:
        errors = self.validate()
        test_samples = self.split("test")
        if len(test_samples) < min_test_prompts:
            errors.append("sealed test split has too few prompts")
        test_buckets = {sample.token_bucket for sample in test_samples}
        if not set(required_token_buckets) <= test_buckets:
            errors.append("sealed test split is missing required token buckets")
        return errors

    def split(self, name: str) -> tuple[CachedKVPrompt, ...]:
        if name not in DATASET_SPLITS:
            raise ValueError(f"unknown dataset split: {name}")
        return tuple(sample for sample in self.samples if sample.split == name)

    def split_sha256(self, name: str) -> str:
        records = [
            asdict(sample) for sample in sorted(self.split(name), key=lambda item: item.prompt_id)
        ]
        raw = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "samples": [asdict(sample) for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CachedKVPromptDataset:
        samples_payload = payload.get("samples")
        if not isinstance(samples_payload, list):
            raise ValueError("cached-KV dataset samples must be a list")
        samples = tuple(CachedKVPrompt(**item) for item in samples_payload)
        return cls(
            samples=samples,
            schema_version=str(payload.get("schema_version", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> CachedKVPromptDataset:
        dataset = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        errors = dataset.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return dataset

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def render_to_token_bucket(
    sample: CachedKVPrompt,
    tokenizer: Any,
    *,
    suffix_tokens: int,
    max_context_items: int = 10000,
) -> tuple[str, list[int]]:
    """Grow deterministic context until the requested prefix and suffix exist."""

    if suffix_tokens <= 0:
        raise ValueError("suffix_tokens must be positive")
    required = sample.token_bucket + suffix_tokens + 1
    low = 0
    high = 1
    while True:
        text = _format_for_model(tokenizer, sample.render(context_items=high))
        token_ids = _encode(tokenizer, text)
        if len(token_ids) >= required:
            break
        if high == max_context_items:
            raise ValueError(f"prompt {sample.prompt_id} cannot reach token bucket")
        low = high
        high = min(high * 2, max_context_items)
    while low + 1 < high:
        middle = (low + high) // 2
        text = _format_for_model(tokenizer, sample.render(context_items=middle))
        token_ids = _encode(tokenizer, text)
        if len(token_ids) >= required:
            high = middle
        else:
            low = middle
    text = _format_for_model(tokenizer, sample.render(context_items=high))
    token_ids = _encode(tokenizer, text)
    if len(token_ids) < required:
        raise ValueError(f"prompt {sample.prompt_id} is shorter than its declared bucket")
    return text, token_ids


def contains_expected_final_answer(text: str, expected_answer: str) -> bool:
    """Match an explicit final-answer assertion without accepting token substrings."""

    expected = expected_answer.strip()
    if not expected:
        return False
    pattern = rf"\bfinal\s+answer\s*:\s*[`*_]*{re.escape(expected)}(?![\w-])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(item) for item in ids]


def _format_for_model(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return prompt
    formatted = apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(formatted, str) or not formatted:
        raise ValueError("tokenizer chat template did not produce model input text")
    return formatted


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()
