#!/usr/bin/env python3
"""Generate the deterministic, explicitly split Qwen3 cached-KV prompt dataset."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from goldenexperience.size_variant.cached_kv_dataset import (
    CachedKVPrompt,
    CachedKVPromptDataset,
)

BUCKETS = (32, 128, 512, 2048)
CATEGORIES = ("math", "code", "prose", "chat")
SPLIT_COUNTS = {"train": 256, "validation": 64, "test": 64}
SPLIT_OFFSETS = {"train": 1000, "validation": 2000, "test": 3000}


def _sample(split: str, index: int) -> CachedKVPrompt:
    value = SPLIT_OFFSETS[split] + index
    category = CATEGORIES[index % len(CATEGORIES)]
    bucket = BUCKETS[(index // len(CATEGORIES)) % len(BUCKETS)]
    if category == "math":
        left = 20 + value % 37
        right = 7 + value % 19
        answer = str(left + right)
        template = (
            "Reference notes:\n{context}\n"
            f"Compute {left} + {right}. Return exactly `Final answer: {answer}` and no other text."
        )
    elif category == "code":
        left = 3 + value % 11
        right = 2 + value % 7
        answer = str(left * right)
        template = (
            "Code review context:\n{context}\n"
            f"What does `print({left} * {right})` output? Return exactly "
            f"`Final answer: {answer}` and no other text."
        )
    elif category == "prose":
        answer = f"MARKER-{value}"
        template = (
            "Background material:\n{context}\n"
            f"Ignore incidental numbers. Return exactly `Final answer: {answer}` and no other text."
        )
    else:
        answer = f"ACK-{value}"
        template = (
            "Conversation memory:\n{context}\n"
            f"Return exactly `Final answer: {answer}` and no other text."
        )
    return CachedKVPrompt(
        prompt_id=f"{split}-{category}-{index:03d}",
        split=split,
        category=category,
        group_id=f"{split}-sealed-{index:03d}",
        token_bucket=bucket,
        template=template,
        context_seed=f"{split}-{category}-seed-{value}",
        expected_answer=answer,
    )


def build_dataset() -> CachedKVPromptDataset:
    samples = tuple(
        _sample(split, index) for split, count in SPLIT_COUNTS.items() for index in range(count)
    )
    dataset = CachedKVPromptDataset(samples=samples)
    errors = dataset.approval_errors()
    if errors:
        raise ValueError("; ".join(errors))
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "qwen3_cached_kv_prompts.json",
    )
    args = parser.parse_args()
    dataset = build_dataset()
    dataset.save(args.output)
    print(f"Wrote {len(dataset.samples)} prompts to {args.output}")
    for split in ("train", "validation", "test"):
        print(f"{split}: {len(dataset.split(split))} sha256={dataset.split_sha256(split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
