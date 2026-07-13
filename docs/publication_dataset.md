# Publication Dataset Freeze

The selective-KV v5 experiment uses a real-data, grouped-prefix benchmark. The builder is
deterministic and fail-closed: it verifies every source byte before parsing, assigns every query
row at most once, materializes exact Qwen3 token buckets, validates every scorer contract, and
publishes no raw sealed row under the public output directory.

## Frozen Sources

`configs/publication_sources.qwen3-v5.json` is the portable source lock. It records the upstream
revision, per-file size and SHA-256, a deterministic Merkle identity for multi-file datasets, the
applicable dataset license, and the required Qwen3 tokenizer/chat-template identities. A different
tokenizer path is rejected before rows are selected. The lock contains:

| Dataset | Frozen revision | Use | License |
| --- | --- | --- | --- |
| LongBench HotpotQA | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | semantic | CC-BY-SA-4.0 |
| LongBench Qasper | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | semantic | CC-BY-4.0 |
| LongBench MultiFieldQA-en | `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` | semantic | MIT (LongBench release) |
| BFCL v4 simple Python | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | semantic | Apache-2.0 |
| GSM8K | `3101c7d5072418e28b9008a6636bde82a006892c` | semantic | MIT |
| MATH | `21a5633873b6a120296cce3e2df9d5550074f4a3` | semantic | MIT |
| HumanEval | `6d43fb980f9fee3c892a914eda09951f772ad10d` | semantic | MIT |
| MBPP | `OpenCompassData-core-20240207` | semantic | CC-BY-4.0 |
| ShareGPT cleaned split | `192ab2185289094fc556ec8ce5ce1e8e587154ca` | runtime trace text | Apache-2.0 |
| BurstGPT | `d895a53bb7b8ec137d0d2fe203b335835a78c10a` | runtime arrival/length trace | CC-BY-4.0 |

LongBench republishes transformed benchmark rows. The lock therefore records the original
HotpotQA and Qasper licenses rather than treating the LongBench repository's MIT software license
as a replacement. Users redistributing raw stores remain responsible for every upstream
attribution and share-alike obligation.

## Source Layout

Install the explicit builder dependencies with `python3 -m pip install -e ".[publication]"`.
Place locked files below one source root using the paths in the lock, for example:

```text
publication_sources/
  bfcl/BFCL_v4_simple_python.json
  bfcl/possible_answer/BFCL_v4_simple_python.json
  burstgpt/BurstGPT_1.csv
  gsm8k/{train,test}.jsonl
  humaneval/HumanEval.jsonl
  longbench/{hotpotqa,qasper,multifieldqa_en}.jsonl
  math/{algebra,...,precalculus}_{train,test}.parquet
  mbpp/mbpp.jsonl
  sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
```

`--source-path DATASET:ROLE=PATH` may override a local location without changing its portable
identity. Overrides still undergo the locked size and SHA-256 checks. Symbolic-link source files,
unknown overrides, files that change during hashing, and malformed rows are rejected.

Audit all bytes before a build:

```bash
golden-publication-benchmark audit-sources \
  --source-lock configs/publication_sources.qwen3-v5.json \
  --source-root /data/publication_sources
```

## Deterministic Split Contract

The six semantic splits contain 13,312 globally unique source queries; the runtime audit adds 512
trace requests. Every split is exactly balanced across 128, 512, 2,048, and 8,192 Qwen3 tokens.
Each exact prefix is decoded and re-encoded before acceptance, so a bucket never means "at least"
or an approximate word count.

The allocation has four isolation rules:

1. `transport_train` uses a distinct rendered prefix family that cannot recur later.
2. GSM8K and MATH use only official train rows before the sealed test; their sealed rows use only
   official test data.
3. Every source query row and rendered suffix is globally unique. In particular,
   `risk_calibration`, `validation`, and `semantic_sealed_test` have disjoint suffix hashes.
4. ShareGPT and BurstGPT occur only in `runtime_audit`. BurstGPT supplies arrival/model/token-count
   metadata and is deterministically paired with distinct ShareGPT text, because BurstGPT is an
   anonymized length trace and contains no prompt text.

LongBench rows use their own document as the prefix. GSM8K, MATH, code, and function-call rows use
shared public demonstration/schema prefixes; demonstration rows are reserved and never become
queries. MATH rows without a balanced final box or with an empty normalized answer are excluded by
the same scorer validation used at evaluation time.

Task scoring is declared per row: LongBench token F1, GSM8K numeric equality, normalized MATH exact
match, sandboxed HumanEval/MBPP tests, and structured BFCL function-call matching. No evaluator
chooses a metric from the prediction.

## Build And Outputs

The raw sealed destination must be outside the public output directory and must not already exist:

```bash
golden-publication-benchmark build \
  --source-lock configs/publication_sources.qwen3-v5.json \
  --source-root /data/publication_sources \
  --tokenizer-model /models/Qwen3-8B \
  --output-dir artifacts/publication_v5 \
  --sealed-output secure/semantic_sealed_test_v5.jsonl
```

The public directory contains `benchmark_manifest.json`, a hash-only `records.jsonl`, the audited
`source_manifest.json`, `build_report.json`, and raw stores for the six non-sealed splits. The
separate sealed file is mode `0400`; its expected SHA-256 is recorded in the public manifest. The
builder uses exclusive destinations and temporary files, and removes partial publications after a
failed validation.

Creating the sealed payload is not an evaluation opening. After the workspace is initialized, only
`golden-v5-pipeline open-semantic-sealed` may read it, and only after all four independent validation
directions pass. Do not inspect, sample, or use the sealed raw file for debugging.
