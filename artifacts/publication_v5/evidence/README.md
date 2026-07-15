# Publication v5 Evidence

This directory contains the curated evidence package for the terminal Qwen3-4B to Qwen3-8B
v4 method-development result. It supports a negative-result claim only. It does not authorize
selector fitting, calibration, validation, semantic-sealed evaluation, or runtime reuse.

## Contents

- `method_dev_report.v4.json.gz` is a deterministic gzip copy of the complete 9,216-row
  method-dev report. Its uncompressed SHA-256 is
  `f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400`.
- `method_dev_candidates.v4.csv` contains all nine registered rank/seed candidates.
- `method_dev_ranks.v4.csv` reproduces rank aggregation and within-rank safe-set unions.
- `method_dev_tasks.v4.csv` and `method_dev_token_buckets.v4.csv` contain deployment-candidate
  breakdowns.
- `method_dev_failure_overlap.v4.csv` contains the complete three-criterion failure partition,
  including zero-count cells.
- `method_dev_safe_sets.v4.csv` separates the fixed deployment result from non-deployable,
  target-derived candidate oracles.
- `method_dev_evidence_manifest.v4.json` binds every source and generated file by SHA-256 and
  records the failed protocol disposition.

## Reproduction

Generate the package from the retained workspace report:

```bash
python3 paper/tools/build_method_dev_evidence.py
```

Verify that every tracked byte is reproducible without rewriting files:

```bash
python3 paper/tools/build_method_dev_evidence.py --check
```

The generator reads only the explicit method-dev report, fit receipt, failed-stage receipt, and
post-failure diagnostic. It rejects any input or output path containing `sealed` and never reads
the semantic-sealed payload.
