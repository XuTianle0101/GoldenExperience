# Publication v5 Artifact Map

Publication v5 records a complete Qwen3-4B to Qwen3-8B full-prefix transport fit followed by a
terminal method-development failure. The selected rank-64/seed-17 candidate reaches
`142/1024 = 0.138671875` oracle-safe coverage against a registered minimum of `0.45`.

## Evidence Chain

```text
workspace initialization (locked)
  -> transport_train collection (completed)
  -> nine-candidate transport fit (completed)
  -> method_dev collection (completed)
  -> method_dev evaluation (failed)
  -X-> selector / calibration / other directions / validation / sealed / runtime
```

The failed stage is evidence for a negative result; it is not an authoritative success receipt.
The implementation of later stages remains testable, but those stages were not executed in this
workspace.

## Files

- `initialization_v4.json` binds pipeline, source code, benchmark, tokenizer, and model identities
  and records the semantic state as `locked`.
- `stages/qwen3_4b_to_8b.fit_transport.v4.json` is the verified fit-stage receipt.
- `stages/qwen3_4b_to_8b.evaluate_method_dev.v4.failed.json` records the failed gate, complete
  candidate matrix, rank aggregation, and verification results.
- `evidence/` contains the deterministic compressed report, all publication tables, and a checksum
  manifest.
- `figures/` contains each plotted value as CSV and both SVG/PDF vector renderings.
- `development/v4_method_dev_diagnostic.json` contains post-failure task, bucket, safe-set, and
  mechanism analysis. Its authority is diagnostic, not runtime approval.
- `development/v4_implementation_verification.json` records tests, build checks, coverage, and
  bounded real-model implementation smoke evidence captured before the formal run.

## Stable Identities

| Object | SHA-256 or ID |
| --- | --- |
| Pipeline | `v5-pipeline-1c6fed3dc231893debb58298` |
| Executable source | `b3d0dcb81e5a528937c1a80858273e2e8f8b1876be3d3691e222959867ef2760` |
| Benchmark manifest | `557cfe1eccd522d19e6a06177b2d86e6b1a55587b8a84cba65732fad4d2bcd4a` |
| Fit manifest content | `7195d0cf59f0c8995ce4065a42587733597d7ab3861c79e4c347f8a5e11e80a0` |
| Method-dev report | `f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400` |

## Verification

```bash
python3 paper/tools/build_method_dev_evidence.py --check --from-archive
python3 paper/tools/build_figures.py --check
python3 paper/tools/check_manuscript.py
```

These commands use tracked public evidence only. They reject any path containing `sealed` and do
not access the semantic payload.
