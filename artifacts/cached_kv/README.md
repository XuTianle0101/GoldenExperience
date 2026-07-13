# Qwen3 Cached-KV Bridge Artifacts

This directory retains only the active validation record and the compact reports needed
to verify it. Generated bridge weights, smoke outputs, and superseded candidates are
ignored and should not be kept in the repository.

- `bidirectional_mixed_refinement_summary_20260713.json` is the authoritative quality and
  gate decision for both directions.
- `mixed_refinement_8b_to_14b_20260713.json` and
  `mixed_refinement_14b_to_8b_20260713.json` retain direction-specific paired deltas,
  holdout behavior, and raw-result SHA-256 bindings.
- `runtime_cost_8b_to_14b_20260713.json` is the authoritative cost rejection. Its two
  compact source reports are `qwen3_8b_to_14b_native_prefill_1776.json` and
  `qwen3_8b_to_14b_cost_1776.json`.

The fixed/scaled baselines, regularization and capacity sweep, SiLU and CKA ablations,
failed all-parameter refinement, constrained teacher-forced refinement, native-generation
step study, and historical runtime experiments are consolidated in
`docs/paper_outline.md`. Their exact original JSON remains available through Git history.

All records in this directory predate manifest v5 and are deprecated development baselines.
They cannot serve as publication validation, risk calibration, semantic sealed, runtime audit,
or approval evidence for selective cross-scale reuse.

The only local raw quality files intentionally retained are the two mixed, holdout-16
per-prompt results under `artifacts/results/`. They are needed to analyze the remaining
forward and reverse exact-answer failures; all earlier smoke and refinement outputs were
removed.

Only a `CachedKVBridgeManifest` whose derived `approved` property is true may be used by
the runtime materializer. Missing held-out accuracy or Mooncake cost evidence keeps a
manifest fail closed.
