# Publication v5 Figures

These figures visualize the terminal v4 method-development result. They support a negative-result
paper and do not imply selector, calibration, validation, sealed-test, or runtime approval.

Each numbered figure is available as:

- CSV containing the exact plotted values;
- accessible SVG with a title and description;
- deterministic, single-page vector PDF using built-in fonts and no timestamps.

## Figure Index

1. `fig01_candidate_coverage`: all nine rank/seed candidates, the fixed deployment marker, the
   non-deployable nine-candidate oracle, and the `0.45` gate.
2. `fig02_full_prefix_by_length`: v3 sampled-prefix versus v4 full-prefix supervision at fixed
   rank 128 and seed 17.
3. `fig03_task_heterogeneity`: task-level safe coverage, greedy agreement, and perplexity drift.
4. `fig04_failure_overlap`: the complete failure partition and overlapping marginal violations.
5. `fig05_method_progression`: deployment and oracle-union coverage for v2, v3, and v4.
6. `fig06_pipeline_stop`: the completed evidence chain, failed method-dev gate, and blocked or
   locked downstream stages.

`figures_manifest.v4.json` binds every input and output by SHA-256. Rebuild and verify with:

```bash
python3 paper/tools/build_figures.py
python3 paper/tools/build_figures.py --check
```

The generator uses only the Python standard library. It reads the tracked public evidence package
and diagnostic receipts, rejects paths containing `sealed`, and does not access the semantic-sealed
payload.
