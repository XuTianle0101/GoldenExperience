# Paper

`paper.md` is the complete artifact-backed manuscript for the publication-v5 terminal negative
result. `references.bib` provides machine-readable bibliography entries, while the manuscript also
contains a rendered numbered reference list so it remains self-contained in plain Markdown.

The empirical claim is intentionally narrow: full-prefix supervision improves a fixed
Qwen3-4B-to-8B candidate, but the registered v4 method fails the method-development coverage gate.
The manuscript contains no approved selector, calibration, other-direction, validation, sealed,
or cross-model runtime claim.

## Reproduce Evidence

```bash
python3 paper/tools/build_method_dev_evidence.py --check
python3 paper/tools/build_figures.py --check
```

These commands rebuild all paper tables and figures in memory, compare them byte-for-byte with the
tracked artifacts, and reject any path containing `sealed`.

## Render

The Markdown file renders directly on Git hosting. With Pandoc installed, a standalone HTML copy
can be built without changing the source:

```bash
pandoc --standalone paper/paper.md --output paper/paper.html
```

The repository tracks the manuscript source and deterministic figure PDFs, not a
machine-dependent rendered manuscript. The final project audit runs tests, lint, type checks,
package builds, link checks, evidence regeneration, source-identity verification, and a locked
sealed-state check.

## Evidence Boundary

- Authoritative fit evidence: `artifacts/publication_v5/stages/`.
- Terminal negative-result package: `artifacts/publication_v5/evidence/`.
- Reproducible plotted values and vectors: `artifacts/publication_v5/figures/`.
- Preregistered training method: `docs/transport_v4.md`.
- Pipeline authority and terminal status: `docs/v5_pipeline.md`.
- Claim-scoped prior-art audit: `docs/related_work_matrix.md`.

The semantic payload is not a paper input and must remain unopened.
