# Artifacts

This directory separates curated, reviewable evidence from ignored runtime and training caches.
Artifact presence does not imply runtime authority; state and receipt contracts remain decisive.

## Publication v5

The current publication result is terminally negative for Qwen3-4B to Qwen3-8B v4:

| Location | Contents | Authority |
| --- | --- | --- |
| `publication_v5/stages/` | Verified fit/collection receipts and failed method-dev receipt | Stage-specific |
| `publication_v5/evidence/` | Compressed 9,216-measurement report and derived CSV tables | Negative result only |
| `publication_v5/figures/` | Deterministic CSV, accessible SVG, and vector PDF figures | Derived negative result |
| `publication_v5/development/` | Mechanism, implementation, cleanup, and prior-art diagnostics | Diagnostic only |
| `publication_v5/initialization_v4.json` | Immutable workspace/model/data/code identities | Initialization receipt |

The registered deployment candidate covers `142/1024 = 0.138671875` prompts, below the `0.45`
gate. Selector, calibration, other-direction, validation, semantic-sealed, and runtime stages are
blocked. No v5 `validation_candidate`, `semantic_approved`, or `approved` artifact exists.

Reproduce the tracked package from a clean clone:

```bash
python3 paper/tools/build_method_dev_evidence.py --check --from-archive
python3 paper/tools/build_figures.py --check
python3 paper/tools/check_manuscript.py
```

See `publication_v5/README.md` for the detailed evidence map.

## Other Curated Artifacts

- `patch_manifest.md`: rendered GoldenExperience LMCache patch contract.
- `kv_baseline/manifests/`: same-model KV offload/reuse manifests.
- `cached_kv/`: retained historical development summaries, not publication-v5 approval.
- `cross_model_runtime/manifests/`: curated historical runtime manifests when explicitly retained.

Raw output directories such as `cache/`, `results/`, and full KV baseline runs are ignored. Keep
large model weights, optimizer checkpoints, raw KV objects, service logs, and machine-specific
payloads outside Git. The compressed method-dev report is a deliberate publication exception: it
is 702,193 bytes, content-bound, and required to reproduce the negative result.

## Sealed Boundary

The semantic payload is not a publication artifact in this directory. Do not inspect, copy,
sample, hash, or package it. The public initialization receipt records only the locked state and
content identity needed by the one-shot protocol. The failed method-dev gate does not authorize
opening it.
