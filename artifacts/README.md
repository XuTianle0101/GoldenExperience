# Artifacts

This directory is reserved for generated experiment assets.

- `patch_manifest.md`: rendered GoldenExperience LMCache patch contract.
- `kv_baseline/`: same-model KV offload/reuse manifests; raw run directories are ignored.
- `results/`: per-request latency, reuse, fallback, and quality logs.
- `calibration/`: manifests or small metadata for projection/translator calibration.
- `figures/`: generated paper figures.

Large generated files are ignored by default. Keep only small metadata, manifests, or
curated result summaries in version control.
