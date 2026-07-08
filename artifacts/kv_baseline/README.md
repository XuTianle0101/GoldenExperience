# KV Baseline Artifacts

`artifacts/kv_baseline/` is for generated same-model KV offload/reuse runs. Raw run
directories are intentionally ignored by Git because Mooncake/LMCache KV cache files can be
multi-GB and are often machine-specific.

## Git Policy

Keep these small files in Git:

- `README.md`: this policy and restore notes.
- `manifests/*.json`: curated proof/seed manifests exported from successful runs.

Do not keep these raw generated files in Git:

- `artifacts/kv_baseline/<run_id>/cache/**` or `cache/mooncake/**`.
- Full LMCache/vLLM/Mooncake logs, pid files, lookup hash logs, temporary offsets.
- Failed experiment outputs; delete failed run directories after collecting diagnostics.

## External Artifact Policy

Store large KV seed payloads outside the repository:

- Local development: `/ssd/ge-kv-seeds/<artifact_id>/`, shared NVMe, or NFS.
- Team reproduction: S3, MinIO, SSH/NFS artifact share, or a DVC remote.
- Public release: release tarball or DVC-managed payload. Avoid normal Git; use Git LFS only
  for rare curated bundles, not frequent raw Mooncake object directories.

A useful seed payload should contain the Mooncake storage root plus the small run metadata
needed to restore it:

```text
<seed>/
  cache/mooncake/...
  metadata.json
  lmc_config.yaml
  disk_prompt.json
  summary.json
  metrics/offload.prom
  metrics/reuse.prom
```

Export a tracked manifest for a successful run:

```bash
python3 scripts/kv_baseline/export_kv_seed_manifest.py \
  artifacts/kv_baseline/<run_id> \
  --artifact-uri s3://bucket/ge-kv-seeds/<artifact_id>.tar.zst \
  --output artifacts/kv_baseline/manifests/<run_id>.json
```

## Reuse Caveat

The current Mooncake Python adapter avoids native `batchIsExist` by using an LMCache MP
process-local key index. A raw Mooncake cache can be reused immediately while the same
LMCache MP process stays alive and only vLLM restarts. Reusing a restored cache after an
LMCache MP restart requires a persistent key-index sidecar, which should be exported with
future seed bundles.
