# KV Baseline Artifacts

`artifacts/kv_baseline/` is for generated same-model KV offload/reuse runs. Raw run
directories are intentionally ignored by Git because Mooncake/LMCache KV cache files can be
multi-GB and are often machine-specific.

## Git Policy

Keep these small files in Git:

- `README.md`: this policy and restore notes.
- `manifests/*.json`: curated proof/seed manifests exported from successful runs.

Historical local manifests whose payloads no longer exist were consolidated into
`docs/paper_outline.md` and removed. New manifests should be committed only when their
external payload URI is durable or when they are required by the active evaluation.

## Local Cleanup Exception

`qwen3_8b_cost_seed_20260713T0245Z/` is intentionally retained outside Git while the
Mooncake physical-reclamation defect remains unresolved. Its 6.69 GB of backing files
are the reproduction evidence for `artifacts/cached_kv/runtime_cost_8b_to_14b_20260713.json`.
Do not treat this local payload as approval evidence or delete it as a routine cache.

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

The Mooncake Python adapter avoids native `batchIsExist` by using an LMCache MP key index.
For normal same-process MP baselines this index is process-local. Cross-model materializer
injection can also provide a persistent sidecar through `GE_MOONCAKE_EXTERNAL_INDEX`, which
LMCache refreshes during lookup. Export that sidecar with any materialized cross-model seed
bundle that must survive an LMCache MP restart.
