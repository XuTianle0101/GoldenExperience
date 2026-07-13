# Selective KV v5 Pipeline Workspace

The v5 pipeline uses one immutable workspace config for all four registered Qwen3
directions. The config binds the frozen benchmark, every split hash, exact model/config/
tokenizer/weight identities, the executable source-tree hash, and the sealed payload hash.
Local model and manifest paths are execution locations; the stable `pipeline_id` is derived
only from content identities so equivalent workspaces have the same semantic identity.

## Initialize

Create the workspace only after `golden-publication-benchmark validate` succeeds:

```bash
golden-v5-pipeline init \
  --workspace artifacts/v5_pipeline \
  --benchmark-manifest artifacts/publication/benchmark.json \
  --repository-root .

golden-v5-pipeline status --workspace artifacts/v5_pipeline
```

Initialization hashes every model shard. The stat-guarded identity cache under
`.pipeline/model_identity_cache.json` avoids repeating that full pass while the files remain
unchanged. `--refresh-identity` forces a new full hash pass.

There is deliberately no CLI option for a semantic sealed payload. Initialization records
only its expected hash and publishes `.pipeline/semantic_sealed.locked.json`. The generic
resume API rejects the `semantic_sealed` stage; a later one-shot guard must first verify
completed validation receipts for all four directions.

## Stage Graph

```text
collect_transport_train -> fit_transport -----------+
                                                     +-> evaluate_method_dev --+
collect_method_dev ---------------------------------+                         |
                                                                               +-> validate
collect_selector_train -> fit_risk -----------------+                         |
                                                     +-> calibrate ------------+
collect_risk_calibration ---------------------------+
collect_validation -----------------------------------------------------------+

all four validate -> one-shot semantic_sealed
semantic_sealed + collect_runtime_audit -> runtime_audit
```

Collection of `semantic_sealed_test` is not a normal stage. The other six public split
collections are independently resumable and are each bound to the corresponding split hash.

## Filesystem Contract

```text
workspace/
  .pipeline/
    config.json                       immutable, mode 0444
    state.json                        lock-serialized mutable state
    state.lock
    semantic_sealed.locked.json       immutable until guarded one-shot access
  objects/<sha256-prefix>/<sha256>.*  immutable content-addressed outputs
  receipts/<direction>/<stage>/<sha256>.json
```

Stage completion first copies and verifies every output into `objects/`, then publishes an
immutable receipt, and only then changes the mutable state to `completed`. A crash before the
last step leaves the stage resumable and may leave only harmless content-addressed objects.
Completed stages are idempotently reused when their input binding is unchanged. Changing a
completed stage's parameters requires a new workspace, preventing silent evidence drift.

Dependency bindings include the upstream stage input plus logical output hashes and sizes.
They exclude timestamps, attempt ids, receipt locations, and host paths, so equivalent runs
produce the same downstream input identity even though their audit receipts differ.

Every state read checks config ownership, stage keys, receipt hashes, object stat identity,
read-only mode, and workspace-relative paths. Consumers request a full object checksum before
loading a dependency. Corrupt, replaced, writable, missing, or symlink-escaping objects fail
closed.
