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

The frozen benchmark must have been created with `--tokenizer-model`; file-level tokenizer
hashes from older development manifests are intentionally incompatible with pipeline model
identities.

## Collect A Split

Each non-sealed split has a separate raw JSONL store. This is deliberate: collecting
`transport_train` cannot even parse validation or calibration content from the same file.

```bash
golden-v5-pipeline collect \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_4b_to_8b \
  --split transport_train \
  --samples datasets/publication/transport_train.jsonl
```

Every JSONL row uses `goldenexperience.publication_raw_sample.v1` and contains
`sample_id`, `prefix_text`, `suffix_query`, `reference`, `evaluation`, and provenance-only
`provenance`. The hash-only benchmark record binds prefix text, suffix/query text, task,
reference, and evaluation settings. The raw-store file hash additionally binds provenance.
The collector rejects missing, duplicate, foreign-split, hash-mismatched, and sealed rows.

For every request, the real collector runs source and target prefix prefill, converts both
DynamicCache objects to `[K/V, layer, head, sampled_key, head_dim]`, captures bounded target
queries, and stores two attention outputs: one recomputed over the exact sampled-key domain
used by the training loss, and the full native attention output retained as a diagnostic.
It also records the bounded native-generation and prompt-tail constants without storing raw
token ids or logits.

Each sample is immediately published as an immutable safetensors object, followed by a
mutable local checkpoint that binds the stage input and full object stat/hash identity. Use
`--resume` after an interruption; verified samples are skipped. A completed stage emits one
content-addressed trace manifest covering exactly the registered split count. `collect`
offers no `semantic_sealed_test` choice and rechecks the executable source-tree hash before
loading any model.

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
