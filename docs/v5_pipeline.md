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

## Fit Transport Candidates

Structure screening is registered only on Qwen3 4B-to-8B. The production command exposes no
rank, seed, loss, epoch, or optimizer override:

```bash
golden-v5-pipeline fit-transport \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_4b_to_8b \
  --device cuda:1
```

The stage trains the Cartesian product of ranks 32/64/128 and seeds 17/29/43. It fixes seed
17 for later deployment, source window 3, three epochs, AdamW at `3e-4` with weight decay
`1e-4`, gradient accumulation 8, gradient clipping at 1.0, and the manifest loss contract.
All nine candidates consume each trace shard synchronously, so a shard is loaded once per
epoch rather than once per candidate. The native-generation and prompt-tail values in a
trace are bounded reference constants; they are included in the registered reported total,
while gradients come from attention-logit KL, attention-output MSE, and transformed-KV
anchor loss.

Normalizers are fit once over transport-train traces. Checkpoint generations atomically bind
the trace manifest, complete training parameters, exact sample/optimizer boundary, model
state, AdamW moments and step counters, and finite cumulative metrics for every candidate.
The pointer is published only after the full nine-file generation exists. `--resume` rejects
partial generations, path or symlink escapes, changed hashes, missing optimizer tensors,
changed hyperparameters, fractional progress, and non-finite state. Runtime weights use the
same metadata contract as a final v5 manifest.

Generated safetensors headers are canonicalized before hashing. This removes serialization
order from trace, checkpoint, normalizer, and weight identities: an interrupted-and-resumed
deterministic run produces the same weight bytes as an uninterrupted run.

The one-shard real-model diagnostic and synthetic resume tests establish only executable and
checkpoint correctness. They do not replace the registered 4,096-sample transport run,
method-dev screening, four-direction validation, or any publication evidence.

## Freeze Structure On Method Dev

After all nine candidates and the independent 1,024-example method-dev traces exist, run:

```bash
golden-v5-pipeline evaluate-method-dev \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_4b_to_8b \
  --samples datasets/publication/method_dev.jsonl \
  --source-device cuda:0 \
  --target-device cuda:1
```

The evaluator rechecks that the raw store is byte-identical to the store bound by trace
collection. For each prompt it performs source and native-target prefix prefill once, then
evaluates every rank/seed transport with a fixed 16-token target continuation. Evaluation is
declared per sample using one of the registered deterministic scorers: exact match,
containment, token F1, numeric exact match with explicit tolerances, structural JSON exact
match, or resource-limited Python tests. Code evaluation runs in an isolated interpreter with
AST import/introspection restrictions, an import allowlist, audit-hook I/O/network/process
blocking, and CPU/address-space/file/process limits. Unknown metrics, options, references,
and thresholds fail before model execution.

`task_score` is semantic preservation, `1 - max(0, native_score - bridge_score)`, averaged
over prompts; it is not native task accuracy. Oracle-safe coverage applies the frozen unsafe
label per prompt: no native-pass regression, at least 0.98 greedy-token agreement, and at
most 2% teacher-forced perplexity drift. Transform time is synchronized on the target device.

The three seeds are aggregated independently for each rank using arithmetic means and
population standard deviations. Rank selection uses the registered lexicographic order:
mean task preservation, mean oracle-safe coverage, mean greedy agreement, then lower mean
P95 transform time. Only seed 17 from the selected rank becomes the deployment artifact.
There is no CLI override for generation length, seed aggregation, selection order, rank, or
deployment seed.

Every completed sample has a stage-input-bound checkpoint. The final detailed report binds
all 9,216 measurements; the frozen structure receipt binds that report, the raw store,
method-dev trace manifest, transport-fit manifest, benchmark, code, selected rank, seed-17
weight object, and deployment quality. Downstream directions must consume this receipt.

A real-model single-prompt diagnostic has exercised prefix prefill, transform, free greedy
generation, semantic scoring, and teacher NLL. Its deliberately undertrained one-step
candidate failed the safety metrics, as expected; this is implementation evidence only and
is not included in method-dev results or used to change the registered selection rule.

## Fit The Other Directions

Once the frozen structure receipt exists, collect `transport_train` for each remaining
direction and invoke the same fit command, for example:

```bash
golden-v5-pipeline fit-transport \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_8b_to_14b \
  --device cuda:1
```

For `qwen3_8b_to_4b`, `qwen3_8b_to_14b`, and `qwen3_14b_to_8b`, the command trains exactly
one candidate: the method-dev-selected rank, source window 3, deployment seed 17, and the
same three-epoch AdamW/loss contract. These directions never repeat rank or seed selection.
The pipeline state itself enforces a cross-direction dependency on the completed 4B-to-8B
`evaluate_method_dev` output, and the directional fit manifest binds the semantic structure
receipt hash. Calling the generic stage API cannot bypass this ordering.

Each directional manifest carries one content-addressed runtime weight object and its
direction-specific normalizer/training metrics. Resume uses the same full model, optimizer,
metric, progress, and input-binding checks as screening fit. A changed selected rank, seed,
window, optimizer, loss, train trace, code hash, or structure receipt fails closed.

## Fit Selector-Only Risk Predictors

After the deployment transport and the independent 2,048-example `selector_train` trace
exist for a direction, fit its uncalibrated source-side ranker:

```bash
golden-v5-pipeline fit-risk \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_4b_to_8b \
  --samples datasets/publication/selector_train.jsonl \
  --source-device cuda:0 \
  --target-device cuda:1 \
  --predictor-device cuda:1
```

The command must be run independently for all four directions. It consumes only
`selector_train`; neither calibration nor validation rows are accepted by this stage. For
4B-to-8B it uses the method-dev-selected rank at deployment seed 17. Every other direction
uses its single frozen directional candidate. The CLI exposes no rank, seed, threshold,
label, hidden-width, epoch, optimizer, or calibration override.

Each example reconstructs the production 169-dimensional feature vector only after a real
source-side sidecar serialization/deserialization round trip. Target-native and transported
continuations are each fixed at 16 greedy tokens and are retained only as hashes and numeric
labels in the report. A row is unsafe when a native-pass becomes a bridge-fail, greedy token
agreement is below 0.98, or teacher-forced perplexity drift exceeds 2%. Task scoring uses the
same frozen deterministic evaluator declared by the sample.

History features are causal: rows are processed in the frozen split's lexicographic sample-id
order, and each prefix group sees only the count, failures, and mean greedy agreement of its
strictly earlier rows. A checkpoint binds this exact history plus the split, trace, raw store,
transport, code, predictions, token sequences, and quantized sidecar hashes. Resume rejects a
checkpoint whose reconstructed history differs; a resumed run therefore produces the same
canonical report and predictor bytes as an uninterrupted run.

The predictor contract is fixed to seed 17, 200 full-batch epochs, AdamW at `1e-3` with
weight decay `1e-4`, and a 169-to-64-to-1 SiLU MLP. Both safe and unsafe selector examples are
required. Training may run on the selected predictor device, while artifact metrics are
recomputed on CPU and include the class-weighted training objective, unweighted log loss,
0.5-threshold accuracy, and tie-aware ROC-AUC. The canonical safetensors artifact contains
only the six runtime tensors and exact `feature_schema_version`/`hidden_size=64` metadata.

The resulting manifest explicitly records `calibrated=false`; this stage ranks risk but
cannot authorize reuse. Its report, predictor, selected transport object, raw sample store,
trace manifest, split hash, code hash, and pipeline identity are all content-bound. The later
`calibrate` stage alone may choose an admission threshold on `risk_calibration`.

## Calibrate The Frozen Predictor

After collecting the independent 2,048-example calibration trace for the same direction,
freeze its admission threshold:

```bash
golden-v5-pipeline calibrate \
  --workspace artifacts/v5_pipeline \
  --direction qwen3_4b_to_8b \
  --samples datasets/publication/risk_calibration.jsonl \
  --source-device cuda:0 \
  --target-device cuda:1
```

Calibration reloads the selector-trained predictor by its immutable object hash and always
scores it on CPU. It never updates predictor tensors and accepts only `risk_calibration` raw
rows/traces. Label generation, quantized sidecar round trips, and per-prefix causal history
use the same frozen evaluator as predictor fitting. The command exposes no threshold,
confidence, accepted-count, risk-bound, predictor-device, or label override.

Every distinct predictor score that would accept at least 300 rows is a candidate threshold.
All rows tied at a score enter or leave together. For each candidate, the stage computes an
exact one-sided Clopper-Pearson regression-risk upper bound using pointwise confidence
`1 - (1 - 0.95) / candidate_count`; this Bonferroni correction provides simultaneous 95%
family-wise coverage across the complete eligible threshold search. The selected threshold
maximizes coverage subject to at least 300 accepted rows and an upper bound no greater than
1%. If no threshold satisfies both constraints, the stage fails without a calibrated
artifact.

Each sample checkpoint binds the stage input, current causal history, frozen predictor,
transport, trace, raw store, and target-derived label. The detailed report records every
source-only probability and unsafe label. Loading a completed calibration replays all 2,048
predictor scores on CPU, reconstructs histories in frozen sample order, repeats the complete
threshold search, and requires exact agreement with the summary. This prevents a modified
threshold, count, tie group, or risk bound from becoming authoritative through summary-only
metadata.

The calibration manifest contains a runtime `RiskGateSpec` with fixed sidecar feature schema,
hidden width 64, OOD threshold 6.0, one required prior shadow sample, the predictor object,
split/report/evaluator hashes, selected threshold, all counts, coverage, simultaneous upper
bound, method, and confidence. It records `calibrated=true`, but remains only validation input;
it does not grant semantic-sealed or production authority.

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
    work/<direction>/fit_transport/   resumable, non-authoritative checkpoints
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
