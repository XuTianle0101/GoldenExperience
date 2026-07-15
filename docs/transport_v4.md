# Full-Prefix Transport Training v4

This document preregistered the next transport-training method after the full v3
method-dev failure. The implementation amendment below was recorded before any v4 fit or
method-dev result was observed. It does not change the runtime transport structure, the
benchmark splits, the rank/seed selection rule, or any quality threshold.

## Motivation And Evidence Boundary

V3 made the generation losses differentiable, but conditioned both teacher and student
on at most 256 sampled prefix KV positions. The independent 1,024-row full-cache
method-dev run failed the unchanged oracle-safe coverage gate:

- v3 rank-128/seed-17 coverage was 115/1024 (`0.1123046875`) against the required
  `0.45`;
- the nine-candidate safe-set union was only 311/1024 (`0.3037109375`), so rank or seed
  selection cannot repair the failure;
- the v3 deployment candidate lost 171 rows that were safe under v2 and gained only 48;
- a post-failure diagnostic found that the sampled-cache 16-token native teacher exactly
  matched the full-cache native sequence on only 453/1024 rows;
- for 512/2048/8192-token prefixes, the match rate was 202/768 (`0.2630208333`);
- only 21 suffixes were truncated, so suffix truncation cannot explain at least 550 of
  the 571 teacher mismatches.

The complete evidence and source bindings are recorded in
`artifacts/publication_v5/development/v3_method_dev_diagnostic.json`. Method-dev is not
used as v4 training data or for threshold tuning. Its failed result supports one
structural correction: make the train-time teacher estimand identical to the deployed
full-prefix cache estimand.

## Frozen Runtime And Screening Contract

The runtime remains `head_aware_transport_v2`: train-only normalizers, a three-layer
source window, learned layer/head mixing, and independent low-rank affine K/V maps per
target layer and KV head. V4 names the training method and fit-manifest schema, not a new
runtime operator.

The screening matrix remains ranks 32/64/128 crossed with seeds 17/29/43. Seed 17 is the
deployment seed. Ridge ratio `1e-3`, three epochs, AdamW learning rate `3e-4`, weight
decay `1e-4`, accumulation 8, gradient clipping 1.0, and the existing five loss weights
remain unchanged. There are exactly 512 optimizer steps per epoch and 1,536 total.

## Full-Prefix Estimand

For every `transport_train` prefix group, the frozen source and target models prefill the
complete registered prefix of 128, 512, 2048, or 8192 tokens. The raw prefix is read only
from the already bound `transport_train` store. Tokenization must reproduce the collected
`token_ids_sha256`; model paths and weights must reproduce the pipeline identities.

For candidate `c`, the student cache is:

```text
T_hat_c = transport_c(full_source_KV, positions=0..prefix_tokens-1)
```

The native teacher consumes the same bounded suffix as v3, but its past cache is the
complete native target KV. The student consumes that suffix and the first 15 native greedy
teacher tokens with `T_hat_c` as its complete past cache. Both paths retain absolute suffix
positions. Suffixes up to 256 tokens remain intact; longer suffixes retain the absolute
first 128 and last 128 positions. The teacher horizon remains 16 tokens.

The differentiable terms remain:

```text
native_generation        = mean CE(student_logits, native_greedy_tokens)
prompt_tail_distillation = mean KL(native_logits || student_logits)
attention_logit_kl       = sampled-prefix attention KL
attention_output_mse     = sampled-prefix attention-output MSE
transformed_kv_anchor    = sampled-prefix native-target KV MSE
```

Their weights remain `1.0`, `0.25`, `0.5`, `0.5`, and `0.1`. The sampled trace is still
used for the three local alignment terms, but never as the past cache for either generation
path.

## Deterministic Grouped Optimization

Each epoch deterministically shuffles prefix groups and deterministically shuffles rows
within each group. The resulting 4,096-row sequence is divided into global consecutive
8-row accumulation windows. A window that crosses a prefix boundary is evaluated as two
or more prefix segments, but all segment losses are divided by eight and one optimizer
step occurs only after the complete window. This preserves 512 steps per epoch.

Within one prefix segment:

1. Reuse the immutable full source and native-target caches for the current group.
2. Generate each row's native full-cache teacher once and share it across candidates.
3. Process candidates in fixed microbatches of three.
4. Compute one activation-checkpointed full transport per candidate for the segment.
5. Accumulate target-model gradients through a detached full-cache proxy for each suffix.
6. Backpropagate the accumulated proxy gradient through the transport once, then add the
   row-weight-equivalent sampled alignment terms.

The proxy procedure is an exact application of the chain rule: target parameters remain
frozen, candidate parameters are disjoint, and the summed gradient with respect to the
transported cache is passed once to the transport graph. It changes memory and recomputation,
not the objective.

## Memory Contract

The longest registered train example has an 8192-token prefix and an 823-token raw suffix
(bounded to 256). A naive nine-candidate full transform failed after one candidate because
the source-window expansion required another 13.5 GiB. Activation checkpointing made all
nine transformed caches fit, but batch-nine target attention still exhausted 78.79 GiB.

The preregistered configuration uses non-reentrant activation checkpointing and candidate
microbatches of three. A pre-implementation tensor-level feasibility probe completed all nine
candidates at a 62.6875 GiB peak and produced finite proxy gradients. It established the
microbatch bound, but was not the final trainable-module memory measurement.

## Pre-Evaluation Implementation Amendment

The final implementation was frozen in commits `36af3e7` and `aace27c`, before creating a
v4 workspace or observing any v4 fit or method-dev metric. It makes the following numerical
and operational details explicit:

- full-prefix transport forward and recomputation use the target model's bfloat16 dtype;
  exported runtime weights and the frozen `head_aware_transport_v2` runtime remain unchanged;
- RoPE and source mixing operate in 256-token chunks under nested non-reentrant activation
  checkpointing;
- the structurally shared source-layer plan is gathered once per target layer and broadcast
  through learned head mixing, removing an exact eightfold duplicate expansion without changing
  the function;
- source and target native models occupy separate CUDA devices, defaulting to `cuda:0` and
  `cuda:1`;
- the run requires PyTorch's default native CUDA allocator (`native_default_v1`). In the pinned
  Torch 2.11.0/CUDA 13.0 stack, `expandable_segments:True` repeatedly caused illegal-memory
  failures on the real 8192-token path, while two independent default-allocator single-row runs
  and one complete 8-row window passed.

The implemented nine-candidate 8192-token single-row path peaked at 50.300 GiB on the target
A100. The complete 8-row accumulation window, including eight native teachers, proxy-gradient
accumulation, sampled alignment, clipping, and one AdamW update, peaked at 50.368 GiB. All 72
trainable parameter-gradient tensors were present and finite. These are implementation
diagnostics only; they do not replace the registered 4,096-row fit or method-dev evaluation.
Any implementation that exceeds the candidate microbatch, silently shortens the prefix,
detaches the past cache, substitutes sampled KV, or enables an unregistered allocator violates
the contract.

## Checkpoint And Identity Contract

V4 checkpoints must bind:

- the pipeline, source tree, benchmark, split, trace, and raw-store hashes;
- exact source/target model and tokenizer identities;
- the full generation-supervision specification;
- prefix-group and within-group order digests for every epoch;
- normalizer and ridge-initializer hashes;
- candidate rank/seed identities, model tensors, AdamW moments, finite metric sums, and the
  exact 8-row optimizer boundary;
- activation-checkpoint mode, candidate microbatch size three, and the full-prefix cache
  mode;
- the `native_default_v1` allocator identity and the executable code hash that binds bfloat16
  compute and 256-token transport chunks.

Resume may recompute deterministic source/target prefixes and teachers. It must reject a
changed group plan, sample order, prefix token hash, cache mode, supervision parameter,
optimizer boundary, or artifact identity. Existing v1/v2/v3 fit manifests remain readable,
but no earlier checkpoint may be resumed as v4.

## Registered Evaluation And Failure Rule

After implementation tests and a fresh content-bound workspace, v4 must rerun the complete
4,096-row fit and the independent 1,024 x 9 method-dev matrix. Selection remains the
registered lexicographic mean task preservation, oracle-safe coverage, greedy agreement,
and negative P95 transform time. Deployment remains seed 17 of the selected rank.

A row remains unsafe if a native task pass becomes a bridge failure, greedy agreement is
below 0.98, or perplexity drift exceeds 2%. The structure is publishable only if deployment
oracle-safe coverage is at least 0.45. A second failure is recorded as negative evidence;
it does not authorize a threshold change, a biased pilot, rank/seed cherry-picking, or access
to selector, calibration, validation, sealed, or runtime splits.

## Recorded Outcome

The registered v4 run completed on 2026-07-15 in pipeline
`v5-pipeline-1c6fed3dc231893debb58298`, with executable source hash
`b3d0dcb81e5a528937c1a80858273e2e8f8b1876be3d3691e222959867ef2760`.
All nine rank/seed candidates consumed 12,288 training rows (4,096 rows over three epochs)
and 1,536 optimizer steps. The fit manifest content hash is
`7195d0cf59f0c8995ce4065a42587733597d7ab3861c79e4c347f8a5e11e80a0`.

The complete 1,024-prompt method-dev matrix then produced 9,216 measurements. The frozen
ordering selected rank 64, so the deployment candidate remained seed 17 at that rank. It
preserved task score at `0.9768618035`, but greedy agreement was only `0.6172485352`,
aggregate perplexity drift was `21.47483717%`, and oracle-safe coverage was
`142/1024 = 0.138671875`. The unchanged `0.45` coverage gate therefore failed.

This was not a seed-selection accident. Only 377 of 1,024 prompts were safe for at least
one of all nine candidates, so even a prohibited per-row oracle over rank and seed reached
only `0.3681640625`, below the registered gate. Of the selected deployment candidate's 882
unsafe rows, 735 violated greedy agreement and 865 violated perplexity drift; 695 violated
both. All four token buckets failed, with safe counts of 30, 36, 34, and 42 for prefix
lengths 128, 512, 2,048, and 8,192 respectively.

The full-prefix correction nevertheless moved the mechanism in the predicted direction.
For rank 128/seed 17 held fixed across v3 and v4, safe count increased from 115 to 159;
the 8,192-token bucket contributed 24 of those additional safe rows. This supports the v3
diagnosis that sampled-prefix teacher mismatch mattered, but also shows that teacher-estimand
alignment alone is insufficient for the fixed low-rank affine runtime operator.

The audited, non-authoritative failure summary is
`artifacts/publication_v5/stages/qwen3_4b_to_8b.evaluate_method_dev.v4.failed.json`; the
mechanism analysis is `artifacts/publication_v5/development/v4_method_dev_diagnostic.json`.
The report file hash is
`f35e9599cea4d56cb1d0a7fad888a7d1bf2cef2602c9f42950162de7662a4400`.
All 1,024 per-sample checkpoints were independently replayed against the report. In
accordance with the preregistered rule, no selector, calibration, validation, semantic-sealed,
or runtime stage is authorized for this workspace.
