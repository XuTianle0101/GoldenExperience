# Target-Logit Transport Training v3

This document freezes the next transport-training method after the registered v2
method-dev failure. It does not change the runtime tensor contract, method-dev split,
candidate ordering, or safety thresholds.

## Scope

The runtime structure remains `head_aware_transport_v2`: train-only normalization,
three-layer/head mixing, and an independent low-rank affine map for each target K/V
head. The v3 name refers to the training method and fit-manifest schema, not a new
runtime operator. Existing v1 and v2 artifacts remain readable.

The registered screening matrix remains ranks 32/64/128 crossed with seeds 17/29/43.
Seed 17 remains the deployment seed. Ridge ratio `1e-3`, three epochs, AdamW at
`3e-4`, weight decay `1e-4`, accumulation 8, clipping 1.0, and the five existing loss
weights remain fixed.

## Differentiable Generation Supervision

For each `transport_train` row, the frozen trace supplies source and native-target KV
at the same at-most-256 stratified prefix positions. For candidate `c`, the trainable
transport produces:

```text
T_hat_c = transport_c(source_sample, absolute_key_positions)
```

The raw suffix/query is tokenized with the pipeline-bound target tokenizer. Suffixes of
at most 256 tokens are used in full. Longer suffixes use the first 128 and last 128
tokens while retaining their original absolute positions. The request is rejected if
the untruncated prefix, suffix, and 16 teacher tokens exceed the model position
contract.

The frozen target model first consumes the bounded suffix with the native sampled
target KV as its past cache. It then generates exactly 16 greedy tokens, matching the
registered method-dev continuation horizon. The logits at those autoregressive steps
are detached as the teacher distribution. All
teacher tensors are derived only from `transport_train` and may be cached in host
memory; they are never model parameters or publication labels.

All candidates for a row are stacked on the batch dimension. One teacher-forced target
forward consumes the same bounded suffix and the first 15 native teacher tokens,
with each candidate's transformed sampled KV as the past cache. Target-model parameters
are frozen, but autograd remains enabled through the past KV. This defines two losses
that were constants in v2:

```text
native_generation       = mean CE(student_logits, native_greedy_tokens)
prompt_tail_distillation = mean KL(teacher_logits || student_logits)
```

The other three losses remain the prefix-attention logit KL, prefix-attention output
MSE, and transformed-KV anchor MSE. The frozen weights remain 1.0, 0.25, 0.5, 0.5,
and 0.1 respectively. Candidate losses are summed for one backward pass so batching
does not couple their parameters.

## Isolation And Identity

- `fit-transport` must receive the exact raw `transport_train` store already bound by
  trace collection. Its byte hash is checked before and after fitting.
- The target model, tokenizer, trace manifest, raw store, supervision parameters,
  source tree, optimizer boundary, normalizer, and ridge initializer are part of the
  stage/checkpoint identity.
- No method-dev, selector, calibration, validation, sealed, or runtime row is available
  to the generation supervisor.
- Teacher generation is deterministic greedy decoding in evaluation mode. Resume may
  recompute a teacher cache, but cannot change the bound training estimand.

## Registered Evaluation

The full 1,024-row method-dev matrix remains 9,216 measurements with a 16-token target
continuation. Rank selection remains lexicographic mean task preservation, oracle-safe
coverage, greedy agreement, and negative P95 transform time. Deployment still uses
seed 17 from the selected rank.

A row is unsafe if any of the following holds:

- a native task pass becomes a bridge failure;
- greedy agreement is below 0.98;
- teacher-forced perplexity drift exceeds 2%.

The frozen structure is published only if deployment oracle-safe coverage is at least
0.45. A failed gate is recorded as a negative result; thresholds are not tuned or
lowered. No downstream risk, calibration, sealed, or runtime stage may run before this
gate passes.

## Falsifiable Hypothesis

The v2 failure is attributed to a mismatch between prefix-only attention/KV fitting and
suffix-specific decoded behavior. V3 succeeds only if direct target-logit gradients lift
the unchanged full method-dev gate to at least 0.45. Rank/seed tuning, the biased
first-16-per-bucket pilot, or post-hoc selective reporting do not count as support.

The sampled-cache teacher is an explicit approximation: it conditions on the registered
256 prefix positions rather than a full native cache. Results must therefore report both
the training objective and the independent full-cache method-dev outcome.
