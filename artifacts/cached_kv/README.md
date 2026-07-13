# Qwen3 Cached-KV Bridge Artifacts

This directory contains small manifests and curated result summaries. Generated
`.safetensors` weights are ignored by Git.

`bidirectional_pipeline_smoke_20260710.json` verifies that both Qwen3 directions execute
the real cached-KV collection, fitting, RoPE translation, and target DynamicCache decode
path. It is deliberately not a calibration artifact: the run used rank 2, four training
positions, one validation prompt, and no sealed test or runtime cost evidence. Both
directions fail the production quality thresholds.

`bidirectional_validation_sweep_20260710.json` records validation-only rank 256/512
results against the bounded non-thinking chat contract. Native task assertions pass, but
both bridge directions remain far below the accuracy thresholds. The sealed test remains
unopened and no candidate weights are checked in.

`scaled_baseline_ab_20260710.json` compares the v1 fixed interpolation baseline with the
v2 learned per-channel baseline under identical rank, data, and prompt settings. v2
improves both directions, but still fails the end-to-end accuracy gates.

`regularization_capacity_sweep_20260710.json` records the validation-only ridge and rank
sweep after the v2 scaled baseline. Stronger regularization and more supervised positions
substantially improve both directions, but the best candidates remain below every
production output-quality threshold.

`expanded_corpus_nonlinear_ab_20260710.json` records the 256-train/64-validation corpus
and compares the v3 linear map with the v4 SiLU correction. v4 improves generation in
both directions, especially 8B to 14B, but neither direction is approved.

`cka_layer_alignment_ab_20260710.json` compares fixed normalized-depth source windows
with train-only monotonic linear-CKA alignment. CKA improves forward generation metrics
but regresses reverse generation and tensor metrics, so it remains an opt-in experiment.

`validation_candidate_8b_to_14b_20260713.json` and
`validation_candidate_14b_to_8b_20260713.json` record the first full bidirectional
validation candidates emitted by the production-gated training path. The rank-512 v4
bridges are bound to content-addressed local Qwen3-8B and Qwen3-14B weights. The reverse
bridge preserves task answers better than the forward bridge, but both fail every
output-quality gate and remain validation-only. Raw candidate manifests, per-prompt
results, and generated weights stay local and are ignored by Git.

`runtime_cost_8b_to_14b_20260713.json` binds 20 isolated Qwen3-14B native-prefill
samples to 20 real Mooncake read-transform-put samples over the same 1776-token prefix.
The materialization P95 is 3.44x native prefill, versus the 0.70x limit. Mooncake also
leaves unaddressable backing files after API rollback, so the cost evidence is explicitly
ineligible for approval even though every temporary object key was removed.

`logit_refinement_ab_8b_to_14b_20260713.json` records a rejected first attempt to
fine-tune all 83,968,000 bridge up-projection and bias parameters through frozen Qwen3-14B
logits. One 16-prompt pass at learning rate 1e-4 reduces validation task score from
0.1875 to 0.015625 and greedy continuation match from 0.1572 to 0.0498. The experiment
remains validation-only and is not selected as the default bridge.

`logit_refinement_parameter_screen_8b_to_14b_20260713.json` compares constrained
bias-only and nonlinear-up-only refinement with lower learning rates, transformed-KV
anchoring, and train-only holdout checkpoint selection. Nonlinear-up-only improves the
reference validation task score from 0.1875 to 0.21875 and perplexity drift from 26.39%
to 13.21%, but remains far below approval thresholds. Because each arm independently
refit a randomized baseline, the result is screening evidence pending a seeded,
same-fit pre/post confirmation.

`logit_refinement_paired_confirmation_8b_to_14b_20260713.json` removes that
confounder with seed 17 and full pre/post validation on one fitted state. Eight
nonlinear-up-only steps improve task score by 0.0625, greedy continuation match by
0.0518, and perplexity drift by 11.67 percentage points without materially changing
KV cosine. All four new task passes are code prompts in the 32/128-token buckets;
512/2048-token task score and prose/chat task score remain zero. The method is therefore
confirmed as directionally useful but is not selected as a production bridge.

`native_generation_refinement_8b_to_14b_20260713.json` aligns the refinement teacher
with the native greedy continuation used by the output-quality gate. Four seeded,
nonlinear-up-only steps improve validation task score from 0.09375 to 0.90625 and
greedy continuation match from 0.0820 to 0.8408. The run adds 52 task passes with no
regressions and generalizes to all four token buckets. Tensor cosine, next-token
agreement, perplexity drift, and remaining output metrics still fail production gates,
so the result remains validation-only and the sealed test stays unopened.

`native_generation_steps_ab_8b_to_14b_20260713.json` compares four and eight
generation-aligned steps under the same seed and rank. Eight steps slightly improve
greedy match and perplexity drift but reduce task score from 0.90625 to 0.890625 by
changing one correct exact answer into the wrong marker. Because the four-prompt
teacher-forced holdout continues improving through that regression, four steps remain
the selected validation configuration and further step scaling is stopped.

`mixed_refinement_8b_to_14b_20260713.json` adds a 0.25-weight prompt-tail loss to the
four-step native-generation objective and selects checkpoints with 16 train-only,
free-running holdout prompts. Relative to pure native-generation it preserves the
0.90625 validation task score while increasing next-token agreement by 0.00586 and
reducing perplexity drift by another 0.49 percentage points. The absolute quality gates
still fail, so the experiment remains validation-only.

`mixed_refinement_14b_to_8b_20260713.json` applies the same seed-17, rank-512,
four-step mixed experiment in the reverse direction. On the paired fitted state it
preserves the 0.84375 validation task score, raises greedy continuation match by 0.10156,
and reduces perplexity drift by 5.32 percentage points. It creates no new validation
task passes, and every production quality gate remains failed, so the reverse bridge is
also validation-only and the sealed test remains unopened.

`bidirectional_mixed_refinement_summary_20260713.json` closes this validation phase.
Greedy match, next-token agreement, and perplexity drift improve in both directions,
while task score improves only from 8B to 14B and merely holds steady from 14B to 8B.
Both directions fail seven quality checks. Consequently the sealed test stays closed,
and chunk batching, batch serialization, further cost optimization, and Mooncake
physical reclamation remain deferred.

Only a `CachedKVBridgeManifest` whose derived `approved` property is true may be used by
the runtime materializer. Missing held-out accuracy or Mooncake cost evidence keeps a
manifest fail closed.
