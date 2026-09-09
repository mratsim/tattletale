# SPEC: transformer suite checks

Check semantics for the transformer suites. PLAYBOOK.md carries the
how-to checklist. Modules: tolerance (assertions and budgets),
recording (fixture stats files), invariants (analytic properties),
provenance (environment rows), selftest (two-sided suite verification).

## Budget tiers

A `ToleranceBudget` is one budget: the tolerances one check allows, how
much difference it accepts. Tiers, in order of strictness:

- `ctBitExact` — deterministic weight math (codec, codebook, scales).
  Compared value-equal. These tensors are pure functions of recorded
  bytes, so any drift is a bug, not rounding.
- `ctElementWise` — same-device computed outputs. Compared with rtol and
  abstol caps. Same-device float math drifts 1-2 ulp (bench_rmsnorm
  measurement), so caps sit just above honest rounding and far below seeded
  fault sizes.
- `ctDistribution` — outputs compared through signatures (order
  statistics, binade-log histograms), match rates, and margin checks.
  No element-wise cap.

Computed outputs are never compared bit-exactly: 1-2 ulp same-device drift
is honest rounding. Recorded fixture files against recorded checksums are
the only
bit-exact comparisons.

## Budget table v0

band: the per-element tolerance `rtol * |expected| + abstol`.
band width: the measured drift of an element divided by its band.

Rows measured on the M4 Max (`defaultBudget` in the tolerance module).
The same-device budgets hold for Metal computations of the norm and rope
paths: the measured Metal replay is bit-exact against the cpu path
(the recorded measurement output of bench_metal_budget in the harness
module, maxUlp 0 over
262144 norm elements and the recorded rope tensors). The attention and
post-residual elementwise caps stay same-device budgets: the measured
Metal cross-device drift is absolute-scale, up to 29504 own ulps at
near-zero elements with a 3.3e-2 mismatch fraction past 4 ulps, while
the quantile fingerprints stay inside 1 ulp and the depth-1 chain band
contains the drift (usage 0.166 attention, 0.250 post-residual, same
measurement). Metal-computed outputs of composed paths therefore compare
under the chain checkpoint band and the greedy margin budgets, never
under these same-device elementwise caps. The device comparison lives in
`budgetRowClass` (harness/device.nim). CUDA budgets wait for the
reference box. The exl3 suites compare through the recorded-summary
surface: the bf16 bounds with the ulp unit taken in fp16, because EXL3
dequantizes to fp16, the recorded payloads coming from the production
CUDA kernel so only a CUDA replay is the bit-exact reference class.
rounding_rmsnorm stays box-only: its suites import the CUDA kernel
archive libpositron_cuda and fail to link on this mac.

| op class      | maxUlp | mismatch frac | extra checks           | measurement artifact
|===============|========|===============|========================|=========================
| obRmsNorm     | 2      | 0             | scale invariant        | bench_rmsnorm.nim (1-2 ulp f32 drift); Metal replay bit-exact (bench_metal_budget)
| obRope        | 2      | 0             | pair norms             | NEOX rotation identity; Metal replay bit-exact (bench_metal_budget)
| obAttention   | 4      | 1e-3          | softmax row sums       | softmax + SDPA rounding; Metal cross-device class: chain band (bench_metal_budget)
| obPostResidual| 4      | 1e-3          |                        | accumulated block drift; Metal cross-device class: chain band (bench_metal_budget)
| obChainCheckpoint | 2 ulp rtol + 2^-3 * depth * meanAbs abstol | 0 | flat-ratio drift scaling, mean drift | chain fixtures (bf16-02-first-8-layers-plus-final), 18-checkpoint Metal replay measurements (band usage 0.022-0.309)
| obLogits      | none   | none          | margin, truncated-KL, tail probability | greedy margins, measured Metal drift floors in the greedy suites

The norm budget cites bench_rmsnorm.nim: same-device f32 accumulation drifts
1-2 ulp there, so the bf16 cap sits at 2 ulp with zero mismatch
allowance. The attention and post-residual budgets accumulate more layers of
rounding and take 4 ulp with a 1e-3 mismatch fraction. The measured
Metal drift of one post-residual block, 0.250 band widths at depth 1,
matches the chain replay measurement of the same block.

## Chain checkpoints and drift scaling

Per-op budgets do not compose along a generation or prefill chain: honest
drift accumulates over the blocks, so a mid-chain checkpoint references
the chain budget, never a per-op budget. The chain checkpoints of the dense
ports are the 8+1 set: the outputs of decoder blocks 0..7 plus the tail,
the decoder stack output taken pre-final-norm. The prefix-8 is
class-complete for both models (0.6B: 28 uniform-attention layers,
0.8B:
24 layers in the period-4 linear x 3 + full pattern, so the prefix is
two full periods), and the tail validates the depth extrapolation to the
full stack.

The blocks between the prefix and the tail carry no per-block fixtures,
by design. The recorded prefix is unit-test territory. The deep blocks
are integration territory, exercised end to end by the greedy chains.
The consequence for late-block per-op regressions: they appear only
when they perturb generation or the tail checkpoint, and the tail
checkpoint band plus the drift-scaling check bound what such a
regression can do to the chain before that.

- Elementwise band: `rtol * |expected| + scale * depth * meanAbs(expected)`
  per element, zero mismatch fraction (`assertChainCheckpoint`). The
  relative term is two bf16 ulp, the honest spread of two correct
  implementations of one op. The absolute term anchors on the bulk scale
  of the recorded checkpoint (its mean absolute value, `meanAbsValue`)
  times the depth times the factor 2^-3.
- The band anchors on the bulk, not on the element's own magnitude:
  the honest cross-device drift of a checkpoint element is
  absolute-scale (measured: the per-binade maximum drift is one near
  constant across binades), because cancellation elements sit far below
  the activation magnitudes that produce them. A relative cap alone is
  blind exactly there, and quantile-ulp caps explode on near-zero
  medians (14 912 ulps observed) and the normalized histogram L1 sits an
  order of magnitude above the per-tensor budget.
- The drift accumulation is linear in depth, not a random walk: the
  per-block reordering differences are bounded but not zero-mean (they
  are deterministic given the inputs), so the worst case is a linear
  sum. The depth-constant of the band derives from the block structure:
  about 13 bf16-rounded intermediates on one element's path per block,
  each moving at most one own ulp, gives 13 * 2^-8 ~ 0.05 per
  block-depth. The safety factor 2.4 absorbs shallow checkpoints (depth 1-3),
  where the normalized internal activations exceed the residual bulk,
  and the factor rounds onto the binade grid at 2^-3.
- Validation measurements corroborate the derived band, they do not set it:
  the Metal replay over the 9 checkpoints of the 0.6B stack measures
  band widths 0.250, 0.309, 0.037, 0.028, 0.022, 0.035, 0.030, 0.050
  at depths 1..8 and 0.043 at the depth-28 tail, zero elements past one
  band everywhere (the budget this replaced measured 0.308 ... 1.111 and
  FAILED at depth 8, see the budget-change log). A checkpoint outside
  its budget is a flaw signal: stale recording, code defect, or wrong
  modeling assumption, and it is resolved by root-cause work, never by
  widening the budget.

### Budget-change procedure

Budgets change only through derivation. A valid change names
the modeling assumption that moved, re-derives the constant from the
new model, and shows the seeded fault corpus still rejecting at the
new budget. Fitting a constant to recorded measurements is forbidden:
the suite would become a description of the last recording instead of
a bound on honest rounding. The README FAQ carries the same rule in
longer form.

### Budget-change log

- Chain checkpoint absolute term, depth scaling: sqrt(depth) -> depth
  anchored on the bulk scale. The 9-point measurement (8 prefix
  checkpoints + the depth-28 tail) falsified the zero-mean random-walk
  assumption behind the sqrt model: the measured band widths grew from
  0.308 at depth 1 to 1.111 at depth 8 (one element past one band),
  11.2 at the tail, while a depth-flat band width is what the walk
  model predicts under the sqrt-scaled band. The per-binade drift
  report (the per-element count of ulp distances) showed the drift is
  absolute-scale and tracks the activation
  bulk (drift over meanAbs between 0.05 and 0.16 at every depth), so
  the per-block reordering differences are bounded but not zero-mean
  and accumulate linearly. Re-derived budget: abstol(depth) = 2^-3 *
  depth * meanAbs(expected). The rejection corpus (chain faults in the
  selftest) keeps rejecting at the new budget. The 3-point recordings
  of the previous budget could not discriminate the two depth models.
  The 9-point set can.
- Drift-scaling check: sqrt-model slack 1.6 -> flat ratio-to-first
  bound with slack 3. Under the depth-linear band the honest band width
  is bounded (measured ratios against the first checkpoint at or under
  1.25), so the flat bound separates honest drift from compounding
  growth: a sqrt-shaped compounding fault crosses the slack exactly at
  the depth-28 tail. A constant per-block bias is depth-linear like
  the honest drift and is invisible to the band and the scaling check
  by design, the mean-drift check polices it.
- Conv-state bit-equality -> derived conv drift budget (OPERATOR-REVIEW,
  pending operator sign-off). The evaluation-order suites demanded bit-equal
  conv states between the two evaluation orders, justified by "the conv
  is causal and both orders run the same conv op". The Metal evaluation-order suite
  measurement falsified the order-invariance assumption: on hardware
  that reorders the conv accumulation the two orders differ by exactly
  one bf16 quantization step at the differing element (0.0078125
  absolute = 0.25 ulp-units at the conv max 7.34375, because the
  differing element sits in a lower binade than the max). New budget:
  `assertConvEvalOrder`, two bf16 ulp-units at the tensor max,
  derived from the rounding model (one reordered f32 accumulator rounds
  to at most one bf16 step of the element, at most one ulp-unit at the
  max, safety factor 2), the measurement calibrates, it does not fit
  the constant. Rejection row: a conv drift past the budget rejects, a
  sub-budget drift accepts (selftest rows).
- Flat state band -> linear-in-decode-length drift bound
  (OPERATOR-REVIEW, pending operator sign-off). The evaluation-order f32 state assert used the flat
  `SsmUlpMargin` cap, four fp32 ulps at the pair max. The Metal T=70
  decode measures 1027.4 fp32 ulps at the pair max 1.032, about 15 fp32
  ulps per step: the cross-order state drift accumulates linearly in the
  decode length, so no flat cap can cover an honest run at a larger
  length. New budget: `assertSsmEvalOrder(a, b, steps)` with the bound
  abstol(steps) = 2^-3 * steps * meanAbs(reference), the
  chainCheckpointAbstol derivation shape reused unchanged, no new
  constants. Rejection row: a
  2x-bound state drift rejects, the measured-scale and just-below-bound
  drifts accept (selftest rows).

### Drift-scaling check

`checkDriftScaling` consumes the measured band widths of the chain
checkpoints and requires them bounded: every later checkpoint against
the first one, under the flat bound with slack 3 and floor 0.75 (the
floor covers an all-zero prefix, the reference device replay). Honest
drift stays bounded because the band grows with depth. A fault that
compounds per block grows its band width past the slack, a mild
sqrt-shaped growth crossing exactly at the tail. The chain suites
verify the check two-sided with synthetic drift sequences (bounded
growth accepted, compounding growth rejected).

### Mean-drift check

The band and the scaling check cannot see a constant per-block bias:
it is depth-linear exactly like the honest drift it hides among. The
mean-drift check polices that fault class on the signed mean of the
checkpoint elementwise differences (`assertChainMeanDrift`, also wired
into `assertChainCheckpoint`).

Derivation. The block internals accumulate in f32: the ATen linear on
bf16 inputs, the RMSNorm mean and rsqrt, and the Gated DeltaNet core
with its state. Consider the longest f32-accumulated reduction that
feeds one checkpoint element, the down projection over the
intermediate dimension, d = intermediate_size (3072 for the 0.6B
stack, 3584 for the 0.8B stack). Two correct reduction orders differ
by the classical backward-error bound of that reduction,

    |error_i| <= (d - 1) * eps_f32 * sum_k |term_ik|,

with eps_f32 = 2^-24 the f32 unit roundoff (`F32UnitRoundoff`). The
signed mean over the N checkpoint elements satisfies

    |mean drift| <= mean_i |error_i|
                 <= (d - 1) * eps_f32 * mean_i( sum_k |term_ik| )
                 <= (d - 1) * eps_f32 * sqrt(d) * bulk

where the last step is the inequality step from the per-term bounds
to the output scale: for zero-mean uncorrelated
reduction factors, the mean absolute term sum is (2/pi) * d *
sigma_w * sigma_h while the reduction output scale is sqrt(d) *
sigma_w * sigma_h, so the term sum runs about 0.8 * sqrt(d) times the
reduction output scale, and the residual stream bulk (the mean
absolute checkpoint value) covers that scale from above. The derived
bound is

    bound = ChainMeanDriftSafety * (d - 1) * eps_f32 * sqrt(d) * bulk

with safety factor 8 (`ChainMeanDriftSafety`), stacking coverage for
the inequality step (about 1.25x), the several composing reductions of
one block (about 2x), the product roundings inside the reductions
(about 2x), and the bf16 rounding at the f32-to-bf16 storage boundary
(its signed mean returns to the f32-scale difference, about 1.5x). Numerically the bound is
about 0.08 times the bulk.

Why CLT optimism is NOT used: the honest signed mean under
random-sign reordering scales like 1/sqrt(N) (measured: at or under
1.0e-4 of the bulk on every checkpoint of both ports), and a 1/sqrt(N)
scaled threshold would sit close to that noise, rejecting honest runs
on unlucky recordings. The bound stays at the worst case of legally
aligned reordering errors instead, which is exactly the shape a
coherent bias takes, so the check can never reject honest rounding
and catches coherent biases above about 8 percent of the bulk.

Corroborating measurement (Metal replay vs cpu recording, 9
checkpoints per port): the signed means measure between +9e-6 and
-1.3e-3, at or under 1.0e-4 of the bulk everywhere, three orders of
magnitude inside the bound.

## Signatures

The tolerance module computes a signature per tensor:

- **Quantiles**: exact order statistics of the ascending sort. Index
  rule: `floor(p · (n − 1))`, no interpolation, computed in f64.
  Fixed probabilities: 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95,
  0.99, plus min and max. Values stored as f32 (native dtype promoted).
  Order statistics are bit-exact cross-implementation. The Python
  twin agrees bit-exactly on the recorded fixture bytes
  (verified in the initial proof-of-concept recording run).
- **Histogram** (margin-critical tensors only): binade-log buckets over
  the bf16 bit pattern. Bucket = (sign, binade −64..63, mantissa top 6
  bits), 14-bit keys in sparse storage, plus dedicated zero and
  subnormal bins. Soft bucketing: each element contributes (2 − low) to
  its bucket and low to the neighbor, where low is the dropped mantissa
  bit — integer math, bit-exact cross-implementation. Bucket width is 4
  ulp (6 of 8 mantissa bits kept). NaN and +Inf raise an error. −Inf
  is whitelisted only where a mask invariant explains it (masked
  attention scores) and stays unbucketed.
- **Comparison**: `assertStats` checks every order statistic within
  `maxUlp` (counted in ulps of the recorded dtype) and the normalized
  histogram L1 within `histL1` (provisional 0.01 until the fault corpus
  measures the residual false-positive rate). `assertMatchRate` checks
  the element-wise fraction drifting past `maxUlp` against
  `maxMismatchFrac`.

### Sidecars

`<fixture>.stats.json.zst` frames sit next to recorded fixtures,
written by the gen_stats tool from fixture bytes (bootstrap: no
re-recording). The sidecar container is the zstd frame, level 19 with
content size and checksum, the byte format is content-exact: the
zstd frame carries the bytes and defines nothing, the JSON payload
inside is the format.
The reader, tests/harness.nim `zstdReadFixture`, resolves the frame path itself
or the bare stem without the container suffix, inflates the payload in
memory and never touches disk. A corrupt or content-size-unknown frame
raises ZstdError.
Schema: hex f32 bit patterns for quantiles, sparse `key:count` histogram
strings, one entry per tensor. Histograms are stored only on the margin-critical
tensor of each fixture family (the deepest chain checkpoint, the rotated
rope output, the layer output). Everything else carries quantile stats.
Sidecars stay under 64 KB per fixture family and add no raw tensor
payloads. Regenerate only after a sanctioned re-record (PLAYBOOK.md).
The descriptor sidecar class (`<fixture>.descriptors.json.zst`,
the `ttt-tf-003-tensor-stats-h2` format
with the descriptor keys) records the numeric summary of tensors
that left a payload at migration time. A sanctioned re-record
regenerates it from the fresh recording through the same writer.

### Format registry

The fixture wire formats carry a registry id in the payload's `schema`
field: `ttt-tf-<id>-<slug>-h2` (h2: harness generation 2). The loaders
accept exactly the three constants below (harness/tolerance.nim); no
legacy branch exists.

| registry id | content | retired predecessor |
|---|---|---|
| `ttt-tf-001-greedy-steps-h2` | greedy generation: per-step chosen token, top-32 support with f32 logits, argmax margin, softmax tail beyond the top-32 | `tt-greedy-2` |
| `ttt-tf-002-logit-decisions-probe-h2` | final logits: per-position argmax id, top-2 competing pair, margin, tail probability, plus the strided 512-word bit-exact probe row of every deciding row | `tt-final-logits-projection-2` (decisions + probe) and `tt-final-logits-projection-1` (decisions only, no probe) |
| `ttt-tf-003-tensor-stats-h2` | `.stats.json.zst` and `.descriptors.json.zst` sidecars: hex f32 quantile bits, sparse packed histogram buckets, descriptor keys only on descriptor-carried entries | the numeric `1` |

h1-era frames (the `schema` field absent, or the numeric `1`) are
retired as of this change: no such frame exists in the tree, and the
loaders accept exactly the three registry constants above.
New formats register at 004+; registration is an operator decision
recorded before any generator emits.

## Fixture contract and scaling model

The fixture budget must hold at 30 to 100 model families. The contract
(README "Internal and external consistency" carries the operator
statement. This section is the binding form):

- **Universal file cap: 256 kB** for every stored tensor file,
  regardless of model dims. State-like tensors never store full at any
  model size: they carry per-step descriptors (bulk mean, signed mean,
  tail probability, per-binade drift report, fingerprint) plus a
  512-word strided f32 probe.
- **Sequence length is free**: a recorded trace uses the minimum step
  count (2 to 3 steps) that exercises the property, and a lossless
  slice of an existing recording is preferred over a re-record.
- **Weights never appear as fixture content**: weights load from the
  checkpoint the suite points at.
- **Per-family payload targets**: ~1.5 MB dense, ~0.5 MB SSM.

Check tiers, in fallback order, per tensor:

```
tier 1  property check on synthetic inputs   -> no fixture bytes
tier 2  recorded summary + descriptors       -> one tiny twin trace
tier 3  full-tensor band check               -> named exception only
```

1. **Tier 1, property check on synthetic inputs** (zero recorded
   fixtures, any model): self-consistency of the two computation
   modes, e.g. the GDN evaluation-order checks (assertSsmEvalOrder, the
   budget-change record in the GDN section). No payload bytes at all.
2. **Tier 2, statistical external check** (one tiny recorded twin
   trace, 2 to 3 tokens): assertStats fingerprint plus
   assertDescriptors against the recorded descriptor entry.
3. **Tier 3, full-tensor check**: only with a named exception in the
   budget table. The chain checkpoint recordings are the sanctioned
   full-tensor class: the drift bound there depends on full-tensor band
   checks and the payloads are already small.

Any full-tensor payload outside the chain class needs a named
exception recorded next to the budget table. An unnamed exception defeats
the size cap: payload sizes can then grow without a recorded operator
decision.

## Future tests: chunk and page boundaries

Chunked prefill and paged KV caches introduce edge sizes. The sweeps to
carry once those paths exist:

- chunk 512: trace sizes 511, 512, 513
- page 16: trace sizes 15, 16, 17
- page 64: trace sizes 63, 64, 65

One model per attention type carries the sweep. The edges are
self-consistency questions between our own code paths, so they run on
the property tier:

- inputs are pattern-generated from a deterministic formula of the
  index, stored as a few pattern parameters. No input tensor is stored
- both paths receive the same pattern input and their outputs are
  compared directly (paged against flat KV, chunked against monolithic
  prefill)
- outputs shrink to descriptors: bulk mean, signed mean, tail
  probability, fingerprint, and a 512-word probe, a few kB per checked
  layer

Expected cost: tens of kB per attention type, against roughly 88 MB for
full recorded tensors at the chunk-512 sweep. The python twin can run
the same pattern inputs, so an external boundary check stores output
descriptors only.

Batch invariance (a batched forward equals the row-by-row forward) is
the candidate double-run: the property exists and one synthetic suite
covers it for a single block. No inference consumer needs the full-model
check yet, so the sweep stays a candidate until a consumer needs it.

Real-activation grounding stays separate: the recorded traces (a few
tokens) carry the drift statistics on real distributions. Pattern inputs
serve the index arithmetic and the masking edges, where the input
distribution does not matter.

## Invariants

Analytic properties checked on seeded random inputs (property mode) and
recorded fixtures (wiring mode). Slacks are explicit arguments with
defaults documented in the invariants module.

### RMSNorm definition

Identity, over the last dimension, recomputed in f32 from the same
inputs:

    output ≈ x · s · rsqrt(mean(x²) + eps)

scale s is `w` for RmsNorm, `1 + w` for RmsNormOne (the checkpoint stores
the offset). bf16 fixture outputs round to 1 bf16 ulp (2⁻⁸ relative), so
fixture wiring passes rtol 1e-2. f32 property checks pass 1e-6.

Detection note: a uniform output scale fault of 1.01 shifts every element
by 1% and violates rtol 1e-2 caps. Honest rounding (≤ 2 ulp, 0.8%)
stays inside.

### RMSNorm scale preservation

RMS(out) ≈ RMS(s) for uncorrelated x and weight. The identity
RMS(x·s)/RMS(x) = RMS(s) holds only in expectation: at normalized width d
the finite-sample deviation of mean(x²) and the x-weight cross term give
the ratio a relative variance of

    (kx · ks + kx − 1) / d

where kx, ks are the empirical kurtosis E[z⁴]/E[z²]² of x and s. The
check allows a 3-sigma slack computed from the tensors themselves. At
d = 256 with Gaussian x and w the slack is ~0.6, so this check detects
gross scale faults only (weight scaling, broadcast errors), not 1% faults.
Fault detection for norm outputs goes through the definition check.

### RoPE pair norms

NEOX rotation pairs (i, i + rotary_dim/2) rotate by (c, s) with c² + s² =
1, preserving each pair norm. The check compares pair-norm tensors before
and after rotation, over the rotated columns only. Pass-through columns
are checked bit-exact by the suites. Position 0 rotates by identity
(cos row 0 is 1, sin row 0 is 0, asserted precondition): rotated columns
at sequence position 0 are bit-exact equal to the input.

### FWHT Parseval

Unnormalized Walsh-Hadamard transform on a d-wide block satisfies
||H·x||² = d·||x||², checked in f32. `hadamard_rotate_128` applies the
1/sqrt(128) norm after the transform, so output energy equals input
energy. A pre_scale scales input energy elementwise before the transform.

### Softmax row sums

Every softmax output row sums to 1 within slack, computed in f32. Slack
1e-5 for f32 logits, 1e-2 for bf16 logits. Row sums alone cannot detect
tail-probability drift (a rescaled and renormalized tail still
sums to 1): logits comparisons need the tail-probability check
alongside row sums.

### GDN evaluation order (budget-change record)

The f32 Gated DeltaNet recurrent state drifts between the two
evaluation orders of the recurrence: the chunked one-shot path and the
step-wise decode order the f32 accumulation differently.

Budget change, recorded 2026-09-09:

- Old budget: exact fp32 agreement against the recorded state tensors,
  with `GdnStateRtol` (two fp32 ulp) plus `GdnStateAbstol` on the
  recorded band, and the decode outputs bit-exact.
- Assumption the old budget leaned on: the recording and every replay run
  the same evaluation order, so fp32 agreement is device-stable. The
  assumption fails on hardware that rounds the accumulation order
  differently, the failure mode the bf16 port already demonstrated.
- New budget: `assertSsmEvalOrder` runs the two evaluation orders and asserts
  `SsmUlpMargin` (4 fp32 ulp) drift at the pair max. The derivation:
  the two orders differ by the order of a length-T sum, the divergence
  concentrates at the largest partially summed element, whose magnitude
  the pair max bounds, so one max-anchored absolute bound covers every
  binade. Measured drift under the cap: 2.0 fp32 ulps (0.8B layer-0
  synthetic T=70), 1.0 (35B layer-0 synthetic T=70), 2.2 (the
  bench_gdn_recurrence decode measurement), 2.4 to 2.7 (the recorded torch
  recurrent-versus-chunked comparison at T=70).
- Block outputs: the bf16 outputs of the two orders stay inside the
  bf16 rounding scale (a few bf16 ulps), asserted at 4 bf16 ulps at the
  output max
  (measured 0.125 at 0.8B and 0.03125 at 35B on synthetic T=70). The
  conv states stay bit-exact: the conv is causal and both orders run
  the same conv op.
- Rejection rows: the drift bound stays absolute-scale, so a scaled
  blowup trips the fingerprint max quantile before the drift budget.
  The bounded-probe and histogram fault classes stay rejected under
  the new bound (selftest rows).

Budget change, cross-device un-skip (OPERATOR-REVIEW, pending operator
sign-off):

- Measured facts, Metal, both orders run on the device: bf16 block
  outputs 0.25 bf16 ulps at the output max 3.375 (cap 4, holds).
  Conv state: exactly one bf16 step at the differing element, 0.25
  ulp-units at the conv max 7.34375 (the element sits in a lower
  binade). f32 state: 1027.4 fp32 ulps at the pair max 1.032 over the
  T=70 decode, about 15 fp32 ulps per step. CPU reference: conv bit-equal, 1.0 fp32
  ulp.
- Assumptions replaced: the conv-state bit-equality bullet above (the
  conv is causal, but the two orders no longer run one conv op on
  hardware that reorders the accumulation), and the flat four-fp32-ulp
  state cap (the per-step drift accumulates linearly over the decode
  length).
- New budgets: `assertConvEvalOrder` (two bf16 ulp-units at the tensor max,
  derivation in the check's comment) and `assertSsmEvalOrder` with the
  `steps` parameter and the linear-in-decode-length drift bound
  abstol(steps) = 2^-3 * steps * meanAbs(reference), the
  chainCheckpointAbstol shape reused unchanged.
- Rejection rows: the selftest evaluation-order corpus rejects a conv drift
  past the budget and a 2x-bound state drift, and accepts the sub-budget
  conv drift, the measured-scale state drift and the just-below-bound
  state drift. The cross-device GDN variant runs for real on the test
  device, no skip line remains.

## Greedy chain checks

`checkGreedyStep` checks one decode step of a greedy chain against its
`ttt-tf-001-greedy-steps-h2` recording (the Format registry). Semantics:

- **Argmax agreement** with the recorded chosen token. On agreement the
  top-32 f32 logits must sit under a margin-scaled cap:
  `epsBase · argmaxMargin + 4 bf16 ulp` at the recorded top logit
  (the ulp floor accounts for honest device drift, the margin term
  accounts for reference sensitivity). A truncated-KL checksum over the top-32
  support and a relative tail-probability checksum close the step.
- **Tie flips**: a divergence whose recorded margin is at or under
  `tieUlps` ulps of the recorded top logit (default 1 — the corpus's
  smallest nonzero margin is exactly one bf16 ulp) is tie-eligible.
  The observed pick must sit inside the tie band (4 bf16 ulp plus
  epsBase around the recorded top-1). The flip counts against
  `maxFlips` and the caller teacher-forces the recorded token back in,
  so every later check must re-converge. Past the cap, or with the pick
  outside the band, the step fails with the localized report.
- **Real divergence**: the report names the step, the recorded margin,
  the observed pick and its logit, and the worst deviating top-32
  support id. See tests/README.md for the misattribution rules.

## Selftest

The selftest verifies the checks themselves, two-sided: a seeded fault
corpus must be rejected, a known-good drift corpus must be accepted.
Detection floors per fault kind are published in the selftest module doc.
Run through `nim test_transformers` (the selftest suite file).

## Recordings

A recording is a `.json.zst` zstd frame holding exactly one JSON
payload, parsed by the consuming suite with jsony against the payload's
registry id (the stats recordings carry `ttt-tf-003-tensor-stats-h2`,
see the Format registry). The frame records its content size and a checksum in
the header: producers write level 19 frames with both fields, and the
reader asserts the content size instead of guessing buffers, so a corrupt
or content-size-unknown frame raises an error. Every recording writes
the environment into PROVENANCE.md through the provenance module as
key-value rows: rendered rows are deterministic and verified by
byte-level re-render. The recording environment tracks upstream,
versions are written as rows, never locked in a lockfile.
No device row:
the recording device derives from the recorded_from row, its last
dash-separated component (harness/device.nim `recordedDevice`).
