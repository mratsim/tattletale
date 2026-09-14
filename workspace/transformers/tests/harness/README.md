# Harness

The check layer between the model code and the recorded truth.
Two assert functions, one uniform record, allowances derived from the error model.

This file is the contract for the assert API.

- tree rules, [tests README](../README.md)
- writing rules, [testing skill](../../../.agents/skills/testing/SKILL.md)
- recording rules, [FIXTURE_GENERATION.md](../testgen/FIXTURE_GENERATION.md)

## API

| function | asserts | kind argument |
|---|---|---|
| `assertStats(actual, statsPath, tensorName, kind, depth = 1, msg = "")` | quantiles, histogram, boundary elements, mean and tail bands, everything the sidecar record carries | the statistical property |
| `assertArgMax(actual, decisionsPath, step, kind, flipCount, msg = "", depth = 1)` | argmax id, truncated KL over the 32-wide top-32, tail probability, flip events | the error model class |

### Kinds (statistical properties, never operations)

| kind | error model | band |
|---|---|---|
| `kElementwise` | reduction-free sequence, drift is per-element rounding | tight, around 2 ulp, zero mismatch |
| `kReduction` | one accumulation point, associativity exposed once | moderate, around 4 ulp, small mismatch fraction, histogram L1 |
| composed chains | `kReduction` composed over checkpoints | `kReduction` band plus the depth argument |

- a new operation maps to a kind by its error model, misclassification
  is a suite bug, not an API extension
- the decision surface has no kind, integer ids carry no quantiles,
  so they belong under `assertArgMax`

## Assert flow

```
assertStats(actual, statsPath, tensorName, kind, depth = 1, msg = "")

  actual:     Tensor                  the suite-computed tensor
  statsPath:  string                  the sidecar stats frame (.json.zst)
  tensorName: string                  the recorded tensor inside the frame
  kind:       RoundingErrorSourceKind kElementwise or kReduction
  depth:      composed-chain depth    default 1

  the frame loads from disk and yields the record for tensorName
  the actual side recomputes the same instruments
  |  the same instruments the recorder took at record time:
  |  order statistics, binade histogram, f64 means, tail probability
  v
  shape assert: element count and ulp datatype must equal the record
  |  mismatch -> HarnessCheckError
  v
  deriveBands(allowanceOf(kind), record.maxMagnitude, depth,
              record.ulpDatatype)
  |  ulpBand = reorders x depth
  |  delta   = ulpBand x ulpStepAt(datatype, |reference|)
  v
  instruments, every fault raises HarnessCheckError carrying msg:
  |  depth 1:      quantiles within ulpBand ulps
  |  depth > 1:    quantiles within the absolute delta
  |  any depth:    the f64 means within delta
  |  depth 1 only: histogram total variation <= floor, tail probability
  |                inside the recorded edge fraction
  v
  return
```

### Why a fingerprint, not word-by-word

```
computed tensor --> fingerprint
                   [ min/max | 9 percentiles | size-bucket histogram
                     | fixed-position spot-checks | 2 f64 means
                     | far-tail probability ]
              --> against the recorded fingerprint, inside the bands
```

- parallel math adds numbers in a different but equally valid order
  than the reference, the last bits differ honestly, the bands say
  how much honest difference looks like
- depth composed operations drift up to depth times more, the band
  widens with the depth argument

### What the fingerprint instruments cannot see

- a single drifting element, away from the checkpoint spots,
  never an extremum, stays invisible, the fingerprint never moves
- the recorded-side comparison carries no exact class, cross-platform
  robustness = the allowances

```
assertArgMax(actual, decisionsPath, step, kind, flipCount, msg = "", depth = 1)

  actual:       Tensor  one logits vector at one decision step
  decisionsPath: string the decisions frame (.json.zst) of the chain
  step:         int     the zero-based frame position
  kind:         RoundingErrorSourceKind  the error model class
  flipCount:    var int  the caller-owned chain-wide flip counter

  the frame parses once per path and stays in the cache for the run,
  the allowances derive at check from the per-stage kind constant x depth,
  uniform for every model (the frame serializes no allowance and no
  flip cap, the cap = the harness MaxTieFlips constant, the counter
  lives in the suite loop)

  instruments, every step, agreement and flip alike:
  |  a non-finite logit rejects
  |  truncatedKl over the shared top-32 <= klBand
  |  tail probability inside its sensitivity-scaled allowance
  v
  pick agreement (observed argmax == record.argmaxId) ->
  |  the set certificate compares the observed top-32 id set against
  |  the recorded top-32 set
  |  equal sets -> return
  |  one boundary swap within one ulp of the record's datatype and
  |  max(delta, 0.05) -> accepted, counted against the chain-wide cap
  |  anything wider -> HarnessError named with both boundary tokens,
  |  values, and the gap

  pick divergence:
  |  recorded margin <= 1 activation ulp AND the pick within
  |  max(delta, 0.05) of the recorded top-1
  |     -> accepted tie flip, counted against the chain-wide cap
  |  otherwise -> HarnessCheckError (divergence, never a tie)
```

Both asserts raise `HarnessCheckError` on every fault and return
normally otherwise. Neither returns a measurement.

### What the decision instruments cannot see

```
rank:         ...  31    32    33    34  ...
recorded:     ...   x    [A]   [B]  ...      top-32 = {.. A ..}
observed:     ...   x    [A]   [B'] ...      B' rose past A

the recorded ids still read the same values, no instrument moves
```

- the observed side is never re-ranked, a rank 32/33 boundary swap
  moves no recorded value and shifts no tail probability, every
  instrument stays green while the ranking changed
- the tail band scales with the recorded tail, a flat distribution
  whose top-32 carries little mass gets a wide tail band, the ranks
  past the top-32 set carry no per-logit guard there
- the observed-side set certificate closes that gap
- every step recomputes the observed row's own top-32 and compares
  the id set against the recorded top-32 set
- one swapped boundary pair counts against the MaxTieFlips budget when
  the two boundary logits sit within one activation ulp of each other
  and agree inside the drift band, a wider set change rejects naming
  both boundary ids and logits
- the record carries every field the certificate needs
- the ulp datatype comes from the frame's "ulp_datatype" key
- an absent key reads ulpBf16
- the retired 002 row sample (probe_bits, 512 strided words per step)
  saw corruption anywhere in the row
- the 005 schema replaced it with the top-32 set and the tail probability
- corruption deep outside the top-32 set stays invisible when the tail
  moves under its allowance

## Uniform record

One recorder (the Python fixture side, `testgen/fixture_stats.py`),
every record identical in shape:

- exact order statistics at fixed probabilities, f32 storage
- binade-log histogram of (sign, binade, mantissa top bits), 4-ulp wide,
  soft bucketing, zero and subnormal bins, the buckets keyed
  by the recorded ulp datatype's pattern
- max, min, tail threshold crossings, carried as values and positions
- bulk and signed mean in f64, tail edge count
- the allowance inputs, max magnitude, meanAbs, depth or step on sequence records

Records live as sidecars (`.json.zst`), never as full recorded
tensors inside the check path.

The EXL3 packed-quantization fixtures are the one exception, justified
by deterministic dequant, no reductions, no associativity.

## Bands are derived, not calibrated

### Two-tier decision law

Two decision tiers, pick strictness and distribution shape:

- the pick stays strict, the decoded token is the recorded argmax id,
  divergence raises unless tie-eligible
- the top-32 distribution is what sampling consumes, the truncated-KL
  band enforces it, and the individual top-32 logits stay
  unguarded per id
- the per-id value band is retired on measurement, the MoE routed-stack
  replay's mid-table id drift is the platform noise floor
- measured drift, dense controls at most 1.5 reference-ulps, the A3B
  routed stack a mean of about 2 id-ulps, p95 7 and max 16.5
  reference-ulps across the MPS/CPU/MLX backends, 9 of 96 steps past
  the reduction class's 4-ulp per-stage allowance, while the truncated
  KL stayed inside the derived 0.5 x delta^2 band on all 96 steps
- the bands still derive from the error model, klBand = 0.5 x delta^2,
  no calibrated constants, no device keys

- the band formulas derive from the error model, reordering
  opportunities times ulp per reordering
- the math lives here and in the harness code, not in a tuned table
- no budget table, no per-op rows, no device-keyed constants, an ulp is
  an ulp on any architecture
- measured drift outside the derived band marks a finding, a defect
  or a wrong derivation
- the band never moves, the code or the math does
- the drift-scaling rule compares band widths (drift over band), never
  raw drifts, with the absolute floor kept

## No exact class

The assert compares recomputed replay instruments with the recorded
instruments:

- all 11 order statistics (min, the nine fixed probabilities, max)
- every histogram bucket count
- both f64 means
- the tail probability against its recorded edge fraction

Every instrument compares under the derived allowances:

- order statistics within `ulpBand` ulps of the recorded datatype,
  depth 1, within the absolute `delta` at greater depth
- the means within `delta`
- the histogram within the total-variation floor

No device name, box name, or recorded-environment entry selects between branches, the comparison itself observes which one holds.
Reasons live in the record's own history:

- x86-recorded fixtures drift one bf16 ulp on arm64 through QKV/SDPA reduction-order differences
- one mid-network coarse-compute ulp amplifies to a logits deviation of 0.033
- batched reductions reassociate

Bit-equality is the codec family's payload contract only.

- the EXL3-00 codec verifies its decoded payload bit for bit
- no other assert carries an exact class

## Self-validation routes through statistical theory

- every self-check carries an analytic answer, computed independently
  of the recorder
- arange has exact order statistics, a constant tensor produces
  one histogram bucket, seeded normals sit near the analytic quantiles
- fixture data never validates the harness
- the fault corpus (injected faults of known size) must be rejected,
  the honest drift corpus must be accepted
- the sparse and single-element floors are measured, then published,
  never promised away
- python and Nim recorders produce identical record text over the synthetic constructions of the selftest corpus
- agreement is the secondary check, the analytic answer stays primary

## How to add a fixture family

1. the generator `tests/testgen/gen_<dtype>_<model>_<NN>_<tier>.py`,
   rules in [FIXTURE_GENERATION.md](../testgen/FIXTURE_GENERATION.md),
   and `tests/linters/lint_gen_scripts.py` runs the rules
2. record the family, sidecar statistics + decision records per tensor
3. the suite, one flat `main`, setup from `tests/layer_utils.nim`,
   asserts from this API only
4. register the granular task inside `config.nims`
5. run the linters (see the [testing skill](../../../.agents/skills/testing/SKILL.md))

## Asserts

Only assertStats and assertArgMax may enforce.

Counted violations:
- any other harness assert proc
- any check* call
- any verify*, ensure*, or require* call
- the `runCppTest` sections

`SPEC.md` and `PLAYBOOK.md` are superseded by this file.
