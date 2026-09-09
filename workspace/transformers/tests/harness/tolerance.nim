
# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tolerance assertions and budgets for transformer suites.
## A `ToleranceBudget` is one budget: the set of tolerances one check allows, how far a computed
## output may drift from its recording before the test fails. Budgets are tiered by comparison
## strictness (see SPEC.md):
## - `ctBitExact` for deterministic weight math (codec, codebook, scales), compared value-equal
## - `ctElementWise` for same-device computed outputs, compared against rtol and abstol caps
## - `ctDistribution` for outputs compared through fingerprints and margin checks instead of element-wise
## caps

import
  std/algorithm,
  std/json,
  std/math,
  std/options,
  std/os,
  std/sequtils,
  std/strutils,
  std/tables,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/zstd/zstd_highlevel

export libtorch_testutils.assertAllClose, libtorch_testutils.assertShape

type
  HarnessCheckError* = object of CatchableError
    ## The one rejection type of the check layer: every detector that rejects a measured
    ## fault raises exactly this type, never a plain AssertionDefect and never a generic
    ## CatchableError. The two-sided selftest and the per-suite rejection rows catch only
    ## this type: a check that fails by crashing (a parse error, an I/O error, a wrong shape)
    ## propagates and fails the run instead of counting as a fault rejection.

type
  CheckTier* = enum
    ## Comparison strictness classes of the budget table.
    ctBitExact
      ## Deterministic weight math. Compared with value equality.
    ctElementWise
      ## Same-device computed outputs. Compared with rtol/abstol caps.
    ctDistribution
      ## Outputs compared through fingerprints, match rates, and margin checks. No element-wise cap
      ## applies.

  ToleranceBudget* = object
    ## One budget: the tolerances one check allows.
    tier*: CheckTier
    rtol*: float64
      ## Relative element-wise cap. Ignored for the ctBitExact and ctDistribution budgets.
    abstol*: float64
      ## Absolute element-wise cap. Ignored for the ctBitExact and ctDistribution budgets.
    maxUlp*: int32
      ## ctDistribution: element-wise match-rate cap, counted in ulps relative to the recorded
      ## dtype. 0 disables the match-rate assert. Logits rows use margin checks instead, see
      ## SPEC.md.
    maxMismatchFrac*: float64
      ## ctDistribution: fraction of elements allowed past maxUlp.
    histL1*: float64
      ## ctDistribution: normalized L1 allowance between fingerprint histograms. Provisional 0.01
      ## until the fault corpus measures the residual false-positive rate, see SPEC.md.

  OpBudgetKind* = enum
    ## Op classes of the budget table v0.
    obRmsNorm
      ## Norm outputs. Measurement artifact: bench_rmsnorm.nim records 1-2 ulp f32 same-device
      ## drift, so the budget caps at 2 bf16 ulp.
    obRope
    obAttention
      ## Attention and mixer outputs (GDN included).
    obPostResidual
      ## Post-residual and FFN outputs.
    obChainCheckpoint
      ## Mid-chain block checkpoints of the chain suites. Per-op budgets never compose: drift
      ## accumulates over the blocks, so the budget is derived from the measured per-block drift on the recorded
      ## chain fixtures (bf16-02-first-8-layers-plus-final), not from a per-op number. The absolute term
      ## scales with sqrt(depth), see chainCheckpointAbstol.
    obLogits
      ## Final logits: margin, truncated-KL, and tail-probability checks only. maxUlp stays 0, the match-rate
      ## assert refuses this budget.

proc maxAbsDiff*(a, b: Tensor): float64 =
  ## Maximum absolute elementwise difference of two tensors, compared in f32.
  (a.to(F.kFloat32) - b.to(F.kFloat32)).abs().max().item(float64)

proc assertWithinBudget*(
    actual, expected: Tensor,
    budget: ToleranceBudget,
    msg = "") =
  ## Check `actual` against `expected` under `budget`.
  ## - ctBitExact: value equality, no tolerance
  ## - ctElementWise: assertAllClose with the budget's rtol/abstol
  ## - ctDistribution: not implemented here, it raises ValueError.
  ##   Distribution comparisons run through assertStats and the fingerprint
  ##   checks instead.
  case budget.tier
  of ctBitExact:
    if not actual.equal(expected):
      raise newException(HarnessCheckError,
        "[ttt] bit-exact budget violated" & (if msg.len > 0: ": " & msg else: ""))
  of ctElementWise:
    assertAllClose(actual, expected,
      rtol = budget.rtol, abstol = budget.abstol, msg = msg)
  of ctDistribution:
    raise newException(ValueError,
      "ctDistribution budgets require fingerprint checks, not yet implemented")

proc worstRelDiff*(actual, expected: Tensor): float64 =
  ## Worst relative element-wise difference in f32, abstol-guarded. Elements with |expected| below 1e-12
  ## compare by absolute difference.
  let a = actual.to(F.kFloat32)
  let e = expected.to(F.kFloat32)
  let absDiff = (a - e).abs()
  let scale = e.abs().clampMin(1e-12'f32)
  (absDiff / scale).max().item(float64)

# ########################################################################### Fingerprints
# ###########################################################################

const
  GdnStateRtol* = 2.4e-7
    ## Relative slack for comparing the f32 Gated DeltaNet recurrent state between the step-decode
    ## path and the chunked one-shot path. Measured drift from the Qwen3.5-0.8B layer-0 state
    ## persistence fixture (gdn-Qwen3.5-0.8B-01): decode outputs match the one-shot trajectory
    ## bit-exactly, while the stored state differs by one f32 ulp (1.49e-8 at binade -3). The chunked
    ## kernel groups the f32 accumulation differently than the step-wise recurrence. Two f32 ulp of relative
    ## slack, with the absolute floor beneath, absorb that honest rounding on any binade.
  GdnStateAbstol* = 1.0e-7
    ## Absolute floor for the GDN recurrent-state slack, covering state elements that sit near zero
    ## where a relative cap alone is blind.

const
  QuantileProbs* = [0.01'f64, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ## Fixed quantile probabilities. Storage and comparison use exact order statistics: index =
    ## floor(p * (n - 1)) of the ascending sort, no interpolation. Python and Nim agree bit-exactly
    ## on the same tensor bytes.

  QuantileCount* = 2 + QuantileProbs.len
    ## min, the 9 fixed quantiles, max.

  HistogramBinades* = 128
    ## Bucket space: binades -64 .. 63, both signs, mantissa top 6 bits
    ## as 14-bit keys in sparse storage, plus the dedicated zero
    ## and subnormal bins. Values outside the binade range raise an error.

  HistKeyZero* = uint16(0xFFFE)
  HistKeySubnormal* = uint16(0xFFFF)

  HistBinadeMin* = -64
  HistBinadeMax* = 63

type
  TensorStats* = object
    ## Fingerprint of one tensor: exact order statistics plus an optional binade-log soft histogram
    ## over the bf16 bit pattern.
    name*: string
    n*: int
    quantiles*: array[QuantileCount, float32]
      ## min, the fixed quantiles, max, stored as f32.
    allowMinusInf*: bool
      ## The recording whitelists -Inf (masked attention scores).
    hasHist*: bool
    histKeys*: seq[uint16]
    histCounts*: seq[uint32]
      ## Soft counts scaled x2. Each element adds (2 - low) toward one bucket and low toward the neighbor
      ## bucket. low is the dropped bf16 mantissa bit. Integer math, bit-exact
      ## cross-implementation.
    histTotal*: uint64
      ## Sum of histCounts. Non-finite whitelisted values stay unbucketed.
    hasDescriptors*: bool
      ## True when the entry carries the descriptor keys: meanAbs, signedMean, tailProbability, and the strided
      ## probe subset. Entries without descriptor keys keep this false and use the fingerprint-only
      ## byte layout.
    meanAbs*: float64
      ## Bulk scale of the recorded tensor: the mean absolute value in f64 over the promoted
      ## values.
    signedMean*: float64
      ## Signed mean of the recorded tensor in f64. The difference between the two signed means is the signed
      ## mean drift of the replay, the detector of the coherent-bias fault class.
    tailProbability*: float64
      ## Fraction of elements strictly above the tail threshold: the recorded max divided by 16,
      ## four binades under the max.
    tailEdge*: int
      ## Recorded count of elements within the drift bound of the tail threshold. The tail
      ## band is this count over n: no element outside the edge band can cross the threshold under honest
      ## drift.
    probeStride*: int
      ## Stride of the recorded probe subset: indices i * probeStride, sized so the probe holds
      ## about 512 f32 words, near 2 kB.
    probeMode*: DescMode
      ## Probe comparison class: dmExact for bit-equal promoted values on the reference device,
      ## dmDrift for the absolute drift bound.
    probeValues*: seq[float32]
      ## Recorded probe values, promoted to f32 bit patterns in index order. bf16 tensors store the exact
      ## f32 promotion.

  DescMode* = enum
    ## Comparison class of a descriptor-carried tensor entry.
    dmNone
      ## Fingerprint-only entry, no descriptor keys, fingerprint-only layout.
    dmExact
      ## Reference-device class: the replay is bit-identical to the recording, so the probe
      ## and the descriptor bands compare with zero drift allowance.
    dmDrift
      ## Drift class of f32 state-like tensors: the replay may drift the derived descriptor
      ## bound per element, absolute-scale.

  FingerprintStatsFile* = object
    ## Fingerprint stats file for one committed fixture, one entry per tensor.
    schema*: int
    source*: string
    tensors*: seq[TensorStats]

proc orderedBits32*(f: float32): uint32 {.inline.} =
  ## Monotone integer map of the f32 pattern: order preserved, sign-flip continuous. Distance
  ## between two ordered words counts ulps.
  let u = cast[uint32](f)
  if (u and 0x80000000'u32) != 0'u32: not u else: u or 0x80000000'u32

proc bf16BitsFromF32*(f: float32): uint16 {.inline.} =
  ## Round-to-nearest-even f32 to bf16 bit pattern, matching torch `.to(bfloat16)` bit-exactly.
  let u = cast[uint32](f)
  uint16(((u + 0x7FFF'u32 + ((u shr 16) and 1'u32)) and 0xFFFF0000'u32) shr 16)

proc bf16ToF32*(b: uint16): float32 {.inline.} =
  ## Exact bf16 promotion: the pattern shifted into the f32 mantissa top.
  cast[float32](uint32(b) shl 16)

proc orderedBits16*(b: uint16): uint32 {.inline.} =
  ## Monotone integer map of the bf16 pattern: order preserved, sign-flip continuous.
  ## The map stays inside 16-bit pattern space. In the ordered space an f32-word promotion would lift
  ## negative values above +Inf, so cross-sign pairs, tiny near-zero values included, report absurd
  ## ulp distances. Distance between two ordered words counts bf16 ulps.
  let w = uint32(b)
  if (w and 0x8000'u32) != 0'u32: (not w) and 0xFFFF'u32
  else: w or 0x8000'u32

proc ulpDistance32*(a, b: float32): int64 {.inline.} =
  ## ulp distance of two f32 values. Numerically equal values share distance 0, a +/-0 pair
  ## included.
  if a == b: 0'i64
  else:
    let d = orderedBits32(a).int64 - orderedBits32(b).int64
    if d < 0: -d else: d

proc ulpDistance16*(a, b: float32): int64 {.inline.} =
  ## ulp distance of two bf16-representable values, counted in bf16 ulps.
  if a == b: 0'i64
  else:
    let d = orderedBits16(bf16BitsFromF32(a)).int64 -
            orderedBits16(bf16BitsFromF32(b)).int64
    if d < 0: -d else: d

proc fp16BitsFromF32*(f: float32): uint16 =
  ## Exact fp16 pattern of an fp16-representable f32 value, recovered by bit
  ## surgery (the promotion leaves the f32 mantissa low 13 bits zero, so no
  ## rounding happens): normals take the f32 exponent and the mantissa top 10
  ## bits, subnormals scale back onto the 2^-24 grid. |f| past the fp16
  ## overflow bound maps to the inf pattern.
  let u = cast[uint32](f)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let expField = (u shr 23) and 0xFF'u32
  let man = u and 0x7FFFFF'u32
  if expField == 0xFF'u32:
    # NaN never reaches the fingerprint paths: the non-finite scan raises first.
    return sign or 0x7C00'u16
  if expField == 0'u32 and man == 0'u32:
    return sign
  if expField.int >= 127 + 16:
    return sign or 0x7C00'u16
  if expField.int >= 127 - 14:
    let e = expField.int - 127 + 15
    return sign or uint16(e.uint32 shl 10) or uint16(man shr 13)
  let scale = abs(f.float64) * 16777216.0
  return sign or uint16(scale.uint32)

proc fp16ToF32*(b: uint16): float32 =
  ## Exact fp16 promotion: the pattern value as f32, subnormals included.
  let sign = uint32(b and 0x8000'u16) shl 16
  let field = (b shr 10) and 0x1F'u16
  let man = b and 0x3FF'u16
  if field == 0'u16:
    let v = if man == 0'u16: 0.0'f32 else: (man.float64 * pow(2.0, -24.0)).float32
    return cast[float32](sign or cast[uint32](v))
  return cast[float32](sign or
    (uint32(field - 15'u16 + 127'u16) shl 23) or (uint32(man) shl 13))

proc ulpDistanceFp16*(a, b: float32): int64 {.inline.} =
  ## ulp distance of two fp16-representable values, counted in fp16 ulps.
  ## The exl3 families use the bf16 bounds with the ulp unit taken in fp16
  ## because EXL3 dequantizes to fp16.
  if a == b: 0'i64
  else:
    let d = orderedBits16(fp16BitsFromF32(a)).int64 -
            orderedBits16(fp16BitsFromF32(b)).int64
    if d < 0: -d else: d

proc tensorStats*(t: Tensor, allowMinusInf = false, withHistogram = true,
    name = ""): TensorStats =
  ## Fingerprint of a contiguous tensor. Quantiles come from the values promoted to f32. The histogram
  ## slices the bf16 bit pattern: sign, binade, mantissa top 6 bits, soft neighbor split on the dropped
  ## bit. NaN and +Inf raise an error. allowMinusInf whitelists -Inf values from masked attention
  ## scores, which then stay unbucketed. Device tensors move to the host before the bit reads:
  ## device memory holds no host storage, so the pointer casts need host bytes.
  let tc = t.contiguous().to(F.kCPU)
  let n = tc.numel()
  if n == 0:
    raise newException(ValueError, "fingerprint of an empty tensor " & name)
  result.n = n
  result.hasHist = withHistogram

  var vals = newSeq[float32](n)
  var bits16: seq[uint16]
  let isBf16 = tc.scalarType() == F.kBfloat16
  let isF16 = tc.scalarType() == F.kFloat16
  if isBf16:
    bits16 = newSeq[uint16](n)
    # bf16 bits are read through an int16 view: data_ptr<T> refuses a T that differs from the tensor
    # dtype.
    let bv = tc.view(F.kInt16).contiguous()
    let raw = cast[ptr UncheckedArray[int16]](bv.data_ptr(int16))
    for i in 0 ..< n:
      let b = uint16(raw[i])
      bits16[i] = b
      vals[i] = bf16ToF32(b)
  elif isF16:
    # fp16 values promote exactly into f32 (the fp16 unit of the exl3
    # families); the histogram buckets still read the bf16 pattern of the
    # promoted values.
    let hv = tc.view(F.kInt16).contiguous()
    let raw = cast[ptr UncheckedArray[int16]](hv.data_ptr(int16))
    for i in 0 ..< n:
      let f = fp16ToF32(uint16(raw[i]))
      vals[i] = f
      if withHistogram:
        bits16.add bf16BitsFromF32(f)
  else:
    let raw = cast[ptr UncheckedArray[float32]](tc.data_ptr(float32))
    for i in 0 ..< n:
      let f = raw[i]
      vals[i] = f
      if withHistogram:
        bits16.add bf16BitsFromF32(f)

  # Non-finite scan: NaN and +Inf always fail, -Inf is whitelisted only through allowMinusInf and never
  # bucketed.
  for i in 0 ..< n:
    let u = cast[uint32](vals[i])
    let expField = (u shr 23) and 0xFF'u32
    let manField = u and 0x7FFFFF'u32
    if expField == 0xFF'u32:
      if manField != 0'u32:
        raise newException(ValueError,
          "fingerprint input holds NaN: " & name)
      if (u and 0x80000000'u32) == 0'u32:
        raise newException(ValueError,
          "fingerprint input holds +Inf: " & name)
      if not allowMinusInf:
        raise newException(ValueError,
          "fingerprint input holds -Inf without the mask whitelist: " & name)

  vals.sort()

  result.quantiles[0] = vals[0]
  result.quantiles[^1] = vals[^1]
  for pi, p in QuantileProbs:
    let idx = floor(p * float64(n - 1)).int
    result.quantiles[pi + 1] = vals[idx]

  if not withHistogram:
    return

  # Binade-log soft histogram over the bf16 pattern. Integer math only.
  var keyCounts = initCountTable[uint16]()
  for i in 0 ..< n:
    let b = bits16[i]
    let sign = b and 0x8000'u16
    let expField = (b shr 7) and 0xFF'u16
    let mant = b and 0x7F'u16
    if expField == 0'u16 and mant == 0'u16:
      keyCounts.inc(HistKeyZero, 2)
    elif expField == 0'u16:
      keyCounts.inc(HistKeySubnormal, 2)
    else:
      let e = expField.int - 127
      if e < HistBinadeMin or e > HistBinadeMax:
        raise newException(ValueError,
          "fingerprint binade " & $e & " outside [" & $HistBinadeMin &
          ", " & $HistBinadeMax & "]: " & name)
      let mtop = mant shr 1
      let low = mant and 1'u16
      let key = (sign shr 2'u16) or (uint16(e - HistBinadeMin) shl 6) or mtop
      keyCounts.inc(key, 2 - low.int)
      if low != 0'u16:
        if mtop == 63'u16:
          if e + 1 <= HistBinadeMax:
            let nkey = (sign shr 2'u16) or
              (uint16(e + 1 - HistBinadeMin) shl 6)
            keyCounts.inc(nkey, 1)
          # else: soft mass at the binade-space edge is dropped
        else:
          keyCounts.inc(key + 1, 1)
  # Sorted sparse storage keeps stats files small and comparison order fixed.
  var pairs: seq[(uint16, uint32)]
  for k, c in keyCounts:
    pairs.add (uint16(k), uint32(c))
  pairs.sort()
  for (k, c) in pairs:
    result.histKeys.add k
    result.histCounts.add c
    result.histTotal += uint64(c)

const
  ChainCheckpointRtol* = 0.0078125
    ## Relative term of the chain checkpoint budget: two bf16 ulp total, one ulp of honest device
    ## drift plus one ulp of slack.
  ChainCheckpointScaleFactor* = 0.125
    ## Bulk-scale factor of the chain checkpoint budget's absolute term: abstol(depth) =
    ## ChainCheckpointScaleFactor * depth * meanAbs(expected).
    ## Derivation from the 8+1 checkpoint chains. SPEC carries the budget-change record.
    ## One decoder block stores about 13 bf16-rounded intermediates on one element path:
    ## - norm outputs, q/k/v, rope, attention out
    ## - both residual adds, gate/up/silu, down projection
    ## Computing the same sum in a different addition order shifts one stored value by one bf16 ulp
    ## of that value, so one block contributes at most 13 * 2^-8 ~ 0.05 of the activation bulk
    ## scale. The contribution accumulates linearly with depth: the differences are deterministic
    ## and not zero-mean, so the worst case is a linear sum, not a random walk. The safety factor
    ## 2.4 absorbs shallow checkpoints (depth 1-3), where the normalized internal
    ## activations exceed the residual bulk, and rounding the factor onto the binade grid gives 2^-3.
    ## Corroborating measurement over the 9 checkpoints of the dense ports, the Metal replay
    ## against the cpu recording on bf16-02-first-8-layers-plus-final: drift over bulk stays inside
    ## 0.05 to 0.16 at every depth 1..8 and 28, so the band usage stays below 0.75
    ## everywhere. The 3-point recordings do not discriminate the two depth models.
  ChainScalingSlack* = 3.0
    ## Drift-scaling slack on the band width ratio against the first checkpoint. The band absolute
    ## term is depth-linear by derivation, so it absorbs the honest depth-linear drift and the honest
    ## band widths stay bounded: measured ratios against the first checkpoint stay at or under 1.25
    ## over the 9 checkpoints of the dense ports. A fault that compounds per block grows its band
    ## width past the slack. With slack 3 a mild sqrt-shaped growth crosses exactly at the depth-28
    ## tail, sqrt(28) ~ 5.3 > 3 > 1.25. A constant per-block bias is depth-linear like the honest
    ## drift, so this check passes it by design and the mean-drift check polices it, see
    ## assertChainCheckpoint.
  ChainScalingFloor* = 0.75
    ## Drift-scaling floor, covering a first checkpoint that measured
    ## zero drift on the reference device replays.

  F32UnitRoundoff* = 5.9604645e-8
    ## Unit roundoff of the f32 accumulators the block internals use: 2^-24, the half-ulp bound the classical
    ## backward-error results are stated against. The block internals accumulate in f32: the ATen
    ## linear on bf16 inputs, the RMSNorm mean and rsqrt, and the Gated DeltaNet core with its state.
    ## See the mean-drift derivation in SPEC.

  ChainMeanDriftSafety* = 8.0
    ## Safety factor of the mean-drift bound. Stacked coverage: the inequality step from the per-term bounds to the output scale, about
    ## 1.25x, from the mean over elements of the reduction term magnitudes against the bulk scale,
    ## the several composing reductions of one block, about 2x, the product roundings inside the reductions,
    ## about 2x, and the bf16 rounding at the f32-to-bf16 storage boundary, about 1.5x, whose signed mean returns to the f32-scale
    ## difference.

proc meanAbsValue*(t: Tensor): float64 =
  ## Returns the bulk scale of a tensor: the mean absolute value, computed in f32. The chain row anchors its absolute
  ## term on this bulk scale rather than on the per-element magnitude: the honest drift of a checkpoint
  ## element is set by the activation scale along its path. Cancellation elements sit far below the terms
  ## that produce them.
  let c = t.contiguous().to(F.kCPU).to(F.kFloat32)
  let n = c.numel()
  let rc = cast[ptr UncheckedArray[float32]](c.data_ptr(float32))
  var sum = 0.0'f64
  for i in 0 ..< n:
    sum += abs(rc[i].float64)
  if n == 0:
    raise newException(ValueError, "bulk scale of an empty tensor")
  sum / n.float64

const
  ChainCheckpointRtolF16* = 0.00390625
    ## Relative term of the chain checkpoint budget for the fp16 dequant
    ## format: four fp16 ulp, the bf16 row's shape re-derived on the fp16
    ## grid (the exl3 families use the bf16 bounds with the ulp unit taken
    ## in fp16 because EXL3 dequantizes to fp16). Two fp16 grid steps are the
    ## honest cross-device drift of one composed stage measured on the mac
    ## replay, and the fp16 grid is eight times finer than the bf16 grid, so
    ## the same one-step-of-slack-plus-rounding coverage the bf16 row carries
    ## needs two more grid steps: four fp16 ulp total. The absolute term is
    ## grid-independent and unchanged.

proc chainCheckpointAbstol*(depth: int, bulkScale: float64): float64 =
  ## Absolute term of the chain checkpoint budget at `depth`: the bulk scale of the checkpoint times
  ## the scale factor times the depth. The honest cross-device drift of a residual-stream element
  ## is absolute-scale, grows linearly with the checkpoint depth, and the scale anchor is the activation
  ## bulk of the block, measured here as the mean absolute value of the recorded checkpoint.
  ChainCheckpointScaleFactor * depth.float64 * bulkScale

proc meanDrift*(actual, expected: Tensor): float64 =
  ## Signed mean of the elementwise difference over the checkpoint elements, computed in f64 from f32-promoted
  ## values. A coherent per-element bias survives the averaging, while the rounding differences
  ## of a different addition order are elementwise symmetric and average out.
  let a = actual.contiguous().to(F.kCPU).to(F.kFloat32)
  let e = expected.contiguous().to(F.kCPU).to(F.kFloat32)
  doAssert a.numel() == e.numel(), "mean drift element count mismatch"
  let n = a.numel()
  let ra = cast[ptr UncheckedArray[float32]](a.data_ptr(float32))
  let re = cast[ptr UncheckedArray[float32]](e.data_ptr(float32))
  var sum = 0.0'f64
  for i in 0 ..< n:
    sum += ra[i].float64 - re[i].float64
  sum / n.float64

proc chainMeanDriftBound*(bulkScale: float64,
    reductionLen = 3072): float64 =
  ## The chain drift bound: the largest signed mean drift that computing the same sums in a
  ## different addition order can produce. When the same sum is accumulated in a different order
  ## the result moves a little, and this bound is the largest movement a different addition order
  ## can explain. It grows with the number of accumulated steps. Full derivation in SPEC,
  ## mean-drift subsection: the signed mean drift
  ## satisfies |mean drift| <= mean_i |error_i| <= (d - 1) * eps_f32 * mean_i of the sum
  ## |term_ik| <= (d - 1) * eps_f32 * sqrt(d) * bulk, with d the longest
  ## f32-accumulated reduction feeding a checkpoint element, the down projection
  ## over the intermediate dimension, eps_f32 the f32 unit roundoff, and the bulk
  ## the mean absolute value of the recorded checkpoint.
  ## The last step is the inequality step from the per-term bounds to the
  ## output scale: for zero-mean uncorrelated reduction factors
  ## the mean absolute term sum is (2/pi) * d * sigma_w * sigma_h while the output scale is sqrt(d)
  ## * sigma_w * sigma_h, so the term sum runs about 0.8 * sqrt(d) times the reduction output
  ## scale, and the residual stream bulk covers that scale from above. The bound stays at the worst
  ## case of same-sign rounding errors from a different addition order, which is exactly the
  ## shape a coherent bias takes, so no 1/sqrt(N) averaging division is applied.
  ChainMeanDriftSafety * (reductionLen - 1).float64 * F32UnitRoundoff *
    sqrt(reductionLen.float64) * bulkScale

proc assertChainMeanDrift*(actual, expected: Tensor,
    reductionLen = 3072, msg = "") =
  ## Check that the signed mean drift of the checkpoint stays inside the chain drift bound.
  ## A different addition order produces elementwise differences with a near-zero signed mean
  ## at the f32 rounding scale, while a coherent bias shifting every element the same way
  ## survives the averaging and trips the bound.
  let d = meanDrift(actual, expected)
  let bound = chainMeanDriftBound(meanAbsValue(expected), reductionLen)
  if abs(d) > bound:
    raise newException(HarnessCheckError,
      "[chain] coherent mean drift " & formatBiggestFloat(d, ffScientific, 3) &
      " exceeds the reordering bound " & formatBiggestFloat(bound, ffScientific, 3) &
      " (bulk " & formatBiggestFloat(meanAbsValue(expected), ffScientific, 3) &
      ")" & (if msg.len > 0: ": " & msg else: ""))

proc maxBandWidths*(actual, expected: Tensor, rtol, abstol: float64):
    tuple[worst: float64, violations: int] =
  ## Returns the worst elementwise band width and the count of elements past one band. Both tensors compare in f32
  ## after promotion. The per-element band is rtol * |expected| + abstol. The relative term accounts for
  ## the per-op rounding of the element's own scale. The absolute term accounts for the cancellation-dominated
  ## small elements, whose relative drift means nothing: honest cross-device drift of a near-zero
  ## element reaches many ulps of its own binade while staying well inside one activation-scale
  ## absolute band.
  let a = actual.contiguous().to(F.kCPU).to(F.kFloat32)
  let e = expected.contiguous().to(F.kCPU).to(F.kFloat32)
  doAssert a.numel() == e.numel(), "band compare element count mismatch"
  let n = a.numel()
  let ra = cast[ptr UncheckedArray[float32]](a.data_ptr(float32))
  let re = cast[ptr UncheckedArray[float32]](e.data_ptr(float32))
  result.worst = 0.0
  for i in 0 ..< n:
    let d = abs(ra[i].float64 - re[i].float64)
    let band = rtol * abs(re[i].float64) + abstol
    if classify(band) == fcNaN or band <= 0.0:
      raise newException(HarnessCheckError,
        "[chain] degenerate band: rtol/abstol produce " & $band &
        " at element " & $i & ", a zero bulk scale turns 0/0 into NaN and any drift passes")
    let w = d / band
    if w > result.worst:
      result.worst = w
    if w > 1.0:
      inc result.violations

proc checkDriftScaling*(drifts: seq[float64], slack = ChainScalingSlack,
    floor = ChainScalingFloor) =
  ## Check that the measured per-checkpoint band widths stay bounded. The band absolute term is
  ## depth-linear, so the honest depth-linear drift keeps the band widths bounded and the ratio
  ## against the first checkpoint near 1, measured at or under 1.25 over the 9 checkpoints. The
  ## check compares every later checkpoint against the first one under the flat bound with slack,
  ## and the floor covers the all-zero reference device replay. A fault that compounds per block
  ## grows its band width past the slack and is rejected, for a mild sqrt-shaped growth exactly
  ## at the tail checkpoint. A constant per-block bias behaves like the honest drift: depth-linear,
  ## so this check passes it by design. The mean-drift check polices it.
  if drifts.len < 2:
    return
  let bound = max(drifts[0] * slack, floor)
  for k in 1 ..< drifts.len:
    if drifts[k] > bound:
      raise newException(HarnessCheckError,
        "[chain] band widths compound with depth, not bounded: checkpoint " &
        $k & " measured " & formatBiggestFloat(drifts[k], ffDecimal, 3) &
        " band widths, bound " & formatBiggestFloat(bound, ffDecimal, 3) &
        " (first checkpoint " & formatBiggestFloat(drifts[0], ffDecimal, 3) & ")")
proc assertChainCheckpoint*(actual, expected: Tensor, depth: int,
    reductionLen = 3072, msg = "",
    rtol: float64 = ChainCheckpointRtol):
    tuple[worst: float64, violations: int, meanDrift: float64] =
  ## Check one mid-chain block checkpoint under the chain budget:
  ##
  ##   computed checkpoint -> maxBandWidths ------------------------\
  ##   recorded checkpoint -> band rtol * |expected| + abstol(depth) --> worst width, violations
  ##   computed and recorded means -> meanDrift --------------------> chain drift bound
  ##
  ## The band absolute term is scale * depth * meanAbs(expected), the bulk scale comes off the
  ## recorded checkpoint itself. Every element must sit inside the band, the mismatch fraction
  ## is zero, and the signed mean drift must stay inside the chain drift bound.
  ## Returns the measured worst band width, the violation count and the signed mean drift. Suites
  ## feed the per-checkpoint worst widths to checkDriftScaling.
  let mw = maxBandWidths(actual, expected, rtol,
    chainCheckpointAbstol(depth, meanAbsValue(expected)))
  if mw.violations != 0:
    raise newException(HarnessCheckError,
      "[chain] checkpoint band violated: " & $mw.violations &
      " elements past one band (worst " &
      formatBiggestFloat(mw.worst, ffDecimal, 3) & " band widths)" &
      (if msg.len > 0: ": " & msg else: ""))
  assertChainMeanDrift(actual, expected, reductionLen, msg)
  result = (worst: mw.worst, violations: mw.violations,
    meanDrift: meanDrift(actual, expected))

proc defaultBudget*(kind: OpBudgetKind, computed: F.DeviceKind,
    reference: F.DeviceKind = F.kCPU): ToleranceBudget =
  ## Budget table v0, measured on the M4 Max. The same-device budgets hold for Metal computations of
  ## the norm and rope paths, measured bit-exact against the cpu path (harness/bench_metal_budget.nim).
  ## The attention and post-residual elementwise caps stay same-device budgets: Metal-computed
  ## outputs of composed paths take the chain checkpoint band, selected through compareClass
  ## (harness/device.nim). CUDA budgets await the reference box. Measurement artifact of the norm
  ## budget: bench_rmsnorm.nim measures 1-2 ulp f32 same-device drift, the floor under the 2 bf16
  ## ulp cap.
  if reference != F.kCPU:
    raise newException(ValueError,
      "budget table v0 only covers recordings made on cpu")
  case kind
  of obRmsNorm:
    result = ToleranceBudget(tier: ctDistribution,
      maxUlp: 2, maxMismatchFrac: 0.0, histL1: 0.01)
  of obRope:
    result = ToleranceBudget(tier: ctDistribution,
      maxUlp: 2, maxMismatchFrac: 0.0, histL1: 0.01)
  of obAttention, obPostResidual:
    result = ToleranceBudget(tier: ctDistribution,
      maxUlp: 4, maxMismatchFrac: 1e-3, histL1: 0.01)
  of obChainCheckpoint:
    # The chain budget absolute term depends on depth and bulk and lives inside assertChainCheckpoint.
    # The budget carries the factor per depth unit alone.
    result = ToleranceBudget(tier: ctElementWise,
      rtol: ChainCheckpointRtol, abstol: ChainCheckpointScaleFactor)
  of obLogits:
    result = ToleranceBudget(tier: ctDistribution,
      maxUlp: 0, maxMismatchFrac: 0.0, histL1: 0.0)
  case computed
  of F.kCPU, F.kMPS: discard
  else:
    raise newException(ValueError,
      "budget table v0 has no row for computed device kind " & $computed)

proc exl3Budget*(kind: OpBudgetKind): ToleranceBudget =
  ## The budget rows of the exl3 families: the bf16 table rows with the ulp
  ## unit taken in fp16, because EXL3 dequantizes to fp16. The exl3 payloads
  ## are recorded from the production CUDA kernel, a recording device the
  ## defaultBudget device guard does not cover, so the rows come off the
  ## table constants directly.
  defaultBudget(kind, F.kCPU)

proc normalizedL1*(a, b: TensorStats): float64 =
  ## Normalized L1 distance between two soft histograms: sum of absolute bucket differences over the combined
  ## mass.
  if not (a.hasHist and b.hasHist):
    return 0.0
  var ai, bi = 0
  var diff = 0'u64
  while ai < a.histKeys.len or bi < b.histKeys.len:
    if bi >= b.histKeys.len or (ai < a.histKeys.len and
        a.histKeys[ai] < b.histKeys[bi]):
      diff += uint64(a.histCounts[ai])
      inc ai
    elif ai >= a.histKeys.len or b.histKeys[bi] < a.histKeys[ai]:
      diff += uint64(b.histCounts[bi])
      inc bi
    else:
      let x = a.histCounts[ai].int64 - b.histCounts[bi].int64
      diff += uint64(if x < 0: -x else: x)
      inc ai
      inc bi
  let total = a.histTotal + b.histTotal
  if total == 0'u64: 0.0 else: diff.float64 / total.float64

# ########################################################################### Descriptor drift bounds
# ###########################################################################

const
  SsmUlpMargin* = 4.0
    ## Evaluation-order and descriptor drift cap for f32 SSM state-like tensors: four fp32 ulps at
    ## the state max magnitude. The two evaluation orders of the recurrence diverge by a few ulps
    ## at the largest divergent element, whose magnitude the state max bounds. Measured drift:
    ## - 2.2 fp32 ulps for the two CPU implementations of one decode step (the
    ## bench_gdn_recurrence decode measurement)
    ## - 2.4 to 2.7 fp32 ulps between the torch recurrent and chunked paths at sequence length 70
    ## The margin stays at 4 across those measurements.

  ConvEvalOrderBudget* = 2.0
    ## Conv-state evaluation-order budget, counted in bf16 ulp-units at the tensor max.
    ## Unit convention: one ulp-unit is one bf16 ulp taken at the tensor max,
    ## the widest bf16 step in the tensor. One bf16 ulp equals 65536 fp32 ulps
    ## at the same magnitude: binade [1, 2) holds the bf16 step 2^-7 against
    ## the fp32 step 2^-23. An element in a lower binade rounds on a narrower
    ## grid, so an absolute drift reads as a fraction of one ulp-unit at the max.
    ## Derivation: the two evaluation orders run one causal conv. The honest
    ## cross-order difference at one element comes from one bf16
    ## rounding step of a differently ordered f32 accumulator, bounded by one
    ## ulp-unit at the max. The budget carries the model worst case times
    ## the safety factor 2: two ulp-units at the max.
    ## Calibration, no constant fitting: the measured Metal cross-order drift
    ## is one bf16 step at the differing element, 0.0078125 absolute = 0.25
    ## ulp-units at the conv max 7.34375, because the differing element sits
    ## in binade [1, 2) whose step is 2^-7 while the max sits in binade
    ## [4, 8) whose step is 2^-5. The budget usage is one eighth. The CPU
    ## reference orders compare bit-exact.

  DescriptorProbeWords = 512
    ## Probe subset size target, in f32 words: about 2 kB of values, the stride spreading the probe
    ## over the whole tensor.

  TailBinadesBelowMax = 4.0
    ## The tail threshold sits four binades under the recorded max: threshold = max / 2^4. A sparse
    ## inflation fault bounded by the max sits in the top four binades of the recorded range, and a blowup
    ## past the max trips the max quantile instead, so the tail fraction moves while the order
    ## statistics below the tail and the histogram mass stay put.

proc ulpFp32At*(m: float64): float64 =
  ## One fp32 ulp at magnitude m: fp32 stores 23 significand bits, so for m in [2^e, 2^(e+1)) the ulp
  ## is 2^(e-23). Zero maps to 0.
  if m <= 0.0:
    return 0.0
  pow(2.0, floor(log2(m)) - 23.0)

proc evalOrderDriftUlps*(a, b: Tensor): float64 =
  ## Measured drift of two same-shape tensors in fp32 ulps at the pair max magnitude: maxAbsDiff
  ## divided by the ulp width at the max. The reporting unit of the
  ## evaluation-order checks and of the descriptor drift bound.
  let pairMax = max(a.to(F.kFloat32).abs().max().item(float64),
    b.to(F.kFloat32).abs().max().item(float64))
  maxAbsDiff(a, b) / ulpFp32At(pairMax)

proc descriptorMaxDrift*(ts: TensorStats): float64 =
  ## The derived per-element drift bound of a descriptor entry, the same constant the probe and the descriptor
  ## bands consume: SsmUlpMargin fp32 ulps at the recorded state max for dmDrift entries, zero for dmExact
  ## entries. The drift is absolute-scale by the chain derivation: the drift of an element follows
  ## the activation bulk along its path, not the element's own binade, so one max-anchored bound
  ## covers every binade.
  if ts.probeMode == dmDrift:
    SsmUlpMargin * ulpFp32At(abs(ts.quantiles[^1].float64))
  else:
    0.0

proc descriptorStatsBudget*(mode: DescMode): ToleranceBudget =
  ## The fingerprint budget of a descriptor entry: dmExact entries compare their order statistics
  ## bit-exactly, dmDrift entries take the SsmUlpMargin fp32-ulp cap at the recorded max,
  ## the same cap constant (SsmUlpMargin fp32 ulps at the recorded max) used by
  ## the evaluation-order checks, the recorded summary bands, and this budget.
  ToleranceBudget(tier: ctDistribution,
    maxUlp: (if mode == dmDrift: int32(SsmUlpMargin) else: 0),
    maxMismatchFrac: 0.0, histL1: 0.01)

proc assertStats*(actual: Tensor, stats: TensorStats,
    budget: ToleranceBudget, msg = "", allowMinusInf = false) =
  ## Check the fingerprint of `actual` against the recorded stats.
  ##
  ##   computed tensor -> fingerprint (count, quantiles, histogram)
  ##   recorded stats  -> same summary -----------------------------> compare
  ##
  ## Two comparison rules, split by entry class:
  ## - Descriptor-carried entries (hasDescriptors): the absolute-scale drift
  ##   bound, the same one assertDescriptors enforces. Every order statistic
  ##   may drift descriptorMaxDrift at most, SsmUlpMargin fp32 ulps at the
  ##   recorded max for dmDrift entries and zero for dmExact entries. The
  ##   per-quantile own-binade ulp counting of the fingerprint-only path
  ##   would contradict that bound at the low binades and does not apply
  ##   here. The histogram L1 compares against budget.histL1.
  ## - Fingerprint-only entries: every order statistic within budget.maxUlp,
  ##   counted in the ulps of the dtype of `actual` (bf16 ulps for bf16
  ##   tensors), histogram L1 at most budget.histL1.
  let fp = tensorStats(actual, allowMinusInf = allowMinusInf,
    withHistogram = stats.hasHist, name =
      (if msg.len > 0: msg else: stats.name))
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if fp.n != stats.n:
    raise newException(HarnessCheckError,
      "[ttt] stats element count " & $fp.n & " != recorded " & $stats.n & ctx)

  proc checkHistL1() =
    if stats.hasHist:
      let l1 = normalizedL1(fp, stats)
      if l1 > budget.histL1:
        raise newException(HarnessCheckError,
          "[ttt] histogram L1 " & $l1 & " exceeds budget " & $budget.histL1 & ctx)

  if stats.hasDescriptors:
    let drift = descriptorMaxDrift(stats)
    var worst = 0.0'f64
    var worstName = ""
    for i in 0 ..< QuantileCount:
      let got = fp.quantiles[i]
      let want = stats.quantiles[i]
      let gInf = classify(got)
      let wInf = classify(want)
      if gInf == fcNegInf and wInf == fcNegInf: continue
      if gInf in {fcNaN, fcInf, fcNegInf} or wInf in {fcNaN, fcInf, fcNegInf}:
        raise newException(HarnessCheckError,
          "[ttt] stats quantile " & $i & " non-finite mismatch: got " &
          $got & " want " & $want & ctx)
      let d = abs(got.float64 - want.float64)
      if d > worst:
        worst = d
        worstName = "quantile " & $i
    if worst > drift:
      raise newException(HarnessCheckError,
        "[ttt] descriptor stats quantile drift " & $worst & " at " & worstName &
        " exceeds the absolute bound " & $drift &
        " (" & $stats.probeMode & " law, " & $SsmUlpMargin & " ulp at max)" & ctx)
    checkHistL1()
    return

  let isBf16 = actual.contiguous().scalarType() == F.kBfloat16
  let isF16 = actual.contiguous().scalarType() == F.kFloat16
  var worst = 0'i64
  var worstName = ""
  for i in 0 ..< QuantileCount:
    let got = fp.quantiles[i]
    let want = stats.quantiles[i]
    let gInf = classify(got)
    let wInf = classify(want)
    if gInf == fcNegInf and wInf == fcNegInf: continue
    if gInf in {fcNaN, fcInf, fcNegInf} or wInf in {fcNaN, fcInf, fcNegInf}:
      raise newException(HarnessCheckError,
        "[ttt] stats quantile " & $i & " non-finite mismatch: got " &
        $got & " want " & $want & ctx)
    let d = if isBf16: ulpDistance16(got, want)
            elif isF16: ulpDistanceFp16(got, want)
            else: ulpDistance32(got, want)
    if d > worst:
      worst = d
      worstName = "quantile " & $i
  if worst > budget.maxUlp.int64:
    raise newException(HarnessCheckError,
      "[ttt] stats drift " & $worst & " ulp at " & worstName &
      " exceeds budget " & $budget.maxUlp & ctx)
  checkHistL1()

proc assertStatsChainBand*(actual: Tensor, stats: TensorStats, depth: int,
    msg = "", rtol: float64 = ChainCheckpointRtol) =
  ## Check the fingerprint of `actual` against the recorded stats under the
  ## chain checkpoint band: the cross-device quantile bound is the
  ## bulk-anchored absolute band (chainCheckpointAbstol), the instrument the
  ## near-zero quantiles need, since quantile-ulp caps explode there (the
  ## SPEC measurement observed 14 912 ulps on a bf16 median). Every bound
  ## comparison is inclusive: a value landing exactly on the bound passes.
  ## Quantiles only: the histogram is the same-device margin instrument, its
  ## provisional L1 budget does not cover bulk-scale cross-device drift (the
  ## SPEC cross-device class compares the quantile fingerprints plus the
  ## chain band).
  let fp = tensorStats(actual, allowMinusInf = stats.allowMinusInf,
    withHistogram = stats.hasHist, name =
      (if msg.len > 0: msg else: stats.name))
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if fp.n != stats.n:
    raise newException(HarnessCheckError,
      "[ttt] stats element count " & $fp.n & " != recorded " & $stats.n & ctx)
  let abstol = chainCheckpointAbstol(depth, meanAbsValue(actual))
  var worst = 0.0'f64
  var worstName = ""
  for i in 0 ..< QuantileCount:
    let got = fp.quantiles[i]
    let want = stats.quantiles[i]
    let gInf = classify(got)
    let wInf = classify(want)
    if gInf == fcNegInf and wInf == fcNegInf: continue
    if gInf in {fcNaN, fcInf, fcNegInf} or wInf in {fcNaN, fcInf, fcNegInf}:
      raise newException(HarnessCheckError,
        "[ttt] stats quantile " & $i & " non-finite mismatch: got " &
        $got & " want " & $want & ctx)
    # The per-quantile band mirrors the elementwise chain band: the relative
    # term covers the recorded value's own grid steps (an outlier element
    # moves one ulp of its own binade), the absolute term anchors on the bulk.
    let d = abs(got.float64 - want.float64)
    let band = rtol * abs(want.float64) + abstol
    if d > worst:
      worst = d
      worstName = "quantile " & $i
    if d > band:
      raise newException(HarnessCheckError,
        "[ttt] stats quantile drift " & $d & " at " & worstName &
        " exceeds the chain band " & $band & " (depth " & $depth & ")" & ctx)
proc assertMatchRate*(actual, expected: Tensor,
    budget: ToleranceBudget, msg = "") =
  ## Check the element-wise match rate: the fraction of elements drifting
  ## past budget.maxUlp, counted in ulps of the dtype, must be at most
  ## budget.maxMismatchFrac. Equal values sit at distance 0. Refuses
  ## budgets without a match-rate cap (logits).
  if budget.maxUlp <= 0:
    raise newException(ValueError,
      "tolerance carries no match-rate cap, use the margin checks instead" &
      (if msg.len > 0: ": " & msg else: ""))
  # The per-element walk reads the storages through raw host pointers:
  # both tensors must be host-resident first. On CUDA a device tensor's
  # data pointer is not readable from the host (the unified-memory devices
  # hid this), so copy to the host before the view.
  let a = actual.contiguous().to(F.kCPU)
  let e = expected.contiguous().to(F.kCPU)
  if a.numel() != e.numel():
    raise newException(HarnessCheckError,
      "[ttt] match-rate element count mismatch" &
      (if msg.len > 0: ": " & msg else: ""))
  let isBf16 = a.scalarType() == F.kBfloat16
  let isF16 = a.scalarType() == F.kFloat16
  if isBf16 != (e.scalarType() == F.kBfloat16):
    raise newException(HarnessCheckError,
      "[ttt] match-rate dtype mismatch" &
      (if msg.len > 0: ": " & msg else: ""))
  if isF16 != (e.scalarType() == F.kFloat16):
    raise newException(HarnessCheckError,
      "[ttt] match-rate dtype mismatch" &
      (if msg.len > 0: ": " & msg else: ""))
  let n = a.numel()
  var mismatches = 0'i64
  var worst = 0'i64
  if isBf16:
    let av = a.view(F.kInt16).contiguous()
    let ev = e.view(F.kInt16).contiguous()
    let ra = cast[ptr UncheckedArray[int16]](av.data_ptr(int16))
    let re = cast[ptr UncheckedArray[int16]](ev.data_ptr(int16))
    for i in 0 ..< n:
      let fa = bf16ToF32(uint16(ra[i]))
      let fe = bf16ToF32(uint16(re[i]))
      if fa == fe: continue
      let d = abs(orderedBits16(uint16(ra[i])).int64 -
                  orderedBits16(uint16(re[i])).int64)
      if d > worst: worst = d
      if d > budget.maxUlp.int64: inc mismatches
  elif isF16:
    let av = a.contiguous().view(F.kInt16).contiguous()
    let ev = e.contiguous().view(F.kInt16).contiguous()
    let ra = cast[ptr UncheckedArray[int16]](av.data_ptr(int16))
    let re = cast[ptr UncheckedArray[int16]](ev.data_ptr(int16))
    for i in 0 ..< n:
      let fa = fp16ToF32(uint16(ra[i]))
      let fe = fp16ToF32(uint16(re[i]))
      if fa == fe: continue
      let d = ulpDistanceFp16(fa, fe)
      if d > worst: worst = d
      if d > budget.maxUlp.int64: inc mismatches
  else:
      let ra = cast[ptr UncheckedArray[float32]](a.data_ptr(float32))
      let re = cast[ptr UncheckedArray[float32]](e.data_ptr(float32))
      for i in 0 ..< n:
        let d = ulpDistance32(ra[i], re[i])
        if d > worst: worst = d
        if d > budget.maxUlp.int64: inc mismatches
  let frac = mismatches.float64 / n.float64
  if frac > budget.maxMismatchFrac:
    raise newException(HarnessCheckError,
      "[ttt] match-rate violated: " & $mismatches & "/" & $n &
      " elements past " & $budget.maxUlp & " ulp (worst " & $worst &
      "), budget " & $budget.maxMismatchFrac &
      (if msg.len > 0: ": " & msg else: ""))

# ########################################################################### Fingerprint stats
# files ###########################################################################

const
  FingerprintStatsFileSchema* = 1
    ## Schema of the `.stats.json` files next to committed fixtures.

  QuantileNames* = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
                    "p99"]
    ## Stats-file keys of the fixed quantiles, in QuantileProbs order.

proc decodeQuantile(s: string): float32 =
  ## Decode one stored quantile: a hex f32 bit pattern or the "-inf" whitelist. Hex keeps storage
  ## bit-exact across implementations.
  if s == "-inf": return NegInf
  if s.startsWith("0x"):
    return cast[float32](uint32(parseHexInt(s)))
  raise newException(ValueError, "bad quantile string: " & s)

type
  # jsony shape of a `.stats.json` file. Quantiles and buckets keep their stored text formats, hex f32
  # bits and packed "key:count" pairs, and decode through decodeTensorStats, the single
  # validation path every decode goes through. The descriptor keys mean_abs, signed_mean, tail_probability, tail_edge,
  # probe_stride, probe_mode and probe_bits are optional: the writer emits them only for descriptor-carried
  # entries, absent keys decode to the defaults through jsony, and fingerprint-only entries keep
  # the historical byte layout.
  QuantilesJson = object
    `min`, `max`: string
    p01, p05, p10, p25, p50, p75, p90, p95, p99: string
  HistogramJson = object
    total: int64
    buckets: string
  TensorStatsJson = object
    n: int
    allow_minus_inf: bool
    quantiles: QuantilesJson
    histogram: Option[HistogramJson]
    mean_abs: string
    signed_mean: string
    tail_probability: string
    tail_edge: int
    probe_stride: int
    probe_mode: string
    probe_bits: string
  StatsFileJson = object
    schema: int
    source: string
    tensors: OrderedTable[string, TensorStatsJson]

static:
  doAssert QuantileNames == ["p01", "p05", "p10", "p25", "p50", "p75",
    "p90", "p95", "p99"]

proc encodeQuantile(v: float32): string =
  ## Store one quantile as a hex f32 bit pattern, or "-inf".
  if classify(v) == fcNegInf: "-inf"
  else: "0x" & toHex(cast[uint32](v), 8)

proc encodeF64(v: float64): string =
  ## Store one f64 descriptor as a hex bit pattern, uppercase, 16 digits. Hex keeps storage
  ## bit-exact across implementations, like the quantile hex format.
  "0x" & toHex(cast[uint64](v), 16)

proc decodeF64(s: string): float64 =
  ## Decode one stored f64 descriptor: 16 hex digits after the 0x prefix, parsed nibble by nibble
  ## into uint64. Negative patterns carry the high bit, so the parse must stay unsigned end to end.
  if s.len != 18 or not s.startsWith("0x"):
    raise newException(ValueError, "bad f64 descriptor string: " & s)
  var u: uint64
  for i in 2 ..< s.len:
    let c = s[i]
    let d: uint64 = case c
      of '0'..'9': uint64(c.ord - 48)
      of 'a'..'f': uint64(c.ord - 87)
      of 'A'..'F': uint64(c.ord - 55)
      else: raise newException(ValueError,
        "bad f64 descriptor string: " & s)
    u = (u shl 4) or d
  cast[float64](u)

proc decodeTensorStats(name: string, j: TensorStatsJson): TensorStats =
  ## Decode one tensor entry of a stats file. The writer keeps bucket keys sorted, so sparse lookup
  ## order is stable.
  result.name = name
  result.n = j.n
  result.allowMinusInf = j.allow_minus_inf
  result.quantiles[0] = decodeQuantile(j.quantiles.`min`)
  result.quantiles[^1] = decodeQuantile(j.quantiles.`max`)
  let raw = [j.quantiles.p01, j.quantiles.p05, j.quantiles.p10,
    j.quantiles.p25, j.quantiles.p50, j.quantiles.p75, j.quantiles.p90,
    j.quantiles.p95, j.quantiles.p99]
  for pi in 0 ..< QuantileProbs.len:
    result.quantiles[pi + 1] = decodeQuantile(raw[pi])
  if j.histogram.isSome:
    result.hasHist = true
    let h = j.histogram.get
    result.histTotal = uint64(h.total)
    var last = 0'u16
    for pair in h.buckets.split(','):
      let kv = pair.split(':')
      let raw = parseBiggestInt(kv[0])
      if raw < 0 or raw > 65535:
        raise newException(ValueError,
          "stats file bucket key out of the uint16 range: " & kv[0] &
          " (a negative key would wrap into the reserved zero and subnormal region)")
      let k = uint16(raw)
      if k <= last and result.histKeys.len > 0:
        raise newException(ValueError,
          "stats file bucket keys out of sorted order: " & $k)
      last = k
      result.histKeys.add k
      result.histCounts.add uint32(parseBiggestInt(kv[1]))
  if j.mean_abs.len > 0:
    result.hasDescriptors = true
    result.meanAbs = decodeF64(j.mean_abs)
    result.signedMean = decodeF64(j.signed_mean)
    result.tailProbability = decodeF64(j.tail_probability)
    result.tailEdge = j.tail_edge
    result.probeStride = j.probe_stride
    result.probeMode = if j.probe_mode == "exact": dmExact
      elif j.probe_mode == "drift": dmDrift
      else: raise newException(ValueError,
        "bad descriptor probe mode: " & j.probe_mode)
    if j.probe_bits.len mod 8 != 0:
      raise newException(ValueError,
        "descriptor probe bits hold " & $j.probe_bits.len &
        " hex chars, not a multiple of 8: " & name)
    var i = 0
    while i < j.probe_bits.len:
      result.probeValues.add cast[float32](
        uint32(parseHexInt(j.probe_bits[i ..< i + 8])))
      i += 8
  elif j.probe_bits.len > 0:
    raise newException(ValueError,
      "descriptor probe bits without the mean_abs key: " & name)

proc statsTensor*(statsFile: FingerprintStatsFile, name: string): TensorStats =
  ## Stats entry of one tensor of a stats file, or an IOError.
  for ts in statsFile.tensors:
    if ts.name == name: return ts
  raise newException(IOError,
    "stats file " & statsFile.source & " holds no stats for tensor " & name)

proc loadFingerprintStats*(path: string): FingerprintStatsFile =
  ## Parse a `.stats.json.zst` stats frame written by harness/gen_stats.nim.
  ## jsony parses the decompressed payload straight into the schema
  ## objects. The hex quantile formats and packed bucket strings decode
  ## through decodeTensorStats with validation.
  let framePath =
    if path.endsWith(".json.zst"): path
    else: path & ".json.zst"
  let j = readFile(framePath).zstdDecompress(string).fromJson(StatsFileJson)
  result.schema = j.schema
  result.source = j.source
  for name, tj in j.tensors:
    result.tensors.add decodeTensorStats(name, tj)

proc encodeTensorStatsBody*(ts: TensorStats): string =
  ## One tensor entry as stored JSON text: hex quantile bits, packed sorted buckets, and the descriptor
  ## keys only when the entry carries them. Field order fixes the byte format on both writer sides.
  result = "{\"n\":" & $ts.n & ",\"allow_minus_inf\":" &
    (if ts.allowMinusInf: "true" else: "false") & ",\"quantiles\":{\"min\":\"" &
    encodeQuantile(ts.quantiles[0]) & "\",\"max\":\"" &
    encodeQuantile(ts.quantiles[^1]) & "\""
  for pi, qn in QuantileNames:
    result.add ",\"" & qn & "\":\"" & encodeQuantile(ts.quantiles[pi + 1]) &
      "\""
  result.add "},\"histogram\":"
  if ts.hasHist:
    var packed = ""
    for i in 0 ..< ts.histKeys.len:
      if i > 0: packed.add ','
      packed.add $ts.histKeys[i]
      packed.add ':'
      packed.add $ts.histCounts[i]
    result.add "{\"total\":" & $int64(ts.histTotal) & ",\"buckets\":\"" &
      packed & "\"}"
  else:
    result.add "null"
  if ts.hasDescriptors:
    var probeBits = ""
    for v in ts.probeValues:
      probeBits.add toHex(cast[uint32](v), 8)
    result.add ",\"mean_abs\":\"" & encodeF64(ts.meanAbs) &
      "\",\"signed_mean\":\"" & encodeF64(ts.signedMean) &
      "\",\"tail_probability\":\"" & encodeF64(ts.tailProbability) &
      "\",\"tail_edge\":" & $ts.tailEdge &
      ",\"probe_stride\":" & $ts.probeStride &
      ",\"probe_mode\":\"" &
      (if ts.probeMode == dmExact: "exact" else: "drift") &
      "\",\"probe_bits\":\"" & probeBits & "\""
  result.add "}"

proc writeFingerprintStats*(path: string, statsFile: FingerprintStatsFile) =
  ## Deterministic stats sidecar writer: sorted bucket keys, hex quantile bits, single-line JSON plus a trailing
  ## newline, inside one zstd frame (level 19, content size and checksum in the frame header).
  ## The emission is hand-rolled so descriptor-carried entries append their keys only when present:
  ## fingerprint-only entries use the fingerprint-only byte layout, and the jsony reader consumes both layouts.
  ## The python twin (fixture_stats.py stats_file_bytes) emits
  ## the same payload bytes: the zstd frame carries the bytes and defines
  ## nothing, the JSON payload inside is the format.
  var s = "{\"schema\":" & $statsFile.schema & ",\"source\":\"" &
    statsFile.source & "\",\"tensors\":{"
  for i, ts in statsFile.tensors:
    if i > 0: s.add ","
    s.add "\"" & ts.name & "\":" & encodeTensorStatsBody(ts)
  s.add "}}\n"
  # seq[byte] to string: the frame is written out byte-exact, the copy
  # mirrors the payload inflation of recording.nim zstdReadFixture.
  writeFile(path, s.zstdCompress(string))

# ########################################################################### Descriptor checks
# ###########################################################################

proc tensorDescriptors*(t: Tensor, name: string, mode: DescMode,
    allowMinusInf = false, withHistogram = false): TensorStats =
  ## Record the fingerprint plus descriptors of one tensor: the recorded
  ## summary of a descriptor-carried entry, what a fixture stores instead
  ## of the full tensor.
  ## - fingerprint: order statistics plus optional histogram (tensorStats)
  ## - bulk mean and signed mean: f64 accumulation over the promoted values
  ##   in index order, the same arithmetic the python twin runs, so the two
  ##   agree bit-exactly
  ## - tail threshold and probe subset: both follow the recorded max
  result = tensorStats(t, allowMinusInf = allowMinusInf,
    withHistogram = withHistogram, name = name)
  let tc = t.contiguous().to(F.kCPU).to(F.kFloat32)
  let n = tc.numel()
  let rc = cast[ptr UncheckedArray[float32]](tc.data_ptr(float32))
  var absSum, sum = 0.0'f64
  for i in 0 ..< n:
    let v = rc[i].float64
    absSum += abs(v)
    sum += v
  result.hasDescriptors = true
  result.probeMode = mode
  result.meanAbs = absSum / n.float64
  result.signedMean = sum / n.float64
  let threshold = abs(result.quantiles[^1].float64) /
    pow(2.0, TailBinadesBelowMax)
  let maxDrift = descriptorMaxDrift(result)
  var tailCount, edgeCount = 0
  for i in 0 ..< n:
    let a = abs(rc[i].float64)
    if a > threshold:
      inc tailCount
    if abs(a - threshold) <= maxDrift:
      inc edgeCount
  result.tailProbability = tailCount.float64 / n.float64
  result.tailEdge = edgeCount
  let stride = if n <= DescriptorProbeWords: 1 else:
    ceilDiv(n, DescriptorProbeWords)
  result.probeStride = stride
  let count = ceilDiv(n, stride)
  for i in 0 ..< count:
    result.probeValues.add rc[i * stride]

proc probeBinadeCensus*(actual: Tensor, ts: TensorStats): string =
  ## Returns the maximum drift per binade over the probe points, the per-element count of ulp
  ## distances grouped by binade. Honest drift is absolute-scale, so the report reads one
  ## near-constant across binades, and a reading that concentrates in one binade localizes the
  ## fault. The value returns as the printed report line, the assertion runs in assertDescriptors.
  let a = actual.contiguous().to(F.kCPU).to(F.kFloat32)
  let stride = ts.probeStride
  let count = ceilDiv(a.numel(), stride)
  let ra = cast[ptr UncheckedArray[float32]](a.data_ptr(float32))
  var binades: seq[int]
  var maxima: seq[float64]
  for i in 0 ..< count:
    let idx = i * stride
    let rec = ts.probeValues[i].float64
    if rec == 0.0:
      continue
    let b = floor(log2(abs(rec))).int
    let d = abs(ra[idx].float64 - rec)
    var slot = -1
    for j, bb in binades:
      if bb == b: slot = j
    if slot < 0:
      binades.add b
      maxima.add d
    elif d > maxima[slot]:
      maxima[slot] = d
  for j in 0 ..< binades.len:
    let line = (if j > 0: ", " else: "") & "2^" & $binades[j] & ": " &
      formatBiggestFloat(maxima[j], ffScientific, 2)
    result.add line

proc assertDescriptors*(actual: Tensor, ts: TensorStats, msg = "") =
  ## Check a computed tensor against the recorded descriptor entry: element count, the bulk and signed
  ## mean bands, the tail-probability band, and the probe subset elementwise.
  ## - dmExact: every comparison allows zero drift. The probe values and the f64 means must reproduce
  ## the recording bit for bit.
  ## - dmDrift: every probe element may drift the derived bound (descriptorMaxDrift) at most,
  ## absolute-scale. The bulk and signed mean bands take the same bound, and the tail band is the recorded
  ## edge count over n, the elements that could cross the threshold under honest drift.
  ## The probe is elementwise corroboration of the fingerprint, not a replacement for it:
  ## corruption outside the probe set is policed by the fingerprint of assertStats and the descriptor
  ## bands together. See the selftest floor table for the measured split.
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if not ts.hasDescriptors:
    raise newException(ValueError,
      "descriptor check on a fingerprint-only entry" & ctx)
  let a = actual.contiguous().to(F.kCPU).to(F.kFloat32)
  if a.numel() != ts.n:
    raise newException(HarnessCheckError,
      "[ttt] descriptor element count " & $a.numel() & " != recorded " &
      $ts.n & ctx)
  let n = a.numel()
  let ra = cast[ptr UncheckedArray[float32]](a.data_ptr(float32))
  let maxDrift = descriptorMaxDrift(ts)

  var absSum, sum = 0.0'f64
  for i in 0 ..< n:
    let v = ra[i].float64
    absSum += abs(v)
    sum += v
  let meanAbs = absSum / n.float64
  let signedMean = sum / n.float64
  if abs(meanAbs - ts.meanAbs) > maxDrift:
    raise newException(HarnessCheckError,
      "[ttt] descriptor meanAbs drift " &
      formatBiggestFloat(meanAbs - ts.meanAbs, ffScientific, 3) &
      " exceeds bound " & formatBiggestFloat(maxDrift, ffScientific, 3) & ctx)
  if abs(signedMean - ts.signedMean) > maxDrift:
    raise newException(HarnessCheckError,
      "[ttt] descriptor signedMean drift " &
      formatBiggestFloat(signedMean - ts.signedMean, ffScientific, 3) &
      " exceeds bound " & formatBiggestFloat(maxDrift, ffScientific, 3) & ctx)

  let threshold = abs(ts.quantiles[^1].float64) / pow(2.0, TailBinadesBelowMax)
  var tailCount = 0
  for i in 0 ..< n:
    if abs(ra[i].float64) > threshold:
      inc tailCount
  let tail = tailCount.float64 / n.float64
  let tailBand = ts.tailEdge.float64 / n.float64
  if abs(tail - ts.tailProbability) > tailBand:
    raise newException(HarnessCheckError,
      "[ttt] descriptor tail probability " & $tail & " vs recorded " &
      $ts.tailProbability & " outside band " & $tailBand & ctx)

  let stride = ts.probeStride
  let count = ceilDiv(n, stride)
  if ts.probeValues.len != count:
    raise newException(HarnessCheckError,
      "[ttt] descriptor probe holds " & $ts.probeValues.len &
      " values, tensor size " & $n & " at stride " & $stride &
      " wants " & $count & ctx)
  var worst = 0.0'f64
  var worstIdx = -1
  for i in 0 ..< count:
    let idx = i * stride
    let rec = ts.probeValues[i]
    let d = abs(ra[idx].float64 - rec.float64)
    if d > worst:
      worst = d
      worstIdx = idx
    if d > maxDrift:
      raise newException(HarnessCheckError,
        "[ttt] descriptor probe element " & $idx & " drift " &
        formatBiggestFloat(d, ffScientific, 3) & " exceeds bound " &
        formatBiggestFloat(maxDrift, ffScientific, 3) & ctx)
  let census = probeBinadeCensus(actual, ts)
  echo "    descriptors passed (mode " & $ts.probeMode & ", worst probe drift " &
    formatBiggestFloat(worst, ffScientific, 3) & " at " & $worstIdx &
    ", bound " & formatBiggestFloat(maxDrift, ffScientific, 3) &
    (if census.len > 0: "; per-binade max drift " & census else: "") & ")"

# ########################################################################### Final logits decision
# projections ###########################################################################

proc bf16Ulp*(binade: int): float64 =
  ## Width of one bf16 ulp for values in [2^binade, 2^(binade+1)). bf16 stores 7 mantissa bits, so the width
  ## is 2^(binade - 7), the value bf16UlpAt returns inside the same binade.
  pow(2.0, (binade - 7).float64)

proc bf16UlpAt*(v: float64): float64 =
  ## Width of one bf16 ulp at magnitude |v| for normal values.
  let u = bf16BitsFromF32(v.float32)
  let e = (u.int shr 7 and 0xFF) - 127 - 7
  pow(2.0, e.float64)

proc fp16Ulp*(binade: int): float64 =
  ## Width of one fp16 ulp for values in [2^binade, 2^(binade+1)). fp16 stores
  ## 10 mantissa bits, so the width is 2^(binade - 10), the value fp16UlpAt
  ## returns inside the same binade.
  pow(2.0, (binade - 10).float64)

proc fp16UlpAt*(v: float64): float64 =
  ## Width of one fp16 ulp at magnitude |v| for normal values. The exl3
  ## families use the bf16 bounds with the ulp unit taken in fp16 because
  ## EXL3 dequantizes to fp16.
  let u = fp16BitsFromF32(v.float32)
  let e = (u.int shr 10 and 0x1F) - 15 - 10
  pow(2.0, e.float64)

# ########################################################################### Evaluation-order
# checks ###########################################################################

proc assertConvEvalOrder*(a, b: Tensor, msg = "") =
  ## Check the conv state under the two evaluation orders of the GDN
  ## recurrence:
  ##
  ##   chunked scan (whole sequence at once) -> conv state A -\
  ##                                                          compare -> ConvEvalOrderBudget
  ##   step-by-step loop (one token at a time) -> conv state B -/
  ##
  ## Floating-point addition is not associative: the two orders round
  ## differently, the budget absorbs the rounding. The budget is the derived
  ## ConvEvalOrderBudget, counted in bf16 ulp-units at the tensor max.
  ## On hardware that reorders the conv accumulation the two orders differ
  ## by one bf16 rounding step, so the budget is derived from that rounding
  ## model and calibrated by the Metal measurement, see the budget-change
  ## record in SPEC.md.
  ## Companion: assertSsmEvalOrder checks the f32 state member.
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if a.shape != b.shape:
    raise newException(HarnessCheckError,
      "evaluation-order shapes disagree" & ctx)
  let maxAt = max(a.to(F.kFloat32).abs().max().item(float64),
    b.to(F.kFloat32).abs().max().item(float64))
  let drift = maxAbsDiff(a, b)
  let unit = bf16UlpAt(maxAt)
  let ulps = drift / unit
  if drift > ConvEvalOrderBudget * unit:
    raise newException(HarnessCheckError,
      "evaluation-order conv drift " & $drift & " = " & $ulps &
      " bf16 ulp-units at max " & $maxAt & " exceeds the budget " &
      $ConvEvalOrderBudget & " ulp-units (" & $(ConvEvalOrderBudget * unit) &
      " absolute)" & ctx)
  echo "    evaluation-order conv drift " & $drift & " = " & $ulps &
    " bf16 ulp-units at max " & $maxAt & ", budget " & $ConvEvalOrderBudget &
    " ulp-units"

proc assertEvalOrderOutput*(drift, refMax: float64, msg = "") =
  ## Check the bf16 block output under the two evaluation orders:
  ##
  ##   per-step outputs, chunked vs step-by-step -> max drift --\
  ##   output max -> bf16 ulp width -----------------------------|--> SsmUlpMargin
  ##
  ## The drift must stay inside SsmUlpMargin bf16 ulps, counted at the
  ## output max. The caller measures both numbers: the drift is the max
  ## elementwise difference over the decode steps, the reference max
  ## supplies the magnitude anchor. The check prints the measured numbers,
  ## the drift line of the suite report.
  let ctx = (if msg.len > 0: ": " & msg else: "")
  let unit = bf16UlpAt(refMax)
  let ulps = drift / unit
  if drift > SsmUlpMargin * unit:
    raise newException(HarnessCheckError,
      "evaluation-order output drift " & $drift & " = " & $ulps &
      " bf16 ulps at output max " & $refMax & " exceeds the " &
      $SsmUlpMargin & " ulp cap" & ctx)
  echo "    evaluation-order output drift " & $drift & " = " & $ulps &
    " bf16 ulps at output max " & $refMax

proc assertSsmEvalOrder*(a, b: Tensor, steps: int, msg = "") =
  ## Check the state under the two evaluation orders of the GDN recurrence:
  ##
  ##   chunked scan (whole sequence at once) -> f32 state A -\
  ##                                                          compare -> drift bound
  ##   step-by-step loop (one token at a time) -> f32 state B -/
  ##
  ## State bound: coherent rounding-error accumulation. The f32
  ## accumulation orders diverge by a per-step difference from the
  ## addition order, with a consistent sign: with T the decode length, the drift adds
  ## up linearly in T (O(T)), not as a zero-mean random walk
  ## (sqrt(T)). The bound reuses the chain derivation
  ## unchanged, no new constants:
  ##   abstol(steps) = 2^-3 * steps * meanAbs(reference)
  ## Source: chainCheckpointAbstol with ChainCheckpointScaleFactor 2^-3,
  ## see the budget-change record in SPEC.md. The caller passes the decode
  ## length as `steps`.
  ## Calibration at the measured point: the T=70
  ## Metal decode measures 1027.4 fp32 ulps at the pair max 1.032, about
  ## 15 fp32 ulps per step of growth, inside the bound. The rejection
  ## rows of the selftest show a 2x-bound corruption still rejects.
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if a.shape != b.shape:
    raise newException(HarnessCheckError,
      "evaluation-order shapes disagree" & ctx)
  let pairMax = max(a.to(F.kFloat32).abs().max().item(float64),
    b.to(F.kFloat32).abs().max().item(float64))
  let drift = maxAbsDiff(a, b)
  let bulk = meanAbsValue(a)
  let bound = chainCheckpointAbstol(steps, bulk)
  let ulps = drift / ulpFp32At(pairMax)
  let boundUlps = bound / ulpFp32At(pairMax)
  if drift > bound:
    raise newException(HarnessCheckError,
      "evaluation-order state drift " & $drift & " = " & $ulps &
      " fp32 ulps at pair max " & $pairMax & " exceeds the " & $steps &
      "-step law bound " & $bound & " (bulk " & $bulk & ")" & ctx)
  echo "    evaluation-order state drift " & $drift & " = " & $ulps &
    " fp32 ulps at pair max " & $pairMax & ", bound " & $bound & " = " &
    $boundUlps & " ulps at pair max (" & $steps & "-step law, bulk " &
    $bulk & ")"

const
  LogitsProjectionSchema* = "tt-final-logits-projection-1"
  ## Schema of the first-generation final_logits.decisions.json.zst
  ## payloads, decisions only.
  LogitsProjectionSchema2* = "tt-final-logits-projection-2"
  ## Schema adding the strided bit-exact sample of every deciding row:
  ## 512 f32 words per position, one every ceil(vocab/512) positions.
  ## The f32 logit agreement cap stays 1e-5 absolute, the committed
  ## top-2 cap.
  ## Schema of the final_logits.decisions.json.zst payloads: one step per position carrying the argmax
  ## id, the top-2 competing pair, the argmax margin and the softmax tail probability beyond the pair.

type
  DecisionStep* = object
    ## One position of a final-logits decision projection.
    position*: int
    argmaxId*: int
    top2Ids*: seq[int]
    top2Logits*: seq[float32]
    argmaxMargin*: float64
    tailProbability*: float64
    probeStride*: int
      ## Strided probe step of the deciding row, 0 when the step carries
      ## no probe (schema 1).
    probeMode*: string
      ## "exact": the probe words compare bit-exactly on the reference
      ## device.
    probeBits*: string
      ## Packed hex f32 probe words, 8 hex digits per word.

  LogitsProjection* = object
    ## Parsed payload of a final_logits.decisions.json.zst fixture.
    schema*: string
    model*: string
    vocabSize*: int
    steps*: seq[DecisionStep]

proc assertProjection*(logits: Tensor, projection: LogitsProjection, msg = "",
    ulpUnitF16 = false) =
  ## Check computed final logits against the recorded decision projection: argmax id and top-2
  ## competing pair per position, the ulp-banded logit row, argmax margin, and the
  ## tail-probability checksum beyond the pair. Raw logits tensors leave the tree, the consumers
  ## read the decisions only.
  ##
  ## The ulp unit follows the recorded dequant grid (the GreedyConfig.ulpUnitF16
  ## switch): fp16 for the exl3 families, bf16 otherwise. The ulp count is the
  ## per-op row budget (4) evaluated at the recorded top-1 logit binade — the
  ## support-wide ulpFloor shape of checkGreedyStep, because the chain
  ## perturbation is a logit-scale absolute drift. bf16 is the loose grid; the
  ## same ulp count on the fp16 grid is a strictly tighter absolute band, never
  ## worse.
  let ulpAt = if ulpUnitF16: fp16UlpAt else: bf16UlpAt
  let ctx = (if msg.len > 0: ": " & msg else: "")
  let row = logits.contiguous().to(F.kCPU).to(F.kFloat32)
  if not (row.dim == 3 and row.size(0) == 1):
    raise newException(HarnessCheckError,
      "final logits must be [1, seq, vocab]" & ctx)
  if row.size(1) != projection.steps.len:
    raise newException(HarnessCheckError,
      "projection step count " & $projection.steps.len &
      " != logits sequence length " & $row.size(1) & ctx)
  for step in projection.steps:
    let flat = row.narrow(1, step.position, 1).squeeze(1).squeeze(0)
    let n = flat.numel()
    let raw = cast[ptr UncheckedArray[float32]](flat.data_ptr(float32))
    var obsArgmax = 0
    var obsTop1 = NegInf
    var obsTop2 = NegInf
    for i in 0 ..< n:
      let v = raw[i].float64
      if v > obsTop1:
        obsTop2 = obsTop1
        obsTop1 = v
        obsArgmax = i
      elif v > obsTop2:
        obsTop2 = v
    let tieEligible = step.argmaxMargin <= ulpAt(step.top2Logits[0].float64) and
      obsArgmax in step.top2Ids and
      abs(obsTop1 - step.top2Logits[0].float64) <=
        max(4.0 * ulpAt(step.top2Logits[0].float64), 1e-5)
    if not (obsArgmax == step.argmaxId or tieEligible):
      raise newException(HarnessCheckError,
        "position " & $step.position & " argmax " & $obsArgmax &
        " != recorded " & $step.argmaxId & ctx)
    for slot in 0 ..< step.top2Ids.len:
      let id = step.top2Ids[slot]
      let diff = abs(raw[id].float64 - step.top2Logits[slot].float64)
      # 4-ulp band at the recorded top-1 logit binade: the chain
      # perturbation is a logit-scale absolute drift (~1 fp16 ulp measured
      # on the rebuilt binary), the support-wide ulpFloor shape of
      # checkGreedyStep.
      if diff > max(1e-5, 4.0 * ulpAt(step.top2Logits[0].float64)):
        raise newException(HarnessCheckError,
          "position " & $step.position & " top2 slot " & $slot &
          " logit drift " & $diff & ctx)
    let marginDrift = abs(obsTop1 - obsTop2 - step.argmaxMargin)
    # The margin combines two banded logits: the cap is the top-2 band
    # twice (2 x 4 ulp at the top-1 binade), not an independent constant.
    if marginDrift >
        max(1e-5, 8.0 * ulpAt(step.top2Logits[0].float64)):
      raise newException(HarnessCheckError,
        "position " & $step.position & " argmax margin drift " & $marginDrift &
        " (recorded margin " & $step.argmaxMargin & ", cap " &
        $max(1e-5, 8.0 * ulpAt(step.top2Logits[0].float64)) & ")" & ctx)
    let probs = F.softmax(flat, dim = -1)
    var kept = 0.0'f64
    for id in step.top2Ids:
      kept += probs[id].item(float64)
    let tail = 1.0 - kept
    # The greedy tailBand row shape (tailBand x max(tail, 1e-3) + 1e-4).
    # The band term derives from the 4-ulp logit floor through softmax
    # sensitivity: a logit-scale shift delta moves the tail probability by
    # (exp(delta) - 1) x tail, so tailBand = exp(4 ulp at the top-1
    # binade) - 1, capped at the bf16 greedy tailBand 0.3 — the 4-ulp floor
    # must never translate into a looser tail row than the bf16 convention
    # (the fp16 grids stay strictly tighter, e.g. 3.1e-2 at logit 8).
    # Measured on the CUDA reference box with the rebuilt binary:
    # 1.65e-3/8.2e-3/1.24e-2 relative at tails 0.28/0.027/0.38. A
    # wrong-weights bug shifts tails by O(1) and stays detected.
    let tailLimit = min(0.3,
      exp(4.0 * ulpAt(step.top2Logits[0].float64)) - 1.0) *
      max(step.tailProbability, 1e-3) + 1e-4
    if abs(tail - step.tailProbability) > tailLimit:
      raise newException(HarnessCheckError,
        "position " & $step.position & " tail probability " & $tail &
        " vs recorded " & $step.tailProbability & " outside band " & $tailLimit & ctx)
    if step.probeBits.len > 0:
      # Schema-2 strided sample (probe_stride, probe_mode, probe_bits on disk)
      # of the deciding row over the reference device. The stride must cover the row with 512 words,
      # indices i * stride, ceil(n / stride) == word count. Compared on the
      # 4-ulp band at the recorded top-1 binade (same chain drift as the
      # top2 cap).
      if not (step.probeStride > 0 and step.probeMode == "exact"):
        raise newException(HarnessCheckError,
          "position " & $step.position & " probe stride or mode malformed" & ctx)
      if (n - 1) div step.probeStride + 1 != 512:
        raise newException(HarnessCheckError,
          "position " & $step.position & " probe stride " & $step.probeStride &
          " does not cover " & $n & " logits with 512 words" & ctx)
      if step.probeBits.len != 512 * 8:
        raise newException(HarnessCheckError,
          "position " & $step.position & " probe word count malformed" & ctx)
      for i in 0 ..< 512:
        let idx = i * step.probeStride
        let rec = cast[float32](uint32(parseHexInt(
          step.probeBits[i * 8 ..< i * 8 + 8])))
        # The chain perturbation is a logit-scale absolute drift, so tail
        # words carry the same absolute noise as the top pair: the band is
        # the 4-ulp floor at the recorded top-1 binade (the support-wide
        # ulpFloor shape of checkGreedyStep).
        if abs(raw[idx] - rec) > max(1e-5,
            4.0 * ulpAt(step.top2Logits[0].float64)):
          raise newException(HarnessCheckError,
            "position " & $step.position & " probe word " & $i &
            " at logit " & $idx & " drifts from the recorded row" & ctx)

# ########################################################################### Greedy prefix checks

type
  GreedyStepRef* = object
    ## Recorded reference for one greedy decode step (tt-greedy-2 schema).
    step*: int
    chosenToken*: int
    top32Ids*: seq[int]
    top32Logits*: seq[float32]
    argmaxMargin*: float64
      ## f32 gap between the top-1 and top-2 logit. Zero means structural tie on the bf16 grid.
    tailProbability*: float64
      ## Softmax probability mass beyond the top-32 support.
    tailRecorded*: bool = true
      ## Whether the recording carries the tail probability at all. The exl3
      ## greedy payloads recorded before the tt-greedy-2 migration hold the
      ## top-10 support only, no tail: the checksum step is skipped and the
      ## margin-scaled cap and the truncated KL carry the step check.

  GreedyConfig* = object
    ## Checks for a greedy chain replay.
    tieUlps*: int
      ## Tie-eligibility in ulps of the recorded logit grid: a margin at or under tieUlps *
      ## bf16Ulp(recorded top logit binade) makes a divergence a tie flip eligible for
      ## teacher-forced recovery. The value 1 matches the recorded margins of the fixture
      ## corpus, whose smallest nonzero margin is one bf16 ulp at logit 16.
    epsBase*: float64
      ## Top-32 f32 logit agreement at a unit recorded margin. The step cap adds epsBase *
      ## argmaxMargin over the 4 bf16 ulp device floor of the attention budget.
    tailBand*: float64
      ## Relative band on the tail-probability checksum, with a 1e-4 absolute floor for vanishing
      ## tails.
    klBand*: float64
      ## Band on the truncated KL over the top-32 support.
    maxFlips*: int
      ## Tie-flip cap per chain. Past the cap the chain fails.
    ulpFloorUlps*: int = 4
      ## Honest cross-device drift floor in ulps of the recorded top logit. 4 sizes the dense
      ## one-GDN-block stacks. The 40-layer 35B MoE stack measured up to 36 bf16 ulps of support
      ## drift on MPS and takes 64.
    ulpUnitF16*: bool = false
      ## Ulp unit of the floor and the tie band: fp16 when set (the exl3
      ## families use the bf16 bounds with the ulp unit taken in fp16 because
      ## EXL3 dequantizes to fp16), bf16 otherwise.

  GreedyState* = object
    ## Per-chain bookkeeping for the greedy checks.
    flips*: int
      ## Tie flips seen so far, teacher-forced recoveries included.

  GreedyVerdict* = enum
    ## Outcome of one greedy step check.
    gvAgree
      ## Argmax agrees with the recording and the step checksums hold.
    gvTieFlip
      ## Near-tie divergence. The caller teacher-forces the recorded token back in and replays.
      ## Flips within the cap are expected. A real divergence or a cap overrun raises
      ## directly, so no verdict value exists for the fail case.

proc greedyFlatRow*(logitsRow: Tensor): Tensor =
  ## Flat f32 CPU copy of the deciding logits row, squeezed to [V]. The copy is made on cpu because
  ## raw pointer reads below are host reads: a row computed on a Metal device holds no host storage.
  var t = logitsRow.contiguous().to(F.kCPU)
  while t.dim > 1:
    t = t.squeeze(0)
  if t.dim != 1:
    raise newException(ValueError, "greedy logits row is not flat")
  t.to(F.kFloat32).contiguous()

proc truncatedKl*(refLogits, obsLogits: seq[float32]): float64 =
  ## KL(p_ref || p_obs) over the shared support, both renormalized there.
  let n = refLogits.len
  var zr, zo = 0.0'f64
  for i in 0 ..< n:
    zr += exp(refLogits[i].float64)
    zo += exp(obsLogits[i].float64)
  for i in 0 ..< n:
    let pr = exp(refLogits[i].float64) / zr
    let po = exp(obsLogits[i].float64) / zo
    if pr > 0.0:
      result += pr * ln(pr / po)

proc observedTailProbability*(row: Tensor, supportIds: seq[int]): float64 =
  ## Softmax mass beyond the top-32 support of one f32 logits row.
  let probs = F.softmax(row, dim = -1)
  let idTensor = F.toTensor(supportIds.mapIt(it.int64))
  let kept = probs.index_select(0, idTensor).to(F.kFloat32)
  1.0 - kept.sum().item(float64)

proc checkGreedyStep*(state: var GreedyState, cfg: GreedyConfig,
    refStep: GreedyStepRef, logitsRow: Tensor): GreedyVerdict =
  ## Check one greedy decode step against its recorded reference:
  ## - argmax agreement with the recorded chosen token
  ## - margin-scaled f32 logit agreement over the top-32 support
  ## - truncated-KL and tail-probability checksums
  ## A near-tie divergence (recorded margin at or under the floor) returns gvTieFlip
  ## within the flip cap. A real divergence or a cap overrun raises a HarnessCheckError
  ## naming the step.
  let row = greedyFlatRow(logitsRow)
  let n = row.numel()
  let raw = cast[ptr UncheckedArray[float32]](row.data_ptr(float32))
  var obsArgmax = 0
  var obsTop1 = NegInf
  var obsTop2 = NegInf
  for i in 0 ..< n:
    let v = raw[i].float64
    if v > obsTop1:
      obsTop2 = obsTop1
      obsTop1 = v
      obsArgmax = i
    elif v > obsTop2:
      obsTop2 = v

  # The ulp floor accounts for the honest device drift, MPS vs cpu reference. The ulp count is per stack
  # (ulpFloorUlps). The margin term accounts for reference sensitivity.
  let ulpAt = if cfg.ulpUnitF16: fp16UlpAt else: bf16UlpAt
  let ulpFloor = cfg.ulpFloorUlps.float64 *
    ulpAt(refStep.top32Logits[0].float64)
  if obsArgmax == refStep.chosenToken:
    # Agreement: margin-scaled logit cap over the recorded support.
    let epsStep = cfg.epsBase * refStep.argmaxMargin + ulpFloor
    var obs32 = newSeq[float32](refStep.top32Logits.len)
    var worst = 0.0'f64
    var worstId = -1
    for i, id in refStep.top32Ids:
      obs32[i] = raw[id]
      let d = abs(obs32[i].float64 - refStep.top32Logits[i].float64)
      if d > worst:
        worst = d
        worstId = id
    if worst > epsStep:
      raise newException(HarnessCheckError,
        "[greedy] step " & $refStep.step & " logit drift " & $worst &
        " at id " & $worstId & " exceeds cap " & $epsStep &
        " (margin " & $refStep.argmaxMargin & ")")
    let kl = truncatedKl(refStep.top32Logits, obs32)
    if kl > cfg.klBand:
      raise newException(HarnessCheckError,
        "[greedy] step " & $refStep.step & " truncated KL " & $kl &
        " exceeds band " & $cfg.klBand)
    if refStep.tailRecorded:
      let tail = observedTailProbability(row, refStep.top32Ids)
      let tailLimit = cfg.tailBand * max(refStep.tailProbability, 1e-3) + 1e-4
      if abs(tail - refStep.tailProbability) > tailLimit:
        raise newException(HarnessCheckError,
          "[greedy] step " & $refStep.step & " tail probability " & $tail &
          " vs recorded " & $refStep.tailProbability & " outside band " &
          $tailLimit)
    return gvAgree

  # Divergence: tie flip or real bug. Tie-eligibility scales with the logit grid: one bf16 ulp is 0.125
  # at logit 16 but 0.25 at logit 32. The floor is tieUlps ulps of the step's own recorded top
  # logit, never an absolute constant.
  if refStep.argmaxMargin <=
      cfg.tieUlps.float64 * ulpAt(refStep.top32Logits[0].float64):
    # The pick must sit at the tie: its logit inside a tie band about the recorded top-1, floored
    # at 4 bf16 ulp and epsBase.
    let tieBand = max(ulpFloor, cfg.epsBase)
    let pickLogit = raw[obsArgmax].float64
    if abs(pickLogit - refStep.top32Logits[0].float64) > tieBand:
      raise newException(HarnessCheckError,
        "[greedy] step " & $refStep.step & " divergence is not a tie: " &
        "pick " & $obsArgmax & " logit " & $pickLogit & " vs recorded " &
        $refStep.top32Logits[0] & " outside band " & $tieBand &
        " (recorded margin " & $refStep.argmaxMargin & ")")
    inc state.flips
    if state.flips > cfg.maxFlips:
      raise newException(HarnessCheckError,
        "[greedy] tie-flip cap " & $cfg.maxFlips & " exceeded at step " &
        $refStep.step & " (pick " & $obsArgmax & " vs recorded " &
        $refStep.chosenToken & ", margin " & $refStep.argmaxMargin & ")")
    return gvTieFlip

  raise newException(HarnessCheckError,
    "[greedy] step " & $refStep.step & " real divergence: pick " &
    $obsArgmax & " (logit " & $raw[obsArgmax] & ", margin " & $(obsTop1 - obsTop2) & ") vs recorded " & $refStep.chosenToken & " (logit " &
    $refStep.top32Logits[0] & ", margin " & $refStep.argmaxMargin & ")")

