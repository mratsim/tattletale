# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Recorded-side checks for the transformer suites, the uniform
## stats record, the argmax decision record, the derived allowances
## both asserts consume, and the `.json.zst` record frames.
##
## Allowance derivation is one function, `deriveBands`
## - inputs, the per-stage error-model allowance and the activation
##   ulp datatype of the measured side
## - no tuned constants, no serialized allowance
## The KL allowance is the perturbation bound KL(p||q) <= 0.5 delta^2.
## Cross-model equivalence comparisons never route through the allowances,
## their epsilon is grounded by measurement.

import
  std/algorithm,
  std/math,
  std/sequtils,
  std/strutils,
  std/tables,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/zstd/zstd_highlevel


# #######################################################################
#                         Recorded-frame geometry
# #######################################################################
const
  MaxTieFlips = 4
    ## Tie-flip cap of the assertArgMax flip counter, harness policy
    ## held in code, never serialized and never a loader parameter.

  QuantileProbs = [0.01'f64, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ## Fixed quantile probabilities, storage and comparison use exact
    ## order statistics:
    ## - index = floor(p * (n - 1)) of the ascending sort, no interpolation

  QuantileCount = 2 + QuantileProbs.len
    ## Min, the 9 fixed quantiles, max.

  QuantileNames = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
                   "p99"]
    ## Record-file keys of the fixed quantiles, in QuantileProbs order.

  HistKeyZero = uint16(0xFFFE)
    ## Histogram key of the exact zero bin.

  HistKeySubnormal = uint16(0xFFFF)
    ## Histogram key of the subnormal bin.

  HistBinadeMin = -64
    ## Lowest binade a histogram bucket covers.

  HistBinadeMax = 63
    ## Highest binade a histogram bucket covers.

  TailBinadesBelowMax = 4
    ## Tail threshold placement, the threshold is recorded max / 2^4.

  TailEdgeSteps = 4
    ## Edge count of the tail instrument, measured in the threshold-binade
    ## ulp steps that give the honest drift allowance past the threshold.

  UniformStatsSchema = "ttt-tf-004-uniform-stats"
    ## Format registry id of the uniform stats record frame, one
    ## record per tensor:
    ## - order statistics (hex f32 bits), packed histogram buckets
    ## - hex f64 means, the tail probability, the edge count

  ArgmaxDecisionsSchema = "ttt-tf-005-argmax-decisions"
    ## Format registry id of the argmax decision frame, one record per
    ## greedy step:
    ## - argmax id, the top-32 set (hex f32 logits), margin, tail probability
    ## - the per-frame `ulp_datatype` key, no serialized instrument constant


# #######################################################################
#                                  Types
# #######################################################################
type
  HarnessCheckError* = object of CatchableError
    ## Rejection type of the recorded-side checks, every fault
    ## detection raises exactly this type:
    ## - a crash (parse error, I/O error, shape mismatch) propagates,
    ##   it fails the run
    ## - a fault rejection is every other raise

  RoundingErrorSourceKind* = enum
    ## Error model of one recorded tensor, keyed by the policed fault class.
    kElementwise
      ## Same-kernel elementwise replay, about two reordering
      ## opportunities per element, zero mismatch allowed.
    kReduction
      ## Reduction-shaped replay (attention, mixer, composed chains),
      ## about four reordering opportunities plus the histogram L1 instrument.

  UlpDatatype* = enum
    ## Ulp datatypes of the recorded fixture families.
    ulpBf16
      ## bf16 storage, 7 mantissa bits, one step is 2^(binade - 7)
    ulpFp16
      ## fp16 storage, 10 mantissa bits, one step is 2^(binade - 10)
    ulpFp32
      ## f32 storage, 23 mantissa bits, one step is 2^(binade - 23)
      ## Reduction-shaped replay (attention, mixer, composed chains),
      ## about four reordering opportunities plus the histogram L1 instrument.

  ArgmaxRecord = object
    ## Statistical descriptors of one recorded greedy step, carrying
    ## nothing beyond them except the recording's error-model class
    ## and ulp datatype
    argmaxId: int
      ## Recorded argmax token id.
    topK: seq[int]
      ## Token ids of the top-32 set, topK[0] == argmaxId.
    topKLogits: seq[float32]
      ## f32 logits of the top-32 set, index-aligned with topK.
    margin: float64
      ## Gap between the top-1 and the top-2 logit, a margin of at most
      ## one ulp of the datatype makes a flipped pick legal.
    tailProbability: float64
      ## Softmax mass beyond the top-32 set.
    kind: RoundingErrorSourceKind
      ## Error model class of the recorded chain:
      ## - the allowances derive at check, the per-stage class constant
      ##   times the depth, never a serialized constant
      ## - the path form takes the class as the assert's kind argument
      ## - the value form reads the record's own field
    ulpDatatype: UlpDatatype
      ## Ulp datatype of the recording and its decision allowances:
      ## - one unit = one representable-value step at the check magnitude
      ## - bf16 over the dense families, fp16 over the EXL3 chains
      ## - the record's datatype when the key carries no name

type
  StatsRecord = object
    ## Uniform stats frame for one recorded fixture, one entry
    ## per recorded tensor.
    ##
    ## Uniform fingerprint of one recorded tensor:
    ## - exact order statistics (min, the 9 fixed quantiles, max, f32)
    ## - soft binade histogram, the buckets keyed by the recorded
    ##   ulp datatype's pattern
    ## - f64 mean absolute value and signed mean, tail probability past
    ##   the threshold (the recorded max / 2^4) with the recorded edge
    ##   count as its band
    ##
    ## Masked rows carry -Inf in the quantiles, -Inf compares -Inf
    ## equal to -Inf in the asserts.
    name: string
      ## Recorded tensor name, the lookup key of the stats frame.
    n: int
      ## Element count of the recorded tensor.
    ulpDatatype: UlpDatatype
      ## Ulp datatype of the recorded dtype, the band reference input.
    quantiles: array[QuantileCount, float32]
      ## Min, the 9 fixed quantiles, max, stored f32, -Inf on masked rows.
    histKeys: seq[uint16]
      ## Sorted sparse keys of the soft histogram, reserved keys included.
    histCounts: seq[uint32]
      ## Soft counts scaled x2, each element splits toward two buckets
      ## on the dropped mantissa bit.
    histTotal: uint64
      ## Sum of the histogram counts, -Inf stays unbucketed.
    meanAbs: float64
      ## Mean absolute value over the finite elements, f64 accumulation.
    signedMean: float64
      ## Signed mean over the finite elements, f64 accumulation, the detector
      ## of a coherent per-element bias.
    tailProbability: float64
      ## Fraction of finite elements strictly above the tail threshold set
      ## four binades under the recorded max.
    tailEdge: int
      ## Recorded count of finite elements within TailEdgeSteps-scale ulp
      ## steps of the threshold, the count that bands the tail comparison.
    maxMagnitude: float64
      ## Largest finite magnitude of the recording, the drift band reference.

  UniformStatsFile = object
    schema: string
    source: string
    tensors: seq[tuple[name: string, record: StatsRecord]]


# #######################################################################
#                              Ulp bit math
# #######################################################################

proc orderedBits32(f: float32): uint32 {.inline.} =
  ## Monotone integer map of the f32 pattern, order preserved, sign-flip continuous.
  ## - the distance between two ordered words counts ulps
  let u = cast[uint32](f)
  if (u and 0x80000000'u32) != 0'u32: not u else: u or 0x80000000'u32

proc orderedBits16(b: uint16): uint32 {.inline.} =
  ## Monotone integer map of the bf16 pattern, order preserved,
  ## sign-flip continuous, kept inside 16-bit pattern space.
  ## - an f32-word promotion lifts negative values above +Inf
  let w = uint32(b)
  if (w and 0x8000'u32) != 0'u32: (not w) and 0xFFFF'u32
  else: w or 0x8000'u32

proc bf16BitsFromF32(f: float32): uint16 {.inline.} =
  ## Round-to-nearest-even f32 to bf16 bit pattern, matching torch `.to(bfloat16)` bit-exactly.
  let u = cast[uint32](f)
  uint16(((u + 0x7FFF'u32 + ((u shr 16) and 1'u32)) and 0xFFFF0000'u32) shr 16)

proc bf16ToF32(b: uint16): float32 {.inline.} =
  ## Exact bf16 promotion, the pattern shifted into the f32 mantissa top.
  cast[float32](uint32(b) shl 16)

proc fp16BitsFromF32(f: float32): uint16 =
  ## Recovers the exact fp16 pattern of an fp16-representable f32
  ## value by bit surgery, the promotion leaves the f32 mantissa low
  ## 13 bits zero, so the recovery rounds nothing at all.
  ##
  ## Contract:
  ## - normals take the f32 exponent and the mantissa top 10 bits,
  ##   subnormals scale back onto the 2^-24 step ladder
  ## - |f| past the fp16 overflow bound maps to the inf pattern
  ## - NaN never reaches the record paths, the non-finite scan raises first
  let u = cast[uint32](f)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let expField = (u shr 23) and 0xFF'u32
  let man = u and 0x7FFFFF'u32
  if expField == 0xFF'u32:
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

proc fp16ToF32(b: uint16): float32 =
  ## Exact fp16 promotion, the pattern value as f32, subnormals included.
  let sign = uint32(b and 0x8000'u16) shl 16
  let field = (b shr 10) and 0x1F'u16
  let man = b and 0x3FF'u16
  if field == 0'u16:
    let v = if man == 0'u16: 0.0'f32 else: (man.float64 * pow(2.0, -24.0)).float32
    return cast[float32](sign or cast[uint32](v))
  return cast[float32](sign or
    (uint32(field - 15'u16 + 127'u16) shl 23) or (uint32(man) shl 13))

proc distance32(a, b: float32): int64 {.inline.} =
  ## Distance of two f32 values in f32 ulps. Numerically equal values
  ## share distance 0, a +/-0 pair included.
  if a == b: 0'i64
  else:
    let d = orderedBits32(a).int64 - orderedBits32(b).int64
    if d < 0: -d else: d

proc distance16(a, b: float32): int64 {.inline.} =
  ## Distance of two bf16-representable values, counted in bf16 ulps.
  if a == b: 0'i64
  else:
    let d = orderedBits16(bf16BitsFromF32(a)).int64 -
            orderedBits16(bf16BitsFromF32(b)).int64
    if d < 0: -d else: d

proc distanceFp16(a, b: float32): int64 {.inline.} =
  ## Distance of two fp16-representable values, counted in fp16 ulps.
  if a == b: 0'i64
  else:
    let d = orderedBits16(fp16BitsFromF32(a)).int64 -
            orderedBits16(fp16BitsFromF32(b)).int64
    if d < 0: -d else: d

proc binadeIndex(v: float64): int =
  ## Binade index of a normal nonzero magnitude, with |v| in [2^e, 2^(e+1))
  ## reading binade e. Raises ValueError for zero and non-finite input.
  if v == 0.0 or classify(v) in {fcInf, fcNegInf, fcNaN}:
    raise newException(ValueError,
      "binade of a zero or non-finite magnitude has no reference: " & $v)
  var man: float64
  var exp: int
  man = frexp(abs(v), exp)
  exp - 1

proc binadeStep(g: UlpDatatype, binade: int): float64 =
  ## Width of one grid step for values in binade [2^e, 2^(e+1)).
  let bits = case g
    of ulpBf16: 7
    of ulpFp16: 10
    of ulpFp32: 23
  pow(2.0, (binade - bits).float64)

proc ulpStepAt(g: UlpDatatype, v: float64): float64 =
  ## Width of one ulp at magnitude |v|, precondition v names a normal nonzero value.
  ##
  ## Reference binade of the allowance derivations:
  ## - the recorded top logit for decisions
  ## - the recorded max magnitude for value records
  ##
  ## Example:
  ##   one bf16 ulp at 17.25 is 2^(4 - 7) = 0.125.
  binadeStep(g, binadeIndex(v))

proc ulpDatatypeName(g: UlpDatatype): string =
  ## Record-file name of one ulp datatype.
  case g
  of ulpBf16: "bf16"
  of ulpFp16: "fp16"
  of ulpFp32: "f32"


proc ulpDistance(g: UlpDatatype, a, b: float32): int64 =
  ## Distance of two values in ulps of the datatype `g`.
  ##
  ## Returns:
  ## the ulp distance, the measurement runs between the two values'
  ## own representable steps, an adjacent-step pair sits 1 apart
  ##
  ## Example:
  ##   bf16 12.5625 and 12.6875 sit 2 ulps apart, one bf16 ulp at 12.5625 is 0.0625.
  case g
  of ulpBf16: distance16(a, b)
  of ulpFp16: distanceFp16(a, b)
  of ulpFp32: distance32(a, b)


# #######################################################################
#                          Statistics primitives
# #######################################################################

proc totalVariation(a, b: tuple[keys: seq[uint16], counts: seq[uint32], total: uint64]): float64 =
  ## Total variation distance between two soft histograms, the summed
  ## absolute bucket difference over the combined mass.
  var ai, bi = 0
  var diff = 0'u64
  while ai < a.keys.len or bi < b.keys.len:
    if bi >= b.keys.len or (ai < a.keys.len and a.keys[ai] < b.keys[bi]):
      diff += uint64(a.counts[ai])
      inc ai
    elif ai >= a.keys.len or b.keys[bi] < a.keys[ai]:
      diff += uint64(b.counts[bi])
      inc bi
    else:
      let x = a.counts[ai].int64 - b.counts[bi].int64
      diff += uint64(if x < 0: -x else: x)
      inc ai
      inc bi
  let total = a.total + b.total
  if total == 0'u64: 0.0 else: diff.float64 / total.float64

proc truncatedKl(refLogits, obsLogits: seq[float32]): float64 =
  ## KL(p_ref || p_obs) over the shared top-32 set, both sides
  ## renormalized over that set.
  let n = refLogits.len
  if n != obsLogits.len:
    raise newException(ValueError, "truncated KL set length mismatch")
  var zr, zo = 0.0'f64
  for i in 0 ..< n:
    zr += exp(refLogits[i].float64)
    zo += exp(obsLogits[i].float64)
  for i in 0 ..< n:
    let pr = exp(refLogits[i].float64) / zr
    let po = exp(obsLogits[i].float64) / zo
    if pr > 0.0:
      result += pr * ln(pr / po)


# Record fingerprint builders
# ----------------------------------------------------------

proc recordMeans(vals: seq[float32]): tuple[n: int, meanAbs: float64, signedMean: float64, maxMagnitude: float64] =
  ## Finite-element means and the largest finite magnitude.
  ## NaN and +Inf raise, -Inf self-declares and stays outside.
  var nFinite = 0
  var absSum, sum = 0.0'f64
  var maxMagnitude = 0.0'f64
  for v in vals:
    let u = cast[uint32](v)
    if (u shr 23 and 0xFF'u32) == 0xFF'u32:
      if (u and 0x7FFFFF'u32) != 0'u32:
        raise newException(ValueError, "stats record input holds NaN")
      if (u and 0x80000000'u32) == 0'u32:
        raise newException(ValueError, "stats record input holds +Inf: ")
      continue
    inc nFinite
    absSum += abs(v.float64)
    sum += v.float64
    if abs(v.float64) > maxMagnitude:
      maxMagnitude = abs(v.float64)
  if nFinite == 0:
    raise newException(ValueError,
      "stats record of an all -Inf tensor has no finite mean")
  result.n = nFinite
  result.meanAbs = absSum / nFinite.float64
  result.signedMean = sum / nFinite.float64
  result.maxMagnitude = maxMagnitude

proc recordQuantiles(vals: seq[float32]): array[11, float32] =
  ## Min, the 9 fixed quantiles, max by exact order statistics,
  ## index = floor(p * (n - 1)) of the ascending sort, never interpolation.
  var sorted = vals
  sorted.sort()
  result[0] = sorted[0]
  result[^1] = sorted[^1]
  for pi in 0 ..< QuantileProbs.len:
    let idx = floor(QuantileProbs[pi] * float64(vals.len - 1)).int
    result[pi + 1] = sorted[idx]

proc recordHistogram(bits16: seq[uint16], name: string):
    tuple[keys: seq[uint16], counts: seq[uint32], total: uint64] =
  ## Soft binade histogram over the recorded values' bf16 patterns,
  ## integer math only, -Inf never buckets
  ## - name, the record context for the out-of-range binade error
  ## Returns:
  ## the sorted sparse histogram as keys, soft counts and the total
  var keyCounts = initCountTable[uint16]()
  for b in bits16:
    let sign = b and 0x8000'u16
    let expField = (b shr 7) and 0xFF'u16
    let mant = b and 0x7F'u16
    if expField == 0xFF'u16:
      continue
    if expField == 0'u16 and mant == 0'u16:
      keyCounts.inc(HistKeyZero, 2)
    elif expField == 0'u16:
      keyCounts.inc(HistKeySubnormal, 2)
    else:
      let e = expField.int - 127
      if e < HistBinadeMin or e > HistBinadeMax:
        raise newException(ValueError,
          "stats record binade " & $e & " outside [" & $HistBinadeMin &
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
        else:
          keyCounts.inc(key + 1, 1)
  var pairs: seq[(uint16, uint32)]
  for k, c in keyCounts:
    pairs.add (uint16(k), uint32(c))
  pairs.sort()
  for (k, c) in pairs:
    result.keys.add k
    result.counts.add c
    result.total += uint64(c)

proc recordTail(vals: seq[float32], ulpDatatype: UlpDatatype):
    tuple[probability: float64, edge: int] =
  ## Tail instrument over the finite elements
  ## - the threshold sits TailBinadesBelowMax binades under the recorded max
  ## - probability, the fraction of finite elements strictly above it
  ## - edge, the count within TailEdgeSteps ulp steps of the threshold,
  ##   the count that bands the tail comparison
  ## Returns:
  ## the tail probability and the recorded edge count
  let maxMagnitude = recordMeans(vals).maxMagnitude
  var tailCount, edgeCount = 0
  var probability = 0.0'f64
  var edge = 0
  if maxMagnitude > 0.0:
    let threshold = maxMagnitude / pow(2.0, TailBinadesBelowMax.float64)
    let edgeBand = TailEdgeSteps.float64 *
      binadeStep(ulpDatatype, binadeIndex(threshold))
    for v in vals:
      let a = abs(v.float64)
      if v != NegInf and a > threshold:
        inc tailCount
      if v != NegInf and abs(a - threshold) <= edgeBand:
        inc edgeCount
    probability = tailCount.float64 / vals.len.float64
  (probability: probability, edge: edge)


# #######################################################################
#                                Recorder
# #######################################################################

proc loadUniformStats(t: Tensor, name = ""): StatsRecord =
  ## Records the uniform stats of one contiguous tensor, one recorder,
  ## no flags, the histogram always taken.
  ##
  ## Contract, non-finite inputs:
  ## - NaN and +Inf raise
  ## - -Inf self-declares (masked attention scores), it stays unbucketed
  ##   and outside the means and the tail
  ## - -Inf compares -Inf equal, -Inf against a finite value is a mismatch
  ##
  ## - device tensors move to the host before the bit reads, device
  ##   memory holds no host storage, so the pointer casts need host bytes.
  let tc = t.contiguous().to(F.kCPU)
  let n = tc.numel()
  if n == 0:
    raise newException(ValueError, "stats record of an empty tensor " & name)
  result.name = name
  result.n = n

  var vals = newSeq[float32](n)
  var bits16 = newSeq[uint16](n)
  case tc.scalarType()
  of F.kBfloat16:
    result.ulpDatatype = ulpBf16
    let bv = tc.view(F.kInt16).contiguous()
    let raw = cast[ptr UncheckedArray[int16]](bv.data_ptr(int16))
    for i in 0 ..< n:
      let b = uint16(raw[i])
      bits16[i] = b
      vals[i] = bf16ToF32(b)
  of F.kFloat16:
    result.ulpDatatype = ulpFp16
    let hv = tc.view(F.kInt16).contiguous()
    let raw = cast[ptr UncheckedArray[int16]](hv.data_ptr(int16))
    for i in 0 ..< n:
      let f = fp16ToF32(uint16(raw[i]))
      vals[i] = f
      bits16[i] = bf16BitsFromF32(f)
  of F.kFloat32:
    result.ulpDatatype = ulpFp32
    let raw = cast[ptr UncheckedArray[float32]](tc.data_ptr(float32))
    for i in 0 ..< n:
      vals[i] = raw[i]
      bits16[i] = bf16BitsFromF32(raw[i])
  else:
    raise newException(ValueError,
      "stats record of unsupported dtype: " & $tc.scalarType() & " " & name)

  let means = recordMeans(vals)
  if means.n == 0:
    raise newException(ValueError,
      "stats record of an all -Inf tensor has no finite mean: " & name)
  result.meanAbs = means.meanAbs
  result.signedMean = means.signedMean
  result.maxMagnitude = means.maxMagnitude
  result.quantiles = recordQuantiles(vals)
  let hist = recordHistogram(bits16, name)
  result.histKeys = hist.keys
  result.histCounts = hist.counts
  result.histTotal = hist.total
  let tail = recordTail(vals, result.ulpDatatype)
  result.tailProbability = tail.probability
  result.tailEdge = tail.edge

# #######################################################################
#                          Allowance derivation
# #######################################################################
type
  BandSet = tuple[ulpBand: int, delta: float64, klBand: float64]
    ## Derived allowances of one comparison.
    ## - ulpBand, honest drift allowance in ulps at the reference magnitude
    ## - delta, the same allowance in absolute scale
    ## - klBand, the truncated-KL allowance, 0.5 x delta^2


proc deriveBands(ulpAllowance: int, top1: float64, depth = 1, datatype: UlpDatatype = ulpBf16, coarseAmplification = 1.0): BandSet =
  ## Returns the drift allowances for one comparison.
  ##
  ## - the reference magnitude = the recorded top-1 logit for decisions,
  ##   the recorded max magnitude for value records
  ## - the bands scale by the composed chain's depth, one stage = one
  ##   allowance of the error-model class
  ## - the depth scaling, the root-sum-square accumulation of independent
  ##   zero-mean per-stage reordering perturbations of `ulpAllowance` ulps,
  ##   the standard deviation scales as sqrt(depth) times the per-stage one
  ##
  ##   per-stage allowance x sqrt(depth) --> ulpBand --> delta --> klBand
  ##
  ## Worked example, a bf16 recording with recorded top-1 17.25:
  ## - one bf16 ulp at 17.25 = 0.125
  ## - the reduction class (4 ulps per stage) gives delta = 0.5
  ##   at depth 1 and delta = 2.75 at depth 28
  let amp = if coarseAmplification >= 1.0: coarseAmplification
    else:
      raise newException(ValueError,
        "coarse amplification below 1.0 widens no allowance, got: " &
        $coarseAmplification)
  result.ulpBand = ceil(ulpAllowance.float64 * sqrt(depth.float64) * amp).int
  result.delta = result.ulpBand.float64 * ulpStepAt(datatype, abs(top1))
  result.klBand = 0.5 * result.delta * result.delta


# #######################################################################
#                              Check engine
# #######################################################################

# Error model classes
# ----------------------------------------------------------

proc perStageAllowance(kind: RoundingErrorSourceKind): int =
  ## Returns the per-stage drift allowance of one error model in ulps, 2
  ## for the elementwise class and 4 for the reduction class, both measured
  ## same-kernel drift at the reference magnitude.
  case kind
  of kElementwise: 2
  of kReduction: 4

proc kindFloors(kind: RoundingErrorSourceKind, n: int): float64 =
  ## Histogram total-variation floor of one error model, the derivation
  ## reads the per-stage ulp drift allowance and the element count:
  ##   - a reassociation drift is a random zero-mean per-element perturbation
  ##   - the net TV is a fluctuation, TV ~ K * sqrt(mean|drift_ulps|)
  ##   - mean|drift_ulps| <= perStageAllowance(kind), and K ~ 7/sqrt(n)
  ## so the floor = 7.0 * sqrt(perStageAllowance(kind) / n),
  ## 7.0 the fluctuation-model calibration measured ~7-9 across recorded tensors.
  # TODO(magic-constant) 7.0 is calibrated across tensors, derive it
  # analytically when a first-principles model exists.
  const FluctuationK = 7.0
  if n <= 0: return 0.0
  FluctuationK * sqrt(perStageAllowance(kind).float64 / n.float64)

proc flatLogitsRow(logitsRow: Tensor): Tensor =
  ## Flat f32 CPU copy of one logits row, squeezed to [V].
  ## The copy lands on cpu, the raw pointer reads are host reads.
  var t = logitsRow.contiguous().to(F.kCPU)
  while t.dim > 1 and t.shape[0] == 1:
    t = t.squeeze(0)
  if t.dim != 1:
    raise newException(ValueError,
      "logits row is not flat, leading dims are not singleton: " &
        $t.shape)
  t.to(F.kFloat32).contiguous()


# #######################################################################
#                                Record IO
# #######################################################################

proc parseHexF32(s: string): float32 =
  ## Returns the f32 the hex string `s` encodes, "0x" plus 8 hex digits,
  ## case-insensitive, ValueError on any other form.
  if s.len != 10 or not s.startsWith("0x"):
    raise newException(ValueError, "bad f32 hex pattern: " & s)
  cast[float32](uint32(parseHexInt(s)))

proc parseHexF64(s: string): float64 =
  ## Returns the f64 the hex string `s` encodes, "0x" plus 16 hex digits,
  ## case-insensitive, ValueError on any other form.
  if s.len != 18 or not s.startsWith("0x"):
    raise newException(ValueError, "bad f64 hex pattern: " & s)
  var u: uint64
  for i in 2 ..< s.len:
    let d: uint64 = case s[i]
      of '0'..'9': uint64(s[i].ord - 48)
      of 'a'..'f': uint64(s[i].ord - 87)
      of 'A'..'F': uint64(s[i].ord - 55)
      else: raise newException(ValueError, "bad f64 hex pattern: " & s)
    u = (u shl 4) or d
  cast[float64](u)


# Stats frame IO
# ----------------------------------------------------------

proc readUniformStats(path: string): UniformStatsFile =
  ## Parses one uniform stats frame in the standardized form.
  ## The argument names either the frame path itself or the bare stem
  ## without the container suffix. Returns the validated records.
  let framePath =
    if path.endsWith(".json.zst"): path
    else: path & ".json.zst"
  let j = parseJson(readFile(framePath).zstdDecompress(string))
  result.schema = j{"schema"}.getStr()
  result.source = j{"source"}.getStr()
  if result.schema != UniformStatsSchema:
    raise newException(ValueError,
      "stats frame schema " & result.schema & " is not " &
      UniformStatsSchema)
  for name, tj in pairs(j{"tensors"}):
    var r = StatsRecord(name: name)
    r.n = tj{"n"}.getInt()
    case tj{"grid"}.getStr()
    of "bf16": r.ulpDatatype = ulpBf16
    of "fp16": r.ulpDatatype = ulpFp16
    of "f32": r.ulpDatatype = ulpFp32
    else:
      raise newException(ValueError,
        "bad stats grid key: " & tj{"grid"}.getStr())
    r.quantiles[0] = parseHexF32(tj{"quantiles"}{"min"}.getStr())
    r.quantiles[^1] = parseHexF32(tj{"quantiles"}{"max"}.getStr())
    for pi in 0 ..< r.quantiles.len - 2:
      r.quantiles[pi + 1] =
        parseHexF32(tj{"quantiles"}{QuantileNames[pi]}.getStr())
    let h = tj{"histogram"}
    r.histTotal = uint64(h{"total"}.getInt())
    var last = 0'u16
    for pair in h{"buckets"}.getStr().split(','):
      let kv = pair.split(':')
      let rawKey = parseBiggestInt(kv[0])
      if rawKey < 0 or rawKey > 0xFFFF:
        raise newException(ValueError,
          "stats frame bucket key outside uint16 range: " & $rawKey)
      let k = uint16(rawKey)
      if k <= last and r.histKeys.len > 0:
        raise newException(ValueError,
          "stats frame bucket keys out of sorted order: " & $k)
      last = k
      r.histKeys.add k
      r.histCounts.add uint32(parseBiggestInt(kv[1]))
    r.meanAbs = parseHexF64(tj{"mean_abs"}.getStr())
    r.signedMean = parseHexF64(tj{"signed_mean"}.getStr())
    r.tailProbability = parseHexF64(tj{"tail_probability"}.getStr())
    r.tailEdge = tj{"tail_edge"}.getInt()
    r.maxMagnitude = parseHexF64(tj{"max_magnitude"}.getStr())
    result.tensors.add (name, r)

proc loadUniformStats(path: string, tensorName: string): StatsRecord =
  ## Loads the uniform stats of one recorded tensor off its sidecar frame,
  ## the record acquisition inside the call.
  ##
  ## Args:
  ## the stats frame path or stem, the recorded tensor name
  for (n, rec) in readUniformStats(path).tensors:
    if n == tensorName:
      return rec
  raise newException(ValueError,
    "stats frame " & path & " carries no tensor " & tensorName)

proc dtypeGrid(node: JsonNode): UlpDatatype =
  ## Ulp datatype behind one decisions frame's `ulp_datatype` key,
  ## an absent key reads ulpBf16.
  if node.kind == JNull: return ulpBf16
  case node.getStr()
  of "bf16": ulpBf16
  of "fp16": ulpFp16
  of "f32": ulpFp32
  else:
    raise newException(ValueError,
      "bad decisions frame dtype key: " & node.getStr())


# Decisions frame IO
# ----------------------------------------------------------

proc loadArgmaxDecisions(path: string): seq[ArgmaxRecord] =
  ## Parses one argmax decision frame in the standardized form
  ##
  ## Args:
  ## - the frame path itself, or the bare stem without the container suffix
  ## - the standardized schema carries hex margins and hex top-32 bits
  ## - the allowances derive at check from the assert's kind argument, no
  ##   serialized allowance, the cap = the harness MaxTieFlips constant
  ## Returns:
  ## the records in frame order
  let framePath =
    if path.endsWith(".json.zst"): path
    else: path & ".json.zst"
  let j = parseJson(readFile(framePath).zstdDecompress(string))
  let schema = j{"schema"}.getStr()
  case schema
  of ArgmaxDecisionsSchema:
    let grid = dtypeGrid(j{"ulp_datatype"})
    for sj in items(j{"steps"}):
      var r = ArgmaxRecord(ulpDatatype: grid)
      r.argmaxId = sj{"argmax_id"}.getInt()
      r.margin = parseHexF64(sj{"margin"}.getStr())
      r.tailProbability = parseHexF64(sj{"tail_probability"}.getStr())
      for idj in items(sj{"top_k"}):
        r.topK.add idj.getInt()
      let bits = sj{"top_k_logits"}.getStr()
      for chunk in bits.split(' '):
        r.topKLogits.add parseHexF32(chunk)
      result.add r
  else:
    raise newException(ValueError,
      "decisions frame schema " & schema & " is not " &
      ArgmaxDecisionsSchema)

proc observedTailProbability(row: Tensor, topKIds: seq[int]): float64 =
  ## Softmax mass beyond the recorded top-32 set of one flat f32 logits row.
  let probs = F.softmax(row, dim = -1)
  let idTensor = F.toTensor(topKIds.mapIt(it.int64))
  let kept = probs.index_select(0, idTensor).to(F.kFloat32)
  1.0 - kept.sum().item(float64)


# Instrument check cores
# ----------------------------------------------------------

proc harnessStats(actual: Tensor, record: StatsRecord, kind: RoundingErrorSourceKind, depth = 1, msg = "") =
  ## Checks the uniform stats of `actual` against one record,
  ## under the allowances derived from perStageAllowance(kind), maxMagnitude,
  ## depth and the record's ulp datatype.
  ##
  ## Contract and comparison plan:
  ## - order statistics and the f64 means compare against the absolute delta
  ##   at every depth, the allowance anchored at the recorded max magnitude,
  ##   the informative precision set by the tensor's scale
  ## - depth scales the band through deriveBands (one allowance per stage),
  ##   it does not switch the unit of measure, at depth 1 the histogram
  ##   total variation and the tail probability carry their own floors
  ## - a zero-magnitude reference scales no allowance, the quantiles
  ##   must equal the recorded zero pattern
  let ctx = (if msg.len > 0: ": " & msg else: "")
  let fp = loadUniformStats(actual,
    (if msg.len > 0: msg else: record.name))
  if fp.n != record.n:
    raise newException(HarnessCheckError,
      "stats element count " & $fp.n & " != recorded " & $record.n & ctx)
  if fp.ulpDatatype != record.ulpDatatype:
    raise newException(HarnessCheckError,
      "stats ulp datatype mismatch: computed " &
      ulpDatatypeName(fp.ulpDatatype) & " vs recorded " &
      ulpDatatypeName(record.ulpDatatype) & ctx)

  proc reject(what: string) =
    raise newException(HarnessCheckError, what & ctx)

  if record.maxMagnitude == 0.0:
    ## All-zero reference, no allowance scales from it.
    for i in 0 ..< record.quantiles.len:
      if cast[uint32](fp.quantiles[i]) != cast[uint32](record.quantiles[i]):
        reject("stats quantile " & $i & " of an all-zero record must " &
          "equal the recorded zero pattern")
    return

  let bands = deriveBands(perStageAllowance(kind), record.maxMagnitude, depth,
    record.ulpDatatype)
  let histFloor = kindFloors(kind, record.n)

  proc quantileFault(i: int): bool =
    ## -Inf self-declares, -Inf equal to -Inf passes and -Inf against
    ## a finite value is a mismatch, NaN and +Inf never reach here.
    let g = classify(fp.quantiles[i])
    let w = classify(record.quantiles[i])
    if g == fcNegInf and w == fcNegInf: return false
    if g in {fcNegInf, fcInf, fcNaN} or w in {fcNegInf, fcInf, fcNaN}:
      reject("stats quantile " & $i & " non-finite mismatch: got " &
        $fp.quantiles[i] & " want " & $record.quantiles[i])
    return true

  var worst = 0.0
  for i in 0 ..< record.quantiles.len:
    if not quantileFault(i): continue
    let d = abs(fp.quantiles[i].float64 - record.quantiles[i].float64)
    if d > worst:
      worst = d
    if d > bands.delta:
      reject("stats quantile " & $i & " drift " & $d &
        " exceeds band " & $bands.delta)


  if abs(fp.meanAbs - record.meanAbs) > bands.delta:
    reject("stats meanAbs drift " &
      formatBiggestFloat(fp.meanAbs - record.meanAbs, ffScientific, 3) &
      " exceeds band " & $bands.delta)
  if abs(fp.signedMean - record.signedMean) > bands.delta:
    reject("stats signedMean drift " &
      formatBiggestFloat(fp.signedMean - record.signedMean, ffScientific, 3) &
      " exceeds band " & $bands.delta)

  if depth == 1:
    block:
      let l1 = totalVariation(
        (keys: fp.histKeys, counts: fp.histCounts, total: fp.histTotal),
        (keys: record.histKeys, counts: record.histCounts,
          total: record.histTotal))
      if l1 > histFloor:
        reject("stats histogram total variation " & $l1 & " exceeds floor " &
          $histFloor)
    let tailBand = record.tailEdge.float64 / record.n.float64
    if abs(fp.tailProbability - record.tailProbability) > tailBand:
      reject("stats tail probability " & $fp.tailProbability &
        " vs recorded " & $record.tailProbability & " outside band " &
        $tailBand)


proc checkArgmaxRow(actual: Tensor, record: ArgmaxRecord, flipCount: var int, msg = "", depth = 1) =
  ## Checks one logits row against the recorded decision.
  ## The instrument core of assertArgMax.
  ##
  ##   computed row -> argmax, top-32 logits, truncated KL, tail
  ##   recorded record -----------------------------------------> allowances
  ##
  ## Every step runs the top-32 instruments, flips included:
  ## - the pick is the decoded token, divergence raises unless tie-eligible
  ## - the truncated KL over the 32 recorded ids stays at most
  ##   0.5 x delta^2, the distribution allowance, the individual top-32
  ##   logits carry no per-id value band
  ## - the tail probability differs from the recorded value by at most
  ##   min(0.3, exp(delta) - 1) x max(recorded tail, 1e-3) + 1e-4
  ##
  ## Allowance form:
  ## - derived at check from the error-model class (perStageAllowance),
  ##   never a serialized constant
  ## - delta = ceil(per-stage ulps x sqrt(depth)) x the datatype's
  ##   one-ulp step at |recorded top-1|, the sqrt depth scaling gives
  ##   the root-sum-square accumulation of per-stage reordering errors
  ##
  ## Worked example, a bf16 recording with recorded top-1 17.25, one
  ## bf16 ulp at 17.25 = 0.125, the reduction class (4 ulps per stage)
  ## gives delta = 0.5 at depth 1 and delta = 2.75 at depth 28.
  ##
  ## Pick divergence:
  ##
  ##   margin <= 1 ulp of the record's datatype at the recorded top-1
  ##     --> pick logit within max(delta, 0.05) of the top-1 --> flip,
  ##     flipCount increments, the MaxTieFlips constant caps it
  ##   anything else --> a real divergence, raised
  ##
  ## The observed row reads the values at the recorded ids, no re-ranking,
  ## the rank-32/33 boundary stays covered by the set certificate,
  ## a rank-32/33 swap the certificate accepts sits within the KL band.
  ## - kind, the error model class the allowances derive from at check
  ## - flipCount, the caller-owned chain-wide counter, the MaxTieFlips
  ##   constant caps it, pick flips increment it
  ## - depth, the composed op count feeding the allowance, a head
  ##   decision over a chain carries the layer count, a single
  ##   step carries 1
  let ctx = (if msg.len > 0: ": " & msg else: "")
  if record.topKLogits.len != record.topK.len:
    raise newException(ValueError,
      "argmax record top-32 ids and logits misaligned" & ctx)
  let row = flatLogitsRow(actual)
  let n = row.numel()
  let raw = cast[ptr UncheckedArray[float32]](row.data_ptr(float32))
  for vi in 0 ..< n:
    if classify(raw[vi]) in {fcNaN, fcInf, fcNegInf}:
      raise newException(HarnessCheckError,
        "argmax logits row carries a non-finite value, no instrument reads NaN")
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

  let top1Rec = record.topKLogits[0].float64
  let bands = deriveBands(perStageAllowance(record.kind), top1Rec, depth,
    record.ulpDatatype)

  # Every step runs the top-32 instruments, flips included, the 32-wide
  # (record width) truncated KL check does not skip flips.
  var obs = newSeq[float32](record.topKLogits.len)
  for i, id in record.topK:
    if id < 0 or id >= n:
      raise newException(HarnessCheckError,
        "argmax record top-32 id " & $id & " outside the row" & ctx)
    obs[i] = raw[id]
  let kl = truncatedKl(record.topKLogits, obs)
  if kl > bands.klBand:
    raise newException(HarnessCheckError,
      "argmax truncated KL " & $kl & " exceeds band " & $bands.klBand &
      " (delta " & $bands.delta & ")" & ctx)
  let tail = observedTailProbability(row, record.topK)
  let tailLimit = min(0.3, exp(bands.delta) - 1.0) *
    max(record.tailProbability, 1e-3) + 1e-4
  if abs(tail - record.tailProbability) > tailLimit:
    raise newException(HarnessCheckError,
      "argmax tail probability " & $tail & " vs recorded " &
      $record.tailProbability & " outside band " & $tailLimit & ctx)

  # Divergence, tie flip or real bug. Tie eligibility reads the ulp
  # grid of the record's datatype, one bf16 step is 0.125 at logit 16
  # and 0.25 at logit 32.
  if obsArgmax != record.argmaxId:
    if record.margin <= ulpStepAt(record.ulpDatatype, abs(top1Rec)):
      let tieBand = max(bands.delta, 0.05)
      let pickLogit = raw[obsArgmax].float64
      if abs(pickLogit - top1Rec) <= tieBand:
        inc flipCount
        if flipCount > MaxTieFlips:
          raise newException(HarnessCheckError,
            "tie-flip cap " & $MaxTieFlips & " exceeded (pick " &
            $obsArgmax & " vs recorded " & $record.argmaxId & ", margin " &
            $record.margin & ")" & ctx)
        return
      raise newException(HarnessCheckError,
        "divergence is not a tie: pick " & $obsArgmax & " logit " &
        $pickLogit & " vs recorded top-1 " & $top1Rec & " outside band " &
        $tieBand & " (recorded margin " & $record.margin & ")" & ctx)
    raise newException(HarnessCheckError,
      "argmax divergence: pick " & $obsArgmax & " (logit " & $raw[obsArgmax] &
      ", margin " & $(obsTop1 - obsTop2) & ") vs recorded " &
      $record.argmaxId & " (logit " & $top1Rec & ", margin " &
      $record.margin & ")" & ctx)


# Public assert surface
# ----------------------------------------------------------

proc assertStats*(actual: Tensor, statsPath: string, tensorName: string, kind: RoundingErrorSourceKind, depth = 1, msg = "") =
  ## Asserts one computed tensor against one recorded sidecar entry,
  ## the record acquisition inside the call.
  ##
  ## Args:
  ## the computed tensor, the stats frame path or stem, the recorded
  ## tensor name, the error kind, the composed depth
  ##
  ## In plain terms the check compares fingerprints, never millions
  ## of numbers one by one
  ## - the fingerprint, the min and max, nine percentiles between,
  ##   a histogram of how many values land in each size bucket,
  ##   two f64 averages, the far-tail probability
  ## - parallel math adds numbers in a different but equally valid order
  ##   than the reference, the last bits differ honestly, the allowances
  ##   say how much honest difference looks like
  ## - the exact-class cases live elsewhere, the EXL3-00 codec verifies
  ##   its decoded payload bit for bit, a single drifting element away
  ##   from the checkpoints stays invisible
  harnessStats(actual, loadUniformStats(statsPath, tensorName),
    kind, depth, msg)

proc assertArgMax*(actual: Tensor, decisionsPath: string, step: int,
    kind: RoundingErrorSourceKind, flipCount: var int, msg = "", depth = 1) =
  ## Asserts one computed logits row against the recorded decision at one
  ## frame position, the record acquisition inside the call.
  ##
  ## Args:
  ## - the computed logits row, the decisions frame path or stem,
  ##   the zero-based frame position
  ## - the error kind
  ## - the caller-owned chain-wide flip counter, the composed depth
  let records = loadArgmaxDecisions(decisionsPath)
  if step < 0 or step >= records.len:
    raise newException(ValueError,
      "decisions frame " & decisionsPath & " position " & $step &
      " outside 0.." & $(records.len - 1))
  var record = records[step]
  record.kind = kind
  checkArgmaxRow(actual, record, flipCount, msg, depth)

