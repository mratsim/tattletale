# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[algorithm, json, math, os, strutils, tables]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/positron/src/kernels/ceramic/linear_quant/hadamard_rotate
import workspace/safetensors/src/safetensors
import workspace/zstd/zstd_highlevel
import ../properties/properties

## Per 128-block of the input axis, `hadamardRotate128Ref` computes
## `dst = fp16round(FWHT(src * preScale) * norm) * postScale`, matching the exllamav3 op `ext.had_r_128`.
##
## Dataflow per 128-block (all arrows elementwise):
##
##   src line --> fp16 preScale mul (one RNE round) --> f64 widen
##   --> FWHT-128 (f64, exact) --> f64->fp32 RNE --> fp32 * norm
##   --> fp16 RNE round --> fp16 postScale mul (one RNE round) --> dst line
##
## Exactness of the float64 butterflies:
##
## - an fp16 value is an integer multiple of 2^-24, and so is any sum of 128 of them
## - the integer magnitude of such a sum stays below 2^47, 47 significand bits
## - the float64 significand spans 53 bits, every such sum fits it exactly
##
## An fp32 butterfly kernel does not share the guarantee:
##
## - the 2-term sum 65504 + 2^-24 already needs 40 significand bits, fp32 spans 24 and rounds
## - an fp32 kernel matches this reference bit for bit only where every partial sum stays
##   inside the 24-bit fp32 significand, an addend exponent spread of up to 13 binades,
##   the domain the validation inputs must respect
##
## Off-contract inputs are undefined behavior and the callers check before launching.
## The contract needs an axis size that is a positive multiple of 128 and a positive size.

import workspace/crucible

func refFwht128*(x: var array[128, float64]) =
  ## Unnormalized 7-stage fast Walsh-Hadamard transform, in place over x.
  ## Stage s combines positions j and j + 2^(s-1) into a+b / a-b.
  var step = 1
  while step < 128:
    var base = 0
    while base < 128:
      for j in 0 ..< step:
        let
          a = x[base + j]
          b = x[base + j + step]
        x[base + j] = a + b
        x[base + j + step] = a - b
      base += 2 * step
    step *= 2

func fp16mul(a, b: float16): float16 =
  ## Exact fp32 products: the fp32 product of two fp16 values is exact, one RNE round back to fp16 is the native fp16 multiply.
  to(to(a, float32) * to(b, float32), float16)

proc hadamardRotate128Ref*(dst, src, preScale, postScale: ptr UncheckedArray[float16],
    M, C: int32, norm: float32,
    withPreScale, withPostScale: static bool) =
  ## Reference of the 128-block Hadamard rotate over fp16 storage, rounding per the module doc's contract.
  ##
  ## Contract:
  ##
  ## - dst, src: two (M, C) tensors of the same shape, fully contiguous, element (m, c) at offset m * C + c
  ## - preScale, postScale: the C-element scale vectors, one value per axis position,
  ##   the same vector serves every tensor line
  ## - a scale slot whose static switch is false is never dereferenced, any valid pointer fills it
  ## - norm: the post-transform normalization factor, 1/sqrt(128) for EXL3
  var row: array[128, float64]
  for i in 0'i32 ..< M:
    for blk in 0'i32 ..< C div 128:
      for k in 0'i32 ..< 128:
        var v = src[i * C + blk * 128 + k]
        when withPreScale:
          v = fp16mul(v, preScale[blk * 128 + k])
        row[k] = v.to(float32).float64
      refFwht128(row)
      for k in 0'i32 ..< 128:
        var v = (row[k].float32 * norm).to(float16)
        when withPostScale:
          v = fp16mul(v, postScale[blk * 128 + k])
        dst[i * C + blk * 128 + k] = v

export properties.fp32ToFp16, properties.fp16ToFp32

const
  FixtureDir* = currentSourcePath().parentDir().parentDir().parentDir().parentDir() /
    "transformers" / "tests" / "fixtures" / "exl3-00-codec" / "Qwen3-0.6B-EXL3-5bpw"
  InvSqrt128* = 0.088388347648'f32

# ═══════════════════════════════════════════════════════════════════════
#  Fixture loading
# ═══════════════════════════════════════════════════════════════════════

proc loadFixtureF16*(path, name: string, elems: int): seq[uint16] =
  var st = Safetensor.open(path)
  let view = st.getMmapView(name)
  doAssert view.size == elems * sizeof(uint16)
  result = newSeq[uint16](elems)
  copyMem(addr result[0], view.data, view.size)

# ═══════════════════════════════════════════════════════════════════════
#  Uniform-stats fingerprint
# ═══════════════════════════════════════════════════════════════════════

const
  HistKeyZero = uint16(0xFFFE)
  HistKeySubnormal = uint16(0xFFFF)
  HistBinadeMin = -64
  HistBinadeMax = 63
  TailBinadesBelowMax = 4
  TailEdgeSteps = 4
  QuantileProbs = [0.01'f64, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

type Fingerprint* = object
  n*: int
  quantiles*: array[11, uint32]
  histKeys*: seq[uint16]
  histCounts*: seq[uint32]
  histTotal*: uint64
  meanAbs*, signedMean*, maxMagnitude*, tailProbability*: uint64
  tailEdge*: int

func bf16BitsFromF32(f: float32): uint16 =
  let u = cast[uint32](f)
  uint16(((u + 0x7FFF'u32 + ((u shr 16) and 1'u32)) and 0xFFFF0000'u32) shr 16)

func binadeIndex(v: float64): int =
  var man: float64
  var e: int
  man = frexp(abs(v), e)
  e - 1

func binadeStepFp16(binade: int): float64 =
  pow(2.0, (binade - 10).float64)

proc statsFingerprint*(bits: openArray[uint16]): Fingerprint =
  doAssert bits.len > 0
  let n = bits.len
  var vals = newSeq[float32](n)
  var bits16 = newSeq[uint16](n)
  for i in 0 ..< n:
    let f = fp16ToFp32(bits[i])
    vals[i] = f
    bits16[i] = bf16BitsFromF32(f)
  result.n = n
  var nFinite = 0
  var absSum, sum = 0.0'f64
  var maxMagnitude = 0.0'f64
  for v in vals:
    let u = cast[uint32](v)
    if (u shr 23 and 0xFF'u32) == 0xFF'u32:
      doAssert (u and 0x7FFFFF'u32) == 0'u32
      doAssert (u and 0x80000000'u32) != 0'u32
      continue
    inc nFinite
    absSum += abs(v.float64)
    sum += v.float64
    if abs(v.float64) > maxMagnitude:
      maxMagnitude = abs(v.float64)
  doAssert nFinite > 0
  result.meanAbs = cast[uint64](absSum / nFinite.float64)
  result.signedMean = cast[uint64](sum / nFinite.float64)
  result.maxMagnitude = cast[uint64](maxMagnitude)
  var sorted = vals
  sorted.sort()
  result.quantiles[0] = cast[uint32](sorted[0])
  result.quantiles[^1] = cast[uint32](sorted[^1])
  for pi in 0 ..< QuantileProbs.len:
    let idx = floor(QuantileProbs[pi] * float64(n - 1)).int
    result.quantiles[pi + 1] = cast[uint32](sorted[idx])
  var keyCounts: array[65536, uint32]
  for b in bits16:
    let sign = b and 0x8000'u16
    let expField = (b shr 7) and 0xFF'u16
    let mant = b and 0x7F'u16
    if expField == 0xFF'u16:
      continue
    if expField == 0'u16 and mant == 0'u16:
      keyCounts[HistKeyZero] += 2
    elif expField == 0'u16:
      keyCounts[HistKeySubnormal] += 2
    else:
      let e = expField.int - 127
      doAssert e >= HistBinadeMin and e <= HistBinadeMax
      let mtop = mant shr 1
      let low = mant and 1'u16
      let key = (sign shr 2'u16) or (uint16(e - HistBinadeMin) shl 6) or mtop
      keyCounts[key] += uint32(2 - low.int)
      if low != 0'u16:
        if mtop == 63'u16:
          if e + 1 <= HistBinadeMax:
            let nkey = (sign shr 2'u16) or (uint16(e + 1 - HistBinadeMin) shl 6)
            keyCounts[nkey] += 1
        else:
          keyCounts[key + 1] += 1
  var entries: seq[(uint16, uint32)]
  for k in 0 ..< 65536:
    if keyCounts[k] != 0:
      entries.add (uint16(k), keyCounts[k])
  entries.sort()
  for (k, c) in entries:
    result.histKeys.add k
    result.histCounts.add c
    result.histTotal += uint64(c)
  let threshold = maxMagnitude / pow(2.0, TailBinadesBelowMax.float64)
  let edgeBand = TailEdgeSteps.float64 * binadeStepFp16(binadeIndex(threshold))
  var tailCount, edgeCount = 0
  for v in vals:
    let a = abs(v.float64)
    if a > threshold:
      inc tailCount
    if abs(a - threshold) <= edgeBand:
      inc edgeCount
  result.tailProbability = cast[uint64](tailCount.float64 / n.float64)
  result.tailEdge = edgeCount

# ═══════════════════════════════════════════════════════════════════════
#  The recorded frames
# ═══════════════════════════════════════════════════════════════════════

type FrameRecord* = object
  n*: int
  quantiles*: array[11, uint32]
  histKeys*: seq[uint16]
  histCounts*: seq[uint32]
  histTotal*: uint64
  meanAbs*, signedMean*, maxMagnitude*, tailProbability*: uint64
  tailEdge*: int

func hexU32(j: JsonNode): uint32 =
  uint32(parseHexInt(j.getStr()))

func hexU64(j: JsonNode): uint64 =
  uint64(parseHexInt(j.getStr()))

proc loadUniformStatsFrame*(path: string): Table[string, FrameRecord] =
  ## Reads one `ttt-tf-004-uniform-stats` frame, one record per tensor name.
  doAssert fileExists(path), "missing stats frame " & path
  let j = parseJson(readFile(path).zstdDecompress(string))
  doAssert j{"schema"}.getStr() == "ttt-tf-004-uniform-stats"
  for name, rec in j{"tensors"}.pairs():
    var r: FrameRecord
    r.n = rec{"n"}.getInt()
    r.quantiles[0] = hexU32(rec{"quantiles"}{"min"})
    r.quantiles[^1] = hexU32(rec{"quantiles"}{"max"})
    for pi, qn in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
      r.quantiles[pi + 1] = hexU32(rec{"quantiles"}{qn})
    for pair in rec{"histogram"}{"buckets"}.getStr().split(','):
      let kv = pair.split(':')
      r.histKeys.add uint16(kv[0].parseInt())
      r.histCounts.add uint32(kv[1].parseInt())
    r.histTotal = uint64(rec{"histogram"}{"total"}.getInt())
    r.meanAbs = hexU64(rec{"mean_abs"})
    r.signedMean = hexU64(rec{"signed_mean"})
    r.maxMagnitude = hexU64(rec{"max_magnitude"})
    r.tailProbability = hexU64(rec{"tail_probability"})
    r.tailEdge = rec{"tail_edge"}.getInt()
    result[name] = r

proc assertFingerprintMatches*(got: Fingerprint, expect: FrameRecord, msg: string) =
  template check(field: untyped, cond: bool) =
    doAssert cond, msg & ": " & astToStr(field) & " mismatch"
  check n, got.n == expect.n
  check quantiles, got.quantiles == expect.quantiles
  check histogram, got.histKeys == expect.histKeys and
    got.histCounts == expect.histCounts and got.histTotal == expect.histTotal
  check mean_abs, got.meanAbs == expect.meanAbs
  check signed_mean, got.signedMean == expect.signedMean
  check max_magnitude, got.maxMagnitude == expect.maxMagnitude
  check tail_probability, got.tailProbability == expect.tailProbability
  check tail_edge, got.tailEdge == expect.tailEdge

# ═══════════════════════════════════════════════════════════════════════
#  Device entry
# ═══════════════════════════════════════════════════════════════════════
type ScaleVariant* = enum
  varNone
  varPre
  varPost
  varBoth

func variantFrameName*(v: ScaleVariant): string =
  case v
  of varNone: "output_none"
  of varPre: "output_pre"
  of varPost: "output_post"
  of varBoth: "output_both"

proc hadamardCombo*(dst, src, preScale, postScale: distinct TensorView, norm: float32, withPreScale, withPostScale: static bool) =
  # Requires export: the launch machinery re-emits the callee name into the importing module, an imported-only symbol does not resolve there.
  # The raw-pointer kernel reads the views' device pointers, M and C come from the dst shape.
  hadamard_rotate_128_fwd(
    cast[ptr UncheckedArray[float16]](dst.data), cast[ptr UncheckedArray[float16]](src.data),
    cast[ptr UncheckedArray[float16]](preScale.data), cast[ptr UncheckedArray[float16]](postScale.data),
    int32(dst.shape[0]), int32(dst.shape[1]), norm, withPreScale, withPostScale)

proc launchHadamardGrid*[E: HwEngine](engine: E, v: ScaleVariant,
    src, preScale, postScale: seq[uint16], M, C: int, norm: float32,
    gridLines, gridBlocks: int): seq[uint16] =
  ## One launch of the requested scale variant with the dst bits read back.
  ##
  ## Args:
  ## - gridLines x gridBlocks: the launch threadgroups, any size is valid through
  ##   the grid-stride loop
  let
    srcMem = engine.allocDevice(float16, M * C)
    dstMem = engine.allocDevice(float16, M * C)
    preMem = engine.allocDevice(float16, C)
    postMem = engine.allocDevice(float16, C)
  engine.copyToDeviceFromHost(srcMem, cast[ptr float16](addr src[0]), M * C)
  engine.copyToDeviceFromHost(preMem, cast[ptr float16](addr preScale[0]), C)
  engine.copyToDeviceFromHost(postMem, cast[ptr float16](addr postScale[0]), C)
  let
    srcLay = make_layout((M, C), (C, 1))
    lineLay = make_layout((C,), (1,))
    # DeviceMem handles are move-only, each BridgeTensor value is built inline
    # over one handle
    srcView = BridgeTensor[typeof(srcMem), typeof(srcLay.shape), typeof(srcLay.stride)](
      data: srcMem, layout: srcLay)
    dstView = BridgeTensor[typeof(dstMem), typeof(srcLay.shape), typeof(srcLay.stride)](
      data: dstMem, layout: srcLay)
    preView = BridgeTensor[typeof(preMem), typeof(lineLay.shape), typeof(lineLay.stride)](
      data: preMem, layout: lineLay)
    postView = BridgeTensor[typeof(postMem), typeof(lineLay.shape), typeof(lineLay.stride)](
      data: postMem, layout: lineLay)
    normVal = norm
  template launchVariant(withPreScale, withPostScale: static bool) =
    engine.launchDeviceFn hadamardCombo(
      dstView, srcView, preView, postView, normVal, withPreScale, withPostScale) do:
      grid = (gridLines, gridBlocks, 1)
      blk = (32, 1)
  case v
  of varNone: launchVariant(false, false)
  of varPre: launchVariant(true, false)
  of varPost: launchVariant(false, true)
  of varBoth: launchVariant(true, true)
  result = newSeq[uint16](M * C)
  engine.copyToHostFromDevice(cast[ptr float16](addr result[0]), dstView.data, M * C)

proc launchHadamard*[E: HwEngine](engine: E, v: ScaleVariant, src, preScale, postScale: seq[uint16], M, C: int, norm: float32): seq[uint16] =
  ## One launch at the natural grid, one threadgroup per tensor line and 128-block.
  launchHadamardGrid(engine, v, src, preScale, postScale, M, C, norm, M, C div 128)

proc referenceHadamard*(src, preScale, postScale: seq[uint16], M, C: int, norm: float32, withPreScale, withPostScale: static bool): seq[uint16] =
  result = newSeq[uint16](M * C)
  hadamardRotate128Ref(cast[ptr UncheckedArray[float16]](addr result[0]),
      cast[ptr UncheckedArray[float16]](unsafeAddr src[0]),
      cast[ptr UncheckedArray[float16]](unsafeAddr preScale[0]),
      cast[ptr UncheckedArray[float16]](unsafeAddr postScale[0]),
      int32(M), int32(C), norm, withPreScale, withPostScale)

proc referenceVariant*(src, preScale, postScale: seq[uint16], M, C: int, norm: float32, v: ScaleVariant): seq[uint16] =
  case v
  of varNone: referenceHadamard(src, preScale, postScale, M, C, norm, false, false)
  of varPre: referenceHadamard(src, preScale, postScale, M, C, norm, true, false)
  of varPost: referenceHadamard(src, preScale, postScale, M, C, norm, false, true)
  of varBoth: referenceHadamard(src, preScale, postScale, M, C, norm, true, true)

export int_tuples, layout_algebra
export ScaleVariant, variantFrameName, hadamardCombo, launchHadamard, launchHadamardGrid,
       referenceVariant
