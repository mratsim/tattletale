# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Seeded support surface for the positron property suites, the randomness source,
## the element-dtype bit surgery, the ulp allowance vocabulary and the seeded input
## generators per kernel family.
##
## One explicitly seeded `PropRng` is the single randomness source of the suites,
## every randomized input of every property suite deriving from it:
##
## - the generator is xorshift64* over this module's own bit operations, deterministic
##   across Nim compiler versions, unlike the std/random module
## - the bit surgery is host-side round-to-nearest-even over the raw patterns, the same
##   rounding that the kernels' element-dtype stores apply
## - the ulp block mirrors the transformers harness's allowance vocabulary, see
##   workspace/transformers/tests/harness/harness.nim, with no import involved

import std/[math, strformat]

# ─── Seeded randomness ───────────────────────────────────────────────

type PropRng* = object
  ## xorshift64* stream state. Must stay nonzero, the zero state is
  ## the fixed point of the recurrence.
  state: uint64

proc initPropRng*(seed: uint64): PropRng =
  ## Starts a deterministic stream from `seed`, mapping a zero seed
  ## to a fixed nonzero state so callers passing 0 get a valid stream.
  let state = if seed == 0: 0x9E3779B97F4A7C15'u64 else: seed
  PropRng(state: state)

proc nextU64*(rng: var PropRng): uint64 =
  ## Returns the next 64-bit value of the stream, the same seed always
  ## yielding the same sequence of values.
  var x = rng.state
  x = x xor (x shr 12)
  x = x xor (x shl 25)
  x = x xor (x shr 27)
  rng.state = x
  result = x * 0x2545F4914F6CDD1D'u64

proc nextF32*(rng: var PropRng; lo, hi: float32): float32 =
  ## Returns a uniform float32 in [lo, hi), using the 24 high bits of the next stream
  ## value at 2⁻²⁴ granularity over the range.
  doAssert hi > lo, "empty range [" & $lo & ", " & $hi & ")"
  let unit = float32(rng.nextU64() shr 40) * (1.0'f32 / 16777216.0'f32)
  lo + unit * (hi - lo)

proc nextInt*(rng: var PropRng; lo, hi: int): int =
  ## Returns a uniform int in [lo, hi), using the high 32 bits
  ## of the next stream value.
  ## - the span `hi - lo` must stay at or below 2³², the width
  ##   addressable by the sampled high bits
  ## - a wider span leaves the upper part of the range unreachable,
  ##   a caller bug the assert turns loud
  doAssert hi > lo, "empty range [" & $lo & ", " & $hi & ")"
  let span = uint64(hi - lo)
  doAssert span <= 0xFFFF_FFFF'u64,
    "nextInt span " & $span & " exceeds the 32-bit width of the sampled high bits"
  lo + int((rng.nextU64() shr 32) mod span)

# ─── Element-dtype bit surgery ───────────────────────────────────────

type ScalarKind* = enum
  ## Element dtype of one kernel binding, the members the property
  ## suites and the ceramic suites instantiate.
  kFloat16
  kBfloat16

proc f32ToBf16*(x: float32): uint16 =
  ## Returns the bf16 round-to-nearest-even bit pattern of one fp32 value,
  ## zero and Inf passing through and NaN short-circuiting to the canonical
  ## quiet pattern, sign kept.
  ##
  ## NaN never goes through the round-to-even increment, whose carry can
  ## move a small odd payload across an Inf/NaN boundary.
  let u = cast[uint32](x)
  if (u and 0x7F800000'u32) == 0x7F800000'u32 and (u and 0x007FFFFF'u32) != 0:
    return if (u shr 31) == 1: 0xFFC0'u16 else: 0x7FC0'u16
  let lsb = (u shr 16) and 1'u32
  uint16((u + 0x7FFF'u32 + lsb) shr 16)

proc bf16ToF32*(h: uint16): float32 =
  ## Returns the exact fp32 widening of a bf16 bit pattern.
  cast[float32](uint32(h) shl 16)

proc fp32ToFp16*(x: float32): uint16 =
  ## Returns the fp16 round-to-nearest-even bit pattern of one fp32 value,
  ## subnormals on the 2⁻²⁴ grid, underflow below the halfway point to zero,
  ## overflow saturating to inf and NaN quieted to the canonical pattern.
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF'u32)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:                           # Inf / NaN
    if mant == 0: return sign or 0x7C00'u16
    return sign or 0x7C00'u16 or 0x0200'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:                             # overflow -> Inf
    return sign or 0x7C00'u16
  if bexp <= 0:                              # subnormal or zero
    if bexp < -10:                           # underflow -> zero
      return sign
    # the f16 subnormal significand is the RNE of m2·2^(aexp-126), a right
    # shift by 126 - aexp with round-to-nearest-even on the remainder
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1'u32)
    if rem > half or (rem == half and (r and 1'u32) == 1'u32):
      r += 1
    return sign or uint16(r)
  # the normal path, dropping the 13 low mantissa bits with round-to-nearest-even
  let half = 0x1000'u32
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1'u32) == 1'u32):
    r += 1
    if r == 0x400:                           # mantissa overflow -> bump exponent
      inc bexp
      r = 0
      if bexp >= 31:                         # a value just below the overflow boundary
        return sign or 0x7C00'u16
  return sign or uint16(uint32(bexp) shl 10) or uint16(r)

proc fp16ToFp32*(h: uint16): float32 =
  ## Returns the exact fp32 widening of an fp16 bit pattern.
  let u = uint32(h)
  let sign = (u and 0x8000'u32) shl 16
  let e = int((u shr 10) and 0x1F'u32)
  var man = u and 0x3FF'u32
  if e == 0:
    if man == 0: return cast[float32](sign)
    # subnormal:
    #   value = man · 2⁻²⁴, built through the fp32 pattern of the
    # magnitude so the sign bit ors onto a positive normal pattern
    let val = float32(man) * 5.960464477539063e-8'f32
    return cast[float32](cast[uint32](val) or sign)
  result = cast[float32](sign or (uint32(e - 15 + 127) shl 23) or (uint32(man) shl 13))

proc narrowTo*(x: float32, dt: ScalarKind): uint16 =
  ## Returns the element dtype's round-to-nearest-even bit pattern of one
  ## fp32 value, the cast the stimulus generators use.
  ##
  ## Example:
  ##
  ##   xBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  if dt == kBfloat16: f32ToBf16(x) else: fp32ToFp16(x)

proc widenTo*(h: uint16, dt: ScalarKind): float32 =
  ## Returns the exact fp32 widening of an element-dtype bit pattern.
  ##
  ## Example:
  ##
  ##   let yGot = yB.hostPtr[i].widenTo(dt).float64
  if dt == kBfloat16: bf16ToF32(h) else: fp16ToFp32(h)

# ─── Ulp allowance vocabulary ────────────────────────────────────────

const
  U32* = 5.9604644775390625e-8        # 2⁻²⁴, the fp32 unit roundoff
  U64* = 1.1102230246251565e-16       # 2⁻⁵³, the fp64 unit roundoff

type UlpDatatype* = enum
  ## Ulp datatypes of the judged element-dtype side.
  ulpBf16
    ## bf16 storage, 7 mantissa bits, one step is 2^(binade - 7)
  ulpFp16
    ## fp16 storage, 10 mantissa bits, one step is 2^(binade - 10)

proc binade*(v: float64): int =
  ## Returns the binade index of a normal nonzero magnitude, a magnitude
  ## |v| in [2^e, 2^(e+1)) reads binade e.
  ##
  ## Raises:
  ## - ValueError for zero and non-finite input
  if v == 0.0 or classify(v) in {fcInf, fcNegInf, fcNaN}:
    raise newException(ValueError,
      "binade of a zero or non-finite magnitude has no reference: " & $v)
  var man: float64
  var exp: int
  man = frexp(abs(v), exp)
  exp - 1

proc binadeStep*(g: UlpDatatype, binade: int): float64 =
  ## Returns the width of one grid step for values in binade [2^e, 2^(e+1)).
  ##
  ## Example:
  ##
  ##   binadeStep(ulpBf16, 4)  # one bf16 ulp at 17.25, 2^(4 - 7) = 0.125
  let bits = case g
    of ulpBf16: 7
    of ulpFp16: 10
  pow(2.0, (binade - bits).float64)

proc ulpStepAt*(g: UlpDatatype, v: float64): float64 =
  ## Returns the width of one ulp at magnitude |v|, the allowance derivations' step unit.
  ##
  ## Contract:
  ## - zero returns 0.0, the callers treat a zero width as no ulp allowance
  ## - a magnitude below the datatype's min normal exponent clamps onto
  ##   the min-normal spacing, the element-dtype subnormal ladder
  ##
  ## Example:
  ##
  ##   ulpStepAt(ulpBf16, 17.25)  # 2^(4 - 7) = 0.125
  if v == 0.0: return 0.0
  let (mant, exp10) = frexp(abs(v))
  doAssert mant >= 0.5 and mant < 1.0
  let minNormalExp = if g == ulpBf16: -126 else: -14
  let floorExp = max(exp10 - 1, minNormalExp)  # floor(log2|v|), clamped at the min normal exponent
  binadeStep(g, floorExp)

proc ulpDatatypeName*(g: UlpDatatype): string =
  ## Returns the record name of one ulp datatype ("bf16" / "fp16").
  case g
  of ulpBf16: "bf16"
  of ulpFp16: "fp16"

# ─── Judgment bookkeeping ────────────────────────────────────────────

type Judge* = object
  ## Per-suite judgment statistics over one judged buffer kind.
  total*: int
  exact*: int
  worstUse*: float64
  worstUlp*: float64
  worstAbs*: float64

proc judge*(j: var Judge; ctx: string; got, want, allow, step: float64) =
  ## Judges one element against its allowance, asserting the in-model inequality
  ## and recording the worst usage, ulp distance and the bit-exact count.
  let diff = abs(got - want)
  doAssert diff <= allow,
    &"{ctx}: |Δ| {diff:.3e} > allowance {allow:.3e}"
  if diff > j.worstAbs: j.worstAbs = diff
  if allow > 0.0:
    let use = diff / allow
    if use > j.worstUse: j.worstUse = use
  if step > 0.0 and diff > 0.0:
    let u = diff / step
    if u > j.worstUlp: j.worstUlp = u
  if diff == 0.0: inc j.exact
  inc j.total

# ─── Seeded input generators ─────────────────────────────────────────

proc l2NormRows*(rng: var PropRng, dt: ScalarKind, rows, cols: int): seq[uint16] =
  ## Returns `rows` l2-normalized element-dtype rows of `cols` entries each, the kernel
  ## contract's post-l2norm query/key shape, each fp32 row being l2-normalized in fp64
  ## and then rounded to the element dtype.
  ##
  ## The normalization keeps the delta-rule recursion bounded over hundreds
  ## of tokens in fp16 y range.
  result = newSeq[uint16](rows * cols)
  for r in 0 ..< rows:
    var norm2 = 0.0'f64
    for c in 0 ..< cols:
      let x = rng.nextF32(-1.0'f32, 1.0'f32).float64
      norm2 += x * x
      result[r * cols + c] = x.float32.narrowTo(dt)
    let inv = 1.0 / sqrt(norm2)
    for c in 0 ..< cols:
      result[r * cols + c] =
        (result[r * cols + c].widenTo(dt).float64 * inv).float32.narrowTo(dt)

type RecCase* = object
  ## One seeded recurrence case (GDN or KDA shapes), the per-token rows
  ## shared by the chunk-scan launches and the per-token decode steps.
  ##
  ## | field    | shape                         |
  ## | -------- | ----------------------------- |
  ## | qBits    | (qkRows, T, Dk) element bits  |
  ## | kBits    | (qkRows, T, Dk) element bits  |
  ## | vBits    | (bhMax, T, Dv) element bits   |
  ## | betaBits | (bhMax, T) element bits       |
  ## | gVals    | (bhMax, T) f32 log decay, GDN |
  ## | state0   | (bhMax, Dv, Dk) f32           |
  qBits*: seq[uint16]
  kBits*: seq[uint16]
  vBits*: seq[uint16]
  betaBits*: seq[uint16]
  gVals*: seq[float32]
  state0*: seq[float32]

proc takeRecCase*(rng: var PropRng, dt: ScalarKind, qkRows, bhMax, T, Dv, Dk: int, kda: bool): RecCase =
  ## Returns one seeded GDN or KDA recurrence case of per-token rows.
  ##
  ## - q/k l2-normalized element rows, v in [-1, 1), beta in [0.2, 0.8),
  ##   state0 in [-1, 1), the initial state never zero
  ## - GDN carries one f32 log decay per value head (bhMax, T) inside `gVals`, KDA
  ##   carries one per key channel (qkRows, T, Dk), both in [-0.5, -0.01)
  var qBits = l2NormRows(rng, dt, qkRows * T, Dk)
  var kBits = l2NormRows(rng, dt, qkRows * T, Dk)
  var vBits = newSeq[uint16](bhMax * T * Dv)
  for i in 0 ..< bhMax * T * Dv:
    vBits[i] = rng.nextF32(-1.0'f32, 1.0'f32).narrowTo(dt)
  var betaBits = newSeq[uint16](bhMax * T)
  for i in 0 ..< bhMax * T:
    betaBits[i] = rng.nextF32(0.2'f32, 0.8'f32).narrowTo(dt)
  var gVals = newSeq[float32](if kda: qkRows * T * Dk else: bhMax * T)
  for i in 0 ..< gVals.len:
    gVals[i] = rng.nextF32(-0.5'f32, -0.01'f32)
  var state0 = newSeq[float32](bhMax * Dv * Dk)
  for i in 0 ..< bhMax * Dv * Dk:
    state0[i] = rng.nextF32(-1.0'f32, 1.0'f32)
  RecCase(qBits: qBits, kBits: kBits, vBits: vBits,
    betaBits: betaBits, gVals: gVals, state0: state0)

proc takeKdaCase*(rng: var PropRng, dt: ScalarKind, qkRows, bhMax, T, Dv, Dk: int): RecCase =
  ## Returns one seeded KDA recurrence case over the same element-dtype surface.
  ## Per-channel g (qkRows, T, Dk) carried in `gVals` under the KDA layout:
  ##
  ## - q/k stay l2-normalized element rows, the kernels widen them exactly
  ## - beta stays (bhMax, T) element bits, widened to f32 by the f32 inputs
  ##   the KDA kernels read
  result = takeRecCase(rng, dt, qkRows, bhMax, T, Dv, Dk, kda = true)

proc segments*(n: int, cuts: varargs[int]): seq[int] =
  ## Returns the segment lengths of a split of n tokens at the caller's
  ## cut positions, an empty `cuts` list giving the whole-sequence run.
  ##
  ## Example:
  ##
  ##   segments(64, 40, 57)  # @[40, 17, 7]
  result = newSeq[int]()
  var prev = 0
  for c in cuts:
    doAssert c > prev and c < n, "cut " & $c & " outside (0, " & $n & ")"
    result.add(c - prev)
    prev = c
  result.add(n - prev)
