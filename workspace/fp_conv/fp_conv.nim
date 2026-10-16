# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Floating point conversions, one generic core over the format spec.
##
## - `F16`, `BF16`, `F8E4M3`, `F8E5M2`: low-precision floats as raw bit patterns
## - `from`: converts into the target, `F16.from(1.0'f32)`, `float32.from(f16value)`
## - conversions are round-to-nearest-even, NaN preserving, sign preserving
## - `F8E4M3` is the OCP finite format: no infinity, overflow saturates
##
## The other formats are IEEE, overflow goes to infinity.
##
## The core is parameterized by the source's (exponent, mantissa) bit widths
## and the target's format spec, so adding a format is a spec, not an algorithm.

import std/math

{.push checks: off, raises: [].}

# ═══════════════════════════════════════════════════════════════
#  Formats
# ═══════════════════════════════════════════════════════════════

type
  F16* = distinct uint16    ## IEEE binary16, E5 M10
  BF16* = distinct uint16   ## Brain float, E8 M7, binary32 truncated
  F8E4M3* = distinct uint8  ## OCP FP8 E4M3fn, finite, no infinity
  F8E5M2* = distinct uint8  ## OCP FP8 E5M2, IEEE-like

template expBits*(T: typedesc[F16]): int = 5
template expBits*(T: typedesc[BF16]): int = 8
template expBits*(T: typedesc[F8E4M3]): int = 4
template expBits*(T: typedesc[F8E5M2]): int = 5

template manBits*(T: typedesc[F16]): int = 10
template manBits*(T: typedesc[BF16]): int = 7
template manBits*(T: typedesc[F8E4M3]): int = 3
template manBits*(T: typedesc[F8E5M2]): int = 2

template bias*(T: typedesc[F16]): int = 15
template bias*(T: typedesc[BF16]): int = 127
template bias*(T: typedesc[F8E4M3]): int = 7
template bias*(T: typedesc[F8E5M2]): int = 15

template hasInf*(T: typedesc[F16]): bool = true
template hasInf*(T: typedesc[BF16]): bool = true
template hasInf*(T: typedesc[F8E4M3]): bool = false
template hasInf*(T: typedesc[F8E5M2]): bool = true

template bits*(x: F16): uint16 = uint16(x)
template bits*(x: BF16): uint16 = uint16(x)
template bits*(x: F8E4M3): uint8 = uint8(x)
template bits*(x: F8E5M2): uint8 = uint8(x)

template fromBits*(T: typedesc[F16], b: uint16): F16 = F16(b)
template fromBits*(T: typedesc[BF16], b: uint16): BF16 = BF16(b)
template fromBits*(T: typedesc[F8E4M3], b: uint8): F8E4M3 = F8E4M3(b)
template fromBits*(T: typedesc[F8E5M2], b: uint8): F8E5M2 = F8E5M2(b)

# ═══════════════════════════════════════════════════════════════
#  Format constants
# ═══════════════════════════════════════════════════════════════

template wrap[T: F16 or BF16 or F8E4M3 or F8E5M2](v: uint32): T =
  when sizeof(T) == 1: T(v.uint8)
  else: T(v.uint16)

template maxFiniteBits(T: typedesc): uint32 =
  ## IEEE: exponent one below all-ones, all-ones mantissa.
  ## e4m3: all-ones exponent, mantissa 0b..110, 0b..111 encodes NaN.
  when T.hasInf:
    (uint32((1 shl T.expBits) - 2) shl T.manBits) or uint32((1 shl T.manBits) - 1)
  else:
    (uint32((1 shl T.expBits) - 1) shl T.manBits) or uint32((1 shl T.manBits) - 2)

template nanBits(T: typedesc): uint32 =
  (uint32((1 shl T.expBits) - 1) shl T.manBits) or uint32((1 shl T.manBits) - 1)

template infBits(T: typedesc): uint32 =
  uint32((1 shl T.expBits) - 1) shl T.manBits

template maxFinite*(T: typedesc[F16]): F16 = wrap[F16](maxFiniteBits(T))
template maxFinite*(T: typedesc[BF16]): BF16 = wrap[BF16](maxFiniteBits(T))
template maxFinite*(T: typedesc[F8E4M3]): F8E4M3 = wrap[F8E4M3](maxFiniteBits(T))
template maxFinite*(T: typedesc[F8E5M2]): F8E5M2 = wrap[F8E5M2](maxFiniteBits(T))

template inf*(T: typedesc[F16]): F16 = wrap[F16](infBits(T))
template inf*(T: typedesc[BF16]): BF16 = wrap[BF16](infBits(T))
template inf*(T: typedesc[F8E5M2]): F8E5M2 = wrap[F8E5M2](infBits(T))

template nan*(T: typedesc[F16]): F16 = wrap[F16](nanBits(T))
template nan*(T: typedesc[BF16]): BF16 = wrap[BF16](nanBits(T))
template nan*(T: typedesc[F8E4M3]): F8E4M3 = wrap[F8E4M3](nanBits(T))
template nan*(T: typedesc[F8E5M2]): F8E5M2 = wrap[F8E5M2](nanBits(T))

template minNormal*(T: typedesc[F16]): F16 = wrap[F16](uint32(1) shl T.manBits)
template minNormal*(T: typedesc[BF16]): BF16 = wrap[BF16](uint32(1) shl T.manBits)
template minNormal*(T: typedesc[F8E4M3]): F8E4M3 = wrap[F8E4M3](uint32(1) shl T.manBits)
template minNormal*(T: typedesc[F8E5M2]): F8E5M2 = wrap[F8E5M2](uint32(1) shl T.manBits)

template eps*(T: typedesc[F16]): F16 = wrap[F16](uint32(T.bias + 1 - T.manBits) shl T.manBits)
template eps*(T: typedesc[BF16]): BF16 = wrap[BF16](uint32(T.bias + 1 - T.manBits) shl T.manBits)
template eps*(T: typedesc[F8E4M3]): F8E4M3 = wrap[F8E4M3](uint32(T.bias + 1 - T.manBits) shl T.manBits)
template eps*(T: typedesc[F8E5M2]): F8E5M2 = wrap[F8E5M2](uint32(T.bias + 1 - T.manBits) shl T.manBits)

# ═══════════════════════════════════════════════════════════════
#  Generic core, IEEE source bits -> small format
# ═══════════════════════════════════════════════════════════════

func toSmall*[T: F16 or BF16 or F8E4M3 or F8E5M2, SE, SM: static int](
    srcBits: uint64): T {.inline.} =
  ## Round-to-nearest-even conversion from an IEEE source's raw bits.
  ##
  ## Args:
  ##   - srcBits: the source raw bits in a uint64 container
  ##   - the source has SE exponent bits, SM mantissa bits
  ##   - SE+SM+1 <= 64 so the shifts stay in range
  const
    TE = T.expBits
    TM = T.manBits
    tBias = (1 shl (TE - 1)) - 1
    tEmax = (1 shl TE) - 1
    saturating = not T.hasInf
    maxE = when saturating: tEmax else: tEmax - 1
    maxMT = (1 shl TM) + (when saturating: (1 shl TM) - 2 else: (1 shl TM) - 1)
    srcBias = (1 shl (SE - 1)) - 1
    srcEmax = (1 shl SE) - 1
    maxFiniteB = maxFiniteBits(T)
    nanB = nanBits(T)
    infB = infBits(T)

  # the sign goes at the target's top bit, not the source's
  let sign = uint32(srcBits shr (SE + SM)) shl (TE + TM)
  let exp = uint32((srcBits shr SM) and ((1'u64 shl SE) - 1))
  let man = srcBits and ((1'u64 shl SM) - 1)

  # source NaN and infinity
  if exp == srcEmax:
    if man != 0:
      return wrap[T](sign or nanB)
    when saturating:
      return wrap[T](sign or nanB)
    else:
      return wrap[T](sign or infB)

  # unbiased source exponent and implicit-one mantissa
  var e: int
  var m: uint64
  if exp == 0:
    e = 1 - srcBias
    m = man
  else:
    e = int(exp) - srcBias
    m = (1'u64 shl SM) or man

  if e >= 1 - tBias:
    # normal target
    let shift = SM - TM
    var mT = m shr shift
    let roundBit = (m shr (shift - 1)) and 1
    let sticky = (m and ((1'u64 shl (shift - 1)) - 1)) != 0
    if roundBit == 1 and (sticky or (mT and 1) == 1):
      mT.inc
    var eT = e + tBias
    if mT == 1'u64 shl (TM + 1):
      # the rounding carried into the exponent
      mT = 1'u64 shl TM
      eT.inc
    when saturating:
      if eT > maxE or (eT == maxE and mT > maxMT):
        return wrap[T](sign or maxFiniteB)
    else:
      if eT >= tEmax:
        return wrap[T](sign or infB)
    return wrap[T](sign or (uint32(eT) shl TM) or uint32(mT - (1'u64 shl TM)))
  else:
    # subnormal target
    let shift = SM - TM + 1 - tBias - e
    if shift > SM + 1:
      # everything rounds away, the round bit is 0
      return wrap[T](sign)
    var mT = m shr shift
    let roundBit = (m shr (shift - 1)) and 1
    let sticky = (m and ((1'u64 shl (shift - 1)) - 1)) != 0
    if roundBit == 1 and (sticky or (mT and 1) == 1):
      mT.inc
    if mT == 1'u64 shl TM:
      # rounds up into the smallest normal
      return wrap[T](sign or (uint32(1) shl TM))
    return wrap[T](sign or uint32(mT))

# ═══════════════════════════════════════════════════════════════
#  Small format -> IEEE, exact through binary64
# ═══════════════════════════════════════════════════════════════

template pow2f64*(k: int): float64 =
  ## 2^k as a binary64, k in [-1074, 1023]
  cast[float64](uint64(k + 1023) shl 52)

func toF64*[T: F16 or BF16 or F8E4M3 or F8E5M2](x: T): float64 {.inline.} =
  ## Exact value of x as a binary64, every small format value
  ## is representable there.
  const
    TE = T.expBits
    TM = T.manBits
    tBias = (1 shl (TE - 1)) - 1
    tEmax = (1 shl TE) - 1
  let b = uint32(x.bits)
  let sign = (b shr (TE + TM)) and 1
  let expT = (b shr TM) and tEmax
  let manT = b and ((1 shl TM) - 1)
  if expT == tEmax and manT != 0:
    return NaN
  if expT == tEmax:
    return if sign == 1: NegInf else: Inf
  if expT == 0:
    if manT == 0:
      return if sign == 1: -0.0 else: 0.0
    # subnormal: manT * 2^(1-tBias-TM), exact in binary64
    let v = float64(manT) * pow2f64(1 - tBias - TM)
    return if sign == 1: -v else: v
  let v = float64((1 shl TM) or int(manT)) * pow2f64(int(expT) - tBias - TM)
  return if sign == 1: -v else: v

# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

template `from`*[T: F16 or BF16 or F8E4M3 or F8E5M2](
    _: typedesc[T], src: float32): T =
  ## Round-to-nearest-even conversion of a binary32
  toSmall[T, 8, 23](uint64(cast[uint32](src)))

template `from`*[T: F16 or BF16 or F8E4M3 or F8E5M2](
    _: typedesc[T], src: float64): T =
  ## Round-to-nearest-even conversion of a binary64
  toSmall[T, 11, 52](cast[uint64](src))

template `from`*[T: F16 or BF16 or F8E4M3 or F8E5M2](
    _: typedesc[T], src: F16 or BF16 or F8E4M3 or F8E5M2): T =
  ## Format to format, exact widening to binary32 then one rounding
  T.from(float32.from(src))

template `from`*(T: typedesc[float32], x: F16 or BF16 or F8E4M3 or F8E5M2): float32 =
  ## Binary32 nearest x, exact
  float32(x.toF64)

template `from`*(T: typedesc[float64], x: F16 or BF16 or F8E4M3 or F8E5M2): float64 =
  ## Exact value of x
  x.toF64

template `$`*[T: F16 or BF16 or F8E4M3 or F8E5M2](x: T): string =
  $(float32.from(x))

{.pop.}
