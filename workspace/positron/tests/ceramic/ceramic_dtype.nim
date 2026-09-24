# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ─── Element-dtype surface, mirrored from transformers, shared by the ceramic suites ───

## Element-dtype surface of the ceramic kernel suites, mirrored from the transformers tier:
##
## - `ScalarKind` with `narrowTo`/`widenTo` mirrors the libtorch `ScalarKind`
##   dtype enum and the transformers layers' `.to(dtype)` conversions
## - the ulp tolerance block (`UlpDatatype`, `binade`, `binadeStep`, `ulpStepAt`, `ulpDatatypeName`)
##   mirrors the transformers harness's allowance vocabulary (workspace/transformers/tests/harness/harness.nim)
##
## - the scalar bit surgery itself (`f32ToBf16`, `fp16ToFp32`, ...) lives in the naive tensors module, this surface routes through it
## - suites' band constants stay suite-local, only this surface is shared

import std/math
import ../naive/naive_tensors

type ScalarKind* = enum
  ## Element dtype of one kernel binding, the libtorch ScalarKind members
  ## the test surface instantiates.
  kFloat16
  kBfloat16

const U32* = 5.9604644775390625e-8        # 2⁻²⁴, the fp32 unit roundoff

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
  ## - zero returns 0.0, the callers treat a zero width as no ulp band
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
  ## Returns the record-file name of one ulp datatype ("bf16" / "fp16").
  case g
  of ulpBf16: "bf16"
  of ulpFp16: "fp16"

proc readRecord*[T](src: ptr UncheckedArray[T]; count: int): seq[T] =
  ## Returns the host-side record of `count` elements read out of one
  ## kernel-written buffer, the bars' and bit-identity checks' raw material.
  ##
  ## Contract:
  ## - `src` stays valid over the whole copy, the seq owns its own storage
  result = newSeq[T](count)
  for i in 0 ..< count:
    result[i] = src[i]
