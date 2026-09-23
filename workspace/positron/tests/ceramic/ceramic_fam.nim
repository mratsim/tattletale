# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ─── Element-dtype band helpers, shared by the ceramic-vs-naive suites ─────────────

## Element-dtype helpers shared by the ceramic kernel suites.
##
## - the bf16/fp16 element-dtype enum, the exact bit-level rounds and widenings
## - the element-dtype ulp width at a value
## - the host-side buffer snapshot
##
## Suites' band constants stay suite-local. The band-model term set
## belongs to each suite, only these helpers are shared.

import std/math
import ../naive/naive_tensors

type Dtype* = enum
  dtypeBf16, dtypeF16

const U32* = 5.9604644775390625e-8        # 2⁻²⁴, the fp32 unit roundoff

proc toDtypeBits*(dt: Dtype, x: float32): uint16 =
  ## Returns the element-dtype round-to-nearest-even bit pattern of an fp32 value.
  if dt == dtypeBf16: f32ToBf16(x) else: fp32ToFp16(x)

proc widenDtype*(dt: Dtype, h: uint16): float32 =
  ## Returns the exact fp32 widening of an element-dtype bit pattern.
  if dt == dtypeBf16: bf16ToF32(h) else: fp16ToFp32(h)

proc widenDtype64*(dt: Dtype, h: uint16): float64 =
  ## Returns the exact fp64 widening of a family dtype bit pattern.
  widenDtype(dt, h).float64

proc dtypeName*(dt: Dtype): string =
  ## Returns the family dtype's printable name.
  if dt == dtypeBf16: "bf16" else: "fp16"

proc dtypeUlp*(dt: Dtype, v: float64): float64 =
  ## Width of one family-dtype ulp at a nonzero |v|, subnormal values
  ## clamp to the family's min-normal spacing.
  if v == 0.0: return 0.0
  let (mant, exp10) = frexp(abs(v))
  doAssert mant >= 0.5 and mant < 1.0
  let minNormalExp = if dt == dtypeBf16: -126 else: -14
  let floorExp = max(exp10 - 1, minNormalExp)  # floor(log2|v|), clamped at the min normal exponent
  let mantBits = if dt == dtypeBf16: 7 else: 10
  result = pow(2.0, float64(floorExp - mantBits))

proc readInto*[T](src: ptr UncheckedArray[T]; count: int): seq[T] =
  ## Host-side snapshot of `count` elements out of a kernel-written buffer,
  ## the bars' and bit-identity checks' raw material.
  result = newSeq[T](count)
  for i in 0 ..< count:
    result[i] = src[i]
