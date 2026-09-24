# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option, this file may not be copied, modified, or distributed except according to those terms.

# ─── Bit-surgery worked examples ─────────────────────────────────────

## Unit checks over the support surface's bit surgery, the known-good IEEE-754
## patterns each rounding case must produce:
##
## - the subnormal-underflow window, the exact 2⁻²⁵ tie and the min-subnormal boundary
## - overflow saturation and the NaN short-circuits
##
## Run command, from the repo root:
## - nim test_positron_properties
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/properties/t_prop_bits.nim

import std/[math, strutils]
import ../properties/properties

const Patterns = [0x0000'u16, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x7C00, 0x7E00,
  0x8000, 0x8001, 0xFBFF, 0xFC00, 0xFE00]

proc check(label: string, got, want: uint16) =
  doAssert got == want,
    label & ": got 0x" & toHex(got.uint32, 4) & ", want 0x" & toHex(want.uint32, 4)

proc main =
  # fp32ToFp16, the subnormal ladder and its underflow window
  # 2⁻²⁵ is the exact halfway point to the min subnormal, ties round to even (0)
  check("fp16 below-half min subnormal", fp32ToFp16(9.193288e-11'f32), 0x0000'u16)
  check("fp16 exact 2^-25 tie rounds to even", fp32ToFp16(2.98023223876953125e-8'f32), 0x0000'u16)
  check("fp16 just above the tie rounds up", fp32ToFp16(3.0e-8'f32), 0x0001'u16)
  check("fp16 min subnormal", fp32ToFp16(5.9604644775390625e-8'f32), 0x0001'u16)
  check("fp16 smallest normal", fp32ToFp16(6.103515625e-5'f32), 0x0400'u16)
  # overflow saturates to inf, the largest finite stays finite
  check("fp16 largest finite stays", fp32ToFp16(65504.0'f32), 0x7BFF'u16)
  check("fp16 overflow saturates", fp32ToFp16(65520.0'f32), 0x7C00'u16)
  check("fp16 negative overflow saturates", fp32ToFp16(-65520.0'f32), 0xFC00'u16)
  # NaN quiets to the canonical pattern, sign kept
  check("fp16 NaN quiets", fp32ToFp16(NaN), 0x7E00'u16)
  check("fp16 -NaN quiets", fp32ToFp16(-NaN), 0xFE00'u16)
  # f32ToBf16, the NaN short-circuit. The round-to-even increment can carry
  # a small odd payload across the Inf/NaN boundary.
  check("bf16 NaN stays NaN", f32ToBf16(NaN), 0x7FC0'u16)
  check("bf16 -NaN keeps the sign", f32ToBf16(-NaN), 0xFFC0'u16)
  check("bf16 overflow saturates", f32ToBf16(3.5e38'f32), 0x7F80'u16)
  check("bf16 exact half ties to even", f32ToBf16(1.00390625'f32), 0x3F80'u16)
  check("bf16 just above ties up", f32ToBf16(1.0039064'f32), 0x3F81'u16)
  # widenings round-trip the finite patterns exactly
  for p in Patterns:
    if p == 0x7E00'u16 or p == 0xFE00'u16: continue  # NaN, no round-trip
    doAssert fp16ToFp32(p).fp32ToFp16 == p,
      "fp16 round-trip fails at 0x" & toHex(p.uint32, 4)
  proc checkWiden(label: string, h: uint16, want: FloatClass) =
    ## the widenings keep the Inf/NaN patterns' IEEE-754 meaning, the fp32
    ## exponent field 255 with the payload shifted through the mantissa
    let got = classify(fp16ToFp32(h))
    doAssert got == want,
      label & ": got " & $got & ", want " & $want
  checkWiden("fp16 +Inf widens to fp32 +Inf", 0x7C00'u16, fcInf)
  checkWiden("fp16 -Inf widens to fp32 -Inf", 0xFC00'u16, fcNegInf)
  checkWiden("fp16 NaN widens to fp32 NaN", 0x7E00'u16, fcNan)
  checkWiden("fp16 -NaN widens to fp32 NaN", 0xFE00'u16, fcNan)

  echo "BIT SURGERY EXAMPLES OK"

main()
