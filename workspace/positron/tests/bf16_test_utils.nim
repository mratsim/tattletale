## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host-side bf16 test data helpers for the positron manual kernel tests:
##
## - RNE (round-to-nearest-even) bf16 rounding,
##   matching torch's float32 → bfloat16 cast
## - seeded random bf16 buffers
##
## Values stay in [-scale, scale], uniform fp32 arithmetic, no clamp. bf16
## denormals, magnitudes below 2^-126, appear at none of the test scales 0.5
## and 1.0: the smallest nonzero magnitude reachable is scale · 2^-24,
## above the denormal floor by 2^100. Metal's float() flushes denormals
## to zero, the host cast keeps them, so any denormal input would split
## kernel results from the reference implementation.

import std/random

func bf16BitsFromFp32*(x: float32): uint16 =
  ## bf16 bit pattern of an fp32, the rounded top 16 bits. Matches
  ## torch's float32 → bfloat16 cast.
  let u = cast[uint32](x)
  let roundingBias = 0x7FFF'u32 + ((u shr 16) and 1u32)
  uint16((u + roundingBias) shr 16)

func bf16BitsToF32*(b: uint16): float32 =
  ## fp32 holding a bf16 bit pattern in its top 16 bits, low 16 bits zero.
  cast[float32](uint32(b) shl 16)

proc randomBf16Buffer*(rng: var Rand, n: int, scale: float32): seq[uint16] =
  ## Returns n uniform random values in [-scale, scale] as bf16 bit patterns.
  result = newSeq[uint16](n)
  for i in 0 ..< n:
    result[i] = bf16BitsFromFp32(float32(rng.rand(1.0'f32)) * 2.0'f32 * scale - scale)
