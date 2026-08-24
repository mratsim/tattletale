## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Host-side test support for the tile layer: the fp16↔fp32 bit-pattern converters
## (round-to-nearest-even) and the deterministic shape draws for the on-device tests.
## Not a test module (no `test_` prefix): `nim test_ceramic` does not run it.

import std/sha1

# ═════════════════════════════════════════════════════════════════════════
#  Host fp16 ↔ fp32 conversions (bit patterns, round-to-nearest-even)
# ═════════════════════════════════════════════════════════════════════════

func fp16ToFp32*(h: uint16): float32 =
  ## Exact binary16 → binary32: every fp16 value is representable in fp32.
  let sign = uint32(h and 0x8000'u16) shl 16
  let e = (h shr 10) and 0x1F'u16
  let m = h and 0x3FF'u16
  if e == 0:
    if m == 0: return cast[float32](sign)    # ±0
    # subnormal: value m·2⁻²⁴, normalized to a normal fp32
    var m2 = uint32(m)
    var e2 = 113
    while (m2 and 0x400'u32) == 0:
      m2 = m2 shl 1
      dec e2
    return cast[float32](sign or (uint32(e2) shl 23) or ((m2 and 0x3FF'u32) shl 13))
  if e == 0x1F:                              # Inf / NaN
    return cast[float32](sign or 0x7F800000'u32 or (uint32(m) shl 13))
  return cast[float32](sign or ((uint32(e) + 112) shl 23) or (uint32(m) shl 13))

func fp32ToFp16*(x: float32): uint16 =
  ## IEEE-754 binary32 → binary16 bit pattern, round-to-nearest-even.
  ## Handles normal, subnormal, zero, Inf, NaN (NaN quieted to a 0x200 payload, the hardware default).
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:                           # Inf / NaN
    if mant == 0: return sign or 0x7C00'u16
    return sign or 0x7C00'u16 or 0x0200'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:                             # overflow → Inf
    return sign or 0x7C00'u16
  if bexp <= 0:                              # subnormal or zero
    if bexp < -10:                           # underflow → zero
      return sign
    # the f16 subnormal significand is RNE(m2·2^(aexp−126)): a right shift
    # of (126 − aexp) with round-to-nearest-even on the remainder
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return sign or uint16(r)
  # normal: drop 13 low mantissa bits with round-to-nearest-even
  let half = 0x1000'u32
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:                           # mantissa overflow → bump exponent
      bexp += 1
      r = 0
      if bexp >= 31:
        return sign or 0x7C00'u16
  return sign or uint16(bexp shl 10) or uint16(r)

# ═════════════════════════════════════════════════════════════════════════
#  Deterministic shape draws (SHA-1 of a fixed seed and a per-draw counter)
# ═════════════════════════════════════════════════════════════════════════
#
# The on-device tests must be impossible to pass by hardcoding dimensions:
# each test draws its shapes from the SHA-1 digest of a fixed seed and a per-draw counter
# (preimage-resistant), so the dimensions cannot be baked into a kernel.
# The digest bytes are the only source of shape parameters.

type
  ShapeDraws* = object
    ## A deterministic hash-draw stream. `seed` is fixed at construction.
    ## `counter` increments per draw. `nextBytes` returns the 20-byte SHA-1 digest
    ## of `seed & ":" & counter`.
    seed*: string
    counter*: int

func initShapeDraws*(seed: string): ShapeDraws =
  ## Starts a draw stream for `seed`. All draws are deterministic: the same seed
  ## always yields the same shapes.
  ShapeDraws(seed: seed, counter: 0)

func nextBytes*(draws: var ShapeDraws): array[20, uint8] =
  ## Returns the 20 digest bytes of `sha1(seed & ":" & counter)`.
  let digest = secureHash(draws.seed & ":" & $draws.counter)
  inc draws.counter
  Sha1Digest(digest)

func drawInRange*(b: array[20, uint8]; i: int; lo, hi: int): int =
  ## Returns a deterministic value in `[lo, hi]` from digest byte `i`.
  ## Callers keep `i` distinct per parameter so the parameters stay
  ## independent.
  doAssert i >= 0 and i < 20
  lo + int(b[i]) mod (hi - lo + 1)

proc checkDrawPins*() =
  ## Verifies the fixed digest bytes and derived shapes for the first draws
  ## of each test seed. A silent hash regression must fail loudly
  ## here, not silently shrink every shape.
  block gemmPins:
    var d = initShapeDraws("gemm")
    let b0 = d.nextBytes()
    doAssert b0[0] == 0x85'u8 and b0[1] == 0xf3'u8 and b0[2] == 0x8f'u8,
      "gemm:0 digest bytes changed"
    doAssert drawInRange(b0, 0, 1, 96) == 38
    doAssert drawInRange(b0, 1, 1, 96) == 52
    doAssert drawInRange(b0, 2, 1, 96) == 48
    let b1 = d.nextBytes()
    doAssert b1 != b0, "gemm draws must differ"
    doAssert drawInRange(b1, 0, 1, 96) == 53
    doAssert drawInRange(b1, 1, 1, 96) == 35
    doAssert drawInRange(b1, 2, 1, 96) == 77
  block rmsnormPins:
    var d = initShapeDraws("rmsnorm")
    let b0 = d.nextBytes()
    doAssert b0[0] == 0x4a'u8 and b0[1] == 0x8f'u8,
      "rmsnorm:0 digest bytes changed"
    doAssert drawInRange(b0, 0, 1, 64) == 11
    doAssert drawInRange(b0, 1, 1, 128) == 16
    let b1 = d.nextBytes()
    doAssert drawInRange(b1, 0, 1, 64) == 48
    doAssert drawInRange(b1, 1, 1, 128) == 44
  block raggedPins:
    var d = initShapeDraws("ragged")
    let b0 = d.nextBytes()
    doAssert b0[0] == 0xa8'u8 and b0[1] == 0x4a'u8 and b0[2] == 0x5b'u8,
      "ragged:0 digest bytes changed"
    doAssert drawInRange(b0, 0, 1, 96) == 73
    doAssert drawInRange(b0, 1, 1, 96) == 75
    doAssert drawInRange(b0, 2, 1, 96) == 92
    # The ragged test's per-row K_eff stream consumes one digest per 20 rows
    # after the shape draw, so the second shape draw is the 6th digest
    # (b0 + 4 kEff digests for the 73-row draw + the draw itself).
    var b1: array[20, uint8]
    for i in 0 ..< 5:
      b1 = d.nextBytes()
    doAssert drawInRange(b1, 0, 1, 96) == 47,
      "ragged second shape draw changed"
    doAssert drawInRange(b1, 1, 1, 96) == 9
    doAssert drawInRange(b1, 2, 1, 96) == 48
  block attnPins:
    var d = initShapeDraws("attn")
    let b0 = d.nextBytes()
    doAssert b0[0] == 0x28'u8 and b0[1] == 0x56'u8 and b0[4] == 0x5e'u8,
      "attn:0 digest bytes changed"
    doAssert drawInRange(b0, 0, 1, 2) == 1
    doAssert drawInRange(b0, 1, 1, 2) == 1
    doAssert 8 * drawInRange(b0, 4, 1, 12) == 88
    let b1 = d.nextBytes()
    doAssert 8 * drawInRange(b1, 4, 1, 12) == 24,
      "attn:1 KV dimension changed"
    let b2 = d.nextBytes()
    doAssert 8 * drawInRange(b2, 4, 1, 12) == 16,
      "attn:2 KV dimension changed"
    doAssert drawInRange(b2, 0, 1, 2) == 1 and drawInRange(b2, 1, 1, 2) == 2,
      "attn:2 batch/head draws changed"
when isMainModule:
  checkDrawPins()
