# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Seeded randomness for the positron naive test tier.
## Every randomized input derives from an explicitly seeded `NaiveRng`,
## the single randomness source of the whole tier.
##
## Generator, xorshift64* over this module's own bit operations:
## - the sequence depends on the seed alone, never on the compiler,
##   a given seed yields the same stream on every Nim version
## - t_naive_harness.nim asserts exact golden stream values
## - std/random is excluded, its output sequence for a given seed is
##   not stable across Nim compiler versions, unlike this module

type
  NaiveRng* = object
    ## xorshift64* stream state. Must stay nonzero, the zero state is
    ## the fixed point of the recurrence.
    state: uint64

proc initNaiveRng*(seed: uint64): NaiveRng =
  ## Starts a deterministic stream from `seed`, mapping a zero seed
  ## to a fixed nonzero state so callers passing 0 get a valid stream.
  let state = if seed == 0: 0x9E3779B97F4A7C15'u64 else: seed
  NaiveRng(state: state)

proc nextU64*(rng: var NaiveRng): uint64 =
  ## Returns the next 64-bit value of the stream.
  ## The same seed always yields the same sequence of values.
  var x = rng.state
  x = x xor (x shr 12)
  x = x xor (x shl 25)
  x = x xor (x shr 27)
  rng.state = x
  result = x * 0x2545F4914F6CDD1D'u64

proc nextF32*(rng: var NaiveRng; lo, hi: float32): float32 =
  ## Returns a uniform float32 in [lo, hi), using the 24 high bits
  ## of the next stream value, at 2⁻²⁴ granularity over the range.
  doAssert hi > lo, "empty range [" & $lo & ", " & $hi & ")"
  let unit = float32(rng.nextU64() shr 40) * (1.0'f32 / 16777216.0'f32)
  lo + unit * (hi - lo)

proc nextInt*(rng: var NaiveRng; lo, hi: int): int =
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
