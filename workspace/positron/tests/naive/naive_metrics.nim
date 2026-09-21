# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Comparison metrics and the untouched-memory sentinel for the positron
## naive test tier.
##
## - worstSeqDiff reports the largest elementwise difference
## - bitExactCount counts bitwise-equal 16-bit pattern pairs
## - GuardedTail verifies a buffer tail the code under test must not
##   touch stays bit-identical

import std/math

func worstSeqDiff*(got, want: openArray[float32]): float64 =
  ## Returns the largest |got[i] - want[i]| over the pair.
  ##
  ## Expected input:
  ## - two equal-length f32 sequences, two empty sequences are equal
  ##   by construction and the result is 0.0
  ## - no NaN on either side, a NaN raises ValueError (a NaN difference reads back as zero)
  doAssert got.len == want.len,
    "sequence length mismatch: got " & $got.len & ", want " & $want.len
  result = 0.0
  for i in 0 ..< got.len:
    let d = abs(float64(got[i]) - float64(want[i]))
    if d.classify == fcNaN:
      raise newException(ValueError,
        "element " & $i & " of the compared pair is NaN")
    if d > result: result = d

func bitExactCount*(got, want: openArray[uint16]): int =
  ## Returns the number of index positions where the two 16-bit patterns
  ## are bitwise equal. Exact-where-legal comparisons (state continuity, untouched lanes)
  ## use this, a partial count is a real mismatch.
  doAssert got.len == want.len,
    "sequence length mismatch: got " & $got.len & ", want " & $want.len
  for i in 0 ..< got.len:
    if got[i] == want[i]: inc result

# Untouched-memory sentinel helpers

type
  GuardedTail*[T] = object
    ## A buffer split into a used region the test hands to the code
    ## under test and a tail region carrying `fillPattern` that must
    ## stay bit-identical through the run.
    data*: seq[T]
    used*: int
    fillPattern*: T

proc initGuardedTail*[T](used, tailElems: int; fillPattern: T): GuardedTail[T] =
  ## Returns `used + tailElems` zero-initialized elements.
  ## - [0, used) left at zero, for the test to write into
  ## - [used, used + tailElems) carrying `fillPattern`
  result.data = newSeq[T](used + tailElems)
  for i in used ..< result.data.len:
    result.data[i] = fillPattern
  result.used = used
  result.fillPattern = fillPattern

proc assertTailUntouched*[T](g: GuardedTail[T]) =
  ## Raises AssertionError when any element at or past `used` fails to hold
  ## the exact `fillPattern` bit pattern. The first offending
  ## index appears within the assert message.
  for i in g.used ..< g.data.len:
    doAssert g.data[i] == g.fillPattern,
      "guarded tail element " & $i & " was modified: expected " &
      $g.fillPattern & ", got " & $g.data[i]

func countChanged*[T](data, snap: openArray[T]): int =
  ## Returns the number of positions where the two sequences differ
  ## bitwise. A detector test asserts the count directly,
  ## an unchanged-region check asserts the count is zero.
  doAssert data.len == snap.len,
    "sequence length mismatch: data " & $data.len & ", snap " & $snap.len
  for i in 0 ..< data.len:
    if data[i] != snap[i]: inc result

func snapshotSeq*[T](data: openArray[T]): seq[T] =
  ## Returns a copy of `data` for a later countChanged check, a region
  ## check passes toOpenArray(buf, first, last).
  result = newSeq[T](data.len)
  for i in 0 ..< data.len:
    result[i] = data[i]
