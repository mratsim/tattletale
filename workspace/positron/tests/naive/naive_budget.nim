# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shape ceilings and run-line plumbing for the positron naive test tier.
##
## - NaivePrefillMaxT and NaiveMaxDk bound every naive reference
##   run's per-test shape, tests needing more state the reason
##   inside their headers
## - runTimed prints each test's wall-clock line

import std/[times, strutils]

const
  NaivePrefillMaxT* = 256
    ## Prefill sequence-length ceiling for a naive reference run.
    ## Tests needing more state the reason in the test header.
  NaiveMaxDk* = 128
    ## Head-dimension ceiling (query/key/value per head) for a naive
    ## reference run. Decode steps pass T = 1, well under NaivePrefillMaxT.
    ## Tests needing more state the reason in the test header.

proc checkShapeBudget*(t, dk: int) =
  ## Asserts a test's shape sits inside the naive-tier ceilings.
  ## Call it at setup with the actual T and dk values the test will use.
  doAssert t <= NaivePrefillMaxT,
    "prefill length " & $t & " exceeds the naive-tier ceiling " & $NaivePrefillMaxT
  doAssert dk <= NaiveMaxDk,
    "head dimension " & $dk & " exceeds the naive-tier ceiling " & $NaiveMaxDk

template runTimed*(name: string, body: untyped) =
  ## Runs `body` in its own scope and prints its wall-clock line.
  ## A failing check raises before the line prints, so a printed line
  ## means the check completed.
  let t0Naive = epochTime()
  block:
    body
  echo "  ", name, ": ", formatFloat(epochTime() - t0Naive, ffDecimal, 3), " s"
