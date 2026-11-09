# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Design-by-contract checks: message tuples, conditions, contracts
##
## Test that the colon form raises with the condition as written
## Test that the (condition, "message") tuple raises with its message
## Test that the loop invariant raises per iteration under {.contracts.}
##
## Run from the repo root:
##   nim c -r --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wip --outdir:build/wip \
##     workspace/tests/t_design_by_contract.nim

import std/strutils
import workspace/design_by_contract

proc failingPre(n: int) =
  precondition: n > 2

proc spelledPre(n: int) =
  precondition: (n > 2, "the inputs must order")

proc spelledInvariant() {.contracts.} =
  var i = 0
  while i < 3:
    {.invariant: (i < 2, "the loop stops at 2").}
    inc i

proc contractsProc(a, b: int) {.contracts, noreturn.} =
  {.precondition: (a == b, "inputs must match").}
  {.postcondition: a + b == 3.}

proc expectDefect(body: proc {.nimcall.}, marker: string) =
  try:
    body()
  except AssertionDefect as e:
    doAssert marker in e.msg, "got: " & e.msg
    return
  doAssert false, "no defect raised, expected: " & marker

expectDefect(proc() = failingPre(1), "precondition violated: n > 2")
expectDefect(proc() = spelledPre(1), "precondition violated: the inputs must order")
expectDefect(proc() = spelledInvariant(), "invariant violated: the loop stops at 2")
expectDefect(proc() = contractsProc(1, 2), "precondition violated: inputs must match")

echo "design_by_contract: all checks passed"
