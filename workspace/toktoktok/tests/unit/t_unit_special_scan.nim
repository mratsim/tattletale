# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Explicit special-scan unit rows over named dictionaries and inputs,
## exact decision list:
## - greedy leftmost-start selection.
## - the first-declared same-start tie rule.
## - ordinary-region non-split.
## - the automaton path agreeing with the two-level path.
## - the empty dictionary passthrough.
## - the empty-pattern rejection.
##
## Fuzz and chunk-boundary coverage lives in tests/fuzzing/t_special_scan_stage.nim.

import std/[monotimes, times]

import workspace/toktoktok/src/machine
import workspace/toktoktok/src/scan

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc collectDecisions(scanner: SpecialScanner, text: string): seq[SpecialDecision] =
  var m = SpecialScan.init(scanner, text)
  for d in m.items:
    result.add d

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  block:
    let sc = SpecialScanner.init(["\n\n", "<|fim|>"], [5, 7])
    check "scan 'hello <|fim|> world' -> ord | fim | ord",
      collectDecisions(sc, "hello <|fim|> world") == @[
        SpecialDecision(lo: 0, hi: 6, specialId: -1),
        SpecialDecision(lo: 6, hi: 13, specialId: 7),
        SpecialDecision(lo: 13, hi: 19, specialId: -1)]

  block:
    # same-start tie:
    #   first declared wins, not longest match
    let sc = SpecialScanner.init(["ab", "abc"], [3, 4])
    check "scan tie 'abc ab' -> 'ab' id 3, one ordinary region, 'ab' id 3",
      collectDecisions(sc, "abc ab") == @[
        SpecialDecision(lo: 0, hi: 2, specialId: 3),
        SpecialDecision(lo: 2, hi: 4, specialId: -1),
        SpecialDecision(lo: 4, hi: 6, specialId: 3)]

  block:
    # automaton path (forceDaac) == two-level path on the same rows
    let sc2 = SpecialScanner.init(["\n\n", "<|fim|>"], [5, 7])
    let scD = SpecialScanner.init(["\n\n", "<|fim|>"], [5, 7], forceDaac = true)
    let text = "a <|fim|> b\n\n c"
    check "scan forceDaac == two-level on 'a <|fim|> b\\n\\n c'",
      collectDecisions(scD, text) == collectDecisions(sc2, text)

  block:
    let sc = SpecialScanner.init([], [])
    check "scan empty dictionary passthrough 'hi' -> one ordinary decision",
      collectDecisions(sc, "hi") == @[SpecialDecision(lo: 0, hi: 2, specialId: -1)]

  block:
    var raised = false
    try:
      discard SpecialScanner.init([""], [1])
    except ValueError:
      raised = true
    check "scan empty pattern rejected with ValueError", raised

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall special-scan unit rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
