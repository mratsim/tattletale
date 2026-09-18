# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Double-array Aho-Corasick core verification.
##
## Reference semantics for the comparison, standard overlapping AC:
##
## | aspect     | semantics                                                                                                                                |
## | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
## | occurrence | every pattern occurrence is reported at its end position with its value, enumerated exhaustively per end position                        |
## | stream     | the automaton must reproduce exactly this stream, byte for byte                                                                          |
## | NUL        | NUL bytes in patterns and haystacks stay in the corpus, block-close CHECK embedding must prevent any false transition into a vacant slot |
##
## Run:
##   $ nim cpp -r --outdir:build/tests --nimcache:nimcache/tests workspace/data_structures/tests/test_aho_corasick.nim

import std/random
import std/algorithm
import std/strutils
import std/importutils
import workspace/data_structures/aho_corasick
privateAccess(AhoCorasick)

type
  Triple = tuple[start, endPos, value: int]

proc naiveOverlapping(text: string, patterns: seq[string]): seq[Triple] =
  ## Exhaustive reference for standard overlapping semantics.
  ## For each end position, every pattern that is a suffix of the prefix
  ## ending there is reported in pattern order.
  for e in 0 ..< text.len:
    for pidx, pat in patterns:
      let s = e + 1 - pat.len
      if s >= 0:
        var ok = true
        for i in 0 ..< pat.len:
          if text[s + i] != pat[i]:
            ok = false
            break
        if ok:
          result.add (s, e + 1, pidx)

proc automatonOverlapping(text: string, patterns: seq[string],
    d: AhoCorasick): seq[Triple] =
  ## Walks the automaton byte by byte, reporting every output chain
  ## entry (start = end - pattern length).
  var state = int32(AhoCorasickRootIdx)
  for i in 0 ..< text.len:
    state = d.nextState(state, uint8(text[i]))
    var op = d.outputHead(state)
    while op != 0:
      let e = i + 1
      let s = e - int(d.outputLength(op))
      result.add (s, e, int(d.outputValue(op)))
      op = d.outputParent(op)

proc check(name: string, ok: bool, failures: var int, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    failures += 1

proc runFuzzRound(r: var Rand, patterns: var seq[string],
    text: var string, alphabet: string) =
  ## Generates one random dictionary plus one random haystack over a small alphabet.
  ## Dense overlap regime:
  ##
  ## | aspect     | description          |
  ## | ---------- | -------------------- |
  ## | patterns   | short patterns       |
  ## | collisions | frequent collisions  |
  ## | chains     | prefix chains        |
  ## | NUL        | NUL bytes in the mix |
  patterns.setLen(0)
  let numPatterns = r.rand(0..14)
  for i in 0 ..< numPatterns:
    var pat = ""
    let patLen = r.rand(1..6)
    for j in 0 ..< patLen:
      pat.add alphabet[r.rand(0..(alphabet.len - 1))]
    if pat notin patterns:
      patterns.add pat
  text.setLen(0)
  let textLen = r.rand(0..220)
  for j in 0 ..< textLen:
    text.add alphabet[r.rand(0..(alphabet.len - 1))]

proc toSeqInt(n: int): seq[int] =
  result = @[]
  for i in 0 ..< n:
    result.add i

proc runTests*() =
  ## Runs every check, echoes a failure count, quits 1 on any failure.
  var failures = 0

  block:
    # Worked example, patterns bcd, ab, a over "abcd".
    # Standard non-overlapping find order reports a then bcd.
    let pats = @["bcd", "ab", "a"]
    let d = buildAhoCorasick(pats, @[0, 1, 2])
    let got = automatonOverlapping("abcd", pats, d)
    check "doc example, overlapping stream matches naive",
      got == naiveOverlapping("abcd", pats), failures, $got

  block:
    # Prefix chain tie reproducer, both patterns match at start 1 of "xabcd"
    # and the overlapping core must report BOTH occurrences.
    # Standard kind, tie policy belongs to the scanner layer.
    let pats = @["bcd", "abcd"]
    let d = buildAhoCorasick(pats, @[0, 1])
    let got = automatonOverlapping("xabcd", pats, d)
    var gotSorted = got
    var wantSorted = naiveOverlapping("xabcd", pats)
    gotSorted.sort(system.cmp)
    wantSorted.sort(system.cmp)
    check "cross-branch suffix case, both occurrences reported",
      gotSorted == wantSorted, failures, $got

  block:
    # Empty pattern set. Root-only automaton, no transitions, no output.
    let emptyPats: seq[string] = @[]
    let d = buildAhoCorasick(emptyPats, @[])
    var state = int32(AhoCorasickRootIdx)
    for c in "anything":
      state = d.nextState(state, uint8(c))
    check "empty dictionary, walk stays at root", state == AhoCorasickRootIdx,
      failures
    check "empty dictionary, no outputs",
      d.outputHead(int32(AhoCorasickRootIdx)) == 0, failures

  block:
    # Determinism. Same input list must give byte-identical arrays.
    let pats = @["bcd", "ab", "a", "abce", "\x00a\x00"]
    let vals = @[0, 1, 2, 3, 4]
    let d1 = buildAhoCorasick(pats, vals)
    let d2 = buildAhoCorasick(pats, vals)
    check "determinism, states identical", d1.states == d2.states, failures
    check "determinism, outputs identical", d1.outputs == d2.outputs, failures

  block:
    # fuzz over several alphabets, many rounds, fixed seeds.
    const Alphabets = [
      "ab",
      "abc",
      "ab\x00",
      "ab\x00c\x01",
      "a\xC3\xA9b",       # multi-byte utf-8 bytes as ordinary symbols
      "ab<|>",
    ]
    const RoundsPerAlphabet = 400
    var patterns: seq[string] = @[]
    var text = ""
    var seed = 20260910'i64
    for alphaIdx in 0 ..< Alphabets.len:
      var r = initRand(seed)
      seed += 1
      for round in 0 ..< RoundsPerAlphabet:
        runFuzzRound(r, patterns, text, Alphabets[alphaIdx])
        let d = buildAhoCorasick(patterns,
            toSeqInt(patterns.len))
        let want = naiveOverlapping(text, patterns)
        let got = automatonOverlapping(text, patterns, d)
        # Chain order within one end position is unspecified by the core
        # (the scanner tie-break is order-insensitive), so the comparison
        # is over sorted match multisets.
        var gotSorted = got
        var wantSorted = want
        gotSorted.sort(system.cmp)
        wantSorted.sort(system.cmp)
        if gotSorted != wantSorted:
          check "fuzz round", false, failures,
            "alpha=" & $alphaIdx & " seed=" & $seed & " round=" & $round &
            " patterns=" & $patterns & " text=" & text.escape &
            " want=" & $want & " got=" & $got
          break

    echo "fuzz rounds completed: ", Alphabets.len * RoundsPerAlphabet

  if failures > 0:
    echo "\n", failures, " check(s) failed"
    quit(1)
  echo "\nall aho-corasick-core checks passed"

when isMainModule:
  runTests()
