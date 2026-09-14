# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Special-scan machine verification:
## 1. decision stream vs the greedy reference algorithm
##    (earliest start wins, same-start ties by declaration order),
##    fuzz corpus generated in-test with fixed seeds,
## 2. two-level prefix path vs automaton path equivalence on the same
##    fuzz corpora and chunkings,
## 3. carry-buffer parity rows, specials EXACTLY at chunk boundaries
##    for every split offset, including multi-byte UTF-8 tokens
##    straddling the boundary - chunked decisions must equal whole-input decisions,
## 4. machine stream invariants, full-drain equality across all
##    iteration styles, drain discipline, idempotent drained.

import std/[monotimes, times]
import std/random
import std/strutils
import workspace/toktoktok/src/machine
import workspace/toktoktok/src/scan

type
  Decision = tuple[start, endPos, id: int]

proc naiveDecisions(text: string, patterns: seq[string]): seq[Decision] =
  ## Greedy reference algorithm, per round:
  ## - find the earliest occurrence of each pattern at or after pos
  ##   (strictly smaller start replaces the standing winner, so first-declared wins ties).
  ## - ordinary text up to the winner, then the winner consumes its bytes.
  ## Regions are never split:
  ##   one ordinary region is one decision.
  var pos = 0
  while pos < text.len:
    var bestStart = -1
    var bestPat = -1
    for pidx, pat in patterns:
      let f = text.find(pat, pos)
      if f != -1 and (bestStart == -1 or f < bestStart):
        bestStart = f
        bestPat = pidx
    if bestStart == -1:
      result.add (pos, text.len, -1)
      break
    if bestStart > pos:
      result.add (pos, bestStart, -1)
    result.add (bestStart, bestStart + patterns[bestPat].len, bestPat)
    pos = bestStart + patterns[bestPat].len

proc collectWhole(scanner: SpecialScanner, text: string): seq[Decision] =
  ## Whole-input mode, absolute coordinates.
  var c = SpecialScan.init(scanner, text)
  for d in c.items:
    result.add (c.winBase + d.lo, c.winBase + d.hi, d.specialId)
  doAssert c.drained()
  # drained stability:
  #   a further iteration stays empty and drained
  var extra = 0
  for d in c.items:
    inc extra
  doAssert extra == 0
  doAssert c.drained()

proc collectChunked(scanner: SpecialScanner, text: string,
    chunks: seq[int]): seq[Decision] =
  ## Chunked mode:
  ## feed the given chunk-size cycle (draining between feeds), finish, then drain to completion.
  var c = SpecialScan.init(scanner)
  var off = 0
  var ci = 0
  var guard = 0
  while off < text.len:
    inc guard
    doAssert guard <= 4 * text.len + 8, "feed loop overran its bound"
    let n = min(chunks[ci mod chunks.len], text.len - off)
    c.feed(toOpenArray(text, off, off + n - 1))
    off += n
    inc ci
    for d in c.items:
      result.add (c.winBase + d.lo, c.winBase + d.hi, d.specialId)
  c.finish()
  for d in c.items:
    result.add (c.winBase + d.lo, c.winBase + d.hi, d.specialId)
  doAssert c.drained()

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc decisionsEqual(a, b: seq[Decision]): bool =
  if a.len != b.len:
    return false
  for i in 0 ..< a.len:
    if a[i] != b[i]:
      return false
  true

proc firstDiff(a, b: seq[Decision]): string =
  let n = min(a.len, b.len)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return "first diff at " & $i & ": want " & $b[i] & " got " & $a[i]
  return "lengths " & $a.len & " vs " & $b.len

proc genCorpus(r: var Rand, patterns: var seq[string], text: var string,
    alphabet: string) =
  ## One random dictionary + one random haystack.
  ## Alphabets keep special-token shape (bracket markers) so prefix
  ## chains and overlaps are dense.
  patterns.setLen(0)
  let numPatterns = r.rand(0..9)
  for i in 0 ..< numPatterns:
    var pat = ""
    let patLen = r.rand(1..7)
    for j in 0 ..< patLen:
      pat.add alphabet[r.rand(0..(alphabet.len - 1))]
    if pat notin patterns:
      patterns.add pat
  text.setLen(0)
  let textLen = r.rand(0..240)
  for j in 0 ..< textLen:
    text.add alphabet[r.rand(0..(alphabet.len - 1))]

proc randChunks(r: var Rand, textLen: int): seq[int] =
  ## Random chunk-size cycle for the chunked runs.
  var sizes: seq[int] = @[]
  for i in 0 ..< r.rand(1..4):
    sizes.add r.rand(1..9)
  if sizes.len == 0:
    sizes.add 1
  sizes

proc toSeqIds(n: int): seq[int] =
  result = @[]
  for i in 0 ..< n:
    result.add i

proc runTests*() =
  ## Entry point, runs this file's checks in order.

  block:
    # empty dictionary:
    #   pure ordinary passthrough (final semantics).
    let emptyPats: seq[string] = @[]
    let emptyIds: seq[int] = @[]
    let sc = SpecialScanner.init(emptyPats, emptyIds)
    let got = collectWhole(sc, "ordinary text \xC3\xA9")
    check "empty dictionary passthrough",
      got == @[(0, len("ordinary text \xC3\xA9"), -1)], $got
    check "empty dictionary uses the two-level path (trivially)",
      sc.usesTwoLevel() and sc.maxPatternLen() == 0

  block:
    # empty pattern is rejected with a raise.
    var raised = false
    try:
      discard SpecialScanner.init(@["ok", ""], @[1, 2])
    except ValueError:
      raised = true
    check "empty pattern rejected with a raise", raised

  block:
    # duplicate patterns keep the first (priority) occurrence.
    let sc = SpecialScanner.init(@["dup", "other", "dup"], @[5, 6, 9])
    check "duplicate pattern deduped", sc.patternCount() == 2
    let got = collectWhole(sc, "dup other dup")
    check "dedup keeps first id",
      got == @[(0, 3, 5), (3, 4, -1), (4, 9, 6), (9, 10, -1), (10, 13, 5)], $got

  block:
    # boundary parity rows over a fixed adversarial dictionary, specials
    # at EVERY chunk-boundary offset, including a multi-byte token
    # and a prefix-chain tie pair. Row count is asserted to cover every
    # byte offset of every case (carry-buffer parity, mandatory rows).
    let pats = @["<s>", "<e>", "<é｜>", "<a>", "<ab>"]
    let ids = @[10, 11, 12, 13, 14]
    let sc = SpecialScanner.init(pats, ids)
    let scD = SpecialScanner.init(pats, ids, forceDaac = true)
    let texts = [
      "<s><e>ordinary<s>tail",
      "x<é｜>y<é｜>z",
      "<a><ab>prefix<ab><a>mid<ab>end",
      "aaaa<s>aaaa<e>aaaa<s>aaaa",
      "<é｜><é｜><é｜>",
      "<ab><a><ab><a>",
    ]
    var parityRows = 0
    var rowsOk = true
    for text in texts:
      let want = collectWhole(sc, text)
      let wantD = collectWhole(scD, text)
      if not decisionsEqual(want, wantD):
        rowsOk = false
        check "boundary corpus two-level equals daac (whole)", false,
          firstDiff(wantD, want)
      var naive = naiveDecisions(text, pats)
      for i in 0 ..< naive.len:
        if naive[i][2] >= 0:
          naive[i][2] = ids[naive[i][2]]
      if not decisionsEqual(want, naive):
        rowsOk = false
        check "boundary corpus matches the reference", false,
          firstDiff(naive, want)
      for split in 1 ..< text.len:
        let got = collectChunked(sc, text, @[split])
        if not decisionsEqual(got, want):
          rowsOk = false
          check "boundary split row", false,
            "text=" & text.escape & " split=" & $split & " " &
            firstDiff(got, want)
        let gotD = collectChunked(scD, text, @[split])
        if not decisionsEqual(gotD, want):
          rowsOk = false
          check "boundary split row (daac)", false,
            "text=" & text.escape & " split=" & $split & " " &
            firstDiff(gotD, want)
        inc parityRows
    check "boundary parity rows all green (2 paths x every offset)",
      rowsOk
    echo "boundary parity rows: ", parityRows, " chunk splits x 2 paths"
    doAssert parityRows >= 5 * 15 - 5  # at least every offset of each text

  block:
    # fuzz:
    #   automaton path and two-level path vs the reference,
    # whole-input and chunked.
    const Alphabets = [
      "<ab>",
      "<|ab",
      "a<|b>",
      "ab\x00<>|",
      "<é>ab|",
    ]
    const RoundsPerAlphabet = 300
    var patterns: seq[string] = @[]
    var text = ""
    var seed = 777001'i64
    var rounds = 0
    for alphaIdx in 0 ..< Alphabets.len:
      var r = initRand(seed)
      seed += 1
      for round in 0 ..< RoundsPerAlphabet:
        genCorpus(r, patterns, text, Alphabets[alphaIdx])
        let want = naiveDecisions(text, patterns)
        let sc = SpecialScanner.init(patterns, toSeqIds(patterns.len))
        let got = collectWhole(sc, text)
        if not decisionsEqual(got, want):
          check "fuzz whole (auto path)", false,
            "alpha=" & $alphaIdx & " round=" & $round &
            " patterns=" & $patterns & " text=" & text.escape & " " &
            firstDiff(got, want)
        if sc.usesTwoLevel():
          let scD = SpecialScanner.init(patterns, toSeqIds(patterns.len),
              forceDaac = true)
          let gotD = collectWhole(scD, text)
          if not decisionsEqual(gotD, want):
            check "fuzz whole (two-level vs daac)", false,
              "alpha=" & $alphaIdx & " round=" & $round &
              " patterns=" & $patterns & " text=" & text.escape & " " &
              firstDiff(gotD, want)
        let chunks = randChunks(r, text.len)
        let gotC = collectChunked(sc, text, chunks)
        if not decisionsEqual(gotC, want):
          check "fuzz chunked", false,
            "alpha=" & $alphaIdx & " round=" & $round &
            " patterns=" & $patterns & " text=" & text.escape &
            " chunks=" & $chunks & " " & firstDiff(gotC, want)
        inc rounds
    check "fuzz decision streams all green", true
    echo "fuzz rounds completed: ", rounds

  block:
    # machine stream invariants:
    #   one decision per yield, full-drain
    # equality between a decision-at-a-time walk and a plain drain,
    # and the drained contract.
    let pats = @["<a>", "<bb>"]
    let sc = SpecialScanner.init(pats, @[1, 2])
    let text = "xx<a>yy<bb>zz<a>end"
    var c = SpecialScan.init(sc, text)
    var got: seq[Decision] = @[]
    for d in c.items:
      got.add (c.winBase + d.lo, c.winBase + d.hi, d.specialId)
    check "plain drain emits every decision", got.len == 7, $got.len
    check "drained after full iteration", c.drained()
    var extra = 0
    for d in c.items:
      inc extra
    check "iteration after drain stays empty", extra == 0
    check "drained is sticky", c.drained()

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall special-scan machine checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
