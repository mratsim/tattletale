# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## RegexScanner machine conformance (object + ctor + ONE items):
## scan-order match spans, window subscans (matches never cross the
## window bounds), resume discipline (cursor advances before each
## yield, so partial consumption resumes exactly), post-drain
## emptiness, and parity with the raw nextMatch loop including
## interleaved scanners sharing one compiled pattern's scratch
## buffers.

import std/[strutils, monotimes, times]

import workspace/regex_engine

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc drain(p: CompiledPattern, text: string, winLo = 0,
    winHi = -1): seq[tuple[start, stop: int]] =
  var m = initRegexScanner(p, text, winLo, winHi)
  for s in m.items:
    result.add s

proc drainRaw(p: CompiledPattern, text: string, winLo = 0,
    winHi = -1): seq[tuple[start, stop: int]] =
  ## The manual scan-driver loop the machine wraps.
  let hi = if winHi < 0: text.len else: winHi
  var offset = winLo
  while offset < hi:
    let (ms, me) = p.nextMatch(text, offset, winLo, hi)
    if ms < 0:
      break
    result.add (ms, me)
    offset = me

const
  TextRows = [
    "ab12 c", "xx12yy34", "don't stop  12 ", "中文 abc 123 中文",
    "", " ", "abc", "123", "a1b2c3", "  padded  ",
  ]

proc main() =
  let p = compilePattern(r"\p{L}+|\p{N}+")

  #
  # machine == manual loop over every row
  #
  var parityFails = 0
  for i, text in TextRows.pairs:
    let viaMachine = drain(p, text)
    let viaRaw = drainRaw(p, text)
    if viaMachine != viaRaw:
      inc parityFails
      echo "SCANNER PARITY MISMATCH row_", $i, ": ", viaMachine,
        " vs ", viaRaw
  check "RegexScanner items == raw nextMatch loop (all rows)",
    parityFails == 0

  #
  # scan-order spans: a known row with literal expectation
  #
  block:
    let got = drain(p, "ab12 c")
    check "scan-order spans on 'ab12 c'",
      got == @[(0, 2), (2, 4), (5, 6)], $got

  #
  # window subscans: matches never start before winLo and never
  # extend past winHi
  #
  block:
    var winFails = 0
    for i, text in TextRows.pairs:
      for lo in 0 .. text.len:
        for hi in lo .. text.len:
          for (ms, me) in drain(p, text, lo, hi):
            if ms < lo or me > hi or ms >= me:
              inc winFails
              echo "WINDOW VIOLATION row_", $i, " [", lo, ",", hi,
                "): match (", ms, ",", me, ")"
    check "window subscans stay inside [winLo, winHi) (all rows x bounds)",
      winFails == 0

  #
  # window subscan equals scanning the sliced text at rebased offsets
  # (the window is a pure restriction, never a re-anchor)
  #
  block:
    var equivFails = 0
    for i, text in TextRows.pairs:
      if text.len < 3:
        continue
      let lo = 1
      let hi = text.len - 1
      var want: seq[tuple[start, stop: int]]
      let sub = text[lo ..< hi]
      for (ms, me) in drainRaw(p, sub):
        want.add (ms + lo, me + lo)
      if drain(p, text, lo, hi) != want:
        inc equivFails
        echo "WINDOW EQUIV MISMATCH row_", $i
    check "window subscan == rebased slice scan (all rows)", equivFails == 0

  #
  # resume discipline: the cursor advances before each yield, so a
  # consumer that stops after k spans resumes exactly there
  #
  block:
    var resumeFails = 0
    for i, text in TextRows.pairs:
      let full = drain(p, text)
      for k in 0 ..< max(full.len, 1):
        var m = initRegexScanner(p, text)
        var head: seq[tuple[start, stop: int]]
        var n = 0
        for s in m.items:
          head.add s
          inc n
          if n == k:
            break
        var tail: seq[tuple[start, stop: int]]
        for s in m.items:
          tail.add s
        if head & tail != full:
          inc resumeFails
          echo "RESUME MISMATCH row_", $i, " k=", $k, ": ",
            (head & tail), " vs ", full
    check "resume after partial consumption (all rows x break points)",
      resumeFails == 0

  #
  # post-drain re-iteration stays empty; empty input yields nothing
  #
  block:
    var drainFails = 0
    var m = initRegexScanner(p, "ab12 c")
    for s in m.items:
      discard
    var extra = 0
    for s in m.items:
      inc extra
    if extra != 0:
      inc drainFails
      echo "post-drain re-iteration yielded ", extra, " spans"
    var mEmpty = initRegexScanner(p, "")
    var emptyCount = 0
    for s in mEmpty.items:
      inc emptyCount
    if emptyCount != 0:
      inc drainFails
      echo "empty input yielded ", emptyCount, " spans"
    check "post-drain emptiness + empty input", drainFails == 0

  #
  # scratch sharing: two interleaved scanners on one compiled pattern
  # (the pattern's attempt buffers are reused per attempt) must each
  # reproduce their own full drain
  #
  block:
    var interFails = 0
    let q = compilePattern(r"\s+(?!\S)|\s+")
    for i, text in TextRows.pairs:
      var ma = initRegexScanner(p, text)
      var mb = initRegexScanner(q, text)
      var sa, sb: seq[tuple[start, stop: int]]
      while true:
        var progressed = false
        for s in ma.items:
          sa.add s
          progressed = true
          break
        for s in mb.items:
          sb.add s
          progressed = true
          break
        if not progressed:
          break
      if sa != drain(p, text):
        inc interFails
        echo "INTERLEAVE MISMATCH (letters) row_", $i, ": ", sa
      if sb != drain(q, text):
        inc interFails
        echo "INTERLEAVE MISMATCH (spaces) row_", $i, ": ", sb
    check "interleaved scanners sharing one pattern keep their own drains",
      interFails == 0

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall regex engine scanner checks passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 /
    1000.0, " s"
