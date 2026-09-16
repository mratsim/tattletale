# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Regex engine semantics vs the bundled PCRE2 engine.
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/regex_engine/tests/t_engine_semantics.nim
##
## Checks:
## - parse and compile determinism, a double compile yields an identical program hash
## - the compile-error model
## - no fixture pattern hits the determinization cap
##
## Segmentation:
## - class bitmap equivalence with classContains, the invalid-byte sentinel matches no class
## - segmentation vs the PCRE2 replica, same compile flags (UTF+UCP) and match flags (NOTEMPTY + NO_UTF_CHECK)
## - frontier DFA == NFA simulation, byte-identical segmentation
##
## Match policy:
## - the first position admitting a non-empty match wins, the scan stops at the first unmatched offset
## - leftmost-first pattern-preference extents
## - the two \s variants, diverging exactly on U+180E

import std/[strutils, monotimes, times]

import workspace/pcre2
import workspace/regex_engine

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

type
  Pcre2Oracle = object
    ## Direct PCRE2 replica of the engine's reference scan.
    ## - same compile flags (UTF+UCP) and match flags
    ##   (NOTEMPTY + NO_UTF_CHECK)
    ## - gap-before-match emission, the scan stops at the first
    ##   unmatched offset, the remainder is one trailing piece
    code: ptr Code
    matchData: ptr MatchData
    ovector: ptr UncheckedArray[int]

proc initOracle(pattern: string): Pcre2Oracle =
  var err: CompileError
  var errOff: csize_t
  result.code = compile(pattern, flag(UTF, UCP), err, errOff)
  if result.code == nil:
    raise newException(ValueError, "oracle compile failed: " & pattern)
  result.matchData = match_data_create_from_pattern(result.code, nil)
  result.ovector = get_ovector_pointer(result.matchData)

proc oracleSeg(o: Pcre2Oracle, text: string): seq[(int, int)] =
  var lastPos = 0
  var offset = 0
  while offset < text.len:
    let rc = match(o.code, text, offset,
      flag(NOTEMPTY, MatchOption.NO_UTF_CHECK), o.matchData, nil)
    if rc == -1:
      break
    doAssert rc > 0
    let ms = o.ovector[0].int
    let me = o.ovector[1].int
    if ms >= text.len or me > text.len:
      break
    if ms > lastPos:
      result.add (lastPos, ms)
    result.add (ms, me)
    lastPos = me
    offset = me
  if lastPos < text.len:
    result.add (lastPos, text.len)

proc engineSeg(p: CompiledPattern, text: string,
    useNfa = false): seq[(int, int)] =
  ## Gap-and-match segmentation through one scan driver.
  var lastPos = 0
  var offset = 0
  while offset < text.len:
    let (ms, me) = (if useNfa: p.nextMatchNfa(text, offset)
                    else: p.nextMatch(text, offset))
    if ms < 0:
      break
    if ms > lastPos:
      result.add (lastPos, ms)
    result.add (ms, me)
    lastPos = me
    offset = me
  if lastPos < text.len:
    result.add (lastPos, text.len)

proc firstMatch(p: CompiledPattern, text: string): tuple[start, stop: int] =
  var m = initRegexScanner(p, text)
  for s in m.items:
    return s
  (-1, -1)

proc firstDiff(a, b: seq[(int, int)]): string =
  let n = min(a.len, b.len)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return "first diff at " & $i & ": want (" & $b[i][0] & "," &
        $b[i][1] & ") got (" & $a[i][0] & "," & $a[i][1] & ")"
  "lengths " & $a.len & " vs " & $b.len

proc progHash(p: CompiledPattern): string =
  for ins in p.prog.items:
    result.add $ord(ins.kind) & ":"
    case ins.kind
    of iChar, iLookNeg:
      result.add $ins.cls & "," & $ins.next
    of iSplit:
      result.add $ins.a & "," & $ins.b
    of iDollar:
      result.add $ins.after
    of iMatch:
      discard

proc lcg(state: var uint64): uint64 =
  state = state * 6364136223846793005'u64 + 1442695040888963407'u64
  state

proc genRandom(state: var uint64): string =
  ## Deterministic adversarial strings over a pattern-dense alphabet:
  ## - caseless-fold partners, whitespace, CJK, combining marks, digits
  ## - escapes, the U+180E whitespace divergence
  const Alphabet = [
    "a", "Z", "1", "9", " ", "\t", "\n", "\r\n", "'", "s", "L", "S", "D",
    "M", "T", "V", "R", "E", "l", "v", "r", "e", "d", "m", "t", "/", "\\",
    "中", "文", "あ", "ン", "。", "é", "é\u0301", "😀", "🇺", "\u{180E}",
    "\u{00A0}", "\u{017F}", "\u{3000}", "\u{2028}", "_", "-", ".", "0",
  ]
  let n = 1 + int(lcg(state) mod 40)
  result = ""
  for i in 0 ..< n:
    result.add Alphabet[int(lcg(state) mod Alphabet.len)]

const
  FuzzRounds = 4000
  TwinRounds = 2000

  # Engine-representative fixture patterns, one label per pattern.
  # Together they exercise every supported construct:
  # - alternation, concatenation, non-capturing + caseless groups
  # - both lookahead forms, the $ assertion
  # - \p{...} classes with complements and ranges
  # - every quantifier shape incl. possessive and counted
  FixturePatterns = [
    ("alt_caseless_lookahead_possessive",
     r"(?i:'s|'t|'re)|\p{L}+|\p{N}+|\s++$|\s+(?!\S)|\s"),
    ("digit_grouping", r"\p{N}{1,3}"),
    ("han_lookahead_lo",
     r"(?!\p{Script=Han})\p{Lo}|[\p{Lt}\p{Lu}\p{Lm}\p{M}]"),
    ("han_run", r"[\p{Script=Han}]+"),
    ("punct_class_ranges", r"[!\[\\\]^_`{|}~][A-Za-z]+"),
    ("counted_overlap", r"a{2,4}|ab|a"),
    ("caseless_class", r"(?i:[st])(?:'s)?|\p{M}+"),
  ]

  # Hand-recorded text rows. Expected segmentation is PCRE2 output,
  # the explicit-expectation blocks below carry their own literals.
  TextRows = [
    "ab  ", "ab \n c", "a\r\n\r\nb", "  leading", "trailing   ",
    "don't Can't ITS st're", "中文abc De\u0301 1234  ",
    "a\u180Eb", "\u180E \u00A0", "1234567", "a12345b678",
    "aab", "abab", "aaaab", "x\ny", "x\n", "x", "x\n\n", "\n\n",
    "!!foo!!bar", "~Az09^", "sS\u{1E9E}t", "é\u0301é", "word--_word",
    "", " ", "\n", "a", "中",
  ]

proc main() =
  var compiled: seq[tuple[label: string, pat: CompiledPattern]]
  var oracles: seq[tuple[label: string, oracle: Pcre2Oracle]]
  for (label, pat) in FixturePatterns.items:
    compiled.add (label, compilePattern(pat, false, label))
    oracles.add (label, initOracle(pat))

  #
  # double-compile determinism, the program bytes must hash identically
  #
  var detFails = 0
  for i, (label, pat) in compiled.pairs:
    if progHash(pat) != progHash(compilePattern(FixturePatterns[i][1])):
      inc detFails
      echo "DETERMINISM FAIL ", label
  check "double-compile determinism (" & $compiled.len & " patterns)",
    detFails == 0, $detFails & " fails"

  #
  # class bitmap == classContains over the ASCII plane + a band above,
  # and the invalid-byte sentinel fails every class
  #
  var bitmapFails = 0
  var bitmapProbes = 0
  for (label, pat) in compiled.items:
    for cls in 0 ..< pat.classes.len:
      let classId = int32(cls)
      for cp in 0'u32 .. 0x2FF'u32:
        if pat.classHas(classId, cp) !=
            pat.classes[cls].classContains(cp):
          inc bitmapFails
          echo "BITMAP MISMATCH ", label, " class=", cls,
            " cp=0x", cp.toHex()
        inc bitmapProbes
      if pat.classHas(classId, 0xFFFFFFFF'u32):
        inc bitmapFails
        echo "BITMAP MISMATCH ", label, " class=", cls,
          " sentinel cp matched"
  check "class bitmap == classContains (ASCII plane + band above)",
    bitmapFails == 0, $bitmapProbes & " probes, " & $bitmapFails & " fails"

  #
  # leftmost-first pattern preference, the first alternative's extent
  # wins even when a later alternative matches longer, a longest-match
  # machine would report the other extent
  #
  block:
    let pref = compilePattern(r"a|ab")
    let got = firstMatch(pref, "ab")
    check "leftmost-first extent wins over the longest extent",
      got == (0, 1), $got

  #
  # NOTEMPTY, a zero-width acceptance is discarded and the scan continues
  #
  block:
    var neFails = 0
    let ws = compilePattern(r"\s*")
    var m1 = initRegexScanner(ws, "abc")
    var c1 = 0
    for s in m1.items:
      inc c1
    if c1 != 0:
      inc neFails
      echo "NOTEMPTY: \\s* on 'abc' yielded ", c1, " matches"
    var m2 = initRegexScanner(ws, "  ab")
    var s2: seq[tuple[start, stop: int]]
    for s in m2.items:
      s2.add s
    if s2 != @[(0, 2)]:
      inc neFails
      echo "NOTEMPTY: \\s* on '  ab' yielded ", s2
    check "NOTEMPTY rows (zero-width acceptance discarded)", neFails == 0

  #
  # the $ assertion matches end of subject or before a final LF
  # (PCRE2 default newline mode), never strictly inside the subject
  #
  block:
    var dollarFails = 0
    let p = compilePattern(r"x$")
    if firstMatch(p, "x\n") != (0, 1):
      inc dollarFails; echo "$ row x\\n: want (0,1)"
    if firstMatch(p, "x") != (0, 1):
      inc dollarFails; echo "$ row x: want (0,1)"
    if firstMatch(p, "x\ny") != (-1, -1):
      inc dollarFails; echo "$ row x\\ny: want no match"
    if firstMatch(p, "x\n\n") != (-1, -1):
      inc dollarFails; echo "$ row x\\n\\n: want no match"
    if firstMatch(p, "x\r\n") != (-1, -1):
      inc dollarFails; echo "$ row x\\r\\n: want no match (LF newline mode)"
    check "$ assertion rows (end of subject or before a final LF)",
      dollarFails == 0

  #
  # the anchored attempt driver, match end at a fixed codepoint
  # boundary with window-bounded dollar semantics
  #
  block:
    var anchoredFails = 0
    let run = compilePattern(r"\s+\s")
    if run.matchAt("  x", 0, 3, 0) != 2:
      inc anchoredFails; echo "matchAt s+s '  x' pos 0: want 2"
    if run.matchAt("  x", 0, 3, 1) != -1:
      inc anchoredFails; echo "matchAt s+s '  x' pos 1: want no match"
    if run.matchAt(" x", 0, 2, 0) != -1:
      inc anchoredFails; echo "matchAt s+s ' x': want no match"
    let tail = compilePattern(r"\s+$")
    if tail.matchAt("ab  ", 0, 4, 2) != 4:
      inc anchoredFails; echo "matchAt s+$ trailing run: want 4"
    if tail.matchAt("ab  cd", 0, 6, 2) != -1:
      inc anchoredFails; echo "matchAt s+$ interior run: want no match"
    if tail.matchAt("ab \ncd", 0, 6, 2) != -1:
      inc anchoredFails; echo "matchAt s+$ non-final LF: want no match"
    # the dollar binds at the window end, not the input end, a caller
    # scanning a slice gets per-slice haystack semantics
    if tail.matchAt("ab  cd", 0, 4, 2) != 4:
      inc anchoredFails; echo "matchAt s+$ window [0,4) of 'ab  cd': want 4"
    check "anchored matchAt rows (fixed start, window-bounded dollar)",
      anchoredFails == 0

  #
  # the two \s variants diverge exactly on U+180E, in the PCRE2 UCP
  # table but dropped from White_Space in Unicode 6.3
  #
  block:
    var wsFails = 0
    let pcre = compilePattern(r"\s")
    let rust = compilePattern(r"\s", whitespaceIsRust = true)
    for cp in [0x180E'u32, 0x00A0'u32]:
      for p in [pcre, rust]:
        # bitmap predicate and range scan must agree on both test values
        if p.classHas(0'i32, cp) != p.classes[0].classContains(cp):
          inc wsFails; echo "BITMAP MISMATCH on \\s variant cp=0x",
            cp.toHex()
    if not pcre.classes[0].classContains(0x180E'u32):
      inc wsFails; echo "PCRE2 \\s lost U+180E"
    if rust.classes[0].classContains(0x180E'u32):
      inc wsFails; echo "Rust \\s gained U+180E"
    if not pcre.classes[0].classContains(0x00A0'u32) or
        not rust.classes[0].classContains(0x00A0'u32):
      inc wsFails; echo "NBSP must sit in both variants"
    check "the two \\s variants diverge exactly on U+180E", wsFails == 0

  #
  # PCRE2 parity, engine segmentation == PCRE2 replica over every
  # fixture pattern, hand-recorded rows + bounded adversarial fuzz
  #
  var fails = 0
  var cases = 0
  proc compareRow(label: string, i: int, text: string) =
    let want = engineSeg(compiled[i].pat, text)
    let got = oracleSeg(oracles[i].oracle, text)
    if want != got:
      inc fails
      echo "PCRE2 ORACLE MISMATCH [", compiled[i].label, "] ", label
      echo "  oracle: ", got[0 ..< min(got.len, 12)], " ",
        firstDiff(want, got)
      echo "  engine: ", want[0 ..< min(want.len, 12)]
    inc cases

  for i in 0 ..< compiled.len:
    for j, text in TextRows.pairs:
      compareRow("row_" & $j, i, text)
  echo "oracle rows: ", compiled.len, " patterns x ", TextRows.len,
    " hand-recorded texts"

  var state: uint64 = 0x243F6A8885A308D3'u64
  for iter in 0 ..< FuzzRounds:
    let text = genRandom(state)
    for i in 0 ..< compiled.len:
      compareRow("fuzz_" & $iter, i, text)
  echo "oracle fuzz: ", FuzzRounds, " rounds x ", compiled.len, " patterns"
  check "segmentation == PCRE2 oracle (all fixture patterns, rows)",
    fails == 0, $fails & " fails / " & $cases & " cases"

  #
  # engine parity check, frontier-DFA scan == NFA-simulation scan
  # with byte-identical segmentation per fixture pattern
  #
  var twinFails = 0
  var twinCases = 0
  for i in 0 ..< compiled.len:
    for j, text in TextRows.pairs:
      let viaDfa = engineSeg(compiled[i].pat, text)
      let viaNfa = engineSeg(compiled[i].pat, text, useNfa = true)
      if viaDfa != viaNfa:
        inc twinFails
        echo "ENGINE TWIN MISMATCH [", compiled[i].label, "] row_", $j,
          " ", firstDiff(viaDfa, viaNfa)
      inc twinCases
    var twinState: uint64 = 0xBB67AE8584CAA73B'u64
    for iter in 0 ..< TwinRounds:
      let text = genRandom(twinState)
      let viaDfa = engineSeg(compiled[i].pat, text)
      let viaNfa = engineSeg(compiled[i].pat, text, useNfa = true)
      if viaDfa != viaNfa:
        inc twinFails
        echo "ENGINE TWIN MISMATCH [", compiled[i].label, "] fuzz_", $iter
      inc twinCases
  echo "twin rows: ", twinCases, " (", compiled.len,
    " patterns x ", TextRows.len, " rows + ", TwinRounds, " fuzz rounds)"
  check "frontier DFA == NFA simulation (byte-identical segmentation)",
    twinFails == 0, $twinFails & " fails / " & $twinCases & " twin cases"

  #
  # compile-error model, unsupported constructs fail loudly at compile
  # time with PatternCompileError, never silently
  #
  block:
    var errFails = 0
    proc expectCompileError(pat: string, what: string) =
      try:
        discard compilePattern(pat)
        inc errFails
        echo "COMPILE ERROR MODEL: '", pat, "' (", what,
          ") compiled without error"
      except PatternCompileError:
        discard
    expectCompileError(r"a+?", "lazy quantifier")
    expectCompileError(r"a\q", "unsupported escape")
    expectCompileError(r"[ab", "unterminated bracket class")
    expectCompileError(r"[b-a]", "reversed class range")
    expectCompileError(r"*a", "quantifier with nothing to quantify")
    expectCompileError(r"a{3,2}", "reversed counted repetition")
    expectCompileError(r"(?=a)", "positive lookahead unsupported")
    expectCompileError(r"(?", "unterminated group prefix")
    check "compile-error model (unsupported constructs rejected)",
      errFails == 0

  #
  # build receipts, every fixture pattern compiled within the packed
  # transition key width and no determinization overflow
  #
  block:
    var overflow = false
    for (label, pat) in compiled.items:
      let (states, edges) = pat.dfaStats()
      let mem = pat.dfaMemoryEstimate()
      overflow = overflow or pat.dfaOverflow
      echo "  ", label, ": dfaStates=", states, " dfaEdges=", edges,
        " memEstBytes=", mem, " overflow=", pat.dfaOverflow
    check "no fixture pattern hit the determinization cap", not overflow

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall regex engine semantics checks passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 /
    1000.0, " s"
