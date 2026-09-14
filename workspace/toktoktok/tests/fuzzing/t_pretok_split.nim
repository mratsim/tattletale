# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Pattern-split pre-tokenization rows (scan.nim pattern-split section):
## - the split scan of every lookahead-bearing family pattern must be
##   byte-identical to the frontier-engine scan of the same family
##   pattern (both engines in this repo, the referee every served segmentation is checked against)
##   on 241,056 chain rows (12 families x 4 corpora + 84 chain-fixture texts + 20,000 adversarial LCG rows).
## - the whitespace adversarial matrix, every row named with its hazard class
##   (end-anchor at a window boundary, priority-order segmentation, drop-last resume).
## - hand-computed segmentation rows.
##
## A further instrument answers the split's simplification question
## with evidence, per family:
## - whether pat1's leftmost-first segmentation equals a longest-match
##   scan over pat1's alternatives (both driven by this repo's engine).
## The verdict is carried by the scan.nim pattern-split section
## (the longest-match contract), this suite is the standing proof.

import std/[os, strutils, monotimes, times, algorithm, json]

import workspace/zstd/zstd_highlevel
import workspace/regex_engine
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
  CorpusDir = TestsDir / "corpus"
  FixturesDir = TestsDir / "fixtures"
  FuzzRounds = 20000
  CollapseFuzzRounds = 2000

type
  FamilyRow = object
    label: string
    fam: Family
    pat: string
    rustWs: bool
    sp: SplitPattern

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc spansToStrings(text: string, spans: seq[(int, int)]): seq[string] =
  for s in spans.items:
    result.add text[s[0] ..< s[1]]

proc spans32ToStrings(text: string,
    spans: seq[tuple[lo, hi: int32]]): seq[string] =
  for s in spans.items:
    result.add text[int(s.lo) ..< int(s.hi)]

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc readFixtureTexts(): seq[string] =
  for family in ["exaone", "step-3.5-flash", "gemma-4"]:
    let frame = FixturesDir / ("pretok_chain_" & family & ".json.zst")
    let rows = parseJson(readFile(frame).zstdDecompress(string))
    for row in rows.items:
      result.add row["text"].getStr()

proc lcg(state: var uint64): uint64 =
  state = state * 6364136223846793005'u64 + 1442695040888963407'u64
  state

proc genRandom(state: var uint64): string =
  ## Deterministic adversarial strings over a pattern-dense alphabet
  ## (contractions, whitespace, CJK, combining marks, digits, slashes, the \s-variant divergence codepoint U+180E).
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

#
# chain drivers:
#   every regex step through one engine or the other,
# the Isolated chain applied level by level over the same pieces
#
proc applyStepFrontier(pat: CompiledPattern, text: string,
    pieceLo, pieceHi: int, outPieces: var seq[tuple[lo, hi: int32]]) =
  var lastEmit = pieceLo
  var offset = pieceLo
  while offset < pieceHi:
    let (ms, me) = pat.nextMatch(text, offset, pieceLo, pieceHi)
    if ms < 0:
      break
    if ms > lastEmit:
      outPieces.add (int32(lastEmit), int32(ms))
    outPieces.add (int32(ms), int32(me))
    lastEmit = me
    offset = me
  if lastEmit < pieceHi:
    outPieces.add (int32(lastEmit), int32(pieceHi))

proc applyStepSplit(step: SplitStep, text: string,
    pieceLo, pieceHi: int, outPieces: var seq[tuple[lo, hi: int32]]) =
  if step.split != nil:
    step.split.scanSplit(text, pieceLo, pieceHi, outPieces)
  else:
    applyStepFrontier(step.pat, text, pieceLo, pieceHi, outPieces)

proc applySpaceStep(text: string, pieceLo, pieceHi: int,
    outPieces: var seq[tuple[lo, hi: int32]]) =
  ## Split(String " ", MergedWithPrevious) replica, shared verbatim
  ## by both engines (no regex surface, nothing to determinize).
  var runStart = pieceLo
  var i = pieceLo
  while i < pieceHi:
    if text[i] == ' ':
      if i == pieceLo or text[i - 1] == ' ':
        if runStart < i:
          outPieces.add (int32(runStart), int32(i))
        outPieces.add (int32(i), int32(i + 1))
      else:
        outPieces.add (int32(runStart), int32(i + 1))
      runStart = i + 1
    inc i
  if runStart < pieceHi:
    outPieces.add (int32(runStart), int32(pieceHi))

proc applyChain(f: Family, text: string, viaSplit: bool): seq[(int, int)] =
  ## Family Isolated chain over the whole input, every regex step
  ## through the frontier engine or (where the step carries the split)
  ## the split scan.
  if familySteps(f).len == 0 or text.len == 0:
    if text.len > 0:
      result.add (0, text.len)
    return
  var level: seq[tuple[lo, hi: int32]] = @[(0'i32, int32(text.len))]
  for step in familySteps(f).items:
    var nxt: seq[tuple[lo, hi: int32]]
    for piece in level.items:
      if step.kind == skRegex:
        if viaSplit:
          applyStepSplit(step, text, piece.lo, piece.hi, nxt)
        else:
          applyStepFrontier(step.pat, text, piece.lo, piece.hi, nxt)
      else:
        applySpaceStep(text, piece.lo, piece.hi, nxt)
    level = system.move(nxt)
  for s in level.items:
    result.add (int(s.lo), int(s.hi))

proc main() =
  var families = [
    FamilyRow(label: "r50k", fam: famR50k, pat: R50kPat, rustWs: false),
    FamilyRow(label: "p50k", fam: famP50k, pat: R50kPat, rustWs: false),
    FamilyRow(label: "cl100k", fam: famCl100k, pat: Cl100kPat, rustWs: false),
    FamilyRow(label: "o200k", fam: famO200k, pat: O200kPat, rustWs: false),
    FamilyRow(label: "kimik25", fam: famKimiK25, pat: KimiK25Pat, rustWs: false),
    FamilyRow(label: "moonlight", fam: famMoonlight, pat: MoonlightPat, rustWs: false),
    FamilyRow(label: "qwen", fam: famQwen, pat: QwenPat, rustWs: false),
    FamilyRow(label: "qwen35", fam: famQwen35, pat: Qwen35Pat, rustWs: false),
    FamilyRow(label: "glm47", fam: famGlm47, pat: Glm47Pat, rustWs: false),
    FamilyRow(label: "ling3", fam: famLing3, pat: Ling3Pat, rustWs: false),
  ]
  const ChainFamilies = [
    ("exaone", famExaone, ExaoneStepPat, true),
    ("step35flash", famStep35Flash, Step35MainPat, true),
  ]
  for row in families.mitems:
    row.sp = splitLookaheadPattern(row.pat, row.rustWs, row.label)

  let corpora = [
    ("sanguozhi", CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 20000),
    ("verne", CorpusDir /
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 20000),
    ("shakespeare", CorpusDir / "pg100-shakespeare.txt.zst", 30000),
    ("sqlite", CorpusDir / "sqlite3.c.zst", 50000),
  ]
  var corpusTexts: seq[string]
  var corpusNames: seq[string]
  var allText = ""
  for (cname, path, maxBytes) in corpora.items:
    let text = readCorpusPrefix(path, maxBytes)
    corpusTexts.add text
    corpusNames.add cname
    allText.add text
  let fixtureTexts = readFixtureTexts()

  #
  # 1. the 241,056-row equality block:
  #   split scan == frontier engine, byte identical,
  #    12 families x (4 corpora + 84 fixtures + 20,000 fuzz)
  #
  var twinFails = 0
  var twinCases = 0
  proc compareTwin(label: string, fam: Family, text: string) =
    let want = spansToStrings(text, applyChain(fam, text, viaSplit = false))
    let got = spansToStrings(text, applyChain(fam, text, viaSplit = true))
    if want != got:
      inc twinFails
      if twinFails <= 3:
        echo "SPLIT TWIN MISMATCH [", label, "] text: ",
          text.replace("\n", "\\n").replace("\r", "\\r")
        echo "  frontier: ", want[0 ..< min(want.len, 16)]
        echo "  split:    ", got[0 ..< min(got.len, 16)]
    inc twinCases

  for (label, fam, _, _) in ChainFamilies.items:
    for i, text in corpusTexts.pairs:
      compareTwin("corpus_" & corpusNames[i], fam, text)
    for i, text in fixtureTexts.pairs:
      compareTwin("fixture_" & $i, fam, text)
    var state: uint64 = 0x243F6A8885A308D3'u64
    for iter in 0 ..< FuzzRounds:
      compareTwin("fuzz_" & $iter, fam, genRandom(state))
  for row in families.items:
    for i, text in corpusTexts.pairs:
      compareTwin("corpus_" & corpusNames[i], row.fam, text)
    for i, text in fixtureTexts.pairs:
      compareTwin("fixture_" & $i, row.fam, text)
    var state: uint64 = 0x243F6A8885A308D3'u64
    for iter in 0 ..< FuzzRounds:
      compareTwin("fuzz_" & $iter, row.fam, genRandom(state))
  echo "twin rows: ", twinCases, " (12 families x 4 corpora + 84 fixtures + ",
    FuzzRounds, " fuzz rows)"
  check "split scan == frontier engine (byte-identical segmentation)",
    twinFails == 0, $twinFails & " fails / " & $twinCases & " twin cases"

  #
  # 3. whitespace adversarial matrix:
  #   every row through the paired scan,
  #    each row named with its hazard class
  #
  var matrixFails = 0
  var matrixCases = 0
  proc compareMatrixRow(hazard, name: string, row: FamilyRow, text: string) =
    let want = spansToStrings(text, applyChain(row.fam, text, viaSplit = false))
    let got = spansToStrings(text, applyChain(row.fam, text, viaSplit = true))
    if want != got:
      inc matrixFails
      echo "MATRIX MISMATCH [", hazard, "/", name, "] text: ",
        text.replace("\n", "\\n").replace("\r", "\\r")
      echo "  frontier: ", want
      echo "  split:    ", got
    inc matrixCases

  proc compareMatrixChainRow(hazard, name: string, fam: Family, text: string) =
    let want = spansToStrings(text, applyChain(fam, text, viaSplit = false))
    let got = spansToStrings(text, applyChain(fam, text, viaSplit = true))
    if want != got:
      inc matrixFails
      echo "MATRIX MISMATCH [", hazard, "/", name, "] text: ",
        text.replace("\n", "\\n").replace("\r", "\\r")
      echo "  frontier: ", want
      echo "  split:    ", got
    inc matrixCases

  proc compareMatrixAll(hazard, name: string, text: string) =
    for (clabel, cfam, _, _) in ChainFamilies.items:
      compareMatrixChainRow(hazard, name & "/" & clabel, cfam, text)
    for row in families.items:
      compareMatrixRow(hazard, name & "/" & row.label, row, text)

  # H1 (end-anchor at a window boundary):
  #   a run reaching the window end
  # takes the pat1-`$` path inside that window and never across it,
  # scanned whole, cut at the run end, and cut inside the run
  for k in [1, 2, 3, 7, 16, 64]:
    let run = repeat(" ", k)
    compareMatrixAll("H1-window-end-run", "k=" & $k, "ab" & run)
    compareMatrixAll("H1-window-end-run-crlf", "k=" & $k, "ab" & run & "\n")
    compareMatrixAll("H1-window-end-tab-run", "k=" & $k, "ab" & repeat("\t", k))
    let mid = "ab" & run & "cd"
    let cut = 2 + k
    compareMatrixAll("H1-region-left-of-run-end", "k=" & $k, mid[0 ..< cut])
    compareMatrixAll("H1-region-right-of-run-end", "k=" & $k,
      mid[cut ..< mid.len])
    compareMatrixAll("H1-region-cut-inside-run", "k=" & $k,
      mid[0 ..< cut - 1])
    compareMatrixAll("H1-region-resume-at-dropped", "k=" & $k,
      mid[cut - 1 ..< mid.len])
  # H3 (drop-last resume):
  #   an interior run of k spaces before a word,
  # the pat2 drop drops the last space and the scan resumes at it
  for k in 1 .. 64:
    compareMatrixAll("H3-drop-last-resume", "k=" & $k,
      "w" & repeat(" ", k) & "x")
    compareMatrixAll("H3-drop-last-resume-tab", "k=" & $k,
      "w" & repeat("\t", k) & "x")
    compareMatrixAll("WS-run-before-digit", "k=" & $k,
      "w" & repeat(" ", k) & "7")
  # H2 (priority order):
  #   a trailing run must segment as ONE piece via
  # the pat1-`$` path, never as the pat2+pat3 pair
  for k in 1 .. 64:
    compareMatrixAll("H2-priority-trailing-run", "k=" & $k,
      "w" & repeat(" ", k))
    compareMatrixAll("H2-priority-trailing-tab-run", "k=" & $k,
      "w" & repeat("\t", k))
  # CRLF and mixed-whitespace runs
  for k in 1 .. 8:
    compareMatrixAll("WS-crlf-run-head", "k=" & $k, repeat("\r\n", k) & "x")
    compareMatrixAll("WS-crlf-run-tail", "k=" & $k, "x" & repeat("\r\n", k))
    compareMatrixAll("WS-crlf-run-mid", "k=" & $k,
      "ab" & repeat(" ", k) & "\r\n" & repeat(" ", k) & "cd")
    compareMatrixAll("WS-lf-run", "k=" & $k, "ab" & repeat("\n", k) & "cd")
  compareMatrixAll("WS-mixed-tab-lf", "row", "a \n\t b")
  compareMatrixAll("WS-mixed-space-tab-crlf", "row", " \t \r\n \n\t x")
  compareMatrixAll("WS-lf-run-window-end", "row", "a\n\n\nb")
  compareMatrixAll("WS-vertical-tab-formfeed", "row", "a\x0B\x0C b")
  # \s-variant and multi-byte whitespace codepoints:
  #   the engine path
  # must honor the family's \s variant (U+180E is PCRE2 \s but not Rust White_Space, U+00A0 / U+3000 / U+2028 are multi-byte \s)
  compareMatrixAll("WS-variant-U180E-single", "row", " \u{180E} x")
  compareMatrixAll("WS-variant-U180E-run", "row", "ab  \u{180E}\u{180E} y")
  compareMatrixAll("WS-variant-U180E-alternating", "row",
    "\u{180E} \u{180E} z")
  compareMatrixAll("WS-nbsp-run", "row", "  \u{00A0} x")
  compareMatrixAll("WS-ideographic-space-run", "row", "\u{3000}\u{3000}x")
  compareMatrixAll("WS-line-sep-mix", "row", " \u{2028}\n x")
  echo "matrix rows: ", matrixCases
  check "whitespace adversarial matrix (split == frontier)",
    matrixFails == 0, $matrixFails & " fails / " & $matrixCases & " rows"

  #
  # 4. hand-computed segmentation rows:
  #    the split scan must produce
  #    the exact recorded pieces
  #
  block:
    let named = [
      (label: "H2 trailing run one piece (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a   ", want: @["a", "   "]),
      (label: "H3 drop-last resume (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a   b", want: @["a", "  ", " b"]),
      (label: "single space joins word (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a b", want: @["a", " b"]),
      (label: "single space before digit (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a 1", want: @["a", " ", "1"]),
      (label: "H2 trailing run one piece (r50k)", pat: R50kPat,
        rustWs: false, text: "a   ", want: @["a", "   "]),
      (label: "H3 drop-last resume (r50k)", pat: R50kPat,
        rustWs: false, text: "a   b", want: @["a", "  ", " b"]),
      (label: "single space joins digit run (r50k)", pat: R50kPat,
        rustWs: false, text: "a 1", want: @["a", " 1"]),
      (label: "tab run interior (r50k)", pat: R50kPat,
        rustWs: false, text: "a\t\t\tb", want: @["a", "\t\t", "\t", "b"]),
      (label: "CRLF mix (r50k)", pat: R50kPat,
        rustWs: false, text: "ab \n cd", want: @["ab", " \n", " cd"]),
      (label: "H3 drop-last resume (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: "a   b", want: @["a", "  ", " b"]),
      (label: "H2 trailing run one piece (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: "a   ", want: @["a", "   "]),
      (label: "space joins Han run (o200k)", pat: O200kPat,
        rustWs: false, text: "a 三國 b", want: @["a", " 三國", " b"]),
      (label: "CRLF run to window end (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "ab \n", want: @["ab", " \n"]),
      (label: "multi-byte ws run interior (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "  \u{00A0} x", want: @["  \u{00A0}", " x"]),
      (label: "U+180E is ws, PCRE2 variant (kimi)", pat: KimiK25Pat,
        rustWs: false, text: " \u{180E} x", want: @[" \u{180E}", " x"]),
      (label: "U+180E not ws, Rust variant (exaone)", pat: ExaoneStepPat,
        rustWs: true, text: " \u{180E} x", want: @[" \u{180E}", " x"]),
      (label: "tab after LF stays its own piece (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a \n\t b", want: @["a", " \n", "\t", " b"]),
      (label: "LF run reaches window end (kimi)", pat: KimiK25Pat,
        rustWs: false, text: "a\n\n\nb", want: @["a", "\n\n\n", "b"]),
    ]
    var namedFails = 0
    for row in named.items:
      let sp = splitLookaheadPattern(row.pat, row.rustWs, "named")
      var spans: seq[tuple[lo, hi: int32]]
      sp.scanSplit(row.text, 0, row.text.len, spans)
      let got = spans32ToStrings(row.text, spans)
      if got != row.want:
        inc namedFails
        echo "NAMED ROW MISMATCH [", row.label, "]: split ", got,
          " want ", row.want
    check "hand-computed segmentation rows (split == recorded pieces)",
      namedFails == 0, $namedFails & " fails / " & $named.len & " rows"

  #
  # 5. the simplification question, answered with evidence:
  #    per family, pat1's leftmost-first segmentation vs a longest-match
  #    scan over pat1's own alternatives
  #    (each alternative compiled standalone, anchored greedy extent, max over alternatives).
  #    The verdict is recorded in the scan.nim pattern-split section.
  #    A flip forces a re-answer of that contract.
  #
  block:
    const ExpectedCollapse = [
      ("r50k", true), ("p50k", true), ("cl100k", false),
      ("o200k", false), ("kimik25", false), ("moonlight", false),
      ("qwen", false), ("qwen35", false), ("glm47", false),
      ("ling3", false), ("exaone", false), ("step35main", false),
    ]
    proc leftmostSeg(p: CompiledPattern, text: string): seq[(int, int)] =
      var offset = 0
      var lastEmit = 0
      while offset < text.len:
        let (ms, me) = p.nextMatch(text, offset, 0, text.len)
        if ms < 0:
          break
        if ms > lastEmit:
          result.add (lastEmit, ms)
        result.add (ms, me)
        lastEmit = me
        offset = me
      if lastEmit < text.len:
        result.add (lastEmit, text.len)

    proc longestSeg(alts: seq[CompiledPattern],
        text: string): seq[(int, int)] =
      var offset = 0
      var lastEmit = 0
      let hi = text.len
      while offset < hi:
        var best = -1
        for alt in alts.items:
          let me = alt.matchAt(text, 0, hi, offset)
          if me > best:
            best = me
        if best < 0:
          let d = decodeCp(text, offset, hi)
          inc offset, (if d.width > 0: d.width else: 1)
          continue
        if offset > lastEmit:
          result.add (lastEmit, offset)
        result.add (offset, best)
        lastEmit = best
        offset = best
      if lastEmit < hi:
        result.add (lastEmit, hi)

    for (label, expected) in ExpectedCollapse.items:
      let pat = if label == "r50k" or label == "p50k": R50kPat
        elif label == "cl100k": Cl100kPat
        elif label == "o200k": O200kPat
        elif label == "kimik25": KimiK25Pat
        elif label == "moonlight": MoonlightPat
        elif label == "qwen": QwenPat
        elif label == "qwen35": Qwen35Pat
        elif label == "glm47": Glm47Pat
        elif label == "ling3": Ling3Pat
        elif label == "exaone": ExaoneStepPat
        else: Step35MainPat
      let rustWs = label == "exaone"
      let p1 = compilePattern(pat1AlternativeStrings(pat).join("|"),
        rustWs, label & "_p1")
      var alts: seq[CompiledPattern]
      for i, alt in pat1AlternativeStrings(pat).pairs:
        alts.add compilePattern(alt, rustWs, label & "_alt" & $i)
      var fails = 0
      var cases = 0
      var firstDiv = ""
      for text in corpusTexts.items:
        let a = spansToStrings(text, leftmostSeg(p1, text))
        let b = spansToStrings(text, longestSeg(alts, text))
        inc cases
        if a != b and fails == 0:
          for i, x in a.pairs:
            if i >= b.len or b[i] != x:
              firstDiv = "corpus: leftmost[" & $i & "]=" & x & " longest[" &
                $i & "]=" & (if i < b.len: b[i] else: "<end>")
              break
      var state: uint64 = 0x243F6A8885A308D3'u64
      for iter in 0 ..< CollapseFuzzRounds:
        let text = genRandom(state)
        let a = spansToStrings(text, leftmostSeg(p1, text))
        let b = spansToStrings(text, longestSeg(alts, text))
        inc cases
        if a != b:
          inc fails
          if firstDiv == "":
            firstDiv = "fuzz " & $iter & ": " & text.replace("\n", "\\n") &
              " leftmost " & $a & " longest " & $b
      echo "collapse ", label, ": leftmost-first == longest-match over ",
        cases, " rows? ", (if fails == 0: "YES" else: "NO (" & $fails &
        " divergent rows)"), (if fails > 0: "  first: " & firstDiv else: "")
      check "collapse verdict (" & label & ") recorded",
        (fails == 0) == expected

  #
  # 6. loud receipts:
  #   split-scan wall vs frontier-engine wall per family
  #    over the 120k allText (median of 11), the build-time
  #    capability tests, and the deterministic-compile state
  #    of the sub-patterns
  #
  block:
    proc median(xs: var seq[float]): float =
      xs.sort()
      xs[xs.len div 2]

    proc scanWall(sp: SplitPattern, text: string): float =
      var spans: seq[tuple[lo, hi: int32]]
      let t0 = getMonoTime()
      sp.scanSplit(text, 0, text.len, spans)
      (getMonoTime() - t0).inMicroseconds.float64 / 1000.0

    proc frontierWall(f: Family, text: string): float =
      let t0 = getMonoTime()
      discard applyChain(f, text, viaSplit = false)
      (getMonoTime() - t0).inMicroseconds.float64 / 1000.0

    var chainSplits: array[2, SplitPattern]
    for i, (clabel, _, cpat, crw) in ChainFamilies.pairs:
      chainSplits[i] = splitLookaheadPattern(cpat, crw, clabel)
      echo "probes ", clabel, ": interiorPlainCapable=",
        chainSplits[i].interiorPlainCapable, " interiorCrlfCapable=",
        chainSplits[i].interiorCrlfCapable
    for row in families.items:
      echo "probes ", row.label, ": interiorPlainCapable=",
        row.sp.interiorPlainCapable, " interiorCrlfCapable=",
        row.sp.interiorCrlfCapable
    var anyOverflow = false
    for sp in chainSplits.items:
      for p in [sp.pat1, sp.pat2, sp.pat3]:
        anyOverflow = anyOverflow or p.dfaOverflow
    for row in families.items:
      for p in [row.sp.pat1, row.sp.pat2, row.sp.pat3]:
        anyOverflow = anyOverflow or p.dfaOverflow
    check "no split sub-pattern hit the determinization cap", not anyOverflow

    for i, (clabel, cfam, _, _) in ChainFamilies.pairs:
      var ws, wf: seq[float]
      for k in 0 ..< 12:
        let a = scanWall(chainSplits[i], allText)
        let b = frontierWall(cfam, allText)
        if k > 0:
          ws.add a
          wf.add b
      echo "timing ", clabel, ": split ",
        median(ws).formatFloat(ffDecimal, 3), " ms, frontier ",
        median(wf).formatFloat(ffDecimal, 3), " ms (120k bytes)"
    for row in families.items:
      var ws, wf: seq[float]
      for k in 0 ..< 12:
        let a = scanWall(row.sp, allText)
        let b = frontierWall(row.fam, allText)
        if k > 0:
          ws.add a
          wf.add b
      echo "timing ", row.label, ": split ",
        median(ws).formatFloat(ffDecimal, 3), " ms, frontier ",
        median(wf).formatFloat(ffDecimal, 3), " ms (120k bytes)"

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall pretok split referee checks passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 /
    1000.0, " s"
