# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Pre-tokenization machine rows, refereed against this repo's
## own implementations and recorded fixtures.
## - per single-pattern family
##   (r50k, p50k, cl100k, o200k, kimik25, moonlight, qwen, qwen35, glm47, ling3) the machine segmentation must be byte-identical
##   to the standalone split scan of the same family pattern
##   (splitLookaheadPattern, the scan.nim pattern-split scan driven directly rather than through the machine driver),
## - the chain-fixture corpus texts
##   (the 4 shared corpus prefixes plus the adversarial rows of the HF chain frames),
## - corpus prefixes of 三國志演義 / verne / shakespeare / sqlite3.c,
## - deterministic adversarial LCG rows over a pattern-dense alphabet,
## - plus the frontier-DFA vs NFA-simulation equality check (12 families),
## - a pull-protocol conformance block (capacity-driven walk, drain discipline),
## - loud build + scan timing receipts (heaviest family called out).
##
## Recorded-fixture referee for the chain families
## (machine pieces == the recorded frame pieces) lives in tests/unit/t_pretok_chain.nim, engine-level
## property checks (double-compile determinism, class-bitmap equivalence) live in workspace/regex_engine/tests.

import std/[os, random, strutils, monotimes, times, json]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/machine
import workspace/regex_engine
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  CorpusDir = TestsDir / "corpus"
  FixturesDir = TestsDir / "fixtures"
  FuzzRounds = 20000

type
  FamilyRow = object
    label: string
    fam: Family
    pat: string
    sp: SplitPattern      # standalone split scan of the family pattern

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

proc collectPieces(fam: Family, text: string): seq[(int, int)] =
  ## Full drain of one pre-tokenization machine
  ## (bounded by the input length in pieces, every piece covers >= 1 byte).
  var r = initPreTokenizer(fam, text)
  var guard = 0
  for piece in r.items:
    inc guard
    doAssert guard <= text.len + 1, "piece stream overran its bound"
    result.add piece

proc dfaSeg(f: Family, text: string): seq[string] =
  spansToStrings(text, collectPieces(f, text))

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

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc readFixtureTexts(): seq[string] =
  ## Texts of every row of every chain fixture frame
  ## (corpus prefixes plus adversarial rows), read whole:
  ## the fuzz comparison is not hot-path code.
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
  ## (contractions, whitespace, CJK, combining marks, digits, slashes).
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

proc main() =
  var families = [
    FamilyRow(label: "r50k", fam: famR50k, pat: R50kPat),
    FamilyRow(label: "p50k", fam: famP50k, pat: R50kPat),
    FamilyRow(label: "cl100k", fam: famCl100k, pat: Cl100kPat),
    FamilyRow(label: "o200k", fam: famO200k, pat: O200kPat),
    FamilyRow(label: "kimik25", fam: famKimiK25, pat: KimiK25Pat),
    FamilyRow(label: "moonlight", fam: famMoonlight, pat: MoonlightPat),
    FamilyRow(label: "qwen", fam: famQwen, pat: QwenPat),
    FamilyRow(label: "qwen35", fam: famQwen35, pat: Qwen35Pat),
    FamilyRow(label: "glm47", fam: famGlm47, pat: Glm47Pat),
    FamilyRow(label: "ling3", fam: famLing3, pat: Ling3Pat),
  ]

  for row in families.mitems:
    row.sp = splitLookaheadPattern(row.pat, false, row.label)

  var fails = 0
  var cases = 0

  proc compareRow(label: string, row: FamilyRow, text: string) =
    let want = dfaSeg(row.fam, text)
    var spans: seq[tuple[lo, hi: int32]]
    row.sp.scanSplit(text, 0, text.len, spans)
    let got = spans32ToStrings(text, spans)
    if got != want:
      inc fails
      echo "MACHINE-SPLIT MISMATCH [", row.label, "] ", label
      echo "  split:  ", got[0 ..< min(got.len, 12)]
      echo "  machine: ", want[0 ..< min(want.len, 12)]
    inc cases

  #
  # engine equality:
  #   every regex step of a family chain can also be driven
  # by the NFA-simulation scan driver (nextMatchNfa), the reference
  # driver of the default frontier-DFA walk (both drivers in this repo).
  # The helpers below apply a chain with the NFA driver so the engine
  # comparison is a true engine-to-engine equality over identical chain semantics.
  #
  proc applyStepNfa(pat: CompiledPattern, text: string, pieceLo, pieceHi: int,
      outPieces: var seq[(int, int)]) =
    var lastEmit = pieceLo
    var offset = pieceLo
    while offset < pieceHi:
      let (ms, me) = pat.nextMatchNfa(text, offset, pieceLo, pieceHi)
      if ms < 0:
        break
      if ms > lastEmit:
        outPieces.add (lastEmit, ms)
      outPieces.add (ms, me)
      lastEmit = me
      offset = me
    if lastEmit < pieceHi:
      outPieces.add (lastEmit, pieceHi)

  proc applySpaceStepNfa(text: string, pieceLo, pieceHi: int,
      outPieces: var seq[(int, int)]) =
    ## Split(String " ", MergedWithPrevious) replica, shared
    ## verbatim by both engines (no regex surface, nothing to determinize)
    var runStart = pieceLo
    var i = pieceLo
    while i < pieceHi:
      if text[i] == ' ':
        if i == pieceLo or text[i - 1] == ' ':
          if runStart < i:
            outPieces.add (runStart, i)
          outPieces.add (i, i + 1)
        else:
          outPieces.add (runStart, i + 1)
        runStart = i + 1
      inc i
    if runStart < pieceHi:
      outPieces.add (runStart, pieceHi)

  proc segmentsViaNfa(f: Family, text: string): seq[(int, int)] =
    ## Chain application with every regex step driven by the NFA
    ## simulation scan driver, the byte-identical counterpart of the machine
    if familySteps(f).len == 0 or text.len == 0:
      if text.len > 0:
        result.add (0, text.len)
      return
    var level = @[(0, text.len)]
    for step in familySteps(f).items:
      var nxt: seq[(int, int)]
      for piece in level.items:
        case step.kind
        of skRegex:
          applyStepNfa(step.pat, text, piece[0], piece[1], nxt)
        of skSpaceMergedPrev:
          applySpaceStepNfa(text, piece[0], piece[1], nxt)
      level = system.move(nxt)
    result = system.move(level)

  # every regex family of the stage, the NFA pass adds the exaone
  # and step-3.5 chains the machine-vs-split rows do not carry
  const TwinFamilies = [
    ("r50k", famR50k), ("p50k", famP50k), ("cl100k", famCl100k),
    ("o200k", famO200k), ("kimik25", famKimiK25),
    ("moonlight", famMoonlight), ("qwen", famQwen), ("qwen35", famQwen35),
    ("glm47", famGlm47), ("ling3", famLing3), ("exaone", famExaone),
    ("step35flash", famStep35Flash),
  ]

  #
  # 1. corpus prefixes (120k chars total across 4 corpora)
  #
  let corpora = [
    ("sanguozhi", CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 20000),
    ("verne", CorpusDir /
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 20000),
    ("shakespeare", CorpusDir / "pg100-shakespeare.txt.zst", 30000),
    ("sqlite", CorpusDir / "sqlite3.c.zst", 50000),
  ]
  var corpusChars = 0
  var corpusTexts: seq[string]
  var corpusNames: seq[string]
  for (cname, path, maxBytes) in corpora.items:
    let text = readCorpusPrefix(path, maxBytes)
    corpusChars += text.len
    corpusTexts.add text
    corpusNames.add cname
    for row in families.items:
      compareRow("corpus_" & cname, row, text)
  echo "corpus: ", corpusChars, " chars x ", families.len, " families"

  #
  # 2. chain-fixture texts (corpus prefixes + adversarial rows)
  #
  let fixtureTexts = readFixtureTexts()
  for i, text in fixtureTexts.pairs:
    for row in families.items:
      compareRow("fixture_" & $i, row, text)
  echo "fixture texts: ", fixtureTexts.len, " x ", families.len, " families"

  #
  # 4. engine equality:
  #   frontier-DFA scan vs NFA-simulation scan over the same corpora,
  #   fixture texts and adversarial fuzz rows, byte identical
  #   segmentation per family (exaone and step-3.5 chains included)
  #
  var twinFails = 0
  var twinCases = 0
  proc compareTwin(label: string, fam: Family, text: string) =
    let viaDfa = spansToStrings(text, collectPieces(fam, text))
    let viaNfa = spansToStrings(text, segmentsViaNfa(fam, text))
    if viaDfa != viaNfa:
      inc twinFails
      echo "ENGINE TWIN MISMATCH [", label, "] ", fam
    inc twinCases

  for (label, fam) in TwinFamilies.items:
    for i, text in corpusTexts.pairs:
      compareTwin("corpus_" & corpusNames[i], fam, text)
    for i, text in fixtureTexts.pairs:
      compareTwin("fixture_" & $i, fam, text)
    var twinState: uint64 = 0x243F6A8885A308D3'u64
    for iter in 0 ..< FuzzRounds:
      compareTwin("fuzz_" & $iter, fam, genRandom(twinState))
  echo "twin rows: ", twinCases, " (", TwinFamilies.len,
    " families x 4 corpora + 84 fixtures + 20000 fuzz rows)"
  check "frontier DFA == NFA simulation (byte-identical segmentation)",
    twinFails == 0, $twinFails & " fails / " & $twinCases & " twin cases"

  #
  # 3. adversarial LCG fuzz
  #
  var state: uint64 = 0x243F6A8885A308D3'u64
  for iter in 0 ..< FuzzRounds:
    let text = genRandom(state)
    for row in families.items:
      compareRow("fuzz_" & $iter, row, text)
  echo "fuzz: ", FuzzRounds, " rounds x ", families.len, " families"
  check "machine segmentation == standalone split scan (all families, rows)",
    fails == 0, $fails & " fails / " & $cases & " cases"

  #
  # machine stream conformance:
  #   full drain == fresh machine per pass,
  # capacity-free walk (the machine shape has no dst buffer), empty
  # input yields nothing, post-drain re-iteration stays empty
  #
  var pullFails = 0
  let protoText = "don't 1234567 trailing   中文字 a\r\n\r\nb"
  for row in families.items:
    let want = collectPieces(row.fam, protoText)
    for pass in 0 ..< 3:
      let got = collectPieces(row.fam, protoText)
      if got != want:
        inc pullFails
        echo "MACHINE DRAIN MISMATCH [", row.label, "] pass ", pass, " ",
          spansToStrings(protoText, got), " vs ",
          spansToStrings(protoText, want)
    var rEmpty = initPreTokenizer(row.fam, "")
    var emptyCount = 0
    for piece in rEmpty.items:
      inc emptyCount
    if emptyCount != 0:
      inc pullFails
      echo "empty input yielded ", emptyCount, " pieces [", row.label, "]"
    var r2 = initPreTokenizer(row.fam, protoText)
    for piece in r2.items:
      discard piece
    var extra = 0
    for piece in r2.items:
      inc extra
    if extra != 0:
      inc pullFails
      echo "post-drain re-iteration yielded ", extra, " pieces [", row.label, "]"
  check "machine stream conformance (drain, empty, post-drain)", pullFails == 0,
    $pullFails & " fails"

  #
  # loud timing receipts:
  #   compile cost per family (first use happens during the corpus scans above) and full-corpus scan cost
  # per family, heaviest scan family called out
  #
  var heaviest = ""
  var heaviestMs = 0.0
  var allText = ""
  for text in corpusTexts.items:
    allText.add text
  for row in families.items:
    var buildMs = chainStats(row.fam).buildMillis
    let t0 = getMonoTime()
    let pieces = collectPieces(row.fam, allText)
    let scanMs = (getMonoTime() - t0).inMicroseconds.float64 / 1000.0
    let tn0 = getMonoTime()
    discard segmentsViaNfa(row.fam, allText)
    let scanNfaMs = (getMonoTime() - tn0).inMicroseconds.float64 / 1000.0
    echo "timing ", row.label, ": compile ", buildMs, " ms, scanDfa ",
      scanMs, " ms, scanNfa ", scanNfaMs, " ms (", pieces.len,
      " pieces over ", allText.len, " bytes)"
    if scanMs > heaviestMs:
      heaviestMs = scanMs
      heaviest = row.label
  echo "heaviest scan family: ", heaviest, " at ", heaviestMs, " ms"
  echo "loud build receipts (instrs per family):"
  for st in allFamilyStats():
    echo "  instrs=", st.instrs, " steps=", st.steps
  echo "frontier DFA build receipts (states/edges after the scans above):"
  var anyOverflow = false
  for (label, fam) in TwinFamilies.items:
    var states = 0
    var edges = 0
    var memEst = 0
    for step in familySteps(fam).items:
      if step.kind == skRegex:
        let (st, ed) = dfaStats(step.pat)
        states += st
        edges += ed
        memEst += dfaMemoryEstimate(step.pat)
        anyOverflow = anyOverflow or step.pat.dfaOverflow
    echo "  ", label, ": dfaStates=", states, " dfaEdges=", edges,
      " memEstBytes=", memEst, " overflow=", anyOverflow
  check "no family pattern hit the determinization cap", not anyOverflow

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall pretok DFA fuzz checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
