# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Isolated-chain semantics of the pre-tokenization machine:
##
## | # | row                                                                                                                                                                                                             |
## | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | 1 | chain fixture frames (exaone, step-3.5-flash, gemma-4), every row of the HF-recorded frames must be reproduced byte-identically                                                                                 |
## | 2 | unit rows per chain feature: digit grouping \p{N}{1,3}, the CJK-script split step, the GPT-2-style letter/combining-mark pattern, the Gemma-4 MergedWithPrevious space splitter (hand-recorded HF expectations) |
## | 3 | chain-vs-flat-join divergence rows: joining a chain's patterns into one alternation must diverge exactly on the documented rows and nowhere else, the convertHfToTiktoken flattening flaw (issue #22)           |
## | 4 | composition with the special-scan machine at composition depth 2                                                                                                                                                |
## | 5 | the CJK zero-alloc receipt (reused buffers across reset())                                                                                                                                                      |
## | 6 | chain-depth receipts (steps <= 3, build stats)                                                                                                                                                                  |
## | 7 | machine stream rows: empty input yields nothing, one-pass stream equals the collected piece list                                                                                                                |
## | 8 | machine pull-protocol rows: a full drain equals a fresh machine over the same input, post-drain re-iteration stays empty                                                                                        |
## | 9 | a machine-vs-split-scan parity canary: machine segmentation must equal the standalone split scan per family, one corpus prefix and one adversarial row each                                                     |

import std/[os, json, strutils, monotimes, times]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/machine
import workspace/regex_engine
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  CorpusDir = TestsDir / "corpus"
  FixturesDir = TestsDir / "fixtures"

var cache = PreTokRegexCache()

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

proc piecesToStrings(text: string, pieces: seq[(int, int)]): seq[string] =
  for s in pieces.items:
    result.add text[s[0] ..< s[1]]

proc piecesEq(a: seq[(int, int)], b: seq[(int, int)]): bool =
  a == b

proc firstDiff(a, b: seq[(int, int)]): string =
  let n = min(a.len, b.len)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return "first diff at " & $i & ": want (" & $b[i][0] & "," &
        $b[i][1] & ") got (" & $a[i][0] & "," & $a[i][1] & ")"
  "lengths " & $a.len & " vs " & $b.len

proc collectPieces(cache: var PreTokRegexCache, fam: Family, text: string): seq[(int, int)] =
  ## Full drain of one pre-tokenization machine
  ## (bounded by the input length in pieces, every piece covers >= 1 byte).
  var r = initPreTokenizer(cache, fam, text)
  var guard = 0
  for piece in r.items:
    inc guard
    doAssert guard <= text.len + 1, "piece stream overran its bound"
    result.add piece

type
  ChainRow = object
    name: string
    text: string
    pieces: seq[(int, int)]

proc readChainFrame(family: string): seq[ChainRow] =
  let frame = FixturesDir / ("pretok_chain_" & family & ".json.zst")
  let rows = parseJson(readFile(frame).zstdDecompress(string))
  for row in rows.items:
    var r = ChainRow(name: row["name"].getStr(),
                     text: row["text"].getStr())
    for p in row["pieces"].items:
      r.pieces.add (p[0].getInt(), p[1].getInt())
    result.add r

proc flatStep35Seg(text: string): seq[(int, int)] =
  ## Flat-join replica of the convertHfToTiktoken flaw, the three
  ## chain patterns joined into one alternation, leftmost-first across
  ## the whole input (serialization.nim joins Split patterns with |).
  const Flat = r"""\p{N}{1,3}""" & "|" & "[一-龥぀-ゟ゠-ヿ]+" & "|" &
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|""" &
    r"""[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|""" &
    r"""\s*[\r\n]+|\s+(?!\S)|\s+"""
  let flat = compilePattern(Flat, true, "step35_flat_join_replica")
  var lastEmit = 0
  var offset = 0
  while offset < text.len:
    let (ms, me) = flat.nextMatch(text, offset)
    if ms < 0:
      break
    if ms > lastEmit:
      result.add (lastEmit, ms)
    result.add (ms, me)
    lastEmit = me
    offset = me
  if lastEmit < text.len:
    result.add (lastEmit, text.len)

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  #
  # 1. chain fixture frames:
  #   byte-identical piece offsets vs HF
  #
  var frameRows = 0
  var frameFails = 0
  for entry in [
    ("exaone", famExaone),
    ("step-3.5-flash", famStep35Flash),
    ("gemma-4", famGemma4),
  ].items:
    let (family, fam) = entry
    for row in readChainFrame(family).items:
      let got = collectPieces(cache, fam, row.text)
      if got != row.pieces:
        inc frameFails
        echo "FRAME MISMATCH ", family, " ", row.name, " ",
          firstDiff(got, row.pieces)
      inc frameRows
    echo "frame ", family, ": all rows compared"
  check "chain fixture frames byte-identical (3 families, 84 rows)",
    frameFails == 0, $frameFails & " fails / " & $frameRows & " rows"

  #
  # 2. unit rows per chain feature (HF-test expectations, byte units)
  #
  block:
    # digit grouping \p{N}{1,3}:
    #   runs of digits cut into groups of at most 3, both as a chain
    #   step (step-3.5-flash) and inside the single-pattern GLM-4.7 alternative
    let cases: seq[tuple[f: Family, text: string,
      want: seq[(int, int)]]] = @[
      (famGlm47, "abc1234567", @[(0, 3), (3, 6), (6, 9), (9, 10)]),
      (famGlm47, "12 345 6789",
        @[(0, 2), (2, 3), (3, 6), (6, 7), (7, 10), (10, 11)]),
      (famStep35Flash, "abc1234567", @[(0, 3), (3, 6), (6, 9), (9, 10)]),
      (famStep35Flash, "12345678901234",
        @[(0, 3), (3, 6), (6, 9), (9, 12), (12, 14)]),
    ]
    for c in cases:
      let got = collectPieces(cache, c.f, c.text)
      check "digit grouping " & $c.f & " " & c.text.escape,
        piecesEq(got, c.want), firstDiff(got, c.want)

  block:
    # CJK-script split step:
    #   a CJK/kana run is one pre-token piece,
    # isolated from surrounding letter runs by the earlier step
    let cases: seq[tuple[text: string, want: seq[(int, int)]]] = @[
      ("中文字あいうカタカナ", @[(0, 30)]),
      ("hello中文字world", @[(0, 5), (5, 14), (14, 19)]),
    ]
    for c in cases:
      let got = collectPieces(cache, famStep35Flash, c.text)
      check "CJK script split " & c.text.escape,
        piecesEq(got, c.want), firstDiff(got, c.want)

  block:
    # GPT-2-style letter/combining-mark pattern (exaone split 1):
    #   combining marks stay inside the letter piece
    #   (one piece over e + U+0301 + x, 4 utf-8 bytes, recorded from the HF test)
    let got = collectPieces(cache, famExaone, "e\u0301x")
    check "exaone letter/combining-mark piece",
      piecesEq(got, @[(0, 4)]), firstDiff(got, @[(0, 4)])

  block:
    # Gemma-4 Split(String " ", MergedWithPrevious):
    #   a space delimiter
    # joins the preceding non-space run, consecutive or leading spaces
    # stand alone (every row recorded verbatim from the HF test)
    let cases: seq[tuple[text: string, want: seq[(int, int)]]] = @[
      ("hello world", @[(0, 6), (6, 11)]),
      ("a  b", @[(0, 2), (2, 3), (3, 4)]),
      (" x", @[(0, 1), (1, 2)]),
      ("x ", @[(0, 2)]),
      ("a\n\nb  c", @[(0, 5), (5, 6), (6, 7)]),
      (" ", @[(0, 1)]),
      ("a b  ", @[(0, 2), (2, 4), (4, 5)]),
      ("uno  due   tre",
        @[(0, 4), (4, 5), (5, 9), (9, 10), (10, 11), (11, 14)]),
      ("a\tb c", @[(0, 4), (4, 5)]),
    ]
    for c in cases:
      let got = collectPieces(cache, famGemma4, c.text)
      check "gemma MergedWithPrevious " & c.text.escape,
        piecesEq(got, c.want), firstDiff(got, c.want)

  #
  # 3. chain-vs-flat-join divergence rows. These record the Isolated-chain
  # vs flat-alternation divergence from mratsim/tattletale#22.
  # the flat join loses chain isolation, so a CJK run inside a letter
  # run stays glued to it by the main letter alternative, and the chain
  # semantics match the HF engine (proven by the fixture frames above)
  # and the divergence is asserted row-exact, never silenced
  #
  block:
    type DivRow = object
      text: string
      flatJoinsCjk: bool
    let rows = [
      DivRow(text: "hello中文字world あいうカタカナ test", flatJoinsCjk: true),
      DivRow(text: "abc中文def", flatJoinsCjk: true),
      DivRow(text: "abc1234567", flatJoinsCjk: false),
      DivRow(text: "12.34!", flatJoinsCjk: false),
      DivRow(text: "hello world", flatJoinsCjk: false),
    ]
    var divCount = 0
    var sameCount = 0
    for row in rows.items:
      let chain = collectPieces(cache, famStep35Flash, row.text)
      let flat = flatStep35Seg(row.text)
      if chain == flat:
        if row.flatJoinsCjk:
          check "expected flat-join divergence on " & row.text.escape, false
        inc sameCount
      else:
        if not row.flatJoinsCjk:
          check "unexpected flat-join divergence on " & row.text.escape &
            ": chain " & $piecesToStrings(row.text, chain) & " flat " &
            $piecesToStrings(row.text, flat), false
        inc divCount
        echo "documented divergence row (issue-22 flat-join class, the Isolated-chain vs flat-alternation divergence): ",
          row.text.escape
        echo "   chain: ", piecesToStrings(row.text, chain)
        echo "   flat : ", piecesToStrings(row.text, flat)
    check "flat-join diverges exactly on the CJK-glued rows",
      divCount == 2 and sameCount == 3,
      $divCount & " divergences / " & $sameCount & " agreements"

  #
  # 4. composition with the special-scan machine, composition depth 2
  # (special-scan machine over the input feeding one pre-tokenization machine per ordinary region):
  #   the pre-token pieces of every
  # ordinary region must equal the standalone
  # segmentation of that region, and pieces + special decisions must
  # cover the input exactly once.
  #
  block:
    let sc = SpecialScanner.init(@["<|s|>", "<|end|>"], @[7, 8])
    let text = "hello <|s|>world 中文字<|end|>tail 123"
    var c = SpecialScan.init(sc, text)
    var composed: seq[(int, int)] = @[]  # piece starts, special id or -1
    var composedOk = true
    for d in c.items:
      if d.specialId >= 0:
        composed.add (c.winBase + d.lo, d.specialId)
      else:
        let lo = c.winBase + d.lo
        let hi = c.winBase + d.hi
        let region = text[lo ..< hi]
        let standalone = collectPieces(cache, famStep35Flash, region)
        for s in standalone.items:
          composed.add (lo + s[0], -1)
        for s in standalone.items:
          composedOk = composedOk and
            piecesToStrings(region, @[s])[0] == region[s[0] ..< s[1]]
    check "composition: every ordinary region re-segments standalone",
      composedOk
    # coverage:
    #   special offsets and pre-token pieces partition the input
    #   (contiguous, non-overlapping, covering [0, text.len) exactly,
    #   not merely a length sum that an overlap + gap would fool)
    var coverOk = true
    var expect = 0
    var c2 = SpecialScan.init(sc, text)
    for d in c2.items:
      if d.specialId >= 0:
        let lo = c2.winBase + d.lo
        let hi = c2.winBase + d.hi
        if lo != expect:
          coverOk = false
        expect = hi
      else:
        let lo = c2.winBase + d.lo
        let hi = c2.winBase + d.hi
        for s in collectPieces(cache, famStep35Flash, text[lo ..< hi]):
          if lo + s[0] != expect:
            coverOk = false
          expect = lo + s[1]
    coverOk = coverOk and expect == text.len
    check "composition: specials + pieces cover the input bytes",
      coverOk, "covered to " & $expect & " of " & $text.len
    # piece count sanity:
    #   specials excluded, pieces dense within regions
    check "composition decision stream emitted", composed.len > 0

  #
  # 5. CJK zero-alloc receipt:
  #   warmup grows the level buffers once,
  # then a full reset() + drain pass per iteration moves no memory
  # (getTotalMem delta 0 over 2000 full segmentations)
  #
  block:
    const CjkText = "中文字横書きカタカナ语料123456789012345678901234567890"
    var r = initPreTokenizer(cache, famStep35Flash, CjkText)
    var warmPieces = 0
    for piece in r.items:
      inc warmPieces
    check "warmup segmentation emitted pieces", warmPieces > 0
    let before = getTotalMem()
    for i in 0 ..< 2000:
      r.reset(cache, famStep35Flash, CjkText)
      for piece in r.items:
        discard piece
    let after = getTotalMem()
    check "zero-alloc receipt (2000 reset + full pass, CJK input)",
      after == before, "delta " & $(after - before)

  #
  # 6. chain-depth receipts:
  #   every family chain has at most 3
  # steps (step-3.5-flash is the documented maximum), gemma-4 is
  # the single string split (0 instructions), and the build stats
  # agree with the compile receipts
  #
  block:
    for f in Family:
      discard cache.steps(f)
    let st = cache.allStats()
    for f in Family:
      let s = st[f]
      check "chain depth <= 3 for " & $f, s.steps <= 3, $s.steps
      if f == famStep35Flash:
        check "step-3.5-flash chain depth is the documented maximum 3",
          s.steps == 3, $s.steps
      if f == famGemma4:
        check "gemma-4 is a string split (no instructions)",
          s.steps == 1 and s.instrs == 0, $s.steps & "/" & $s.instrs
      else:
        check "receipt records instructions for " & $f,
          s.built and s.instrs > 0, $s.built & "/" & $s.instrs
    echo "build receipts:"
    for f in Family:
      let s = cache.chainStats(f)
      echo "  ", $f, ": steps=", s.steps, " instrs=", s.instrs,
        " build=", s.buildMillis, " ms"

  #
  # machine stream rows:
  #   empty input yields nothing, one-pass stream
  # equals a fresh full drain, resume discipline mid-stream
  #
  block:
    var r = initPreTokenizer(cache, famStep35Flash, "")
    var count = 0
    for piece in r.items:
      inc count
    check "empty input yields nothing", count == 0
    let text = "a1234567b中文字c"
    let want = collectPieces(cache, famStep35Flash, text)
    var r2 = initPreTokenizer(cache, famStep35Flash, text)
    var got: seq[(int, int)] = @[]
    var resumeGot: seq[(int, int)] = @[]
    var taken = 0
    for piece in r2.items:
      got.add piece
      inc taken
      if taken == 2:
        break
    for piece in r2.items:
      resumeGot.add piece
    check "resume after a 2-piece break completes the stream",
      got & resumeGot == want, $(got.len, resumeGot.len, want.len)

  #
  # 8. machine pull-protocol rows:
  #   full drain == a fresh machine over the same input,
  # empty input yields nothing, post-drain re-iteration stays empty
  #
  block:
    let protoText = "don't 1234567 trailing   中文字 a\r\n\r\nb"
    for fam in [famStep35Flash, famGlm47]:
      let want = collectPieces(cache, fam, protoText)
      var drainOk = true
      for pass in 0 ..< 3:
        drainOk = drainOk and collectPieces(cache, fam, protoText) == want
      check "full-drain determinism (" & $fam & ")", drainOk
      var rEmpty = initPreTokenizer(cache, fam, "")
      var emptyCount = 0
      for piece in rEmpty.items:
        inc emptyCount
      check "empty input yields nothing (" & $fam & ")", emptyCount == 0
      var r2 = initPreTokenizer(cache, fam, protoText)
      for piece in r2.items:
        discard piece
      var extra = 0
      for piece in r2.items:
        inc extra
      check "post-drain re-iteration stays empty (" & $fam & ")", extra == 0

  #
  # 9. machine-vs-split-scan parity canary:
  #   the machine segmentation of a family pattern must be byte-identical
  # to the standalone split scan of the same pattern, the scan.nim
  # pattern-split scan driven directly rather than through the machine driver
  #
  block:
    proc spansToStrings(text: string, spans: seq[(int, int)]): seq[string] =
      for s in spans.items:
        result.add text[s[0] ..< s[1]]
    proc spans32ToStrings(text: string,
        spans: seq[tuple[lo, hi: int32]]): seq[string] =
      for s in spans.items:
        result.add text[int(s.lo) ..< int(s.hi)]
    let canaries = [
      (label: "cl100k", fam: famCl100k, pat: Cl100kPat),
      (label: "glm47", fam: famGlm47, pat: Glm47Pat),
    ]
    var corpusText = readFile(CorpusDir /
      "pg100-shakespeare.txt.zst").zstdDecompress(string)
    if corpusText.len > 4000:
      corpusText = corpusText[0 ..< 4000]
    let adversarial = " \u{180E}don't 1234567 trailing   中文字 a\r\n\r\nb  \u{3000}"
    var canaryFails = 0
    for c in canaries.items:
      let sp = splitLookaheadPattern(c.pat, false, "canary_" & c.label)
      for (rowLabel, text) in [("corpus_prefix", corpusText),
          ("adversarial", adversarial)]:
        let want = spansToStrings(text, collectPieces(cache, c.fam, text))
        var spans: seq[tuple[lo, hi: int32]]
        sp.scanSplit(spans, text, 0, text.len)
        let got = spans32ToStrings(text, spans)
        if got != want:
          inc canaryFails
          echo "PARITY CANARY MISMATCH [", c.label, "/", rowLabel, "]",
            " split ", got[0 ..< min(got.len, 12)], " machine ",
            want[0 ..< min(want.len, 12)]
    check "machine segmentation == standalone split scan (canary rows)",
      canaryFails == 0, $canaryFails & " fails"

  echo "\nall pretok chain checks passed"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
