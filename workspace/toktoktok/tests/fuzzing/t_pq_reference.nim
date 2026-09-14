# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Priority-queue reference-implementation suite
## (tests/fuzzing/pq_bpe_reference.nim, the test-side merge core with no served path).
##
## Checks, per family with a staged rank table
## (r50k, p50k, cl100k, o200k, kimik2.5):
## - recorded fixtures, the ordinary path (whole-piece rank hit first, else the PQ merge)
##   reproduces the id streams recorded from the tiktoken library over
##   the same rank files
##   (tests/fixtures/small/tiktoken_*.json.zst, no specials).
## - recorded tie rows, on synthetic colliding-rank rank tables the PQ
##   merge reproduces the id streams recorded from the naive-merge
##   specification implementation while that referee existed
##   (tests/fixtures/pq_reference_ties.json.zst, ordinary and merge-core streams per row)
##   - the leftmost-tie contract on duplicate ranks has no in-tree
##   referee left, so its verdicts are recorded.
## - 1-byte double-add, the merge core emits the recorded double-add
##   structure (id twice for a 1-byte piece),
##   engine history:
##     warm == cold == fresh-engine id streams,
##   machine walk:
##     piece-by-piece consumption of the pre-tokenization
##     machine == the full ordinary drain,
##   structure check:
##     PairIndex merge path == the pair-validation-cache path,
##     id streams byte-identical,
##   merge tables:
##     load receipts + SplitTable structural invariants.
## Over the 4 corpora (三國志演義 CJK-dense, verne, shakespeare, sqlite3.c), chain + special-pretokenization fixture texts,
## engineered tie-storm rows, and deterministic adversarial LCG rows
## (the invariance and structure-check rows). Plus the zero-alloc receipt.

import std/[os, strutils, algorithm, monotimes, times, tables, json]
import pkg/jsony

import workspace/zstd/zstd_highlevel
import pq_bpe_reference
import workspace/toktoktok/src/merge
import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/scan
import workspace/toktoktok/src/machine

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  TokenizersDir = TestsDir / "tokenizers"
  CorpusDir = TestsDir / "corpus"
  FixturesDir = TestsDir / "fixtures"
  FuzzRounds = 20000

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc lcg(state: var uint64): uint64 =
  state = state * 6364136223846793005'u64 + 1442695040888963407'u64
  state

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc frameTexts(frame: string): seq[string] =
  let rows = parseJson(readFile(FixturesDir / frame).zstdDecompress(string))
  for row in rows.items:
    result.add row["text"].getStr()

proc firstDiff(a, b: seq[int]): int =
  for i in 0 ..< min(a.len, b.len):
    if a[i] != b[i]:
      return i
  -1

proc collectPieces(fam: Family, text: string): seq[(int, int)] =
  ## Full drain of one pre-tokenization machine
  ## (bounded by the input length in pieces, every piece covers >= 1 byte).
  var r = initPreTokenizer(fam, text)
  var guard = 0
  for piece in r.items:
    inc guard
    doAssert guard <= text.len + 1, "piece stream overran its bound"
    result.add piece

# ------------------------------------------------------------------------
# per-piece encoders over one segmentation
# ------------------------------------------------------------------------

proc pqCoreIds(engine: BpeEngine, text: string,
    pieces: seq[(int, int)]): seq[int] =
  var m = MergeBuf.init()
  for s in pieces.items:
    bytePairEncodePQ(engine, m, text.toOpenArrayByte(0, text.len - 1), s[0], s[1], result)

proc pqCoreIdsLegacy(engine: BpeEngine, text: string,
    pieces: seq[(int, int)]): seq[int] =
  var m = MergeBuf.init()
  for s in pieces.items:
    bytePairEncodePQLegacy(engine, m, text.toOpenArrayByte(0, text.len - 1), s[0], s[1], result)

proc pqOrdIds(engine: BpeEngine, text: string,
    pieces: seq[(int, int)]): seq[int] =
  var m = MergeBuf.init()
  for s in pieces.items:
    encodeSegmentPQ(engine, m, text.toOpenArrayByte(0, text.len - 1), s[0], s[1], result)

proc machineIds(engine: BpeEngine, fam: Family, text: string): seq[int] =
  ## Piece-by-piece consumption of the pre-tokenization machine, one
  ## piece in flight, encodeSegmentPQ per piece into a reused scratch,
  ## exactly the consumption pattern of the pipeline machine.
  var r = initPreTokenizer(fam, text)
  var m = MergeBuf.init()
  var scratch: seq[int] = @[]
  for piece in r.items:
    scratch.setLen(0)
    encodeSegmentPQ(engine, m, text.toOpenArrayByte(0, text.len - 1),
      piece[0], piece[1], scratch)
    for x in scratch.items:
      result.add x

# Check selector:
#   arg1 = check number (1..8, default all), arg2 = family
# index into `families` (0..4, applied to the family-loop checks).
# One check at a time fits inside a 120 s referee window and each check
# reports its own wall time. "all" runs the full suite unchanged.

proc gateActive(gate: string): bool =
  let args = commandLineParams()
  result = args.len < 1 or args[0] == "all" or args[0] == gate

proc famActive(famIdx: int): bool =
  let args = commandLineParams()
  if args.len < 2 or args[1] == "all":
    return true
  result = parseInt(args[1]) == famIdx

type
  RecordedFixture = object
    name: string
    text: string
    tokenIds: seq[int]
    tokenizer: string

  RecordedTie = object
    name: string
    text: string
    ordinaryIds: seq[int]
    coreIds: seq[int]

proc genFuzz(state: var uint64): string =
  const Alphabet = [
    "a", "b", "c", "d", "e", " ", " ", "1", "2", "9", ".", ",", "\n",
    "\r\n", "\t", "'", "s", "t", "he", "th", "er", "中", "文", "の",
    "😀", "\u{00A0}", "_", "-", "0", "L", "L", "L",
  ]
  let n = 1 + int(lcg(state) mod 48)
  result = ""
  for i in 0 ..< n:
    result.add Alphabet[int(lcg(state) mod Alphabet.len)]

proc main() =
  if not fileExists(TokenizersDir / "gpt2-tokenizer.json"):
    echo "MISSING tokenizer data: ", TokenizersDir
    echo "stage it with: python3 ",
      TestsDir /
      "fetch_test_tokenizers.py"
    quit(1)

  var families = [
    (label: "r50k", fam: famR50k, pat: R50kPat, file: "r50k_base.tiktoken",
     recorded: "tiktoken_r50k_base.json.zst"),
    (label: "p50k", fam: famP50k, pat: R50kPat, file: "p50k_base.tiktoken",
     recorded: "tiktoken_p50k_base.json.zst"),
    (label: "cl100k", fam: famCl100k, pat: Cl100kPat, file: "cl100k_base.tiktoken",
     recorded: "tiktoken_cl100k_base.json.zst"),
    (label: "o200k", fam: famO200k, pat: O200kPat, file: "o200k_base.tiktoken",
     recorded: "tiktoken_o200k_base.json.zst"),
    (label: "kimik2.5", fam: famKimiK25, pat: KimiK25Pat, file: "kimik2.5.tiktoken",
     recorded: "tiktoken_kimik2.5.json.zst"),
  ]

  let corpora = [
    ("sanguozhi", CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 20000),
    ("verne", CorpusDir /
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 20000),
    ("shakespeare", CorpusDir / "pg100-shakespeare.txt.zst", 30000),
    ("sqlite", CorpusDir / "sqlite3.c.zst", 50000),
  ]

  var fixtureTexts: seq[string] = @[]
  for frame in ["pretok_chain_exaone.json.zst",
                "pretok_chain_step-3.5-flash.json.zst",
                "pretok_chain_gemma-4.json.zst",
                "special_pretok_kimik2.5.json.zst",
                "special_pretok_step-3.5-flash.json.zst",
                "special_pretok_exaone.json.zst"]:
    for t in frameTexts(frame):
      fixtureTexts.add t

  let tieStorms = [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "11111111111111111111",
    "                    ",
    "abababababababababababababababab",
    "the the the the the the the the",
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
    "................................",
    "a1b2c3d4e5f6g7h8a1b2c3d4e5f6g7h8",
    "\xE4\xB8\xAD\xE6\x96\x87\xE4\xB8\xAD\xE6\x96\x87\xE4\xB8\xAD\xE6\x96\x87",
    "\xE4\xB8\xAD\xE6\x96\x87 123 456 789 \xE4\xB8\xAD\xE6\x96\x87",
    "\xEF\xBB\xBFleading bom \xEF\xBB\xBFagain",
    "a\r\nb\r\nc\r\n\r\n\r\n\r\nd",
    "((((((((((()))))))))))",
    "0123456789012345678901234567890123456789",
    "                    x",
    "x                    ",
  ]

  var corpusTexts: seq[string] = @[]
  for (cname, path, maxBytes) in corpora.items:
    corpusTexts.add readCorpusPrefix(path, maxBytes)

  var sharedTexts: seq[string] = @[]
  for t in corpusTexts.items:
    sharedTexts.add t
  for t in fixtureTexts.items:
    sharedTexts.add t
  for t in tieStorms.items:
    sharedTexts.add t
  block:
    var state: uint64 = 0x243F6A8885A308D3'u64
    for iter in 0 ..< FuzzRounds:
      sharedTexts.add genFuzz(state)

  var totalIds = 0
  block gate1:
    # recorded fixtures:
    #   the PQ ordinary path reproduces the id streams
    # recorded from the tiktoken library per rank file
    if not gateActive("1"):
      break gate1
    let bt0 = getMonoTime()
    for fIdx in 0 ..< families.len:
      let row = families[fIdx]
      if not famActive(fIdx):
        continue
      let ft0 = getMonoTime()
      let tok = loadTiktokenCodec(TokenizersDir / row.file)
      let engine = BpeEngine.init(tok.ranks)
      engine.ensureBuilt() # loud lazy-build receipt prints here
      let content = readFile(FixturesDir / "small" / row.recorded).
        zstdDecompress(string)
      let recorded = content.fromJson(seq[RecordedFixture])
      var fails = 0
      var ids = 0
      for fx in recorded.items:
        let pieces = collectPieces(row.fam, fx.text)
        let pOrd = pqOrdIds(engine, fx.text, pieces)
        if pOrd != fx.tokenIds:
          inc fails
          if fails <= 3:
            echo "RECORDED FAIL [", row.label, "] ", fx.name,
              " first diff ", firstDiff(fx.tokenIds, pOrd)
        ids += fx.tokenIds.len
      check "ordinary path == recorded tiktoken ids [" & row.label & "]",
        fails == 0, $fails & " fails / " & $ids & " ids / " &
        $recorded.len & " rows"
      totalIds += ids
      echo "timing [gate1/", row.label, "] wall ",
        (getMonoTime() - ft0).inMilliseconds, " ms"
    echo "timing [gate1] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"

  #
  # 2. recorded tie rows:
  #   synthetic colliding-rank tables, the PQ merge
  #    reproduces the recorded naive-merge streams (ordinary + core)
  #
  block gate2:
    if not gateActive("2"):
      break gate2
    let bt0 = getMonoTime()
    var allFails = 0
    var allRows = 0
    let content = readFile(FixturesDir / "pq_reference_ties.json.zst").
      zstdDecompress(string)
    let recorded = content.fromJson(seq[RecordedTie])
    for tRow in 0 ..< 3:
      var state: uint64 = 0xBB3E68C1EEF02A11'u64 + uint64(tRow) * 0x9E37'u64
      var ranksTable: Table[seq[byte], int]
      # every single byte must be a key, else the merge-core output
      # lookup would KeyError on surviving single-byte parts
      for b in 0 .. 255:
        ranksTable[@[byte(b)]] = b
      var nKeys = 0
      while nKeys < 40:
        let n = 2 + int(lcg(state) mod 3)
        var key: seq[byte] = @[]
        for i in 0 ..< n:
          key.add byte(97 + int(lcg(state) mod 3))
        if key notin ranksTable:
          ranksTable[key] = int(lcg(state) mod 8) # heavy rank collisions
          inc nKeys
      let engine = BpeEngine.init(ranksTable)
      engine.ensureBuilt()
      let fam = famQwen # pattern only shapes the segmentation
      for fx in recorded.items:
        let wantTable = "table" & $tRow & "-"
        if not fx.name.startsWith(wantTable):
          continue
        let pieces = collectPieces(fam, fx.text)
        let pOrd = pqOrdIds(engine, fx.text, pieces)
        let pCore = pqCoreIds(engine, fx.text, pieces)
        if pOrd != fx.ordinaryIds:
          inc allFails
          if allFails <= 3:
            echo "TIE ORD FAIL table ", tRow, " ", fx.name, " first diff ",
              firstDiff(fx.ordinaryIds, pOrd)
        if pCore != fx.coreIds:
          inc allFails
          if allFails <= 3:
            echo "TIE CORE FAIL table ", tRow, " ", fx.name, " first diff ",
              firstDiff(fx.coreIds, pCore)
        inc allRows
    check "recorded tie rows byte-identical (3 colliding tables x " &
      "500 rows, duplicate ranks)", allFails == 0,
      $allFails & " fails / " & $allRows & " rows"
    echo "timing [gate2] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  #
  # 3. 1-byte double-add:
  #   the merge core emits the recorded double-add
  #   structure (the tiktoken fast-path add is not exclusive with the merge output loop), the id value comes from the codec's
  #    rank table, the structure assertion is this suite's
  #
  block gate3:
    if not gateActive("3"):
      break gate3
    let bt0 = getMonoTime()
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    var m = MergeBuf.init()
    for piece in ["a", " ", "z"]:
      let r = tok.ranks[@[byte(piece[0])]]
      var pqIds: seq[int] = @[]
      bytePairEncodePQ(engine, m, piece.toOpenArrayByte(0, piece.len - 1), 0, piece.len, pqIds)
      check "1-byte double-add [" & piece & "]",
        pqIds == @[r, r], $pqIds & " vs rank " & $r

    echo "timing [gate3] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  #
  # 4. engine-history invariance:
  #   cold-vs-warm equality, 20000-pass
  #    overwrite reproducibility, content-key discipline under reset()
  #    reuse with different texts (aliasing). The hit-receipt rows went
  #    with the deleted pair/word memos. Their redundancy property
  #    lives in t_cache_redundancy.nim
  #
  block gate4:
    if not gateActive("4"):
      break gate4
    let bt0 = getMonoTime()
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    # texts with non-vocabulary pieces so the merge path is exercised,
    # and shared piece/pair contents across the texts
    let textA = "qqqzzz wwxxyy qqqzzz wwxxyy alpha 12345"
    let textB = "wwxxyy qqqzzz beta wwxxyy qqqzzz 67890"
    let textC = "qqqzzz gamma wwxxyy delta qqqzzz 24680"

    proc ordIds(engine: BpeEngine, text: string): seq[int] =
      let pieces = collectPieces(famCl100k, text)
      pqOrdIds(engine, text, pieces)

    # cold-vs-warm:
    #   encode in different orders, results identical
    let eCold = BpeEngine.init(tok.ranks)
    let wA = ordIds(eCold, textA)
    let wB = ordIds(eCold, textB)
    let wC = ordIds(eCold, textC)
    let eWarm = BpeEngine.init(tok.ranks)
    discard ordIds(eWarm, textB) # warm in reverse order
    discard ordIds(eWarm, textC)
    discard ordIds(eWarm, textA)
    check "cold-vs-warm engine-history equality (3 texts)",
      ordIds(eWarm, textA) == wA and ordIds(eWarm, textB) == wB and
      ordIds(eWarm, textC) == wC
    # direct-mapped overwrite:
    #   more distinct segments than slots,
    # second pass must reproduce the first exactly
    let eOv = BpeEngine.init(tok.ranks)
    var state: uint64 = 0x1234ABCD5678EF90'u64
    var pass1: seq[seq[int]] = @[]
    for i in 0 ..< 20000:
      let t = genFuzz(state)
      pass1.add ordIds(eOv, t)
    state = 0x1234ABCD5678EF90'u64
    var ovFails = 0
    for i in 0 ..< 20000:
      let t = genFuzz(state)
      if ordIds(eOv, t) != pass1[i]:
        inc ovFails
    check "direct-mapped overwrite reproducibility (20000 x 2)",
      ovFails == 0, $ovFails & " fails"

    # aliasing:
    #   one cursor reset() over three different texts, spans are offsets,
    # keys must follow content, outputs must match
    # a fresh engine
    let eFresh = BpeEngine.init(tok.ranks)
    let wantA = ordIds(eFresh, textA)
    let wantB = ordIds(eFresh, textB)
    let wantC = ordIds(eFresh, textC)
    var gotA = machineIds(eFresh, famCl100k, textA)
    # reset + full-drain discipline for textB then textC on the same
    # pre-tokenization machine (every buffer reused across rebinds)
    var r = initPreTokenizer(famCl100k, textB)
    var m = MergeBuf.init()
    var scratch: seq[int] = @[]
    var gotB: seq[int] = @[]
    for piece in r.items:
      scratch.setLen(0)
      encodeSegmentPQ(eFresh, m, textB.toOpenArrayByte(0, textB.len - 1),
        piece[0], piece[1], scratch)
      for x in scratch.items:
        gotB.add x
    r.reset(famCl100k, textC)
    var gotC: seq[int] = @[]
    for piece in r.items:
      scratch.setLen(0)
      encodeSegmentPQ(eFresh, m, textC.toOpenArrayByte(0, textC.len - 1),
        piece[0], piece[1], scratch)
      for x in scratch.items:
        gotC.add x
    check "reset() content-key discipline (3 texts, one cursor)",
      gotA == wantA and gotB == wantB and gotC == wantC

    echo "timing [gate4] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  #
  # 5. machine consumption rows:
  #   piece-by-piece walk == full drain,
  #    empty input yields nothing, post-drain re-iteration stays empty
  #
  block gate5:
    if not gateActive("5"):
      break gate5
    let bt0 = getMonoTime()
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    let text = "don't stop 1234567 believing   中文字符 a\r\n\r\nb ....hold on"
    let want = machineIds(engine, famCl100k, text)
    check "reference drain nonempty", want.len > 10, $want.len
    var pullFails = 0
    # piece-at-a-time walk with a fresh machine per pass equals the full
    # drain of a reused machine (engine-history invariance rows already cover the cold/warm axis)
    var rWarm = initPreTokenizer(famCl100k, text)
    var mWarm = MergeBuf.init()
    var scratchWarm: seq[int] = @[]
    var warmGot: seq[int] = @[]
    for piece in rWarm.items:
      scratchWarm.setLen(0)
      encodeSegmentPQ(engine, mWarm, text.toOpenArrayByte(0, text.len - 1),
        piece[0], piece[1], scratchWarm)
      for x in scratchWarm.items:
        warmGot.add x
    if warmGot != want:
      inc pullFails
      echo "reused-machine walk mismatch: ", warmGot.len, " vs ", want.len
    var rEmpty = initPreTokenizer(famCl100k, "")
    var emptyCount = 0
    for piece in rEmpty.items:
      inc emptyCount
    if emptyCount != 0:
      inc pullFails
      echo "empty input yielded ", emptyCount, " pieces"
    for i in 0 ..< 3:
      var extra = 0
      for piece in rWarm.items:
        inc extra
      if extra != 0:
        inc pullFails
        echo "post-drain re-iteration yielded ", extra, " pieces"
    check "machine consumption rows (walk, empty, post-drain)", pullFails == 0

    echo "timing [gate5] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  # 6. zero-alloc receipt:
  #   warm engine, 2000 full fresh-machine +
  #   drain passes move no memory (per-pass scratch reuses freed allocator blocks)
  #
  block gate6:
    if not gateActive("6"):
      break gate6
    let bt0 = getMonoTime()
    const CjkText = "中文字横書きカタカナ语料 1234567890 mixed 漢字 runs."
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    var warm = 0
    block warmup:
      var r = initPreTokenizer(famCl100k, CjkText)
      var m = MergeBuf.init()
      var scratch: seq[int] = @[]
      for piece in r.items:
        scratch.setLen(0)
        encodeSegmentPQ(engine, m,
          CjkText.toOpenArrayByte(0, CjkText.len - 1), piece[0], piece[1], scratch)
        warm += scratch.len
    check "warmup drain emitted ids", warm > 0
    let before = getTotalMem()
    for i in 0 ..< 2000:
      var r = initPreTokenizer(famCl100k, CjkText)
      var m = MergeBuf.init()
      var scratch: seq[int] = @[]
      for piece in r.items:
        scratch.setLen(0)
        encodeSegmentPQ(engine, m,
          CjkText.toOpenArrayByte(0, CjkText.len - 1), piece[0], piece[1], scratch)
    let after = getTotalMem()
    check "zero-alloc receipt (2000 reset + drain passes, CJK input)",
      after == before, "delta " & $(after - before)

    echo "timing [gate6] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  #
  # 7. merge tables:
  #   load receipts + SplitTable structural invariants
  #    pair count and worst-bucket bound, zero self-check failures,
  #    singles self-split, children ids below the parent, children
  #    bytes concatenating to the parent bytes)
  #
  block gate7:
    if not gateActive("7"):
      break gate7
    let bt0 = getMonoTime()
    for row in families.items:
      let tok = loadTiktokenCodec(TokenizersDir / row.file)
      let engine = BpeEngine.init(tok.ranks)
      engine.ensureBuilt()
      echo "load receipt [", row.label, "] pairs=", engine.mt.pairCount,
        " worstBucket=", engine.mt.worstBucket, " slots=2^",
        engine.mt.pairIndex.slotBits, " selfCheckFails=",
        engine.mt.selfCheckFails, " tablesMem=",
        engine.mt.tablesMemBytes(), " bytes"
      check "self-check zero failures [" & row.label & "]",
        engine.mt.selfCheckFails == 0, $engine.mt.selfCheckFails & " fails"
      check "worst bucket bounded [" & row.label & "]",
        engine.mt.worstBucket >= 1 and
        engine.mt.worstBucket <= PairIndexMaxBucket,
        $engine.mt.worstBucket
      var structFails = 0
      let n = engine.mt.starts.len
      for d in 0 ..< n:
        let st = int(engine.mt.starts[d])
        let ln = int(engine.mt.lens[d])
        let L = int(engine.mt.splitLeft[d])
        let R = int(engine.mt.splitRight[d])
        if ln == 1:
          if L != d or R != d:
            inc structFails
          continue
        if L >= d or R >= d:
          inc structFails
          continue
        let lst = int(engine.mt.starts[L])
        let llen = int(engine.mt.lens[L])
        let rst = int(engine.mt.starts[R])
        let rlen = int(engine.mt.lens[R])
        if llen + rlen != ln:
          inc structFails
          continue
        var ok = true
        for i in 0 ..< llen:
          if engine.mt.arena[st + i] != engine.mt.arena[lst + i]:
            ok = false
            break
        if ok:
          for i in 0 ..< rlen:
            if engine.mt.arena[st + llen + i] != engine.mt.arena[rst + i]:
              ok = false
              break
        if not ok:
          inc structFails
      check "SplitTable structural invariants [" & row.label & "]",
        structFails == 0, $structFails & " fails over " & $n & " tokens"
      if row.label == "kimik2.5":
        check "kimik2.5 pair derivation count (probed 296461)",
          engine.mt.pairCount == 296461, $engine.mt.pairCount
        check "kimik2.5 merged-ids contiguity asserted at load",
          int(engine.mt.lens[255]) == 1 and
          int(engine.mt.lens[256]) > 1, "boundary ids 255/256 shapes"

    echo "timing [gate7] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  #
  # 8. structure check:
  #   PairIndex/SplitTable merge path vs the pair-validation-cache path
  #    over the corpus prefixes, fixtures, tie storms and the 20k LCG
  #    fuzz rows, id streams byte-identical per piece
  #
  block gate8:
    if not gateActive("8"):
      break gate8
    let bt0 = getMonoTime()
    var twinFails = 0
    var twinIds = 0
    for row in families.items:
      let tok = loadTiktokenCodec(TokenizersDir / row.file)
      let engine = BpeEngine.init(tok.ranks)
      engine.ensureBuilt()
      for text in sharedTexts.items:
        let pieces = collectPieces(row.fam, text)
        let legacy = pqCoreIdsLegacy(engine, text, pieces)
        let indexed = pqCoreIds(engine, text, pieces)
        if legacy != indexed:
          inc twinFails
          if twinFails <= 3:
            echo "TWIN FAIL [", row.label, "] first diff ",
              firstDiff(legacy, indexed), " text head ",
              text[0 ..< min(text.len, 60)].escape
        twinIds += indexed.len
      check "structure twin byte-identical (PairIndex vs pair cache) [" &
        row.label & "]", twinFails == 0,
        $twinFails & " fails / " & $twinIds & " ids"

    echo "timing [gate8] wall ", (getMonoTime() - bt0).inMilliseconds, " ms"
  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall PQ reference checks passed (", totalIds, " ids gated)"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
