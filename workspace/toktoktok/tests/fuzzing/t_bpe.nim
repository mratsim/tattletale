# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## The backtracking merge core's cross-check suite, the served lazy-DP
## backtracking encoder (src/merge.nim) vs the priority-queue reference
## implementation (pq_bpe_reference.bytePairEncodePQ / encodeSegmentPQ, test-side, no served path), byte-identical id streams.
##
## Two independent encoders agreeing byte for byte on every
## family and corpus, fixture, adversarial and fuzz row is
## the strongest cheap guard in the engine.
##
## The served encoders' id values are locked to ground truth
## by the recorded-fixture suites
## (test_fixtures_small_*, the recorded tiktoken/HF id streams through the pipeline).
##
## The PQ reference is locked to the same recordings in t_pq_reference,
## so bt == PQ locks bt to them transitively.
##
## Equivalence scope (the recorded limit):
## - the bt validity predicate is rank-only (a PairIndex hit below the split-chain limit freezes the adjacency invalid),
##   the PQ tie rule is rank+leftmost.
## - on rank tables with DISTINCT ranks, every real checkpoint shape
##   (tiktoken ids and HF vocab ids are unique per token), the two
##   encoders agree, duplicate-rank tables are outside bt's contract,
##   and the synthetic tables here use distinct (non-dense, gapped) ranks.
## - the PQ reference carries the leftmost-tie contract itself
##   (t_pq_reference check 2, recorded rows).
##
## Checks:
## - family parity (5 families x corpora + fixture frames + tie storms + adversarial rows).
## - 20k LCG fuzz rows per family (100k total, bt == PQ).
## - synthetic distinct-rank tables (3 tables x 5000 rows).
## - named validity-predicate rows (direct-merge, no-merge, frozen-invalid).
## - named bt ordinary rows, and the zero-alloc receipt over the bt steady state.

{.experimental: "views".}
import std/[os, strutils, algorithm, monotimes, times, tables, json]

import workspace/zstd/zstd_highlevel
import pq_bpe_reference
import workspace/toktoktok/src/merge
import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  TokenizersDir = TestsDir / "tokenizers"
  CorpusDir = TestsDir / "corpus"
  FixturesDir = TestsDir / "fixtures"

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

proc genRow(state: var uint64): string =
  const Alphabet = [
    "a", "b", "c", "d", "e", " ", " ", "1", "2", "9", ".", ",", "\n",
    "\r\n", "\t", "'", "s", "t", "he", "th", "er", "中", "文", "の",
    "😀", "\u{00A0}", "_", "-", "0", "L", "L", "L",
  ]
  let n = 1 + int((lcg(state) shr 11) mod 48)
  result = ""
  for i in 0 ..< n:
    result.add Alphabet[int((lcg(state) shr 11) mod Alphabet.len)]

# ------------------------------------------------------------------------
# per-piece encoders over one segmentation
# ------------------------------------------------------------------------

proc btOrdIds(engine: BpeEngine, bt: var BacktrackBuf, text: string,
    pieces: seq[(int, int)]): seq[int] =
  var scratch: seq[int] = @[]
  let data = text.toOpenArrayByte(0, text.len - 1)
  for s in pieces.items:
    scratch.setLen(0)
    encodeSegment(engine, bt, data, s[0], s[1], scratch)
    for x in scratch.items:
      result.add x

proc pqOrdIds(engine: BpeEngine, text: string,
    pieces: seq[(int, int)]): seq[int] =
  var m = MergeBuf.init()
  var scratch: seq[int] = @[]
  let data = text.toOpenArrayByte(0, text.len - 1)
  for s in pieces.items:
    scratch.setLen(0)
    encodeSegmentPQ(engine, m, data, s[0], s[1], scratch)
    for x in scratch.items:
      result.add x

# ------------------------------------------------------------------------
# gates
# ------------------------------------------------------------------------

const AdversarialRows = [
  "ababababababababababababababababababababababababababababababababab",
  "LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
  "                                    ",
  "\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n",
  "中文字符中文字符中文字符中文字符中文字符中文字符",
  "\xE4\xB8\xAD\xE6\x96\x87 123 456 \xE4\xB8\xAD\xE6\x96\x87 789",
  "\xEF\xBB\xBFleading bom \xEF\xBB\xBFagain \xEF\xBB\xBFmore",
  "don't stop 1234567 believing   中文字符 a\r\n\r\nb ....hold on",
  "((((((((((()))))))))))abababab[[[[[[[[",
  "012345678901234567890123456789012345678901234567890123456789",
  "                    x                    y                    z",
  "a b a b a b a b a b a b a b a b a b a b a b a b a b a b a b",
]

proc gateFamily(label: string, fam: Family, codec: TiktokenCodec,
    engine: BpeEngine, texts: seq[string]): tuple[fails, ids: int] =
  var bt = BacktrackBuf.init()
  var r = initPreTokenizer(fam, "")
  for text in texts.items:
    r.reset(fam, text)
    var pieces: seq[(int, int)] = @[]
    for piece in r.items:
      pieces.add piece
    let bOrd = btOrdIds(engine, bt, text, pieces)
    let pOrd = pqOrdIds(engine, text, pieces)
    if bOrd != pOrd:
      inc result.fails
      echo "TWIN FAIL [", label, "] first diff ", firstDiff(pOrd, bOrd),
        " text head ", text[0 ..< min(text.len, 60)].escape
    result.ids += bOrd.len

proc gateActive(gate: string): bool =
  let args = commandLineParams()
  result = args.len < 1 or args[0] == "all" or args[0] == gate

proc main() =
  if not fileExists(TokenizersDir / "kimik2.5.tiktoken"):
    echo "MISSING tokenizer data: ", TokenizersDir
    echo "stage it with: python3 ",
      TestsDir /
      "fetch_test_tokenizers.py"
    quit(1)

  var families = [
    (label: "r50k", fam: famR50k, file: "r50k_base.tiktoken"),
    (label: "p50k", fam: famP50k, file: "p50k_base.tiktoken"),
    (label: "cl100k", fam: famCl100k, file: "cl100k_base.tiktoken"),
    (label: "o200k", fam: famO200k, file: "o200k_base.tiktoken"),
    (label: "kimik2.5", fam: famKimiK25, file: "kimik2.5.tiktoken"),
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

  var sharedTexts: seq[string] = @[]
  for (cname, path, maxBytes) in corpora.items:
    sharedTexts.add readCorpusPrefix(path, maxBytes)
  for t in fixtureTexts.items:
    sharedTexts.add t
  for t in tieStorms.items:
    sharedTexts.add t
  for t in AdversarialRows.items:
    sharedTexts.add t

  block gate1:
    if not gateActive("1"):
      break gate1
    let t0 = getMonoTime()
    for row in families.items:
      let ft0 = getMonoTime()
      let tok = loadTiktokenCodec(TokenizersDir / row.file)
      let engine = BpeEngine.init(tok.ranks)
      engine.ensureBuilt()
      let (fails, ids) = gateFamily(row.label, row.fam, tok, engine,
        sharedTexts)
      check "bt == PQ byte-identical [" & row.label & "]",
        fails == 0, $fails & " fails / " & $ids & " ids"
      echo "timing [gate1/", row.label, "] wall ",
        (getMonoTime() - ft0).inMilliseconds, " ms"
    echo "timing [gate1] wall ", (getMonoTime() - t0).inMilliseconds, " ms"

  block gate2:
    # 20k LCG fuzz rows per family (100k total):
    #   bt == PQ every row
    if not gateActive("2"):
      break gate2
    let t0 = getMonoTime()
    const RowsPerFamily = 20000
    var twinFails = 0
    var totalIds = 0
    for row in families.items:
      let ft0 = getMonoTime()
      let tok = loadTiktokenCodec(TokenizersDir / row.file)
      let engine = BpeEngine.init(tok.ranks)
      engine.ensureBuilt()
      var bt = BacktrackBuf.init()
      var r = initPreTokenizer(row.fam, "")
      var labelSeed = 0'u64
      for c in row.label:
        labelSeed = labelSeed * 131 + uint64(c)
      var state: uint64 = 0x243F6A8885A308D3'u64 + labelSeed
      for i in 0 ..< RowsPerFamily:
        let text = genRow(state)
        r.reset(row.fam, text)
        var pieces: seq[(int, int)] = @[]
        for piece in r.items:
          pieces.add piece
        let bOrd = btOrdIds(engine, bt, text, pieces)
        let pOrd = pqOrdIds(engine, text, pieces)
        if bOrd != pOrd:
          inc twinFails
          if twinFails <= 3:
            echo "TWIN FAIL [", row.label, "] row ", i, " first diff ",
              firstDiff(pOrd, bOrd), " text ", text.escape
        totalIds += bOrd.len
      echo "timing [gate2/", row.label, "] wall ",
        (getMonoTime() - ft0).inMilliseconds, " ms"
    check "bt == PQ twin, 100k LCG rows", twinFails == 0,
      $twinFails & " fails / " & $totalIds & " ids"
    echo "timing [gate2] wall ", (getMonoTime() - t0).inMilliseconds, " ms"

  block gate3:
    # synthetic distinct-rank tables (gapped, not dense):
    #   bt == PQ over fuzz rows
    if not gateActive("3"):
      break gate3
    let t0 = getMonoTime()
    var allFails = 0
    for tRow in 0 ..< 3:
      var state: uint64 = 0xBB3E68C1EEF02A11'u64 + uint64(tRow) * 0x9E37'u64
      var ranksTable: Table[seq[byte], int]
      # every single byte must be a key, else bt dead-ends where the PQ
      # reference raises (the recorded dead-end semantics divergence).
      # The merged tokens are built by simulated training, a new token
      # concatenates two existing ones and takes a rank ABOVE both parts.
      # Children sort below the parent in dense order, the order every
      # trained rank table carries. The split derivation + validity
      # limit walk stand on this invariant.
      for b in 0 .. 255:
        ranksTable[@[byte(b)]] = b
      var toks: seq[seq[byte]] = @[]
      for b in 0 .. 255:
        toks.add @[byte(b)]
      var nextRank = 1000
      var nKeys = 0
      while nKeys < 40:
        let i = int((lcg(state) shr 11) mod uint64(toks.len))
        let j = int((lcg(state) shr 11) mod uint64(toks.len))
        var key = toks[i]
        for c in toks[j]:
          key.add c
        if key.len <= 6 and key notin ranksTable:
          ranksTable[key] = nextRank
          nextRank += 1 + int((lcg(state) shr 11) mod 7) # distinct, gapped ranks
          toks.add key
          inc nKeys
      let engine = BpeEngine.init(ranksTable)
      engine.ensureBuilt()
      let fam = famQwen # pattern only shapes the segmentation
      var bt = BacktrackBuf.init()
      var r = initPreTokenizer(fam, "")
      var state2: uint64 = 0x51633E2D9F5C7F21'u64
      for iter in 0 ..< 5000:
        let n = 1 + int((lcg(state2) shr 11) mod 24)
        var text = ""
        for i in 0 ..< n:
          text.add char(97 + int((lcg(state2) shr 11) mod 4))
        r.reset(fam, text)
        var pieces: seq[(int, int)] = @[]
        for piece in r.items:
          pieces.add piece
        let bOrd = btOrdIds(engine, bt, text, pieces)
        let pOrd = pqOrdIds(engine, text, pieces)
        if bOrd != pOrd:
          inc allFails
          if allFails <= 3:
            echo "SYNTH FAIL table ", tRow, " row ", iter, " text ",
              text.escape, " bt ", bOrd, " pq ", pOrd
    check "synthetic distinct-rank tables byte-identical (3 x 5000 rows)",
      allFails == 0, $allFails & " fails"
    echo "timing [gate3] wall ", (getMonoTime() - t0).inMilliseconds, " ms"

  block gate4:
    # named validity-predicate rows over kimik2.5, direct-merge
    # invalid rows, no-merge valid, frozen-invalid (valid split, unchosen)
    if not gateActive("4"):
      break gate4
    let t0 = getMonoTime()
    let tok = loadTiktokenCodec(TokenizersDir / "kimik2.5.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    # direct-merge, any 2-byte token's two bytes merge, predicate false
    var directChecked = false
    for d in 0 ..< engine.mt.starts.len:
      if directChecked or int(engine.mt.lens[d]) != 2:
        continue
      let st = int(engine.mt.starts[d])
      let u = uint32(engine.trie.lookup(engine.mt.arena, st, st + 1))
      let v = uint32(engine.trie.lookup(engine.mt.arena, st + 1, st + 2))
      if u < 256 and v < 256:
        check "isValidTokenPair direct-merge bytes invalid",
          not isValidTokenPair(engine, u, v)
        directChecked = true
    # no-merge, two single-byte tokens whose byte concat is no token,
    # predicate true (dense ids 0..255 are the single-byte tokens, the concat key is read off the arena, not spelled from the ids)
    var noMergeChecked = false
    for u in 0 ..< 256:
      if noMergeChecked:
        break
      let bu = engine.mt.arena[int(engine.mt.starts[u])]
      for v in 0 ..< 256:
        let bv = engine.mt.arena[int(engine.mt.starts[v])]
        var key: array[2, byte] = [bu, bv]
        if engine.trie.lookup(key) < 0:
          check "isValidTokenPair no-merge byte pair valid",
            isValidTokenPair(engine, uint32(u), uint32(v)),
            "bytes (" & $bu & ", " & $bv & ") concat lookup " &
              $engine.trie.lookup(key) & " pairIndex " &
              $engine.mt.pairIndex.lookup(uint32(u), uint32(v)) &
              " split " & $engine.mt.splitLeft[u] & "," &
              $engine.mt.splitRight[u] & " / " & $engine.mt.splitLeft[v] &
              "," & $engine.mt.splitRight[v]
          noMergeChecked = true
          break
    # frozen-invalid:
    #   a valid split (u, v) of token d whose recorded
    # split differs -> the merge process would merge (u, v) -> false
    var frozenFound = false
    for d in 0 ..< engine.mt.starts.len:
      if frozenFound or int(engine.mt.lens[d]) < 2:
        continue
      if engine.mt.splitLeft[d] == int32(d):
        continue # self-split (unreachable-shape receipt case)
      let st = int(engine.mt.starts[d])
      let en = st + int(engine.mt.lens[d])
      for s in 1 ..< int(engine.mt.lens[d]):
        let a = engine.trie.lookup(engine.mt.arena, st, st + s)
        let b = engine.trie.lookup(engine.mt.arena, st + s, en)
        if a < 0 or b < 0:
          continue
        if a != int(engine.mt.splitLeft[d]) or b != int(engine.mt.splitRight[d]):
          check "isValidTokenPair frozen-invalid (valid unchosen split)",
            not isValidTokenPair(engine, uint32(a), uint32(b)),
            "token " & $d & " split (" & $a & ", " & $b & ") vs chosen (" &
              $engine.mt.splitLeft[d] & ", " & $engine.mt.splitRight[d] & ")"
          frozenFound = true
          break
    check "frozen-invalid pair exists in kimik2.5", frozenFound
    echo "timing [gate4] wall ", (getMonoTime() - t0).inMilliseconds, " ms"

  block gate5:
    # zero-alloc receipt:
    #   warm bt steady state over a CJK-heavy input
    if not gateActive("5"):
      break gate5
    let t0 = getMonoTime()
    const CjkText = "中文字横書きカタカナ语料 1234567890 mixed 漢字 runs."
    let tok = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let engine = BpeEngine.init(tok.ranks)
    engine.ensureBuilt()
    var bt = BacktrackBuf.init()
    var r = initPreTokenizer(famCl100k, "")
    var warm = 0
    block warmup:
      r.reset(famCl100k, CjkText)
      var scratch: seq[int] = @[]
      let data = CjkText.toOpenArrayByte(0, CjkText.len - 1)
      for piece in r.items:
        scratch.setLen(0)
        encodeSegment(engine, bt, data, piece[0], piece[1], scratch)
        warm += scratch.len
    check "warmup drain emitted ids", warm > 0
    let before = getTotalMem()
    for i in 0 ..< 2000:
      r.reset(famCl100k, CjkText)
      var scratch: seq[int] = @[]
      let data = CjkText.toOpenArrayByte(0, CjkText.len - 1)
      for piece in r.items:
        scratch.setLen(0)
        encodeSegment(engine, bt, data, piece[0], piece[1], scratch)
    let after = getTotalMem()
    check "zero-alloc receipt (2000 reset + drain passes, bt core)",
      after == before, "delta " & $(after - before)
    echo "timing [gate5] wall ", (getMonoTime() - t0).inMilliseconds, " ms"

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall backtracking cross-check rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ",
    (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
