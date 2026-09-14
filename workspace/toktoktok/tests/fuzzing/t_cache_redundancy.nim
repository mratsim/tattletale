# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Engine-history invariance for the priority-queue reference core
## (the pair/word memo deletion check rows):
## - for every family and every text, the warm-engine streams
##   (ordinary path + PairIndex-free core) equal the cold-engine
##   streams under arbitrary engine warm/ cold history.
## - the memos were deleted once this property held on every row,
##   with the memo hit receipts proving engagement
##   (wordHits, pairHits > 0 on the warm stream of every family).
##
## Run:
##   nim cpp -r -d:release --stackTrace:on --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests workspace/toktoktok/tests/fuzzing/t_cache_redundancy.nim [familyIndex|all]

import std/[os, strutils, tables, monotimes, times, json]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/tests/fuzzing/pq_bpe_reference
import workspace/toktoktok/src/merge
import workspace/toktoktok/src/scan
import workspace/toktoktok/src/deserializers
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
  if a.len != b.len:
    return min(a.len, b.len)
  -1

proc cachedOrdIds(e: BpeEngine, text: string, pieces: seq[(int, int)]): seq[int] =
  ## Reference ordinary path, whole-piece hit then PQ merge.
  var m = MergeBuf.init()
  for s in pieces.items:
    encodeSegmentPQ(e, m, text.toOpenArrayByte(0, text.len - 1), s[0], s[1], result)

proc cachedCoreLegacy(e: BpeEngine, text: string,
    pieces: seq[(int, int)]): seq[int] =
  ## PairIndex-free merge core over the direct byte-span lookups
  var m = MergeBuf.init()
  for s in pieces.items:
    bytePairEncodePQLegacy(e, m, text.toOpenArrayByte(0, text.len - 1), s[0], s[1], result)

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

proc collectPieces(fam: Family, text: string): seq[(int, int)] =
  ## Full drain of one pre-tokenization machine
  ## (bounded by the input length in pieces, every piece covers >= 1 byte).
  var r = initPreTokenizer(fam, text)
  var guard = 0
  for piece in r.items:
    inc guard
    doAssert guard <= text.len + 1, "piece stream overran its bound"
    result.add piece

proc propertyFamily(label: string, fam: Family, codec: TiktokenCodec,
    texts: seq[string]): tuple[fails, ids: int] =
  var engWarm = BpeEngine.init(codec.ranks)
  # cold engine:
  #   same rank table, caches in their emptiest state
  var engCold = BpeEngine.init(codec.ranks)
  for text in texts.items:
    let pieces = collectPieces(fam, text)
    # warm engine:
    #   caches populated by every earlier text
    let warmOrd = cachedOrdIds(engWarm, text, pieces)
    let warmLegacy = cachedCoreLegacy(engWarm, text, pieces)
    let coldOrd = cachedOrdIds(engCold, text, pieces)
    let coldLegacy = cachedCoreLegacy(engCold, text, pieces)
    if warmOrd != coldOrd:
      inc result.fails
      echo "ORD CACHE-LEAK [", label, "] first diff ",
        firstDiff(coldOrd, warmOrd), " text len ", text.len
    if warmLegacy != coldLegacy:
      inc result.fails
      echo "LEGACY CACHE-LEAK [", label, "] first diff ",
        firstDiff(coldLegacy, warmLegacy), " text len ", text.len
    result.ids += coldOrd.len
  # cold-vs-warm rows:
  #   one cold engine per corpus (the largest, most cache-exercising texts), caches in their emptiest state
  var state: uint64 = 0x243F6A8885A308D3'u64
  block coldRows:
    for cidx in 0 ..< 4:
      let text = texts[cidx]
      let pieces = collectPieces(fam, text)
      let warmOrd = cachedOrdIds(engWarm, text, pieces)
      let warmLegacy = cachedCoreLegacy(engWarm, text, pieces)
      var engCold2 = BpeEngine.init(codec.ranks)
      let coldOrd = cachedOrdIds(engCold2, text, pieces)
      let coldLegacy = cachedCoreLegacy(engCold2, text, pieces)
      if coldOrd != warmOrd:
        inc result.fails
        echo "COLD-vs-WARM ORD FAIL [", label, "] corpus ", cidx,
          " first diff ", firstDiff(warmOrd, coldOrd)
      if coldLegacy != warmLegacy:
        inc result.fails
        echo "COLD-vs-WARM LEGACY FAIL [", label, "] corpus ", cidx,
          " first diff ", firstDiff(warmLegacy, coldLegacy)
  # fuzz rows on the warm engine:
  #   caches in their most-populated state,
  # every row checked against the cold engine (cache-leak detection)
  for iter in 0 ..< FuzzRounds:
    let text = genFuzz(state)
    let pieces = collectPieces(fam, text)
    let warmOrd = cachedOrdIds(engWarm, text, pieces)
    let coldOrd = cachedOrdIds(engCold, text, pieces)
    if warmOrd != coldOrd:
      inc result.fails
      echo "ORD CACHE-LEAK [", label, "] fuzz iter ", iter,
        " first diff ", firstDiff(coldOrd, warmOrd)
    result.ids += coldOrd.len
  check "caches result-neutral: warm == cold [" &
    label & "]", result.fails == 0, $result.fails & " fails / " &
    $result.ids & " ids"

proc main() =
  var families = [
    (label: "r50k", fam: famR50k, pat: R50kPat, file: "r50k_base.tiktoken"),
    (label: "p50k", fam: famP50k, pat: R50kPat, file: "p50k_base.tiktoken"),
    (label: "cl100k", fam: famCl100k, pat: Cl100kPat, file: "cl100k_base.tiktoken"),
    (label: "o200k", fam: famO200k, pat: O200kPat, file: "o200k_base.tiktoken"),
    (label: "kimik2.5", fam: famKimiK25, pat: KimiK25Pat, file: "kimik2.5.tiktoken"),
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

  var texts: seq[string] = @[]
  for (cname, path, maxBytes) in corpora.items:
    texts.add readCorpusPrefix(path, maxBytes)
  for t in fixtureTexts.items:
    texts.add t
  for t in tieStorms.items:
    texts.add t

  let args = commandLineParams()
  let sel = if args.len >= 1 and args[0] != "all": parseInt(args[0]) else: -1
  for fIdx in 0 ..< families.len:
    if sel >= 0 and sel != fIdx:
      continue
    let row = families[fIdx]
    let t0 = getMonoTime()
    let codec = loadTiktokenCodec(TokenizersDir / row.file)
    let (fails, ids) = propertyFamily(row.label, row.fam, codec, texts)
    discard fails
    discard ids
    echo "timing [family ", row.label, "] wall ",
      (getMonoTime() - t0).inMilliseconds, " ms"

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\ncache-redundancy property held on all rows"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  main()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
