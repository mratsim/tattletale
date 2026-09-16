# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Pipeline machine verification:
## 1. chunk-boundary invariance:
##    chunked feeds + finish equal the whole-input stream for every
##    single-chunk split offset,
## 2. machine stream rows:
##    partial consumption + resume, post-drain re-iteration empty,
##    drained sticky, feed-restart guard.

import std/[os, strutils, tables, json, monotimes, times]

import workspace/zstd/zstd_highlevel
import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/pipeline
import workspace/toktoktok/src/scan

const
  TestsDir = currentSourcePath().parentDir().parentDir()
    ## tests/ root (the suite lives one level down)
  CorpusDir = TestsDir / "corpus"
  TokenizersDir = TestsDir / "tokenizers"
  FixturesDir = TestsDir / "fixtures"

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

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

proc specialDict(codec: TiktokenCodec): tuple[pats: seq[string], ids: seq[int]] =
  ## Special dictionary extracted from the live codec table (same-start tie keeps the longest match), never assumed.
  for token, id in codec.specials:
    result.pats.add token
    result.ids.add id

proc collect(p: pipeline.TokPipeline, text: string): seq[int] =
  ## Whole-input machine drain with restart, so one pipeline serves
  ## every text of a family
  ## (the wall budget forbids a rebuild per text, the vocab trie build dominates at large vocabularies).
  pipeline.resetText(p, text)
  for id in p.items:
    result.add id
  doAssert p.drained()

proc collectChunked(p: pipeline.TokPipeline, text: string,
    split: int): seq[int] =
  ## Chunked mode with one fixed split offset, draining between feeds,
  ## the pipeline is reused, beginStream restarts the stream.
  pipeline.beginStream(p)
  pipeline.feed(p, toOpenArray(text, 0, split - 1))
  for id in p.items:
    result.add id
  pipeline.feed(p, toOpenArray(text, split, text.len - 1))
  for id in p.items:
    result.add id
  pipeline.finishStream(p)
  for id in p.items:
    result.add id
  doAssert p.drained()

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  let families = [
    (label: "r50k", fam: scan.Family.famR50k, file: "r50k_base.tiktoken",
     pat: scan.R50kPat),
    (label: "cl100k", fam: scan.Family.famCl100k,
     file: "cl100k_base.tiktoken", pat: scan.Cl100kPat),
    (label: "kimik2.5", fam: scan.Family.famKimiK25,
     file: "kimik2.5.tiktoken", pat: scan.KimiK25Pat),
  ]

  let corpora = [
    ("sanguozhi", CorpusDir /
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 20000),
    ("verne", CorpusDir /
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 20000),
    ("shakespeare", CorpusDir / "pg100-shakespeare.txt.zst", 30000),
    ("sqlite", CorpusDir / "sqlite3.c.zst", 50000),
  ]

  var texts: seq[string] = @[]
  for (cname, path, maxBytes) in corpora.items:
    texts.add readCorpusPrefix(path, maxBytes)
  for frame in ["pretok_chain_exaone.json.zst",
                "pretok_chain_step-3.5-flash.json.zst",
                "pretok_chain_gemma-4.json.zst",
                "special_pretok_kimik2.5.json.zst",
                "special_pretok_step-3.5-flash.json.zst",
                "special_pretok_exaone.json.zst"]:
    for t in frameTexts(frame):
      texts.add t
  for storm in [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "11111111111111111111",
    "abababababababababababababababab",
    "a1b2c3d4e5f6g7h8a1b2c3d4e5f6g7h8",
    "\xE4\xB8\xAD\xE6\x96\x87 123 456 789 \xE4\xB8\xAD\xE6\x96\x87",
    "a\r\nb\r\nc\r\n\r\n\r\n\r\nd",
    "0123456789012345678901234567890123456789",
  ]:
    texts.add storm

  var t0 = getMonoTime()
  #
  # 1. chunk-boundary invariance:
  #   every single-chunk split offset
  #
  block:
    let codec = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let (pats, ids) = specialDict(codec)
    var cache = PreTokRegexCache()
    let text = "hello <|s|>world 1234567<|e|>tail 中文字 a\r\n\r\nb"
    var pChunk = pipeline.TokPipeline.init(cache, codec.ranks, pats, ids,
      scan.Family.famCl100k)
    let want = collect(pChunk, text)
    var splitFails = 0
    for split in 1 ..< text.len:
      let got = collectChunked(pChunk, text, split)
      if got != want:
        inc splitFails
        if splitFails <= 3:
          echo "CHUNK FAIL split ", split, ": ", got.len, " vs ", want.len,
            " ids, first diff ", firstDiff(want, got)
    check "chunk-boundary invariance (every split offset)",
      splitFails == 0, $splitFails & " fails over " & $want.len & " ids"

  #
  # 2. machine stream rows
  #
  block:
    let codec = loadTiktokenCodec(TokenizersDir / "cl100k_base.tiktoken")
    let (pats, ids) = specialDict(codec)
    var cache = PreTokRegexCache()
    let text = "hello <|s|>world 1234567<|e|>tail"
    var p = pipeline.TokPipeline.init(cache, codec.ranks, pats, ids, scan.Family.famCl100k)
    pipeline.resetText(p, text)
    var got: seq[int] = @[]
    var taken = 0
    for id in p.items:
      got.add id
      inc taken
      if taken == 3:
        break
    check "partial consumption holds ids", got.len == 3, $got.len
    for id in p.items:
      got.add id
    let want = collect(p, text)
    check "resume completes the stream identically", got == want
    check "drained sticky", p.drained()
    var extra = 0
    for id in p.items:
      inc extra
    check "post-drain re-iteration empty", extra == 0
    # feed with an ordinary decision in flight asserts:
    #   drain only the first piece's id of the region, leave
    #   the machine mid-decision
    block:
      var pMid = pipeline.TokPipeline.init(cache, codec.ranks, @["<a>"], @[999],
        scan.Family.famCl100k)
      pipeline.beginStream(pMid)
      pipeline.feed(pMid, toOpenArray("x<a>hello world", 0, 13))
      var takenIds = 0
      for id in pMid.items:
        inc takenIds
        if takenIds == 1:
          break
      check "partial drain took the region piece id", takenIds == 1
      var raised = false
      try:
        pipeline.feed(pMid, toOpenArray("more", 0, 3))
      except AssertionDefect:
        raised = true
      check "feed with a decision in flight asserts", raised

    # feed-restart guard:
    #   specials-empty config, drained machine
    var p2 = pipeline.TokPipeline.init(cache, codec.ranks, @[], @[], scan.Family.famCl100k)
    pipeline.resetText(p2, "abc")
    var count = 0
    for id in p2.items:
      inc count
    check "empty-specials pipeline emits the ordinary ids", count > 0
    check "drained before restart", p2.drained()
    # the empty-specials dictionary holds no decisions mid-stream, so
    # a second feed on the drained machine is legal
    var fedOk = true
    try:
      var p3 = pipeline.TokPipeline.init(cache, codec.ranks, @[], @[], scan.Family.famCl100k)
      pipeline.beginStream(p3)
      pipeline.feed(p3, toOpenArray("x<a>", 0, 2))
      for id in p3.items:
        discard id
      pipeline.feed(p3, toOpenArray("b>", 0, 1))
    except AssertionDefect:
      fedOk = false
    check "feed on a drained specials-empty machine is legal", fedOk

  echo "\nall pipeline machine checks passed, wall ",
    (getMonoTime() - t0).inMilliseconds.float64 / 1000.0, " s"
when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
