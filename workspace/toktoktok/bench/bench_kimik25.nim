# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kimik2.5-ranks speed board, full-pipeline wall over the four corpus
## prefixes (verne-5k, shakespeare-10k, sanguozhi-10k CJK, sqlite-50k), Nim-native medians.
## Reference comparisons come from running this bench and the cross-check suites, not from recorded results.
##
##   corpus ──▶ pre-tokenization (pattern split or frontier scan)
##                │
##                ├─▶ scan-only row: the piece stream alone
##                ├─▶ bpe-only row: the naive merge core over the pieces
##                └─▶ bpe-only-bt row: the backtracking tail over the pieces
##
## Rows per corpus:
## - one warm-up rep dropped from the timing, the engine tables, pre-tokenizer buffers and rank table reach steady state.
## - `reps` timed reps, the median wall and MB/s per row.
## - stage-split rows, the four rows below.
##
##   | row                | wall                                                                                                                                     |
##   | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
##   | scan-only          | the pre-tokenization machine's piece stream alone, the pattern-split scan where the step carries one, the frontier-engine scan otherwise |
##   | scan-only-frontier | the same machine forced down the frontier-engine scan, the A/B row against the pattern-split scan                                        |
##   | bpe-only           | the ordinary naive-merge tail, whole-piece hit first else the naive merge core per piece                                                 |
##   | bpe-only-bt        | the ordinary backtracking tail, whole-piece hit first else the bt core                                                                   |
##
## Id stream:
##   | check       | contract                                                                                                                                               |
##   | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
##   | assertion   | every timed rep's id stream equals the reference stream built untimed per corpus                                                                       |
##   | whole-piece | rank hit first, else the naive merge core (`bpe_codec.bytePairEncode`) per pre-tokenized piece, independent of the pipeline's served backtracking tail |
##   | parity      | belongs to the test suites, this assert is the bench's sanity check                                                                                    |
##
## Build + run (-d:release, wall guard ~120 s), from the worktree root:
##   nim cpp -d:release --hints:off --warnings:off --outdir:build/bench \
##     --nimcache:nimcache/bench workspace/toktoktok/bench/bench_kimik25.nim [reps]

{.experimental: "views".}
import std/[os, monotimes, times, strutils, tables, algorithm]

import workspace/zstd/zstd_highlevel
import workspace/regex_engine
import workspace/toktoktok/src/deserializers
import workspace/toktoktok/src/scan
import workspace/toktoktok/src/pipeline
import workspace/toktoktok/src/merge
import workspace/toktoktok/src/bpe_codec {.all.}

const
  WorkDir = currentSourcePath().parentDir()
  ToktoktokDir = WorkDir.parentDir()
  CorpusDir = ToktoktokDir / "tests" / "corpus"
  TokenizersDir = ToktoktokDir / "tests" / "tokenizers"

type
  CorpusRow = tuple[name: string, file: string, prefixBytes: int]

const
  Corpora: array[4, CorpusRow] = [
    ("verne-5k",
      "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 5000),
    ("shakespeare-10k", "pg100-shakespeare.txt.zst", 10000),
    ("sanguozhi-10k",
      "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 10000),
    ("sqlite-50k", "sqlite3.c.zst", 50000),
  ]

proc nowMs(): float =
  (getMonoTime() - MonoTime()).inNanoseconds.float / 1e6

proc readCorpusPrefix(path: string, maxBytes: int): string =
  var text = readFile(path).zstdDecompress(string)
  if text.len > maxBytes:
    text = text[0 ..< maxBytes]
  text

proc bpeOnlyOrd(cache: var PreTokRegexCache, ranks: Table[seq[byte], int], text: string): seq[int] =
  ## Ordinary BPE tail over the family's pre-tokenized pieces:
  ## - whole-piece rank hit first, else the naive merge core
  ##   (`bpe_codec.bytePairEncode`) per piece.
  ## - the naive core is the library's tiktoken-reference merge core,
  ##   the wired pipeline serves the backtracking core instead.
  ##
  ## Serves both as the board's id reference and as the timed bpe-only stage row.
  var scratch: seq[int] = @[]
  var piece: seq[byte] = @[]
  var r = initPreTokenizer(cache, famKimiK25, text)
  for (lo, hi) in r.items:
    scratch.setLen(0)
    piece = cast[seq[byte]](text[lo ..< hi])
    if piece in ranks:
      result.add ranks[piece]
    else:
      bytePairEncode(scratch, piece, ranks)
      for x in scratch.items:
        result.add x

proc frontierPieces(cache: var PreTokRegexCache, fam: Family, text: string): seq[tuple[lo, hi: int]] =
  ## Pre-tokenization piece stream driven down the frontier-engine scan
  ## (step.pat.nextMatch per pattern step):
  ##
  ## the A/B row against the machine's default (pattern-split) scan
  ## and the frontier-scan + backtracking-tail composition row.
  var level = @[(0, text.len)]
  for step in cache.steps(fam).items:
    var nxt: seq[(int, int)]
    for (pieceLo, pieceHi) in level.items:
      var lastEmit = pieceLo
      var offset = pieceLo
      while offset < pieceHi:
        let (ms, me) = step.pat.nextMatch(text, offset, pieceLo, pieceHi)
        if ms < 0:
          break
        if ms > lastEmit:
          nxt.add (lastEmit, ms)
        nxt.add (ms, me)
        lastEmit = me
        offset = me
      if lastEmit < pieceHi:
        nxt.add (lastEmit, pieceHi)
    level = system.move(nxt)
  result = level

proc bpeOnlyBt(engine: BpeEngine, bt: var BacktrackBuf,
    pieces: seq[tuple[lo, hi: int]], text: string): seq[int] =
  ## Ordinary backtracking tail (the wired ordinary encoder) over pre-computed pieces.
  var buf: seq[int] = @[]
  let data = text.toOpenArrayByte(0, text.len - 1)
  for (lo, hi) in pieces:
    buf.setLen(0)
    encodeSegment(engine, buf, bt, data, lo, hi)
    for x in buf.items:
      result.add x

proc drainPipeline(pipe: TokPipeline, text: sink string): seq[int] =
  pipe.resetText(text)
  for id in pipe.items:
    result.add id

proc main() =
  var reps = 7
  if paramCount() > 0:
    reps = parseInt(paramStr(1))
  doAssert reps >= 5, "the board is a median of at least 5 reps"
  echo "[bench-kimik2.5] full-pipeline board, reps=", reps

  let codec = loadTiktokenCodec(TokenizersDir / "kimik2.5.tiktoken")
  var cache: PreTokRegexCache
  # ONE pipeline for the whole board:
  #   its engine builds once, the loud load receipts print at the first
  #   encode below, every rep reuses it through resetText
  #   (the machine protocol's zero-alloc steady state, a fresh pipeline per rep would pay the full table build per rep).
  let pipe = TokPipeline.init(cache, codec.ranks, @[], @[], famKimiK25)
  # The reference walk runs on its own engine
  # (the pipeline's engine field is internal), both engines build once,
  # outside every timed window (the first reference encode and the first warm-up rep).
  let refEngine = BpeEngine.init(codec.ranks)

  for (name, file, prefixBytes) in Corpora.items:
    let text = readCorpusPrefix(CorpusDir / file, prefixBytes)
    let reference = bpeOnlyOrd(cache, codec.ranks, text)
    var walls: seq[float] = @[]
    var ids0: seq[int] = @[]
    for k in 0 ..< reps + 1: # rep 0 = warm-up, dropped
      let w0 = nowMs()
      let ids = drainPipeline(pipe, text)
      let w1 = nowMs()
      doAssert ids == reference, "id stream diverged from the reference"
      if k == 0:
        ids0 = ids
      else:
        walls.add w1 - w0
    walls.sort()
    let median = walls[walls.len div 2]
    let mbs = text.len.float / 1e6 / (median / 1e3)
    echo "[bench-kimik2.5] ", name, " bytes=", text.len,
      " ids=", ids0.len, " medianMs=", median.formatFloat(ffDecimal, 3),
      " minMs=", walls[0].formatFloat(ffDecimal, 3),
      " MB/s=", mbs.formatFloat(ffDecimal, 1),
      " (reference stream == every timed rep)"

    # stage split:
    #   scan-only wall (piece stream alone), bpe-only wall
    # (naive merge core over the pre-computed pieces, ids re-asserted)
    var scanWalls: seq[float] = @[]
    var scanPieces = 0
    for k in 0 ..< reps + 1:
      let w0 = nowMs()
      var r = initPreTokenizer(cache, famKimiK25, text)
      scanPieces = 0
      for (lo, hi) in r.items:
        inc scanPieces
      let w1 = nowMs()
      if k > 0:
        scanWalls.add w1 - w0
    scanWalls.sort()
    let scanMedian = scanWalls[scanWalls.len div 2]
    echo "[bench-kimik2.5] ", name, " scan-only pieces=", scanPieces,
      " medianMs=", scanMedian.formatFloat(ffDecimal, 3),
      " MB/s=", (text.len.float / 1e6 / (scanMedian / 1e3)).formatFloat(ffDecimal, 1)

    var scanFWalls: seq[float] = @[]
    var scanFPieces = 0
    for k in 0 ..< reps + 1:
      let w0 = nowMs()
      scanFPieces = frontierPieces(cache, famKimiK25, text).len
      let w1 = nowMs()
      if k > 0:
        scanFWalls.add w1 - w0
    doAssert scanFPieces == scanPieces, "frontier A/B row piece count diverged"
    scanFWalls.sort()
    let scanFMedian = scanFWalls[scanFWalls.len div 2]
    echo "[bench-kimik2.5] ", name, " scan-only-frontier pieces=", scanFPieces,
      " medianMs=", scanFMedian.formatFloat(ffDecimal, 3),
      " MB/s=", (text.len.float / 1e6 / (scanFMedian / 1e3)).formatFloat(ffDecimal, 1)

    var bpeWalls: seq[float] = @[]
    var pieces: seq[tuple[lo, hi: int]] = @[]
    block:
      var r = initPreTokenizer(cache, famKimiK25, text)
      for piece in r.items:
        pieces.add piece
    var bpeIds: seq[int] = @[]
    for k in 0 ..< reps + 1:
      let w0 = nowMs()
      bpeIds = bpeOnlyOrd(cache, codec.ranks, text)
      let w1 = nowMs()
      doAssert bpeIds == reference, "bpe-only stream diverged"
      if k > 0:
        bpeWalls.add w1 - w0
    bpeWalls.sort()
    let bpeMedian = bpeWalls[bpeWalls.len div 2]
    echo "[bench-kimik2.5] ", name, " bpe-only(naive) medianMs=",
      bpeMedian.formatFloat(ffDecimal, 3),
      " MB/s=", (text.len.float / 1e6 / (bpeMedian / 1e3)).formatFloat(ffDecimal, 1)

    var btWalls: seq[float] = @[]
    var bt = BacktrackBuf.init()
    var idsBt: seq[int] = @[]
    for k in 0 ..< reps + 1:
      let w0 = nowMs()
      idsBt.setLen(0)
      var scratch: seq[int] = @[]
      let data = text.toOpenArrayByte(0, text.len - 1)
      for (lo, hi) in pieces.items:
        scratch.setLen(0)
        encodeSegment(refEngine, scratch, bt, data, lo, hi)
        for x in scratch.items:
          idsBt.add x
      let w1 = nowMs()
      doAssert idsBt == reference, "bpe-only-bt stream diverged"
      if k > 0:
        btWalls.add w1 - w0
    btWalls.sort()
    let btMedian = btWalls[btWalls.len div 2]
    echo "[bench-kimik2.5] ", name, " bpe-only-bt medianMs=",
      btMedian.formatFloat(ffDecimal, 3),
      " MB/s=", (text.len.float / 1e6 / (btMedian / 1e3)).formatFloat(ffDecimal, 1)

    var m2Walls: seq[float] = @[]
    var btM2 = BacktrackBuf.init()
    var idsM2: seq[int] = @[]
    for k in 0 ..< reps + 1:
      let w0 = nowMs()
      idsM2 = bpeOnlyBt(refEngine, btM2, frontierPieces(cache, famKimiK25, text), text)
      let w1 = nowMs()
      doAssert idsM2 == reference, "frontier-scan + backtracking-tail stream diverged"
      if k > 0:
        m2Walls.add w1 - w0
    m2Walls.sort()
    let m2Median = m2Walls[m2Walls.len div 2]
    echo "[bench-kimik2.5] ", name, " frontier-scan+bt-tail medianMs=",
      m2Median.formatFloat(ffDecimal, 3),
      " MB/s=", (text.len.float / 1e6 / (m2Median / 1e3)).formatFloat(ffDecimal, 1)

when isMainModule:
  main()
