# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Top-level pipeline machine for toktoktok. Input text in, token ids out, one scan decision in flight at a time.
##
##   input text
##      │ scan machine (specials)
##      ├─ special decision ──► its id yields directly
##      └─ ordinary decision ──► RegionPreTok rebinds to the scan
##                                 machine's [lo, hi) window
##                                 │ pieces stream one at a time
##                                 └─► each piece encodes through the
##                                     BPE tail stage, ids yield before
##                                     the next decision
##
## RegionPreTok addresses the scan window in place. TokPipeline is a ref
## object that keeps the object at one address so the interior pointer does not dangle.

import std/[tables]

import ./machine
import workspace/regex_engine
import ./scan
import ./merge

type
  RegionPreTok* {.final.} = object
    ## Region-restricted pre-tokenization stage (internal helper, one per pipeline):
    ## - piece offsets index the host buffer (the scan machine's window),
    ##   the region bounds the pieces.
    ## - level buffers reused across rebinds.
    host*: ptr string
    steps*: seq[SplitStep]
    regionLo*, regionHi*: int
    pieces: seq[tuple[lo, hi: int32]]
    head: int
    built: bool
    scratchA, scratchB: seq[tuple[lo, hi: int32]]

proc buildPieces(r: var RegionPreTok) =
  ## Applies the Isolated chain over the region, level by level, reusing
  ## the scratch buffers, seeded with the region instead of the whole input.
  r.pieces.setLen(0)
  r.scratchA.setLen(0)
  r.scratchA.add (int32(r.regionLo), int32(r.regionHi))
  for level in 0 ..< r.steps.len:
    let step = r.steps[level]
    if level mod 2 == 0:
      r.scratchB.setLen(0)
      for piece in r.scratchA.items:
        applyStep(step, r.scratchB, r.host[], piece[0], piece[1])
    else:
      r.scratchA.setLen(0)
      for piece in r.scratchB.items:
        applyStep(step, r.scratchA, r.host[], piece[0], piece[1])
  if r.steps.len mod 2 == 1:
    r.pieces = system.move(r.scratchB)
  else:
    r.pieces = system.move(r.scratchA)
  r.built = true

proc initRegionPreTok(steps: seq[SplitStep], host: ptr string): RegionPreTok =
  ## Empty-region placeholder stage (rebound per ordinary decision).
  RegionPreTok(host: host, steps: steps, regionLo: 0, regionHi: 0)

proc rebind(r: var RegionPreTok, regionLo, regionHi: int) {.inline.} =
  ## Binds the stage to a new region of the same host buffer, reusing
  ## every buffer (zero-alloc steady state).
  r.regionLo = regionLo
  r.regionHi = regionHi
  r.head = 0
  r.built = false

iterator items*(r: var RegionPreTok): tuple[lo, hi: int] {.inline.} =
  ## Yields the pre-token pieces of the bound region as offsets into
  ## the host buffer.
  if not r.built:
    r.buildPieces()
  while r.head < r.pieces.len:
    let piece = r.pieces[r.head]
    inc r.head
    yield (int(piece.lo), int(piece.hi))

# ------------------------------------------------------------------------
# Pipeline machine
# ------------------------------------------------------------------------

type
  TokPipeline* = ref object
    ## Composed encoder machine with interior pointer stability:
    ## - one scan decision in flight at a time, its piece ids stream
    ##   through the piece buffer to the consumer.
    ## - the ref object keeps the machine at a stable address, so the interior
    ##   pointer never dangles.
    scanner: SpecialScanner
    engine: BpeEngine
    scan: SpecialScan
    pretok: RegionPreTok
    host: ptr string
    bt: BacktrackBuf
    buf: seq[int]
    pos: int
    mid: bool

proc drained*(p: TokPipeline): bool {.inline.} =
  ## True once the whole stream is encoded and every id was yielded.
  ## Further iteration yields nothing.
  (not p.mid) and p.pos >= p.buf.len and p.scan.drained()

proc takeRegion(p: TokPipeline, d: SpecialDecision) {.inline.} =
  ## Points the pre-tokenization stage at the next ordinary region,
  ## the stage is rebound in place (buffers reused), the piece buffer
  ## and position state reset, merge buffers and caches persist.
  p.pretok.rebind(d.lo, d.hi)
  p.mid = true

proc init*(_: type TokPipeline, cache: var PreTokRegexCache,
    ranks: Table[seq[byte], int], specialPatterns: openArray[string],
    specialIds: openArray[int], family: Family): TokPipeline =
  ## Builds the pipeline machine for one tokenizer configuration, build contract:
  ## - mergeable ranks, special dictionary in codec table order and family chain
  ##   (the longest match wins a same-start tie, extract the order from the live codec table, never assume it)
  new result
  result.scanner = SpecialScanner.init(specialPatterns, specialIds)
  result.engine = BpeEngine.init(ranks)
  result.scan = SpecialScan.init(result.scanner, "")
  result.host = addr result.scan.win
  result.pretok = initRegionPreTok(cache.steps(family), result.host)
  result.bt = BacktrackBuf.init()
  result.buf = newSeq[int](0)
  result.pos = 0
  result.mid = false

proc resetText*(p: TokPipeline, text: sink string) {.inline.} =
  ## Whole-input mode, the text moves into the scan window (zero copy),
  ## the stream is complete from the start.
  doAssert not p.mid and p.pos >= p.buf.len and
    not p.scan.decisionsQueued(),
    "pipeline restart with ids pending"
  p.scan = SpecialScan.init(p.scanner, text)
  p.pretok.host = addr p.scan.win
  p.buf.setLen(0)
  p.pos = 0
  p.mid = false

proc beginStream*(p: TokPipeline) {.inline.} =
  ## Chunked mode, feed chunks then finishStream. No decision may be
  ## in flight when the stream restarts.
  doAssert not p.mid and p.pos >= p.buf.len and
    not p.scan.decisionsQueued(),
    "pipeline restart with ids pending"
  p.scan = SpecialScan.init(p.scanner)
  p.pretok.host = addr p.scan.win
  p.buf.setLen(0)
  p.pos = 0
  p.mid = false

proc feed*(p: TokPipeline, chunk: openArray[char]) {.inline.} =
  ## Appends the next stream chunk (chunked mode), feed contract:
  ## - the caller must have drained the ids of every decision taken so
  ##   far (no decision in flight).
  ## - queued-but-unconsumed decisions are fine, their offsets stay
  ##   valid through the feed.
  doAssert not p.mid, "ordinary decision in flight: drain ids before feed"
  doAssert p.pos >= p.buf.len, "piece ids pending: drain before feed"
  p.scan.feed(chunk)

proc finishStream*(p: TokPipeline) {.inline.} =
  ## Marks the stream complete (chunked mode), idempotent.
  p.scan.finish()

iterator items*(p: TokPipeline): int {.inline.} =
  ## Yields the token id stream, the in-flight decision drains first,
  ## then scan decisions are taken one at a time, the yield contract:
  ## - special decisions yield their id directly.
  ## - ordinary decisions pre-tokenize their region and stream BPE ids
  ##   through the piece buffer.
  ## - partial consumption resumes from the machine's fields, position
  ##   state advances before each yield (the machine protocol).
  while true:
    if p.pos < p.buf.len:
      let id = p.buf[p.pos]
      inc p.pos
      yield id
      continue
    if p.mid:
      var tookPiece = false
      for piece in p.pretok.items:
        p.buf.setLen(0)
        p.engine.encodeSegment(p.buf, p.bt,
          p.host[].toOpenArrayByte(0, p.host[].len - 1),
          piece.lo, piece.hi)
        p.pos = 0
        tookPiece = true
        break
      if tookPiece:
        continue
      p.mid = false
      continue
    var tookDecision = false
    for d in p.scan.items:
      if d.specialId >= 0:
        yield d.specialId
      else:
        p.takeRegion(d)
      tookDecision = true
      break
    if tookDecision:
      continue
    break

