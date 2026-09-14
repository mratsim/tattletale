# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Top-level pipeline machine for toktoktok (the composed encoder):
## input text -> special-scan decisions -> per-ordinary-region
## pre-tokenization -> BPE ids yielded element by element.
##
## Reference structure mirrored:
## - toktoktok encodeWithSpecialTokens emits one id per special token
##   and encodes each ordinary slice independently through encodeOrdinaryImpl.
## - here the special-scan machine emits the same decisions, each
##   ordinary decision feeds one pre-tokenization machine restricted
##   to that region (a pattern never sees across a special token, exactly like the predecessor's per-slice PCRE2 subject),
##   and each pre-token piece is encoded by the BPE tail stage
##   (encodeSegment over the double-array vocab trie with the PairIndex merge structures).
## - the pipeline machine composes the three stages by hand, one scan
##   decision in flight at a time, its ids yielded to the consumer
##   before the next decision is taken
##   (the machine-flattening rule of the machine protocol, a driving machine keeps one upstream element in flight instead of stacking adapters).
##
## Machine shape (src/machine.nim):
## - object + ctor + ONE items.
## - no collect proc, the one-shot convenience is a plain for loop in the tests.
## - zero-alloc steady state, the piece buffer and merge scratch are
##   machine-owned and reused across regions and reset calls.
##
## DEVIATION from the pure value-object machine shape, TokPipeline is
## a ref object:
## - the pre-tokenization stage must address the scan machine's
##   window buffer in place (its piece offsets index that shared buffer),
##   a value object would move when the ctor returns, so the interior
##   pointer would dangle.
## - the ref keeps the object stable at one address for its whole lifetime,
##   everything else about the machine shape (ONE items, no collect, zero-alloc hot path) holds.
##
## Region-restricted pre-tokenization:
## - the stage-3 machine covers a whole input string, the pipeline
##   needs the [lo, hi) window of the scan machine's own buffer
##   (decision offsets index scan.win).
## - RegionPreTok holds a pointer to that shared buffer (no copy),
##   seeds its level buffers with the region instead of the whole input,
##   and applies the family chain with the same window discipline
##   (winLo/winHi = piece bounds, so the dollar horizon binds at the region edge exactly like the predecessor's per-slice PCRE2 subject).
## - the default regex scan is the pattern-split one where the step
##   carries it.
##
## Buffer validity and feeding:
## - decision offsets index the scan machine's window and stay valid
##   until the next feed. The pipeline holds at most one ordinary
##   decision in flight, so feed requires no decision in flight
##   (asserted, a caller drains the machine before the next feed).
## - whole-input mode moves the caller's text into the scan window
##   (zero copy), chunked mode carries undecided bytes across feeds
##   per the carry-buffer discipline.
## - ordinary stretches produce no decision until the next special
##   or the stream end (decisions are emitted only once final), so a chunked
##   stream with no specials mid-stream starts emitting ids at finish
##   (the stage-2 decision-finality design, chunking-independent, chunk-boundary invariance is a suite-level invariant row).
##
## Byte-level note:
## - the vocabularies of the staged checkpoints are byte rank tables
##   (HF byte-level vocabularies decoded to raw bytes by the HF conversion), so no byte-to-unicode remap stage participates in the encode path.
## - normalization is the identity for these checkpoints.
## - bytelevel/normalization machines stay out of the encode path.

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

proc applyStep(input: string, pieceLo, pieceHi: int, step: SplitStep,
    outPieces: var seq[tuple[lo, hi: int32]]) {.inline.} =
  ## One Isolated chain step over one piece, the stage-3 step
  ## application mirror (PCRE2-scan-mirror segmentation, window = piece):
  ## - the default regex scan is the pattern-split one where the step
  ##   carries it, the frontier-engine scan otherwise.
  case step.kind
  of skRegex:
    if step.split != nil:
      scanSplit(step.split, input, pieceLo, pieceHi, outPieces)
      return
    var lastEmit = pieceLo
    var offset = pieceLo
    while offset < pieceHi:
      let (ms, me) = step.pat.nextMatch(input, offset, pieceLo, pieceHi)
      if ms < 0:
        break
      if ms > lastEmit:
        outPieces.add (int32(lastEmit), int32(ms))
      outPieces.add (int32(ms), int32(me))
      lastEmit = me
      offset = me
    if lastEmit < pieceHi:
      outPieces.add (int32(lastEmit), int32(pieceHi))
  of skSpaceMergedPrev:
    var runStart = pieceLo
    var i = pieceLo
    while i < pieceHi:
      if input[i] == ' ':
        if i == pieceLo or input[i - 1] == ' ':
          if runStart < i:
            outPieces.add (int32(runStart), int32(i))
          outPieces.add (int32(i), int32(i + 1))
        else:
          outPieces.add (int32(runStart), int32(i + 1))
        runStart = i + 1
      inc i
    if runStart < pieceHi:
      outPieces.add (int32(runStart), int32(pieceHi))

proc buildPieces(r: var RegionPreTok) =
  ## Applies the Isolated chain over the region, level by level, reusing
  ## the scratch buffers (mirror of the stage-3 build, seeded with the region instead of the whole input).
  r.pieces.setLen(0)
  r.scratchA.setLen(0)
  r.scratchA.add (int32(r.regionLo), int32(r.regionHi))
  for level in 0 ..< r.steps.len:
    let step = r.steps[level]
    if level mod 2 == 0:
      r.scratchB.setLen(0)
      for piece in r.scratchA.items:
        applyStep(r.host[], piece[0], piece[1], step, r.scratchB)
    else:
      r.scratchA.setLen(0)
      for piece in r.scratchB.items:
        applyStep(r.host[], piece[0], piece[1], step, r.scratchA)
  if r.steps.len mod 2 == 1:
    r.pieces = system.move(r.scratchB)
  else:
    r.pieces = system.move(r.scratchA)
  r.built = true

proc initRegionPreTok*(steps: seq[SplitStep], host: ptr string): RegionPreTok =
  ## Empty-region placeholder stage (rebound per ordinary decision).
  RegionPreTok(host: host, steps: steps, regionLo: 0, regionHi: 0)

proc rebind*(r: var RegionPreTok, regionLo, regionHi: int) {.inline.} =
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
  ## Composed encoder machine with interior pointer stability:
  ## - one scan decision in flight at a time, its piece ids stream
  ##   through the piece buffer to the consumer.
  ## - the ref shape is the flagged deviation from the value-object
  ##   machine shape (see the module header, the field contract follows).
  TokPipeline* = ref object
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

proc init*(_: type TokPipeline, ranks: Table[seq[byte], int],
    specialPatterns: openArray[string], specialIds: openArray[int],
    family: Family): TokPipeline =
  ## Builds the pipeline machine for one tokenizer configuration, build contract:
  ## - mergeable ranks, special dictionary in tie-priority order
  ##   (the predecessor's same-start tie rule keeps the first pattern declared, extract the order from the live codec table, never assume it),
  ##   family chain.
  new result
  result.scanner = SpecialScanner.init(specialPatterns, specialIds)
  result.engine = BpeEngine.init(ranks)
  result.scan = SpecialScan.init(result.scanner, "")
  result.host = addr result.scan.win
  result.pretok = initRegionPreTok(familySteps(family), result.host)
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
        p.engine.encodeSegment(p.bt,
          p.host[].toOpenArrayByte(0, p.host[].len - 1),
          piece.lo, piece.hi, p.buf)
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

proc scannerBuildMillis*(p: TokPipeline): float64 {.inline.} =
  ## Special-dictionary automaton build wall time.
  p.scanner.buildMillis()
