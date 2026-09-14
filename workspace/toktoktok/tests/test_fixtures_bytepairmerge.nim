# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Byte-pair merge fixture rows drive the 15 recorded
## piece/rank/expected rows through the merge cores over the static
## structures (bytePairEncodePQ = the naive merge-core mirror, encodeSegmentPQ / encodeSegment = the segment entry):
## - row expectations are the recorded ids, rows the engine
##   structure cannot serve as-recorded carry the documented engine
##   behavior instead, each marked in its row comment.
## - a rank table with no derivable split halves (a mergeable token whose halves are not tokens themselves) builds no PairIndex:
##   the row table extends by the 256 single bytes at the loader's
##   byte-filler ranks (1000000+i, the convertHfToTiktoken convention),
##   which leaves the merge dynamics of the recorded pairs untouched.
## - a rank table missing a single byte of the piece, the priority-
##   queue core mirrors the naive KeyError as BpeError, the backtracking
##   core emits the tokens placed before its dead end.
## - a 1-byte piece through the naive core carries the double-add
##   quirk ([id, id]), the segment entry's whole-piece branch emits it once.
## - a rank table with no token for any piece byte cannot build
##   the engine at all (empty pair set), the row runs over a minimal
##   constructible seed table and asserts the BpeError mirror.
## The bt == PQ == recorded equality on served vocabularies is
## the cross-check suite's contract (tests/fuzzing/t_bpe.nim),
## synthetic tables may legitimately diverge
## (the emoji row carries both hand-computed segmentations).
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/toktoktok/tests/test_fixtures_bytepairmerge.nim

import std/unittest
import std/base64
import std/tables
import std/os
import std/sequtils
import std/json
import std/strutils

import workspace/toktoktok/tests/fuzzing/pq_bpe_reference
import workspace/toktoktok/src/merge
import workspace/zstd/zstd_highlevel

const FixturePath = currentSourcePath().parentDir() /
  "fixtures" / "bytepairmerge" / "bytepairmerge"

proc b64DecodeToBytes(b64_str: string): seq[byte] =
  let decoded = decode(b64_str)
  result = newSeq[byte](decoded.len)
  for i in 0..<decoded.len:
    result[i] = byte(decoded[i])

type
  Fixture = object
    description: string
    inputBytes: seq[byte]
    ranks: Table[seq[byte], int]
    expectedTokens: seq[int]

proc parseFixture(node: JsonNode): Fixture =
  result.description = node["description"].getStr()
  result.inputBytes = node["input_bytes"].getElems().mapIt(it.getInt().byte)
  result.ranks = initTable[seq[byte], int]()
  for k, v in node["ranks"]:
    let keyBytes = b64DecodeToBytes(k)
    result.ranks[keyBytes] = v.getInt()
  result.expectedTokens = node["expected_tokens"].getElems().mapIt(it.getInt())

proc addByteFillers(ranks: var Table[seq[byte], int]) =
  ## Byte-filler ranks for the 256 single bytes:
  ## - makes a recorded table structurally constructible (every mergeable token's split halves become tokens).
  ## - the recorded mergeable ranks stay untouched, their values stay
  ##   below every filler rank.
  for b in 0 ..< 256:
    let key = @[byte(b)]
    if not ranks.hasKey(key):
      ranks[key] = 1000000 + b

proc addConstructibleSeed(ranks: var Table[seq[byte], int]) =
  ## Minimal table with one derivable split pair:
  ## - a PairIndex needs at least one mergeable token whose halves are tokens.
  ## - the seed pair's ranks sit above the recorded ranks and below the byte fillers,
  ##   so no merge on a recorded piece can select them.
  ranks[@[byte(0)]] = 5
  ranks[@[byte(1)]] = 6
  ranks[@[byte(0), byte(1)]] = 7

proc encodePQDirect(e: BpeEngine, piece: seq[byte]): seq[int] =
  ## Naive merge-core mirror over the whole piece
  ## (bytePairEncodePQ, the direct whole-piece call shape).
  var m = MergeBuf.init()
  bytePairEncodePQ(e, m, piece, 0, piece.len, result)

proc encodeSegPQ(e: BpeEngine, piece: seq[byte]): seq[int] =
  var m = MergeBuf.init()
  encodeSegmentPQ(e, m, piece, 0, piece.len, result)

proc encodeSegBt(e: BpeEngine, piece: seq[byte]): seq[int] =
  var bt = BacktrackBuf.init()
  encodeSegment(e, bt, piece, 0, piece.len, result)

proc expectBpeError(body: proc()) =
  ## Asserts the call raises BpeError:
  ##   the priority-queue core's mirror
  ## of the naive output loop's unranked-span KeyError.
  var raised = false
  try:
    body()
  except BpeError:
    raised = true
  check raised

proc runBytePairMergeTests() =
  let content = readFile(FixturePath & ".json.zst").zstdDecompress(string)
  let fixtureNodes = parseJson(content).getElems()

  for fixtureNode in fixtureNodes:
    let fixture = parseFixture(fixtureNode)

    test fixture.description:
      let piece = fixture.inputBytes

      case fixture.description
      of "Simple ASCII text 'hello'", "Simple ASCII 'the'",
          "Multiple newlines '\\n\\n\\n'",
          "Code-like 'def foo():\\n    return 42'",
          "Two identical bytes '\\x00\\x00'", "High bytes (above 127)",
          "Math symbols '𝜑² + 𝜑 + 1 ≡ 0'":
        # every split half of every mergeable token is itself a token,
        # so the recorded table builds as-is and all three encoders
        # emit the recorded ids
        let e = BpeEngine.init(fixture.ranks)
        check encodePQDirect(e, piece) == fixture.expectedTokens
        check encodeSegPQ(e, piece) == fixture.expectedTokens
        check encodeSegBt(e, piece) == fixture.expectedTokens

      of "Chinese '你好世界' -你好 world",
          "Single Chinese character '界' (one 3-byte UTF-8 char split across merges)",
          "Greek text 'Ελληνικά'":
        # the recorded tokens' split halves are absent as tokens (no PairIndex),
        # with the byte fillers the merge dynamics stay the recorded
        # ones (the recorded pairs decide, the fillers never win a rank comparison),
        # and all three encoders emit the recorded ids
        var ranks = fixture.ranks
        ranks.addByteFillers()
        let e = BpeEngine.init(ranks)
        check encodePQDirect(e, piece) == fixture.expectedTokens
        check encodeSegPQ(e, piece) == fixture.expectedTokens
        check encodeSegBt(e, piece) == fixture.expectedTokens

      of "Single byte 'a'":
        # the recorded table is single-byte only (no pair set)
        # and the piece byte is ranked:
        #   the segment entry's whole-piece branch
        # emits the recorded id once, the naive core adds it before
        # the merge walk and again at the output walk, the recorded
        # double-add quirk (the naive core emits [33, 33] against the recorded [33])
        var ranks = fixture.ranks
        ranks.addByteFillers()
        ranks.addConstructibleSeed()
        let e = BpeEngine.init(ranks)
        check encodeSegPQ(e, piece) == fixture.expectedTokens
        check encodeSegBt(e, piece) == fixture.expectedTokens
        check encodePQDirect(e, piece) == @[33, 33]

      of "Bytes with no merge ranks available":
        # an empty rank table builds no engine, over the minimal
        # constructible seed the piece bytes stay unranked and the naive
        # core's output walk raises on the first unranked span @[1]
        var ranks = initTable[seq[byte], int]()
        ranks.addConstructibleSeed()
        let e = BpeEngine.init(ranks)
        expectBpeError(proc() = discard encodePQDirect(e, piece))

      of "Chinese period + newline '。\\n' - combined 4-byte token regression":
        # the piece's first three bytes have no single-byte rank:
        # the naive core's output walk raises on the unranked (227,) span
        # (the naive core's missing-rank failure), the segment entry's
        # whole-piece branch emits the combined token id 10155 in both encoders
        # (the ordinary path can serve the piece the naive direct call cannot represent)
        let e = BpeEngine.init(fixture.ranks)
        expectBpeError(proc() = discard encodePQDirect(e, piece))
        check encodeSegPQ(e, piece) == @[10155]
        check encodeSegBt(e, piece) == @[10155]

      of "Chinese text with 。\\n combined token in context":
        # unranked bytes 131 and 136 sit mid-piece, the priority-queue
        # core raises at the output walk (missing-rank failure on @[131]),
        # through the direct call and through the segment entry alike,
        # the backtracking core emits the tokens placed before its dead end
        # ([60412, 229], the tail byte has no rank to continue the walk),
        # the documented dead-end semantics of a vocabulary missing single bytes
        let e = BpeEngine.init(fixture.ranks)
        expectBpeError(proc() = discard encodePQDirect(e, piece))
        expectBpeError(proc() = discard encodeSegPQ(e, piece))
        check encodeSegBt(e, piece) == @[60412, 229]

      of "Emoji '🌍' - Earth globe":
        # priority-queue rank order merges (159,140) first (rank 234),
        # consuming the (240,159) pair:
        #   rank [300, 234, 220].
        # The backtracking core's longest-match walk places (240,159) then
        # (140,141) instead:
        #   [12520, 235], the hand-computed greedy segmentation
        #   of this synthetic table
        #   (the two encoders' equality contract holds on the served families, exercised by the cross-check suite)
        var ranks = fixture.ranks
        ranks.addByteFillers()
        let e = BpeEngine.init(ranks)
        check encodePQDirect(e, piece) == fixture.expectedTokens
        check encodeSegPQ(e, piece) == fixture.expectedTokens
        check encodeSegBt(e, piece) == @[12520, 235]

      else:
        checkpoint("unclassified fixture row: " & fixture.description)
        fail()

when isMainModule:
  import std/[monotimes, times]
  let suiteWallStart = getMonoTime()
  runBytePairMergeTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
