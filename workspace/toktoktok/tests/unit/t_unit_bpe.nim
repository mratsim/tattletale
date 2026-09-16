# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
##
## Explicit BPE merge unit rows for the served lazy-DP encoder
## (src/merge.nim encodeSegment) over one hand-built rank table:
## - named table, named piece, exact id expectations
## - the whole-piece hit beating the merge path
## - the two-disjoint-merges piece
##
## The rank table is deliberately not a dense permutation:
## - duplicate rank 2 on 'ab'/'ba' keeps the tiktoken-shape asserts
##   of buildMergeTables out of the way
## - the distinct-rank contract of the validity predicate is noted
##   on the encodeSegment helper

import std/[tables, monotimes, times]

import workspace/toktoktok/src/merge

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    raise newException(AssertionDefect,
      "FAIL " & name & (if detail.len > 0: " " & detail else: ""))

proc encodeSegment1(e: BpeEngine, text: string): seq[int] =
  ## Served encoder over the hand table.
  ## - distinct ranks are the contract of the bt validity predicate
  ##   (rank-only, the naive merge tie rule is rank+leftmost).
  ## - the rows take whole-piece hits or disjoint merges, where the duplicate
  ##   rank never selects.
  var bt = BacktrackBuf.init()
  encodeSegment(e, result, bt, text.toOpenArrayByte(0, text.len - 1), 0,
      text.len)

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  # hand-built rank table:
  # - ranks chosen with a duplicate so the dense order
  #   exercises the byte-lexicographic tiebreak.
  # - byte-sorted key order, a(0), ab(2), aba(5), b(1), ba(2)
  # - dense id order, a=0, b=1, ab=2, ba=3, aba=4
  var ranks = initTable[seq[byte], int]()
  ranks[@[byte('a')]] = 0
  ranks[@[byte('b')]] = 1
  ranks[@[byte('a'), byte('b')]] = 2
  ranks[@[byte('b'), byte('a')]] = 2
  ranks[@[byte('a'), byte('b'), byte('a')]] = 5
  let e = BpeEngine.init(ranks)

  block:
    check "encodeSegment whole piece 'ab' -> [2]",
      encodeSegment1(e, "ab") == @[2]
    check "encodeSegment whole piece beats the merge path ('aba' -> [5])",
      encodeSegment1(e, "aba") == @[5]
    check "encodeSegment 'abab' two disjoint merges -> [2, 2]",
      encodeSegment1(e, "abab") == @[2, 2]

  echo "\nall bpe unit rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
