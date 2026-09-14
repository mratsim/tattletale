# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Explicit BPE merge unit rows over one hand-built rank table,
## named table, named piece, exact id expectations:
## - the whole-piece hit beating the merge path.
## - one- and two-merge pieces.
## - the leftmost-pair rule on equal ranks.
## - the 1-byte double-add quirk.
## - the unknown-byte BpeError (the naive KeyError mirror).
##
## Deliberately NOT a dense permutation (duplicate rank 2 on 'ab'/'ba'),
## the rank table keeps the tiktoken-shape asserts
## of buildMergeTables out of the way. Fuzz coverage lives in the paired suite tests/fuzzing/t_pq_reference.nim.

import std/[tables, monotimes, times]

import workspace/toktoktok/tests/fuzzing/pq_bpe_reference
import workspace/toktoktok/src/merge

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc encodePQ(e: BpeEngine, text: string): seq[int] =
  var m = MergeBuf.init()
  bytePairEncodePQ(e, m, text.toOpenArrayByte(0, text.len - 1), 0,
      text.len, result)

proc encodeSegmentPq1(e: BpeEngine, text: string): seq[int] =
  var m = MergeBuf.init()
  encodeSegmentPQ(e, m, text.toOpenArrayByte(0, text.len - 1), 0,
      text.len, result)

proc encodeSegment1(e: BpeEngine, text: string): seq[int] =
  ## Bt counterpart of encodeSegmentPq1 over the same hand table:
  ## - distinct ranks required, the bt validity predicate is rank-only,
  ##   the naive/PQ tie rule is rank+leftmost, so duplicate-rank tables
  ##   are outside bt's contract.
  ## - 'bab' on this table is the recorded divergence row, PQ [2, 1]
  ##   (leftmost (b,a) merge) vs bt [1, 2].
  var bt = BacktrackBuf.init()
  encodeSegment(e, bt, text.toOpenArrayByte(0, text.len - 1), 0,
      text.len, result)

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
    check "encodeSegmentPQ whole piece 'ab' -> [2]",
      encodeSegmentPq1(e, "ab") == @[2]
    check "encodeSegmentPQ whole piece beats the merge path ('aba' -> [5])",
      encodeSegmentPq1(e, "aba") == @[5]
    check "encodeSegmentPQ whole piece 'ba' -> [2]",
      encodeSegmentPq1(e, "ba") == @[2]

  block:
    check "bytePairEncodePQ 'ab' one merge -> [2]", encodePQ(e, "ab") == @[2]
    check "bytePairEncodePQ 'ba' one merge -> [2]", encodePQ(e, "ba") == @[2]
    check "bytePairEncodePQ 'aba' two merges -> [5]",
      encodePQ(e, "aba") == @[5]
    check "bytePairEncodePQ 'abab' two disjoint merges -> [2, 2]",
      encodePQ(e, "abab") == @[2, 2]

  block:
    # equal ranks on (b,a) and (a,b):
    #   the heap pops slot 0
    # first (the naive leftmost rule) -> [ba, b] -> [2, 1]
    check "bytePairEncodePQ 'bab' leftmost tie -> [2, 1]",
      encodePQ(e, "bab") == @[2, 1]

  block:
    # 1-byte double-add quirk:
    #   direct bytePairEncodePQ emits the id twice
    check "bytePairEncodePQ 'a' double-add quirk -> [0, 0]",
      encodePQ(e, "a") == @[0, 0]

  block:
    # bt counterpart of the ordinary rows above
    # (the whole-piece branch and the disjoint-merge piece, distinct ranks hold on every row)
    check "encodeSegment whole piece 'ab' -> [2]",
      encodeSegment1(e, "ab") == @[2]
    check "encodeSegment whole piece beats the merge path ('aba' -> [5])",
      encodeSegment1(e, "aba") == @[5]
    check "encodeSegment 'abab' two disjoint merges -> [2, 2]",
      encodeSegment1(e, "abab") == @[2, 2]

  block:
    var raised = false
    try:
      discard encodePQ(e, "q")
    except BpeError:
      raised = true
    check "bytePairEncodePQ unknown byte raises BpeError", raised

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall bpe unit rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
