# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   nim test_toktoktok  # from the worktree root
## Explicit dense-id and vocab-trie unit rows over named byte-sorted
## tables:
## - duplicate-rank dense id assignment (rank-ascending, byte-lexicographic tiebreak).
## - trie lookups over a named prefix chain (hits, misses, byte-slice lookups).
## - property/fuzz coverage in tests/fuzzing/t_dense_ids.nim and tests/fuzzing/t_vocab_trie.nim.

import std/[monotimes, times]

import workspace/toktoktok/src/merge

var checkFailures = 0

proc check(name: string, ok: bool, detail = "") =
  if ok:
    echo "PASS ", name
  else:
    echo "FAIL ", name, " ", detail
    checkFailures += 1

proc runTests*() =
  ## Entry point, runs this file's checks in order.
  block:
    # byte-sorted input a(0), ab(5), b(0), the rank-ascending order
    # is a, b (both rank 0, byte tiebreak), ab, dense ids a=0, b=1, ab=2
    var keys: seq[seq[byte]] = @[@[byte('a')], @[byte('a'), byte('b')],
                                 @[byte('b')]]
    var ranks: seq[int] = @[0, 5, 0]
    check "denseIdsOfSorted duplicate ranks 'a','ab','b' -> [0, 2, 1]",
      denseIdsOfSorted(keys, ranks) == @[int32 0, int32 2, int32 1]

  block:
    # no duplicate ranks:
    #   dense ids are the rank-ascending order of the byte-sorted input
    var keys: seq[seq[byte]] = @[@[byte('a')], @[byte('a'), byte('b')],
                                 @[byte('b')]]
    var ranks: seq[int] = @[2, 0, 1]
    check "denseIdsOfSorted distinct ranks -> [2, 0, 1]",
      denseIdsOfSorted(keys, ranks) == @[int32 2, int32 0, int32 1]

  block:
    # the trie is valued in the caller's id space (the engine passes dense ids):
    #   keys sorted a, ab, abc, b with dense ids 0..3
    var keys: seq[seq[byte]] = @[@[byte('a')], @[byte('a'), byte('b')],
                                 @[byte('a'), byte('b'), byte('c')],
                                 @[byte('b')]]
    var ranks: seq[int] = @[0, 1, 2, 3]
    let t = buildTrie(keys, ranks)
    check "trie lookups: hits a=0 ab=1 abc=2 b=3",
      t.lookup("a") == 0 and t.lookup("ab") == 1 and t.lookup("abc") == 2 and
      t.lookup("b") == 3
    check "trie lookups: misses abd, ba, empty hit",
      t.lookup("abd") == -1 and t.lookup("ba") == -1 and
      t.lookup("") == -1

  if checkFailures > 0:
    echo "\n", checkFailures, " check(s) failed"
    quit(1)
  echo "\nall merge-table/trie unit rows passed"

when isMainModule:
  let suiteWallStart = getMonoTime()
  runTests()
  echo "\nwall ", (getMonoTime() - suiteWallStart).inMilliseconds.float64 / 1000.0, " s"
