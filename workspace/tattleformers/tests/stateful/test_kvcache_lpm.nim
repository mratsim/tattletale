# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## KVCache longest-prefix-match suite (CPU, model-free).
##
## Test that longest prefix match resolves pred and succ candidates with the signed divergence point.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" workspace/tattleformers/tests/stateful/test_kvcache_lpm.nim
##

import workspace/tattleformers/src/stateful/kvcache

proc testLpmPredLongerMatch(): bool =
  ## Children: 0: [A, B, C, D, E] (pred candidate), 1: [A, B, D, E, A] (succ candidate).
  ## Query: [A, B, C, D, F].
  ##
  ## WAVL search at root:
  ## 1. query > child0 at position 4: right, child0 recorded as the pred candidate
  ## 2. query < child1 at position 2: left, child1 recorded as the succ candidate
  ##
  ## pred match: 4 tokens long, the query prefix A B C D.
  ## Expected: child0 returned with a 4-token pred match.
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = A,B,C,D,E, child1 = A,B,D,E,A, query = A,B,C,D,F (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 3, 4, 0]
  let query  = @[0'u32, 1, 2, 3, 5]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 4, "A,B,C,D: should find child0's first 4 tokens"

proc testLpmPredSuccEqualMatch(): bool =
  ## Children: 0: [A, B, C, D, E] (pred candidate), 1: [A, B, C, F, F] (succ candidate).
  ## Query: [A, B, C, E, F].
  ##
  ## Both children diverge from the query at position 3: pred and succ match
  ## the same 3-token prefix A B C.
  ## Expected: either child returned with a 3-token match.
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = A,B,C,D,E, child1 = A,B,C,F,F, query = A,B,C,E,F (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 2, 5, 5]
  let query  = @[0'u32, 1, 2, 4, 5]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 3, "A,B,C: both children have same prefix"

proc testLpmSuccLongerMatch(): bool =
  ## Children: 0: [A, B, C, D, E] (pred candidate), 1: [A, B, C, F, F] (succ candidate).
  ## Query: [A, B, C, F, E].
  ##
  ## The succ candidate child1 matches 4 tokens (A,B,C,F), the pred candidate
  ## only 3 (A,B,C).
  ## Expected: child1 returned with a 4-token succ match.
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = A,B,C,D,E, child1 = A,B,C,F,F, query = A,B,C,F,E (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 2, 5, 5]
  let query  = @[0'u32, 1, 2, 5, 4]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 4, "A,B,C,F: should find child1's first 4 tokens"

proc testLpmSuccLongerMatchPages(): bool =
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = page 1, child1 = page 2 (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 2, 5, 5]
  let query  = @[0'u32, 1, 2, 5, 4]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 4
  doAssert r.pages[0] == 2, "should get child1's page, not child0's"

proc testLpmPredLongerMatchPages(): bool =
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = page 1, child1 = page 2 (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 3, 4, 0]
  let query  = @[0'u32, 1, 2, 3, 5]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 4
  doAssert r.pages[0] == 1, "should get child0's page, not child1's"

proc testLpmZeroTokenMatch(): bool =
  ## Children: 0: [A, B, C, D, E], 1: [B, C, D, E, F].
  ## Query: [Z, X, Y, W, V].
  ##
  ## WAVL search at root:
  ## 1. query[0]=Z > child0[0]=A: right, child0 recorded as pred
  ## 2. query[0]=Z > child1[0]=B: right, child1 recorded as pred
  ## No succ recorded (query > all children, always went right).
  ## lpmCmp(query, child1): Z > B at position 0, returns +1.
  ## match = abs(+1) - 1 = 0, bestLen stays 0: no child matches.
  ## Expected: root with 0 tokens matched and no pages.
  result = true
  var cache = KVCache[uint32, int].new()
  # child0 = A,B,C,D,E, child1 = B,C,D,E,F, query = Z,X,Y,W,V (letters are token values)
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[1'u32, 2, 3, 4, 5]
  let query  = @[25'u32, 23, 24, 22, 21]

  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  let r = cache.lpm(query)
  doAssert r.totalTokenMatched == 0
  doAssert r.pages.len == 0

proc runAll() =
  doAssert testLpmPredLongerMatch()
  doAssert testLpmPredSuccEqualMatch()
  doAssert testLpmSuccLongerMatch()
  doAssert testLpmSuccLongerMatchPages()
  doAssert testLpmPredLongerMatchPages()
  doAssert testLpmZeroTokenMatch()

when isMainModule:
  runAll()
