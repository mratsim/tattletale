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
##   nim c -r -d:release --nimcache:nimcache/wipM07trie_lpm workspace/tattleformers/tests/stateful/test_kvcache_lpm.nim
##

import std/unittest
import std/math
import std/importutils
import workspace/tattleformers/src/stateful/kvcache {.all.}

privateAccess(PagedRadixNode)
privateAccess(KVCache)

suite "LPM ,  pred/succ resolution with signed divergence point":

  test "Scenario 1: pred has longer match (pred=4, succ=2)":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, D, E, A]   (succ candidate)
    ## Query: [A, B, C, D, F]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 4: right, child0 recorded as the pred candidate
    ##   2. query < child1 at position 2: left, child1 recorded as the succ candidate
    ##
    ## pred match: 4 tokens long, the query prefix A B C D
    ## succ match: 2 tokens long, the query prefix A B
    ## Expected: child0 returned with a 4-token pred match
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
    check r.totalTokenMatched == 4  # A,B,C,D ,  should find child0's first 4 tokens

  test "Scenario 2: pred and succ have equal match length":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, C, F, F]   (succ candidate)
    ## Query: [A, B, C, E, F]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 3: right, child0 recorded as the pred candidate
    ##   2. query < child1 at position 3: left, child1 recorded as the succ candidate
    ##
    ## pred match: 3 tokens long, the query prefix A B C
    ## succ match: 3 tokens long, the run A,B,C
    ## Expected: either child returned with a 3-token match
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
    check r.totalTokenMatched == 3  # A,B,C ,  both children have same prefix

  test "Scenario 3: succ has longer match":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, C, F, F]   (succ candidate)
    ## Query: [A, B, C, F, E]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 3: right, child0 recorded as the pred candidate
    ##   2. query < child1 at position 4: left, child1 recorded as the succ candidate
    ##
    ## pred match: 3 tokens long, the query prefix A B C
    ## succ match: 4 tokens long, the run A,B,C,F
    ## Expected: child1 returned with a 4-token succ match
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
    check r.totalTokenMatched == 4  # A,B,C,F ,  should find child1's first 4 tokens

  test "Scenario 3b: succ has longer match, verify pages":
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
    check r.totalTokenMatched == 4
    check r.pages[0] == 2  # should get child1's page, not child0's

  test "Scenario 1b: pred has longer match, verify pages":
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
    check r.totalTokenMatched == 4
    check r.pages[0] == 1  # should get child0's page, not child1's

  test "Zero token match: query shares no prefix with any child":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [B, C, D, E, F]   (succ candidate)
    ## Query: [Z, X, Y, W, V]
    ##
    ## WAVL search at root:
    ##   1. query[0]=Z > child0[0]=A → right, record child0 as pred
    ##   2. query[0]=Z > child1[0]=B → right, record child1 as pred
    ## No succ recorded (query > all children, always went right).
    ## pred=child1 (last right turn), succ=nil.
    ## lpmCmp(query, child1): Z > B at position 0 → returns +1
    ## match = abs(+1) - 1 = 0
    ## bestLen stays 0 → return nil → LPM returns root with 0 matched.
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
    check r.totalTokenMatched == 0
    check r.pages.len == 0
