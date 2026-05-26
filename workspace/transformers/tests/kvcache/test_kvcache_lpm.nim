import std/unittest
import std/math
import std/importutils
import ../../src/stateful/kvcache {.all.}

privateAccess(PagedRadixNode)
privateAccess(KVCache)

suite "LPM — pred/succ resolution with signed divergence point":

  test "Scenario 1: pred has longer match (pred=4, succ=2)":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, D, E, A]   (succ candidate)
    ## Query: [A, B, C, D, F]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 4 (F vs E) → right, record child0 as pred
    ##   2. query < child1 at position 2 (C vs D) → left (nil), record child1 as succ
    ##
    ## pred match length: 4  (A,B,C,D match)
    ## succ match length: 2  (A,B match)
    ## Expected: return child0 (pred, match=4)
    var cache = KVCache[uint32, int].new()
    let child0 = @[0'u32, 1, 2, 3, 4]  # A,B,C,D,E
    let child1 = @[0'u32, 1, 3, 4, 0]  # A,B,D,E,A
    let query  = @[0'u32, 1, 2, 3, 5]  # A,B,C,D,F

    discard cache.lpm(child0)
    cache.graftPages(child0, @[1])
    discard cache.lpm(child1)
    cache.graftPages(child1, @[2])

    let r = cache.lpm(query)
    check r.totalTokenMatched == 4  # A,B,C,D — should find child0's first 4 tokens

  test "Scenario 2: pred and succ have equal match length":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, C, F, F]   (succ candidate)
    ## Query: [A, B, C, E, F]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 3 (E vs D) → right, record child0 as pred
    ##   2. query < child1 at position 3 (E vs F) → left (nil), record child1 as succ
    ##
    ## pred match length: 3  (A,B,C match)
    ## succ match length: 3  (A,B,C match)
    ## Expected: return either child (match=3)
    var cache = KVCache[uint32, int].new()
    let child0 = @[0'u32, 1, 2, 3, 4]  # A,B,C,D,E
    let child1 = @[0'u32, 1, 2, 5, 5]  # A,B,C,F,F
    let query  = @[0'u32, 1, 2, 4, 5]  # A,B,C,E,F

    discard cache.lpm(child0)
    cache.graftPages(child0, @[1])
    discard cache.lpm(child1)
    cache.graftPages(child1, @[2])

    let r = cache.lpm(query)
    check r.totalTokenMatched == 3  # A,B,C — both children have same prefix

  test "Scenario 3: succ has longer match":
    ## Children:
    ##   0: [A, B, C, D, E]   (pred candidate)
    ##   1: [A, B, C, F, F]   (succ candidate)
    ## Query: [A, B, C, F, E]
    ##
    ## WAVL search at root:
    ##   1. query > child0 at position 3 (F vs D) → right, record child0 as pred
    ##   2. query < child1 at position 4 (E vs F) → left (nil), record child1 as succ
    ##
    ## pred match length: 3  (A,B,C match)
    ## succ match length: 4  (A,B,C,F match)
    ## Expected: return child1 (succ, match=4)
    var cache = KVCache[uint32, int].new()
    let child0 = @[0'u32, 1, 2, 3, 4]  # A,B,C,D,E
    let child1 = @[0'u32, 1, 2, 5, 5]  # A,B,C,F,F
    let query  = @[0'u32, 1, 2, 5, 4]  # A,B,C,F,E

    discard cache.lpm(child0)
    cache.graftPages(child0, @[1])
    discard cache.lpm(child1)
    cache.graftPages(child1, @[2])

    let r = cache.lpm(query)
    check r.totalTokenMatched == 4  # A,B,C,F — should find child1's first 4 tokens

  test "Scenario 3b: succ has longer match, verify pages":
    var cache = KVCache[uint32, int].new()
    let child0 = @[0'u32, 1, 2, 3, 4]  # page 1
    let child1 = @[0'u32, 1, 2, 5, 5]  # page 2
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
    let child0 = @[0'u32, 1, 2, 3, 4]  # page 1
    let child1 = @[0'u32, 1, 3, 4, 0]  # page 2
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
    let child0 = @[0'u32, 1, 2, 3, 4]   # A,B,C,D,E
    let child1 = @[1'u32, 2, 3, 4, 5]   # B,C,D,E,F
    let query  = @[25'u32, 23, 24, 22, 21]  # Z,X,Y,W,V

    discard cache.lpm(child0)
    cache.graftPages(child0, @[1])
    discard cache.lpm(child1)
    cache.graftPages(child1, @[2])

    let r = cache.lpm(query)
    check r.totalTokenMatched == 0
    check r.pages.len == 0
