## Tests for PagedRadixTrie KV Cache
##
## Each test is a standalone proc so that ARC/ORC destructors fire
## immediately after the proc returns (stack frame cleanup).

import
  std/unittest,
  std/math,
  std/importutils,
  ../../src/stateful/kvcache {.all.}

privateAccess(PagedRadixNode)
privateAccess(KVCache)

# ── Local helpers ──────────────────────────────────────
func sumLeafCount[T, P](cs: openArray[PagedRadixNode[T, P]]): int32 =
  for c in cs: result += c.subtree_sum_leaves

func sumStagingLock[T, P](cs: openArray[PagedRadixNode[T, P]]): int32 =
  for c in cs: result += c.subtree_sum_locked

func makePages(nTokens: int): seq[int] =
  let nPages = ceilDiv(nTokens, 256)
  result = newSeq[int](nPages)
  for i in 0..<nPages: result[i] = i + 1

# ════════════════════════════════════════════════════════
# LPM
# ════════════════════════════════════════════════════════
proc testLPMEmptyTrie(): bool =
  var cache = KVCache[uint32, int].new()
  let r = cache.lpm(@[1'u32, 2, 3])
  doAssert r.pages.len == 0
  doAssert r.totalTokenMatched == 0
  doAssert r.lastLevelMatched == 0
  doAssert cache.root.children.len == 0
  doAssert cache.root.subtree_sum_locked == 1
  result = true

proc testLPMSameRootSecondCall(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  let r = cache.lpm(@[4'u32, 5, 6])
  doAssert r.totalTokenMatched == 0
  doAssert cache.root.children.len == 0
  result = true

proc testLPMFindsLeafChild(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  cache.graftPages(@[1'u32, 2, 3], [1])
  doAssert cache.root.subtree_sum_locked == 0
  let r = cache.lpm(@[1'u32, 2, 3])
  doAssert r.totalTokenMatched == 3
  doAssert r.pages == @[1]
  doAssert cache.root.children.len == 0
  doAssert cache.root.subtree_sum_locked == 1
  result = true

# ════════════════════════════════════════════════════════
# GraftPages
# ════════════════════════════════════════════════════════
proc testGraftPagesOnEmptyRoot(): bool =
  var cache = KVCache[uint32, int].new()
  let tokens = @[1'u32, 2, 3]
  discard cache.lpm(tokens)
  cache.graftPages(tokens, [1, 2, 3])
  doAssert cache.root.tokens == tokens
  doAssert cache.root.pages == @[1, 2, 3]
  doAssert cache.root.subtree_sum_pages == 3
  doAssert cache.root.subtree_sum_leaves == 1
  result = true

proc testGraftPagesMultipleCalls(): bool =
  var cache = KVCache[uint32, int].new()
  var tokens = @[1'u32, 2, 3]
  discard cache.lpm(tokens)
  cache.graftPages(tokens, [1])
  doAssert cache.root.pages == @[1]
  # Extend
  var tokens2 = @[1'u32, 2, 3, 7'u32]
  discard cache.lpm(tokens2)
  cache.graftPages(tokens2, [1, 2])
  doAssert cache.root.pages == @[1, 2]
  doAssert cache.root.tokens.len == 4
  result = true

proc testGraftPagesPropagatesDecode(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[42'u32])
  cache.graftPages(@[42'u32], [1])
  doAssert cache.root.subtree_oldest_decode == 1
  result = true

proc testGraftPagesReleasesLock(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  doAssert cache.root.subtree_sum_locked == 1
  cache.graftPages(@[1'u32, 2, 3], [1])
  doAssert cache.root.subtree_sum_locked == 0
  result = true

proc testGraftPagesWalksToRoot(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  cache.graftPages(@[1'u32, 2, 3], [1])
  var p: PagedRadixNode[uint32, int] = cache.root
  while p != nil:
    doAssert p.subtree_sum_locked == 0
    doAssert p.subtree_oldest_decode == 1
    p = p.parent
  result = true

# ════════════════════════════════════════════════════════
# Evict
# ════════════════════════════════════════════════════════
proc testEvictUnlockedLeaf(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  doAssert cache.root.children.len == 2
  cache.evict()
  doAssert cache.root.subtree_sum_leaves == 1
  result = true

proc testEvictLockedLeaf(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  cache.root.children[0].subtree_sum_locked = 1
  cache.evict()
  doAssert cache.root.subtree_sum_leaves == 1
  cache.evict()
  result = true

proc testEvictEmptyTree(): bool =
  var cache = KVCache[uint32, int].new()
  cache.evict()
  result = true

proc testEvictPropagatesLeaves(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  doAssert cache.root.subtree_sum_leaves == 2
  cache.evict()
  doAssert cache.root.subtree_sum_leaves == 1
  result = true

# ════════════════════════════════════════════════════════
# findEvictionCandidate
# ════════════════════════════════════════════════════════
proc testFindCandidateUnlocked(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  cache.root.children[0].subtree_sum_locked = 1
  let victim = findEvictionCandidate(cache)
  doAssert victim != nil
  doAssert victim == cache.root.children[1]
  doAssert victim.evictable
  cache.root.children[0].subtree_sum_locked = 0
  result = true

proc testFindCandidateAllLocked(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  cache.root.children[0].subtree_sum_locked = 1
  cache.root.children[1].subtree_sum_locked = 1
  let victim = findEvictionCandidate(cache)
  doAssert victim == nil
  result = true

# ════════════════════════════════════════════════════════
# Invariants (A5, C2)
# ════════════════════════════════════════════════════════
proc testInvariantA5(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  doAssert cache.root.subtree_sum_leaves == sumLeafCount(cache.root.children)
  result = true

proc testInvariantC2(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  doAssert cache.root.subtree_sum_locked == sumStagingLock(cache.root.children)
  result = true

proc testInvariantA5AfterLPM(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  discard cache.lpm(@[1'u32, 2, 3])
  discard cache.lpm(@[4'u32, 5, 6])
  doAssert cache.root.subtree_sum_leaves == sumLeafCount(cache.root.children)
  result = true

proc testInvariantC2AfterLPM(): bool =
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3]); cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6]); cache.graftPages(@[4'u32, 5, 6], [2])
  doAssert cache.root.subtree_sum_locked == sumStagingLock(cache.root.children)
  result = true

# ════════════════════════════════════════════════════════
# Regression tests
# ════════════════════════════════════════════════════════
proc testRegressionWalkDownOOB(): bool =
  ## walkDown: root<256 tok matches all input, skips child check
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  cache.graftPages(@[1'u32, 2, 3], [1])
  let r = cache.lpm(@[1'u32, 2, 3])
  doAssert r.totalTokenMatched == 3
  result = true

proc testRegressionForkReturn(): bool =
  ## graftPages fork branch returns, doesn't fall through to append
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6])
  cache.graftPages(@[4'u32, 5, 6], [2])
  doAssert cache.root.children.len == 2
  result = true

proc testRegressionLPMPageSlice(): bool =
  ## LPM returns pages only for matched portion, not full node
  var cache = KVCache[uint32, int].new()
  var bigTokens = newSeq[uint32](1280)
  for i in 0..<1280: bigTokens[i] = uint32(i)
  discard cache.lpm(bigTokens)
  cache.graftPages(bigTokens, [1,2,3,4,5])
  var partial = newSeq[uint32](1024)
  for i in 0..<1024: partial[i] = uint32(i)
  let r = cache.lpm(partial)
  doAssert r.pages.len == 4
  result = true

proc testRegressionEvictCompressRerevict(): bool =
  ## evict + compressPath + re-evict from merged leaf
  var cache = KVCache[uint32, int].new()
  discard cache.lpm(@[1'u32, 2, 3])
  cache.graftPages(@[1'u32, 2, 3], [1])
  discard cache.lpm(@[4'u32, 5, 6])
  cache.graftPages(@[4'u32, 5, 6], [2])
  cache.evict()
  cache.evict()
  doAssert cache.root.subtree_sum_leaves == 0
  result = true

proc testRegressionMultiUserRootLinking(): bool =
  ## multi-user fork on root linking
  var cache = KVCache[uint32, int].new()
  for i in 0..<20:
    var tok = newSeq[uint32](256)
    for j in 0..<256:
      tok[j] = uint32(i * 256 + j)
    discard cache.lpm(tok)
    cache.graftPages(tok, [i+1])
  doAssert cache.root.children.len == 20
  result = true

proc testRegressionPartialMatchSiblingFork(): bool =
  ## Scenario: target has MORE content after the match point.
  ## The fork branch must handle it (not partial match).
  ##
  ## Uses page-aligned splits so fork's round_step_down works:
  ## 1. Insert A = [0..511] (2 pages) → root leaf
  ## 2. Insert B = [1000..1511] → fork at page 0: P→[A(512), B(512)]
  ## 3. Insert C = [0..255,2000,2001] (258 tok — match A's first 256)
  ##    Fork at page 1 boundary (lastLevelMatched=256 >= 256).
  ##
  ## Expected: P → { forkNode(256) → [target(256), sibling(2)], B(512) }
  var cache = KVCache[uint32, int].new()

  # Step 1: A = [0..511]
  var aFull = newSeq[uint32](512)
  for i in 0..<512: aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(512))
  doAssert cache.root.tokens.len == 512

  # Step 2: B = [1000..1511] — different, forces fork
  var bFull = newSeq[uint32](512)
  for i in 0..<512: bFull[i] = uint32(1000 + i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(512))
  doAssert cache.root.children.len == 2, "root should have 2 children after fork"

  # Step 3: C matches first 256 of A (full page), then diverges
  var cTok = newSeq[uint32](258)
  for i in 0..<256: cTok[i] = uint32(i)         # match A[0..255]
  cTok[256] = 2000; cTok[257] = 2001             # diverge
  discard cache.lpm(cTok)
  cache.graftPages(cTok, makePages(258))

  # LPM on C: should match all 258 tokens (shared 256 + sibling 2)
  let r = cache.lpm(cTok)
  doAssert r.totalTokenMatched == 258,
    "LPM matched " & $r.totalTokenMatched & ", expected 258"

  # LPM on A: should still match all 512
  let rA = cache.lpm(aFull)
  doAssert rA.totalTokenMatched == 512,
    "LPM on A matched " & $rA.totalTokenMatched & ", expected 512"

  # LPM on B: should still match all 512
  let rB = cache.lpm(bFull)
  doAssert rB.totalTokenMatched == 512,
    "LPM on B matched " & $rB.totalTokenMatched & ", expected 512"
  result = true

# ════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════
proc runTests*() =
  suite "LPM":
    test "LPM on empty trie returns root without creating children":
      check testLPMEmptyTrie()
    test "LPM second call returns root again, still no children":
      check testLPMSameRootSecondCall()
    test "LPM finds leaf child without creating extra children":
      check testLPMFindsLeafChild()

  suite "GraftPages":
    test "graftPages on empty root creates leaf":
      check testGraftPagesOnEmptyRoot()
    test "graftPages multiple calls accumulate on same leaf":
      check testGraftPagesMultipleCalls()
    test "graftPages propagates subtree_oldest_decode up":
      check testGraftPagesPropagatesDecode()
    test "graftPages releases lock and updates timestamps":
      check testGraftPagesReleasesLock()
    test "graftPages walks entire path to root":
      check testGraftPagesWalksToRoot()

  suite "Evict":
    test "evict unlocked leaf after graft":
      check testEvictUnlockedLeaf()
    test "cannot evict locked leaf":
      check testEvictLockedLeaf()
    test "evict on empty tree does not crash":
      check testEvictEmptyTree()
    test "evict propagates subtree_sum_leaves up":
      check testEvictPropagatesLeaves()

  suite "findEvictionCandidate":
    test "finds unlocked leaf when one exists":
      check testFindCandidateUnlocked()
    test "returns nil when all leaves locked":
      check testFindCandidateAllLocked()

  suite "Invariants (A5, C2)":
    test "A5 subtree_sum_leaves partition":
      check testInvariantA5()
    test "C2 subtree_sum_locked partition":
      check testInvariantC2()
    test "A5 after multiple LPMs (no tree mutation)":
      check testInvariantA5AfterLPM()
    test "C2 after multiple LPMs (locks balanced)":
      check testInvariantC2AfterLPM()

  suite "Regression (bugs found during PagedRadixTrie rewrite)":
    test "walkDown: root<256 tok matches all input, skips child check":
      check testRegressionWalkDownOOB()
    test "fork branch returns, doesn't fall through to append":
      check testRegressionForkReturn()
    test "LPM returns pages only for matched portion":
      check testRegressionLPMPageSlice()
    test "evict + compressPath + re-evict from merged leaf":
      check testRegressionEvictCompressRerevict()
    test "multi-user fork on root linking":
      check testRegressionMultiUserRootLinking()
    test "partial match sibling fork at correct level":
      check testRegressionPartialMatchSiblingFork()

when isMainModule:
  runTests()
