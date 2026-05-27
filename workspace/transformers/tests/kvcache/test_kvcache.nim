## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for PagedRadixTrie KV Cache
##
## Each test is a standalone proc so that ARC/ORC destructors fire
## immediately after the proc returns (stack frame cleanup).

import
  std/unittest,
  std/math,
  std/importutils,
  ../../src/stateful/kvcache {.all.},
  ../../src/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)

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

proc testRegressionLpmFourTokenCollision(): bool =
  ## Two children share tokens[0..3] but diverge at token[4].
  ## A 4-token WAVL key returns 0 (collision) → findChild returns wrong child.
  ## Full-page comparator resolves this correctly.
  var cache = KVCache[uint32, int].new()
  # Child A: [1,2,3,4,100,101,...,354]
  var aTok = newSeq[uint32](256)
  aTok[0] = 1; aTok[1] = 2; aTok[2] = 3; aTok[3] = 4
  for i in 4..<256: aTok[i] = uint32(100 + i)
  discard cache.lpm(aTok)
  cache.graftPages(aTok, makePages(256))
  # Child B: [1,2,3,4,200,201,...,454] — same first 4, diverges at [4]
  var bTok = newSeq[uint32](256)
  bTok[0] = 1; bTok[1] = 2; bTok[2] = 3; bTok[3] = 4
  for i in 4..<256: bTok[i] = uint32(200 + i)
  discard cache.lpm(bTok)
  cache.graftPages(bTok, makePages(256))
  doAssert cache.root.children.len == 2,
    "root should have 2 children after fork"
  # LPM for A must find A (not B), despite 4-token prefix collision
  let rA = cache.lpm(aTok)
  doAssert rA.totalTokenMatched == 256,
    "LPM on A matched " & $rA.totalTokenMatched & ", expected 256"
  # LPM for B must find B (not A)
  let rB = cache.lpm(bTok)
  doAssert rB.totalTokenMatched == 256,
    "LPM on B matched " & $rB.totalTokenMatched & ", expected 256"
  result = true

proc testRegressionCompressPathRecursiveAndReKey(): bool =
  ## P3: Recursive compressPath when grandparent becomes single-child.
  ##
  ## Build: root → [forkA(256) → [A_tail(512), C_tail(256)], B(256)]
  ## Evict through the tree. After recursive compress, root is a leaf.
  var cache = KVCache[uint32, int].new()

  # Step 1: Unique conversation A (3 pages = 768 tok)
  var seqA = newSeq[uint32](768)
  for i in 0..<768: seqA[i] = uint32(i)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(768))

  # Step 2: Unique conversation B (1 page = 256 tok, different first page)
  var seqB = newSeq[uint32](256)
  for i in 0..<256: seqB[i] = uint32(5000+i)
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(256))
  # root → [A(768), B(256)]

  # Step 3: C shares A's first 256 tokens, diverges after
  var seqC = newSeq[uint32](512)
  for i in 0..<256: seqC[i] = uint32(i)
  for i in 256..<512: seqC[i] = uint32(6000+i)
  discard cache.lpm(seqC)
  cache.graftPages(seqC, makePages(512))
  # root → [forkA(256) → [A_tail(512), C_tail(256)], B(256)]

  # Evict through the tree
  cache.evict()
  cache.evict()
  # After two evictions, root should be a leaf (recursive compress cleaned up)
  # No single-child nodes (E1 invariant)
  doAssert cache.root.children.len == 0
  doAssert cache.root.parent == nil
  result = true

proc testRegressionCompressPathPreservesTimestamp(): bool =
  ## Verify compressPath does NOT change subtree_oldest_decode — parent
  ## and child share the same timestamp from the last graftPages walk-up.
  ##
  ## Build: root → [forkA(256) → [A_tail, C_tail], B(256)]
  ## After evicting one child under forkA, compressPath fires. forkA's
  ## timestamp equals the absorbed child's (same walk-up path).
  var cache = KVCache[uint32, int].new()

  var seqA = newSeq[uint32](768)
  for i in 0..<768: seqA[i] = uint32(i)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(768))

  var seqB = newSeq[uint32](256)
  for i in 0..<256: seqB[i] = uint32(5000+i)
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(256))

  var seqC = newSeq[uint32](512)
  for i in 0..<256: seqC[i] = uint32(i)
  for i in 256..<512: seqC[i] = uint32(6000+i)
  discard cache.lpm(seqC)
  cache.graftPages(seqC, makePages(512))
  # root → [forkA(256) → [A_tail, C_tail], B(256)]

  # Lock B so eviction must descend through forkA
  cache.root.children[1].subtree_sum_locked = 1

  # Capture forkA's timestamp before compression
  let forkA = cache.root.children[0]
  let forkAoldestBefore = forkA.subtree_oldest_decode

  cache.evict()
  # forkA had 2 children, now has 1 → compressPath fires

  # compressPath sets parent.subtree_oldest_decode = only.subtree_oldest_decode.
  # But parent and only shared the same last graftPages walk-up (both were
  # set to kvClock in the same walkUpUpdate). The assignment is a no-op.
  doAssert forkA.subtree_oldest_decode == forkAoldestBefore,
    "compressPath must not change subtree_oldest_decode"
  result = true


# ════════════════════════════════════════════════════════
# classifyGraft unit test
# ════════════════════════════════════════════════════════
proc testClassifyGraft(): bool =
  ## Verify the 5-branch decision table.
  # gcFullMatch when targetMatchLen == tokensLen
  doAssert classifyGraft(10, 10, 10, 10, true, false) == gcFullMatch
  # gcPartialMatch: lastLevel < 256, hasParent, lastLevel == targetTokLen
  doAssert classifyGraft(200, 300, 200, 200, true, false) == gcPartialMatch
  # gcRootNewChild: lastLevel < 256, no parent, root has children
  doAssert classifyGraft(0, 100, 0, 0, false, true) == gcRootNewChild
  # gcFork: lastLevel < targetTokLen (not full match, not sub-page)
  doAssert classifyGraft(256, 512, 256, 512, true, false) == gcFork
  # gcAppend: fallthrough (lastLevel >= targetTokLen)
  doAssert classifyGraft(512, 768, 512, 512, true, false) == gcAppend
  result = true

# ════════════════════════════════════════════════════════
# appendOp leaves == 0 boundary
# ════════════════════════════════════════════════════════
proc testAppendOpLeavesZero(): bool =
  ## appendOp sets subtree_sum_leaves = 1 when it was 0.
  ## This happens after compressPath where the merged parent
  ## inherits the child's subtree_sum_leaves.
  var cache = KVCache[uint32, int].new()
  let tokA = makeTokens(256)
  let tokB = makeTokens(512)
  discard cache.lpm(tokA)
  cache.graftPages(tokA, makePages(256))
  discard cache.lpm(tokB)
  cache.graftPages(tokB, makePages(512))
  # Evict A — leaves B with 1 page at root
  cache.evict()
  # Root now has 1 child with subtree_sum_leaves = 0 after compressPath
  # Append to B should set leaves to 1
  let tokC = makeTokens(768)
  discard cache.lpm(tokC)
  cache.graftPages(tokC, makePages(768))
  doAssert cache.root.subtree_sum_leaves == 1,
    "appendOp should set subtree_sum_leaves to 1"
  result = true

# ════════════════════════════════════════════════════════
# evict root-with-no-parent
# ════════════════════════════════════════════════════════
proc testEvictRootWithNoParent(): bool =
  var cache = KVCache[uint32, int].new()
  let tokens = makeTokens(256)
  discard cache.lpm(tokens)
  cache.graftPages(tokens, makePages(256))
  # First evict removes child, root becomes empty leaf
  cache.evict()
  # Second evict: root is a leaf with no parent → replace root
  cache.evict()
  doAssert cache.root.subtree_sum_leaves == 0
  doAssert cache.root.pages.len == 0
  result = true

# ════════════════════════════════════════════════════════
# findEvictionCandidate wavlNext skip
# ════════════════════════════════════════════════════════
proc testEvictionCandidateSkipLocked(): bool =
  var cache = KVCache[uint32, int].new()
  # Different first tokens → separate children at root
  let seqA = @[0'u32] & makeTokens(255)
  let seqB = @[1'u32] & makeTokens(255)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(256))
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(256))
  # Lock seqA via LPM (locks persist without matching graftPages)
  discard cache.lpm(seqA)
  # findEvictionCandidate should skip locked seqA, find seqB
  let candidate = cache.findEvictionCandidate()
  doAssert candidate != nil, "should find an evictable leaf"
  doAssert candidate.tokens.len == 256
  doAssert not candidate.isLocked, "candidate must not be locked"
  result = true

# ════════════════════════════════════════════════════════
# walkUpUpdate WAVL re-key
# ════════════════════════════════════════════════════════
proc testWalkUpUpdateReKey(): bool =
  ## Verify that walkUpUpdate properly re-keys the eviction tree.
  ## After graftPages, the eviction tree should reflect the new
  ## subtree_oldest_decode ordering.
  var cache = KVCache[uint32, int].new()
  let seqA = makeTokens(256)
  let seqB = makeTokens(256)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(256))  # oldest_decode = 1
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(256))  # oldest_decode = 2
  # A is older than B. Eviction should evict A first.
  let candidate = cache.findEvictionCandidate()
  doAssert candidate != nil
  doAssert candidate.tokens == seqA, "oldest should be evicted first"
  result = true

# ════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════
proc runTests*() =
  suite "classifyGraft":
    test "5-branch decision table":
      check testClassifyGraft()

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
    test "appendOp sets subtree_sum_leaves = 1 when it was 0":
      check testAppendOpLeavesZero()

  suite "Evict":
    test "evict unlocked leaf after graft":
      check testEvictUnlockedLeaf()
    test "cannot evict locked leaf":
      check testEvictLockedLeaf()
    test "evict on empty tree does not crash":
      check testEvictEmptyTree()
    test "evict propagates subtree_sum_leaves up":
      check testEvictPropagatesLeaves()
    test "evict root-with-no-parent replaces root":
      check testEvictRootWithNoParent()

  suite "findEvictionCandidate":
    test "finds unlocked leaf when one exists":
      check testFindCandidateUnlocked()
    test "returns nil when all leaves locked":
      check testFindCandidateAllLocked()
    test "wavlNext skip-locked path":
      check testEvictionCandidateSkipLocked()
    test "walkUpUpdate re-keys eviction tree":
      check testWalkUpUpdateReKey()

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
    test "4-token WAVL LPM collision resolved by full-page compare":
      check testRegressionLpmFourTokenCollision()
    test "compressPath recursive + eviction tree re-key (P3)":
      check testRegressionCompressPathRecursiveAndReKey()
    test "compressPath preserves subtree_oldest_decode (no-op assignment)":
      check testRegressionCompressPathPreservesTimestamp()
when isMainModule:
  runTests()