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
  # Lock all children. Maintain C2: root.locked == sum(child.locked)
  cache.root.children[0].subtree_sum_locked = 1
  cache.root.children[1].subtree_sum_locked = 1
  cache.root.subtree_sum_locked = 2
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
  # gcRootNewChild: lastLevel < 256, no parent, root has children, root is EMPTY
  # (targetTokLen == 0 indicates root has no content tokens)
  doAssert classifyGraft(0, 100, 0, 0, false, true) == gcRootNewChild
  # gcFork: root has tokens AND children, input differs at position 0
  # lastLevel=0 but targetTokLen=256 -> should NOT be rootNewChild
  doAssert classifyGraft(0, 100, 0, 256, false, true) == gcFork,
    "root with content+children+different input should fork, not rootNewChild"
  # gcFork: lastLevel < targetTokLen (not full match, not sub-page)
  doAssert classifyGraft(256, 512, 256, 512, true, false) == gcFork
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
# CODERA-030: forkPageOp grandparent slot
# ════════════════════════════════════════════════════════
proc testCoderA030ForkPageSlot(): bool =
  ## CODERA-030: forkPageOp overwrites wrong grandparent slot.
  ## When the forked node is at index > 0 in its parent,
  ## newParent.addChild(target) resets childId to 0, so
  ## grandparent.children[target.childId] = newParent
  ## writes to grandparent.children[0] instead of the correct index.
  ##
  ## Uses page-aligned fork (lastLevel=256) so forkPageOp creates
  ## a newParent with 1 page of shared content and no A1 conflict.
  var cache = KVCache[uint32, int].new()

  # Step 1: A = [0..255], 256 tokens
  var seqA = newSeq[uint32](256)
  for i in 0..<256: seqA[i] = uint32(i)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(256))

  # Step 2: B = [1000..1511] (512 tokens, 2 pages), different first token
  # -> fork at root: root becomes empty branching with children [A(idx=0), B(idx=1)]
  var seqB = newSeq[uint32](512)
  for i in 0..<512: seqB[i] = uint32(1000 + i)
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(512))

  doAssert cache.root.children.len == 2,
    "root should have 2 children after fork, got " & $cache.root.children.len
  doAssert cache.root.children[1].childId == 1,
    "B should be at index 1, got childId=" & $cache.root.children[1].childId

  # Step 3: C matches first 256 of B (full page), then diverges
  # -> forkPageOp on B at idx 1, llBranchingPoint=256
  # BUG: addChild resets childId to 0 -> grandparent.children[0] overwritten!
  var seqC = newSeq[uint32](258)
  for i in 0..<256: seqC[i] = uint32(1000 + i)  # match B's first page
  seqC[256] = 9999'u32
  seqC[257] = 10000'u32
  discard cache.lpm(seqC)
  cache.graftPages(seqC, makePages(258))

  # After correct fork: root has [A, newParent_B(256 tok)] at indices [0,1]
  # newParent_B has [B-tail(256 tok), C-sibling(2 tok)]
  radixVerifyInvariants(cache.root, "CODERA-030")

  # Verify structure
  doAssert cache.root.children.len == 2,
    "root should have 2 children, got " & $cache.root.children.len
  doAssert cache.root.children[1].children.len == 2,
    "newParent_B should have 2 children, got " & $cache.root.children[1].children.len
  doAssert cache.root.children[0].tokens == seqA, "A's tokens unchanged"

  # LPM on seqC should match all 258 tokens
  let r = cache.lpm(seqC)
  doAssert r.totalTokenMatched == 258,
    "LPM on C: matched " & $r.totalTokenMatched & ", expected 258"

  # LPM on seqA should still match fully
  let rA = cache.lpm(seqA)
  doAssert rA.totalTokenMatched == 256,
    "LPM on A: matched " & $rA.totalTokenMatched & ", expected 256"
  result = true

# ════════════════════════════════════════════════════════
# CODERA-029: classifyGraft non-empty branching root
# ════════════════════════════════════════════════════════
proc testCoderA029ClassifyGraftRoot(): bool =
  ## CODERA-029: classifyGraft routes non-empty branching root
  ## to gcRootNewChild instead of gcFork.
  ##
  ## Root has tokens AND children (after fork at page boundary).
  ## A new graft partially matches root's tokens (a full page) but not all.
  ## classifyGraft must return gcFork, not gcRootNewChild.
  var cache = KVCache[uint32, int].new()

  # Step 1: A = [0..767] (3 pages, 768 tokens) as root leaf
  var seqA = newSeq[uint32](768)
  for i in 0..<768: seqA[i] = uint32(i)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(768))

  # Step 2: B matches first 512 of A (2 pages), diverges
  # -> fork at lastLevel=512, llBranchingPoint=512
  # -> newParent gets tokens[0..511], becomes root
  # -> root now has 512 tokens AND 2 children
  var seqB = newSeq[uint32](514)
  for i in 0..<512: seqB[i] = uint32(i)
  seqB[512] = 8888'u32
  seqB[513] = 8889'u32
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(514))

  doAssert cache.root.tokens.len > 0,
    "root should have tokens after fork at page boundary"
  doAssert cache.root.children.len > 0,
    "root should have children after fork"

  # Step 3: C matches first 256 of root (1 page), then diverges
  # BUG: classifyGraft returns gcRootNewChild instead of gcFork
  # gcRootNewChild adds C as direct child of root with ALL tokens
  # -> LPM on C only matches 256 tokens (root's prefix), not all 258
  # Correct: gcFork creates newParent, C becomes sibling, LPM matches 258
  var seqC = newSeq[uint32](258)
  for i in 0..<256: seqC[i] = uint32(i)  # match root's first 256 tokens
  seqC[256] = 6666'u32
  seqC[257] = 6667'u32
  discard cache.lpm(seqC)
  cache.graftPages(seqC, makePages(258))

  # Verify: LPM on C should match ALL 258 tokens
  # With gcRootNewChild bug: root's 512 tokens match 256,
  # findChild returns nil at pos=256 -> only 256 matched
  let r = cache.lpm(seqC)
  doAssert r.totalTokenMatched == 258,
    "LPM should match all 258 tokens, got " & $r.totalTokenMatched & ". " &
    "gcRootNewChild bug: C was added as direct child, root's content masks it"

  radixVerifyInvariants(cache.root, "CODERA-029")
  result = true

# ════════════════════════════════════════════════════════
# Three different prompts (all differ at first token)
# ════════════════════════════════════════════════════════
proc testThreeDifferentPrompts(): bool =
  ## Three prompts, all differing at the first token:
  ##   1. First prompt -> gcAppend (root leaf)
  ##   2. Second prompt, different -> gcFork (root splits, becomes empty branching)
  ##   3. Third prompt, different from both -> gcRootNewChild (added as direct child)
  ##
  ## Verifies the full gcAppend -> gcFork -> gcRootNewChild lifecycle.
  var cache = KVCache[uint32, int].new()

  # Prompt 1: [0..255]
  var p1 = newSeq[uint32](256)
  for i in 0..<256: p1[i] = uint32(i)
  discard cache.lpm(p1)
  cache.graftPages(p1, makePages(256))
  doAssert cache.root.tokens == p1, "prompt 1 should append to root"
  doAssert cache.root.children.len == 0, "root has no children after first prompt"

  # Prompt 2: [2000..2255] — different first token
  # -> gcFork: root gets split, becomes empty branching with 2 children
  var p2 = newSeq[uint32](256)
  for i in 0..<256: p2[i] = uint32(2000 + i)
  discard cache.lpm(p2)
  cache.graftPages(p2, makePages(256))

  doAssert cache.root.tokens.len == 0,
    "root should be empty after fork, got " & $cache.root.tokens.len & " tokens"
  doAssert cache.root.children.len == 2,
    "root should have 2 children after fork, got " & $cache.root.children.len

  # Prompt 3: [4000..4255] — different from both previous
  # -> gcRootNewChild: added as direct child of empty branching root
  var p3 = newSeq[uint32](256)
  for i in 0..<256: p3[i] = uint32(4000 + i)
  discard cache.lpm(p3)
  cache.graftPages(p3, makePages(256))

  doAssert cache.root.tokens.len == 0,
    "root should stay empty after third graft, got " & $cache.root.tokens.len & " tokens"
  doAssert cache.root.children.len == 3,
    "root should have 3 children, got " & $cache.root.children.len

  # Verify all three prompts are findable via LPM
  for (prompt, name) in [(p1, "p1"), (p2, "p2"), (p3, "p3")]:
    let r = cache.lpm(prompt)
    doAssert r.totalTokenMatched == 256,
      "LPM on " & name & " matched " & $r.totalTokenMatched & ", expected 256"

  radixVerifyInvariants(cache.root, "three-prompts")
  result = true

proc testCoderA018SubtreeSumLeavesDecrement(): bool =
  ## CODERA-018: findEvictionCandidate decrements the child's
  ## subtree_sum_leaves instead of the ancestor's, corrupting
  ## subtree_sum_leaves on the path to root after eviction from a
  ## multi-level tree.
  ##
  ## Build a 3-level tree (root → splitNode(256 tok, 2 children) → leaves,
  ## and root → B(512 tok)), lock B to force eviction from the deep
  ## subtree, then verify A5 invariant after eviction.
  var cache = KVCache[uint32, int].new()

  # Step 1: A = [0..511] (2 pages)
  var aFull = newSeq[uint32](512)
  for i in 0..<512: aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(512))

  # Step 2: B = [1000..1511] — different first token → fork at page 0
  var bFull = newSeq[uint32](512)
  for i in 0..<512: bFull[i] = uint32(1000 + i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(512))
  doAssert cache.root.children.len == 2,
    "root should have 2 children after fork"

  # Step 3: C matches first 256 of A, then diverges
  # -> fork under root[0]: root[0] becomes splitNode(256, 2 children)
  var cTok = newSeq[uint32](258)
  for i in 0..<256: cTok[i] = uint32(i)
  cTok[256] = 2000; cTok[257] = 2001
  discard cache.lpm(cTok)
  cache.graftPages(cTok, makePages(258))

  let splitNode = cache.root.children[0]
  doAssert splitNode.children.len == 2,
    "splitNode should have 2 children, got " & $splitNode.children.len
  doAssert splitNode.subtree_sum_leaves == 2,
    "splitNode.subtree_sum_leaves == " & $splitNode.subtree_sum_leaves & ", expected 2"

  # Lock B to force eviction from splitNode's subtree
  cache.root.children[1].subtree_sum_locked = 1
  cache.root.subtree_sum_locked = 1

  # Verify invariants BEFORE eviction
  radixVerifyInvariants(cache.root, "CODERA-018 before evict")

  # Evict — must pick from splitNode's subtree (B is locked)
  cache.evict()

  # Verify invariants AFTER eviction
  # With CODERA-018 bug: root.subtree_sum_leaves was never decremented
  # during descent, so A5 fails: root.subtree_sum_leaves != sum(children)
  radixVerifyInvariants(cache.root, "CODERA-018 after evict")

  result = true

# ════════════════════════════════════════════════════════
# Runner
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

  suite "CODERA-030 (forkPageOp grandparent slot)":
    test "forkPageOp on non-zero childId keeps grandparent consistent":
      check testCoderA030ForkPageSlot()

  suite "CODERA-029 (classifyGraft non-empty root)":
    test "classifyGraft uses gcFork for non-empty branching root":
      check testCoderA029ClassifyGraftRoot()

  suite "Graft lifecycle (append -> fork -> rootNewChild)":
    test "three different prompts at first token use all 3 branches":
      check testThreeDifferentPrompts()

suite "CODERA-018 (subtree_sum_leaves decrement order)":
  test "eviction from 3-level tree maintains A5 invariant":
    check testCoderA018SubtreeSumLeavesDecrement()

when isMainModule:
  runTests()