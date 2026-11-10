# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Fork stability: interleaved decode + fork scenarios (CPU, model-free).
##
## Test that forks and re-grafts keep every subtree routable through LPM
## and the structural invariants intact, including a fork whose divergence
## falls inside the target's first page.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" \
##     workspace/tattleformers/tests/stateful/test_fork_stability.nim
##

import
  std/importutils,
  workspace/tattleformers/src/stateful/kvcache {.all.},
  workspace/tattleformers/tests/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)

# ── Helpers ──────────────────────────────────────────────
proc makeTokens(grp, uid, n: int): seq[uint32] =
  const SYS = 4096
  result = newSeq[uint32](n)
  for i in 0..<min(n, SYS):
    result[i] = uint32(i)
  if n <= SYS:
    return
  for i in 0..<min(n-SYS, 512):
    result[SYS+i] = (grp*1000 + i).uint32
  let uOff = SYS + 512
  for i in uOff..<n:
    let x = uid*10000 + (i-uOff) mod 997
    result[i] = x.uint32


proc checkInvariants(n: PagedRadixNode[uint32, int]):
    tuple[leaves: int32, locks: int32] =
  if n.children.len > 0:
    var childLeaves, childLocks: int32
    for c in n.children:
      let (cl, ck) = checkInvariants(c)
      childLeaves += cl
      childLocks += ck
    doAssert n.subtree_sum_leaves == childLeaves,
      "A5 violated at depth " & $n.depth_in_pages
    doAssert n.subtree_sum_locked == childLocks,
      "C2 violated at depth " & $n.depth_in_pages
  result = (n.subtree_sum_leaves, n.subtree_sum_locked)

# ═══════════════════════════════════════════════════════════════════════
# Fork stability tests
# ═══════════════════════════════════════════════════════════════════════
proc testForkBelowA(): bool =
  ## A matches 1024, B forks at 768 (below A), A grafts
  result = true
  var cache = KVCache[uint32, int].new()
  let full = makeTokens(0, 0, 1280)
  discard cache.lpm(full)
  cache.graftPages(full, makePages(1280))

  let aTokens = makeTokens(0, 0, 1024)
  let rA = cache.lpm(aTokens)
  doAssert rA.pages.len == 4

  var bTok = makeTokens(0, 0, 768)
  bTok.add([9999'u32, 10000'u32, 10001'u32])
  discard cache.lpm(bTok)
  cache.graftPages(bTok, makePages(bTok.len))

  cache.graftPages(aTokens, makePages(aTokens.len))

  let (leaves, _) = checkInvariants(cache.root)
  doAssert leaves > 0

proc testForkAboveA(): bool =
  ## A matches 768, B forks at 1280 (above A), A grafts
  result = true
  var cache = KVCache[uint32, int].new()
  let full = makeTokens(0, 0, 1280)
  discard cache.lpm(full)
  cache.graftPages(full, makePages(1280))

  let aTokens = makeTokens(0, 0, 768)
  let rA = cache.lpm(aTokens)
  doAssert rA.pages.len == 3

  var bTok = makeTokens(0, 0, 1280)
  bTok.add([8888'u32, 8889'u32])
  discard cache.lpm(bTok)
  cache.graftPages(bTok, makePages(bTok.len))

  cache.graftPages(aTokens, makePages(aTokens.len))

  let (leaves, _) = checkInvariants(cache.root)
  doAssert leaves > 0

proc testContinuationStress(): bool =
  ## Many sequential extends keep structure valid
  result = true
  var cache = KVCache[uint32, int].new()
  var oneMore = makeTokens(0, 0, 8192)
  oneMore.add(9999'u32)

  discard cache.lpm(oneMore)
  cache.graftPages(oneMore, makePages(oneMore.len))

  for i in 0..<10:
    discard cache.lpm(oneMore)
    cache.graftPages(oneMore, makePages(oneMore.len))

  doAssert cache.root.subtree_sum_leaves > 0
  let (leaves, locks) = checkInvariants(cache.root)
  doAssert leaves > 0
  doAssert locks == 0

proc testSubPageForkStaysRoutable(): bool =
  ## A fork whose divergence falls inside the target's first page.
  ##
  ## - the fork point rounds below the page boundary, the trie cannot
  ##   split pages
  ## - the new sequence enters as a copy-on-write sibling of the target
  ##   under the same parent
  ## - LPM over the target, the sibling and an unrelated branch each match
  ##   their full length
  ## - every structural invariant passes
  ##
  ## Structure:
  ##
  ## - A of 512 tokens and B (different first token) fork the root
  ## - C shares the first 5 tokens of A, diverging inside A's first page
  ## - C therefore grafts against A, the non-root target
  result = true
  var cache = KVCache[uint32, int].new()

  var seqA = makeTokens(512)
  discard cache.lpm(seqA)
  cache.graftPages(seqA, makePages(512))

  var seqB = newSeq[uint32](512)
  for i in 0..<512:
    seqB[i] = uint32(5000 + i)
  discard cache.lpm(seqB)
  cache.graftPages(seqB, makePages(512))

  var seqC = newSeq[uint32](512)
  for i in 0..<5:
    seqC[i] = uint32(i)
  for i in 5..<512:
    seqC[i] = uint32(7000 + i)
  discard cache.lpm(seqC)
  cache.graftPages(seqC, makePages(512))

  # C enters as a sibling of A under the same parent, no new level.
  # Expected root children: A, B and C.
  doAssert cache.root.children.len == 3,
    "got " & $cache.root.children.len & " root children, expected 3"
  let (leaves, locks) = checkInvariants(cache.root)
  doAssert leaves == 3
  doAssert locks == 0
  radixVerifyInvariants(cache.root, "sub-page fork")

  # LPM locks the walked path, the counts below run over the locked tree
  doAssert cache.lpm(seqA).totalTokenMatched == 512,
    "A's subtree must stay routable after the sub-page fork"
  doAssert cache.lpm(seqB).totalTokenMatched == 512,
    "B's subtree must stay routable after the sub-page fork"
  doAssert cache.lpm(seqC).totalTokenMatched == 512,
    "C must graft fully despite diverging inside A's first page"

proc runAll() =
  doAssert testForkBelowA()
  doAssert testForkAboveA()
  doAssert testContinuationStress()
  doAssert testSubPageForkStaysRoutable()

when isMainModule:
  runAll()
