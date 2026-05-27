## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tests for fork stability: interleaved decode + fork scenarios
##
## Each test is a standalone proc so ARC/ORC destructors fire immediately.

import
  std/unittest,
  std/math,
  std/importutils,
  ../../src/stateful/kvcache {.all.},
  ../../src/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)

# ── Helpers ──────────────────────────────────────────────
proc makeTokens(grp, uid, n: int): seq[uint32] =
  const SYS = 4096
  result = newSeq[uint32](n)
  for i in 0..<min(n, SYS):
    result[i] = uint32(i)
  if n <= SYS: return
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

# ════════════════════════════════════════════════════════
# Fork stability tests
# ════════════════════════════════════════════════════════
proc testForkBelowA(): bool =
  ## A matches 1024, B forks at 768 (below A), A grafts
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
  result = true

proc testForkAboveA(): bool =
  ## A matches 768, B forks at 1280 (above A), A grafts
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
  result = true

proc testContinuationStress(): bool =
  ## Many sequential extends keep structure valid
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
  result = true

# ════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════
proc runTests*() =
  suite "Fork stability (interleaved decode + fork)":
    test "A matches 1024, B forks at 768 (below A), A grafts":
      check testForkBelowA()
    test "A matches 768, B forks at 1280 (above A), A grafts":
      check testForkAboveA()
    test "Continuation stress: many sequential extends":
      check testContinuationStress()

when isMainModule:
  runTests()
