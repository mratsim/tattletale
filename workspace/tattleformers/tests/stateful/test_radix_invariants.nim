# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Radix trie invariant suite (CPU, model-free).
##
## Test that radixVerifyInvariants accepts every routable child structure
## and rejects the unroutable one.
##
## - siblings may share a partial first-page prefix, their keys still
##   diverge inside the page
## - two children with identical first-page keys cannot be told apart
##   through the child index
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" \
##     workspace/tattleformers/tests/stateful/test_radix_invariants.nim
##

import
  std/importutils,
  workspace/tattleformers/src/stateful/kvcache {.all.},
  workspace/tattleformers/tests/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)

# Test that invariants hold after basic operations
proc testInvariantsEmptyCache(): bool =
  result = true
  var cache = KVCache[uint32, int].new()
  radixVerifyInvariants(cache.root, "empty cache")

proc testInvariantsAfterSingleGraft(): bool =
  result = true
  var cache = KVCache[uint32, int].new()
  var tok = newSeq[uint32](256)
  for i in 0..<256:
    tok[i] = uint32(i)
  discard cache.lpm(tok)
  cache.graftPages(tok, makePages(256))
  radixVerifyInvariants(cache.root, "after single graft")

proc testInvariantsAfterFork(): bool =
  result = true
  var cache = KVCache[uint32, int].new()

  # Insert A = [0..511]
  var aFull = newSeq[uint32](512)
  for i in 0..<512:
    aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(512))

  radixVerifyInvariants(cache.root, "after A")

  # Insert B = [1000..1511], different first page, forces a fork
  var bFull = newSeq[uint32](512)
  for i in 0..<512:
    bFull[i] = 1000'u32 + uint32(i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(512))

  radixVerifyInvariants(cache.root, "after B (fork)")

  # Insert C = [0..255, 2000, 2001], matching A's first page
  var cTok = newSeq[uint32](258)
  for i in 0..<256:
    cTok[i] = uint32(i)
  cTok[256] = 2000
  cTok[257] = 2001
  discard cache.lpm(cTok)
  cache.graftPages(cTok, makePages(258))

  radixVerifyInvariants(cache.root, "after C (page-aligned fork)")
  result = true

proc testInvariantsAfterEviction(): bool =
  result = true
  var cache = KVCache[uint32, int].new()

  # Insert two branches
  var aFull = newSeq[uint32](256)
  for i in 0..<256:
    aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(256))

  var bFull = newSeq[uint32](256)
  for i in 0..<256:
    bFull[i] = 1000'u32 + uint32(i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(256))

  radixVerifyInvariants(cache.root, "before eviction")

  # Evict one leaf
  cache.evict()

  radixVerifyInvariants(cache.root, "after eviction")
  result = true

proc testValidatorAcceptsPartialFirstPageSiblings(): bool =
  ## Two children whose keys share a partial first-page prefix stay routable:
  ## the child index orders by the signed divergence point, which is nonzero
  ## for any pair that diverges inside the page. grafting a short sequence
  ## against a shorter one forks the root into exactly this shape.
  result = true
  var cache = KVCache[uint32, int].new()
  let child0 = @[0'u32, 1, 2, 3, 4]
  let child1 = @[0'u32, 1, 3, 4, 0]  # shares tokens 0, 1 with child0
  discard cache.lpm(child0)
  cache.graftPages(child0, @[1])
  discard cache.lpm(child1)
  cache.graftPages(child1, @[2])

  doAssert cache.root.children.len == 2
  radixVerifyInvariants(cache.root, "partial first-page siblings")

proc testValidatorRejectsIdenticalFirstPageKeys(): bool =
  ## Two children with identical first-page keys are unroutable: the child index
  ## resolves a query to one of them, so the other subtree is unreachable.
  ##
  ## - the child index itself refuses the insert (duplicate key)
  ## - the tree below is built directly, exercising the validator's own defense
  result = true
  let dup = makeTokens(256)
  let c0 = PagedRadixNode[uint32, int](
    tokens: dup, pages: @[1], depth_in_pages: 0, subtree_sum_leaves: 1)
  let c1 = PagedRadixNode[uint32, int](
    tokens: dup, pages: @[2], depth_in_pages: 0, subtree_sum_leaves: 1)
  let broken = PagedRadixNode[uint32, int](
    children: @[c0, c1], subtree_sum_leaves: 2)
  doAssertRaises KVCacheDefect:
    radixVerifyInvariants(broken, "identical first-page keys")

proc testValidatorRejectsIdenticalShortKeys(): bool =
  ## Same as the full-page case at sub-page width: two children with equal
  ## short keys cannot be told apart.
  result = true
  let dup = @[7'u32, 8, 9]
  let c0 = PagedRadixNode[uint32, int](
    tokens: dup, pages: @[1], depth_in_pages: 0, subtree_sum_leaves: 1)
  let c1 = PagedRadixNode[uint32, int](
    tokens: dup, pages: @[2], depth_in_pages: 0, subtree_sum_leaves: 1)
  let broken = PagedRadixNode[uint32, int](
    children: @[c0, c1], subtree_sum_leaves: 2)
  doAssertRaises KVCacheDefect:
    radixVerifyInvariants(broken, "identical short keys")

proc runAll() =
  doAssert testInvariantsEmptyCache()
  doAssert testInvariantsAfterSingleGraft()
  doAssert testInvariantsAfterFork()
  doAssert testInvariantsAfterEviction()
  doAssert testValidatorAcceptsPartialFirstPageSiblings()
  doAssert testValidatorRejectsIdenticalFirstPageKeys()
  doAssert testValidatorRejectsIdenticalShortKeys()

when isMainModule:
  runAll()
