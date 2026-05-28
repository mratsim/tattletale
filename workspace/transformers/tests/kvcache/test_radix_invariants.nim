## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/unittest,
  std/math,
  std/importutils,
  workspace/transformers/src/stateful/kvcache {.all.},
  workspace/transformers/src/stateful/stateful_testutils

privateAccess(PagedRadixNode)
privateAccess(KVCache)


# Test that invariants hold after basic operations
proc testInvariantsEmptyCache(): bool =
  var cache = KVCache[uint32, int].new()
  radixVerifyInvariants(cache.root, "empty cache")
  result = true

proc testInvariantsAfterSingleGraft(): bool =
  var cache = KVCache[uint32, int].new()
  var tok = newSeq[uint32](256)
  for i in 0..<256: tok[i] = uint32(i)
  discard cache.lpm(tok)
  cache.graftPages(tok, makePages(256))
  radixVerifyInvariants(cache.root, "after single graft")
  result = true

proc testInvariantsAfterFork(): bool =
  var cache = KVCache[uint32, int].new()

  # Insert A = [0..511]
  var aFull = newSeq[uint32](512)
  for i in 0..<512: aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(512))

  radixVerifyInvariants(cache.root, "after A")

  # Insert B = [1000..1511] — different, forces fork
  var bFull = newSeq[uint32](512)
  for i in 0..<512: bFull[i] = 1000'u32 + uint32(i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(512))

  radixVerifyInvariants(cache.root, "after B (fork)")

  # Insert C = [0..255, 2000, 2001] — matches first page of A
  var cTok = newSeq[uint32](258)
  for i in 0..<256: cTok[i] = uint32(i)
  cTok[256] = 2000; cTok[257] = 2001
  discard cache.lpm(cTok)
  cache.graftPages(cTok, makePages(258))

  radixVerifyInvariants(cache.root, "after C (sub-page fork)")
  result = true

proc testInvariantsAfterEviction(): bool =
  var cache = KVCache[uint32, int].new()

  # Insert two branches
  var aFull = newSeq[uint32](256)
  for i in 0..<256: aFull[i] = uint32(i)
  discard cache.lpm(aFull)
  cache.graftPages(aFull, makePages(256))

  var bFull = newSeq[uint32](256)
  for i in 0..<256: bFull[i] = 1000'u32 + uint32(i)
  discard cache.lpm(bFull)
  cache.graftPages(bFull, makePages(256))

  radixVerifyInvariants(cache.root, "before eviction")

  # Evict one leaf
  cache.evict()

  radixVerifyInvariants(cache.root, "after eviction")
  result = true

proc runTests*() =
  suite "Radix Trie Invariants":
    test "empty cache": check testInvariantsEmptyCache()
    test "after single graft": check testInvariantsAfterSingleGraft()
    test "after fork": check testInvariantsAfterFork()
    test "after eviction": check testInvariantsAfterEviction()

when isMainModule:
  runTests()
