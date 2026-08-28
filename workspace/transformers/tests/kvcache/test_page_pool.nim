## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import workspace/libtorch
import workspace/libtorch_testutils
import workspace/transformers/src/stateful/page_pool
import workspace/transformers/src/stateful/kvcache

# Page + KVCache lifecycle tests, public API only. PageObj fields
# (index, pool) are private, so the tests interact via borrow()
# and the trie's graftPages/evict lifecycle. The pool tests need
# libtorch linked at runtime.

proc testEmptyKvCacheLpm(): bool =
  var cache = KVCache[uint32, Page].new()
  let tokens = @[1'u32, 2, 3]
  let matched = cache.lpm(tokens)
  doAssert matched.totalTokenMatched == 0
  doAssert matched.pages.len == 0
  cache.graftPages(tokens, newSeq[Page]())
  result = true

proc testGraftPagesEmptyPageSeq(): bool =
  var cache = KVCache[uint32, Page].new()
  let tokens = @[1'u32, 2, 3]
  # Pages carry a nil pool, so =destroy is a no-op
  cache.graftPages(tokens, newSeq[Page]())
  result = true

proc testEvictReturnsFreedPages(): bool =
  # Graft two single-token sequences, each with one "page"
  # Note: Page here is just a token type for the trie.
  type DummyPage = int  # int payload, no GPU pool for trie-only tests
  var dcache = KVCache[uint32, DummyPage].new()
  dcache.graftPages(@[1'u32], @[42])
  dcache.graftPages(@[2'u32], @[43])
  let freed = dcache.evict()
  doAssert freed > 0
  result = true

proc testLayerViewSlabSlice(): bool =
  let pool = PagePool.init(4, 3, 2, 128, kFloat16, kCPU)
  let kv = pool.layerView(1)
  let k = kv.kView
  let v = kv.vView
  doAssert k.size(0) == 4
  doAssert k.size(1) == TokensPerPage
  doAssert k.size(2) == 2
  doAssert k.size(3) == 128
  doAssert v.size(0) == 4
  let off = cast[int](k.data_ptr()) - cast[int](pool.layerView(0).kView.data_ptr())
  doAssert off == 1 * TokensPerPage * 2 * 128 * k.element_size()
  doAssert k.strides()[0] == 3 * TokensPerPage * 2 * 128
  result = true

when isMainModule:
  runCppTest("empty KVCache[uint32, Page] LPM", testEmptyKvCacheLpm)
  runCppTest("graftPages with an empty page sequence", testGraftPagesEmptyPageSeq)
  runCppTest("evict returns page count > 0 after graftPages", testEvictReturnsFreedPages)
  runCppTest("layerView selects the per-layer slab slice", testLayerViewSlabSlice)
