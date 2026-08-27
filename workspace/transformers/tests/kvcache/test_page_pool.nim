## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/unittest
import std/importutils
import workspace/libtorch
import workspace/transformers/src/stateful/page_pool
import workspace/transformers/src/stateful/kvcache

# Page + KVCache lifecycle tests, public API only. PageObj fields
# (index, pool) are private, so the tests interact via borrow()
# and the trie's graftPages/evict lifecycle. The pool tests need
# libtorch linked at runtime.

suite "Page + KVCache integration (P = Page, public API only)":
  test "empty KVCache[uint32, Page] LPM":
    var cache = KVCache[uint32, Page].new()
    let tokens = @[1'u32, 2, 3]
    let matched = cache.lpm(tokens)
    check matched.totalTokenMatched == 0
    check matched.pages.len == 0
    cache.graftPages(tokens, newSeq[Page]())

  test "graftPages with Page ref, then LPM retrieves":
    var cache = KVCache[uint32, Page].new()
    let tokens = @[1'u32, 2, 3]
    # Pages carry a nil pool, so =destroy is a no-op
    cache.graftPages(tokens, newSeq[Page]())

  test "evict returns page count > 0 after graftPages":
    var cache = KVCache[uint32, Page].new()
    # Graft two single-token sequences, each with one "page"
    # Note: Page here is just a token type for the trie.
    type DummyPage = int  # int payload, no GPU pool for trie-only tests
    var dcache = KVCache[uint32, DummyPage].new()
    dcache.graftPages(@[1'u32], @[42])
    dcache.graftPages(@[2'u32], @[43])
    let freed = dcache.evict()
    check freed > 0

suite "PagePool layer views":
  test "layerView selects the per-layer slab slice":
    let pool = PagePool.init(4, 3, 2, 128, kFloat16, kCPU)
    let kv = pool.layerView(1)
    let k = kv.kView
    let v = kv.vView
    check k.size(0) == 4
    check k.size(1) == TokensPerPage
    check k.size(2) == 2
    check k.size(3) == 128
    check v.size(0) == 4
    let off = cast[int](k.data_ptr()) - cast[int](pool.layerView(0).kView.data_ptr())
    check off == 1 * TokensPerPage * 2 * 128 * k.element_size()
    check k.strides()[0] == 3 * TokensPerPage * 2 * 128
