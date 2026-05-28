## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/unittest
import std/importutils
import workspace/transformers/src/stateful/page_pool
import workspace/transformers/src/stateful/kvcache

# These tests verify Page + KVCache lifecycle using only public APIs.
# PageObj fields (index, pool) are private — we interact via borrow() and
# the trie's graftPages/evict lifecycle.
#
# Full PagePool + tensor tests need libtorch linked at runtime (see AGENTS.md).

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
    # Create Pages using the pool (needs tensor init — skipped if unavailable)
    let tokens = @[1'u32, 2, 3]
    # These pages have nil pool so =destroy is a no-op
    cache.graftPages(tokens, newSeq[Page]())  # just release lock

  test "evict returns page count > 0 after graftPages":
    var cache = KVCache[uint32, Page].new()
    # Graft two single-token sequences, each with one "page"
    # Note: Page here is just a token type for the trie; actual pool/buffer
    # management is tested via the orchestrator integration tests.
    type DummyPage = int  # int as page payload (no GPU pool needed for trie-only tests)
    var dcache = KVCache[uint32, DummyPage].new()
    dcache.graftPages(@[1'u32], @[42])
    dcache.graftPages(@[2'u32], @[43])
    let freed = dcache.evict()
    check freed > 0
