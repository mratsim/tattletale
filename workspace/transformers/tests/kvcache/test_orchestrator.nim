# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Orchestrator integration tests — regression tests for BUG-B-001 and BUG-B-002.
##
## These tests use a full PagePool (CPU-backed) to exercise the orchestrator's
## page lifecycle management with real Page objects.
##
## Bugs reproduced:
##   BUG-B-001: kv_position never updated after prefill → OOB crash on decode
##              at page boundary.
##   BUG-B-002: COW partial-match page inserted before fully-matched pages
##              → page ordering corruption in ctx.pages.
##
## Test design:
##   Each bug gets a standalone proc returning bool (same pattern as
##   test_kvcache.nim).  Tests are registered in suites under runTests().
##
## Compilation:
##   Needs libtorch linked (same as test_page_pool.nim).  Use:
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests --nimcache:nimcache/tests \
##       workspace/transformers/tests/kvcache/test_orchestrator.nim

import
  std/unittest,
  std/importutils,
  workspace/libtorch as F,
  ../../src/stateful/kvcache,        # TokensPerPage, ceilDiv
  ../../src/stateful/page_pool,       # Page, pageIndex
  ../../src/stateful/inference_context,
  ../../src/stateful/orchestrator

# ═════════════════════════════════════════════════════════════════════════════
# Test configuration
# ═════════════════════════════════════════════════════════════════════════════

const
  TestLayers = 1
  TestKvHeads = 1
  TestHeadDim = 4
  TestMaxSeq = 4096
  TestNumPages = 32

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

proc makeOrchestrator(): Orchestrator =
  ## Create a minimal orchestrator backed by CPU tensors.
  result = Orchestrator.init(
    num_layers = TestLayers,
    batch_size = 1,
    kv_heads = TestKvHeads,
    max_seq = TestMaxSeq,
    head_dim = TestHeadDim,
    num_pages = TestNumPages,
    dtype = kBFloat16,
    device = kCPU
  )

proc makeTokens(n: int): seq[uint32] =
  ## Sequential tokens `[0, 1, 2, ..., n-1]` for reproducible LPM matching.
  result = newSeq[uint32](n)
  for i in 0 ..< n:
    result[i] = uint32(i)

# ═════════════════════════════════════════════════════════════════════════════
# BUG-B-001: kv_position never updated after prefill → OOB on decode at
# page boundary.
#
# Root cause chain:
#   1. startSequence sets kv_position = matched.totalTokenMatched (0 for new prompt)
#   2. Prefill forward pass writes KV at positions 0..seq_len-1 but NEVER
#      updates kv_position.
#   3. After 256-token prefill, kv_position is still 0.
#   4. decodeStep checks: kv_position > 0 and (kv_position mod 256) == 0
#      → 0 > 0 is false → no page allocated.
#   5. attn.forward computes pageIdx = 256 / 256 = 1 → ctx.pages[1] → OOB crash.
# ═════════════════════════════════════════════════════════════════════════════

proc testBugB001KvPositionUpdate(): bool =
  ## Prefill exactly TokensPerPage (256) tokens, then decodeStep.
  ## Without the fix (ctx.kv_position = input_ids.len), this would crash
  ## because the first decodeStep doesn't allocate a new page.
  var orc = makeOrchestrator()
  let tokens = makeTokens(TokensPerPage)  # exactly 256 tokens = 1 page

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContext()

  # After prefill of 256 tokens, kv_position MUST be 256
  doAssert ctx.kv_position == TokensPerPage,
    "[BUG-B-001] kv_position should be " & $TokensPerPage &
    " after " & $TokensPerPage & "-token prefill, got " & $ctx.kv_position

  # ctx.pages should have exactly 1 page (256 tokens = 1 page)
  doAssert ctx.pages.len == 1,
    "[BUG-B-001] Expected 1 page for " & $TokensPerPage &
    " tokens, got " & $ctx.pages.len

  # decodeStep at position 256 should succeed (allocate new page at boundary)
  orc.decodeStep(position = TokensPerPage, token_id = 42'u32, device = kCPU)

  # After decode, kv_position should be 257
  doAssert ctx.kv_position == TokensPerPage + 1,
    "[BUG-B-001] kv_position should be " & $(TokensPerPage + 1) &
    " after 1 decode step, got " & $ctx.kv_position

  # Pages should now be 2 (crossed page boundary → new page allocated)
  doAssert ctx.pages.len == 2,
    "[BUG-B-001] Expected 2 pages after crossing page boundary, got " &
    $ctx.pages.len

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# BUG-B-002: COW partial-match page inserted before fully-matched pages,
# corrupting ctx.pages ordering.
#
# Scenario: LPM matches 300 tokens (256 full + 44 partial).
#   - matched.pages[0] = trie page for positions 0-255
#   - matched.pages[^1] = trie page for positions 256-511 (only 44 valid)
#
# Buggy ordering:
#   ctx.pages[0] = COW copy of matched.pages[^1] (positions 256-299) ← WRONG
#   ctx.pages[1] = matched.pages[0] (positions 0-255) ← WRONG
#
# Correct ordering (after fix):
#   ctx.pages[0] = matched.pages[0] (positions 0-255) ← correct
#   ctx.pages[1] = COW copy of matched.pages[^1] (positions 256-299) ← correct
#
# Verification strategy:
#   1. Populate trie with 512-token sequence → 2 pages borrowed (indices 0,1)
#   2. endSequence grafts both pages into trie
#   3. startSequence with 300 tokens → LPM matches both pages
#   4. COW borrows page index 2 from pool (next free)
#   5. Check ctx.pages[0].index == 0 (fully-matched page, correct position)
#   6. Check ctx.pages[1].index == 2 (COW page, correct position)
# ═════════════════════════════════════════════════════════════════════════════

proc testBugB002CowPageOrdering(): bool =
  ## Verify page ordering after LPM with partial match:
  ##   fully-matched pages first, COW page last.
  var orc = makeOrchestrator()

  # ── Phase 1: Populate the trie with a 512-token sequence ──
  let seq1Tokens = makeTokens(TokensPerPage * 2)  # 512 tokens = 2 pages
  orc.startSequence(seq1Tokens)
  orc.endSequence()
  # After endSequence: both pages (indices 0 and 1) are grafted into the trie
  # Pool free_indices: [31, 30, ..., 2] (indices 0 and 1 are in use by trie)

  # ── Phase 2: Prefill 300 tokens (same prefix → LPM match) ──
  # 300 = 256 full + 44 partial: matched.pages[0] (index 0) fully matched,
  # matched.pages[1] (index 1) partially matched → COW needed.
  let seq2Tokens = makeTokens(300)
  orc.startSequence(seq2Tokens)
  let ctx = orc.getInferenceContext()

  # After startSequence with 300-token LPM match (256 + 44 partial):
  #   ctx.pages should have 2 entries:
  #     - [0]: fully-matched page (index 0 from trie)
  #     - [1]: COW page (index 2, freshly borrowed)
  doAssert ctx.pages.len == 2,
    "[BUG-B-002] Expected 2 pages for 300-token match (1 full + 1 COW), got " &
    $ctx.pages.len

  # Verify ordering: first page must be the fully-matched page (index 0)
  doAssert ctx.pages[0].pageIndex() == 0,
    "[BUG-B-002] ctx.pages[0] should be the fully-matched page (index 0), " &
    "got index " & $ctx.pages[0].pageIndex()

  # Verify ordering: second page must be the COW page (index 2, newly borrowed)
  doAssert ctx.pages[1].pageIndex() == 2,
    "[BUG-B-002] ctx.pages[1] should be the COW page (index 2), " &
    "got index " & $ctx.pages[1].pageIndex()

  # Verify kv_position reflects total matched
  doAssert ctx.kv_position == 300,
    "[BUG-B-002] kv_position should be 300 after 300-token match, got " &
    $ctx.kv_position

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# Additional lifecycle tests
# ═════════════════════════════════════════════════════════════════════════════

proc testOrchestratorLifecycleRoundtrip(): bool =
  ## Verify that startSequence → endSequence → startSequence works
  ## and contexts are properly reset between sequences.
  var orc = makeOrchestrator()
  let tokens = makeTokens(128)  # less than 1 page

  # Sequence 1
  orc.startSequence(tokens)
  let ctx = orc.getInferenceContext()
  # kv_position reflects total prefill tokens (BUG-B-001 fix)
  doAssert ctx.kv_position == 128,
    "Expected kv_position=128 after 128-token prefill, got " &
    $ctx.kv_position
  doAssert ctx.pages.len == 1

  orc.endSequence()
  # After endSequence, context should be cleared
  doAssert ctx.pages.len == 0,
    "Expected 0 pages after endSequence, got " & $ctx.pages.len
  doAssert ctx.kv_position == 0,
    "Expected kv_position=0 after endSequence, got " & $ctx.kv_position

  # Sequence 2 — reuse orchestrator
  orc.startSequence(tokens)
  doAssert ctx.kv_position == 128,
    "Expected kv_position=128 after second 128-token prefill, got " &
    $ctx.kv_position
  doAssert ctx.pages.len == 1,
    "Expected 1 page for second sequence, got " & $ctx.pages.len

  result = true

proc testOrchestratorDecodePageBoundaryAllocation(): bool =
  ## Verify page allocation on decode page boundary crossing.
  ## Prefill 255 tokens (just under 1 page), then decode until we cross
  ## the boundary and a new page is allocated.
  var orc = makeOrchestrator()
  let tokens = makeTokens(TokensPerPage - 1)  # 255 tokens

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContext()

  doAssert ctx.pages.len == 1,
    "Expected 1 page for 255 tokens, got " & $ctx.pages.len
  doAssert ctx.kv_position == TokensPerPage - 1,
    "kv_position should be 255 after 255-token prefill, got " & $ctx.kv_position

  # Decode at position 255 (0-indexed) — stays within page 0
  orc.decodeStep(position = TokensPerPage - 1, token_id = 100'u32, device = kCPU)
  doAssert ctx.pages.len == 1,
    "Expected still 1 page after decode within page boundary, got " & $ctx.pages.len
  doAssert ctx.kv_position == TokensPerPage,
    "kv_position should be 256 after crossing boundary, got " & $ctx.kv_position

  # Decode at position 256 — crosses into page 1
  orc.decodeStep(position = TokensPerPage, token_id = 101'u32, device = kCPU)
  doAssert ctx.pages.len == 2,
    "Expected 2 pages after crossing page boundary, got " & $ctx.pages.len
  doAssert ctx.kv_position == TokensPerPage + 1,
    "kv_position should be 257, got " & $ctx.kv_position

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# Test runner
# ═════════════════════════════════════════════════════════════════════════════

proc runTests*() =
  suite "Orchestrator — BUG-B-001 (kv_position not updated after prefill)":
    test "prefill 256 tokens → decodeStep at boundary (no crash)":
      check testBugB001KvPositionUpdate()

  suite "Orchestrator — BUG-B-002 (COW page ordering inverted)":
    test "LPM 300 tokens (256 full + 44 partial) → correct page order":
      check testBugB002CowPageOrdering()

  suite "Orchestrator — page boundary decode allocation":
    test "prefill 255 tokens → decode across page boundary":
      check testOrchestratorDecodePageBoundaryAllocation()

  suite "Orchestrator — lifecycle round-trip":
    test "startSequence → endSequence → startSequence reuse":
      check testOrchestratorLifecycleRoundtrip()

when isMainModule:
  runTests()
